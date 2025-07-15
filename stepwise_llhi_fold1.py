import os
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln
import matplotlib.pyplot as plt
from numpy import unwrap
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helper functions reused from ``GLM_all.py``
# -----------------------------------------------------------------------------

def create_raised_cosine_basis(n_basis=16, history_ms=160, dt=1):
    """Raised cosine basis used for spike history."""
    ttb = np.arange(dt, history_ms + dt, dt, dtype=np.float32)
    log_t = np.log(ttb)
    centers = np.linspace(log_t[0], log_t[-1], n_basis, dtype=np.float32)
    width = centers[1] - centers[0]
    basis = [
        (np.cos(np.clip((log_t - c) * np.pi / (2 * width), -np.pi, np.pi)) + 1) / 2
        for c in centers
    ]
    return np.stack(basis, axis=1)


def spike_history_design_fast(spikes: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fast convolution of spike vector with history basis."""
    T, K = len(spikes), basis.shape[1]
    rev = basis[::-1, :]
    cols = [np.convolve(spikes, rev[:, k], mode="full")[:T] for k in range(K)]
    return np.stack(cols, axis=1)


def bin_col(series: pd.Series, n_bins: int, mn=None, mx=None) -> np.ndarray:
    if mn is None:
        mn = series.min()
    if mx is None:
        mx = series.max()
    edges = np.linspace(mn, mx, n_bins + 1)
    out = np.digitize(series, edges) - 1
    return np.clip(out, 0, n_bins - 1).astype(np.int16)


def repeat_df(df: pd.DataFrame, f: int = 4) -> pd.DataFrame:
    return df.loc[df.index.repeat(f)].reset_index(drop=True)


def upsample_df(df: pd.DataFrame, f: int = 4) -> pd.DataFrame:
    old = np.arange(len(df))
    new = np.linspace(0, len(df) - 1, len(df) * f)
    return pd.DataFrame({c: np.interp(new, old, df[c].values).astype(np.float32) for c in df.columns})


def log_likelihood_poisson(y: np.ndarray, mu: np.ndarray) -> float:
    eps = 1e-12
    return np.sum(y * np.log(mu + eps) - mu - gammaln(y + 1))


# -----------------------------------------------------------------------------
# Build design matrices (same processing as ``GLM_all.py``)
# -----------------------------------------------------------------------------

def build_inputs():
    """Load spike and feature matrices and align to 1000 Hz."""
    with h5py.File("spike_binary_200Hz.h5", "r") as hf:
        spike_arr = hf["spike_binary"][:].astype(np.float32)
    neuron_names = [f"neuron_{i + 1}" for i in range(spike_arr.shape[1])]

    beh = pd.read_csv("F5D10_outdoor_modified.csv").drop(columns=["Index"], errors="ignore")
    pos = pd.read_csv("positions_F5D10_outdoor.csv")[["head_x", "head_y"]]
    dlc = pd.read_csv("final_filtered_F5D10_outdoor_50hz.csv")[["head_v", "bodyCenter1_v"]]
    imu = pd.read_csv("F5D10_outdoor_IMU_features_basic.csv")[["roll", "yaw", "pitch", "speed_z"]]

    pos["x_bin"] = (pos["head_x"] // (4 * 4.611473)).astype(int)
    pos["y_bin"] = (pos["head_y"] // (4 * 4.611473)).astype(int)
    uniq = pos[["x_bin", "y_bin"]].drop_duplicates().reset_index(drop=True)
    uniq["pos_idx"] = np.arange(len(uniq))
    pos = pos.merge(uniq, on=["x_bin", "y_bin"], how="left")

    beh_2h = repeat_df(beh)
    dlc_2h = upsample_df(dlc)
    imu["roll"] = unwrap(imu["roll"])
    imu["yaw"] = unwrap(imu["yaw"])
    imu["pitch"] = unwrap(imu["pitch"])
    imu_2h = upsample_df(imu)
    pos_2h = repeat_df(pos[["pos_idx"]]).rename(columns={"pos_idx": "position"})

    dlc_bin = pd.DataFrame({c: (bin_col(dlc_2h[c], 24, -180, 180) if "angle" in c else bin_col(dlc_2h[c], 30)) for c in dlc_2h})
    imu_bin = pd.DataFrame({c: (bin_col(imu_2h[c], 24, -np.pi, np.pi) if c in ["roll", "yaw", "pitch"] else bin_col(imu_2h[c], 30)) for c in imu_2h})

    cat_df = pd.concat([dlc_bin, imu_bin, pos_2h], axis=1).reset_index(drop=True)
    cats = []
    for col in cat_df:
        if col in ["roll", "yaw", "pitch"]:
            cats.append(np.arange(24))
        elif col in ["head_v", "bodyCenter1_v", "speed_z"]:
            cats.append(np.arange(30))
        elif col == "position":
            cats.append(np.arange(uniq.shape[0]))
        else:
            cats.append(np.arange(30))
    encoder = OneHotEncoder(categories=cats, sparse_output=True, handle_unknown="ignore")
    X_cat = encoder.fit_transform(cat_df).astype(np.float32)

    X_beh = sparse.csr_matrix(beh_2h.values.astype(np.float32))
    X_both = sparse.hstack([X_beh, X_cat], format="csr")

    feat_beh = list(beh_2h.columns)
    feat_cat = encoder.get_feature_names_out(cat_df.columns.tolist())
    history_basis = create_raised_cosine_basis(16, 160)
    hist_names = [f"history_{i + 1}" for i in range(history_basis.shape[1])]
    all_names = feat_beh + list(feat_cat) + hist_names + ["intercept"]

    start = int(1.238591123957225e4 * 200)
    end = int(1.588404693733597e4 * 200)

    Y = spike_arr[start:end].astype(np.float32)
    X_al = X_both[: end - start]

    return X_al, Y, history_basis, neuron_names, all_names


# -----------------------------------------------------------------------------
# Stepwise LLHi and prediction
# -----------------------------------------------------------------------------

def analyze_stepwise(
    fold: int = 1,
    neuron_index: int = 0,
    index_range=None,
    n_iter: int = 100,
    bin_ms: int = 20,
    sigma_ms: int = 40,
    out_dir: str = "stepwise_analysis",
):
    X_all, Y_all, history_basis, neuron_names, mapping = build_inputs()

    kf = KFold(n_splits=5, shuffle=False)
    splits = list(kf.split(X_all))
    tr_idx, va_idx = splits[fold - 1]
    X_va = X_all[va_idx]
    y_va = Y_all[va_idx, neuron_index].astype(np.float32)

    coef_df = pd.read_csv(f"cv_results/fold{fold}_train_coefficients.csv", index_col=0)
    coef = coef_df.loc[neuron_names[neuron_index]].values.astype(np.float32)

    # parse mapping indices
    idx_beh = [i for i, name in enumerate(mapping) if name in coef_df.columns[: len(mapping)]]
    idx_roll = [i for i, name in enumerate(mapping) if name.startswith("roll_")]
    idx_yaw = [i for i, name in enumerate(mapping) if name.startswith("yaw_")]
    idx_pitch = [i for i, name in enumerate(mapping) if name.startswith("pitch_")]
    idx_speed = [i for i, name in enumerate(mapping) if name.startswith("speed_z_")]
    idx_hist = [i for i, name in enumerate(mapping) if name.startswith("history_")]
    intercept_idx = len(mapping) - 1

    # compute spike history design
    with h5py.File("spike_binary_200Hz.h5", "r") as hf:
        spikes = hf["spike_binary"][:, neuron_index].astype(np.float32)
    H = spike_history_design_fast(spikes, history_basis)
    H = H[start:end]
    H_va = H[va_idx]

    X_aug = sparse.hstack([X_va, sparse.csr_matrix(H_va)], format="csr")

    ll_list = []
    rates = {}

    def predict(weights):
        lin = X_aug.dot(weights[:-1]) + weights[-1]
        return np.exp(lin)

    def sample_and_rate(mu):
        rng = np.random.default_rng()
        samples = rng.poisson(mu[:, None], size=(len(mu), n_iter))
        mean_pred = samples.mean(axis=1)
        base_ms = 5
        bin_size = max(1, int(bin_ms / base_ms))

        def _rate(arr):
            trunc = (len(arr) // bin_size) * bin_size
            arr = arr[:trunc].reshape(-1, bin_size).sum(axis=1)
            return arr / (bin_ms / 1000.0)

        true_rate = _rate(y_va)
        pred_rate = _rate(mean_pred)
        sigma_bins = sigma_ms / bin_ms
        true_smooth = gaussian_filter1d(true_rate, sigma=sigma_bins)
        pred_smooth = gaussian_filter1d(pred_rate, sigma=sigma_bins)
        return true_smooth, pred_smooth

    active = np.zeros_like(coef[:-1], dtype=bool)

    steps = [
        ("behavior", idx_beh),
        ("rpy", idx_roll + idx_yaw + idx_pitch),
        ("speed_z", idx_speed),
        ("history", idx_hist),
    ]

    prev_mu = np.exp(np.full_like(y_va, coef[-1]))
    prev_ll = log_likelihood_poisson(y_va, prev_mu)
    ll_list.append(("baseline", prev_ll, 0.0))
    true_smooth, pred_smooth = sample_and_rate(prev_mu)
    rates["baseline"] = pred_smooth

    for name, idxs in steps:
        active[idxs] = True
        weights = np.concatenate([coef[:-1] * active, [coef[-1]]])
        mu = predict(weights)
        ll = log_likelihood_poisson(y_va, mu)
        llhi = (ll - prev_ll) / np.maximum(y_va.sum(), 1)
        ll_list.append((name, ll, llhi))
        prev_ll = ll
        prev_mu = mu
        _, pred_smooth = sample_and_rate(mu)
        rates[name] = pred_smooth

    # prepare output dataframe
    base_ms = 5
    bin_size = max(1, int(bin_ms / base_ms))
    trunc = (len(y_va) // bin_size) * bin_size
    times_ms = np.arange(trunc // bin_size) * bin_ms
    df = pd.DataFrame({"time_ms": times_ms, "true_rate": true_smooth})
    for name in rates:
        df[f"pred_{name}"] = rates[name]

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.csv")
    df.to_csv(csv_path, index=False)

    if index_range is not None:
        start_i, end_i = index_range
        sl = slice(start_i, end_i)
    else:
        sl = slice(None)

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_ms"].iloc[sl], df["true_rate"].iloc[sl], label="True", linewidth=2)
    colors = {
        "baseline": "gray",
        "behavior": "tab:blue",
        "rpy": "tab:orange",
        "speed_z": "tab:green",
        "history": "tab:red",
    }
    for name in rates:
        plt.plot(df["time_ms"].iloc[sl], df[f"pred_{name}"].iloc[sl], label=name, color=colors.get(name, None), alpha=0.6)
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Fold {fold}, Neuron {neuron_index}")
    plt.legend()
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    llhi_df = pd.DataFrame(ll_list, columns=["step", "ll", "llhi"])
    llhi_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}_llhi.csv")
    llhi_df.to_csv(llhi_path, index=False)
    print(f"Saved {csv_path}, {img_path} and {llhi_path}")


if __name__ == "__main__":
    index_range = (8000, 10000)
    analyze_stepwise(
        fold=1,
        neuron_index=0,
        index_range=index_range,
        n_iter=100,
        bin_ms=20,
        sigma_ms=40,
    )
