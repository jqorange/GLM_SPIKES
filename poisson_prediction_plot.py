import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def analyze_fold(
    fold,
    neuron_index=0,
    index_range=None,
    n_iter=100,
    bin_ms=20,
    sigma_ms=40,
    pred_h5_pattern="cv_results/fold{fold}_pred.h5",
    out_dir="prediction_analysis",
):
    """Analyze predictions for a given fold and neuron.

    Parameters
    ----------
    fold : int
        Fold index (1-based).
    neuron_index : int, optional
        Neuron index to analyze, by default 0.
    index_range : tuple or None, optional
        Range of binned indices to plot, e.g. (0, 100). If None, use full range.
    n_iter : int, optional
        Number of Poisson sampling iterations, by default 100.
    bin_ms : int, optional
        Binning size in milliseconds, by default 20.
    sigma_ms : int, optional
        Sigma for Gaussian smoothing in milliseconds, by default 40.
    pred_h5_pattern : str, optional
        Pattern for prediction h5 files.
    out_dir : str, optional
        Output directory for CSV and figure.
    """
    pred_path = pred_h5_pattern.format(fold=fold)
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(pred_path, "r") as hf:
        mu = hf["pred"][:, neuron_index].astype(np.float32)
        y_true = hf["true"][:, neuron_index].astype(np.float32)

    rng = np.random.default_rng()
    samples = rng.poisson(mu[:, None], size=(len(mu), n_iter))
    mean_pred = samples.mean(axis=1)

    base_ms = 5  # original data sampled at 200 Hz -> 5 ms per bin
    bin_size = max(1, int(bin_ms / base_ms))

    def _rate(arr):
        trunc = len(arr) // bin_size * bin_size
        arr = arr[:trunc].reshape(-1, bin_size).sum(axis=1)
        return arr / (bin_ms / 1000.0)

    true_rate = _rate(y_true)
    pred_rate = _rate(mean_pred)

    sigma_bins = sigma_ms / bin_ms
    true_smooth = gaussian_filter1d(true_rate, sigma=sigma_bins)
    pred_smooth = gaussian_filter1d(pred_rate, sigma=sigma_bins)

    times_ms = np.arange(len(true_smooth)) * bin_ms
    df = pd.DataFrame({
        "time_ms": times_ms,
        "true_rate": true_smooth,
        "pred_rate": pred_smooth,
    })

    csv_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.csv")
    df.to_csv(csv_path, index=False)

    start, end = (index_range if index_range is not None else (0, len(df)))
    sl = slice(start, end)

    plt.figure(figsize=(10, 4))
    plt.plot(df["time_ms"].iloc[sl], df["true_rate"].iloc[sl], label="True")
    plt.plot(df["time_ms"].iloc[sl], df["pred_rate"].iloc[sl], label="Pred")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Fold {fold} Neuron {neuron_index}")
    plt.legend()
    plt.tight_layout()
    img_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Saved {csv_path} and {img_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze GLM fold predictions")
    parser.add_argument("fold", type=int, help="Fold index (1-5)")
    parser.add_argument("neuron", type=int, help="Neuron index")
    parser.add_argument("--start", type=int, default=0, help="Start index after binning")
    parser.add_argument("--end", type=int, default=None, help="End index after binning")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of Poisson draws")
    args = parser.parse_args()

    idx_range = (args.start, args.end) if args.end is not None else None

    analyze_fold(args.fold, neuron_index=args.neuron, index_range=idx_range, n_iter=args.n_iter)
