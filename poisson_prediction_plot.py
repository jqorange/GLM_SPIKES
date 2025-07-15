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
    out_dir="prediction_analysis",
):
    """Analyze predictions for a given fold and neuron."""

    pred_path = f"cv_results/fold{fold}_pred.h5"
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(pred_path, "r") as hf:
        mu = hf["pred"][:, neuron_index].astype(np.float32)
        y_true = hf["true"][:, neuron_index].astype(np.float32)

    # sample n_iter Poisson realizations around the rate mu
    rng = np.random.default_rng()
    samples = rng.poisson(mu[:, None], size=(len(mu), n_iter))
    mean_pred = samples.mean(axis=1)

    # re‐bin into larger windows
    base_ms = 5  # 200Hz → 5 ms per bin
    bin_size = max(1, int(bin_ms / base_ms))

    def _rate(arr):
        trunc = (len(arr) // bin_size) * bin_size
        arr = arr[:trunc].reshape(-1, bin_size).sum(axis=1)
        return arr / (bin_ms / 1000.0)

    true_rate = _rate(y_true)
    pred_rate = _rate(mean_pred)

    # smooth
    sigma_bins = sigma_ms / bin_ms
    true_smooth = gaussian_filter1d(true_rate, sigma=sigma_bins)
    pred_smooth = gaussian_filter1d(pred_rate, sigma=sigma_bins)

    # prepare DataFrame
    times_ms = np.arange(len(true_smooth)) * bin_ms
    df = pd.DataFrame({
        "time_ms": times_ms,
        "true_rate": true_smooth,
        "pred_rate": pred_smooth,
    })

    # save CSV
    csv_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.csv")
    df.to_csv(csv_path, index=False)

    # slice to requested range
    start, end = (index_range if index_range is not None else (0, len(df)))
    sl = slice(start, end)

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["time_ms"].iloc[sl], df["true_rate"].iloc[sl], label="True")
    plt.plot(df["time_ms"].iloc[sl], df["pred_rate"].iloc[sl], label="Pred")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (Hz)")
    plt.title(f"Fold {fold}, Neuron {neuron_index}")
    plt.legend()
    plt.tight_layout()

    img_path = os.path.join(out_dir, f"fold{fold}_neuron{neuron_index}.png")
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"Saved {csv_path} and {img_path}")


if __name__ == "__main__":

    index_range = (8000, 10000)  # bin‐index slice, or None for whole
    n_iter = 100  # Poisson sampling iterations
    bin_ms = 20  # ms per re‐bin
    sigma_ms = 40  # Gaussian smoothing (ms)

    folds = range(1, 6)
    neurons = range(142)
    for fold in folds:
        out_dir = f"prediction_analysis/{fold}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for neuron in neurons:

            analyze_fold(
                fold=fold,
                neuron_index=neuron,
                index_range=index_range,
                n_iter=n_iter,
                bin_ms=bin_ms,
                sigma_ms=sigma_ms,
                out_dir=out_dir,
            )
