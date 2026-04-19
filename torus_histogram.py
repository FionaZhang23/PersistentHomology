# torus_histogram.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from tadasets import torus
from ripser import ripser
import codim_functions as cf  # quiver_codim, weighted_quiver_codim

OUT_DIR = Path("/Users/fionazhang/PycharmProjects/PersistentHomology/graphs/torus_histograms")

# ------------------ helpers ------------------
def sample_torus(n: int, c: float, a: float, noise: float, seed: int) -> np.ndarray:
    """Version-compatible sampler for tadasets.torus."""
    try:
        return torus(n=n, c=c, a=a, noise=noise, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return torus(n=n, c=c, a=a, noise=noise)

def h1_intervals(X: np.ndarray) -> List[Tuple[float, float]]:
    dgms = ripser(X, maxdim=1)["dgms"]
    if len(dgms) < 2:
        return []
    H1 = dgms[1]
    return [tuple(map(float, row)) for row in H1]

def gather_codim_samples(
    n_trials: int, n_points: int, a: float, c: float, noise: float, base_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unweighted_array, weighted_array) of length n_trials."""
    rng = np.random.default_rng(base_seed + int(10_000*a + 100*c + 1000*noise + n_points))
    unweighted, weighted = [], []
    for _ in range(n_trials):
        seed = int(rng.integers(0, 2**31 - 1))
        X = sample_torus(n=n_points, c=c, a=a, noise=noise, seed=seed)
        H1 = h1_intervals(X)
        unweighted.append(cf.quiver_codim(H1, assume_sorted=False))
        weighted.append(cf.weighted_quiver_codim(H1, assume_sorted=False))
    return np.asarray(unweighted, float), np.asarray(weighted, float)

def _int_bins_for_row(arrs: List[np.ndarray]) -> np.ndarray:
    """Integer-aligned bins covering all arrays in the row."""
    vmin = min(a.min() for a in arrs) if arrs else 0.0
    vmax = max(a.max() for a in arrs) if arrs else 1.0
    vmin = float(np.floor(vmin) - 0.5)
    vmax = float(np.ceil(vmax) + 0.5)
    return np.arange(vmin, vmax + 1.0, 1.0)

def _cont_bins_for_row(arrs: List[np.ndarray], num_bins: int = 20) -> np.ndarray:
    """Shared continuous bins across a row (weighted case)."""
    vmin = min(a.min() for a in arrs) if arrs else 0.0
    vmax = max(a.max() for a in arrs) if arrs else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return np.linspace(vmin, vmax, num_bins + 1)

def _annotate_stats(ax, values: np.ndarray):
    mu = float(values.mean()) if values.size else 0.0
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    ax.text(
        0.98, 0.98, f"mean={mu:.3f}\nstd={sd:.3f}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.8, boxstyle="round,pad=0.25")
    )

# ------------------ grid plotting ------------------
def plot_grid_for_configs(
    results: Dict[Tuple[int, float], Dict[int, Tuple[np.ndarray, np.ndarray]]],
    n_list: List[int],
    fig_title: str,
    weighted: bool,
    out_path: Path,
):
    """
    results[(c, noise)][n] -> (unweighted_arr, weighted_arr)
    Rows keyed by (c, noise) in iteration order; columns over n_list.
    """
    rows = list(results.keys())  # list of (c, noise)
    nrows, ncols = len(rows), len(n_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 2.8*nrows), squeeze=False, constrained_layout=True)

    for r, (c, noise) in enumerate(rows):
        # collect arrays for the row and choose shared bins
        row_arrays = []
        for n in n_list:
            uw, w = results[(c, noise)][n]
            row_arrays.append(w if weighted else uw)

        if weighted:
            bins = _cont_bins_for_row(row_arrays, num_bins=20)
            xlabel = "Weighted codimension (per trial)"
        else:
            bins = _int_bins_for_row(row_arrays)
            xlabel = "Unweighted codimension (per trial)"

        for cidx, n in enumerate(n_list):
            ax = axes[r, cidx]
            uw, w = results[(c, noise)][n]
            vals = w if weighted else uw
            ax.hist(vals, bins=bins, edgecolor="black")
            _annotate_stats(ax, vals)

            if r == 0:
                ax.set_title(f"n={n}")
            if cidx == 0:
                ax.set_ylabel(f"c={c}, σ={noise}")
            if r == nrows - 1:
                ax.set_xlabel(xlabel)

    fig.suptitle(fig_title, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# ------------------ main run ------------------
if __name__ == "__main__":
    # Settings from your request
    N_TRIALS = 100          # uppercase N: trials
    a = 1                   # minor radius
    c_list = [1, 5, 10]
    sigma_list = [0.0, 0.2, 0.4]
    n_list = [100, 500, 1000]
    BASE_SEED = 1701

    # Compute and cache results: results[(c, noise)][n] = (unweighted_array, weighted_array)
    results: Dict[Tuple[int, float], Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for c in c_list:
        for noise in sigma_list:
            key = (c, noise)
            results[key] = {}
            for n_points in n_list:
                uw, w = gather_codim_samples(
                    n_trials=N_TRIALS, n_points=n_points, a=a, c=c, noise=noise, base_seed=BASE_SEED
                )
                results[key][n_points] = (uw, w)

    # Make output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Unweighted grid
    plot_grid_for_configs(
        results=results,
        n_list=n_list,
        fig_title=f"Torus codimension histograms (UNWEIGHTED) — a={a}, N={N_TRIALS}, rows=(c,σ), cols=n",
        weighted=False,
        out_path=OUT_DIR / "torus_hist_grid_unweighted_a1_N100_rows_c-sigma_cols_n.svg",
    )

    # Weighted grid
    plot_grid_for_configs(
        results=results,
        n_list=n_list,
        fig_title=f"Torus codimension histograms (WEIGHTED) — a={a}, N={N_TRIALS}, rows=(c,σ), cols=n",
        weighted=True,
        out_path=OUT_DIR / "torus_hist_grid_weighted_a1_N100_rows_c-sigma_cols_n.svg",
    )

    print(f"Saved SVG grids under: {OUT_DIR}")
x