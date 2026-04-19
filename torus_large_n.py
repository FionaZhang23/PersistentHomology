from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ripser import ripser
from tadasets import torus

import codim_functions as cf


# ============================================================
# Project paths
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "graphs" / "torus_histograms_large_n"


# ============================================================
# Main experiment settings
# ============================================================
N_TRIALS = 100                     # number of torus samples per histogram
A = 1.0                            # minor radius
C = 5.0                            # fixed major radius, per your request
SIGMA_LIST = [0.0, 0.2, 0.4]       # noise levels
N_LIST = [1000, 10000, 100000, 1000000]
BASE_SEED = 1701

# Ripser settings
MAXDIM = 1
THRESH = None

# For large n, exact VR persistence can be extremely slow.
# This script switches to greedy permutation after EXACT_CUTOFF.
# If you really want exact on all n, set N_PERM_LARGE = None and
# make EXACT_CUTOFF very large, but that may be impractical.
EXACT_CUTOFF = 10000
N_PERM_LARGE = 2000


# ============================================================
# Helpers
# ============================================================
def safe_name(x: float) -> str:
    """Make a float filename-safe."""
    return str(x).replace("-", "m").replace(".", "p")


def sample_torus(n: int, c: float, a: float, noise: float, seed: int) -> np.ndarray:
    """
    Version-compatible sampler for tadasets.torus.
    """
    try:
        return torus(n=n, c=c, a=a, noise=noise, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return torus(n=n, c=c, a=a, noise=noise)


def h1_intervals(X: np.ndarray, n_perm: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Compute H1 persistence intervals from ripser.
    Filters out malformed / non-finite bars.
    """
    kwargs = {"maxdim": MAXDIM}
    if THRESH is not None:
        kwargs["thresh"] = THRESH
    if n_perm is not None:
        kwargs["n_perm"] = n_perm

    dgms = ripser(X, **kwargs)["dgms"]

    if len(dgms) < 2:
        return []

    H1 = dgms[1]
    out = []
    for row in H1:
        b, d = float(row[0]), float(row[1])
        if np.isfinite(b) and np.isfinite(d) and b <= d:
            out.append((b, d))
    return out


def choose_n_perm(n_points: int) -> Optional[int]:
    """
    Exact for smaller n, greedy permutation for larger n.
    """
    if N_PERM_LARGE is None:
        return None
    if n_points <= EXACT_CUTOFF:
        return None
    return N_PERM_LARGE


def gather_codim_samples(
    n_trials: int,
    n_points: int,
    a: float,
    c: float,
    noise: float,
    base_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
        unweighted_array, weighted_array
    each of length n_trials.
    """
    rng = np.random.default_rng(
        base_seed + int(n_points + 1000 * noise + 100 * c + 10 * a)
    )

    n_perm = choose_n_perm(n_points)

    unweighted = []
    weighted = []

    for trial_idx in range(n_trials):
        seed = int(rng.integers(0, 2**31 - 1))
        X = sample_torus(n=n_points, c=c, a=a, noise=noise, seed=seed)
        H1 = h1_intervals(X, n_perm=n_perm)

        uw = cf.quiver_codim(H1, assume_sorted=False)
        w = cf.weighted_quiver_codim(H1, assume_sorted=False)

        unweighted.append(float(uw))
        weighted.append(float(w))

        print(
            f"[torus] sigma={noise}, n={n_points}, trial={trial_idx + 1}/{n_trials}, "
            f"bars={len(H1)}, n_perm={n_perm}"
        )

    return np.asarray(unweighted, dtype=float), np.asarray(weighted, dtype=float)


def _int_bins_for_row(arrs: List[np.ndarray]) -> np.ndarray:
    """
    Integer-aligned bins for unweighted codimension.
    """
    vmin = min(a.min() for a in arrs) if arrs else 0.0
    vmax = max(a.max() for a in arrs) if arrs else 1.0
    vmin = float(np.floor(vmin) - 0.5)
    vmax = float(np.ceil(vmax) + 0.5)
    return np.arange(vmin, vmax + 1.0, 1.0)


def _cont_bins_for_row(arrs: List[np.ndarray], num_bins: int = 20) -> np.ndarray:
    """
    Shared continuous bins for weighted codimension.
    """
    vmin = min(a.min() for a in arrs) if arrs else 0.0
    vmax = max(a.max() for a in arrs) if arrs else 1.0
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return np.linspace(vmin, vmax, num_bins + 1)


def _annotate_stats(ax, values: np.ndarray) -> None:
    mu = float(values.mean()) if values.size else 0.0
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    ax.text(
        0.98, 0.98,
        f"mean={mu:.3f}\nstd={sd:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="0.6",
            alpha=0.8,
            boxstyle="round,pad=0.25"
        )
    )


def summary_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)) if arr.size else 0.0,
        "q25": float(np.quantile(arr, 0.25)) if arr.size else 0.0,
        "median": float(np.median(arr)) if arr.size else 0.0,
        "q75": float(np.quantile(arr, 0.75)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
    }


def save_trial_level_csv(
    results: Dict[float, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    out_path: Path
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shape", "c", "a", "sigma", "n_points", "trial",
            "unweighted_codim", "weighted_codim"
        ])
        for noise in SIGMA_LIST:
            for n_points in N_LIST:
                uw, w = results[noise][n_points]
                for idx, (u, ww) in enumerate(zip(uw, w), start=1):
                    writer.writerow([
                        "torus", C, A, noise, n_points, idx, float(u), float(ww)
                    ])


def save_summary_csv(
    results: Dict[float, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    out_path: Path
) -> None:
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shape", "c", "a", "sigma", "n_points", "codim_type",
            "mean", "std", "min", "q25", "median", "q75", "max"
        ])

        for noise in SIGMA_LIST:
            for n_points in N_LIST:
                uw, w = results[noise][n_points]

                s_uw = summary_stats(uw)
                writer.writerow([
                    "torus", C, A, noise, n_points, "unweighted",
                    s_uw["mean"], s_uw["std"], s_uw["min"], s_uw["q25"],
                    s_uw["median"], s_uw["q75"], s_uw["max"]
                ])

                s_w = summary_stats(w)
                writer.writerow([
                    "torus", C, A, noise, n_points, "weighted",
                    s_w["mean"], s_w["std"], s_w["min"], s_w["q25"],
                    s_w["median"], s_w["q75"], s_w["max"]
                ])


def save_npy_arrays(
    results: Dict[float, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    out_dir: Path
) -> None:
    for noise in SIGMA_LIST:
        for n_points in N_LIST:
            uw, w = results[noise][n_points]
            sigma_tag = safe_name(noise)

            np.save(
                out_dir / f"torus_unweighted_c_{safe_name(C)}_a_{safe_name(A)}_sigma_{sigma_tag}_n_{n_points}.npy",
                uw
            )
            np.save(
                out_dir / f"torus_weighted_c_{safe_name(C)}_a_{safe_name(A)}_sigma_{sigma_tag}_n_{n_points}.npy",
                w
            )


def plot_grid_for_noise_levels(
    results: Dict[float, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    n_list: List[int],
    fig_title: str,
    weighted: bool,
    out_path: Path,
) -> None:
    """
    Rows = sigma
    Cols = n
    """
    nrows = len(SIGMA_LIST)
    ncols = len(n_list)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        squeeze=False,
        constrained_layout=True
    )

    for r, noise in enumerate(SIGMA_LIST):
        row_arrays = []
        for n_points in n_list:
            uw, w = results[noise][n_points]
            row_arrays.append(w if weighted else uw)

        if weighted:
            bins = _cont_bins_for_row(row_arrays, num_bins=20)
            xlabel = "Weighted codimension (per trial)"
        else:
            bins = _int_bins_for_row(row_arrays)
            xlabel = "Unweighted codimension (per trial)"

        for cidx, n_points in enumerate(n_list):
            ax = axes[r, cidx]
            uw, w = results[noise][n_points]
            vals = w if weighted else uw

            ax.hist(vals, bins=bins, edgecolor="black")
            _annotate_stats(ax, vals)

            if r == 0:
                ax.set_title(f"n={n_points}")
            if cidx == 0:
                ax.set_ylabel(f"σ={noise}")
            if r == nrows - 1:
                ax.set_xlabel(xlabel)

    fig.suptitle(fig_title, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting torus experiment...")
    print(f"Output directory: {OUT_DIR}")
    print(f"N_TRIALS={N_TRIALS}")
    print(f"C={C}, A={A}")
    print(f"SIGMA_LIST={SIGMA_LIST}")
    print(f"N_LIST={N_LIST}")
    print(f"EXACT_CUTOFF={EXACT_CUTOFF}, N_PERM_LARGE={N_PERM_LARGE}")

    # results[noise][n_points] = (unweighted_array, weighted_array)
    results: Dict[float, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}

    for noise in SIGMA_LIST:
        results[noise] = {}
        for n_points in N_LIST:
            print("-" * 70)
            print(f"Running torus for sigma={noise}, n={n_points}")
            uw, w = gather_codim_samples(
                n_trials=N_TRIALS,
                n_points=n_points,
                a=A,
                c=C,
                noise=noise,
                base_seed=BASE_SEED
            )
            results[noise][n_points] = (uw, w)

    # Save arrays
    save_npy_arrays(results, OUT_DIR)

    # Save CSVs
    save_trial_level_csv(results, OUT_DIR / "torus_trial_level_results.csv")
    save_summary_csv(results, OUT_DIR / "torus_summary_results.csv")

    # Unweighted histogram grid
    plot_grid_for_noise_levels(
        results=results,
        n_list=N_LIST,
        fig_title=(
            f"Torus codimension histograms (UNWEIGHTED) — "
            f"c={C}, a={A}, N={N_TRIALS}, rows=σ, cols=n"
        ),
        weighted=False,
        out_path=OUT_DIR / "torus_hist_grid_unweighted.png",
    )

    # Weighted histogram grid
    plot_grid_for_noise_levels(
        results=results,
        n_list=N_LIST,
        fig_title=(
            f"Torus codimension histograms (WEIGHTED) — "
            f"c={C}, a={A}, N={N_TRIALS}, rows=σ, cols=n"
        ),
        weighted=True,
        out_path=OUT_DIR / "torus_hist_grid_weighted.png",
    )

    print("Torus experiment finished.")
    print(f"Saved all outputs under: {OUT_DIR}")