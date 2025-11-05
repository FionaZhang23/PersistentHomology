# experiments_codim.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from tadasets import dsphere, torus
from ripser import ripser
import codim_functions as cf  # your module with quiver_codim & weighted_quiver_codim

# ---------- Output root (figures only, SVG) ----------
OUTPUT_ROOT = Path("/Users/fionazhang/PycharmProjects/PersistentHomology/graphs")


# ---------- Helpers ----------
def sample_circle(n: int, noise: float, seed: int) -> np.ndarray:
    return dsphere(n=n, d=1, noise=noise, seed=seed)

def sample_torus(n: int, c: float, a: float, noise: float, seed: int) -> np.ndarray:
    # angle-uniform branch for reproducibility
    return torus(n=n, c=c, a=a, noise=noise, seed=seed)

def h1_barcode_points(X: np.ndarray) -> List[Tuple[float, float]]:
    dgms = ripser(X, maxdim=1)["dgms"]
    H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2), dtype=float)
    return [tuple(row) for row in H1]

def plot_hist_svg(values: np.ndarray, title: str, xlabel: str, out_svg: Path, integer_bins: bool = False) -> None:
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))

    if values.size == 0:
        plt.title(title + " (no values)")
    else:
        vmin, vmax = float(values.min()), float(values.max())
        if integer_bins:
            bins = np.arange(np.floor(vmin) - 0.5, np.ceil(vmax) + 1.5, 1.0)
            plt.hist(values, bins=bins, edgecolor="black")
        else:
            if np.isclose(vmin, vmax):
                plt.hist(values, bins=1, edgecolor="black")
            else:
                bins = np.linspace(vmin, vmax, 21)  # ~20 bins
                plt.hist(values, bins=bins, edgecolor="black")

    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches="tight")
    plt.close()


# ---------- Circle experiments (ALL with 500 trials) ----------
def run_circle_experiments():
    """
    - Re-run n=100 with 500 trials (noise=0.1).
    - Grid: n in {200, 300, 400, 500}, noise in {0.1, 0.2, 0.3, 0.4}, 500 trials each.
    Save SVGs under graphs/circle/.
    """
    rng_base = 2025
    figs_dir = OUTPUT_ROOT / "circle"

    # (A) n=100, 500 trials
    do_circle_case(
        figs_dir=figs_dir,
        base_seed=rng_base,
        n_trials=500,
        n_points=100,
        noise=0.1
    )

    # (B) grid with 500 trials each
    for n_points in [200, 300, 400, 500]:
        for noise in [0.1, 0.2, 0.3, 0.4]:
            do_circle_case(
                figs_dir=figs_dir,
                base_seed=rng_base,
                n_trials=500,
                n_points=n_points,
                noise=noise
            )

def do_circle_case(figs_dir: Path, base_seed: int, n_trials: int, n_points: int, noise: float):
    rng = np.random.default_rng(base_seed)

    unweighted_vals = []
    weighted_vals = []

    for _ in range(n_trials):
        seed = int(rng.integers(0, 2**31 - 1))
        X = sample_circle(n_points, noise, seed)
        H1 = h1_barcode_points(X)

        unweighted_vals.append(cf.quiver_codim(H1, assume_sorted=False))
        weighted_vals.append(cf.weighted_quiver_codim(H1, assume_sorted=False))

    unweighted_vals = np.array(unweighted_vals, dtype=float)
    weighted_vals = np.array(weighted_vals, dtype=float)

    stem = f"circle_n{n_points}_T{n_trials}_sigma{noise:.2f}"

    # Unweighted (integer-valued)
    plot_hist_svg(
        unweighted_vals,
        title=f"Unweighted codimension over {n_trials} noisy circles (n={n_points}, σ={noise})",
        xlabel="Unweighted codimension (per trial)",
        out_svg=figs_dir / f"{stem}_unweighted.svg",
        integer_bins=True
    )

    # Weighted (continuous)
    plot_hist_svg(
        weighted_vals,
        title=f"Weighted codimension over {n_trials} noisy circles (n={n_points}, σ={noise})",
        xlabel="Weighted codimension (per trial)",
        out_svg=figs_dir / f"{stem}_weighted.svg",
        integer_bins=False
    )


# ---------- Torus experiments (same as before, save SVGs to graphs/torus) ----------
def run_torus_experiments():
    """
    n = 100, trials = 100,
    noise ∈ {0, 0.1, 0.2, 0.3, 0.4},
    integer grid 1 <= a <= c <= 10.
    """
    figs_dir = OUTPUT_ROOT / "torus"
    base_seed = 4242
    n_trials = 100
    n_points = 100
    noise_list = [0.0, 0.1, 0.2, 0.3, 0.4]

    for a in range(1, 11):
        for c in range(a, 11):  # ensure c >= a
            for noise in noise_list:
                do_torus_case(figs_dir, base_seed, n_trials, n_points, a, c, noise)

def do_torus_case(figs_dir: Path, base_seed: int, n_trials: int, n_points: int, a: float, c: float, noise: float):
    rng = np.random.default_rng(base_seed + int(a*100 + c*10 + noise*1000))

    unweighted_vals = []
    weighted_vals = []

    for _ in range(n_trials):
        seed = int(rng.integers(0, 2**31 - 1))
        X = sample_torus(n_points, c=c, a=a, noise=noise, seed=seed)
        H1 = h1_barcode_points(X)

        unweighted_vals.append(cf.quiver_codim(H1, assume_sorted=False))
        weighted_vals.append(cf.weighted_quiver_codim(H1, assume_sorted=False))

    unweighted_vals = np.array(unweighted_vals, dtype=float)
    weighted_vals = np.array(weighted_vals, dtype=float)

    stem = f"torus_a{a}_c{c}_n{n_points}_T{n_trials}_sigma{noise:.2f}"

    # Unweighted (integer)
    plot_hist_svg(
        unweighted_vals,
        title=f"Unweighted codimension on torus (a={a}, c={c}, n={n_points}, σ={noise}, T={n_trials})",
        xlabel="Unweighted codimension (per trial)",
        out_svg=figs_dir / f"{stem}_unweighted.svg",
        integer_bins=True
    )

    # Weighted (continuous)
    plot_hist_svg(
        weighted_vals,
        title=f"Weighted codimension on torus (a={a}, c={c}, n={n_points}, σ={noise}, T={n_trials})",
        xlabel="Weighted codimension (per trial)",
        out_svg=figs_dir / f"{stem}_weighted.svg",
        integer_bins=False
    )


# ---------- Main ----------
if __name__ == "__main__":
    print(f"Saving all figures (SVG) to: {OUTPUT_ROOT}")
    '''
    run_circle_experiments()
    '''
    run_torus_experiments()
    print("Done.")
