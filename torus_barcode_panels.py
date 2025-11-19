# torus_barcode_panels.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt

from tadasets import torus
from ripser import ripser
import persim  # scikit-tda plotting
import codim_functions as cf  # <- for weighted_quiver_codim

# -------- Output folder --------
OUT_DIR = Path("/Users/fionazhang/PycharmProjects/PersistentHomology/graphs/torus_barcodes")

# -------- Sampling helpers --------
def sample_torus(n: int, c: float, a: float, noise: float, seed: int) -> np.ndarray:
    """
    Version-compatible torus sampler: try with 'seed' first; if not supported, fall back.
    """
    try:
        return torus(n=n, c=c, a=a, noise=noise, seed=seed)
    except TypeError:
        np.random.seed(seed)
        return torus(n=n, c=c, a=a, noise=noise)

def ripser_h1(X: np.ndarray) -> np.ndarray:
    """Return H1 diagram as an (k,2) ndarray (birth, death). Empty if none."""
    dgms = ripser(X, maxdim=1)["dgms"]
    if len(dgms) < 2:
        return np.empty((0, 2), dtype=float)
    return np.asarray(dgms[1], dtype=float)

# -------- Plotting helpers --------
def plot_single_barcode(ax, intervals: np.ndarray, xmax_cap: float | None = None, color="black"):
    """
    Draw a barcode on a given Axes for an (k,2) array of (birth, death).
    If death is inf, clamp to xmax_cap (if provided).
    """
    if intervals.size == 0:
        ax.text(0.5, 0.5, "No H1 bars", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return

    # sort by length desc for readability
    lengths = intervals[:, 1] - intervals[:, 0]
    order = np.argsort(-lengths)
    ints = intervals[order]

    ys = np.arange(len(ints))
    for y, (b, d) in zip(ys, ints):
        x1 = float(b)
        x2 = float(d if np.isfinite(d) else (xmax_cap if xmax_cap is not None else b))
        ax.hlines(y, x1, x2, color=color, linewidth=1.5)
        ax.vlines([x1, x2], y - 0.12, y + 0.12, color=color, linewidth=1.0)

    ax.set_ylim(-0.5, len(ints) - 0.5)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

def plot_single_diagram(ax, H1: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float]):
    """
    Use scikit-tda persim to plot the H1 persistence diagram on the given Axes.
    """
    H1_plot = H1[np.isfinite(H1).all(axis=1)] if H1.size else H1
    persim.plot_diagrams(H1_plot, ax=ax, show=False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

# -------- Panel builder --------
def save_torus_panel_with_barcodes_and_diagrams(
    a: float, c: float, n: int, num_samples: int,
    noise: float, base_seed: int = 2025
) -> Path:
    """
    For a fixed (a,c,noise), sample `num_samples` times, compute H1,
    and create a 5x2 grid: left column = barcode, right column = persim diagram.
    Also annotate each row with the weighted codimension value.
    """
    rng = np.random.default_rng(base_seed + int(10_000 * a + 100 * c + 10 * noise))

    H1_list: List[np.ndarray] = []
    wcodim_list: List[float] = []

    for _ in range(num_samples):
        seed = int(rng.integers(0, 2**31 - 1))
        X = sample_torus(n=n, c=c, a=a, noise=noise, seed=seed)
        H1 = ripser_h1(X)
        H1_list.append(H1)
        # weighted codimension (expects list of tuples)
        wcodim = cf.weighted_quiver_codim([tuple(row) for row in H1], assume_sorted=False)
        wcodim_list.append(float(wcodim))

    # Determine consistent x/y bounds across all samples (ignore infs)
    finite_births = [pt[0] for H1 in H1_list for pt in H1 if np.isfinite(pt).all()]
    finite_deaths = [pt[1] for H1 in H1_list for pt in H1 if np.isfinite(pt).all()]
    if finite_births and finite_deaths:
        xmin = min(finite_births)
        xmax = max(finite_deaths)
        if np.isclose(xmin, xmax):
            xmax = xmin + 1.0
    else:
        xmin, xmax = 0.0, 1.0
    pad = 0.05 * (xmax - xmin)
    xcap = xmax
    xlim = (xmin - pad, xmax + pad)
    ylim = xlim  # square limits for diagram (birth vs death)

    # Build figure: 5 rows × 2 cols (barcode | diagram)
    rows = num_samples
    fig, axes = plt.subplots(rows, 2, figsize=(12, 2.2 * rows), constrained_layout=True, sharex="col")

    if rows == 1:
        axes = np.array([axes])  # shape (1,2)

    for k in range(rows):
        ax_bar = axes[k, 0]
        ax_dgm = axes[k, 1]

        plot_single_barcode(ax_bar, H1_list[k], xmax_cap=xcap, color="black")
        ax_bar.set_xlim(*xlim)
        ax_bar.set_ylabel(f"S{k+1}", rotation=0, labelpad=15, va="center")

        plot_single_diagram(ax_dgm, H1_list[k], xlim=xlim, ylim=ylim)

        # annotate weighted codimension on the diagram panel (top-right, small box)
        ax_dgm.text(
            0.98, 0.98, f"weighted codim = {wcodim_list[k]:.4f}",
            transform=ax_dgm.transAxes, ha="right", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.5", boxstyle="round,pad=0.25", alpha=0.8)
        )

        if k == rows - 1:
            ax_bar.set_xlabel("Scale (birth → death)")
            ax_dgm.set_xlabel("Birth")
            ax_dgm.set_ylabel("Death")

    axes[0, 0].set_title("H1 Barcode")
    axes[0, 1].set_title("H1 Persistence Diagram")
    fig.suptitle(f"Torus (a={a}, c={c}, n={n}, samples={num_samples}, noise={noise})", y=1.02)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = OUT_DIR / f"torus_a{a}_c{c}_n{n}_T{num_samples}_sigma{noise:.2f}_H1_barcode+diagram.svg"
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    return out_svg

# -------- Main: a=1, c=1..10, noise ∈ {0.0,0.1,0.2,0.3,0.4} --------
if __name__ == "__main__":
    print(f"Saving SVGs to: {OUT_DIR}")
    a = 1
    n = 1000
    num_samples = 5
    noise_list = [0.0, 0.1, 0.2, 0.3, 0.4]

    for c in range(1, 11):
        for noise in noise_list:
            p = save_torus_panel_with_barcodes_and_diagrams(
                a=a, c=c, n=n, num_samples=num_samples, noise=noise, base_seed=2025
            )
            print(f"Wrote: {p}")
    print("Done.")
