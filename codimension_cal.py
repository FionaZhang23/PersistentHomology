from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tadasets import dsphere,torus
from ripser import ripser
import persim
import matplotlib
matplotlib.use("Agg")
def codimension_barcode(barcode: List[Tuple[float, float]], assume_sorted: bool = False) -> int:
    """
    Compute the codimension over an H_n barcode per rule:
      count (i, j) with b_i < b_j, d_i < d_j, and b_j <= d_i.

    Parameters
    ----------
    barcode : list of (birth, death)
    assume_sorted : if True, assumes barcode already sorted by birth (ascending)

    Returns
    -------
    total_codim : int
    """
    if barcode is None:
        return 0
    # clean & (optionally) sort
    bars = [(float(b), float(d)) for (b, d) in barcode if b <= d]
    if not assume_sorted:
        bars.sort(key=lambda x: x[0])  # sort by birth

    m = len(bars)
    total = 0

    for i in range(m):
        b1, d1 = bars[i]
        # only need to check j > i because b_j >= b_i after sorting; we require b_j > b_i
        for j in range(i + 1, m):
            b2, d2 = bars[j]
            if b2 > d1:
                # births are sorted, so all later bars start after d1 too → can stop early
                break
            # check the rule: b1 < b2, d1 < d2, b2 <= d1
            if b1 < b2 and d1 < d2 and b2 <= d1:
                total += 1

    return total
N_TRIALS = 100     # number of independent samplings
N_POINTS = 100     # points per circle
NOISE_SD = 0.3    # Gaussian noise std dev added to coordinates
BASE_SEED = 2025   # set for reproducibility (change if you want different runs)

# --- Helper: sample one noisy circle in R^2 (uses tadasets.dsphere) ---
def sample_circle(n: int, noise: float, seed: int) -> np.ndarray:
    # dsphere with d=1 gives points on S^1 embedded in R^2
    return dsphere(n=n, d=1, noise=noise, seed=seed)

# --- Run experiment ---
codims = []
rng = np.random.default_rng(BASE_SEED)

for trial in range(N_TRIALS):
    # Different seed per trial (but reproducible across runs via BASE_SEED)
    seed = int(rng.integers(0, 2**31 - 1))
    X = sample_circle(N_POINTS, NOISE_SD, seed)

    dgms = ripser(X, maxdim=1)["dgms"]
    H1 = dgms[1]  # numpy array of shape (k, 2)
    # Convert rows to tuples for your function
    codim = codimension_barcode([tuple(row) for row in H1], assume_sorted=False)
    codims.append(codim)

codims = np.array(codims)

# --- Report and plot histogram (integer bins centered on integers) ---
print(f"Trials: {N_TRIALS}, points per circle: {N_POINTS}, noise σ={NOISE_SD}")
print(f"Mean codimension = {codims.mean():.3f}, Std = {codims.std(ddof=1):.3f}")
print(f"Min = {codims.min()}, Max = {codims.max()}")

# Integer-aligned bins
bins = np.arange(codims.min() - 0.5, codims.max() + 1.5, 1.0)

plt.figure(figsize=(7, 4.5))
plt.hist(codims, bins=bins, edgecolor='black')
plt.xlabel("Codimension (per trial)")
plt.ylabel("Count")
plt.title(f"Codimension distribution over {N_TRIALS} noisy circles (n={N_POINTS}, σ={NOISE_SD})")
plt.tight_layout()
plt.savefig("codimension_hist_0.3.png", dpi=300, bbox_inches="tight")

'''
bars_1 = [(0.0, 0.2), (0.3, 0.5), (0.6, 0.7), (0.8, 1.0)]
print(codimension_barcode(bars_1))  # -> 0
bars_2 = [(0.0, 0.2), (0, 0.4), (0.3, 1.0), (0.7, 1.0)]
print(codimension_barcode(bars_2))  # -> 0
bars_3 = [(0.0, 0.2), (0, 0.4), (0.3, 0.7), (0.35, 1.0)]
print(codimension_barcode(bars_3))  # -> 0
bars_4 = [(0.0, 0.2), (0.2, 0.5), (0.3, 0.7), (0.52, 1.0)]
print(codimension_barcode(bars_3))  # -> 0

# sample a (possibly noisy) 2-D circle; vary seed so each run differs
X = dsphere(n=100, d=1, noise=0.1)

# persistent homology (Rips) up to H1
dgms = ripser(X, maxdim=1)["dgms"]
H0 = dgms[0]
H1 = dgms[1]
print(codimension_barcode(H1))

D = np.array([[0, 1],
              [1, 0]], dtype=float)
out = ripser(D, maxdim=1, distance_matrix=True)
H1 = out["dgms"][1]


X = torus(n=1000, c=2.0, a=1.0, noise=0.1)
dgms = ripser(X, maxdim=1)["dgms"]
H0 = dgms[0]
H1 = dgms[1]
print(codimension_barcode(H1))
'''