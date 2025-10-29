import numpy as np
import matplotlib.pyplot as plt
from tadasets import dsphere,torus
from ripser import ripser
import persim
import matplotlib
matplotlib.use("Agg")
import codim_functions as cf

N_TRIALS = 100     # number of independent samplings
N_POINTS = 100     # points per circle
NOISE_SD = 0.4    # Gaussian noise std dev added to coordinates
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
    codim = cf.weighted_quiver_codim([tuple(row) for row in H1], assume_sorted=False)
    codims.append(codim)

codims = np.array(codims)

# --- Report and plot histogram (integer bins centered on integers) ---
print(f"Trials: {N_TRIALS}, points per circle: {N_POINTS}, noise σ={NOISE_SD}")
print(f"Mean codimension = {codims.mean():.3f}, Std = {codims.std(ddof=1):.3f}")
print(f"Min = {codims.min()}, Max = {codims.max()}")

plt.figure(figsize=(7, 4.5))
plt.hist(codims, bins='auto', edgecolor='black')
plt.xlabel("Weighted codimension (per trial)")
plt.ylabel("Count")
plt.title(f"Weighted codimension over {N_TRIALS} noisy circles (n={N_POINTS}, σ={NOISE_SD})")
plt.tight_layout()
plt.savefig("weighted_codimension_hist_0.4.svg", dpi=300, bbox_inches="tight")
plt.close()

'''
bars_1 = [(0.0, 0.2), (0.3, 0.5), (0.6, 0.7), (0.8, 1.0)]
print(cf.quiver_codim(bars_1))  # -> 0
bars_2 = [(0.0, 0.2), (0, 0.4), (0.3, 1.0), (0.7, 1.0)]
print(cf.quiver_codim(bars_2))  # -> 0
bars_3 = [(0.0, 0.2), (0, 0.4), (0.3, 0.7), (0.35, 1.0)]
print(cf.quiver_codim(bars_3))  # -> 0
bars_4 = [(0.0, 0.2), (0.2, 0.5), (0.3, 0.7), (0.52, 1.0)]
print(cf.quiver_codim(bars_3))  # -> 0

# sample a (possibly noisy) 2-D circle; vary seed so each run differs
X = dsphere(n=100, d=1, noise=0.1)

# persistent homology (Rips) up to H1
dgms = ripser(X, maxdim=1)["dgms"]
H0 = dgms[0]
H1 = dgms[1]
print(cf.quiver_codim(H1))
print(cf.weighted_quiver_codim(H1))


X = torus(n=1000, c=2.0, a=1.0, noise=0.1)
dgms = ripser(X, maxdim=1)["dgms"]
H0 = dgms[0]
H1 = dgms[1]
print(cf.quiver_codim(H1))
'''