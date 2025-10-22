import numpy as np
import matplotlib.pyplot as plt
from tadasets import dsphere,torus
from ripser import ripser
import persim  # for plot_diagrams

# -------- params you can change --------
n_reps = 20      # how many times to sample
n_points = 500     # points per circle
noise = 0.3     # set 0.0, 0.1, 0.2, etc.
# --------------------------------------

plt.rcParams["figure.dpi"] = 150

# make the big canvas: n_reps rows, 2 cols (scatter | diagrams)
fig, axes = plt.subplots(n_reps, 2, figsize=(10, 3.1 * n_reps))
if n_reps == 1:
    axes = np.array([axes])  # ensure 2D indexing axes[row, col]

for i in range(n_reps):
    # sample a (possibly noisy) 2-D circle; vary seed so each run differs
    X = dsphere(n=n_points, d=1, noise=noise, seed=i)

    # persistent homology (Rips) up to H1
    dgms = ripser(X, maxdim=1)["dgms"]

    # left: point cloud
    ax_scatter = axes[i, 0]
    ax_scatter.scatter(X[:, 0], X[:, 1], s=6)
    ax_scatter.set_aspect('equal', 'box')
    ax_scatter.set_title(f"Circle (noise={noise}, run={i+1})")
    ax_scatter.set_xlabel("x"); ax_scatter.set_ylabel("y")
    ax_scatter.grid(True, alpha=0.3)

    # right: persistence diagrams (H0/H1)
    ax_dgm = axes[i, 1]
    persim.plot_diagrams(dgms, ax=ax_dgm, show=False, title="Persistence diagrams (H0/H1)")

fig.tight_layout()
fig.savefig("all_circles_noise0.3.png")
print("Saved figure -> all_circles_and_diagrams.png")

X = torus(n=1000, c=2.0, a=1.0)
'''
# sample a (possibly noisy) 2-D circle; vary seed so each run differs
X = dsphere(n=1000, d=1, noise=0.0)

# persistent homology (Rips) up to H1
dgms = ripser(X, maxdim=1)["dgms"]
H0 = dgms[0]
H1 = dgms[1]
print(H0)
'''