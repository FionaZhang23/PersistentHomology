import matplotlib
matplotlib.use("TkAgg")   # or "Qt5Agg"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gudhi.datasets.generators import points
'''
# ---------- Generate samples ----------
# 1. Points on a 2-sphere (surface of sphere in R^3)
sphere_points = points.sphere(n_samples=1000, ambient_dim=3, radius=1.0, sample="random")

# 2. Points on a 2-torus (dim=2 → embedded in R^4, but we visualize only first 3 coords)
torus_points = points.torus(n_samples=1000, dim=2, sample="random")

# ---------- Plot ----------
fig = plt.figure(figsize=(12, 6))

# Sphere
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2],
            s=10, c="blue", alpha=0.6)
ax1.set_title("Random points on Sphere (S²)")

# Torus
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(torus_points[:, 0], torus_points[:, 1], torus_points[:, 2],
            s=10, c="red", alpha=0.6)
ax2.set_title("Random points on Torus (projected to 3D)")

plt.show()
'''

import matplotlib
matplotlib.use("TkAgg")    # or "Qt5Agg"
import matplotlib.pyplot as plt
import numpy as np
import gudhi
from gudhi.datasets.generators import points

# ---------------- Parameters ----------------
n = 20                # number of repeated samplings
n_samples = 200         # points per sampling
radius = 1.0            # circle radius

# ---------------- Create figure ----------------
cols = 5
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols,
                         figsize=(5*cols, 5*rows),
                         subplot_kw={'aspect':'equal'})
axes = axes.flatten()

# ---------------- Sampling loop ----------------
for i in range(n):
    # Use Gudhi: circle is a 1-sphere in R²
    circle_points = points.sphere(n_samples=n_samples,
                                  ambient_dim=2,
                                  radius=radius,
                                  sample="random")

    ax = axes[i]
    ax.scatter(circle_points[:, 0], circle_points[:, 1],
               s=10, c='blue', alpha=0.6)
    ax.set_title(f"Run {i+1}")
    ax.set_xlim(-1.2*radius, 1.2*radius)
    ax.set_ylim(-1.2*radius, 1.2*radius)
    ax.axis('off')

# Hide unused subplots if n is not a multiple of cols
for j in range(n, len(axes)):
    axes[j].axis('off')

fig.suptitle(f"{n} Independent Gudhi Circle (S¹) Samplings", fontsize=16)
plt.tight_layout()
plt.show()
