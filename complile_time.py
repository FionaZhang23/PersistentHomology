# tda_ph_timing_onepass.py
# ------------------------------------------------------------
# (A) Sample different numbers of points on a circle (NO noise)
#     and time a single ripser run for each n.
# (B) With n=1000, sample circle, sphere, torus (NO noise)
#     and time a single ripser run for each shape.
# Neat console tables. No plotting. No medians.
# ------------------------------------------------------------

import time
import numpy as np
from ripser import ripser
import tadasets

# -------------------- Config (edit here) --------------------
ns = [500, 1000, 1500, 2000]  # sizes for (A)
maxdim_A = 1   # circle vs n
maxdim_B = 2   # shapes compare at n=1000
n_fixed  = 1000
# ------------------------------------------------------------

# --------------- (A) Circle: timing vs n -------------------
print("\n(A) Persistent homology timing vs number of points (circle, no noise)")
print("    single run per n")
print("----------------------------------------------------------------")
print(f"{'n':>6}  {'time_s':>12}  {'maxdim':>6}")

for n in ns:
    # SAMPLE: fresh circle (1-sphere in R^2), NO noise
    X = tadasets.dsphere(n=n, d=1)  # no noise, no seed => fresh sample
    t0 = time.perf_counter()
    _ = ripser(X, maxdim=maxdim_A)  # compute PH
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"{n:6d}  {elapsed:12.6f}  {maxdim_A:6d}")

# --------------- (B) Shapes @ n=1000: one run each ----------
print("\n(B) Persistent homology timing at n=1000 (no noise): circle vs sphere vs torus")
print("    single run per shape")
print("-----------------------------------------------------------------------")
print(f"{'shape':<22}  {'n':>6}  {'time_s':>12}  {'maxdim':>6}")

# Circle (1-sphere in R^2)
X = tadasets.dsphere(n=n_fixed, d=1)
t0 = time.perf_counter()
_ = ripser(X, maxdim=maxdim_B)
t1 = time.perf_counter()
print(f"{'circle':<22}  {n_fixed:6d}  {t1 - t0:12.6f}  {maxdim_B:6d}")

# Sphere (2-sphere in R^3)
X = tadasets.sphere(n=n_fixed)
t0 = time.perf_counter()
_ = ripser(X, maxdim=maxdim_B)
t1 = time.perf_counter()
print(f"{'sphere':<22}  {n_fixed:6d}  {t1 - t0:12.6f}  {maxdim_B:6d}")

# Torus in R^3
X = tadasets.torus(n=n_fixed, c=2.0, a=1.0)
t0 = time.perf_counter()
_ = ripser(X, maxdim=maxdim_B)
t1 = time.perf_counter()
print(f"{'torus':<22}  {n_fixed:6d}  {t1 - t0:12.6f}  {maxdim_B:6d}")

print("\nDone.")
