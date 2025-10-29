from typing import List, Tuple

def _pair_codimension(
    bar_i: Tuple[float, float],
    bar_j: Tuple[float, float]
) -> int:
    """
    Check whether two intervals (birth_i, death_i) and (birth_j, death_j)
    contribute 1 to codimension under the rule:

        b_i < b_j,
        d_i < d_j,
        b_j <= d_i.

    Assumes we're treating bar_i as the "earlier" bar and bar_j as the "later" bar.
    Returns 1 if the rule is satisfied, else 0.
    """
    b_i, d_i = bar_i
    b_j, d_j = bar_j

    if (b_i < b_j) and (d_i < d_j) and (b_j <= d_i):
        return 1
    else:
        return 0


def quiver_codim(
    barcode: List[Tuple[float, float]],
    assume_sorted: bool = False
) -> int:
    """
    Compute the codimension over an H_n barcode per rule:
      Count all pairs (i, j) with:
          b_i < b_j,
          d_i < d_j,
          b_j <= d_i.

    Parameters
    ----------
    barcode : list of (birth, death)
        The persistence intervals.
    assume_sorted : bool
        If True, assumes barcode is already sorted by birth time ascending.

    Returns
    -------
    total_codim : int
        Sum over all qualifying pairs.
    """
    if barcode is None:
        return 0

    # Filter malformed intervals and coerce to float
    bars = [(float(b), float(d)) for (b, d) in barcode if b <= d]

    # Sort by birth if needed
    if not assume_sorted:
        bars.sort(key=lambda x: x[0])

    m = len(bars)
    total = 0

    for i in range(m):
        b_i, d_i = bars[i]
        # only check j > i since births are nondecreasing after sort
        for j in range(i + 1, m):
            b_j, d_j = bars[j]

            # early exit: if b_j > d_i, no later bar will work either
            if b_j > d_i:
                break

            total += _pair_codimension(
                (b_i, d_i),
                (b_j, d_j)
            )

    return total

def weighted_quiver_codim(
    barcode: List[Tuple[float, float]],
    assume_sorted: bool = False
) -> float:
    """
    Compute the weighted quiver codimension:

        (1 / L^2) * sum_{i<j} [ len_i * len_j * codim(bar_i, bar_j) ]

    where
        len_i = d_i - b_i,
        L     = max_k len_k  (max bar length in the barcode),
        codim(bar_i, bar_j) is 0/1 from _pair_codimension.

    If barcode is empty, or all bars have zero length so L=0,
    this returns 0.0 to avoid division by zero.

    Parameters
    ----------
    barcode : list of (birth, death)
        Persistence intervals (e.g. H1 barcode from ripser).
    assume_sorted : bool
        If True, we assume `barcode` is already sorted by birth ascending.

    Returns
    -------
    score : float
        Weighted codimension value in [0, +∞), normalized by L^2.
        (If all bars same scale, this puts the score on a comparable scale.)
    """
    if barcode is None:
        return 0.0

    # Filter invalid intervals and coerce to float
    bars = [(float(b), float(d)) for (b, d) in barcode if b <= d]

    if not bars:
        return 0.0

    # Sort by birth if needed
    if not assume_sorted:
        bars.sort(key=lambda x: x[0])

    # Precompute lengths
    lengths = [d - b for (b, d) in bars]

    # Max length L
    L = max(lengths)

    if L <= 0.0:
        # all bars are zero-length or invalid for weighting
        return 0.0

    m = len(bars)
    weighted_sum = 0.0

    for i in range(m):
        b_i, d_i = bars[i]
        len_i = lengths[i]

        for j in range(i + 1, m):
            b_j, d_j = bars[j]
            len_j = lengths[j]

            # pruning: if b_j > d_i, no later bar can satisfy b_k <= d_i
            if b_j > d_i:
                break

            c_ij = _pair_codimension((b_i, d_i), (b_j, d_j))  # 0 or 1
            if c_ij:
                weighted_sum += len_i * len_j

    # normalize
    return weighted_sum / (L * L)