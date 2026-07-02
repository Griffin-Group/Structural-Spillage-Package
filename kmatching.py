"""UC -> SC plane-wave matching (k-unfolding).

Maps each crystalline-UC plane wave onto the corresponding plane wave in
the amorphous SC's gamma-only plane-wave set, so the SC occupied subspace
can be decomposed into UC k-point blocks for spillage.
"""

import numpy as np


def match_uc_to_sc(p_amorph, p_full_uc, b_sc, index_list, n_pw_sc, verbose=True):
    """Match UC plane waves to SC plane waves via integer Miller indices.

    Both `p_amorph` (SC G+k, Cartesian) and `p_full_uc` (UC G+k for all UC
    k-points, concatenated, Cartesian) are converted to integer (h,k,l)
    indices in the SC reciprocal lattice `b_sc`. UC plane waves are then
    matched to SC plane waves by exact integer equality — this avoids hash
    collisions that a scalar coordinate sum (px+py+pz) could produce.

    Parameters
    ----------
    p_amorph : (n_pw_sc, 3) array
        SC gamma-point G+k vectors, Cartesian.
    p_full_uc : (n_pw_uc_total, 3) array
        UC G+k vectors for all UC k-points, concatenated, Cartesian.
    b_sc : (3, 3) array
        SC reciprocal lattice (rows = b1, b2, b3); used to invert both
        `p_amorph` and `p_full_uc` into SC-fractional Miller indices.
    index_list : list[int]
        Number of UC plane waves per UC k-point, in `p_full_uc` order,
        before filtering.
    n_pw_sc : int
        Total number of SC plane waves (for the completeness check).
    verbose : bool
        Print matching diagnostics.

    Returns
    -------
    sortidx : (n_matched,) int array
        For each matched UC plane wave, the index into the SC plane-wave
        arrays (`p_amorph` order) it corresponds to.
    index_list : list[int]
        Number of matched plane waves per UC k-point, rebuilt after
        filtering; same order/length as the input `index_list`.
    n_matched : int
        Total matched plane waves; equals `n_pw_sc` for a correctly set up
        calculation (enforced by the assertion below).
    """
    if verbose:
        print("Building index map...")

    # use the SC's own reciprocal lattice to invert both sets of G+k
    # vectors back to integers, so both are expressed in SC fractional
    # coordinates
    b_inv  = np.linalg.inv(b_sc)
    sc_hkl = np.round(p_amorph  @ b_inv).astype(np.int32)
    uc_hkl = np.round(p_full_uc @ b_inv).astype(np.int32)

    if verbose:
        print(f"  sc_hkl range: {sc_hkl.min(axis=0)} … {sc_hkl.max(axis=0)}")
        print(f"  uc_hkl range: {uc_hkl.min(axis=0)} … {uc_hkl.max(axis=0)}")
        print(f"  sc_hkl first 3: {sc_hkl[:3]}")
        print(f"  uc_hkl first 3: {uc_hkl[:3]}")

    # sanity: sc_hkl should be exact integers (residuals ~ 0); uc_hkl should
    # also be exact integers if the UC k-grid folds cleanly onto the SC gamma
    sc_resid = np.round(p_amorph @ b_inv) - (p_amorph @ b_inv)
    uc_resid = np.round(p_full_uc @ b_inv) - (p_full_uc @ b_inv)
    if verbose:
        print(f"  sc_hkl rounding residual max: {np.max(np.abs(sc_resid)):.4f} (expect < 0.01)")
        print(f"  uc_hkl rounding residual max: {np.max(np.abs(uc_resid)):.4f} (expect < 0.01)")

    # encode (h,k,l) as a single int64 with an offset so all values are
    # non-negative (without offset negative indices collide: e.g.
    # (-10,5,3) == (-9,-995,3) mod 1000)
    offset = int(np.max(np.abs(np.concatenate([sc_hkl, uc_hkl])))) + 1
    n      = 2 * offset + 1
    sc_enc = ((sc_hkl[:, 0] + offset).astype(np.int64) * n**2 +
              (sc_hkl[:, 1] + offset) * n +
              (sc_hkl[:, 2] + offset))
    uc_enc = ((uc_hkl[:, 0] + offset).astype(np.int64) * n**2 +
              (uc_hkl[:, 1] + offset) * n +
              (uc_hkl[:, 2] + offset))

    original_index_list = list(index_list)

    # step 1: find UC plane waves present in SC
    uc_in_sc = np.isin(uc_enc, sc_enc)

    # step 2: build raw sortidx (may have rare duplicates from float->int rounding)
    sort_sc     = np.argsort(sc_enc, kind='stable')
    sorted_sc   = sc_enc[sort_sc]
    raw_sortidx = sort_sc[np.searchsorted(sorted_sc, uc_enc[uc_in_sc])]

    # step 3: deduplicate — keep first UC occurrence per SC index
    _, first = np.unique(raw_sortidx, return_index=True)
    first    = np.sort(first)
    sortidx  = raw_sortidx[first]

    # step 4: propagate deduplication mask back to full UC array
    uc_positions  = np.where(uc_in_sc)[0]
    uc_final_mask = np.zeros(len(uc_in_sc), dtype=bool)
    uc_final_mask[uc_positions[first]] = True

    n_matched = len(sortidx)
    if verbose:
        print(f"  Matched plane waves: {n_matched} / SC: {n_pw_sc}, UC: {len(uc_enc)}")
    # for a correctly set up calculation all SC plane waves must be matched
    # exactly; a shortfall means the UC k-grid does not tile the SC Gamma
    # point cleanly, or ENCUT differs between the two calculations
    assert n_matched == n_pw_sc, \
        f"Incomplete match ({n_matched}/{n_pw_sc}) — check N1, N2, N3 and that ENCUT is identical in all WAVECARs"

    # step 5: rebuild index_list per UC k-point after filtering
    start = 0
    new_index_list = [int(np.sum(uc_final_mask[start:(start := start + l)]))
                       for l in original_index_list]

    assert len(np.unique(sortidx)) == len(sortidx), "sortidx has duplicate SC indices"

    return sortidx, new_index_list, n_matched
