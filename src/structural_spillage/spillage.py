"""Compute structural spillage between two supercell WAVECARs, unfolded
onto the crystalline UC k-mesh, and the Appendix D spin-orbit plane-wave
spillage.

Functions here return arrays; writing output files is left to the caller
(see cli.py).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kmatching import unfold_onto_uc
from .utils import _detect_N, orth
from .wavecar_io import (SCWavecarData, UCKmesh, first_kpoint_coeffs, load_sc_wavecar,
                         load_uc_kmesh, select_gamma_coeffs)


@dataclass
class Inputs:
    """Loaded WAVECARs plus the UC -> SC plane-wave unfolding indices."""
    uc: UCKmesh
    xtal: SCWavecarData
    amor: SCWavecarData
    sortidx: np.ndarray
    index_list: list
    xtal_nosoc: Optional[SCWavecarData] = None
    amor_soc: Optional[SCWavecarData] = None


@dataclass
class PerBand:
    """Per-band spillage of individual reference bands against a comparison subspace."""
    energies: np.ndarray
    gamma: np.ndarray
    band_k_weight: np.ndarray   # (n_bands, n_uc_k) raw PW weight of each band per UC k-block
    dominant_k: np.ndarray


@dataclass
class SpillageResult:
    spillage: np.ndarray        # gamma_qB(k), one per UC k-point
    nocc: np.ndarray            # N_occ(k) = Tr_k[P_xtal]
    per_band: Optional[PerBand] = None
    per_band_nosoc: Optional[PerBand] = None
    per_band_nosoc_soc: Optional[PerBand] = None

    @property
    def normalized(self):
        """Spillage per UC k-point normalized by N_occ(k), roughly in [0,1]."""
        return self.spillage / self.nocc


@dataclass
class SopwResult:
    gamma_pw: np.ndarray        # gamma_sopw(G), in SC plane-wave order
    total: float
    normalized: float           # total / N_occ
    per_band: Optional[PerBand] = None


def _ncl_occ_ortho_basis(sc, n_pw):
    """Orthonormal occupied-subspace basis from a native NCL/SOC SCWavecarData.

    Ported from the SOC spinor basis construction previously duplicated for
    xtal (--xtal-sc) and, when needed, amor SOC (--amor-soc).
    """
    coeff = first_kpoint_coeffs(sc.coeffs)
    occ_bands = np.where(sc.occs == 1.0)[0]
    V = np.zeros((2 * n_pw, len(occ_bands)), dtype=complex)
    for idx, j in enumerate(occ_bands):
        V[:, idx] = coeff[j].flatten()
    Q, n = orth(V)
    ortho = (Q.T).reshape((n, 2, n_pw))
    return ortho, n


def _trivial_embed_nosoc_bands(sc, n_pw):
    """Occupied std bands trivially embedded as [psi, 0] spinors (ISPIN=1 only).

    Unlike the amor trivial-embedding (Vtriv/Qtriv in compute_structural_spillage),
    this keeps each band distinct rather than SVD-combining them, so per-band
    identity (energy, individual overlap) survives for the per-band outputs.
    """
    assert not sc.ispin2, "trivial [psi,0] embedding here only supports ISPIN=1"
    coeff = select_gamma_coeffs(sc.coeffs, sc.ispin2)[0]   # (n_bands, n_pw)
    occ_bands = np.where(sc.occs == 1.0)[0]
    bands = np.zeros((len(occ_bands), 2, n_pw), dtype=complex)
    for idx, j in enumerate(occ_bands):
        bands[idx, 0, :] = coeff[j]
    return bands, sc.energies[occ_bands]


def per_band_structural_spillage(bands_raw, energies, amor_ortho, sortidx, index_list):
    """Per-band structural spillage of individual `bands_raw` against the
    occupied-subspace projector spanned by `amor_ortho`.

    `bands_raw` — (n_bands, 2, n_pw) raw (un-normalized) spinor coefficients,
    in the SC's own plane-wave order (not yet unfolded onto UC k-blocks).
    `amor_ortho` — (n_amor, 2, n_pw) orthonormal amor occupied-subspace basis,
    already reordered into UC-k-block (sortidx) order.

    gamma_n = 1 - sum_m |<bands_n/||bands_n|| | amor_ortho_m>|^2 — the
    normalization avoids a spurious offset from PAW augmentation (raw PW
    norm is < 1). The per-UC-k weight columns instead use the *raw*
    (unnormalized) PW magnitudes, so they preserve that PW-completeness
    information rather than rescaling every band to unit weight.
    """
    n_bands = bands_raw.shape[0]
    bands_sorted = bands_raw[:, :, sortidx]

    norm_sq = np.einsum('nap, nap -> n', np.conj(bands_sorted), bands_sorted).real
    bands_normed = bands_sorted / np.sqrt(norm_sq)[:, None, None]

    X = np.einsum('nap, map -> nm', np.conj(bands_normed), amor_ortho, optimize="optimal")
    gamma = 1.0 - np.sum(np.abs(X) ** 2, axis=1)

    pw_weight = np.abs(bands_sorted) ** 2
    band_k_weight = np.zeros((n_bands, len(index_list)))
    start = 0
    for ki, l in enumerate(index_list):
        band_k_weight[:, ki] = pw_weight[:, :, start:start + l].sum(axis=(1, 2))
        start += l
    dominant_k = np.argmax(band_k_weight, axis=1)

    return PerBand(energies=energies, gamma=gamma, band_k_weight=band_k_weight, dominant_k=dominant_k)


def load_inputs(xtal_uc, xtal_sc, amor_sc, *, uc_soc=False, amor_ncl=False, amor_soc=None,
                xtal_sc_nosoc=None, n1=None, n2=None, n3=None):
    """Load the UC k-mesh and all SC WAVECARs, run sanity diagnostics, and
    match UC plane waves onto the amorphous SC's plane-wave ordering.

    Arguments mirror the CLI flags (--xtal-uc, --xtal-sc, ...); `amor_soc`
    and `xtal_sc_nosoc` are optional WAVECAR paths.
    """

    # ── load UC k-mesh ────────────────────────────────────────────────────
    print("Loading UC WAVECAR...")
    uc = load_uc_kmesh(xtal_uc, vasp_type='ncl' if uc_soc else 'std')
    N1 = _detect_N(uc.kpoints[:, 0], n1)
    N2 = _detect_N(uc.kpoints[:, 1], n2)
    N3 = _detect_N(uc.kpoints[:, 2], n3)
    print(f"  Supercell repeats: N1={N1}, N2={N2}, N3={N3}")
    b_amorph = np.array([uc.b[0] / N1, uc.b[1] / N2, uc.b[2] / N3])
    print(f"  UC k-points ({len(uc.kpoints)}): first few = "
          f"{[list(k) for k in uc.kpoints[:4]]}{'...' if len(uc.kpoints) > 4 else ''}")
    print(f"  UC pw at gamma (k-pt 0): {uc.n_pw_gamma}")

    # ── load SC WAVECARs (xtal-SC, amor-SC, optional xtal-SC-noSOC) ───────
    print(f"Loading SC B WAVECAR ({'SOC/ncl' if amor_ncl else 'noSOC/std'})...")
    amor = load_sc_wavecar(amor_sc, vasp_type='ncl' if amor_ncl else 'std')
    print(f"  noSOC gap: {amor.gap * 1000:.1f} meV")
    if amor.gap < 0.01:
        print(f"WARNING: noSOC system gap too small ({amor.gap * 1000:.1f} meV) — system may not be insulating")
    print(f"  SC_amor pw at gamma: {amor.n_pw_gamma}")

    print("Loading SOC SC WAVECAR (largest)...")
    xtal = load_sc_wavecar(xtal_sc, vasp_type='ncl')
    print(f"  SOC gap: {xtal.gap * 1000:.1f} meV")
    assert xtal.gap > 0.01, f"SOC system gap too small ({xtal.gap:.4f} eV) — may not be insulating"
    print(f"  SC_xtal pw at gamma: {xtal.n_pw_gamma}")

    xtal_nosoc = None
    if xtal_sc_nosoc is not None:
        print("Loading noSOC xtal SC WAVECAR...")
        xtal_nosoc = load_sc_wavecar(xtal_sc_nosoc, vasp_type='std')
        print(f"  noSOC xtal gap: {xtal_nosoc.gap * 1000:.1f} meV")
        assert xtal_nosoc.gap > 0.01, f"noSOC xtal gap too small ({xtal_nosoc.gap:.4f} eV)"
        assert xtal_nosoc.n_pw_gamma == xtal.n_pw_gamma, \
            f"noSOC xtal n_pw ({xtal_nosoc.n_pw_gamma}) != SOC xtal n_pw ({xtal.n_pw_gamma}) — check ENCUT"

    amor_soc_data = None
    if amor_soc is not None:
        print("Loading amor SOC SC WAVECAR...")
        amor_soc_data = load_sc_wavecar(amor_soc, vasp_type='ncl')
        print(f"  amor SOC gap: {amor_soc_data.gap * 1000:.1f} meV")
        assert amor_soc_data.n_pw_gamma == xtal.n_pw_gamma, \
            f"amor SOC n_pw ({amor_soc_data.n_pw_gamma}) != xtal n_pw ({xtal.n_pw_gamma}) — check ENCUT"

    N_OCC = int(round(np.sum(amor.occs)))
    N_OCC_SOC = int(round(np.sum(xtal.occs)))
    print(f"Occupied bands — noSOC: {N_OCC} per spin, SOC: {N_OCC_SOC} total")

    # ── lattice / geometry diagnostics ──────────────────────────────────────
    print("Lattice diagnostics:")
    uc_lens = np.linalg.norm(uc.a, axis=1)
    # element-wise ratio (NaN for zero-component hex lattice vectors is expected)
    print(f"  SC_xtal / UC ratio (element-wise):\n{xtal.a / uc.a}")
    print(f"  SC_amor / UC ratio (element-wise):\n{amor.a / uc.a}")
    # norm-based ratio avoids NaN — should be [N1, N2, N3]
    print(f"  SC_xtal / UC lattice vector lengths: {np.linalg.norm(xtal.a, axis=1) / uc_lens}")
    print(f"  SC_amor / UC lattice vector lengths: {np.linalg.norm(amor.a, axis=1) / uc_lens}")
    print(f"  b_SC_amor vs b_amorph (b_UC/N) max abs diff: {np.max(np.abs(amor.b - b_amorph)):.3e}")
    print(f"  b_SC_xtal vs b_amorph max abs diff: {np.max(np.abs(xtal.b - b_amorph)):.3e}")

    # ── sanity check: SC xtal and SC amor must agree (both are gamma-point SCs)
    print(f"Plane waves — SC_xtal: {xtal.n_pw_gamma}, SC_amor: {amor.n_pw_gamma}, "
          f"UC total: {sum(uc.index_list)}")
    assert xtal.n_pw_gamma == amor.n_pw_gamma, \
        f"SC xtal ({xtal.n_pw_gamma}) != SC amor ({amor.n_pw_gamma}) — incompatible calculations"

    # ── match UC plane waves onto SC plane waves ───────────────────────────
    sortidx, index_list = unfold_onto_uc(uc, amor)

    return Inputs(uc=uc, xtal=xtal, amor=amor, sortidx=sortidx, index_list=index_list,
                  xtal_nosoc=xtal_nosoc, amor_soc=amor_soc_data)


def compute_structural_spillage(amor, xtal, sortidx, index_list, *, uc_kpoints=None, amor_ncl=False,
                                xtal_nosoc=None, amor_soc=None,
                                per_band=False, per_band_nosoc=False, per_band_nosoc_soc=False):
    """Build the occupied-subspace spinor bases and compute the structural
    spillage (eq 45) via a full matrix-product sum over UC k-blocks.

    This is the tool's namesake quantity. Not to be confused with the
    Appendix D "spin-orbit plane-wave spillage" (sopw, `compute_sopw`),
    which compares SOC vs noSOC states within the amorphous system alone.

    `uc_kpoints` is only used to report the Gamma-point value. The per-band
    flags select which per-band breakdowns to include in the result:
    `per_band` (xtal SOC vs amor), `per_band_nosoc` (xtal noSOC vs amor,
    needs `xtal_nosoc`), `per_band_nosoc_soc` (xtal noSOC vs amor SOC, needs
    `xtal_nosoc` and `amor_soc`).
    """
    if per_band_nosoc and xtal_nosoc is None:
        raise ValueError("per_band_nosoc requires xtal_nosoc")
    if per_band_nosoc_soc and (xtal_nosoc is None or amor_soc is None):
        raise ValueError("per_band_nosoc_soc requires xtal_nosoc and amor_soc")

    N_OCC = int(round(np.sum(amor.occs)))
    N_OCC_SOC = int(round(np.sum(xtal.occs)))

    print("Building coefficient matrices...")
    n_pw = xtal.n_pw_gamma   # SC SOC G-point count (ncl stores same n_pw as std)
    vsa  = 2 * n_pw

    coeff_a = select_gamma_coeffs(amor.coeffs, amor.ispin2)
    coeff_c = first_kpoint_coeffs(xtal.coeffs)

    # SC B spinor basis:
    #   noSOC (default): trivially embed std bands as [ψ,0] and [0,ψ] → 2×N_OCC spinors
    #   SOC (amor_ncl): use native ncl spinors directly → N_OCC_B spinors
    print(f"  Building SC B spinor basis ({'native NCL spinors' if amor_ncl else 'trivial embedding'})...")
    occ_bands_nsoc = np.where(amor.occs == 1.0)[0]
    if amor_ncl:
        # ncl WAVECAR: coeff_a[0] shape (n_bands, 2*n_pw); each band is a flattened 2-spinor
        Vtriv = np.zeros((vsa, len(occ_bands_nsoc)), dtype=complex)
        for idx, n1 in enumerate(occ_bands_nsoc):
            Vtriv[:, idx] = coeff_a[0][n1].flatten()
    else:
        # std WAVECAR: coeff_a shape ISPIN=1→(1,n_band,n_pw); ISPIN=2→(n_spin,1,n_band,n_pw)
        # trivially embed: spin-up → [ψ,0], spin-down → [0,ψ]
        Vtriv = np.zeros((vsa, 2 * len(occ_bands_nsoc)), dtype=complex)
        for idx, n1 in enumerate(occ_bands_nsoc):
            if amor.ispin2:
                Vtriv[0:n_pw,   idx]                      = coeff_a[0, 0, n1]
                Vtriv[n_pw:vsa, idx + len(occ_bands_nsoc)] = coeff_a[1, 0, n1]
            else:
                Vtriv[0:n_pw,   idx]                      = coeff_a[0, n1]
                Vtriv[n_pw:vsa, idx + len(occ_bands_nsoc)] = coeff_a[0, n1]
    Qtriv, ntriv = orth(Vtriv)
    a_nsoc_ortho = (Qtriv.T).reshape((ntriv, 2, n_pw))
    del Vtriv, Qtriv

    print("  Building SOC spinor basis...")
    occ_bands_soc = np.where(xtal.occs == 1.0)[0]
    c_xtal_ortho, nxtal = _ncl_occ_ortho_basis(xtal, n_pw)

    print(f"  Retained vectors after SVD — SOC: {nxtal}, noSOC: {ntriv}")
    print(f"  Expected — SOC: {N_OCC_SOC}, noSOC: {2 * N_OCC} (= 2×{N_OCC} spin-up↑+dn↓ for ISPIN=1)")

    # If ranks differ (disorder pushes in-gap states across the Fermi level), truncate
    # to the smaller subspace and warn — the spillage is still well-defined.
    if nxtal != ntriv:
        n_min = min(nxtal, ntriv)
        print(f"WARNING: SOC rank ({nxtal}) != noSOC rank ({ntriv}); "
              f"truncating both to {n_min} (likely in-gap states near E_F).")
        c_xtal_ortho = c_xtal_ortho[:n_min]
        a_nsoc_ortho = a_nsoc_ortho[:n_min]
        nxtal = ntriv = n_min

    c_xtal_ortho_c = np.conj(c_xtal_ortho)
    a_nsoc_ortho_c = np.conj(a_nsoc_ortho)

    w1_check = np.einsum('nba, nba ->', c_xtal_ortho, c_xtal_ortho_c)
    w2_check = np.einsum('nba, nba ->', a_nsoc_ortho, a_nsoc_ortho_c)
    print(f"  Tr[P_xtal] = {np.real(w1_check):.4f}, expected {N_OCC_SOC}")
    print(f"  Tr[P_amor] = {np.real(w2_check):.4f}, expected {N_OCC}")

    pw_trace_xtal = np.einsum('nba, nba -> a', c_xtal_ortho, c_xtal_ortho_c)
    pw_trace_amor = np.einsum('nba, nba -> a', a_nsoc_ortho, a_nsoc_ortho_c)
    print(f"  Per-pw trace SOC:   min={pw_trace_xtal.min():.4f}, max={pw_trace_xtal.max():.4f}, "
          f"mean={pw_trace_xtal.mean():.4f}, expected ~{N_OCC_SOC / n_pw:.4f}")
    print(f"  Per-pw trace noSOC: min={pw_trace_amor.min():.4f}, max={pw_trace_amor.max():.4f}, "
          f"mean={pw_trace_amor.mean():.4f}, expected ~{N_OCC / n_pw:.4f}")

    print("Computing structural spillage...")
    c_xtal_ortho   = c_xtal_ortho[:, :, sortidx]
    c_xtal_ortho_c = c_xtal_ortho_c[:, :, sortidx]
    a_nsoc_ortho   = a_nsoc_ortho[:, :, sortidx]
    a_nsoc_ortho_c = a_nsoc_ortho_c[:, :, sortidx]

    # aa: diagonal of P_amor summed over ALL plane waves — P_amor is a global
    # projector (amorphous gamma-only SC has no k-block structure).
    aa = np.einsum('mak, mak -> k', a_nsoc_ortho, a_nsoc_ortho_c, optimize="optimal")

    p1 = np.array([], dtype='complex')
    p2 = np.array([], dtype='complex')
    p4 = np.array([], dtype='complex')
    start = 0
    for l in index_list:
        i1, i2 = start, start + l
        p1 = np.append(p1, np.einsum('nak, nbg, mbg, mak -> k',
                                      c_xtal_ortho[:, :, i1:i2], c_xtal_ortho_c[:, :, i1:i2],
                                      a_nsoc_ortho[:, :, i1:i2], a_nsoc_ortho_c[:, :, i1:i2],
                                      optimize="optimal"))
        p2 = np.append(p2, np.einsum('nak, nbg, mbg, mak -> k',
                                      a_nsoc_ortho[:, :, i1:i2], a_nsoc_ortho_c[:, :, i1:i2],
                                      c_xtal_ortho[:, :, i1:i2], c_xtal_ortho_c[:, :, i1:i2],
                                      optimize="optimal"))
        # p4 = Tr_k[P_xtal_k^2] = Tr_k[P_xtal_k] = N_occ(k) per plane wave
        p4 = np.append(p4, np.einsum('nak, nbg, mbg, mak -> k',
                                      c_xtal_ortho[:, :, i1:i2], c_xtal_ortho_c[:, :, i1:i2],
                                      c_xtal_ortho[:, :, i1:i2], c_xtal_ortho_c[:, :, i1:i2],
                                      optimize="optimal"))
        start += l

    spillage_pw = 0.5 * np.real(p4) + 0.5 * np.real(aa) - 0.5 * np.real(p1) - 0.5 * np.real(p2)

    qB_spillage, qB_nocc, start = [], [], 0
    for l in index_list:
        s   = np.sum(spillage_pw[start:start + l])
        n_k = np.sum(np.real(p4[start:start + l]))   # N_occ(k) = Tr_k[P_xtal_k]
        print(f"  k-point spillage: {s:.6f}  N_occ(k)={n_k:.1f}")
        qB_spillage.append(s)
        qB_nocc.append(n_k)
        start += l

    if uc_kpoints is not None:
        gamma_idx = int(np.argmin(np.linalg.norm(uc_kpoints, axis=1)))
        print(f"  Γ-point spillage: {qB_spillage[gamma_idx]:.4f}  N_occ(Γ)={qB_nocc[gamma_idx]:.1f}")

    result = SpillageResult(spillage=np.array(qB_spillage), nocc=np.array(qB_nocc))

    if per_band:
        print("Computing per-band structural spillage (xtal SOC vs amor)...")
        bands_raw = np.stack([coeff_c[j] for j in occ_bands_soc])   # (N_OCC_SOC, 2, n_pw), SC order
        energies = xtal.energies[occ_bands_soc]
        result.per_band = per_band_structural_spillage(bands_raw, energies, a_nsoc_ortho,
                                                       sortidx, index_list)

    if per_band_nosoc:
        print("Computing per-band structural spillage (xtal noSOC vs amor noSOC)...")
        bands_raw, energies = _trivial_embed_nosoc_bands(xtal_nosoc, n_pw)
        result.per_band_nosoc = per_band_structural_spillage(bands_raw, energies, a_nsoc_ortho,
                                                             sortidx, index_list)

    if per_band_nosoc_soc:
        print("Computing per-band structural spillage (xtal noSOC vs amor SOC)...")
        bands_raw, energies = _trivial_embed_nosoc_bands(xtal_nosoc, n_pw)
        amor_soc_ortho, _ = _ncl_occ_ortho_basis(amor_soc, n_pw)
        amor_soc_ortho = amor_soc_ortho[:, :, sortidx]
        result.per_band_nosoc_soc = per_band_structural_spillage(bands_raw, energies, amor_soc_ortho,
                                                                 sortidx, index_list)

    return result


def compute_sopw(amor, amor_soc, sortidx, index_list, *, per_band=False):
    """Appendix D spin-orbit plane-wave spillage: compares the occupied
    subspaces of --amor-soc and --amor-sc (noSOC) — SOC on/off calculations
    of the SAME structure — with no UC k-block restriction, unlike
    compute_structural_spillage's gamma_qB(k), which compares two DIFFERENT
    structures and restricts cross terms to matching UC k-blocks. See the
    README for the full derivation of why these are different quantities.

    `sortidx`/`index_list` are only used for the per-band breakdown
    (`per_band=True`), which compares individual amor SOC bands against the
    amor noSOC subspace.
    """
    print("\n── Appendix D: spin-orbit plane-wave spillage ──────────────────────────")
    n_pw = amor.n_pw_gamma
    vsa = 2 * n_pw

    assert not amor.ispin2, "SOPW trivial embedding currently only supports ISPIN=1 --amor-sc"
    coeff_a = select_gamma_coeffs(amor.coeffs, amor.ispin2)[0]   # (n_bands, n_pw)
    occ_bands_nsoc = np.where(amor.occs == 1.0)[0]
    N_OCC_NSOC = len(occ_bands_nsoc)

    coeff_soc = first_kpoint_coeffs(amor_soc.coeffs)
    occ_bands_soc = np.where(amor_soc.occs == 1.0)[0]
    N_OCC_SOC = len(occ_bands_soc)
    print(f"  Amor SOC:   {N_OCC_SOC} occupied bands, n_pw={n_pw}")
    print(f"  Amor noSOC: {N_OCC_NSOC} occupied bands, n_pw={n_pw}")

    # If ranks don't match the expected 2x (disorder pushes in-gap states across
    # E_F), truncate the SOC side and warn — mirrors the same pattern used for
    # the xtal/amor rank check in compute_structural_spillage.
    if N_OCC_SOC != 2 * N_OCC_NSOC:
        n_min = min(N_OCC_SOC, 2 * N_OCC_NSOC)
        print(f"WARNING: amor SOC bands ({N_OCC_SOC}) != 2×noSOC ({N_OCC_NSOC}); "
              f"truncating SOC to {n_min} (likely in-gap states near E_F).")
        occ_bands_soc = occ_bands_soc[:n_min]
        N_OCC_SOC = n_min

    # native NCL spinors, kept raw (pre-orthonormalization) for the per-band output
    soc_raw = np.stack([coeff_soc[j] for j in occ_bands_soc])   # (N_OCC_SOC, 2, n_pw)
    soc_energies = amor_soc.energies[occ_bands_soc]

    # trivial embedding: band n -> [psi_n, 0] (idx), band n -> [0, psi_n] (idx+N_OCC_NSOC)
    nsoc_raw = np.zeros((2 * N_OCC_NSOC, 2, n_pw), dtype=complex)
    for idx, j in enumerate(occ_bands_nsoc):
        psi = coeff_a[j]
        nsoc_raw[idx, 0, :] = psi
        nsoc_raw[idx + N_OCC_NSOC, 1, :] = psi

    Qsoc, nsoc = orth(soc_raw.reshape(N_OCC_SOC, vsa).T)
    P_soc = (Qsoc.T).reshape((nsoc, 2, n_pw))
    Qnsoc, nnsoc = orth(nsoc_raw.reshape(2 * N_OCC_NSOC, vsa).T)
    P_nsoc = (Qnsoc.T).reshape((nnsoc, 2, n_pw))

    print(f"  Retained vectors — SOC: {nsoc}, noSOC: {nnsoc}, expected: {N_OCC_SOC}")
    if not (nsoc == nnsoc == N_OCC_SOC):
        n_min = min(nsoc, nnsoc, N_OCC_SOC)
        print(f"WARNING: SOPW rank mismatch (SOC={nsoc}, noSOC={nnsoc}, expected={N_OCC_SOC}); "
              f"truncating to {n_min}.")
        P_soc, P_nsoc = P_soc[:n_min], P_nsoc[:n_min]
        nsoc = nnsoc = N_OCC_SOC = n_min

    P_soc_c  = np.conj(P_soc)
    P_nsoc_c = np.conj(P_nsoc)

    w1_check = np.einsum('nba, nba ->', P_soc, P_soc_c)
    w2_check = np.einsum('nba, nba ->', P_nsoc, P_nsoc_c)
    print(f"  Tr[P_soc]  = {np.real(w1_check):.4f}, expected {N_OCC_SOC}")
    print(f"  Tr[P_nsoc] = {np.real(w2_check):.4f}, expected {N_OCC_SOC}")

    print("Computing spin-orbit plane-wave spillage...")
    # No UC k-block restriction — G' is summed over the full plane-wave basis
    # unconditionally, unlike compute_structural_spillage's per-block loop.
    w1 = np.einsum('nap, nbg, mbg, map -> p', P_soc,  P_soc_c,  P_soc,  P_soc_c,  optimize="optimal")
    w2 = np.einsum('nap, nbg, mbg, map -> p', P_nsoc, P_nsoc_c, P_nsoc, P_nsoc_c, optimize="optimal")
    w3 = np.einsum('nap, nbg, mbg, map -> p', P_soc,  P_soc_c,  P_nsoc, P_nsoc_c, optimize="optimal")
    w4 = np.einsum('nap, nbg, mbg, map -> p', P_nsoc, P_nsoc_c, P_soc,  P_soc_c,  optimize="optimal")

    gamma_sopw = 0.5 * np.real(w1) + 0.5 * np.real(w2) - 0.5 * np.real(w3) - 0.5 * np.real(w4)
    total_sopw = np.sum(gamma_sopw)
    norm_sopw  = total_sopw / N_OCC_SOC

    print(f"  Spin-orbit PW spillage (Appendix D): {total_sopw:.6f}")
    print(f"  Normalized sopw (total / N_occ): {norm_sopw:.6f}  (max=1 if SOC fully reshapes subspace)")

    result = SopwResult(gamma_pw=gamma_sopw, total=total_sopw, normalized=norm_sopw)

    if per_band:
        print("Computing per-band spin-orbit spillage...")
        result.per_band = per_band_structural_spillage(soc_raw, soc_energies, P_nsoc[:, :, sortidx],
                                                       sortidx, index_list)

    return result
