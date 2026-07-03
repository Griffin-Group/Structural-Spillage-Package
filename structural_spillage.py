"""Compute structural spillage between two supercell WAVECARs, unfolded
onto the crystalline UC k-mesh.
"""

import argparse

import numpy as np

from kmatching import match_uc_to_sc
from utils import _detect_N, orth
from wavecar_io import (first_kpoint_coeffs, load_sc_wavecar, load_uc_kmesh,
                         select_gamma_coeffs)


def parse_args():
    p = argparse.ArgumentParser(
        description='Compute structural spillage between two supercell WAVECARs, '
                     'unfolded onto the crystalline UC k-mesh.')

    required = p.add_argument_group('required inputs')
    required.add_argument('--xtal-sc', required=True,
                           help='Crystalline supercell WAVECAR (SOC, gamma-only); defines the reference occupied subspace')
    required.add_argument('--amor-sc', required=True,
                           help='Amorphous (or second crystalline) supercell WAVECAR (noSOC by default, gamma-only); '
                                'add --amor-ncl if this is also an SOC WAVECAR')
    required.add_argument('--xtal-uc', required=True,
                           help='Crystalline unit cell WAVECAR (N1×N2×N3 k-mesh); defines unit cell k-point blocks for unfolding')

    soc = p.add_argument_group('SOC handling / optional WAVECARs')
    soc.add_argument('--uc-soc', action='store_true',
                      help='Treat --xtal-uc as an SOC WAVECAR (vasp_type=ncl). '
                           'Default is std (noSOC). Only Gpoints and kpoints are read from UC — '
                           'both formats give identical G-vectors.')
    soc.add_argument('--amor-ncl', action='store_true',
                      help='Treat --amor-sc as an SOC WAVECAR; uses native spinors instead of trivial embedding')
    soc.add_argument('--amor-soc', default=None,
                      help='Amorphous SOC WAVECAR (SOC, gamma-only); enables Appendix D spin-orbit PW spillage')
    soc.add_argument('--xtal-sc-nosoc', default=None,
                      help='Crystalline noSOC SC WAVECAR (std, gamma-only). When provided, also computes '
                           'xtal-noSOC vs amor-noSOC and xtal-noSOC vs amor-SOC per-band overlaps.')

    outputs = p.add_argument_group('output files')
    outputs.add_argument('--out-spillage', default='spillage.txt',
                          help='Output: raw spillage per UC k-point')
    outputs.add_argument('--out-sopw', default='sopw_spillage.txt',
                          help='Output: per-PW spin-orbit spillage (Appendix D, only written if --amor-soc is given)')
    outputs.add_argument('--out-per-band-sopw', default=None,
                          help='Output: per-band spin-orbit spillage — rows are amor SOC bands, '
                               'columns are [energy_eV, gamma_sopw]. '
                               'gamma_n = 1 - sum_m |<psi_SOC_n | psi_noSOC_m>|^2; '
                               'high gamma near E_F signals band inversion.')
    outputs.add_argument('--out-per-band', default=None,
                          help='Output: per-band structural spillage — rows are xtal SOC bands, '
                               'columns are [energy_eV, gamma_struct, w_k0, w_k1, ..., dominant_k]. '
                               'gamma_n = 1 - sum_m |<psi_xtal_n/||psi_xtal_n|| | psi_amor_m>|^2; '
                               'high gamma means that xtal band is not representable in the amorphous subspace.')
    outputs.add_argument('--out-per-band-nosoc', default=None,
                          help='Output: per-band structural spillage, xtal noSOC vs amor noSOC. '
                               'gamma_n = 1 - sum_m |<[psi_n,0]/||psi_n|| | a_nsoc_m>|^2. '
                               'Requires --xtal-sc-nosoc.')
    outputs.add_argument('--out-per-band-nosoc-soc', default=None,
                          help='Output: per-band structural spillage, xtal noSOC vs amor SOC. '
                               'gamma_n = 1 - sum_m |<[psi_n,0]/||psi_n|| | a_soc_m>|^2. '
                               'Requires --xtal-sc-nosoc and --amor-soc.')

    supercell = p.add_argument_group('supercell size overrides')
    supercell.add_argument('--n1', type=int, default=None,
                            help='Supercell repeat along a1 (auto-detected from UC k-points if omitted)')
    supercell.add_argument('--n2', type=int, default=None,
                            help='Supercell repeat along a2 (auto-detected from UC k-points if omitted)')
    supercell.add_argument('--n3', type=int, default=None,
                            help='Supercell repeat along a3 (auto-detected from UC k-points if omitted)')

    return p.parse_args()


def load_inputs(args):
    """Load the UC k-mesh and all SC WAVECARs, run sanity diagnostics, and
    match UC plane waves onto the amorphous SC's plane-wave ordering.

    Returns (uc, amor, xtal, xtal_nosoc, sortidx, index_list, N_OCC, N_OCC_SOC).
    `xtal_nosoc` is None unless --xtal-sc-nosoc was given.
    """

    # ── load UC k-mesh ────────────────────────────────────────────────────
    print("Loading UC WAVECAR...")
    uc = load_uc_kmesh(args.xtal_uc, vasp_type='ncl' if args.uc_soc else 'std')
    N1 = _detect_N(uc.kpoints[:, 0], args.n1)
    N2 = _detect_N(uc.kpoints[:, 1], args.n2)
    N3 = _detect_N(uc.kpoints[:, 2], args.n3)
    print(f"  Supercell repeats: N1={N1}, N2={N2}, N3={N3}")
    b_amorph = np.array([uc.b[0] / N1, uc.b[1] / N2, uc.b[2] / N3])
    print(f"  UC k-points ({len(uc.kpoints)}): first few = "
          f"{[list(k) for k in uc.kpoints[:4]]}{'...' if len(uc.kpoints) > 4 else ''}")
    print(f"  UC pw at gamma (k-pt 0): {uc.n_pw_gamma}")

    # ── load SC WAVECARs (xtal-SC, amor-SC, optional xtal-SC-noSOC) ───────
    print(f"Loading SC B WAVECAR ({'SOC/ncl' if args.amor_ncl else 'noSOC/std'})...")
    amor = load_sc_wavecar(args.amor_sc, vasp_type='ncl' if args.amor_ncl else 'std')
    print(f"  noSOC gap: {amor.gap * 1000:.1f} meV")
    if amor.gap < 0.01:
        print(f"WARNING: noSOC system gap too small ({amor.gap * 1000:.1f} meV) — system may not be insulating")
    print(f"  SC_amor pw at gamma: {amor.n_pw_gamma}")

    print("Loading SOC SC WAVECAR (largest)...")
    xtal = load_sc_wavecar(args.xtal_sc, vasp_type='ncl')
    print(f"  SOC gap: {xtal.gap * 1000:.1f} meV")
    assert xtal.gap > 0.01, f"SOC system gap too small ({xtal.gap:.4f} eV) — may not be insulating"
    print(f"  SC_xtal pw at gamma: {xtal.n_pw_gamma}")

    xtal_nosoc = None
    if args.xtal_sc_nosoc is not None:
        print("Loading noSOC xtal SC WAVECAR...")
        xtal_nosoc = load_sc_wavecar(args.xtal_sc_nosoc, vasp_type='std')
        print(f"  noSOC xtal gap: {xtal_nosoc.gap * 1000:.1f} meV")
        assert xtal_nosoc.gap > 0.01, f"noSOC xtal gap too small ({xtal_nosoc.gap:.4f} eV)"
        assert xtal_nosoc.n_pw_gamma == xtal.n_pw_gamma, \
            f"noSOC xtal n_pw ({xtal_nosoc.n_pw_gamma}) != SOC xtal n_pw ({xtal.n_pw_gamma}) — check ENCUT"

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
    # p_amorph: SC gamma-point G+k vectors, Cartesian (uses the amorphous
    # SC's own reciprocal lattice, matching kmatching's expectation that
    # both p_amorph and p_full are expressed in the same basis).
    p_amorph = amor.Gpoints_gamma @ amor.b + amor.kpoint_gamma @ amor.b
    sortidx, index_list, n_matched = match_uc_to_sc(
        p_amorph, uc.p_full, amor.b, uc.index_list, xtal.n_pw_gamma)

    return uc, amor, xtal, xtal_nosoc, sortidx, index_list, N_OCC, N_OCC_SOC


def compute_structural_spillage(amor, xtal, sortidx, index_list, uc_kpoints, N_OCC, N_OCC_SOC, args):
    """Build the occupied-subspace spinor bases and compute the structural
    spillage (eq 45) via a full matrix-product sum over UC k-blocks.

    This is the tool's namesake quantity. Not to be confused with the
    Appendix D "spin-orbit plane-wave spillage" (sopw) computed elsewhere,
    which compares SOC vs noSOC states within the amorphous system alone.
    Writes `args.out_spillage` and returns `qB_spillage` (raw spillage per
    UC k-point, in the same order as `uc_kpoints`/`index_list`).
    """
    print("Building coefficient matrices...")
    n_pw = xtal.n_pw_gamma   # SC SOC G-point count (ncl stores same n_pw as std)
    vsa  = 2 * n_pw

    coeff_a = select_gamma_coeffs(amor.coeffs, amor.ispin2)
    coeff_c = first_kpoint_coeffs(xtal.coeffs)

    # SC B spinor basis:
    #   noSOC (default): trivially embed std bands as [ψ,0] and [0,ψ] → 2×N_OCC spinors
    #   SOC (--amor-ncl): use native ncl spinors directly → N_OCC_B spinors
    print(f"  Building SC B spinor basis ({'native NCL spinors' if args.amor_ncl else 'trivial embedding'})...")
    occ_bands_nsoc = np.where(amor.occs == 1.0)[0]
    if args.amor_ncl:
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
    Vxtal = np.zeros((vsa, len(occ_bands_soc)), dtype=complex)
    for idx, j in enumerate(occ_bands_soc):
        Vxtal[:, idx] = coeff_c[j].flatten()
    Qxtal, nxtal = orth(Vxtal)
    c_xtal_ortho = (Qxtal.T).reshape((nxtal, 2, n_pw))
    del Vxtal, Qxtal

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

    gamma_idx = int(np.argmin(np.linalg.norm(uc_kpoints, axis=1)))
    print(f"  Γ-point spillage: {qB_spillage[gamma_idx]:.4f}  N_occ(Γ)={qB_nocc[gamma_idx]:.1f}")
    np.savetxt(args.out_spillage, qB_spillage)
    return qB_spillage


def main():
    args = parse_args()
    uc, amor, xtal, xtal_nosoc, sortidx, index_list, N_OCC, N_OCC_SOC = load_inputs(args)
    compute_structural_spillage(amor, xtal, sortidx, index_list, uc.kpoints, N_OCC, N_OCC_SOC, args)

    # ── optional per-band outputs — not yet implemented ─────────────────────

    # ── Appendix D: spin-orbit plane-wave spillage — not yet implemented ────


if __name__ == '__main__':
    main()
