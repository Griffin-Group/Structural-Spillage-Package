"""Compute structural spillage between two supercell WAVECARs, unfolded
onto the crystalline UC k-mesh.
"""

import argparse

import numpy as np

from kmatching import match_uc_to_sc
from utils import _detect_N
from wavecar_io import load_sc_wavecar, load_uc_kmesh


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
    outputs.add_argument('--out-norm-spillage', default='spillage_norm.txt',
                          help='Output: spillage normalized by N_occ(k) — fraction in [0,1]')
    outputs.add_argument('--out-sopw', default='sopw_spillage.txt',
                          help='Output: per-PW spin-orbit spillage (Appendix D, only written if --amor-soc is given)')
    outputs.add_argument('--out-norm-sopw', default='sopw_norm.txt',
                          help='Output: scalar normalized sopw = total_sopw / N_OCC_SOC_AMOR (Appendix D)')
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


def main():
    args = parse_args()

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

    # ── build spinor bases and compute plane-wave spillage ─────────────────

    # ── optional per-band outputs ───────────────────────────────────────────

    # ── Appendix D: spin-orbit plane-wave spillage (if --amor-soc given) ───

    raise NotImplementedError


if __name__ == '__main__':
    main()
