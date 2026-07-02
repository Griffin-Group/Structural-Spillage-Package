"""Compute structural spillage between two supercell WAVECARs, unfolded
onto the crystalline UC k-mesh.
"""

import argparse


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
    # wavecar_io.load_uc_kmesh(args.xtal_uc, vasp_type='ncl' if args.uc_soc else 'std')

    # ── load SC WAVECARs (xtal-SC, amor-SC, optional xtal-SC-noSOC) ───────
    # wavecar_io.load_sc_wavecar(...) for each

    # ── match UC plane waves onto SC plane waves ───────────────────────────
    # kmatching.match_uc_to_sc(...)

    # ── build spinor bases and compute plane-wave spillage ─────────────────

    # ── optional per-band outputs ───────────────────────────────────────────

    # ── Appendix D: spin-orbit plane-wave spillage (if --amor-soc given) ───

    raise NotImplementedError


if __name__ == '__main__':
    main()
