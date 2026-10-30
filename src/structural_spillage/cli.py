"""Command-line interface (`structural-spillage`, or `python -m structural_spillage`).

Loads the WAVECARs, runs the computations in spillage.py, and writes the
requested output files.
"""

import argparse

import numpy as np

from .spillage import compute_sopw, compute_structural_spillage, load_inputs


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog='structural-spillage',
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
                          help='Output: spillage per UC k-point normalized by N_occ(k) — '
                               'fraction of the occupied subspace not reproduced, roughly in [0,1]')
    outputs.add_argument('--out-sopw', default='sopw_spillage.txt',
                          help='Output: per-PW spin-orbit spillage (Appendix D, only written if --amor-soc is given)')
    outputs.add_argument('--out-norm-sopw', default='sopw_norm.txt',
                          help='Output: scalar normalized sopw = total_sopw / N_occ (Appendix D)')
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

    args = p.parse_args(argv)
    # fail before loading any WAVECARs
    if args.out_per_band_nosoc is not None and args.xtal_sc_nosoc is None:
        p.error('--out-per-band-nosoc requires --xtal-sc-nosoc')
    if args.out_per_band_nosoc_soc is not None and (args.xtal_sc_nosoc is None or args.amor_soc is None):
        p.error('--out-per-band-nosoc-soc requires --xtal-sc-nosoc and --amor-soc')
    return args


def save_per_band(path, pb, gamma_label):
    """Write a PerBand result: rows = bands, columns =
    [energy_eV, gamma_label, w_k0, w_k1, ..., dominant_k]."""
    k_hdr = '  '.join(f'w_k{i}' for i in range(pb.band_k_weight.shape[1]))
    out = np.column_stack([pb.energies, pb.gamma, pb.band_k_weight, pb.dominant_k])
    np.savetxt(path, out, header=f'energy_eV  {gamma_label}  {k_hdr}  dominant_k')
    print(f"  Saved to {path}  ({len(pb.gamma)} bands)")

    top10 = np.argsort(pb.gamma)[::-1][:10]
    print(f"  Top 10 bands by {gamma_label}:")
    print(f"  {'band':>5}  {'energy(eV)':>12}  {gamma_label:>16}  {'dom_k':>6}")
    for n in top10:
        print(f"  {n:>5}  {pb.energies[n]:>12.4f}  {pb.gamma[n]:>16.4f}  {pb.dominant_k[n]:>6}")


def main(argv=None):
    args = parse_args(argv)
    inp = load_inputs(args.xtal_uc, args.xtal_sc, args.amor_sc,
                      uc_soc=args.uc_soc, amor_ncl=args.amor_ncl, amor_soc=args.amor_soc,
                      xtal_sc_nosoc=args.xtal_sc_nosoc, n1=args.n1, n2=args.n2, n3=args.n3)

    res = compute_structural_spillage(inp.amor, inp.xtal, inp.sortidx, inp.index_list,
                                      uc_kpoints=inp.uc.kpoints, amor_ncl=args.amor_ncl,
                                      xtal_nosoc=inp.xtal_nosoc, amor_soc=inp.amor_soc,
                                      per_band=args.out_per_band is not None,
                                      per_band_nosoc=args.out_per_band_nosoc is not None,
                                      per_band_nosoc_soc=args.out_per_band_nosoc_soc is not None)
    np.savetxt(args.out_spillage, res.spillage)
    np.savetxt(args.out_norm_spillage, res.normalized)
    for path, pb, label in [(args.out_per_band, res.per_band, 'gamma_struct'),
                            (args.out_per_band_nosoc, res.per_band_nosoc, 'gamma_struct_nosoc'),
                            (args.out_per_band_nosoc_soc, res.per_band_nosoc_soc, 'gamma_struct_nosoc_soc')]:
        if pb is not None:
            save_per_band(path, pb, label)

    if inp.amor_soc is not None:
        sopw = compute_sopw(inp.amor, inp.amor_soc, inp.sortidx, inp.index_list,
                            per_band=args.out_per_band_sopw is not None)
        np.savetxt(args.out_sopw, sopw.gamma_pw)
        np.savetxt(args.out_norm_sopw, [sopw.normalized])
        if sopw.per_band is not None:
            save_per_band(args.out_per_band_sopw, sopw.per_band, 'gamma_sopw')


if __name__ == '__main__':
    main()
