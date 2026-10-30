"""Plot per-band spillage outputs (energy vs gamma), one figure per input file.

Reads the txt files written by per_band_structural_spillage (--out-per-band,
--out-per-band-nosoc, --out-per-band-nosoc-soc, --out-per-band-sopw): rows are
bands, columns are [energy_eV, gamma_*, w_k0, ..., w_kN, dominant_k]. Only the
first two columns (energy, gamma) are plotted; w_k*/dominant_k are per-band
UC-k unfolding detail, not part of this summary view.
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# dataviz reference palette (references/palette.md): categorical slot 1 (blue)
# for the single-series marker, slot 8 (red) reserved for the optional E_F line.
MARKER_COLOR = '#2a78d6'
EF_COLOR     = '#e34948'
TEXT_COLOR   = '#0b0b0b'
GRID_COLOR   = '#dddddd'


def _read_header(path):
    with open(path) as f:
        first = f.readline()
    assert first.startswith('#'), \
        f"{path}: expected a '# ...' header line from per_band_structural_spillage"
    cols = first.lstrip('#').split()
    return cols[0], cols[1]   # energy column name, gamma column name


def plot_per_band(path, out_dir=None, efermi=None, dpi=150):
    energy_col, gamma_col = _read_header(path)
    data = np.loadtxt(path)
    energy, gamma = data[:, 0], data[:, 1]

    # energy_col is a raw absolute VASP eigenvalue (efermi is only used upstream
    # to pick occupied bands, never subtracted before saving) — shift here so
    # the x-axis reads relative to E_F, the physically meaningful reference.
    if efermi is not None:
        energy = energy - efermi
        xlabel = f'{energy_col} $- E_F$ (eV)'
    else:
        xlabel = f'{energy_col} (eV)'

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(energy, gamma, s=18, c=MARKER_COLOR, alpha=0.6, linewidths=0)
    if efermi is not None:
        ax.axvline(0.0, color=EF_COLOR, linestyle='--', linewidth=1.2, label='$E_F$')
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)

    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(gamma_col, color=TEXT_COLOR)
    ax.set_title(os.path.splitext(os.path.basename(path))[0], color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)
    fig.tight_layout()

    out_dir = out_dir or os.path.dirname(path) or '.'
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Plot per-band spillage (energy vs gamma), one PNG per input file.')
    p.add_argument('files', nargs='+',
                    help='Per-band output txt file(s), e.g. per_band_struct_soc_nsoc.txt')
    p.add_argument('--out-dir', default=None,
                    help='Directory for output PNGs (default: alongside each input file)')
    p.add_argument('--efermi', type=float, default=None,
                    help='Optional Fermi level (eV) to mark with a dashed line')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    for f in args.files:
        plot_per_band(f, out_dir=args.out_dir, efermi=args.efermi, dpi=args.dpi)


if __name__ == '__main__':
    main()
