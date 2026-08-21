# Structural-Spillage-Package
This work builds off Structural spillage: an efficient method to identify non-crystalline topological materials (https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042011) with a more efficient implementation of the structural spillage calculations.

Note, this is only an indicator of band inversion in the perturbative limit of a crystalline structure. Work is ongoing to quantify what this limit is (e.g. in terms of local atomic environments/orbitals).

## Method
The quasi-Bloch structural spillage at UC k-point **k** (paper eq. 2b):

$$
\gamma_{\mathrm{qB}}(\mathbf{k}) = \frac{1}{2}\left\lbrace \left[\sum_{G\alpha} P^{\alpha\alpha}_{\mathbf{k+G},\mathbf{k+G}}\right] + \tilde{n}_{\mathrm{occ}}(\mathbf{k}) - \sum_{G\alpha}\sum_{G'\beta}\left[P^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}\tilde{P}^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}} + \tilde{P}^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}P^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}}\right] \right\rbrace
$$

where $P$ is the occupied-subspace projector of the crystalline reference (`--xtal-sc`) and $\tilde{P}$ is the projector of the comparison system (`--amor-sc`), both expressed in the plane-wave basis and unfolded onto the UC k-mesh (`--xtal-uc`). This is implemented in `compute_structural_spillage` in `structural_spillage.py`: `p4` is $\sum_{G\alpha} P^{\alpha\alpha}$, `aa` is $\tilde{n}_{\mathrm{occ}}(\mathbf{k})$, and `p1`/`p2` are the two cross terms $P\tilde{P}$ and $\tilde{P}P$.

$\gamma_{\mathrm{qB}}(\mathbf{k})$ itself (the structural spillage from the paper), the
per-band structural spillage, and the spin-orbit plane-wave spillage (Appendix D, aggregate
and per-band) below are all implemented.

## Per-band structural spillage
Rather than the aggregate $\gamma_{\mathrm{qB}}(\mathbf{k})$ above, `per_band_structural_spillage`
resolves the spillage per individual band $n$ of the crystalline reference:

$$
\gamma_n = 1 - \sum_m \left|\left\langle \psi_n / \lVert\psi_n\rVert \,\middle|\, \psi_m \right\rangle\right|^2
$$

where $\psi_n$ is one occupied band of the reference system (normalized to remove the
sub-unity PW norm left by PAW augmentation) and $\{\psi_m\}$ spans the occupied-subspace
projector of the comparison system. A high $\gamma_n$ means that band $n$ is not
representable in the comparison subspace — e.g. a band pushed into the gap by disorder, or
inverted by SOC. Each output also reports $w_{k_i}$, the raw (un-normalized) PW weight of
band $n$ in each UC k-block, and `dominant_k`, the UC k-point that band unfolds from most
strongly.

Three variants are available, differing in which reference bands and which comparison
subspace are used:

| Flag | Reference bands ($\psi_n$) | Comparison subspace ($\psi_m$) | Requires | Corresponds to |
|---|---|---|---|---|
| `--out-per-band` | `--xtal-sc` (SOC) | `--amor-sc` | — | Per-band breakdown of the main $\gamma_{\mathrm{qB}}(\mathbf{k})$ / `--out-spillage` |
| `--out-per-band-nosoc` | `--xtal-sc-nosoc`, trivially embedded as $[\psi_n, 0]$ | `--amor-sc` | `--xtal-sc-nosoc` | Isolates the structure/disorder axis from SOC entirely — both sides noSOC |
| `--out-per-band-nosoc-soc` | `--xtal-sc-nosoc`, trivially embedded as $[\psi_n, 0]$ | `--amor-soc` | `--xtal-sc-nosoc`, `--amor-soc` | Crystal noSOC vs. disordered+SOC — mixes the structure and SOC axes into one comparison |

## Spin-orbit plane-wave spillage (Appendix D)
A different quantity from $\gamma_{\mathrm{qB}}(\mathbf{k})$ above, despite the similar form.
Where the structural spillage compares two *different* structures (`--xtal-sc` vs
`--amor-sc`) and restricts cross terms to plane waves mapping to the *same* UC k-point,
the spin-orbit plane-wave spillage compares SOC vs noSOC calculations of the *same* structure
(`--amor-soc` vs `--amor-sc`).

$$
\gamma_{\mathrm{sopw}}(\mathbf{G}) = \frac{1}{2}\left\lbrace Q^{\alpha\alpha}_{\mathbf{G},\mathbf{G}} + \tilde{Q}^{\alpha\alpha}_{\mathbf{G},\mathbf{G}} - \sum_{\mathbf{G}'\alpha\beta}\left[Q^{\alpha\beta}_{\mathbf{G},\mathbf{G}'}\tilde{Q}^{\beta\alpha}_{\mathbf{G}',\mathbf{G}} + \tilde{Q}^{\alpha\beta}_{\mathbf{G},\mathbf{G}'}Q^{\beta\alpha}_{\mathbf{G}',\mathbf{G}}\right] \right\rbrace
$$

where $Q$ is the occupied-subspace projector of `--amor-soc` and $\tilde{Q}$ is the projector
of `--amor-sc` (noSOC) — both projectors on the same structure, differing only in whether SOC
was included. The total spin-orbit spillage is $\sum_{\mathbf{G}} \gamma_{\mathrm{sopw}}(\mathbf{G})$
(`--out-sopw`), normalized by $N_{\mathrm{occ}}$ for `--out-norm-sopw`. `compute_sopw` runs automatically whenever `--amor-soc` is given (alongside
`compute_structural_spillage`), and additionally writes `--out-per-band-sopw` — rows are
`--amor-soc` bands, columns `[energy_eV, gamma_sopw, w_k0, ..., dominant_k]` — if that flag
is set, comparing individual `--amor-soc` bands against the `--amor-sc` subspace.

## Requirements
Python 3.9+ (uses the walrus operator). Install dependencies with:
```
pip install -r requirements.txt
```

## Test cases
### Bismuthene bi-layer
We provide the commands to calculate structural spillage for Bismuthene bi-layer for example.

- `uc/2nd/WAVECAR` — crystalline unit cell, 5×5×1 k-mesh, SOC (`vasp_type=ncl`)
- `supercell/WAVECAR` — crystalline 50-atom supercell, gamma-only, SOC
- `dis/WAVECAR` — the same supercell disordered (amorphized), gamma-only, noSOC

Run:
```
python structural_spillage.py \
  --xtal-uc  /global/cfs/cdirs/m4590/spillage_data/Bi_data/uc/2nd/WAVECAR \
  --uc-soc \
  --xtal-sc  /global/cfs/cdirs/m4590/spillage_data/Bi_data/supercell/WAVECAR \
  --amor-sc  /global/cfs/cdirs/m4590/spillage_data/Bi_data/dis/WAVECAR \
  --out-spillage tests/bi_smoke/spillage.txt
```
`--uc-soc` is required because the UC WAVECAR was written with `LSORBIT=.TRUE.`.

We confirm that the value at the $\Gamma$ point (2.37) matches the original spillage code.

### Bi2Se3 (crystalline)
Unlike the bismuthene case above, this compares the crystalline SOC supercell against
the same nominal structure without SOC — no actual disorder — so it isolates the
SOC-driven band inversion at $\Gamma$ that makes Bi2Se3 a topological insulator. This
should yield the same value as pymatgen's spin orbit spillage.

```
python structural_spillage.py \
  --xtal-uc  test_data/r2scan_uc_nosoc/WAVECAR \
  --xtal-sc  /global/cfs/cdirs/m4590/spillage_data/DOS_crys_supercell/soc/WAVECAR \
  --amor-sc  /global/cfs/cdirs/m4590/spillage_data/DOS_crys_supercell/noSOC/WAVECAR \
  --out-spillage tests/bi2se3_smoke/spillage.txt
```
We confirm that the value at the $\Gamma$ point (2.7) matches pymatgen spin orbit spillage.

### Bismuthene bi-layer (per-band + SOPW)
The bismuthene case above only exercises `--out-spillage`. With two more WAVECARs it also
covers all three per-band structural outputs, and — since `--amor-soc` is now present —
`compute_sopw` runs automatically too, covering the Appendix D outputs in the same command:

- `supercell_nosoc/WAVECAR` — noSOC on the same crystalline `supercell/` structure
  (generated via `submit_bi_xtal_nosoc.sh` in `AmorphousTDA`)
- `dis_soc/WAVECAR` — SOC on the same disordered `dis/` structure
  (generated via `submit_bi_dis_soc.sh`)

```
python structural_spillage.py \
  --xtal-uc       test_data/Bi_data/uc/2nd/WAVECAR \
  --uc-soc \
  --xtal-sc       test_data/Bi_data/supercell/WAVECAR \
  --xtal-sc-nosoc test_data/Bi_data/supercell_nosoc/WAVECAR \
  --amor-sc       test_data/Bi_data/dis/WAVECAR \
  --amor-soc      test_data/Bi_data/dis_soc/WAVECAR \
  --out-spillage           tests/bi_smoke/spillage.txt \
  --out-norm-spillage      tests/bi_smoke/spillage_norm.txt \
  --out-per-band           tests/bi_smoke/per_band_struct_soc_nsoc.txt \
  --out-per-band-nosoc     tests/bi_smoke/per_band_struct_nosoc_nsoc.txt \
  --out-per-band-nosoc-soc tests/bi_smoke/per_band_struct_nosoc_soc.txt \
  --out-sopw               tests/bi_smoke/sopw.txt \
  --out-norm-sopw          tests/bi_smoke/sopw_norm.txt \
  --out-per-band-sopw      tests/bi_smoke/per_band_sopw.txt
```

Deep valence bands score lowest (least impacted SOC/noSOC and crystal/disorder); states nearest $E_F$ score highest, as
expected for the bands most reshaped by disorder and SOC. 

## Plotting
`plot_per_band.py` plots any of the per-band txt files above (energy vs gamma), one PNG per
input file, saved alongside each input by default:

```
python plot_per_band.py tests/bi_smoke/per_band_struct_soc_nsoc.txt tests/bi_smoke/per_band_sopw.txt
```

Pass `--efermi <value>` to draw a dashed reference line at $E_F$, and `--out-dir` to write the
PNGs elsewhere. It only reads the `energy_eV`/`gamma_*` columns — the `w_k*`/`dominant_k`
unfolding detail in each file isn't plotted.
