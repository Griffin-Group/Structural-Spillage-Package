# Structural-Spillage-Package
This work builds off Structural spillage: an efficient method to identify non-crystalline topological materials (https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042011) with a more efficient implementation of the structural spillage calculations.

## Method
The quasi-Bloch structural spillage at UC k-point **k** (paper eq. 2b):

$$
\gamma_{\mathrm{qB}}(\mathbf{k}) = \frac{1}{2}\left\lbrace \left[\sum_{G\alpha} P^{\alpha\alpha}_{\mathbf{k+G},\mathbf{k+G}}\right] + \tilde{n}_{\mathrm{occ}}(\mathbf{k}) - \sum_{G\alpha}\sum_{G'\beta}\left[P^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}\tilde{P}^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}} + \tilde{P}^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}P^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}}\right] \right\rbrace
$$

where $P$ is the occupied-subspace projector of the crystalline reference (`--xtal-sc`) and $\tilde{P}$ is the projector of the comparison system (`--amor-sc`), both expressed in the plane-wave basis and unfolded onto the UC k-mesh (`--xtal-uc`). This is implemented in `compute_structural_spillage` in `structural_spillage.py`: `p4` is $\sum_{G\alpha} P^{\alpha\alpha}$, `aa` is $\tilde{n}_{\mathrm{occ}}(\mathbf{k})$, and `p1`/`p2` are the two cross terms $P\tilde{P}$ and $\tilde{P}P$.

$\gamma_{\mathrm{qB}}(\mathbf{k})$ itself (the structural spillage from the paper) and the
per-band structural spillage below are implemented. The per-band spin-orbit spillage and the
Appendix D spin-orbit plane-wave spillage are not yet implemented.

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

| Flag | Reference bands ($\psi_n$) | Comparison subspace ($\psi_m$) | Requires |
|---|---|---|---|
| `--out-per-band` | `--xtal-sc` (SOC) | `--amor-sc` | — |
| `--out-per-band-nosoc` | `--xtal-sc-nosoc`, trivially embedded as $[\psi_n, 0]$ | `--amor-sc` | `--xtal-sc-nosoc` |
| `--out-per-band-nosoc-soc` | `--xtal-sc-nosoc`, trivially embedded as $[\psi_n, 0]$ | `--amor-soc` | `--xtal-sc-nosoc`, `--amor-soc` |

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

### Bismuthene bi-layer (per-band)
The bismuthene case above only exercises `--out-spillage`. With two more WAVECARs it also
covers all three per-band outputs:

- `supercell_nosoc/WAVECAR` — noSOC on the same crystalline `supercell/` structure
  (generated via `submit_bi_xtal_nosoc.sh` in `AmorphousTDA`)
- `dis_soc/WAVECAR` — SOC on the same disordered `dis/` structure
  (generated via `submit_bi_dis_soc.sh`)

Local copies of both live in `test_data/Bi_data/{supercell_nosoc,dis_soc}/WAVECAR`, alongside
the other three, and are what the command below uses.

```
python structural_spillage.py \
  --xtal-uc       test_data/Bi_data/uc/2nd/WAVECAR \
  --uc-soc \
  --xtal-sc       test_data/Bi_data/supercell/WAVECAR \
  --xtal-sc-nosoc test_data/Bi_data/supercell_nosoc/WAVECAR \
  --amor-sc       test_data/Bi_data/dis/WAVECAR \
  --amor-soc      test_data/Bi_data/dis_soc/WAVECAR \
  --out-spillage           tests/bi_smoke/spillage.txt \
  --out-per-band           tests/bi_smoke/per_band_struct_soc_nsoc.txt \
  --out-per-band-nosoc     tests/bi_smoke/per_band_struct_nosoc_nsoc.txt \
  --out-per-band-nosoc-soc tests/bi_smoke/per_band_struct_nosoc_soc.txt
```

Deep valence bands score lowest (least impacted SOC/noSOC and crystal/disorder); states nearest $E_F$ score highest, as
expected for the bands most reshaped by disorder and SOC. 
