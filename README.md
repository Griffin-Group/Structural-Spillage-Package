# Structural-Spillage-Package
This work builds off Structural spillage: an efficient method to identify non-crystalline topological materials (https://link.aps.org/doi/10.1103/PhysRevResearch.5.L042011) with a more efficient implementation of the structural spillage calculations.

## Method
The quasi-Bloch structural spillage at UC k-point **k** (paper eq. 2b):

$$
\gamma_{\mathrm{qB}}(\mathbf{k}) = \frac{1}{2}\left\lbrace \left[\sum_{G\alpha} P^{\alpha\alpha}_{\mathbf{k+G},\mathbf{k+G}}\right] + \tilde{n}_{\mathrm{occ}}(\mathbf{k}) - \sum_{G\alpha}\sum_{G'\beta}\left[P^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}\tilde{P}^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}} + \tilde{P}^{\alpha\beta}_{\mathbf{k+G},\mathbf{k+G'}}P^{\beta\alpha}_{\mathbf{k+G'},\mathbf{k+G}}\right] \right\rbrace
$$

where $P$ is the occupied-subspace projector of the crystalline reference (`--xtal-sc`) and $\tilde{P}$ is the projector of the comparison system (`--amor-sc`), both expressed in the plane-wave basis and unfolded onto the UC k-mesh (`--xtal-uc`). This is implemented in `compute_structural_spillage` in `structural_spillage.py`: `p4` is $\sum_{G\alpha} P^{\alpha\alpha}$, `aa` is $\tilde{n}_{\mathrm{occ}}(\mathbf{k})$, and `p1`/`p2` are the two cross terms $P\tilde{P}$ and $\tilde{P}P$.

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
