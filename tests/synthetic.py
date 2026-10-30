"""Tiny synthetic stand-ins for WAVECAR data, so the spillage math can be
tested in milliseconds without VASP output.

Geometry: cubic UC with b = I, a 2x1x1 supercell, and a 2-point UC k-mesh
(Gamma and X = (1/2,0,0)). Every UC plane wave with G in {-1,0,1}^3 is kept at
each k, so the SC gamma-point basis is exactly the union of the two UC
k-blocks (27 plane waves each, 54 total). The SC plane waves are stored in
shuffled order to exercise the UC->SC matching.

Every band lives inside a single UC k-block (spanned by a random orthonormal
block-local basis u_0, u_1, ...), so the crystalline projector is
block-diagonal in UC k, like a real crystal, and every spillage has a
closed-form value.
"""

import numpy as np

from structural_spillage import Inputs, compute_structural_spillage
from structural_spillage.kmatching import unfold_onto_uc
from structural_spillage.wavecar_io import SCWavecarData, UCKmesh, occupancy_from_energies

N1 = 2
UC_KPOINTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
B_UC = np.eye(3)
B_SC = np.diag([1.0 / N1, 1.0, 1.0])
_G = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
N_PW_BLOCK = len(_G)
N_PW = N1 * N_PW_BLOCK

# raw PW norm below 1, mimicking PAW augmentation; per-band outputs must normalize it away
PW_NORM = 0.9


class Geometry:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        p_full = np.concatenate([(_G + k) @ B_UC for k in UC_KPOINTS])
        self.uc = UCKmesh(b=B_UC, a=np.eye(3), kpoints=UC_KPOINTS, p_full=p_full,
                          index_list=[N_PW_BLOCK] * len(UC_KPOINTS), n_pw_gamma=N_PW_BLOCK)
        # SC plane wave s is UC plane wave perm[s]
        self.perm = rng.permutation(N_PW)
        self.Gpoints_sc = np.round(p_full[self.perm] @ np.linalg.inv(B_SC))
        self.block_idx = [np.where(self.perm // N_PW_BLOCK == ki)[0] for ki in range(len(UC_KPOINTS))]
        self.basis = []
        for _ in UC_KPOINTS:
            M = rng.normal(size=(N_PW_BLOCK, N_PW_BLOCK)) + 1j * rng.normal(size=(N_PW_BLOCK, N_PW_BLOCK))
            self.basis.append(np.linalg.qr(M)[0])

    def state(self, ki, i, rotate_to=None, theta=0.0):
        """SC-ordered PW vector cos(theta) u_i + sin(theta) u_rotate_to in k-block ki."""
        v = np.cos(theta) * self.basis[ki][:, i]
        if rotate_to is not None:
            v = v + np.sin(theta) * self.basis[ki][:, rotate_to]
        out = np.zeros(N_PW, dtype=complex)
        out[self.block_idx[ki]] = PW_NORM * v
        return out


def up(psi):
    return np.stack([psi, np.zeros_like(psi)])


def dn(psi):
    return np.stack([np.zeros_like(psi), psi])


def make_sc(geom, occupied, unoccupied):
    """SCWavecarData (one gamma k-point, ISPIN=1). Bands are (n_pw,) arrays
    for a std WAVECAR or (2, n_pw) spinors for an ncl one."""
    bands = np.array(list(occupied) + list(unoccupied))
    energies = np.concatenate([np.linspace(-2.0, -1.0, len(occupied)),
                               np.linspace(1.0, 2.0, len(unoccupied))])
    occs, gap = occupancy_from_energies(energies, 0.0)
    return SCWavecarData(b=B_SC, a=np.diag([float(N1), 1.0, 1.0]), energies=energies, efermi=0.0,
                         occs=occs, gap=gap, ispin2=False, coeffs=[bands],
                         Gpoints_gamma=geom.Gpoints_sc, kpoint_gamma=np.zeros(3), n_pw_gamma=N_PW)


def amor_nosoc_bands(geom, rotate_first=0.0):
    """Occupied u_0, u_1 in each k-block (4 bands); first optionally rotated toward u_2."""
    occ = [geom.state(ki, i) for ki in range(N1) for i in (0, 1)]
    if rotate_first:
        occ[0] = geom.state(0, 0, rotate_to=2, theta=rotate_first)
    unocc = [geom.state(ki, 5) for ki in range(N1)]
    return occ, unocc


def soc_bands(geom, rotate_first=0.0, orthogonal=False):
    """Spinor set [psi,0], [0,psi] for each amor_nosoc_bands psi (8 bands).

    rotate_first: rotate the first spinor's psi (block 0, u_0, spin up) toward u_2.
    orthogonal: use u_2, u_3 instead of u_0, u_1 (orthogonal to amor_nosoc_bands).
    """
    occ = []
    for ki in range(N1):
        for i in (0, 1):
            psi = geom.state(ki, i + 2 if orthogonal else i)
            occ += [up(psi), dn(psi)]
    if rotate_first:
        occ[0] = up(geom.state(0, 0, rotate_to=2, theta=rotate_first))
    unocc = [s(geom.state(ki, 5)) for ki in range(N1) for s in (up, dn)]
    return occ, unocc


def match(geom, amor):
    return unfold_onto_uc(geom.uc, amor, verbose=False)


def inputs(geom, amor, xtal, xtal_nosoc=None, amor_soc=None):
    """What load_inputs would return for these synthetic WAVECARs."""
    sortidx, index_list = match(geom, amor)
    return Inputs(uc=geom.uc, xtal=xtal, amor=amor, sortidx=sortidx, index_list=index_list,
                  xtal_nosoc=xtal_nosoc, amor_soc=amor_soc)


def run_spillage(geom, amor, xtal, **kw):
    sortidx, index_list = match(geom, amor)
    return compute_structural_spillage(amor, xtal, sortidx, index_list,
                                       uc_kpoints=geom.uc.kpoints, **kw)
