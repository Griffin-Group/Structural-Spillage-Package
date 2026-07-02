"""WAVECAR loading helpers.

VASP does not reliably write occupancies to WAVECAR; every loader here
derives them from band energies vs the Fermi level instead (both are
stored reliably in the file). A small tolerance avoids miscounting bands
right at the Fermi level.
"""

import gc
from dataclasses import dataclass

import numpy as np
from pymatgen.io.vasp.outputs import Wavecar

OCC_TOL = 1e-3


def band_energies_gamma(wc):
    """(n_bands,) energies at k-point 0, spin-up, from a loaded Wavecar.

    band_energy shape is (n_kpoints, n_bands, n_data) for ISPIN=1,
    (n_spin, n_kpoints, n_bands, n_data) for ISPIN=2. Spin-down is
    degenerate for non-magnetic systems, so spin-up is used in both cases.
    Returns (energies, ispin2) — callers need `ispin2` to interpret the
    corresponding `wc.coeffs` via `select_gamma_coeffs`.
    """
    be = np.array(wc.band_energy)
    ispin2 = (be.ndim == 4)
    energies = be[0, 0, :, 0] if ispin2 else be[0, :, 0]
    return energies, ispin2


def occupancy_from_energies(energies, efermi, tol=OCC_TOL):
    """Occupation mask (1.0/0.0) and HOMO-LUMO gap from energies vs E_F."""
    occs = (energies < efermi + tol).astype(float)
    gap = energies[occs == 0].min() - energies[occs == 1].max()
    return occs, gap


def select_gamma_coeffs(coeffs, ispin2):
    """Gamma-point (k=0) coefficients as a plain array, handling ISPIN.

    Multi-k WAVECARs store a different n_pw per k-point, so converting
    `coeffs` directly to an array fails unless gamma is selected first
    (`np.array([coeffs[0]])`, ISPIN=1 case). ISPIN=2 gamma-only WAVECARs
    are already a single k-point, so the full conversion is safe as-is.
    """
    return np.array(coeffs) if ispin2 else np.array([coeffs[0]])


def first_kpoint_coeffs(coeffs):
    """Coefficients at the first (gamma) k-point, without the outer k-axis.

    Used for NCL/SOC WAVECARs, where each band is already a flattened
    2-spinor and downstream code indexes bands directly (`coeffs[j]`)
    rather than through a k-index.
    """
    return np.array(coeffs[0])


@dataclass
class SCWavecarData:
    """Everything needed from one occupied-subspace supercell WAVECAR."""
    b: np.ndarray
    a: np.ndarray
    energies: np.ndarray
    efermi: float
    occs: np.ndarray
    gap: float
    ispin2: bool
    coeffs: object          # raw wc.coeffs; shape/handling depends on ispin2 and vasp_type
    Gpoints_gamma: np.ndarray
    kpoint_gamma: np.ndarray
    n_pw_gamma: int


def load_sc_wavecar(filename, vasp_type='std', verbose=False):
    """Load a supercell WAVECAR and derive its occupied subspace.

    Frees the underlying Wavecar object before returning — only the
    fields needed downstream are kept, since WAVECARs can be large.
    """
    wc = Wavecar(filename=filename, verbose=verbose, vasp_type=vasp_type)
    b = np.array(wc.b)
    a = np.array(wc.a)
    energies, ispin2 = band_energies_gamma(wc)
    efermi = wc.efermi
    occs, gap = occupancy_from_energies(energies, efermi)
    coeffs = wc.coeffs
    Gpoints_gamma = np.array(wc.Gpoints[0])
    kpoint_gamma = np.array(wc.kpoints[0])
    n_pw_gamma = len(Gpoints_gamma)

    data = SCWavecarData(b=b, a=a, energies=energies, efermi=efermi,
                          occs=occs, gap=gap, ispin2=ispin2, coeffs=coeffs,
                          Gpoints_gamma=Gpoints_gamma, kpoint_gamma=kpoint_gamma,
                          n_pw_gamma=n_pw_gamma)
    del wc
    gc.collect()
    return data


@dataclass
class UCKmesh:
    """UC k-mesh data needed to unfold SC plane waves onto UC k-blocks."""
    b: np.ndarray
    a: np.ndarray
    kpoints: np.ndarray
    p_full: np.ndarray      # Cartesian G+k vectors, concatenated over all k-points
    index_list: list        # number of plane waves per UC k-point, in `p_full` order
    n_pw_gamma: int


def load_uc_kmesh(filename, vasp_type='std', verbose=False):
    """Load a UC WAVECAR and extract its k-mesh for unfolding.

    Only Gpoints and kpoints are used; std and ncl WAVECARs give identical
    G-vectors, so `vasp_type` should just match how the file was written.
    """
    wc = Wavecar(filename=filename, verbose=verbose, vasp_type=vasp_type)
    b = np.array(wc.b)
    a = np.array(wc.a)
    kpoints = np.array(wc.kpoints)

    p_chunks, index_list = [], []
    for i in range(len(kpoints)):
        k_cart = np.dot(kpoints[i], wc.b)
        p_chunks.append(np.dot(wc.Gpoints[i], wc.b) + k_cart)
        index_list.append(len(wc.Gpoints[i]))
    p_full = np.concatenate(p_chunks, axis=0)
    n_pw_gamma = len(wc.Gpoints[0])

    data = UCKmesh(b=b, a=a, kpoints=kpoints, p_full=p_full,
                   index_list=index_list, n_pw_gamma=n_pw_gamma)
    del wc
    gc.collect()
    return data
