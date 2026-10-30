from pymatgen.io.vasp.outputs import Wavecar
import numpy as np
import gc
import argparse

def orth(A):
    # orthoganlize a matrix A using SVD
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.amax(s) * np.finfo(float).eps
    num = np.sum(s > tol, dtype=int)
    return u[:, :num], num

def _detect_N(kfrac_col, override):
    # detect supercell repeats
    if override is not None:
        return override
    vals = np.abs(kfrac_col)
    nonzero = vals[vals > 1e-6]
    if len(nonzero) == 0:
        return 1
    return int(np.round(1.0 / np.min(nonzero)))
