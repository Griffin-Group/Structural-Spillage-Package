"""Unit tests for the small helpers (no WAVECARs needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

import synthetic as syn
from structural_spillage.cli import parse_args
from structural_spillage.kmatching import match_uc_to_sc
from structural_spillage.plotting import plot_per_band
from structural_spillage.utils import _detect_N, orth
from structural_spillage.wavecar_io import band_energies_gamma, occupancy_from_energies


def test_orth_drops_dependent_columns():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(20, 3)) + 1j * rng.normal(size=(20, 3))
    A = np.column_stack([A, A[:, 0] + 2 * A[:, 1]])   # 4th column is dependent
    Q, n = orth(A)
    assert n == 3
    assert np.allclose(Q.conj().T @ Q, np.eye(3))
    # same column space: projecting A onto Q loses nothing
    assert np.allclose(Q @ (Q.conj().T @ A), A)


@pytest.mark.parametrize("kcol, override, expected", [
    ([0.0, 0.2, 0.4, -0.4, -0.2], None, 5),
    ([0.0, 0.5], None, 2),
    ([0.0, 0.0], None, 1),
    ([0.0, 0.5], 4, 4),
])
def test_detect_N(kcol, override, expected):
    assert _detect_N(np.array(kcol), override) == expected


def test_occupancy_from_energies_tolerance():
    # a band 0.5 meV above E_F is within OCC_TOL and counts as occupied
    occs, gap = occupancy_from_energies(np.array([-2.0, -1.0, 0.0005, 1.0]), 0.0)
    assert list(occs) == [1, 1, 1, 0]
    assert gap == pytest.approx(1.0 - 0.0005)


def test_band_energies_gamma_ispin():
    nb = 4
    be1 = np.zeros((3, nb, 3)); be1[0, :, 0] = np.arange(nb)
    e, ispin2 = band_energies_gamma(SimpleNamespace(band_energy=be1))
    assert not ispin2 and list(e) == [0, 1, 2, 3]

    be2 = np.zeros((2, 1, nb, 3)); be2[0, 0, :, 0] = -np.arange(nb)
    e, ispin2 = band_energies_gamma(SimpleNamespace(band_energy=be2))
    assert ispin2 and list(e) == [0, -1, -2, -3]


def test_match_recovers_permutation():
    geom = syn.Geometry(seed=3)
    p_sc = geom.Gpoints_sc @ syn.B_SC
    sortidx, index_list, n = match_uc_to_sc(p_sc, geom.uc.p_full, syn.B_SC,
                                            geom.uc.index_list, syn.N_PW, verbose=False)
    assert n == syn.N_PW
    assert index_list == [syn.N_PW_BLOCK, syn.N_PW_BLOCK]
    # UC plane wave u sits at SC index sortidx[u]
    assert np.allclose(p_sc[sortidx], geom.uc.p_full)
    assert np.array_equal(geom.perm[sortidx], np.arange(syn.N_PW))


def test_match_fails_on_missing_sc_planewave():
    geom = syn.Geometry()
    # UC k-mesh missing one plane wave the SC has (e.g. ENCUT mismatch)
    p_uc = geom.uc.p_full[1:]
    index_list = [syn.N_PW_BLOCK - 1, syn.N_PW_BLOCK]
    with pytest.raises(AssertionError, match="Incomplete match"):
        match_uc_to_sc(geom.Gpoints_sc @ syn.B_SC, p_uc, syn.B_SC, index_list, syn.N_PW, verbose=False)


REQUIRED = ["--xtal-uc", "uc", "--xtal-sc", "sc", "--amor-sc", "amor"]


@pytest.mark.parametrize("extra", [
    ["--out-per-band-nosoc", "x.txt"],
    ["--out-per-band-nosoc-soc", "x.txt", "--xtal-sc-nosoc", "n"],
    ["--out-per-band-nosoc-soc", "x.txt", "--amor-soc", "s"],
])
def test_cli_rejects_per_band_without_inputs(extra):
    # checked before any WAVECAR is loaded
    with pytest.raises(SystemExit):
        parse_args(REQUIRED + extra)


def test_plot_per_band_writes_png(tmp_path):
    path = tmp_path / "per_band.txt"
    data = np.column_stack([np.linspace(-3, 0, 6), np.linspace(0, 1, 6), np.ones(6), np.zeros(6)])
    np.savetxt(path, data, header="energy_eV  gamma_struct  w_k0  dominant_k")
    out = plot_per_band(str(path), efermi=-0.5)
    assert out.endswith("per_band.png")
    assert (tmp_path / "per_band.png").stat().st_size > 0
