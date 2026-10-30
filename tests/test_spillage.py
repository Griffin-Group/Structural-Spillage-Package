"""End-to-end spillage tests on synthetic subspaces with closed-form answers.

Setup (see synthetic.py): 2 UC k-blocks; amor noSOC occupies u_0, u_1 in each
block, so with trivial [psi,0]/[0,psi] embedding each block holds 4 occupied
spinors. Rotating one spinor by theta toward an unoccupied direction changes
Tr[P Q] by sin^2(theta), so the spillage of that block is sin^2(theta).
"""

import numpy as np
import pytest

import synthetic as syn
from structural_spillage import cli, compute_sopw, compute_structural_spillage

THETA = 0.3
S2 = np.sin(THETA) ** 2


@pytest.fixture
def geom():
    return syn.Geometry(seed=0)


@pytest.fixture
def amor(geom):
    return syn.make_sc(geom, *syn.amor_nosoc_bands(geom))


def test_identical_subspaces_zero_spillage(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom))
    res = syn.run_spillage(geom, amor, xtal)
    assert np.allclose(res.spillage, 0.0, atol=1e-10)
    assert np.allclose(res.normalized, 0.0, atol=1e-10)


def test_one_rotated_band(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom, rotate_first=THETA))
    res = syn.run_spillage(geom, amor, xtal)
    # only k-block 0 (Gamma) is affected; 4 occupied spinors per block
    assert np.allclose(res.spillage, [S2, 0.0])
    assert np.allclose(res.nocc, [4.0, 4.0])
    assert np.allclose(res.normalized, [S2 / 4, 0.0])


def test_orthogonal_subspaces_max_spillage(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom, orthogonal=True))
    res = syn.run_spillage(geom, amor, xtal)
    # 1/2 (Tr P + Tr Q) per block, normalized spillage = 1
    assert np.allclose(res.spillage, [4.0, 4.0])
    assert np.allclose(res.normalized, [1.0, 1.0])


def test_amor_ncl_native_spinors(geom):
    # amor_ncl: a SOC amor WAVECAR identical to the xtal gives zero spillage
    xtal = syn.make_sc(geom, *syn.soc_bands(geom))
    amor_ncl = syn.make_sc(geom, *syn.soc_bands(geom))
    res = syn.run_spillage(geom, amor_ncl, xtal, amor_ncl=True)
    assert np.allclose(res.spillage, 0.0, atol=1e-10)


def test_per_band_struct(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom, rotate_first=THETA))
    pb = syn.run_spillage(geom, amor, xtal, per_band=True).per_band
    # rows: 8 occupied xtal spinors
    assert np.allclose(pb.gamma, [S2] + [0.0] * 7, atol=1e-10)
    # raw (un-normalized) PW weight, so PW_NORM^2 survives in band_k_weight
    w = syn.PW_NORM ** 2
    assert np.allclose(pb.band_k_weight[:4], [w, 0.0])
    assert np.allclose(pb.band_k_weight[4:], [0.0, w])
    assert list(pb.dominant_k) == [0] * 4 + [1] * 4


def test_per_band_nosoc_variants(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom))
    xtal_nosoc = syn.make_sc(geom, *syn.amor_nosoc_bands(geom, rotate_first=THETA))
    amor_soc = syn.make_sc(geom, *syn.soc_bands(geom))
    res = syn.run_spillage(geom, amor, xtal, xtal_nosoc=xtal_nosoc, amor_soc=amor_soc,
                           per_band_nosoc=True, per_band_nosoc_soc=True)
    for pb in (res.per_band_nosoc, res.per_band_nosoc_soc):
        assert np.allclose(pb.gamma, [S2, 0.0, 0.0, 0.0], atol=1e-10)


def test_per_band_requires_xtal_nosoc(geom, amor):
    xtal = syn.make_sc(geom, *syn.soc_bands(geom))
    sortidx, index_list = syn.match(geom, amor)
    with pytest.raises(ValueError, match="requires xtal_nosoc"):
        compute_structural_spillage(amor, xtal, sortidx, index_list, per_band_nosoc=True)


@pytest.mark.parametrize("theta", [0.0, THETA])
def test_sopw(geom, amor, theta):
    amor_soc = syn.make_sc(geom, *syn.soc_bands(geom, rotate_first=theta))
    sortidx, index_list = syn.match(geom, amor)
    res = compute_sopw(amor, amor_soc, sortidx, index_list, per_band=True)
    s2 = np.sin(theta) ** 2
    assert res.total == pytest.approx(s2, abs=1e-10)
    assert np.sum(res.gamma_pw) == pytest.approx(s2, abs=1e-10)
    assert res.normalized == pytest.approx(s2 / 8, abs=1e-10)
    assert np.allclose(res.per_band.gamma, [s2] + [0.0] * 7, atol=1e-10)


def test_cli_writes_outputs(geom, amor, tmp_path, monkeypatch):
    # run cli.main end to end with load_inputs swapped for synthetic data
    xtal = syn.make_sc(geom, *syn.soc_bands(geom, rotate_first=THETA))
    xtal_nosoc = syn.make_sc(geom, *syn.amor_nosoc_bands(geom))
    amor_soc = syn.make_sc(geom, *syn.soc_bands(geom, rotate_first=THETA))
    monkeypatch.setattr(cli, "load_inputs",
                        lambda *a, **kw: syn.inputs(geom, amor, xtal, xtal_nosoc, amor_soc))
    monkeypatch.chdir(tmp_path)
    cli.main(["--xtal-uc", "uc", "--xtal-sc", "sc", "--amor-sc", "amor",
              "--xtal-sc-nosoc", "n", "--amor-soc", "s",
              "--out-per-band", "pb.txt", "--out-per-band-nosoc", "pb_nosoc.txt",
              "--out-per-band-nosoc-soc", "pb_nosoc_soc.txt", "--out-per-band-sopw", "pb_sopw.txt"])

    assert np.allclose(np.loadtxt("spillage.txt"), [S2, 0.0])
    assert np.allclose(np.loadtxt("spillage_norm.txt"), [S2 / 4, 0.0])
    assert np.loadtxt("sopw_norm.txt") == pytest.approx(S2 / 8)
    assert np.loadtxt("sopw_spillage.txt").shape == (syn.N_PW,)
    for name, label in [("pb.txt", "gamma_struct"), ("pb_nosoc.txt", "gamma_struct_nosoc"),
                        ("pb_nosoc_soc.txt", "gamma_struct_nosoc_soc"), ("pb_sopw.txt", "gamma_sopw")]:
        with open(name) as f:
            assert f.readline().split() == ["#", "energy_eV", label, "w_k0", "w_k1", "dominant_k"]
        assert np.loadtxt(name).shape[1] == 5
