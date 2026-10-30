"""Tests on real VASP WAVECARs (the README smoke tests).

Each test checks the Gamma-point spillage against an independent value (the
original spillage code for Bi, pymatgen's spin-orbit spillage for Bi2Se3),
then compares the full output to the stored smoke-run results in reference/.

Skipped unless SPILLAGE_TEST_DATA points at the example WAVECAR directory
(Bi_data/, r2scan_uc_nosoc/, DOS_crys_supercell/; see the paths below).
These load several GB of WAVECARs, so run them on a compute node:

    SPILLAGE_TEST_DATA=/path/to/data pytest tests/test_real_wavecars.py -v -s
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
REF = Path(__file__).resolve().parent / "reference"
DATA = os.environ.get("SPILLAGE_TEST_DATA")

pytestmark = pytest.mark.skipif(not DATA, reason="set SPILLAGE_TEST_DATA to run real-WAVECAR tests")


def run_cli(tmp_path, *argv):
    # src/ on the path so this works from a checkout without `pip install`
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO / "src"),
                                                                    os.environ.get("PYTHONPATH")])))
    subprocess.run([sys.executable, "-m", "structural_spillage", *map(str, argv)],
                   cwd=tmp_path, env=env, check=True)


def test_bismuthene(tmp_path):
    d = Path(DATA).resolve() / "Bi_data"
    run_cli(tmp_path,
            "--xtal-uc", d / "uc/2nd/WAVECAR", "--uc-soc",
            "--xtal-sc", d / "supercell/WAVECAR",
            "--xtal-sc-nosoc", d / "supercell_nosoc/WAVECAR",
            "--amor-sc", d / "dis/WAVECAR",
            "--amor-soc", d / "dis_soc/WAVECAR",
            "--out-spillage", "spillage.txt",
            "--out-norm-spillage", "spillage_norm.txt",
            "--out-sopw", "sopw.txt",
            "--out-norm-sopw", "sopw_norm.txt")
    spill = np.loadtxt(tmp_path / "spillage.txt")
    assert spill[0] == pytest.approx(2.37, abs=0.005)   # Gamma value, matches original spillage code
    np.testing.assert_allclose(spill, np.loadtxt(REF / "bi_spillage.txt"), rtol=1e-6)
    np.testing.assert_allclose(np.loadtxt(tmp_path / "spillage_norm.txt"),
                               np.loadtxt(REF / "bi_spillage_norm.txt"), rtol=1e-6)
    np.testing.assert_allclose(np.loadtxt(tmp_path / "sopw_norm.txt"),
                               np.loadtxt(REF / "bi_sopw_norm.txt"), rtol=1e-6)


def test_bi2se3_crystalline(tmp_path):
    d = Path(DATA).resolve()
    run_cli(tmp_path,
            "--xtal-uc", d / "r2scan_uc_nosoc/WAVECAR",
            "--xtal-sc", d / "DOS_crys_supercell/soc/WAVECAR",
            "--amor-sc", d / "DOS_crys_supercell/noSOC/WAVECAR",
            "--out-spillage", "spillage.txt")
    spill = np.loadtxt(tmp_path / "spillage.txt")
    assert spill[0] == pytest.approx(2.74, abs=0.005)   # Gamma value, matches pymatgen SOC spillage
    np.testing.assert_allclose(spill, np.loadtxt(REF / "bi2se3_spillage.txt"), rtol=1e-6)
