import numpy as np
import numpy.testing as npt

import AFQ.data as afd
from AFQ.utils.conversion import matlab_tractography, matlab_mori_groups

import os.path as op

DATA_PATH = op.join(op.abspath(op.dirname(__file__)), "../../tests/data")


def test_matlab_tractography():
    sft = matlab_tractography(
        op.join(DATA_PATH, "WholeBrainFG_test.mat"),
        afd.read_mni_template())
    npt.assert_equal(len(sft.streamlines), 2)


def test_matlab_mori_groups():
    fiber_groups = matlab_mori_groups(
        op.join(DATA_PATH, "MoriGroups_Test.mat"),
        afd.read_mni_template())
    npt.assert_equal(len(fiber_groups.keys()), 20)
    npt.assert_equal(len(fiber_groups['CST_R'].streamlines), 2)
