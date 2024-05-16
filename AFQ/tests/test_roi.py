import numpy as np
from scipy.spatial.distance import cdist

from AFQ.bundle_rec.roi import (
    check_sls_with_inclusion,
    check_sl_with_inclusion,
    check_sl_with_exclusion)


streamline1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
streamline2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
roi1 = np.array([[1, 2, 3], [4, 5, 6]])
roi2 = np.array([[7, 8, 9], [10, 11, 12]])
include_rois = [roi1, roi2]
exclude_rois = [roi1]
include_roi_tols = [10, 10]
exclude_roi_tols = [1]


def test_check_sls_with_inclusion():
    sls = [streamline1, streamline2]
    result = list(check_sls_with_inclusion(
        sls, include_rois, include_roi_tols))
    assert result[0][0] is True
    assert np.allclose(
        result[0][1][0], cdist(streamline1, roi1, 'sqeuclidean'))
    assert np.allclose(
        result[0][1][1], cdist(streamline1, roi2, 'sqeuclidean'))
    assert result[1][0] is False


def test_check_sl_with_inclusion_pass():
    result, dists = check_sl_with_inclusion(
        streamline1, include_rois, include_roi_tols)
    assert result is True
    assert len(dists) == 2


def test_check_sl_with_inclusion_fail():
    result, dists = check_sl_with_inclusion(
        streamline2, include_rois, include_roi_tols)
    assert result is False
    assert dists == []


def test_check_sl_with_exclusion_pass():
    result = check_sl_with_exclusion(
        streamline1, exclude_rois, exclude_roi_tols)
    assert result is False


def test_check_sl_with_exclusion_fail():
    result = check_sl_with_exclusion(
        streamline2, exclude_rois, exclude_roi_tols)
    assert result is True
