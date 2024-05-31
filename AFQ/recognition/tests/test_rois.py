import numpy as np
import AFQ.recognition.roi as abr
import nibabel as nib
import numpy.testing as npt
import numpy as np
from scipy.spatial.distance import cdist
from AFQ.recognition.roi import (
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


def test_clean_by_endpoints():
    sl = [np.array([[1, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1],
                    [4, 1, 1]]),
          np.array([[1, 1, 2],
                    [2, 1, 2],
                    [3, 1, 2],
                    [4, 1, 2]]),
          np.array([[1, 1, 1],
                    [2, 1, 1],
                    [3, 1, 1]]),
          np.array([[1, 1, 1],
                    [2, 1, 1]])]

    atlas = np.zeros((20, 20, 20))

    # Targets:
    atlas[1, 1, 1] = 1
    atlas[1, 1, 2] = 2
    atlas[4, 1, 1] = 3
    atlas[4, 1, 2] = 4

    target_img_start = nib.Nifti1Image(
        np.logical_or(atlas == 1, atlas == 2).astype(np.float32), np.eye(4))
    target_img_end = nib.Nifti1Image(
        np.logical_or(atlas == 3, atlas == 4).astype(np.float32), np.eye(4))

    clean_idx_start = list(abr.clean_by_endpoints(
        sl, target_img_start, 0))
    clean_idx_end = list(abr.clean_by_endpoints(
        sl, target_img_end, -1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 0, 0]))

    # If tol=1, the third streamline also gets included
    clean_idx_start = list(abr.clean_by_endpoints(
        sl, target_img_start, 0, tol=1))
    clean_idx_end = list(abr.clean_by_endpoints(
        sl, target_img_end, -1, tol=1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 1, 0]))


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
