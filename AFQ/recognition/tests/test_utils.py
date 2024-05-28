import numpy.testing as npt
import numpy as np
import nibabel as nib
import os.path as op

import AFQ.data.fetch as afd
import AFQ.recognition.curvature as abv
import AFQ.recognition.utils as abu
import AFQ.recognition.cleaning as abc


from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy.data.fetcher as fetcher


hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_img = nib.load(hardi_fdata)
file_dict = afd.read_stanford_hardi_tractography()
streamlines = file_dict['tractography_subsampled.trk']
tg = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
tg.to_vox()
streamlines = tg.streamlines


def test_segment_sl_curve():
    sl_disp_0 = abv.sl_curve(streamlines[4], 4)
    npt.assert_array_almost_equal(
        sl_disp_0,
        [[-0.236384, -0.763855, 0.60054],
         [0.232594, -0.867859, -0.439],
         [0.175343, 0.001082, -0.984507]])

    sl_disp_1 = abv.sl_curve(streamlines[2], 4)
    mean_angle_diff = abv.sl_curve_dist(sl_disp_0, sl_disp_1)
    npt.assert_almost_equal(mean_angle_diff, 1.701458, decimal=3)


def test_segment_clip_edges():
    sls = tg.streamlines
    idx = np.arange(len(tg.streamlines))
    accepted_sls = sls[[4, 10, 11]]
    accepted_ix = idx[[4, 10, 11]]
    bundle_roi_dists = np.zeros((len(sls), 3))
    bundle_roi_dists[4, :] = [5, 10, 15]
    bundle_roi_dists[10, :] = [3, 6, 9]
    bundle_roi_dists[11, :] = [10, 10, 10]
    cut_sls = abu.cut_sls_by_dist(
        accepted_sls,
        bundle_roi_dists[accepted_ix],
        [0, 2])
    npt.assert_array_equal(
        cut_sls[0],
        accepted_sls[0][5:15])
    npt.assert_array_equal(
        cut_sls[1],
        accepted_sls[1][3:9])
    npt.assert_array_equal(
        cut_sls[2],
        accepted_sls[2][9:11])


def test_segment_orientation():
    cleaned_idx = \
        abc.clean_by_orientation(streamlines, primary_axis=1)
    npt.assert_equal(np.sum(cleaned_idx), 93)
    cleaned_idx_tol = \
        abc.clean_by_orientation(streamlines, primary_axis=1, tol=50)
    npt.assert_(np.sum(cleaned_idx_tol) < np.sum(cleaned_idx))

    cleaned_idx = \
        abc.clean_by_orientation(streamlines, primary_axis=2)
    cleaned_idx_tol = \
        abc.clean_by_orientation(streamlines, primary_axis=2, tol=33)
    npt.assert_array_equal(cleaned_idx_tol, cleaned_idx)
