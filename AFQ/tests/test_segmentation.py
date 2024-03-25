import os.path as op

import pytest

import numpy as np
import numpy.testing as npt

import nibabel as nib
import dipy.data as dpd
import dipy.data.fetcher as fetcher
from dipy.stats.analysis import afq_profile
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.data.fetch as afd
import AFQ.segmentation as seg
import AFQ.registration as reg


dpd.fetch_stanford_hardi()
hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_img = nib.load(hardi_fdata)
hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
file_dict = afd.read_stanford_hardi_tractography()
reg_template = afd.read_mni_template()
mapping = reg.read_mapping(
    file_dict['mapping.nii.gz'],
    hardi_img,
    reg_template)
streamlines = file_dict['tractography_subsampled.trk']
tg = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
tg.to_vox()
streamlines = tg.streamlines
templates = afd.read_templates()
cst_r_curve_ref = StatefulTractogram([[
    [4.09933186, -27.86049652, -8.57825184],
    [4.18979505, -27.30110527, -7.7542993 ],
    [4.31947752, -26.78867352, -6.90792487],
    [4.48837897, -26.32320413, -6.03912657],
    [4.74019388, -25.95165819, -5.14551435],
    [4.99200908, -25.58011067, -4.25190286],
    [5.29724114, -25.17955789, -3.39071481],
    [5.70181712, -24.72099485, -2.60239253],
    [6.10747528, -24.18646430, -1.8638705 ],
    [6.56050014, -23.51795578, -1.27745605]]],
    reg_template, Space.RASMM)

bundles = {'Left Corticospinal': {
                    'include': [
                        templates['CST_roi1_L'],
                        templates['CST_roi2_L']],
                    'prob_map': templates['CST_L_prob_map'],
                    'cross_midline': None},
           'Right Corticospinal': {
                    'include': [
                        templates['CST_roi1_R'],
                        templates['CST_roi2_R']],
                    'prob_map': templates['CST_R_prob_map'],
                    "curvature": {
                        "sft": cst_r_curve_ref,
                        "cut": True, "thresh": 45},
                    'cross_midline': None}}

def test_segment():
    segmentation = seg.Segmentation()
    segmentation.segment(bundles,
                         tg,
                         mapping,
                         nib.load(hardi_fdata))
    fiber_groups = segmentation.fiber_groups

    # We asked for 2 fiber groups:
    npt.assert_equal(len(fiber_groups), 2)
    # Here's one of them:
    CST_R_sl = fiber_groups['Right Corticospinal']
    # Let's make sure there are streamlines in there:
    npt.assert_(len(CST_R_sl) > 0)
    # Calculate the tract profile for a volume of all-ones:
    tract_profile = afq_profile(
        np.ones(nib.load(hardi_fdata).shape[:3]),
        CST_R_sl.streamlines, np.eye(4))
    npt.assert_almost_equal(tract_profile, np.ones(100))

    clean_sl = seg.clean_bundle(CST_R_sl)
    npt.assert_equal(len(clean_sl), len(CST_R_sl))


@pytest.mark.nightly
def test_segment_no_prob():
    # What if you don't have probability maps?
    bundles_no_prob = {
        'Left Corticospinal': {
            'include': [
                templates['CST_roi1_L'],
                templates['CST_roi2_L']],
            'cross_midline': False},
        'Right Corticospinal': {
            'include': [
                templates['CST_roi1_R'],
                templates['CST_roi2_R']],
            'cross_midline': False}}

    segmentation = seg.Segmentation()
    segmentation.segment(bundles_no_prob,
                         tg,
                         mapping,
                         nib.load(hardi_fdata))
    fiber_groups = segmentation.fiber_groups

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']) > 0)


def test_segment_return_idx():
    # Test with the return_idx kwarg set to True:
    segmentation = seg.Segmentation(return_idx=True)
    segmentation.segment(bundles,
                         tg,
                         mapping,
                         nib.load(hardi_fdata))
    fiber_groups = segmentation.fiber_groups

    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']['sl']) > 0)
    npt.assert_(len(fiber_groups['Right Corticospinal']['idx']) > 0)


def test_segment_keep_space():
    # Test with the return_idx kwarg set to True:
    segmentation = seg.Segmentation(return_idx=True)
    # We move the tg to rasmm to make sure that it ends
    # up there
    tg.to_rasmm()
    orig_space = tg.space
    segmentation.segment(bundles,
                         tg,
                         mapping,
                         nib.load(hardi_fdata),
                         reset_tg_space=True)

    npt.assert_equal(tg.space, orig_space)


def test_segment_sl_curve():
    sl_disp_0 = seg.sl_curve(streamlines[4], 4)
    npt.assert_array_almost_equal(
        sl_disp_0,
        [[-0.236384, -0.763855,  0.60054 ],
         [ 0.232594, -0.867859, -0.439   ],
         [ 0.175343,  0.001082, -0.984507]])

    sl_disp_1 = seg.sl_curve(streamlines[2], 4)
    mean_angle_diff = seg.sl_curve_dist(sl_disp_0, sl_disp_1)
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
    cut_sls = seg._cut_sls_by_dist(
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


@pytest.mark.nightly
def test_segment_clip_edges_api():
    # Test with the clip_edges kwarg set to True:
    segmentation = seg.Segmentation(clip_edges=True)

    fiber_groups, _ = segmentation.segment(
        bundles,
        tg,
        mapping,
        nib.load(hardi_fdata))

    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']) > 0)


def test_segment_reco():
    # get bundles for reco method
    bundles_reco = afd.read_hcp_atlas(16)
    bundle_names = ['whole_brain', 'CST_R', 'CST_L']
    for key in list(bundles_reco):
        if key not in bundle_names:
            bundles_reco.pop(key, None)

    # Try recobundles method
    segmentation = seg.Segmentation(seg_algo='Reco',
                                    progressive=False,
                                    greater_than=10,
                                    rm_small_clusters=1,
                                    rng=np.random.RandomState(seed=8))
    fiber_groups, _ = segmentation.segment(
        bundles_reco, tg, mapping,
        nib.load(hardi_fdata))

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['CST_R']) > 0)


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
        np.logical_or(atlas==1, atlas==2).astype(np.float32), np.eye(4))
    target_img_end = nib.Nifti1Image(
        np.logical_or(atlas==3, atlas==4).astype(np.float32), np.eye(4))

    clean_idx_start = list(seg.clean_by_endpoints(
        sl, target_img_start, 0))
    clean_idx_end = list(seg.clean_by_endpoints(
        sl, target_img_end, -1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 0, 0]))

    # If tol=1, the third streamline also gets included
    clean_idx_start = list(seg.clean_by_endpoints(
        sl, target_img_start, 0, tol=1))
    clean_idx_end = list(seg.clean_by_endpoints(
        sl, target_img_end, -1, tol=1))
    npt.assert_array_equal(np.logical_and(
        clean_idx_start, clean_idx_end), np.array([1, 1, 1, 0]))


def test_exclusion_ROI():
    segmentation = seg.Segmentation(
        filter_by_endpoints=False
    )
    slf_bundle = {
        'Left Superior Longitudinal': {
            'include': [
                templates['SLF_roi1_L'],
                templates['SLF_roi2_L']],
            'cross_midline': None}}

    # tractogram where 1 streamline goes through include ROIs only
    # and the other goes through both include and exclude ROIs
    slf_tg = StatefulTractogram(
        np.asarray(
            [
                [
                    [8, 53, 39], [8, 50, 39], [8, 45, 39],
                    [30, 41, 61], [28, 61, 38]],
                [
                    [8, 53, 39], [8, 50, 39], [8, 45, 39],
                    [30, 41, 62], [20, 44, 34]]
            ]).astype(float),
        hardi_img, Space.VOX)
    fiber_groups, _ = segmentation.segment(
        slf_bundle,
        slf_tg,
        mapping,
        nib.load(hardi_fdata))
     
    npt.assert_equal(len(fiber_groups["Left Superior Longitudinal"]), 2)

    slf_bundle['Left Superior Longitudinal']['exclude'] =\
        [templates["SLFt_roi2_L"]]

    fiber_groups, _ = segmentation.segment(
        slf_bundle,
        slf_tg,
        mapping,
        nib.load(hardi_fdata))

    npt.assert_equal(len(fiber_groups["Left Superior Longitudinal"]), 1)


def test_segment_orientation():
    cleaned_idx = \
        seg.clean_by_orientation(streamlines, primary_axis=1)
    npt.assert_equal(np.sum(cleaned_idx), 93)
    cleaned_idx_tol = \
        seg.clean_by_orientation(streamlines, primary_axis=1, tol=50)
    npt.assert_(np.sum(cleaned_idx_tol) < np.sum(cleaned_idx))

    cleaned_idx = \
        seg.clean_by_orientation(streamlines, primary_axis=2)
    cleaned_idx_tol = \
        seg.clean_by_orientation(streamlines, primary_axis=2, tol=33)
    npt.assert_array_equal(cleaned_idx_tol, cleaned_idx)


def test_segment_sampled_streamlines():

    # default segmentation
    segmentation = seg.Segmentation()
    fiber_groups, _ = segmentation.segment(
        bundles,
        tg,
        mapping,
        nib.load(hardi_fdata))

    # Already using a subsampled tck
    # the Right Corticospinal has two streamlines and
    # Left Corticospinal has none
    npt.assert_(0 < len(fiber_groups['Right Corticospinal']))

    # number of streamlines to sample
    nb_streamlines = int(len(tg)*0.8)

    # sample and segment streamlines
    sampled_segmentation = seg.Segmentation(
        nb_streamlines=nb_streamlines
    )

    sampled_fiber_groups, _ = sampled_segmentation.segment(
        bundles,
        tg,
        mapping,
        nib.load(hardi_fdata))

    # sampled streamlines should equal the sample number
    npt.assert_equal(len(sampled_segmentation.tg), nb_streamlines)

    # sampled streamlines should be subset of the original streamlines
    npt.assert_(
        np.all(
            np.isin(
                sampled_segmentation.tg.streamlines._data,
                tg.streamlines._data
            )
        )
    )

    # expect the number of resulting streamlines to be more than 0 but less
    # than default; given that random sample and given there are only two
    # streamlines less than equal
    npt.assert_(0 <= len(
        sampled_fiber_groups['Right Corticospinal']) <= len(
            fiber_groups['Right Corticospinal']))
