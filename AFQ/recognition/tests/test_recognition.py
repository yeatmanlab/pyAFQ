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
import AFQ.registration as reg
import AFQ.recognition.cleaning as abc
from AFQ.recognition.recognize import recognize


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
    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles,
        reg_template)

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

    clean_sl = abc.clean_bundle(CST_R_sl)
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

    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles_no_prob,
        reg_template)

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']) > 0)


def test_segment_return_idx():
    # Test with the return_idx kwarg set to True:
    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles,
        reg_template,
        return_idx=True)

    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']['sl']) > 0)
    npt.assert_(len(fiber_groups['Right Corticospinal']['idx']) > 0)


@pytest.mark.nightly
def test_segment_clip_edges_api():
    # Test with the clip_edges kwarg set to True:
    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles,
        reg_template,
        clip_edges=True)
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['Right Corticospinal']) > 0)


def test_segment_reco():
    # get bundles for reco method
    bundles_reco = afd.read_hcp_atlas(16)
    bundle_names = ['CST_R', 'CST_L']
    for key in list(bundles_reco):
        if key not in bundle_names:
            bundles_reco.pop(key, None)

    # Try recobundles method
    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles_reco,
        reg_template,
        rng=np.random.RandomState(seed=8))

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['CST_R']) > 0)


def test_exclusion_ROI():
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
    fiber_groups, _ = recognize(
        slf_tg,
        nib.load(hardi_fdata),
        mapping,
        slf_bundle,
        reg_template)
     
    npt.assert_equal(len(fiber_groups["Left Superior Longitudinal"]), 2)

    slf_bundle['Left Superior Longitudinal']['exclude'] =\
        [templates["SLFt_roi2_L"]]

    fiber_groups, _ = recognize(
        slf_tg,
        nib.load(hardi_fdata),
        mapping,
        slf_bundle,
        reg_template)

    npt.assert_equal(len(fiber_groups["Left Superior Longitudinal"]), 1)


def test_segment_sampled_streamlines():
    fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles,
        reg_template)

    # Already using a subsampled tck
    # the Right Corticospinal has two streamlines and
    # Left Corticospinal has none
    npt.assert_(0 < len(fiber_groups['Right Corticospinal']))

    # number of streamlines to sample
    nb_streamlines = int(len(tg)*0.8)

    # sample and segment streamlines
    sampled_fiber_groups, _ = recognize(
        tg,
        nib.load(hardi_fdata),
        mapping,
        bundles,
        reg_template,
        nb_streamlines=nb_streamlines)

    # sampled streamlines should be subset of the original streamlines
    npt.assert_(
        np.all(
            np.isin(
                sampled_fiber_groups['Right Corticospinal'].streamlines._data,
                fiber_groups['Right Corticospinal'].streamlines._data
            )
        )
    )

    # expect the number of resulting streamlines to be more than 0 but less
    # than default; given that random sample and given there are only two
    # streamlines less than equal
    npt.assert_(0 <= len(
        sampled_fiber_groups['Right Corticospinal']) <= len(
            fiber_groups['Right Corticospinal']))
