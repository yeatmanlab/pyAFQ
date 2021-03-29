import os.path as op

import pytest

import numpy as np
import numpy.testing as npt

import nibabel as nib
import dipy.data as dpd
import dipy.data.fetcher as fetcher
import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from dipy.stats.analysis import afq_profile
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.segmentation as seg
import AFQ.models.dti as dti


dpd.fetch_stanford_hardi()
hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_img = nib.load(hardi_fdata)
hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
file_dict = afd.read_stanford_hardi_tractography()
mapping = file_dict['mapping.nii.gz']
streamlines = file_dict['tractography_subsampled.trk']
tg = StatefulTractogram(streamlines, hardi_img, Space.RASMM)
tg.to_vox()
streamlines = tg.streamlines
templates = afd.read_templates()
bundles = {'CST_L': {'ROIs': [templates['CST_roi1_L'],
                              templates['CST_roi2_L']],
                     'rules': [True, True],
                     'prob_map': templates['CST_L_prob_map'],
                     'cross_midline': None},
           'CST_R': {'ROIs': [templates['CST_roi1_R'],
                              templates['CST_roi1_R']],
                     'rules': [True, True],
                     'prob_map': templates['CST_R_prob_map'],
                     'cross_midline': None}}


def test_segment():
    segmentation = seg.Segmentation()
    segmentation.segment(bundles,
                         tg,
                         hardi_fdata,
                         hardi_fbval,
                         hardi_fbvec,
                         mapping=mapping)
    fiber_groups = segmentation.fiber_groups

    # We asked for 2 fiber groups:
    npt.assert_equal(len(fiber_groups), 2)
    # Here's one of them:
    CST_R_sl = fiber_groups['CST_R']
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
    bundles_no_prob = {'CST_L': {'ROIs': [templates['CST_roi1_L'],
                                          templates['CST_roi2_L']],
                                 'rules': [True, True],
                                 'cross_midline': False},
                       'CST_R': {'ROIs': [templates['CST_roi1_R'],
                                          templates['CST_roi1_R']],
                                 'rules': [True, True],
                                 'cross_midline': False}}

    segmentation = seg.Segmentation()
    segmentation.segment(bundles_no_prob,
                         tg,
                         hardi_fdata,
                         hardi_fbval,
                         hardi_fbvec,
                         mapping=mapping)
    fiber_groups = segmentation.fiber_groups

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['CST_R']) > 0)


def test_segment_return_idx():
    # Test with the return_idx kwarg set to True:
    segmentation = seg.Segmentation(return_idx=True)
    segmentation.segment(bundles,
                         tg,
                         hardi_fdata,
                         hardi_fbval,
                         hardi_fbvec,
                         mapping=mapping)
    fiber_groups = segmentation.fiber_groups

    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['CST_R']['sl']) > 0)
    npt.assert_(len(fiber_groups['CST_R']['idx']) > 0)


def test_segment_keep_space():
    # Test with the return_idx kwarg set to True:
    segmentation = seg.Segmentation(return_idx=True)
    # We move the tg to rasmm to make sure that it ends
    # up there
    tg.to_rasmm()
    orig_space = tg.space
    segmentation.segment(bundles,
                         tg,
                         hardi_fdata,
                         hardi_fbval,
                         hardi_fbvec,
                         mapping=mapping,
                         reset_tg_space=True)

    npt.assert_equal(tg.space, orig_space)


@pytest.mark.nightly
def test_segment_clip_edges():
    # Test with the clip_edges kwarg set to True:
    segmentation = seg.Segmentation(clip_edges=True)

    fiber_groups = segmentation.segment(bundles,
                                        tg,
                                        hardi_fdata,
                                        hardi_fbval,
                                        hardi_fbvec,
                                        mapping=mapping)

    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_(len(fiber_groups['CST_R']) > 0)


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
    fiber_groups = segmentation.segment(bundles_reco, tg,
                                        hardi_fdata,
                                        hardi_fbval,
                                        hardi_fbvec)

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

    clean_sl = seg.clean_by_endpoints(sl, [1, 2], [3, 4], atlas=atlas)
    npt.assert_equal(list(clean_sl), sl[:2])

    clean_results = list(seg.clean_by_endpoints(sl, [1, 2], [3, 4],
                                                atlas=atlas,
                                                return_idx=True))
    clean_idx = []
    clean_sl = []
    for res in clean_results:
        clean_sl.append(res[0])
        clean_idx.append(res[1])
    npt.assert_equal(list(clean_sl), sl[:2])
    npt.assert_equal(clean_idx, np.array([0, 1]))

    # If tol=1, the third streamline also gets included
    clean_sl = seg.clean_by_endpoints(sl, [1, 2], [3, 4], tol=1, atlas=atlas)
    npt.assert_equal(list(clean_sl), sl[:3])

    # Provide the Nx3 array of indices instead.
    idx_start = np.array(np.where(atlas == 1)).T
    idx_end = np.array(np.where(atlas == 3)).T

    clean_sl = seg.clean_by_endpoints(sl, idx_start, idx_end, atlas=atlas)
    npt.assert_equal(list(clean_sl), np.array([sl[0]]))

    # Sometimes no requirement for one side:
    clean_sl = seg.clean_by_endpoints(sl, [1], None, atlas=atlas)
    npt.assert_equal(list(clean_sl), [sl[0], sl[2], sl[3]])


def test_segment_sampled_streamlines():

    # default segmentation
    segmentation = seg.Segmentation()
    fiber_groups = segmentation.segment(
        bundles,
        tg,
        hardi_fdata,
        hardi_fbval,
        hardi_fbvec,
        mapping=mapping
    )

    # Already using a subsampled tck
    # the CST_R has two streamlines and CST_L has none
    npt.assert_(0 < len(fiber_groups['CST_R']))

    # number of streamlines to sample
    nb_streamlines = int(len(tg)*0.8)

    # sample and segment streamlines
    sampled_segmentation = seg.Segmentation(
        nb_streamlines=nb_streamlines
    )

    sampled_fiber_groups = sampled_segmentation.segment(
        bundles,
        tg,
        hardi_fdata,
        hardi_fbval,
        hardi_fbvec,
        mapping=mapping
    )

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
    npt.assert_(0 <= len(sampled_fiber_groups['CST_R']) <= len(fiber_groups['CST_R']))
