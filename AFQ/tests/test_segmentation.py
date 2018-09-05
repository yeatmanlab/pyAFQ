import os.path as op

import pytest

import numpy as np
import numpy.testing as npt

import nibabel as nib
import dipy.data as dpd
import dipy.data.fetcher as fetcher
import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu

import AFQ.data as afd
import AFQ.segmentation as seg


def test_segment():
    dpd.fetch_stanford_hardi()
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_img = nib.load(hardi_fdata)
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    file_dict = afd.read_stanford_hardi_tractography()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    streamlines = dts.Streamlines(dtu.move_streamlines([s for s in streamlines if s.shape[0] > 100],
                                                        np.linalg.inv(hardi_img.affine)))

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

    fiber_groups = seg.segment(hardi_fdata,
                               hardi_fbval,
                               hardi_fbvec,
                               streamlines,
                               bundles,
                               mapping=mapping,
                               as_generator=True)

    # We asked for 2 fiber groups:
    npt.assert_equal(len(fiber_groups), 2)
    # There happen to be 8 fibers in the right CST:
    CST_R_sl = fiber_groups['CST_R']
    npt.assert_equal(len(CST_R_sl), 7)
    # Calculate the tract profile for a volume of all-ones:
    tract_profile = seg.calculate_tract_profile(
        np.ones(nib.load(hardi_fdata).shape[:3]),
        CST_R_sl)
    npt.assert_almost_equal(tract_profile, np.ones(100))

    # Test providing an array input to calculate_tract_profile:
    tract_profile = seg.calculate_tract_profile(
        np.ones(nib.load(hardi_fdata).shape[:3]),
        seg._resample_bundle(CST_R_sl, 100))

    npt.assert_almost_equal(tract_profile, np.ones(100))
    clean_sl = seg.clean_fiber_group(CST_R_sl)
    # Since there are only 8 streamlines here, nothing should happen:
    npt.assert_equal(clean_sl, CST_R_sl)

    # Setting minimum number of streamlines to a smaller number and
    # threshold to a relatively small number will exclude some streamlines:
    clean_sl = seg.clean_fiber_group(CST_R_sl, min_sl=2, clean_threshold=2)
    npt.assert_equal(len(clean_sl), 3)

    # What if you don't have probability maps?
    bundles = {'CST_L': {'ROIs': [templates['CST_roi1_L'],
                                  templates['CST_roi2_L']],
                         'rules': [True, True],
                         'cross_midline': False},
               'CST_R': {'ROIs': [templates['CST_roi1_R'],
                                  templates['CST_roi1_R']],
                         'rules': [True, True],
                         'cross_midline': False}}

    fiber_groups = seg.segment(hardi_fdata,
                               hardi_fbval,
                               hardi_fbvec,
                               streamlines,
                               bundles,
                               mapping=mapping,
                               as_generator=True)

    # This condition should still hold
    npt.assert_equal(len(fiber_groups), 2)
    npt.assert_equal(len(fiber_groups['CST_R']), 8)


def test_gaussian_weights():
    # Some bogus x,y,z coordinates
    x = np.arange(10)
    y = np.arange(10)
    z = np.arange(10)
    # Create a distribution for which we can predict the weights we would
    # expect to get:
    bundle = np.array([np.array([x, y, z]).T + 1,
                       np.array([x, y, z]).T - 1])
    # In this case, all nodes receives an equal weight of 0.5:
    w = seg.gaussian_weights(bundle)
    npt.assert_equal(w, np.ones(bundle.shape[:-1]) * 0.5)

    # Here, some nodes are twice as far from the mean as others
    bundle = np.array([np.array([x, y, z]).T + 2,
                       np.array([x, y, z]).T + 1,
                       np.array([x, y, z]).T - 1,
                       np.array([x, y, z]).T - 2])
    w = seg.gaussian_weights(bundle)

    # And their weights should be halved
    # XXX Need to check how to transform this through the
    # Gaussian distribution!
    npt.assert_almost_equal(w[0], w[1] / 2)