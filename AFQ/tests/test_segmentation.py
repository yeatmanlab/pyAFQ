import os.path as op

import pytest

import numpy as np
import numpy.testing as npt

import nibabel as nib
import dipy.data as dpd
import dipy.data.fetcher as fetcher

import AFQ.data as afd
import AFQ.segmentation as seg


def test_segment():
    dpd.fetch_stanford_hardi()
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    file_dict = afd.read_stanford_hardi_tractograpy()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    templates = afd.read_templates()
    bundles = {'CST_L': {'ROIs': [templates['CST_roi1_L'],
                                  templates['CST_roi2_L']],
                         'rules': [True, True]},
               'CST_R': {'ROIs': [templates['CST_roi1_R'],
                                  templates['CST_roi1_R']],
                         'rules': [True, True]}}

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
    CST_R_sl = list(fiber_groups['CST_R'])
    npt.assert_equal(len(CST_R_sl), 8)
    # Calculate the tract profile for a volume of all-ones:
    tract_profile = seg.calculate_tract_profile(
        np.ones(nib.load(hardi_fdata).shape[:3]),
        CST_R_sl)
    npt.assert_equal(tract_profile, np.ones(100))


def test_gaussian_weights():
    # Some bogus x,y,z coordinates
    x = np.arange(10)
    y = np.arange(10)
    z = np.arange(10)
    # This has the wrong shape (2, 3, 10):
    bundle = np.array([[x, y, z], [x, y, z]])
    pytest.raises(ValueError, seg.gaussian_weights, bundle)
    # Reallocate with the right shape. This time, we're going to create a
    # distribution for which we can predict the weights we would expect
    # to get:
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
