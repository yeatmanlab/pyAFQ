import numpy as np
import numpy.testing as npt
import os.path as op

import nibabel as nib
import nibabel.tmpdirs as nbtmp
import dipy.data.fetcher as fetcher

import AFQ.bundles as bdl

hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")


def test_bundles_class():
    # Example Segmentation results
    img = nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4))
    bundles = {'CST_L': {'sl': [[[-80.5, -120.5, -60.5],
                                 [-80.5, -94.5, -36.5],
                                 [-78.5, -68.7, -12.6]],
                                [[-80.5, -120.5, -60.5],
                                 [-80.5, -94.5, -36.5],
                                 [-78.5, -68.7, -12.6]]],
                         'idx': [0, 1]},
               'CST_R': {'sl': [[[-80.5, -120.5, -60.5],
                                 [-80.5, -94.5, -36.5],
                                 [-78.5, -68.7, -12.6]],
                                [[-80.5, -120.5, -60.5],
                                 [-80.5, -94.5, -36.5],
                                 [-78.5, -68.7, -12.6]]],
                         'idx': [0, 1]}}

    with nbtmp.InTemporaryDirectory() as tmpdir:
        # save in bundles class for bundles class tests
        bundles_og = bdl.Bundles(reference=img,
                                 bundles_dict=bundles,
                                 using_idx=True)
        bundles_og.save_bundles(file_path=tmpdir)

        # load bundles again
        bundles = bdl.Bundles(reference=img)
        bundle_names = ['CST_L', 'CST_R']
        bundles.load_bundles(bundle_names, file_path=tmpdir)

    # check loaded bundles are same
    npt.assert_equal(len(bundles.bundles), len(bundles_og.bundles))
    npt.assert_equal(len(bundles.bundles['CST_L'].streamlines),
                     len(bundles_og.bundles['CST_L'].streamlines))
    npt.assert_equal(len(bundles.bundles['CST_R'].streamlines),
                     len(bundles_og.bundles['CST_R'].streamlines))
    npt.assert_equal(bundles.space, bundles_og.space)
    npt.assert_equal(bundles.bundles['CST_L'].space_attributes,
                     bundles_og.bundles['CST_L'].space_attributes)
    npt.assert_equal(bundles.origin, bundles_og.origin)
    npt.assert_array_equal(
        bundles.bundles['CST_L'].data_per_streamline['idx'],
        bundles_og.bundles['CST_L'].data_per_streamline['idx'])

    # test tract profiles
    profiles = bundles.tract_profiles(
        np.ones(nib.load(hardi_fdata).shape[:3]),
        'test_subject', n_points=1)
    npt.assert_almost_equal(profiles.Value, np.zeros(2))

    # test clean bundles
    bundles.clean_bundles()
    npt.assert_equal(len(bundles.bundles), len(bundles_og.bundles))
