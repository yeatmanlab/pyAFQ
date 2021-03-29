import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd

from AFQ.registration import (register_dwi, write_mapping,
                              read_mapping, syn_register_dwi,
                              slr_registration)

import AFQ.data as afd

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.align.imwarp import DiffeomorphicMap


MNI_T2 = afd.read_mni_template()
hardi_img, gtab = dpd.read_stanford_hardi()
MNI_T2_data = MNI_T2.get_fdata()
MNI_T2_affine = MNI_T2.affine
hardi_data = hardi_img.get_fdata()
hardi_affine = hardi_img.affine
b0 = hardi_data[..., gtab.b0s_mask]
mean_b0 = np.mean(b0, -1)

# We select some arbitrary chunk of data so this goes quicker:
subset_b0 = mean_b0[40:50, 40:50, 40:50]
subset_dwi_data = nib.Nifti1Image(hardi_data[40:50, 40:50, 40:50],
                                  hardi_affine)
subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)


def test_slr_registration():
    # have to import subject sls
    file_dict = afd.read_stanford_hardi_tractography()
    streamlines = file_dict['tractography_subsampled.trk']

    # have to import sls atlas
    afd.fetch_hcp_atlas_16_bundles()
    atlas_fname = op.join(
        afd.afq_home,
        'hcp_atlas_16_bundles',
        'Atlas_in_MNI_Space_16_bundles',
        'whole_brain',
        'whole_brain_MNI.trk')
    hcp_atlas = load_tractogram(
        atlas_fname,
        'same', bbox_valid_check=False)

    with nbtmp.InTemporaryDirectory() as tmpdir:
        mapping = slr_registration(streamlines,
                                   hcp_atlas.streamlines,
                                   moving_affine=subset_b0_img.affine,
                                   static_affine=subset_t2_img.affine,
                                   moving_shape=subset_b0_img.shape,
                                   static_shape=subset_t2_img.shape,
                                   progressive=False,
                                   greater_than=10,
                                   rm_small_clusters=1,
                                   rng=np.random.RandomState(seed=8))
        warped_moving = mapping.transform(subset_b0)

        npt.assert_equal(warped_moving.shape, subset_t2.shape)
        mapping_fname = op.join(tmpdir, 'mapping.npy')
        write_mapping(mapping, mapping_fname)
        file_mapping = read_mapping(mapping_fname,
                                    subset_b0_img,
                                    subset_t2_img)

        # Test that it has the same effect on the data:
        warped_from_file = file_mapping.transform(subset_b0)
        npt.assert_equal(warped_from_file, warped_moving)

        # Test that it is, attribute by attribute, identical:
        for k in mapping.__dict__:
            assert (np.all(mapping.__getattribute__(k) ==
                           file_mapping.__getattribute__(k)))


def test_syn_register_dwi():
    warped_b0, mapping = syn_register_dwi(subset_dwi_data, gtab,
                                          template=subset_t2_img,
                                          radius=1)
    npt.assert_equal(isinstance(mapping, DiffeomorphicMap), True)
    npt.assert_equal(warped_b0.shape, subset_t2_img.shape)


def test_register_dwi():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
        # Use an abbreviated data-set:
        img = nib.load(fdata)
        data = img.get_fdata()[..., :10]
        nib.save(nib.Nifti1Image(data, img.affine),
                 op.join(tmpdir, 'data.nii.gz'))
        # Save a subset:
        bvals = np.loadtxt(fbval)
        bvecs = np.loadtxt(fbvec)
        np.savetxt(op.join(tmpdir, 'bvals.txt'), bvals[:10])
        np.savetxt(op.join(tmpdir, 'bvecs.txt'), bvecs[:10])
        reg_file = register_dwi(op.join(tmpdir, 'data.nii.gz'),
                                op.join(tmpdir, 'bvals.txt'),
                                op.join(tmpdir, 'bvecs.txt'))
        npt.assert_(op.exists(reg_file))

