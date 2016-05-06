import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.core.gradients as dpg

from AFQ.registration import (syn_registration, register_series, register_dwi,
                              c_of_mass, translation, rigid, affine)


def test_syn_registration():
    MNI_T2 = dpd.read_mni_template()
    ni, gtab = dpd.read_stanford_hardi()
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()
    hardi_data = ni.get_data()
    hardi_affine = ni.get_affine()
    b0 = hardi_data[..., gtab.b0s_mask]
    mean_b0 = np.mean(b0, -1)

    # We select some arbitrary chunk of data so this goes quicker:
    subset_b0 = mean_b0[40:50, 40:50, 40:50]
    subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
    warped_moving, mapping = syn_registration(subset_b0,
                                              subset_t2,
                                              moving_affine=hardi_affine,
                                              static_affine=MNI_T2_affine,
                                              step_length=0.1,
                                              metric='CC',
                                              dim=3,
                                              level_iters=[10, 10, 5],
                                              sigma_diff=2.0,
                                              prealign=None)

    npt.assert_equal(warped_moving.shape, subset_t2.shape)


def test_register_series():
    fdata, fbval, fbvec = dpd.get_data('small_64D')
    img = nib.load(fdata)
    gtab = dpg.gradient_table(fbval, fbvec)
    ref_idx = np.where(gtab.b0s_mask)
    transformed_list, affine_list = register_series(img,
                                                    ref=ref_idx,
                                                    pipeline=[c_of_mass,
                                                              translation,
                                                              rigid,
                                                              affine])


def test_register_dwi():
    fdata, fbval, fbvec = dpd.get_data('small_64D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
        # Use an abbreviated data-set:
        img = nib.load(fdata)
        data = img.get_data()[..., :10]
        nib.save(nib.Nifti1Image(data, img.affine),
                 op.join(tmpdir, 'data.nii.gz'))
        # Convert from npy to txt:
        bvals = np.load(fbval)
        bvecs = np.load(fbvec)
        np.savetxt(op.join(tmpdir, 'bvals.txt'), bvals[:10])
        np.savetxt(op.join(tmpdir, 'bvecs.txt'), bvecs[:10])
        reg_file = register_dwi(op.join(tmpdir, 'data.nii.gz'),
                                op.join(tmpdir, 'bvals.txt'),
                                op.join(tmpdir, 'bvecs.txt'))
        npt.assert_(op.exists(reg_file))
