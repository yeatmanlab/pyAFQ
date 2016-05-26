import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.core.gradients as dpg

from AFQ.registration import (syn_registration, register_series, register_dwi,
                              c_of_mass, translation, rigid, affine,
                              streamline_registration, write_mapping,
                              read_mapping, syn_register_dwi)

from dipy.tracking.utils import move_streamlines

from AFQ.utils.streamlines import write_trk

MNI_T2 = dpd.read_mni_template()
hardi_img, gtab = dpd.read_stanford_hardi()
MNI_T2_data = MNI_T2.get_data()
MNI_T2_affine = MNI_T2.get_affine()
hardi_data = hardi_img.get_data()
hardi_affine = hardi_img.get_affine()
b0 = hardi_data[..., gtab.b0s_mask]
mean_b0 = np.mean(b0, -1)

# We select some arbitrary chunk of data so this goes quicker:
subset_b0 = mean_b0[40:50, 40:50, 40:50]
subset_dwi_data = nib.Nifti1Image(hardi_data[40:50, 40:50, 40:50],
                                  hardi_affine)
subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)


def test_syn_registration():
    with nbtmp.InTemporaryDirectory() as tmpdir:
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
        mapping_fname = op.join(tmpdir, 'mapping.nii.gz')
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
    mapping = syn_register_dwi(subset_dwi_data, gtab, template=subset_t2_img)


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


def test_streamline_registration():
    sl1 = [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
           np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])]
    affine = np.eye(4)
    affine[:3, 3] = np.random.randn(3)
    sl2 = list(move_streamlines(sl1, affine))
    aligned, matrix = streamline_registration(sl2, sl1)
    npt.assert_almost_equal(matrix, np.linalg.inv(affine))
    npt.assert_almost_equal(aligned[0], sl1[0])
    npt.assert_almost_equal(aligned[1], sl1[1])

    # We assume the two tracks come from the same space, but it might have
    # some affine associated with it:
    base_aff = np.eye(4) * np.random.rand()
    base_aff[:3, 3] = np.array([1, 2, 3])
    base_aff[3, 3] = 1

    with nbtmp.InTemporaryDirectory() as tmpdir:
        for use_aff in [None, base_aff]:
            fname1 = op.join(tmpdir, 'sl1.trk')
            fname2 = op.join(tmpdir, 'sl2.trk')
            if use_aff is not None:
                # Move the streamlines to this other space, and report it:
                write_trk(fname1,
                          move_streamlines(sl1, np.linalg.inv(use_aff)),
                          use_aff)
                write_trk(fname2,
                          move_streamlines(sl2, np.linalg.inv(use_aff)),
                          use_aff)
            else:
                write_trk(fname1, sl1)
                write_trk(fname2, sl2)

            aligned, matrix = streamline_registration(fname2, fname1)
            npt.assert_almost_equal(aligned[0], sl1[0], decimal=5)
            npt.assert_almost_equal(aligned[1], sl1[1], decimal=5)
