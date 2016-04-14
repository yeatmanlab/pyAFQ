import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.core.gradients as dpg
import dipy.io as dio
from dipy.sims.voxel import multi_tensor_dki

from AFQ import dki


def make_sim_data(out_fbval, out_fbvec, out_fdata, out_shape=(5, 6, 7)):
    """
    Create a synthetic data-set with a 2-shell acquisition

    out_fbval, out_fbvec, out_fdata : str
        Full paths to generated data and bval/bvec files

    out_shape : tuple
        The 3D shape of the output volum

    """
    # This is one-shell (b=1000) data:
    fimg, fbvals, fbvecs = dpd.get_data('small_64D')
    img = nib.load(fimg)
    bvals, bvecs = dio.read_bvals_bvecs(fbvals, fbvecs)
    gtab = dpg.gradient_table(bvals, bvecs)
    # So  we create two shells out of it
    bvals_2s = np.concatenate((bvals, bvals * 2), axis=0)
    bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
    gtab_2s = dpg.gradient_table(bvals_2s, bvecs_2s)

    # Simulate a signal based on the DKI model:
    mevals_cross = np.array([[0.00099, 0, 0], [0.00226, 0.00087, 0.00087],
                             [0.00099, 0, 0], [0.00226, 0.00087, 0.00087]])
    angles_cross = [(80, 10), (80, 10), (20, 30), (20, 30)]
    fie = 0.49
    frac_cross = [fie * 50, (1 - fie) * 50, fie * 50, (1 - fie) * 50]
    # Noise free simulates
    signal_cross, dt_cross, kt_cross = multi_tensor_dki(gtab_2s, mevals_cross,
                                                        S0=100,
                                                        angles=angles_cross,
                                                        fractions=frac_cross,
                                                        snr=None)
    DWI = np.zeros(out_shape + (len(gtab_2s.bvals), ))
    DWI[:] = signal_cross
    nib.save(nib.Nifti1Image(DWI, img.affine), out_fdata)
    np.savetxt(out_fbval, bvals_2s)
    np.savetxt(out_fbvec, bvecs_2s)


def test_fit_dki_inputs():
    data_files = ["String in a list"]
    bval_files = "just a string"
    bvec_files = "just another string"
    npt.assert_raises(ValueError, dki.fit_dki, data_files, bval_files,
                      bvec_files)


def test_fit_dki():
    fdata, fbval, fbvec = dpd.get_data('small_101D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
        file_dict = dki.fit_dki(fdata, fbval, fbvec, out_dir=tmpdir)
        for f in file_dict.values():
            op.exists(f)


def test_predict_dki():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_sim_data(fbval, fbvec, fdata)
        file_dict = dki.fit_dki(fdata, fbval, fbvec, out_dir=tmpdir)
        params_file = file_dict['params']
        gtab = dpg.gradient_table(fbval, fbvec)
        predict_fname = dki.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_data()
        npt.assert_almost_equal(prediction, nib.load(fdata).get_data())
