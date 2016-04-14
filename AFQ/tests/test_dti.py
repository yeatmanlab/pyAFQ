import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.io as dio
import dipy.core.gradients as dpg
import dipy.data as dpd
from dipy.sims.voxel import single_tensor


from AFQ import dti

def make_sim_data(out_fbval, out_fbvec, out_fdata, out_shape=(5, 6, 7)):
    """
    Create a synthetic data-set with a 2-shell acquisition

    out_fbval, out_fbvec, out_fdata : str
        Full paths to generated data and bval/bvec files

    out_shape : tuple
        The 3D shape of the output volum

    """
    fimg, fbvals, fbvecs = dpd.get_data('small_64D')
    img = nib.load(fimg)
    bvals, bvecs = dio.read_bvals_bvecs(fbvals, fbvecs)
    gtab = dpg.gradient_table(bvals, bvecs)

    # Simulate a signal based on the DTI model:
    signal = single_tensor(gtab, S0=150)
    DWI = np.zeros(out_shape + (len(gtab.bvals), ))
    DWI[:] = signal
    nib.save(nib.Nifti1Image(DWI, img.affine), out_fdata)
    np.savetxt(out_fbval, bvals)
    np.savetxt(out_fbvec, bvecs)


def test_fit_dti():
    # Let's see whether we can pass a list of files for each one:
    fdata1, fbval1, fbvec1 = dpd.get_data('small_101D')
    fdata2, fbval2, fbvec2 = dpd.get_data('small_101D')

    with nbtmp.InTemporaryDirectory() as tmpdir:
        file_dict = dti.fit_dti([fdata1, fdata2],
                                [fbval1, fbval2],
                                [fbvec1, fbvec2],
                                out_dir=tmpdir)
        for f in file_dict.values():
            npt.assert_(op.exists(f))


def test_predict_dti():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_sim_data(fbval, fbvec, fdata)
        file_dict = dti.fit_dti(fdata, fbval, fbvec, out_dir=tmpdir)
        params_file = file_dict['params']
        gtab = dpg.gradient_table(fbval, fbvec)
        predict_fname = dti.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_data()
        npt.assert_almost_equal(prediction, nib.load(fdata).get_data())
