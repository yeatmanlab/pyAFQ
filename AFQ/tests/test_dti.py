import os
import os.path as op
import shutil

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.core.gradients as dpg
import dipy.data as dpd

from AFQ import dti
from AFQ.utils.testing import make_dti_data


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
        fbval = op.join(tmpdir, 'dti.bval')
        fbvec = op.join(tmpdir, 'dti.bvec')
        fdata = op.join(tmpdir, 'dti.nii.gz')
        make_dti_data(fbval, fbvec, fdata)
        file_dict = dti.fit_dti(fdata, fbval, fbvec, out_dir=tmpdir)
        params_file = file_dict['params']
        gtab = dpg.gradient_table(fbval, fbvec)
        predict_fname = dti.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_data()
        npt.assert_almost_equal(prediction, nib.load(fdata).get_data())

        # If you have a mask into the volume, you will predict only that
        # part of the volume:
        mask = np.zeros(prediction.shape[:3], dtype=bool)
        mask[2:4, 2:4, 2:4] = 1
        file_dict = dti.fit_dti(fdata, fbval, fbvec, mask=mask,
                                out_dir=tmpdir)
        params_file = file_dict['params']
        predict_fname = dti.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_data()
        npt.assert_almost_equal(prediction[mask],
                                nib.load(fdata).get_data()[mask])


def test_cli():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        dwi, bval, bvec = dpd.get_data("small_25")
        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")
        # Call script
        cmd = " ".join(["pyAFQ_dti", "-d" , "small_25.nii.gz",
               "-l", "small_25.bval", "-c", "small_25.bvec"])
        out = os.system(cmd)
        assert out ==  0
