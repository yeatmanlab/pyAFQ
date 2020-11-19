import os
import os.path as op

import numpy.testing as npt
import pytest

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.core.gradients as dpg

from AFQ.models import dki
from AFQ.utils.testing import make_dki_data


def test_fit_dki_inputs():
    data_files = ["String in a list"]
    bval_files = "just a string"
    bvec_files = "just another string"
    pytest.raises(ValueError, dki.fit_dki, data_files, bval_files,
                  bvec_files)


def test_fit_dki():
    fdata, fbval, fbvec = dpd.get_fnames('small_101D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
        file_dict = dki.fit_dki(fdata, fbval, fbvec, out_dir=tmpdir)
        for f in file_dict.values():
            op.exists(f)


def test_predict_dki():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_dki_data(fbval, fbvec, fdata)
        file_dict = dki.fit_dki(fdata, fbval, fbvec, out_dir=tmpdir)
        params_file = file_dict['params']
        gtab = dpg.gradient_table(fbval, fbvec)
        predict_fname = dki.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_fdata()
        npt.assert_almost_equal(prediction, nib.load(fdata).get_fdata())
