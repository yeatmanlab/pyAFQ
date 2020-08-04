import os
import os.path as op
import shutil

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

from dipy.core.geometry import vector_norm
import dipy.core.gradients as dpg
import dipy.data as dpd

from AFQ.models import dti
from AFQ._fixes import in_place_norm
from AFQ.utils.testing import make_dti_data


def test_fit_dti():
    # Let's see whether we can pass a list of files for each one:
    fdata1, fbval1, fbvec1 = dpd.get_fnames('small_101D')
    fdata2, fbval2, fbvec2 = dpd.get_fnames('small_101D')

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
        prediction = nib.load(predict_fname).get_fdata()
        npt.assert_almost_equal(prediction, nib.load(fdata).get_fdata())

        # If you have a mask into the volume, you will predict only that
        # part of the volume:
        mask = np.zeros(prediction.shape[:3], dtype=bool)
        mask[2:4, 2:4, 2:4] = 1
        file_dict = dti.fit_dti(fdata, fbval, fbvec, mask=mask,
                                out_dir=tmpdir)
        params_file = file_dict['params']
        predict_fname = dti.predict(params_file, gtab, S0_file=fdata,
                                    out_dir=tmpdir)
        prediction = nib.load(predict_fname).get_fdata()
        npt.assert_almost_equal(prediction[mask],
                                nib.load(fdata).get_fdata()[mask])


def test_cli():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dti.bval')
        fbvec = op.join(tmpdir, 'dti.bvec')
        fdata = op.join(tmpdir, 'dti.nii.gz')
        make_dti_data(fbval, fbvec, fdata)
        cmd = " ".join(
            ["pyAFQ_dti",
             "-d", op.join(tmpdir, "dti.nii.gz"),
             "-l", op.join(tmpdir, "dti.bval"),
             "-c", op.join(tmpdir, "dti.bvec")])
        out = os.system(cmd)
        assert out ==  0
        assert op.exists(op.join(tmpdir, 'dti', 'dti_params.nii.gz'))


def test_inplace_norm():
    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec)
    norm2 = in_place_norm(vec)
    npt.assert_equal(norm1, norm2)

    vec = [[8.0, 15.0, 0.0], [0.0, 36.0, 77.0]]
    norm1 = vector_norm(vec)
    norm2 = in_place_norm(vec)
    npt.assert_equal(norm1, norm2)

    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec, keepdims=True)
    norm2 = in_place_norm(vec, keepdims=True)
    npt.assert_equal(norm1, norm2)

    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec, axis=0)
    norm2 = in_place_norm(vec, axis=0)
    npt.assert_equal(norm1, norm2)
