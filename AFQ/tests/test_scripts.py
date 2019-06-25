import os.path as op

import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

from AFQ.utils.testing import (make_dki_data, make_dti_data, ScriptRunner,
                               assert_image_shape_affine)

runner = ScriptRunner(script_sdir='bin',
                      module_sdir='AFQ',
                      debug_print_var='AFQ_DEBUG_PRINT')


def test_fit_dki():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_dki_data(fbval, fbvec, fdata)
        cmd = ["pyAFQ_dki", "-d", fdata, "-l", fbval, "-c", fbvec,
               "-o", tmpdir]
        out = runner.run_command(cmd)
        npt.assert_equal(out[0], 0)
        # Get expected values
        names = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK']
        for n in names:
            fname = op.join(tmpdir, "dki_%s.nii.gz" % n)
            img = nib.load(fdata)
            affine = img.affine
            shape = img.shape[:-1]
            assert_image_shape_affine(fname, shape, affine)


def test_predict_dki():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_dki_data(fbval, fbvec, fdata)
        cmd1 = ["pyAFQ_dki", "-d", fdata, "-l", fbval, "-c", fbvec,
                "-o", tmpdir]
        out = runner.run_command(cmd1)
        npt.assert_equal(out[0], 0)

        # Get expected values
        fparams = op.join(tmpdir, "dki_params.nii.gz")
        cmd2 = ["pyAFQ_dki_predict", "-p", fparams, "-l", fbval, "-c", fbvec,
                "-o", tmpdir, '-b', '0']
        out = runner.run_command(cmd2)
        npt.assert_equal(out[0], 0)
        pred = nib.load(op.join(tmpdir, "dki_prediction.nii.gz")).get_fdata()
        data = nib.load(op.join(tmpdir, "dki.nii.gz")).get_fdata()
        npt.assert_array_almost_equal(pred, data)


def test_fit_dti():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dti.bval')
        fbvec = op.join(tmpdir, 'dti.bvec')
        fdata = op.join(tmpdir, 'dti.nii.gz')
        make_dti_data(fbval, fbvec, fdata)
        cmd = ["pyAFQ_dti", "-d", fdata, "-l", fbval, "-c", fbvec,
               "-o", tmpdir, '-b', '0']
        out = runner.run_command(cmd)
        npt.assert_equal(out[0], 0)
        # Get expected values
        names = ['FA', 'MD', 'AD', 'RD']
        for n in names:
            fname = op.join(tmpdir, "dti_%s.nii.gz" % n)
            img = nib.load(fdata)
            affine = img.affine
            shape = img.shape[:-1]
            assert_image_shape_affine(fname, shape, affine)


def test_predict_dti():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dti.bval')
        fbvec = op.join(tmpdir, 'dti.bvec')
        fdata = op.join(tmpdir, 'dti.nii.gz')
        make_dti_data(fbval, fbvec, fdata)
        cmd1 = ["pyAFQ_dti", "-d", fdata, "-l", fbval, "-c", fbvec,
                "-o", tmpdir]
        out = runner.run_command(cmd1)
        npt.assert_equal(out[0], 0)
        # Get expected values
        fparams = op.join(tmpdir, "dti_params.nii.gz")
        cmd2 = ["pyAFQ_dti_predict", "-p", fparams, "-l", fbval, "-c", fbvec,
                "-o", tmpdir, '-b', '0']
        out = runner.run_command(cmd2)
        npt.assert_equal(out[0], 0)
        pred = nib.load(op.join(tmpdir, "dti_prediction.nii.gz")).get_fdata()
        data = nib.load(op.join(tmpdir, "dti.nii.gz")).get_fdata()
        npt.assert_array_almost_equal(pred, data)
