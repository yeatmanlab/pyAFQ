from dipy.tests import scriptrunner as dts

runner = dts.ScriptRunner(script_sdir='bin',
                          debug_print_var='AFQ_DEBUG_PRINT')

run_command = runner.run_command


def test_dipy_fit_tensor_again():
    with InTemporaryDirectory():
        dwi, bval, bvec = dpd.get_data("small_25")
        # Copy data to tmp directory
        shutil.copyfile(dwi, "small_25.nii.gz")
        shutil.copyfile(bval, "small_25.bval")
        shutil.copyfile(bvec, "small_25.bvec")
        # Call script
        cmd = ["dipy_fit_tensor", "--mask=none", "small_25.nii.gz"]
        out = run_command(cmd)
        assert_equal(out[0], 0)
        # Get expected values
        img = nib.load("small_25.nii.gz")
        affine = img.get_affine()
        shape = img.shape[:-1]
        # Check expected outputs
        assert_image_shape_affine("small_25_fa.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_t2di.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_dirFA.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_ad.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_md.nii.gz", shape, affine)
        assert_image_shape_affine("small_25_rd.nii.gz", shape, affine)
