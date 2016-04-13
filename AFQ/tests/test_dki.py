import os.path as op

import numpy.testing as npt
import nibabel.tmpdirs as nbtmp
import dipy.data as dpd

from AFQ import dki


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
