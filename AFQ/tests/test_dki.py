from AFQ import dki
import numpy.testing as npt


def test_fit_dki_inputs():
    data_files = ["String in a list"]
    bval_files = "just a string"
    bvec_files = "just another string"
    npt.assert_raises(ValueError, dki.fit_dki, data_files, bval_files,
                      bvec_files)
