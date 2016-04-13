import os.path as op
import numpy.testing as npt
import nibabel.tmpdirs as nbtmp
import dipy.data as dpd

from AFQ import dti


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
            op.exists(f)
