import os.path as op

import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.core.gradients as dpg
import dipy.data as dpd
from dipy.reconst.shm import calculate_max_order


from AFQ import csd
from AFQ.utils.testing import make_dti_data


def test_fit_csd():
    fdata, fbval, fbvec = dpd.get_data()
    with nbtmp.InTemporaryDirectory() as tmpdir:
        for sh_order in [4, 8]:
            fname = csd.fit_csd(fdata, fbval, fbvec,
                                out_dir=tmpdir, sh_order=sh_order)
            npt.assert_(op.exists(fname))
            sh_coeffs_img = nib.load(fname)
            npt.assert_equal(sh_order,
                             calculate_max_order(sh_coeffs_img.shape[-1]))
