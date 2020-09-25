import os.path as op

import numpy as np
import numpy.testing as npt

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
from dipy.reconst.shm import calculate_max_order

from AFQ.models import csd


def test_fit_csd():
    fdata, fbval, fbvec = dpd.get_fnames('small_64D')
    with nbtmp.InTemporaryDirectory() as tmpdir:
        # Convert from npy to txt:
        bvals = np.loadtxt(fbval)
        bvecs = np.loadtxt(fbvec)
        np.savetxt(op.join(tmpdir, 'bvals.txt'), bvals)
        np.savetxt(op.join(tmpdir, 'bvecs.txt'), bvecs)
        for msmt in [True, False]:
            for sh_order in [4, 6]:
                fname = csd.fit_csd(
                    fdata,
                    op.join(tmpdir, 'bvals.txt'),
                    op.join(tmpdir, 'bvecs.txt'),
                    out_dir=tmpdir,
                    sh_order=sh_order,
                    msmt=msmt)

                npt.assert_(op.exists(fname))
                sh_coeffs_img = nib.load(fname)
                npt.assert_equal(sh_order,
                                 calculate_max_order(sh_coeffs_img.shape[-1]))
