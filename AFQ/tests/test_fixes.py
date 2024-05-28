import nibabel.tmpdirs as nbtmp
import nibabel as nib
from nibabel.streamlines import ArraySequence as Streamlines

import numpy as np

import os.path as op
import numpy.testing as npt

import dipy.core.gradients as dpg
from dipy.data import default_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel

from AFQ._fixes import gaussian_weights as gaussian_weights_fast

from AFQ.utils.testing import make_dki_data
from AFQ._fixes import gwi_odf


def test_GQI_fix():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dki.bval')
        fbvec = op.join(tmpdir, 'dki.bvec')
        fdata = op.join(tmpdir, 'dki.nii.gz')
        make_dki_data(fbval, fbvec, fdata)
        gtab = dpg.gradient_table(fbval, fbvec)
        data = nib.load(fdata).get_fdata()

        gqmodel = GeneralizedQSamplingModel(
            gtab,
            sampling_length=1.2)

        odf_ours = gwi_odf(gqmodel, data)

        odf_theirs = gqmodel.fit(data).odf(default_sphere)

        npt.assert_array_almost_equal(odf_ours, odf_theirs)


def test_mahal_fix():
    sls = [
        [[8.0, 53, 39], [8, 50, 39], [8, 45, 39], [30, 41, 61], [28, 61, 38]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [30, 41, 62], [20, 44, 34]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [50, 67, 88], [10, 10, 20]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [35, 43, 65], [25, 55, 35]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [40, 50, 70], [15, 15, 25]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [45, 54, 75], [12, 22, 32]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [32, 48, 68], [28, 58, 40]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [38, 52, 72], [18, 38, 28]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [34, 44, 64], [21, 41, 31]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [36, 46, 66], [23, 53, 33]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [37, 47, 67], [24, 54, 34]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [39, 49, 69], [19, 39, 29]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [33, 53, 73], [22, 42, 32]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [31, 51, 71], [26, 56, 36]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [29, 59, 79], [27, 57, 37]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [28, 58, 78], [17, 47, 27]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [27, 57, 77], [16, 36, 26]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [26, 56, 76], [14, 24, 34]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [25, 55, 75], [13, 23, 33]],
        [[8, 53, 39], [8, 50, 39], [8, 45, 39], [24, 54, 74], [11, 21, 31]]
    ]
    sls_array =  np.asarray(sls).astype(float)
    results = np.asarray([
        [0., 0., 0., 1.718654, 1.550252],
        [0., 0., 0., 2.202227, 0.7881],
        [0., 0., 0., 3.415999, 2.689814]])
    npt.assert_array_almost_equal(
        gaussian_weights_fast(
            sls_array, n_points=None,
            return_mahalnobis=True,
            stat=np.mean)[:3], results)
