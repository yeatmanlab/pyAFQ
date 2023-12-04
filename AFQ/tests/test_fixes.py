import nibabel.tmpdirs as nbtmp
import nibabel as nib
from nibabel.streamlines import ArraySequence as Streamlines

import numpy as np

import os.path as op
import numpy.testing as npt

import dipy.core.gradients as dpg
from dipy.data import default_sphere
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.tracking.streamline import set_number_of_points

from AFQ._fixes import gaussian_weights as gaussian_weights_fast
from dipy.stats.analysis import gaussian_weights

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
                [
                    [8.0, 53, 39], [8, 50, 39], [8, 45, 39],
                    [30, 41, 61], [28, 61, 38]],
                [
                    [8, 53, 39], [8, 50, 39], [8, 45, 39],
                    [30, 41, 62], [20, 44, 34]],
                [
                    [8, 53, 39], [8, 50, 39], [8, 45, 39],
                    [50, 67, 88], [10, 10, 20]]
            ]
    sls_array =  np.asarray(sls).astype(float)
    results = np.asarray([
        [0.      , 0.      , 0.      , 1.185854, 2.14735],
        [0.      , 0.      , 0.      , 1.185854, 1.556795],
        [0.      , 0.      , 0.      , 1.274755, 2.23296]])
    npt.assert_array_almost_equal(
        gaussian_weights_fast(
            sls_array, n_points=5,
            return_mahalnobis=True, stat=np.mean), results)
