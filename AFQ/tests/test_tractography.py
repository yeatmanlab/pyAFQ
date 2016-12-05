import os.path as op
import numpy as np
import numpy.testing as npt

import nibabel.tmpdirs as nbtmp

from AFQ.csd import fit_csd
from AFQ.dti import fit_dti
from AFQ.tractography import track
from AFQ.utils.testing import make_tracking_data


seeds = np.array([[-80., -120., -60.],
                  [-81, -121, -61]])


tmpdir = nbtmp.InTemporaryDirectory()
fbval = op.join(tmpdir.name, 'dti.bval')
fbvec = op.join(tmpdir.name, 'dti.bvec')
fdata = op.join(tmpdir.name, 'dti.nii.gz')
make_tracking_data(fbval, fbvec, fdata)


def test_csd_tracking():
    for sh_order in [4, 8, 10]:
        fname = fit_csd(fdata, fbval, fbvec,
                        response=((0.0015, 0.0003, 0.0003), 100),
                        sh_order=8, lambda_=1, tau=0.1, mask=None,
                        out_dir=tmpdir.name)
        for directions in ["det", "prob"]:
            sl_serial = track(fname, directions,
                              max_angle=30., sphere=None,
                              seed_mask=None,
                              seeds=seeds,
                              stop_mask=None,
                              stop_threshold=0.2,
                              step_size=0.5,
                              n_jobs=1,
                              engine="serial")
            npt.assert_equal(sl_serial[0].shape[-1], 3)
            for engine in ["dask", "joblib"]:
                for backend in ["threading"]:
                    sl_parallel = track(fname, directions,
                                        max_angle=30., sphere=None,
                                        seed_mask=None,
                                        seeds=seeds,
                                        stop_mask=None,
                                        stop_threshold=0.2,
                                        step_size=0.5,
                                        n_jobs=2,
                                        engine=engine,
                                        backend=backend)
                    npt.assert_equal(sl_parallel[0].shape[-1], 3)

                    if directions == 'det':
                        npt.assert_almost_equal(sl_parallel[0], sl_serial[0])


def test_dti_tracking():
    fdict = fit_dti(fdata, fbval, fbvec)
    for directions in ["det", "prob"]:
        sl_serial = track(fdict['params'],
                          directions,
                          max_angle=30.,
                          sphere=None,
                          seed_mask=None,
                          seeds=seeds,
                          stop_mask=None,
                          stop_threshold=0.2,
                          step_size=0.5,
                          engine="serial")
        npt.assert_equal(sl_serial[0].shape[-1], 3)
        for engine in ["dask", "joblib"]:
            for backend in ["threading"]:
                sl_parallel = track(fdict['params'],
                                    directions,
                                    max_angle=30.,
                                    sphere=None,
                                    seed_mask=None,
                                    seeds=seeds,
                                    stop_mask=None,
                                    stop_threshold=0,
                                    step_size=0.5,
                                    n_jobs=2,
                                    engine=engine,
                                    backend=backend)
                npt.assert_equal(sl_parallel[0].shape[-1], 3)

                if directions == 'det':
                    npt.assert_almost_equal(sl_parallel[0], sl_serial[0])
