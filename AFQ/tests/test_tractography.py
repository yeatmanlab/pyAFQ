import os.path as op
import numpy as np
import numpy.testing as npt

import nibabel.tmpdirs as nbtmp

from AFQ.csd import fit_csd
from AFQ.dti import fit_dti
from AFQ.tractography import track
from AFQ.utils.testing import make_tracking_data


seeds = np.array([[-80., -120., -60.],
                  [-81, -121, -61],
                  [-81, -120, -60]])


tmpdir = nbtmp.InTemporaryDirectory()
fbval = op.join(tmpdir.name, 'dti.bval')
fbvec = op.join(tmpdir.name, 'dti.bvec')
fdata = op.join(tmpdir.name, 'dti.nii.gz')
make_tracking_data(fbval, fbvec, fdata)

min_length = 20
step_size = 0.5


def test_csd_tracking():
    for sh_order in [4, 8, 10]:
        fname = fit_csd(fdata, fbval, fbvec,
                        response=((0.0015, 0.0003, 0.0003), 100),
                        sh_order=8, lambda_=1, tau=0.1, mask=None,
                        out_dir=tmpdir.name)
        for directions in ["det", "prob"]:
            sl = track(fname, directions,
                       odf_model="CSD",
                       max_angle=30.,
                       sphere=None,
                       seed_mask=None,
                       n_seeds=seeds,
                       stop_mask=None,
                       step_size=step_size,
                       min_length=min_length).streamlines

            npt.assert_(len(sl[0]) >= step_size * min_length)


def test_dti_tracking():
    fdict = fit_dti(fdata, fbval, fbvec)
    for directions in ["det", "prob"]:
        sl = track(fdict['params'],
                   directions,
                   max_angle=30.,
                   sphere=None,
                   seed_mask=None,
                   n_seeds=1,
                   step_size=step_size,
                   min_length=min_length).streamlines
        npt.assert_(len(sl[0]) >= min_length * step_size)
