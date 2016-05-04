import os.path as op
import numpy as np
import numpy.testing as npt

import nibabel.tmpdirs as nbtmp

import dipy.data as dpd

from AFQ.csd import fit_csd
import AFQ.tractography as track
from AFQ.utils.testing import make_tracking_data


def test_track_csd():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        fbval = op.join(tmpdir, 'dti.bval')
        fbvec = op.join(tmpdir, 'dti.bvec')
        fdata = op.join(tmpdir, 'dti.nii.gz')
        make_tracking_data(fbval, fbvec, fdata)
        fname = fit_csd(fdata, fbval, fbvec,
                        response=((0.0015, 0.0003, 0.0003), 100),
                        sh_order=8, lambda_=1, tau=0.1, mask=None,
                        out_dir=tmpdir)

        sl = track.csd_deterministic(fname, max_angle=30., sphere=None,
                                     seed_mask=None,
                                     seed_density=[1, 1, 1],
                                     stop_mask=None,
                                     stop_threshold=0.2,
                                     step_size=0.5)

        # Generate the first streamline:
        sl0 = next(sl._generate_streamlines())
        npt.assert_equal(sl0.shape[-1], 3)
