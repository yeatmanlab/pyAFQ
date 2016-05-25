import os.path as op
import numpy as np
import numpy.testing as npt
import nibabel.tmpdirs as nbtmp
from AFQ.utils import streamlines as aus
from dipy.tracking.utils import move_streamlines


def test_read_write_trk():
    sl = [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
          np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])]

    with nbtmp.InTemporaryDirectory() as tmpdir:
        fname = op.join(tmpdir, 'sl.trk')
        aus.write_trk(fname, sl)
        new_sl = aus.read_trk(fname)
        npt.assert_equal(list(new_sl), sl)

        # What happens if this set of streamlines has some funky affine
        # associated with it?
        aff = np.eye(4) * np.random.rand()
        aff[:3, 3] = np.array([1, 2, 3])
        aff[3, 3] = 1
        # We move the streamlines, and report the inverse of the affine:
        aus.write_trk(fname, move_streamlines(sl, aff),
                      affine=np.linalg.inv(aff))
        # When we read this, we get back what we put in:
        new_sl = aus.read_trk(fname)
        # Compare each streamline:
        for new, old in zip(new_sl, sl):
            npt.assert_almost_equal(new, old)
