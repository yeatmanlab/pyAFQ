import os.path as op
import numpy as np
import numpy.testing as npt
import nibabel.tmpdirs as nbtmp
from AFQ.utils import streamlines as aus

def test_read_write_trk():
    sl = [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
          np.array([[0, 0, 0], [0, 0.5, 0.5 ], [0, 1, 1]])]

    with nbtmp.InTemporaryDirectory() as tmpdir:
        fname = op.join(tmpdir, 'sl.trk')
        aus.write_trk(fname, sl)
        new_sl = aus.read_trk(fname)
        npt.assert_equal(list(new_sl), sl)
