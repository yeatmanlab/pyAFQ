import os.path as op
import numpy as np
import numpy.testing as npt
import nibabel as nib
import nibabel.tmpdirs as nbtmp
from AFQ.utils import streamlines as aus
import dipy.tracking.utils as dtu


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
        aus.write_trk(fname, dtu.move_streamlines(sl, aff),
                      affine=np.linalg.inv(aff))
        # When we read this, we get back what we put in:
        new_sl = aus.read_trk(fname)
        # Compare each streamline:
        for new, old in zip(new_sl, sl):
            npt.assert_almost_equal(new, old, decimal=4)

def test_bundles_to_tgram():
    bundles = {'b1': [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
                      np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])],
               'b2': [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
                      np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])]}

    bundle_dict = {'b1': {'uid': 1}, 'b2':{'uid': 2}}
    affine = np.array([[2., 0., 0., -80.],
                       [0., 2., 0., -120.],
                       [0., 0., 2., -60.],
                       [0., 0., 0., 1.]])
    tgram = aus.bundles_to_tgram(bundles, bundle_dict, affine)
    new_bundles = aus.tgram_to_bundles(tgram, bundle_dict)
    npt.assert_equal(new_bundles, bundles)


def test_add_bundles():
    t1 = nib.streamlines.Tractogram(
            [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
             np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])])

    t2 = nib.streamlines.Tractogram(
            [np.array([[0, 0, 0], [0, 0, 0.5], [0, 0, 1], [0, 0, 1.5]]),
             np.array([[0, 0, 0], [0, 0.5, 0.5], [0, 1, 1]])])

    aus.add_bundles(t1, t2)