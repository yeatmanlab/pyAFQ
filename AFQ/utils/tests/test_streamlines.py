import numpy as np
import numpy.testing as npt
import nibabel as nib
from AFQ.utils import streamlines as aus
import dipy.tracking.streamline as dts
from dipy.io.stateful_tractogram import StatefulTractogram, Space


def test_SegmentedSFT():

    affine = np.array([[2., 0., 0., -80.],
                       [0., 2., 0., -120.],
                       [0., 0., 2., -60.],
                       [0., 0., 0., 1.]])
    img = nib.Nifti1Image(np.ones((10, 10, 10, 30)), affine)

    bundles = {'ba': StatefulTractogram([np.array([[0, 0, 0],
                                                   [0, 0, 0.5],
                                                   [0, 0, 1],
                                                   [0, 0, 1.5]]),
                                         np.array([[0, 0, 0],
                                                   [0, 0.5, 0.5],
                                                   [0, 1, 1]])],
                                         img,
                                         Space.VOX),
               'bb': StatefulTractogram([np.array([[0, 0, 0],
                                                   [0, 0, 0.5],
                                                   [0, 0, 2],
                                                   [0, 0, 2.5]]),
                                         np.array([[0, 0, 0],
                                                   [0, 0.5, 0.5],
                                                   [0, 2, 2]])],
                                         img,
                                         Space.VOX)}

    seg_sft = aus.SegmentedSFT(bundles, Space.VOX)
    for k1 in bundles.keys():
        for sl1, sl2 in zip(
                bundles[k1].streamlines,
                seg_sft.get_bundle(k1).streamlines):
            npt.assert_equal(sl1, sl2)


def test_split_streamline():
    streamlines = dts.Streamlines([np.array([[1.,2.,3.],
                                    [4.,5.,6.]]),
                                   np.array([[7.,8.,9.],
                                    [10.,11.,12.],
                                    [13., 14., 15.]])])
    assert streamlines == streamlines
    sl_to_split = 1
    split_idx = 1
    new_streamlines = aus.split_streamline(streamlines, sl_to_split, split_idx)
    test_streamlines = dts.Streamlines([np.array([[1., 2., 3.],
                                                  [4., 5., 6.]]),
                                        np.array([[7., 8., 9.]]),
                                        np.array([[10., 11., 12.],
                                                  [13., 14., 15.]])])

    # Test equality of the underlying dict items:
    for k in new_streamlines.__dict__.keys():
        if isinstance(new_streamlines.__dict__[k], np.ndarray):
            npt.assert_array_equal(
                new_streamlines.__dict__[k],
                test_streamlines.__dict__[k]
                )
        else:
            assert new_streamlines.__dict__[k] == test_streamlines.__dict__[k]
