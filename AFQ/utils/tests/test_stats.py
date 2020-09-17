import numpy as np
import numpy.testing as npt
import AFQ.utils.stats as AFS


def test_contrast_index():
    x1 = np.asarray([1, 2, 3.0, 1.1])
    x2 = np.asarray([0.9, 3.0, 5.0, 0.8])
    contrast_index1 = AFS.contrast_index(x1, x2)

    npt.assert_almost_equal(contrast_index1, [
        0.05263158*2,
        -0.2*2,
        -0.25*2,
        0.15789474*2])
