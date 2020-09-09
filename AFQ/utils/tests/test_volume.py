import numpy as np
import pytest

import AFQ.utils.volume as AFV


def test_patch_up_roi():
    roi_bad = np.zeros((10, 10, 10))
    roi_good = np.ones((10, 10, 10))

    AFV.patch_up_roi(roi_good)
    with pytest.raises(ValueError):
        AFV.patch_up_roi(roi_bad)
