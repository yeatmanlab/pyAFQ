import numpy as np
import numpy.testing as npt

import AFQ.data.fetch as afd
import nibabel as nib


def test_read_roi():
    aff1 = np.eye(4)
    template = nib.Nifti1Image(np.ones((10, 10, 10)), aff1)
    aff2 = aff1[:]
    aff2[0, 0] = -1
    roi = nib.Nifti1Image(np.zeros((10, 10, 10)), aff2)
    img = afd.read_resample_roi(roi, resample_to=template)
    npt.assert_equal(img.affine, template.affine)
