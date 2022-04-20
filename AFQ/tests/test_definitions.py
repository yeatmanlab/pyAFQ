import os.path as op
import numpy as np
import numpy.testing as npt
import pytest

from bids.layout import BIDSLayout

import AFQ.definitions.image as afm
from AFQ.definitions.image import *  # interprets images from eval
from AFQ.definitions.mapping import *  # interprets mappings from eval
from AFQ.tests.test_api import create_dummy_bids_path


def test_str_instantiates_mixin():
    thresh_image = afm.ThresholdedScalarImage("dti_fa", lower_bound=0.2)
    thresh_image_str = thresh_image.str_for_toml()
    thresh_image_from_str = eval(thresh_image_str)

    npt.assert_(thresh_image.combine == thresh_image_from_str.combine)
    npt.assert_(thresh_image.lower_bound ==
                thresh_image_from_str.lower_bound)
    npt.assert_(thresh_image.upper_bound ==
                thresh_image_from_str.upper_bound)


def test_resample_image():
    image_data = np.zeros((2, 2, 2), dtype=bool)
    image_data[0] = True
    dwi_data = np.zeros((2, 2, 2, 5))
    image_affine = np.eye(4)
    dwi_affine = np.eye(4) * 2
    npt.assert_array_equal(
        afm._resample_image(image_data, None, image_affine, dwi_affine),
        image_data)
    npt.assert_array_equal(
        afm._resample_image(image_data, dwi_data, image_affine, dwi_affine),
        image_data)

    image_data = np.zeros((3, 3, 3), dtype=bool)
    image_data[0] = True
    resampled_image = afm._resample_image(
        image_data, dwi_data, image_affine, dwi_affine)
    npt.assert_array_equal(
        resampled_image.shape,
        dwi_data[..., 0].shape)
    npt.assert_equal(
        resampled_image.dtype,
        image_data.dtype)


@pytest.mark.parametrize("subject", ["01", "02"])
@pytest.mark.parametrize("session", ["01", "02"])
def test_find_path(subject, session):
    bids_dir = create_dummy_bids_path(2, 2)
    bids_layout = BIDSLayout(bids_dir, derivatives=True)

    test_dwi_path = bids_layout.get(
        subject=subject, session=session, return_type="filename",
        suffix="dwi", extension="nii.gz"
    )[0]

    image_file = ImageFile(suffix="seg", filters={'scope': 'synthetic'})
    image_file.find_path(bids_layout, test_dwi_path, subject, session)

    assert image_file.fnames[session][subject] == op.join(
        bids_dir, "derivatives", "dmriprep", "sub-" + subject,
        "ses-" + session, "anat", "seg.nii.gz"
    )

    other_sub = "01" if subject == "02" else "02"
    with pytest.raises(ValueError):
        image_file.find_path(
            bids_layout,
            test_dwi_path,
            subject=other_sub,
            session=session,
        )
