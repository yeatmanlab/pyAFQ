import os.path as op
import numpy as np
import numpy.testing as npt
import pytest

from bids.layout import BIDSLayout

import AFQ.definitions.mask as afm
import AFQ.definitions.mapping as ama
from AFQ.definitions.mask import *  # interprets masks from eval
from AFQ.definitions.mapping import *  # interprets mappings from eval
from AFQ.definitions.utils import Definition
from AFQ.tests.test_api import create_dummy_bids_path


def test_str_instantiates_mixin():
    thresh_mask = afm.ThresholdedScalarMask("dti_fa", lower_bound=0.2)
    thresh_mask_str = thresh_mask.str_for_toml()
    thresh_mask_from_str = eval(thresh_mask_str)

    npt.assert_(thresh_mask.combine == thresh_mask_from_str.combine)
    npt.assert_(thresh_mask.lower_bound ==
                thresh_mask_from_str.lower_bound)
    npt.assert_(thresh_mask.upper_bound ==
                thresh_mask_from_str.upper_bound)


def test_resample_mask():
    mask_data = np.zeros((2, 2, 2), dtype=bool)
    mask_data[0] = True
    dwi_data = np.zeros((2, 2, 2, 5))
    mask_affine = np.eye(4)
    dwi_affine = np.eye(4) * 2
    npt.assert_array_equal(
        afm._resample_mask(mask_data, None, mask_affine, dwi_affine),
        mask_data)
    npt.assert_array_equal(
        afm._resample_mask(mask_data, dwi_data, mask_affine, dwi_affine),
        mask_data)

    mask_data = np.zeros((3, 3, 3), dtype=bool)
    mask_data[0] = True
    resampled_mask = afm._resample_mask(
        mask_data, dwi_data, mask_affine, dwi_affine)
    npt.assert_array_equal(
        resampled_mask.shape,
        dwi_data[..., 0].shape)
    npt.assert_equal(
        resampled_mask.dtype,
        mask_data.dtype)


@pytest.mark.parametrize("subject", ["01", "02"])
@pytest.mark.parametrize("session", ["01", "02"])
def test_find_path(subject, session):
    bids_dir = create_dummy_bids_path(2, 2)
    bids_layout = BIDSLayout(bids_dir, derivatives=True)

    test_dwi_path = bids_layout.get(
        subject=subject, session=session, return_type="filename",
        suffix="dwi", extension="nii.gz"
    )[0]

    mask_file = MaskFile("seg", {'scope': 'synthetic'})
    mask_file.find_path(bids_layout, test_dwi_path, subject, session)

    assert mask_file.fnames[session][subject] == op.join(
        bids_dir, "derivatives", "dmriprep", "sub-" + subject,
        "ses-" + session, "anat", "seg.nii.gz"
    )

    other_sub = "01" if subject == "02" else "02"
    with pytest.raises(ValueError):
        mask_file.find_path(
            bids_layout,
            test_dwi_path,
            subject=other_sub,
            session=session,
        )
