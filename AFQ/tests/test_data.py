import numpy as np
import numpy.testing as npt

import AFQ.data.fetch as afd
import nibabel as nib

def test_aal_to_regions():
    atlas = np.zeros((20, 20, 20, 5))
    # Just one region:
    atlas[0, 0, 0, 0] = 1
    regions = ["leftfrontal"]
    idx = afd.aal_to_regions(regions, atlas=atlas)
    npt.assert_equal(idx, np.array(np.where(atlas[..., 0] == 1)).T)

    # More than one region:
    atlas[1, 1, 1, 0] = 2
    regions = ["leftfrontal", "rightfrontal"]
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx, np.array(np.where((atlas[..., 0] == 1) | (atlas[..., 0] == 2))).T
    )

    # Use one of the additional volumes
    atlas[1, 1, 1, 1] = 1
    regions = ["leftfrontal", "rightfrontal", "cstsuperior"]
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx,
        np.array(
            np.where((atlas[..., 0] == 1) | (atlas[..., 0] == 2) | (atlas[..., 1] == 1))
        ).T,
    )


def test_bundles_to_aal():
    atlas = np.zeros((20, 20, 20, 5))
    affine = np.identity(4)

    atlas[0, 0, 0, 0] = 1

    targets = afd.bundles_to_aal(
        ["ATR_L"], nib.Nifti1Image(atlas, affine))
    npt.assert_equal(np.sum(targets["ATR_L_start"].get_fdata()), 1)
    npt.assert_equal(np.sum(targets["ATR_L_end"].get_fdata()), 0)

    atlas[0, 0, 1, 0] = 2

    targets = afd.bundles_to_aal(
        ["ATR_L", "ATR_R"], nib.Nifti1Image(atlas, affine))
    npt.assert_equal(np.sum(targets["ATR_L_start"].get_fdata()), 1)
    npt.assert_equal(np.sum(targets["ATR_L_end"].get_fdata()), 0)
    npt.assert_equal(np.sum(targets["ATR_R_start"].get_fdata()), 1)
    npt.assert_equal(np.sum(targets["ATR_R_end"].get_fdata()), 0)

    atlas = nib.Nifti1Image(atlas, affine)

    targets = afd.bundles_to_aal([], atlas)
    assert len(targets) == 0

    targets = afd.bundles_to_aal(["HCC_L"], atlas)
    npt.assert_equal(targets, {'HCC_L_end': None, 'HCC_L_start': None})

    targets = afd.bundles_to_aal(["VOF"], atlas)
    npt.assert_equal(targets, {'VOF_end': None, 'VOF_start': None})

    
def test_read_roi():
    aff1 = np.eye(4)
    template = nib.Nifti1Image(np.ones((10, 10, 10)), aff1)
    aff2 = aff1[:]
    aff2[0, 0] = -1
    roi = nib.Nifti1Image(np.zeros((10, 10, 10)), aff2)
    img = afd.read_resample_roi(roi, resample_to=template)
    npt.assert_equal(img.affine, template.affine)
