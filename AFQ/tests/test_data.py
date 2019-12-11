
import numpy as np
import numpy.testing as npt

import AFQ.data as afd


def test_aal_to_regions():
    atlas = np.zeros((20, 20, 20, 5))
    # Just one region:
    atlas[0, 0, 0, 0] = 1
    regions = ['leftfrontal']
    idx = afd.aal_to_regions(regions, atlas=atlas)
    npt.assert_equal(idx, np.array(np.where(atlas[..., 0] == 1)).T)

    # More than one region:
    atlas[1, 1, 1, 0] = 2
    regions = ['leftfrontal', 'rightfrontal']
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx,
        np.array(np.where((atlas[..., 0] == 1) | (atlas[..., 0] == 2))).T)

    # Use one of the additional volumes
    atlas[1, 1, 1, 1] = 1
    regions = ['leftfrontal', 'rightfrontal', 'cstsuperior']
    idx = afd.aal_to_regions(regions, atlas=atlas)

    npt.assert_equal(
        idx,
        np.array(np.where((atlas[..., 0] == 1)
                          | (atlas[..., 0] == 2)
                          | (atlas[..., 1] == 1))).T)


def test_bundles_to_aal():
    atlas = np.zeros((20, 20, 20, 5))

    atlas[0, 0, 0, 0] = 1

    targets = afd.bundles_to_aal(["ATR_L"], atlas)
    npt.assert_equal(targets,
                     [[None, np.array(np.where(atlas[..., 0] == 1)).T]])


    atlas[0, 0, 1, 0] = 2

    targets = afd.bundles_to_aal(["ATR_L", "ATR_R"], atlas)
    npt.assert_equal(targets,
                     [[None, np.array(np.where(atlas[..., 0] == 1)).T],
                      [None, np.array(np.where(atlas[..., 0] == 2)).T]])
