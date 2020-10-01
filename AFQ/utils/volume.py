import logging
import numpy as np

import scipy.ndimage as ndim
from skimage.filters import gaussian
from skimage.morphology import convex_hull_image
from scipy.spatial.qhull import QhullError
from scipy.spatial.distance import dice

from dipy.io.utils import (create_nifti_header, get_reference_info)
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map


def patch_up_roi(roi, bundle_name="ROI"):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed.

    sigma : float
        The sigma for initial Gaussian smoothing.

    truncate : float
        The truncation for the Gaussian

    bundle_name : str, optional
        Name of bundle, which may be useful for error messages.
        Default: None

    Returns
    -------
    ROI after dilation and hole-filling
    """

    hole_filled = ndim.binary_fill_holes(roi > 0)
    if not np.any(hole_filled):
        raise ValueError((
            f"{bundle_name} found to be empty after "
            "applying the mapping."))
    try:
        return convex_hull_image(hole_filled)
    except QhullError:
        return hole_filled


def density_map(tractogram, n_sls=None, to_vox=False):
    """
    Create a streamline density map.
    based on:
    https://dipy.org/documentation/1.1.1./examples_built/streamline_formats/

    Parameters
    ----------
    tractogram : StatefulTractogram
        Stateful tractogram whose streamlines are used to make
        the density map.
    n_sls : int or None
        n_sls to randomly select to make the density map.
        If None, all streamlines are used.
        Default: None
    to_vox : bool
        Whether to put the stateful tractogram in VOX space before making
        the density map.

    Returns
    -------
    Nifti1Image containing the density map.
    """
    if to_vox:
        tractogram.to_vox()

    sls = tractogram.streamlines
    if n_sls is not None:
        sls = select_random_set_of_streamlines(sls, n_sls)

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = density_map(sls, np.eye(4), vol_dims)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img


def dice_coeff(img1, img2):
    """
    Compute Dice's coefficient between two imagess.

    Parameters
    ----------
    img1 : Nifti1Image
        One image to compare.
    img2 : Nifti1Image
        The other image to compare.

    Returns
    -------
    The dice similarity between the images.
    """
    # scipy's dice function returns the dice *dissimilarity*
    return 1 - dice(
        img1.get_fdata().flatten().astype(bool),
        img2.get_fdata().flatten().astype(bool))
