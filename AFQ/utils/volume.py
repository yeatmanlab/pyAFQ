import scipy.ndimage as ndim
from skimage.filters import gaussian
from skimage.morphology import convex_hull_image
from scipy.spatial.qhull import QhullError


def patch_up_roi(roi):
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

    Returns
    -------
    ROI after dilation and hole-filling
    """

    hole_filled = ndim.binary_fill_holes(roi > 0)
    try:
        return convex_hull_image(hole_filled)
    except QhullError:
        return hole_filled
