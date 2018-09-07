import scipy.ndimage as ndim
from skimage.filters import gaussian

def patch_up_roi(roi):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed

    Returns
    -------
    ROI after dilation and hole-filling
    """

    return ndim.binary_fill_holes(
        ndim.binary_dilation(gaussian(roi)).astype(float))
