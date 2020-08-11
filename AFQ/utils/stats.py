def contrast_index(x1, x2):
    """
    Calculate the contrast index between two arrays.
    The contrast index is symmetrical with respect to
    the choice of the baseline for comparison

    Parameters
    ----------
    x1 : ndarray of floats
        An ndarray to compare. The contrast index will have positive values
        where x1 > x2.
    x2 : ndarray of floats
        An ndarray to compare. The contrast index will have negative values
        where x2 > x1.

    Returns
    -------
    contrast_index : ndarray of floats
        Contrast index calculated by doing (x1 - x2) / (x1 + x2)
    """
    return (x1 - x2) / (x1 + x2)
