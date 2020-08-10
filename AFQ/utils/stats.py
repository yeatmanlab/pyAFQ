def contrast_index(x1, x2):
    """
    Calculate the contrast index between two arrays.
    The contrast index is symmetrical with respect to
    the choice of the baseline for comparison
    """
    return (x1 - x2) / (x1 + x2)
