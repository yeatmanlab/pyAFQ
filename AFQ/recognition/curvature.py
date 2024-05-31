import numpy as np

import dipy.tracking.streamlinespeed as dps


def sl_curve(sl, n_points):
    """
    Calculate the direction of the displacement between
    each point along a streamline

    Parameters
    ----------
    sl : 2d array-like
        Streamline to calcualte displacements for.
    n_points : int
        Number of points to resample the streamline to

    Returns
    -------
    2d array of shape (len(sl)-1, 3) with displacements
    between each point in sl normalized to 1.
    """
    # Resample to a standardized number of points
    resampled_sl = dps.set_number_of_points(
        sl,
        n_points)

    # displacement at each point
    resampled_sl_diff = np.diff(resampled_sl, axis=0)

    # normalize this displacement
    resampled_sl_diff = resampled_sl_diff / np.linalg.norm(
        resampled_sl_diff, axis=1)[:, None]

    return resampled_sl_diff


def sl_curve_dist(curve1, curve2):
    """
    Calculate the mean angle using the directions of displacement
    between two streamlines

    Parameters
    ----------
    curve1, curve2 : 2d array-like
        Two curves calculated from sl_curve.

    Returns
    -------
    The mean angle between each curve across all steps, in radians
    """
    return np.mean(np.arccos(np.sum(curve1 * curve2, axis=1)))
