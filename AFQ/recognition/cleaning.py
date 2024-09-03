import numpy as np
import logging

from scipy.stats import zscore

import dipy.tracking.streamline as dts
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.recognition.utils as abu
from AFQ._fixes import gaussian_weights


logger = logging.getLogger('AFQ')


def clean_by_orientation(streamlines, primary_axis, tol=None):
    """
    Compute the cardinal orientation of each streamline

    Parameters
    ----------
    streamlines : sequence of N by 3 arrays
        Where N is number of nodes in the array, the collection of
        streamlines to filter down to.

    Returns
    -------
    cleaned_idx, indicies of streamlines that passed cleaning
    """
    axis_diff = np.zeros((len(streamlines), 3))
    endpoint_diff = np.zeros((len(streamlines), 3))
    for ii, sl in enumerate(streamlines):
        # endpoint diff is between first and last
        endpoint_diff[ii, :] = np.abs(sl[0, :] - sl[-1, :])
        # axis diff is difference between the nodes, along
        axis_diff[ii, :] = np.sum(np.abs(np.diff(sl, axis=0)), axis=0)

    orientation_along = np.argmax(axis_diff, axis=1)
    along_accepted_idx = orientation_along == primary_axis
    if tol is not None:
        percentage_primary = 100 * axis_diff[:, primary_axis] / np.sum(
            axis_diff, axis=1)
        logger.debug((
            "Maximum primary percentage found: "
            f"{np.max(percentage_primary)}"))
        along_accepted_idx = np.logical_and(
            along_accepted_idx, percentage_primary > tol)

    orientation_end = np.argmax(endpoint_diff, axis=1)
    end_accepted_idx = orientation_end == primary_axis

    cleaned_idx = np.logical_and(
        along_accepted_idx,
        end_accepted_idx)

    return cleaned_idx


def clean_bundle(tg, n_points=100, clean_rounds=5, distance_threshold=3,
                 length_threshold=4, min_sl=20, stat='mean',
                 return_idx=False):
    """
    Clean a segmented fiber group based on the Mahalnobis distance of
    each streamline

    Parameters
    ----------
    tg : StatefulTractogram class instance or ArraySequence
        A whole-brain tractogram to be segmented.
    n_points : int, optional
        Number of points to resample streamlines to.
        Default: 100
    clean_rounds : int, optional.
        Number of rounds of cleaning based on the Mahalanobis distance from
        the mean of extracted bundles. Default: 5
    distance_threshold : float, optional.
        Threshold of cleaning based on the Mahalanobis distance (the units are
        standard deviations). Default: 3.
    length_threshold: float, optional
        Threshold for cleaning based on length (in standard deviations). Length
        of any streamline should not be *more* than this number of stdevs from
        the mean length.
    min_sl : int, optional.
        Number of streamlines in a bundle under which we will
        not bother with cleaning outliers. Default: 20.
    stat : callable or str, optional.
        The statistic of each node relative to which the Mahalanobis is
        calculated. Default: `np.mean` (but can also use median, etc.)
    return_idx : bool
        Whether to return indices in the original streamlines.
        Default: False.
    Returns
    -------
    A StatefulTractogram class instance containing only the streamlines
    that have a Mahalanobis distance smaller than `clean_threshold` from
    the mean of each one of the nodes.
    """
    # Convert string to callable, if that's what you got.
    if isinstance(stat, str):
        stat = getattr(np, stat)

    if hasattr(tg, "streamlines"):
        streamlines = tg.streamlines
    else:
        streamlines = dts.Streamlines(tg)

    # We don't even bother if there aren't enough streamlines:
    if len(streamlines) < min_sl:
        logger.warning((
            "Mahalanobis cleaning halted early"
            " due to low streamline count"))
        if return_idx:
            return tg, np.arange(len(streamlines))
        else:
            return tg

    # Resample once up-front:
    fgarray = np.asarray(abu.resample_tg(streamlines, n_points))

    # Keep this around, so you can use it for indexing at the very end:
    idx = np.arange(len(fgarray))
    # get lengths of each streamline
    lengths = np.array([sl.shape[0] for sl in streamlines])
    # We'll only do this for clean_rounds
    rounds_elapsed = 0
    idx_belong = idx
    while (rounds_elapsed < clean_rounds) and (np.sum(idx_belong) > min_sl):
        # Update by selection:
        idx = idx[idx_belong]
        fgarray = fgarray[idx_belong]
        lengths = lengths[idx_belong]
        rounds_elapsed += 1

        # This calculates the Mahalanobis for each streamline/node:
        m_dist = gaussian_weights(
            fgarray, return_mahalnobis=True,
            n_points=None, stat=stat)
        logger.debug(f"Shape of fgarray: {np.asarray(fgarray).shape}")
        logger.debug(f"Shape of m_dist: {m_dist.shape}")
        logger.debug(f"Maximum m_dist: {np.max(m_dist)}")
        logger.debug((
            f"Maximum m_dist for each fiber: "
            f"{np.max(m_dist, axis=1)}"))

        length_z = zscore(lengths)
        logger.debug(f"Shape of length_z: {length_z.shape}")
        logger.debug(f"Maximum length_z: {np.max(length_z)}")
        logger.debug((
            "length_z for each fiber: "
            f"{length_z}"))

        if not (
                np.any(m_dist > distance_threshold)
                or np.any(length_z > length_threshold)):
            break
        # Select the fibers that have Mahalanobis smaller than the
        # threshold for all their nodes:
        idx_dist = np.all(m_dist < distance_threshold, axis=-1)
        idx_len = length_z < length_threshold
        idx_belong = np.logical_and(idx_dist, idx_len)

        if np.sum(idx_belong) < min_sl:
            # need to sort and return exactly min_sl:
            idx_belong = np.argsort(np.sum(
                m_dist, axis=-1))[:min_sl].astype(int)
            logger.debug((
                f"At rounds elapsed {rounds_elapsed}, "
                "minimum streamlines reached"))
        else:
            idx_removed = idx_belong == 0
            logger.debug((
                f"Rounds elapsed: {rounds_elapsed}, "
                f"num removed: {np.sum(idx_removed)}"))
            logger.debug(f"Removed indicies: {np.where(idx_removed)[0]}")

    # Select based on the variable that was keeping track of things for us:
    if hasattr(tg, "streamlines"):
        out = StatefulTractogram(tg.streamlines[idx], tg, tg.space)
    else:
        out = streamlines[idx]
    if return_idx:
        return out, idx
    else:
        return out
