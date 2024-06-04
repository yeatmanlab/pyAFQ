import numpy as np

import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts


def clean_by_other_density_map(this_bundle_sls,
                               other_bundle_sls,
                               node_thresh,
                               img):
    """
    Remove fibers that overlap with more than node_thresh nodes
    """
    other_bundle_density_map = dtu.density_map(
        other_bundle_sls, np.eye(4), img.shape[:3])
    fiber_probabilities = dts.values_from_volume(
        other_bundle_density_map,
        this_bundle_sls, np.eye(4))
    cleaned_idx = np.zeros(len(this_bundle_sls), dtype=np.bool8)
    for ii, fp in enumerate(fiber_probabilities):
        cleaned_idx[ii] = np.sum(np.asarray(fp) >= 1) <= node_thresh
    return cleaned_idx


def clean_relative_to_other_core(core, this_fgarray, other_fgarray):
    """
    Remove any fibers that are on the wrong side of the core
    """
    if core == 'anterior':
        core_axis = 1
        core_direc = -1
    elif core == 'posterior':
        core_axis = 1
        core_direc = 1
    elif core == 'superior':
        core_axis = 2
        core_direc = -1
    elif core == 'inferior':
        core_axis = 2
        core_direc = 1
    elif core == 'right':
        core_axis = 0
        core_direc = -1
    elif core == 'left':
        core_axis = 0
        core_direc = 1

    cutoff_line = np.mean(other_fgarray[:, core_axis, :])
    sls_projected = this_fgarray[:, core_axis, :]
    if core_direc == -1:
        cleaned_idx_core = sls_projected < cutoff_line
    else:
        cleaned_idx_core = sls_projected > cutoff_line
    return np.all(cleaned_idx_core, axis=1)
