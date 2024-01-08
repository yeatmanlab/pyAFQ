import cuslines

import numpy as np
from math import radians
from tqdm import tqdm

from dipy.data import small_sphere
from dipy.reconst.shm import OpdtModel
from dipy.reconst import shm
from dipy.tracking import utils
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from nibabel.streamlines.array_sequence import concatenate

from AFQ.tractography.tractography import get_percentile_threshold


# Modified from https://github.com/dipy/GPUStreamlines/blob/master/run_dipy_gpu.py
def gpu_track(data, gtab, seed_img, stop_img,
              seed_threshold, stop_threshold, thresholds_as_percentages,
              max_angle, step_size, sampling_density, ngpus):
    """
    Perform GPU tractography on DWI data.

    Parameters
    ----------
    data : ndarray
        DWI data.
    gtab : GradientTable
        The gradient table.
    seed_img : Nifti1Image
        Float or binary mask describing the ROI within which we seed for
        tracking.
    stop_img : Nifti1Image
        A float or binary mask that determines a stopping criterion
        (e.g. FA).
    seed_threshold : float
        The value of the seed_img above which tracking is seeded.
    stop_threshold : float
        The value of the stop_img below which tracking is
        terminated.
    thresholds_as_percentages : bool
        Interpret seed_threshold and stop_threshold as percentages of the
        total non-nan voxels in the seed and stop mask to include
        (between 0 and 100), instead of as a threshold on the
        values themselves. 
    max_angle : float
        The maximum turning angle in each step.
    step_size : float
        The size of a step (in mm) of tractography.
    sampling_density : int
        The seeding density; how many seeds in each voxel on
        each dimension (for example, 2 => [2, 2, 2]).
    ngpus : int
        Number of GPUs to use.
    Returns
    -------
    """
    chunk_size = 100000
    sh_order = 6

    seed_data = seed_img.get_fdata()
    stop_data = stop_img.get_fdata()

    if len(np.unique(seed_data)) > 2:
        if thresholds_as_percentages:
            seed_threshold = get_percentile_threshold(
                seed_data, seed_threshold)
        seed_data = seed_data > seed_threshold

    if thresholds_as_percentages:
        stop_threshold = get_percentile_threshold(
            stop_data, stop_threshold)

    model = OpdtModel(gtab, sh_order=sh_order, min_signal=1)
    fit_matrix = model._fit_matrix
    delta_b, delta_q = fit_matrix

    sphere = small_sphere
    theta = sphere.theta
    phi = sphere.phi
    sampling_matrix, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)

    b0s_mask = gtab.b0s_mask
    dwi_mask = ~b0s_mask
    x, y, z = model.gtab.gradients[dwi_mask].T
    _, theta, phi = shm.cart2sphere(x, y, z)
    B, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
    H = shm.hat(B)
    R = shm.lcr_matrix(H)

    gpu_tracker = cuslines.GPUTracker(
        radians(max_angle),
        1.0,
        stop_threshold,
        step_size,
        data.astype(np.float64), H, R, delta_b, delta_q,
        b0s_mask.astype(np.int32), stop_data.astype(np.float64),
        sampling_matrix,
        sphere.vertices, sphere.edges.astype(np.int32),
        ngpus=ngpus, rng_seed=0)

    seed_mask = utils.seeds_from_mask(
        seed_data, density=sampling_density, affine=np.eye(4))

    global_chunk_sz = chunk_size * ngpus
    nchunks = (seed_mask.shape[0] + global_chunk_sz - 1) // global_chunk_sz

    streamlines_ls = [None] * nchunks
    with tqdm(total=seed_mask.shape[0]) as pbar:
        for idx in range(int(nchunks)):
            streamlines_ls[idx] = gpu_tracker.generate_streamlines(
                seed_mask[idx * global_chunk_sz:(idx + 1) * global_chunk_sz])
            pbar.update(
                seed_mask[idx * global_chunk_sz:(idx + 1) * global_chunk_sz].shape[0])

    sft = StatefulTractogram(
        concatenate(streamlines_ls, 0),
        seed_img, Space.VOX)

    return sft
