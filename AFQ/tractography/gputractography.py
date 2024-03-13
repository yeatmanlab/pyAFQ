import cuslines

import numpy as np
from math import radians
from tqdm import tqdm
import logging

from dipy.data import small_sphere
from dipy.reconst.shm import OpdtModel, CsaOdfModel
from dipy.reconst import shm
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from nibabel.streamlines.array_sequence import concatenate

from AFQ.tractography.utils import gen_seeds, get_percentile_threshold


logger = logging.getLogger('AFQ')


# Modified from https://github.com/dipy/GPUStreamlines/blob/master/run_dipy_gpu.py
def gpu_track(data, gtab, seed_img, stop_img, odf_model,
              seed_threshold, stop_threshold, thresholds_as_percentages,
              max_angle, step_size, n_seeds, random_seeds, rng_seed, ngpus,
              chunk_size):
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
    odf_model : str, optional
        One of {"OPDT", "CSA"}
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
    n_seeds : int
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask. Default: 1
    random_seeds : bool
        If True, n_seeds is total number of random seeds to generate.
        If False, n_seeds encodes the density of seeds to generate.
    rng_seed : int
        random seed used to generate random seeds if random_seeds is
        set to True. Default: None    ngpus : int
        Number of GPUs to use.
    chunk_size : int
        Chunk size for GPU tracking.
    Returns
    -------
    """
    sh_order = 6

    seed_data = seed_img.get_fdata()
    stop_data = stop_img.get_fdata()

    if thresholds_as_percentages:
        stop_threshold = get_percentile_threshold(
            stop_data, stop_threshold)

    if odf_model.lower() == "opdt":
        model_type = cuslines.ModelType.OPDT
        model = OpdtModel(
            gtab,
            sh_order=sh_order,
            smooth=0.006,
            min_signal=1)
        fit_matrix = model._fit_matrix
        delta_b, delta_q = fit_matrix
    elif odf_model.lower() == "csa":
        model_type = cuslines.ModelType.CSAODF
        model = CsaOdfModel(
            gtab, sh_order=sh_order,
            smooth=0.006, min_signal=1)
        fit_matrix = model._fit_matrix
        delta_b = fit_matrix
        delta_q = fit_matrix
    else:
        raise ValueError((
            f"odf_model must be 'opdt' or "
            f"'csa', not {odf_model}"))

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

    sph_edges = sphere.edges
    sph_verticies = sphere.vertices

    gtargs = {}
    for var_name in [
        "data",
        "H", "R", "delta_b", "delta_q",
        "b0s_mask", "stop_data", "sampling_matrix",
            "sph_verticies", "sph_edges"]:
        var = locals()[var_name]

        if var_name in ["b0s_mask", "sph_edges"]:
            dtype = np.int32
        else:
            dtype = np.float64

        if not np.asarray(var).flags['C_CONTIGUOUS']:
            logger.warning(f"{var_name} is not C contiguous. Copying...")
            gtargs[var_name] = np.ascontiguousarray(
                var, dtype=dtype)
        else:
            gtargs[var_name] = np.asarray(var, dtype=dtype)

    gpu_tracker = cuslines.GPUTracker(
        model_type,
        radians(max_angle),
        1.0,
        stop_threshold,
        step_size,
        0.25,  # relative peak threshold
        radians(45),  # min separation angle
        gtargs["data"], gtargs["H"], gtargs["R"],
        gtargs["delta_b"], gtargs["delta_q"],
        gtargs["b0s_mask"], gtargs["stop_data"],
        gtargs["sampling_matrix"],
        gtargs["sph_verticies"], gtargs["sph_edges"],
        ngpus=ngpus, rng_seed=0)

    seeds = gen_seeds(
        seed_data, seed_threshold,
        n_seeds, thresholds_as_percentages,
        random_seeds, rng_seed, np.eye(4))

    global_chunk_sz = chunk_size * ngpus
    nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz

    streamlines_ls = [None] * nchunks
    with tqdm(total=seeds.shape[0]) as pbar:
        for idx in range(int(nchunks)):
            streamlines_ls[idx] = gpu_tracker.generate_streamlines(
                seeds[idx * global_chunk_sz:(idx + 1) * global_chunk_sz])
            pbar.update(
                seeds[idx * global_chunk_sz:(idx + 1) * global_chunk_sz].shape[0])

    sft = StatefulTractogram(
        concatenate(streamlines_ls, 0),
        seed_img, Space.VOX)

    return sft
