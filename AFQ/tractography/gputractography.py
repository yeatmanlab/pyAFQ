import cuslines

import numpy as np
from math import radians
from tqdm import tqdm
import logging

from dipy.reconst.shm import OpdtModel, CsaOdfModel
from dipy.reconst import shm
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from nibabel.streamlines.array_sequence import concatenate
from nibabel.streamlines.tractogram import Tractogram

from trx.trx_file_memmap import TrxFile

from AFQ.tractography.utils import gen_seeds, get_percentile_threshold


logger = logging.getLogger('AFQ')


# Modified from https://github.com/dipy/GPUStreamlines/blob/master/run_dipy_gpu.py
def gpu_track(data, gtab, seed_img, stop_img,
              odf_model, sphere, directions,
              seed_threshold, stop_threshold, thresholds_as_percentages,
              max_angle, step_size, n_seeds, random_seeds, rng_seed, use_trx, ngpus,
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
        set to True. Default: None
    use_trx : bool
        Whether to use trx.
    ngpus : int
        Number of GPUs to use.
    chunk_size : int
        Chunk size for GPU tracking.
    Returns
    -------
    """
    sh_order = 8

    seed_data = seed_img.get_fdata()
    stop_data = stop_img.get_fdata()

    if thresholds_as_percentages:
        stop_threshold = get_percentile_threshold(
            stop_data, stop_threshold)

    theta = sphere.theta
    phi = sphere.phi
    sampling_matrix, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)

    if directions == "boot":
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
            model_type = cuslines.ModelType.CSA
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
    else:
        if directions == "prob":
            model_type = cuslines.ModelType.PROB
        else:
            model_type = cuslines.ModelType.PTT
        model = shm.SphHarmModel(gtab)
        model.cache_set("sampling_matrix", sphere, sampling_matrix)
        model_fit = shm.SphHarmFit(model, data, None)
        data = model_fit.odf(sphere).clip(min=0)
        delta_b = sampling_matrix
        delta_q = sampling_matrix

    b0s_mask = gtab.b0s_mask
    dwi_mask = ~b0s_mask
    x, y, z = model.gtab.gradients[dwi_mask].T
    _, theta, phi = shm.cart2sphere(x, y, z)
    B, _, _ = shm.real_sym_sh_basis(sh_order, theta, phi)
    H = shm.hat(B)
    R = shm.lcr_matrix(H)

    gpu_tracker = cuslines.GPUTracker(
        model_type,
        radians(max_angle),
        1.0,
        stop_threshold,
        step_size,
        0.25,  # relative peak threshold
        radians(45),  # min separation angle
        data.astype(np.float64), H.astype(np.float64), R.astype(np.float64),
        delta_b.astype(np.float64), delta_q.astype(np.float64),
        b0s_mask.astype(np.int32), stop_data.astype(
            np.float64), sampling_matrix.astype(np.float64),
        sphere.vertices.astype(np.float64), sphere.edges.astype(np.int32),
        ngpus=ngpus, rng_seed=0)

    seeds = gen_seeds(
        seed_data, seed_threshold,
        n_seeds, thresholds_as_percentages,
        random_seeds, rng_seed, np.eye(4))

    global_chunk_sz = chunk_size * ngpus
    nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz

    # TODO: this code duplicated with GPUStreamlines...
    # should probably be moved up to trx or cudipy at some point
    if use_trx:
        # Will resize by a factor of 2 if these are exceeded
        sl_len_guess = 100
        sl_per_seed_guess = 3
        n_sls_guess = sl_per_seed_guess * len(seeds.shape[0])

        # trx files use memory mapping
        trx_file = TrxFile(
            reference=seed_img,
            nb_streamlines=n_sls_guess,
            nb_vertices=n_sls_guess * sl_len_guess)
        offsets_idx = 0
        sls_data_idx = 0

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(int(nchunks)):
                streamlines = gpu_tracker.generate_streamlines(
                    seeds[idx * global_chunk_sz:(idx + 1) * global_chunk_sz])
                tractogram = Tractogram(
                    streamlines, affine_to_rasmm=seed_img.affine)
                tractogram.to_world()
                sls = tractogram.streamlines

                new_offsets_idx = offsets_idx + len(sls._offsets)
                new_sls_data_idx = sls_data_idx + len(sls._data)

                if new_offsets_idx > trx_file.header["NB_STREAMLINES"]\
                        or new_sls_data_idx > trx_file.header["NB_VERTICES"]:
                    print("TRX resizing...")
                    trx_file.resize(nb_streamlines=new_offsets_idx
                                    * 2, nb_vertices=new_sls_data_idx * 2)

                # TRX uses memmaps here
                trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
                trx_file.streamlines._offsets[offsets_idx:
                                              new_offsets_idx] = offsets_idx + sls._offsets
                trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = sls._lengths

                offsets_idx = new_offsets_idx
                sls_data_idx = new_sls_data_idx
                pbar.update(
                    seeds[idx * global_chunk_sz:(idx + 1) * global_chunk_sz].shape[0])
        trx_file.resize()

        return trx_file
    else:
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
