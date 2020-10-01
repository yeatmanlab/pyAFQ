import numpy as np
import nibabel as nib
import dipy.reconst.shm as shm
import logging

from scipy.spatial.distance import dice

import dipy.data as dpd
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
import dipy.tracking.utils as dtu
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.stopping_criterion import (ThresholdStoppingCriterion,
                                              CmcStoppingCriterion,
                                              ActStoppingCriterion)
from dipy.io.utils import (create_nifti_header, get_reference_info)
from dipy.tracking.streamline import select_random_set_of_streamlines
from dipy.tracking.utils import density_map


from AFQ._fixes import (VerboseLocalTracking, VerboseParticleFilteringTracking,
                        tensor_odf)

import AFQ.registration as reg


def track(params_file, directions="det", max_angle=30., sphere=None,
          seed_mask=None, seed_threshold=0, n_seeds=1, random_seeds=False,
          rng_seed=None, stop_mask=None, stop_threshold=0, step_size=0.5,
          min_length=10, max_length=1000, odf_model="DTI",
          tracker="local"):
    """
    Tractography

    Parameters
    ----------
    params_file : str, nibabel img.
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.
    directions : str
        How tracking directions are determined.
        One of: {"det" | "prob"}
    max_angle : float, optional.
        The maximum turning angle in each step. Default: 30
    sphere : Sphere object, optional.
        The discretization of direction getting. default:
        dipy.data.default_sphere.
    seed_mask : array, optional.
        Float or binary mask describing the ROI within which we seed for
        tracking.
        Default to the entire volume (all ones).
    seed_threshold : float, optional.
        A value of the seed_mask below which tracking is terminated.
        Default to 0.
    n_seeds : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask.
    random_seeds : bool
        Whether to generate a total of n_seeds random seeds in the mask.
        Default: False.
    rng_seed : int
        random seed used to generate random seeds if random_seeds is
        set to True. Default: None
    stop_mask : array or str, optional.
        If array: A float or binary mask that determines a stopping criterion
        (e.g. FA).
        If tuple: it contains a sequence that is interpreted as:
        (pve_wm, pve_gm, pve_csf), each item of which is either a string
        (full path) or a nibabel img to be used in particle filtering
        tractography.
        A tuple is required if tracker is set to "pft".
        Defaults to no stopping (all ones).
    stop_threshold : float or tuple, optional.
        If float, this a value of the stop_mask below which tracking is
        terminated (and stop_mask has to be an array).
        If str, "CMC" for Continuous Map Criterion [Girard2014]_.
                "ACT" for Anatomically-constrained tractography [Smith2012]_.
        A string is required if the tracker is set to "pft".
        Defaults to 0 (this means that if no stop_mask is passed,
        we will stop only at the edge of the image).
    step_size : float, optional.
        The size (in mm) of a step of tractography. Default: 1.0
    min_length: int, optional
        The miminal length (mm) in a streamline. Default: 10
    max_length: int, optional
        The miminal length (mm) in a streamline. Default: 250
    odf_model : str, optional
        One of {"DTI", "CSD", "DKI"}. Defaults to use "DTI"
    tracker : str, optional
        Which strategy to use in tracking. This can be the standard local
        tracking ("local") or Particle Filtering Tracking ([Girard2014]_).
        One of {"local", "pft"}. Default: "local"

    Returns
    -------
    list of streamlines ()

    References
    ----------
    .. [Girard2014] Girard, G., Whittingstall, K., Deriche, R., &
        Descoteaux, M. Towards quantitative connectivity analysis: reducing
        tractography biases. NeuroImage, 98, 266-278, 2014.
    """
    logger = logging.getLogger('AFQ.tractography')

    logger.info("Loading Image...")
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    model_params = params_img.get_fdata()
    affine = params_img.affine
    odf_model = odf_model.upper()
    directions = directions.lower()

    logger.info("Generating Seeds...")
    if isinstance(n_seeds, int):
        if seed_mask is None:
            seed_mask = np.ones(params_img.shape[:3])
        elif seed_mask.dtype != 'bool':
            seed_mask = seed_mask > seed_threshold
        if random_seeds:
            seeds = dtu.random_seeds_from_mask(seed_mask, seeds_count=n_seeds,
                                               seed_count_per_voxel=False,
                                               affine=affine,
                                               random_seed=rng_seed)
        else:
            seeds = dtu.seeds_from_mask(seed_mask,
                                        density=n_seeds,
                                        affine=affine)
    else:
        # If user provided an array, we'll use n_seeds as the seeds:
        seeds = n_seeds
    if sphere is None:
        sphere = dpd.default_sphere

    logger.info("Getting Directions...")
    if directions == "det":
        dg = DeterministicMaximumDirectionGetter
    elif directions == "prob":
        dg = ProbabilisticDirectionGetter

    if odf_model == "DTI" or odf_model == "DKI":
        evals = model_params[..., :3]
        evecs = model_params[..., 3:12].reshape(params_img.shape[:3] + (3, 3))
        odf = tensor_odf(evals, evecs, sphere)
        dg = dg.from_pmf(odf, max_angle=max_angle, sphere=sphere)
    elif odf_model == "CSD":
        dg = dg.from_shcoeff(model_params, max_angle=max_angle, sphere=sphere)

    if tracker == "local":
        if stop_mask is None:
            stop_mask = np.ones(params_img.shape[:3])

        if stop_mask.dtype == 'bool':
            stopping_criterion = ThresholdStoppingCriterion(stop_mask,
                                                            0.5)
        else:
            stopping_criterion = ThresholdStoppingCriterion(stop_mask,
                                                            stop_threshold)

        my_tracker = VerboseLocalTracking

    elif tracker == "pft":
        if not isinstance(stop_threshold, str):
            raise RuntimeError(
                "You are using PFT tracking, but did not provide a string ",
                "'stop_threshold' input. ",
                "Possible inputs are: 'CMC' or 'ACT'")
        if not isinstance(stop_mask, tuple):
            raise RuntimeError(
                "You are using PFT tracking, but did not provide a tuple for",
                "`stop_mask`",
                "input. Expected a (pve_wm, pve_gm, pve_csf) tuple.")
        pves = []
        pve_imgs = []
        vox_sizes = []
        for ii, pve in enumerate(stop_mask):
            if isinstance(pve, str):
                img = nib.load(pve)
            else:
                img = pve
            pve_imgs.append(img)
            pves.append(pve_imgs[-1].get_fdata())
        average_voxel_size = np.mean(vox_sizes)
        pve_wm_img, pve_gm_img, pve_csf_img = pve_imgs
        pve_wm_data, pve_gm_data, pve_csf_data = pves
        pve_wm_data = reg.resample(pve_wm_data, model_params[..., 0],
                                   pve_wm_img.affine,
                                   params_img.affine)
        pve_gm_data = reg.resample(pve_gm_data, model_params[..., 0],
                                   pve_gm_img.affine,
                                   params_img.affine)
        pve_csf_data = reg.resample(pve_csf_data, model_params[..., 0],
                                    pve_csf_img.affine,
                                    params_img.affine)

        vox_sizes.append(np.mean(params_img.header.get_zooms()[:3]))

        my_tracker = VerboseParticleFilteringTracking
        if stop_threshold == "CMC":
            stopping_criterion = CmcStoppingCriterion.from_pve(
                pve_wm_data,
                pve_gm_data,
                pve_csf_data,
                step_size=step_size,
                average_voxel_size=average_voxel_size)
        elif stop_threshold == "ACT":
            stopping_criterion = ActStoppingCriterion.from_pve(
                pve_wm_data,
                pve_gm_data,
                pve_csf_data)

    logger.info("Tracking...")

    return _tracking(my_tracker, seeds, dg, stopping_criterion, params_img,
                     step_size=step_size, min_length=min_length,
                     max_length=max_length, random_seed=rng_seed)


def _tracking(tracker, seeds, dg, stopping_criterion, params_img,
              step_size=0.5, min_length=10, max_length=1000,
              random_seed=None):
    """
    Helper function
    """
    if len(seeds.shape) == 1:
        seeds = seeds[None, ...]

    tracker = tracker(
        dg,
        stopping_criterion,
        seeds,
        params_img.affine,
        step_size=step_size,
        min_length=min_length,
        max_length=max_length,
        random_seed=random_seed)

    return StatefulTractogram(tracker, params_img, Space.RASMM)


def create_density_map(tractogram, n_sls=None, to_vox=False):
    """
    Write streamline density maps.
    based on:
    https://dipy.org/documentation/1.1.1./examples_built/streamline_formats/

    Parameters
    ----------
    tractogram : StatefulTractogram
        Stateful tractogram whose streamlines are used to make
        the density map.
    n_sls : int or None
        n_sls to randomly select to make the density map.
        If None, all streamlines are used.
        Default: None
    to_vox : bool
        Whether to put the stateful tractogram in VOX space before making
        the density map.

    Returns
    -------
    Nifti1Image containing the density map.
    """
    if to_vox:
        tractogram.to_vox()

    sls = tractogram.streamlines
    if n_sls is not None:
        sls = select_random_set_of_streamlines(sls, n_sls)

    affine, vol_dims, voxel_sizes, voxel_order = get_reference_info(tractogram)
    tractogram_density = density_map(sls, np.eye(4), vol_dims)
    nifti_header = create_nifti_header(affine, vol_dims, voxel_sizes)
    density_map_img = nib.Nifti1Image(tractogram_density, affine, nifti_header)

    return density_map_img


def compute_dice_similarity_coefficient(density_map_img1, density_map_img2):
    """
    Compute the Dice dissimilarity between two density maps
    using scipy's dice.

    Parameters
    ----------
    density_map_img1 : Nifti1Image
        One density map to compare.
    density_map_img2 : Nifti1Image
        The other density map to compare.

    Returns
    -------
    The dice similarity between the density maps.
    """
    return 1 - dice(
        density_map_img1.get_fdata().flatten().astype(bool),
        density_map_img2.get_fdata().flatten().astype(bool))
