import numpy as np
import nibabel as nib
import dipy.reconst.shm as shm
import logging

import dipy.data as dpd
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
import dipy.tracking.utils as dtu
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion

from AFQ._fixes import VerboseLocalTracking, tensor_odf


def track(params_file, directions="det", max_angle=30., sphere=None,
          seed_mask=None, n_seeds=1, random_seeds=False, stop_mask=None,
          stop_threshold=0, step_size=0.5, min_length=10, max_length=1000,
          odf_model="DTI"):
    """
    Tractography

    Parameters
    ----------
    params_file : str, nibabel img.
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.
    directions : str
        How tracking directions are determined.
        One of: {"deterministic" | "probablistic"}
    max_angle : float, optional.
        The maximum turning angle in each step. Default: 30
    sphere : Sphere object, optional.
        The discretization of direction getting. default:
        dipy.data.default_sphere.
    seed_mask : array, optional.
        Binary mask describing the ROI within which we seed for tracking.
        Default to the entire volume.
    n_seeds : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask.
    random_seeds : bool
        Whether to generate a total of n_seeds random seeds in the mask.
        Default: XXX.
    stop_mask : array, optional.
        A floating point value that determines a stopping criterion (e.g. FA).
        Default to no stopping (all ones).
    stop_threshold : float, optional.
        A value of the stop_mask below which tracking is terminated. Default to
        0 (this means that if no stop_mask is passed, we will stop only at
        the edge of the image)
    step_size : float, optional.
        The size (in mm) of a step of tractography. Default: 1.0
    min_length: int, optional
        The miminal length (mm) in a streamline. Default: 10
    max_length: int, optional
        The miminal length (mm) in a streamline. Default: 250
    odf_model : str, optional
        One of {"DTI", "CSD"}. Defaults to use "DTI"
    Returns
    -------
    list of streamlines ()
    """
    logger = logging.getLogger('AFQ.Tractography')

    logger.info("Loading Image...")
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    model_params = params_img.get_fdata()
    affine = params_img.affine

    logger.info("Generating Seeds...")
    if isinstance(n_seeds, int):
        if seed_mask is None:
            seed_mask = np.ones(params_img.shape[:3])
        if random_seeds:
            seeds = dtu.random_seeds_from_mask(seed_mask, seeds_count=n_seeds,
                                               seed_count_per_voxel=False,
                                               affine=affine)
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

    if odf_model == "DTI":
        evals = model_params[..., :3]
        evecs = model_params[..., 3:12].reshape(params_img.shape[:3] + (3, 3))
        odf = tensor_odf(evals, evecs, sphere)
        dg = dg.from_pmf(odf, max_angle=max_angle, sphere=sphere)
    elif odf_model == "CSD":
        dg = dg.from_shcoeff(model_params, max_angle=max_angle, sphere=sphere)

    if stop_mask is None:
        stop_mask = np.ones(params_img.shape[:3])

    threshold_classifier = ThresholdStoppingCriterion(stop_mask,
                                                      stop_threshold)
    logger.info("Tracking...")

    return _local_tracking(seeds, dg, threshold_classifier, params_img,
                           step_size=step_size, min_length=min_length,
                           max_length=max_length)


def _local_tracking(seeds, dg, threshold_classifier, params_img,
                    step_size=0.5, min_length=10, max_length=1000):
    """
    Helper function
    """
    if len(seeds.shape) == 1:
        seeds = seeds[None, ...]
    tracker = VerboseLocalTracking(dg,
                                   threshold_classifier,
                                   seeds,
                                   params_img.affine,
                                   step_size=step_size,
                                   min_length=min_length,
                                   max_length=max_length)

    return StatefulTractogram(tracker, params_img, Space.RASMM)
