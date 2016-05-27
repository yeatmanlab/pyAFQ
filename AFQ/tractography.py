import numpy as np
import nibabel as nib
import dipy.reconst.shm as shm
import dipy.tracking.local as dtl
import dipy.tracking.utils as dtu
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
import dipy.data as dpd
from dipy.tracking.local.localtrack import local_tracker

from AFQ.dti import tensor_odf
from AFQ.utils.parallel import parfor


class ParallelLocalTracking(dtl.LocalTracking):
    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
                 return_all=True, n_jobs=-1):
        dtl.LocalTracking.__init__(self,
                       direction_getter,
                       tissue_classifier,
                       seeds,
                       affine,
                       step_size,
                       max_cross=max_cross,
                       maxlen=maxlen,
                       fixedstep=fixedstep,
                       return_all=return_all)
        self.n_jobs = n_jobs

    def _generate_streamlines(self):
        N = self.maxlen
        dg = self.direction_getter
        tc = self.tissue_classifier
        ss = self.step_size
        fixed = self.fixed
        max_cross = self.max_cross
        vs = self._voxel_size

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((N + 1, 3), dtype=float)
        B = F.copy()
        sl = []
        return parfor(self.track_from_seed, self.seeds,
                      func_args=[B, lin, offset, dg, max_cross, tc, ss, fixed,
                                 vs, F])


    def track_from_seed(self, s, B, lin, offset, dg, max_cross, tc, ss, fixed,
                        vs, F):
        s = np.dot(lin, s) + offset
        directions = dg.initial_direction(s)
        if directions.size == 0 and self.return_all:
            # only the seed position
            return [s]
        directions = directions[:max_cross]
        for first_step in directions:
            stepsF, tissue_class = local_tracker(dg, tc, s, first_step,
                                                 vs, F, ss, fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            first_step = -first_step
            stepsB, tissue_class = local_tracker(dg, tc, s, first_step,
                                                 vs, B, ss, fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue

            if stepsB == 1:
                streamline = F[:stepsF].copy()
            else:
                parts = (B[stepsB-1:0:-1], F[:stepsF])
                streamline = np.concatenate(parts, axis=0)
            return streamline


def track(params_file, directions="det",
          max_angle=30., sphere=None,
          seed_mask=None, seeds=2,
          stop_mask=None, stop_threshold=0.2, step_size=0.5,
          n_jobs=-1):
    """
    Deterministic tracking using CSD

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
    seed : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds.
    stop_mask : array, optional.
        A floating point value that determines a stopping criterion (e.g. FA).
        Default to no stopping (all ones).
    stop_threshold : float, optional.
        A value of the stop_mask below which tracking is terminated. Default to
        0.2.
    step_size : float, optional.

    Returns
    -------
    LocalTracking object.
    """
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    model_params = params_img.get_data()
    affine = params_img.get_affine()

    if isinstance(seeds, int):
        if seed_mask is None:
            seed_mask = np.ones(params_img.shape[:3])
        seeds = dtu.seeds_from_mask(seed_mask,
                                    density=seeds,
                                    affine=affine)
    if sphere is None:
        sphere = dpd.default_sphere

    if directions == "det":
        dg = DeterministicMaximumDirectionGetter
    elif directions == "prob":
        dg = ProbabilisticDirectionGetter

    # These are models that have ODFs (there might be others in the future...)
    if model_params.shape[-1] == 12 or model_params.shape[-1] == 27:
        model = "ODF"
    # Could this be an SHM model? If the max order is a whole even number, it
    # might be:
    elif shm.calculate_max_order(model_params.shape[-1]) % 2 == 0:
        model = "SHM"

    if model == "SHM":
        dg = dg.from_shcoeff(model_params, max_angle=max_angle, sphere=sphere)

    elif model == "ODF":
        evals = model_params[..., :3]
        evecs = model_params[..., 3:12].reshape(params_img.shape[:3] + (3, 3))
        odf = tensor_odf(evals, evecs, sphere)
        dg = dg.from_pmf(odf, max_angle=max_angle, sphere=sphere)

    if stop_mask is None:
        stop_mask = np.ones(params_img.shape[:3])

    threshold_classifier = dtl.ThresholdTissueClassifier(stop_mask,
                                                         stop_threshold)

    if n_jobs == 1:
        streamlines = dtl.LocalTracking(dg, threshold_classifier,
                                        seeds, affine,
                                        step_size=step_size,
                                        return_all=True)
    else:
        streamlines = ParallelLocalTracking(dg, threshold_classifier,
                                            seeds, affine,
                                            step_size=step_size,
                                            return_all=True,
                                            n_jobs=n_jobs)


    return list(streamlines._generate_streamlines())
