from itertools import chain

import numpy as np
import nibabel as nib
import dipy.reconst.shm as shm
import AFQ.localtracking as dtl


import dipy.data as dpd
from dipy.align import Bunch
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
import dipy.tracking.utils as dtu
from dipy.tracking.local import ThresholdTissueClassifier
from dipy.tracking.local.localtrack import local_tracker

from AFQ.dti import tensor_odf
from AFQ.utils.parallel import parfor

"""
The class implemented below is based on dipy/tracking/localtracking.py

Copyright: Dipy developers. See
https://github.com/nipy/dipy/blob/master/LICENSE for details.
"""
# enum TissueClass (tissue_classifier.pxd) is not accessible
# from here. To be changed when minimal cython version > 0.21.
# cython 0.21 - cpdef enum to export values into Python-level namespace
# https://github.com/cython/cython/commit/50133b5a91eea348eddaaad22a606a7fa1c7c457
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)


class ParallelLocalTracking(object):
    """A streamline generator for local tracking methods"""

    @staticmethod
    def _get_voxel_size(affine):
        """Computes the voxel sizes of an image from the affine.

        Checks that the affine does not have any shear because local_tracker
        assumes that the data is sampled on a regular grid.

        """
        lin = affine[:3, :3]
        dotlin = np.dot(lin.T, lin)
        # Check that the affine is well behaved
        if not np.allclose(np.triu(dotlin, 1), 0.):
            msg = ("The affine provided seems to contain shearing, data must "
                   "be acquired or interpolated on a regular grid to be used "
                   "with `LocalTracking`.")
            raise ValueError(msg)
        return np.sqrt(dotlin.diagonal())

    def __init__(self, direction_getter, tissue_classifier, seeds, affine,
                 step_size, max_cross=None, maxlen=5000, fixedstep=True,
                 return_all=True, n_jobs=-1, engine="dask",
                 backend="multiprocessing"):
        """Creates streamlines by using local fiber-tracking.

        Parameters
        ----------
        direction_getter : instance of DirectionGetter
            Used to get directions for fiber tracking.
        tissue_classifier : instance of TissueClassifier
            Identifies endpoints and invalid points to inform tracking.
        seeds : array (N, 3)
            Points to seed the tracking. Seed points should be given in point
            space of the track (see ``affine``).
        affine : array (4, 4)
            Coordinate space for the streamline point with respect to voxel
            indices of input data. This affine can contain scaling, rotational,
            and translational components but should not contain any shearing.
            An identity matrix can be used to generate streamlines in "voxel
            coordinates" as long as isotropic voxels were used to acquire the
            data.
        step_size : float
            Step size used for tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels. By default all initial directions are tracked.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        fixedstep : bool
            If true, a fixed stepsize is used, otherwise a variable step size
            is used.
        return_all : bool
            If true, return all generated streamlines, otherwise only
            streamlines reaching end points or exiting the image.
        """
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        self._voxel_size = self._get_voxel_size(affine)
        self.affine = affine
        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        self.step_size = step_size
        self.fixed = fixedstep
        self.max_cross = max_cross
        self.maxlen = maxlen
        self.return_all = return_all

        # Get inverse transform (lin/offset) for seeds:
        inv_A = np.linalg.inv(self.affine)
        self.lin = inv_A[:3, :3]
        self.offset = inv_A[:3, 3]
        # Preallocate:
        self.F = np.zeros((self.maxlen + 1, 3), dtype=float)
        # Parallelization:
        self.n_jobs = n_jobs
        self.backend = backend
        self.engine = engine

    def generate_streamlines(self):
        streamlines = parfor(self._track, self.seeds, n_jobs=self.n_jobs,
                             backend=self.backend, engine=self.engine)
        streamlines = list(chain(*streamlines))
        return dtu.move_streamlines(streamlines,
                                    self.affine)

    def _track(self, s):
        s = np.dot(self.lin, s) + self.offset
        directions = self.direction_getter.initial_direction(s)
        if directions.size == 0 and self.return_all:
            # only the seed position
            return [s]
        directions = directions[:self.max_cross]
        B = self.F.copy()
        streamlines = []
        for first_step in directions:
            stepsF, tissue_class = local_tracker(self.direction_getter, self.tissue_classifier,s,first_step,self._voxel_size,self.F, self.step_size,self.fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            stepsB, tissue_class = local_tracker(self.direction_getter, self.tissue_classifier, s, -first_step, self._voxel_size, B, self.step_size, self.fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue

            if stepsB == 1:
                streamlines.append(self.F[:stepsF].copy())
            else:
                parts = (B[stepsB-1:0:-1], self.F[:stepsF])
                streamlines.append(np.concatenate(parts, axis=0))

        return streamlines



def track(params_file, directions="det", max_angle=30., sphere=None,
          seed_mask=None, seeds=2, stop_mask=None, stop_threshold=0.2,
          step_size=0.5, n_jobs=-1, backend="multiprocessing", engine="dask"):
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

    threshold_classifier = ThresholdTissueClassifier(stop_mask,
                                                     stop_threshold)
    if n_jobs == 1:
        engine="serial"

    tracker = ParallelLocalTracking(dg,
                                    threshold_classifier,
                                    seeds,
                                    affine,
                                    step_size=step_size,
                                    return_all=True,
                                    n_jobs=n_jobs,
                                    backend=backend,
                                    engine=engine)

    return list(tracker.generate_streamlines())
