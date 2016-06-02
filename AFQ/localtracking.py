"""
This is based on dipy/tracking/localtracking.py

Copyright: Dipy developers. See
https://github.com/nipy/dipy/blob/master/LICENSE for details.
"""


import numpy as np

from dipy.tracking.local.localtrack import local_tracker
from dipy.align import Bunch
from dipy.tracking import utils
from AFQ.utils.parallel import parfor

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
                 step_size, max_cross=None, maxlen=500, fixedstep=True,
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
        self.direction_getter = direction_getter
        self.tissue_classifier = tissue_classifier
        self.seeds = seeds
        if affine.shape != (4, 4):
            raise ValueError("affine should be a (4, 4) array.")
        self.affine = affine
        self._voxel_size = self._get_voxel_size(affine)
        self.step_size = step_size
        self.fixed = fixedstep
        self.max_cross = max_cross
        self.maxlen = maxlen
        self.return_all = return_all
                # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        self.lin = inv_A[:3, :3]
        self.offset = inv_A[:3, 3]
        self.F = np.empty((self.maxlen + 1, 3), dtype=float)
        self.n_jobs = n_jobs
        self.backend = backend
        self.engine = engine

    def generate_streamlines(self):
        streamlines = parfor(self._track, self.seeds, n_jobs=self.n_jobs,
                             backend=self.backend, engine=self.engine)
        return utils.move_streamlines(streamlines,
                                      self.affine)

    def _track(self, s):
        s = np.dot(self.lin, s) + self.offset
        directions = self.direction_getter.initial_direction(s)
        if directions.size == 0 and self.return_all:
            # only the seed position
            return [s]
        directions = directions[:self.max_cross]
        B = self.F.copy()
        for first_step in directions:
            stepsF, tissue_class = local_tracker(self.direction_getter,
                                                 self.tissue_classifier,
                                                 s, first_step,
                                                 self._voxel_size,
                                                 self.F, self.step_size,
                                                 self.fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            first_step = -first_step
            stepsB, tissue_class = local_tracker(self.direction_getter,
                                                 self.tissue_classifier,
                                                 s,
                                                 first_step,
                                                 self._voxel_size,
                                                 B,
                                                 self.step_size,
                                                 self.fixed)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue

            if stepsB == 1:
                streamline = self.F[:stepsF].copy()
            else:
                parts = (B[stepsB-1:0:-1], self.F[:stepsF])
                streamline = np.concatenate(parts, axis=0)
            return streamline
