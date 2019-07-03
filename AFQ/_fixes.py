import numpy as np

from scipy.special import lpmv, gammaln

from tqdm import tqdm
from dipy.align import Bunch
from dipy.tracking.local import LocalTracking
import random
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)


def spherical_harmonics(m, n, theta, phi):
    """
    An implementation of spherical harmonics that overcomes conda compilation
    issues. See: https://github.com/nipy/dipy/issues/852
    """
    x = np.cos(phi)
    val = lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (gammaln(n - m + 1) - gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val


class ParallelLocalTracking(LocalTracking):
    # this function is copied from https://github.com/nipy/dipy
    # and modified for parallelization / progress bar
    def _generate_streamlines(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        self.lin = inv_A[:3, :3]
        self.offset = inv_A[:3, 3]

        self.F = np.empty((self.max_length + 1, 3), dtype=float)
        self.B = self.F.copy()
        self.pbar = tqdm(total=self.seeds.shape[0])

        # parfor(self._generate_streamlines_helper, self.seeds)
        for s in self.seeds:
            self._generate_streamlines_helper(s)

        self.pbar.close()

    def _generate_streamlines_helper(self, s):
        s = np.dot(self.lin, s) + self.offset
        # Set the random seed in numpy and random
        if self.random_seed is not None:
            s_random_seed = hash(np.abs((np.sum(s)) + self.random_seed)) \
                % (2**32 - 1)
            random.seed(s_random_seed)
            np.random.seed(s_random_seed)
        directions = self.direction_getter.initial_direction(s)
        if directions.size == 0 and self.return_all:
            # only the seed position
            yield [s]
        directions = directions[:self.max_cross]
        for first_step in directions:
            stepsF, tissue_class = self._tracker(s, first_step, self.F)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            first_step = -first_step
            stepsB, tissue_class = self._tracker(s, first_step, self.B)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            if stepsB == 1:
                streamline = self.F[:stepsF].copy()
            else:
                parts = (self.B[stepsB - 1:0:-1], self.F[:stepsF])
                streamline = np.concatenate(parts, axis=0)
            yield streamline
        self.pbar.update(1)