import numpy as np

from scipy.special import lpmv, gammaln

#from tqdm import tqdm
from dipy.align import Bunch
from dipy.tracking.local import LocalTracking
import random
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)
from dask.distributed import Client
from dask import delayed
from collections import deque

import sys

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
    def _generate_streamlines(self, recent_results_size=20):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        for s in self.seeds:
            s = np.dot(lin, s) + offset
            yield delayed(self._generate_streamlines_helper)(s).compute()

        #pbar = tqdm(total=self.seeds.shape[0])

        # client = Client(processes=False)
        # #[lin, offset, F, B] = client.scatter([lin, offset, F, B])

        # if self.seeds.shape[0] < recent_results_size:
        #     recent_results_size = self.seeds.shape[0]
        # recent_results = deque([])

        # for s in self.seeds[:recent_results_size]:
        #     s = np.dot(lin, s) + offset
        #     recent_results.appendleft(client.submit(
        #         self._generate_streamlines_helper, s))
        
        # for s in self.seeds[recent_results_size:]:   
        #     s = np.dot(lin, s) + offset         
        #     recent_results.appendleft(client.submit(
        #         self._generate_streamlines_helper, s))
            
        #     #pbar.update(1)
        #     yield recent_results.pop().result()
            
        # for _ in recent_results_size:
        #     #pbar.update(1)
        #     yield recent_results.pop().result()

        # client.close()
        # #pbar.close()

    def _generate_streamlines_helper(self, s):
        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        # Set the random seed in numpy and random
        if self.random_seed is not None:
            s_random_seed = hash(np.abs((np.sum(s)) + self.random_seed)) \
                % (2**32 - 1)
            random.seed(s_random_seed)
            np.random.seed(s_random_seed)
        directions = self.direction_getter.initial_direction(s)
        if directions.size == 0 and self.return_all:
            # only the seed position
            return [s]
        directions = directions[:self.max_cross]
        for first_step in directions:
            stepsF, tissue_class = self._tracker(s, first_step, F)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            first_step = -first_step
            stepsB, tissue_class = self._tracker(s, first_step, B)
            if not (self.return_all or
                    tissue_class == TissueTypes.ENDPOINT or
                    tissue_class == TissueTypes.OUTSIDEIMAGE):
                continue
            if stepsB == 1:
                streamline = F[:stepsF].copy()
            else:
                parts = (B[stepsB - 1:0:-1], F[:stepsF])
                streamline = np.concatenate(parts, axis=0)
            return streamline


def in_place_norm(vec, axis=-1, keepdims=False, delvec=True):
    """ Return Vectors with Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    -------------
    vec : array_like
        Vectors to norm. Squared in the process of calculating the norm.
    axis : int, optional
        Axis over which to norm. By default norm over last axis. If `axis` is
        None, `vec` is flattened then normed. Default is -1.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as `vec`,
        with shape 1 on `axis`. Default is False.
    delvec : bool, optional
        If True, vec is deleted as soon as possible.
        If False, vec is not deleted, but still squared. Default is True.

    Returns
    ---------
    norm : array
        Euclidean norms of vectors.

    Examples
    --------
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec)
    array([ 17.,  85.])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, keepdims=True)
    array([[ 17.],
           [ 85.]])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, axis=0)
    array([  8.,  39.,  77.])
    """
    vec = np.asarray(vec)

    if keepdims:
        ndim = vec.ndim
        shape = vec.shape

    np.square(vec, out=vec)
    vec_norm = vec.sum(axis)
    if delvec:
        del vec
    try:
        np.sqrt(vec_norm, out=vec_norm)
    except TypeError:
        vec_norm = vec_norm.astype(float)
        np.sqrt(vec_norm, out=vec_norm)

    if keepdims:
        if axis is None:
            shape = [1] * ndim
        else:
            shape = list(shape)
            shape[axis] = 1
        vec_norm = vec_norm.reshape(shape)

    return vec_norm


def tensor_odf(evals, evecs, sphere):
    """
    Calculate the tensor Orientation Distribution Function

    Parameters
    ----------
    evals : array (4D)
        Eigenvalues of a tensor. Shape (x, y, z, 3).
    evecs : array (5D)
        Eigenvectors of a tensor. Shape (x, y, z, 3, 3)
    sphere : sphere object
        The ODF will be calculated in each vertex of this sphere.
    """
    mask = np.where((evals[..., 0] > 0)
                    & (evals[..., 1] > 0)
                    & (evals[..., 2] > 0))

    projection = np.dot(sphere.vertices, evecs[mask])
    projection /= np.sqrt(evals[mask])
    proj_norm = in_place_norm(projection)
    proj_norm **= -3
    proj_norm /= 4 * np.pi * np.sqrt(np.prod(evals[mask], -1))

    odf = np.zeros((evals.shape[:3] + (sphere.vertices.shape[0],)))
    odf[mask] = proj_norm.T
    return odf
