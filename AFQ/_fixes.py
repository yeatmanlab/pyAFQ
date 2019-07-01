import numpy as np
from scipy.special import lpmv, gammaln


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
    delvec: bool, optional
        If True, vec is deleted as soon as possible. Default is True. 
        If False, vec is not deleted, but still squared.

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
    
    vec *= vec
    vec_norm = vec.sum(axis)
    if delvec:
        del vec
    vec_norm **= 0.5
    
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