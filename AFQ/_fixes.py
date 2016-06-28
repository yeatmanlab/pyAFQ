import types

import numpy as np
from scipy.special import lpmv, gammaln
from scipy.spatial.distance import cdist

import dipy.reconst.dki as dki


def dki_prediction(dki_params, gtab, S0=150):
    """
    In Dipy versions < 0.12, there is a bug in DKI prediction, that doesn't
    allow using volumes of S0. This is temporary fix until the bug is fixed
    upstream. See: https://github.com/nipy/dipy/pull/1028

    For now, we provide this as a fix, to monkey-patch into dipy.reconst.dki

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 150

    Returns
    --------
    S : (..., N) ndarray
        Simulated signal based on the DKI model:

    .. math::

        S=S_{0}e^{-bD+\frac{1}{6}b^{2}D^{2}K}
    """
    evals, evecs, kt = dki.split_dki_param(dki_params)

    # Define DKI design matrix according to given gtab
    A = dki.design_matrix(gtab)

    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab.bvals)))
    if isinstance(S0, np.ndarray):
        S0_vol = np.reshape(S0, (len(fevals)))
    else:
        S0_vol = S0
    # looping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = dki.lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        if isinstance(S0_vol, np.ndarray):
            this_S0 = S0_vol[v]
        else:
            this_S0 = S0_vol
        X = np.concatenate((dt, fkt[v] * MD * MD,
                            np.array([np.log(this_S0)])),
                           axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


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


def _orient_generator(out, roi1, roi2):
    """
    Helper function to `orient_by_rois`

    Performs the inner loop separately. This is needed, because functions with
    `yield` always return a generator
    """
    for idx, sl in enumerate(out):
        dist1 = cdist(sl, roi1, 'euclidean')
        dist2 = cdist(sl, roi2, 'euclidean')
        min1 = np.argmin(dist1, 0)
        min2 = np.argmin(dist2, 0)
        if min1[0] > min2[0]:
            yield sl[::-1]
        else:
            yield sl


def _orient_list(out, roi1, roi2):
    """
    Helper function to `orient_by_rois`

    Performs the inner loop separately. This is needed, because functions with
    `yield` always return a generator.

    Flips the streamlines in place (as needed) and returns a reference to the
    updated list.
    """
    for idx, sl in enumerate(out):
        dist1 = cdist(sl, roi1, 'euclidean')
        dist2 = cdist(sl, roi2, 'euclidean')
        min1 = np.argmin(dist1, 0)
        min2 = np.argmin(dist2, 0)
        if min1[0] > min2[0]:
            out[idx] = sl[::-1]
    return out


def orient_by_rois(streamlines, roi1, roi2, in_place=False,
                   as_generator=False, affine=None):
    """Orient a set of streamlines according to a pair of ROIs

    Parameters
    ----------
    streamlines : list or generator
        List or generator of 2d arrays of 3d coordinates. Each array contains
        the xyz coordinates of a single streamline.
    roi1, roi2 : ndarray
        Binary masks designating the location of the regions of interest, or
        coordinate arrays (n-by-3 array with ROI coordinate in each row).
    in_place : bool
        Whether to make the change in-place in the original list
        (and return a reference to the list), or to make a copy of the list
        and return this copy, with the relevant streamlines reoriented.
        Default: False.
    as_generator : bool
        Whether to return a generator as output. Default: False
    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.

    Returns
    -------
    streamlines : list or generator
        The same 3D arrays as a list or generator, but reoriented with respect
        to the ROIs

    Examples
    --------
    >>> streamlines = [np.array([[0, 0., 0],
    ...                          [1, 0., 0.],
    ...                          [2, 0., 0.]]),
    ...                np.array([[2, 0., 0.],
    ...                          [1, 0., 0],
    ...                          [0, 0,  0.]])]
    >>> roi1 = np.zeros((4, 4, 4), dtype=bool)
    >>> roi2 = np.zeros_like(roi1)
    >>> roi1[0, 0, 0] = True
    >>> roi2[1, 0, 0] = True
    >>> orient_by_rois(streamlines, roi1, roi2)
    [array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 2.,  0.,  0.]]), array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 2.,  0.,  0.]])]

    """
    # If we don't already have coordinates on our hands:
    if len(roi1.shape) == 3:
        roi1 = np.asarray(np.where(roi1.astype(bool))).T
    if len(roi2.shape) == 3:
        roi2 = np.asarray(np.where(roi2.astype(bool))).T

    if affine is not None:
        roi1 = apply_affine(affine, roi1)
        roi2 = apply_affine(affine, roi2)

    if as_generator:
        if in_place:
            w_s = "Cannot return a generator when in_place is set to True"
            raise ValueError(w_s)
        return _orient_generator(streamlines, roi1, roi2)

    # If it's a generator on input, we may as well generate it
    # here and now:
    if isinstance(streamlines, types.GeneratorType):
        out = list(streamlines)

    elif in_place:
        out = streamlines
    else:
        # Make a copy, so you don't change the output in place:
        out = deepcopy(streamlines)

    return _orient_list(out, roi1, roi2)
