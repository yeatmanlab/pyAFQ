import numpy as np
import dipy.reconst.dki as dki


def dki_prediction(dki_params, gtab, S0=150):
    """
    In Dipy versions < 0.12, there is a bug in DKI prediction, that doesn't allow

    See: https://github.com/nipy/dipy/pull/1028

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
        X = np.concatenate((dt, fkt[v]*MD*MD,
                            np.array([np.log(this_S0)])),
                           axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig
