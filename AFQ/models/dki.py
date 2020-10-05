import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import dki
from dipy.reconst import dki_micro
from dipy.core.ndindex import ndindex

import AFQ.utils.models as ut


__all__ = ["fit_dki", "predict"]


def _fit(gtab, data, mask=None):
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    return dkimodel.fit(data, mask=mask)


def fit_dki(data_files, bval_files, bvec_files, mask=None,
            min_kurtosis=-1, max_kurtosis=3, out_dir=None, b0_threshold=50):
    """
    Fit the DKI model, save files with derived maps

    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.
    mask : ndarray, optional
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    min_kurtosis : float, optional
        The minimal plausible value of kurtosis. Default: -1.
    max_kurtosis : float, optional
        The maximal plausible value of kurtosis. Default: 3.
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: maps get stored in the same directory as the last DWI file
        in `data_files`.
    b0_threshold : float


    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    Maps that are calculated: FA, MD, AD, RD, MK, AK, RK

    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files,
                                            bvec_files, mask=mask,
                                            b0_threshold=b0_threshold)

    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    FA = dkifit.fa
    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd
    MK = dkifit.mk(min_kurtosis, max_kurtosis)
    AK = dkifit.ak(min_kurtosis, max_kurtosis)
    RK = dkifit.rk(min_kurtosis, max_kurtosis)
    params = dkifit.model_params

    maps = [FA, MD, AD, RD, MK, AK, RK, params]
    names = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK', 'params']

    if out_dir is None:
        if isinstance(data_files, list):
            out_dir = op.join(op.split(data_files[0])[0], 'dki')
        else:
            out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dki_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths


def avs_dki_df(gtab, data, mask=None, min_signal=1.0e-6):
    r""" Computes mean diffusion kurtosis

    Parameters
    ----------
    gtab : a GradientTable class instance
        The gradient table containing diffusion acquisition parameters.
    data : ndarray ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    mask : array, optional
        A boolean array used to mark the coordinates in the data that should
        be analyzed that has the shape data.shape[:-1]
    min_signal : float
        The minimum signal value. Needs to be a strictly positive
        number. Default: 1.0e-6.

    Returns
    -------
    params : ndarray ([X, Y, Z, ...], 3)
        All parameters estimated from the dir fit DKI model.
        Parameters are ordered as follows:
            1) Direct Mean Diffusivity measure
            2) Direct Mean Kurtosis measure
            3) Direct S0 estimate

    Reference
    ---------
    Henriques, R.N., Correia, M.M., Interpreting age-related changes based
    on the mean signal diffusion kurtosis. 25th Annual Meeting of ISMRM,
    Honolulu, April 22-28, 2017
    """
    params = np.zeros(data.shape[:-1] + (3,))

    bmag = int(np.log10(gtab.bvals.max()))
    b = gtab.bvals.copy() / (10 ** (bmag - 1))  # normalize b units
    b = b.round() * (10 ** (bmag - 1))
    uniqueb = np.unique(b)
    nb = len(uniqueb)

    B = np.zeros((nb, 3))
    B[:, 0] = -uniqueb
    B[:, 1] = 1.0 / 6.0 * uniqueb ** 2
    B[:, 2] = np.ones(nb)

    ng = np.zeros(nb)
    for bi in range(nb):
        ng[bi] = np.sum(b == uniqueb[bi])
    ng = np.sqrt(ng)

    # Prepare mask
    if mask is None:
        mask = np.ones(data.shape[:-1], dtype=bool)
    else:
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
        mask = np.array(mask, dtype=bool, copy=False)

    index = ndindex(mask.shape)
    sig = np.zeros(nb)
    for v in index:
        if mask[v]:
            for bi in range(nb):
                sig[bi] = np.mean(data[v][b == uniqueb[bi]])

            # Define weights as diag(sqrt(ng) * yn**2)
            W = np.diag(ng * sig**2)

            BTW = np.dot(B.T, W)
            inv_BT_W_B = np.linalg.pinv(np.dot(BTW, B))
            invBTWB_BTW = np.dot(inv_BT_W_B, BTW)
            p = np.dot(invBTWB_BTW, np.log(sig))
            p[1] = p[1] / (p[0]**2)
            p[2] = np.exp(p[2])
            params[v] = p
    return params


def fit_mdki(data_files, bval_files, bvec_files, mask=None, out_dir=None,
             b0_threshold=50):
    """
    Fit the DKI model, save files with derived maps

    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.
    mask : ndarray, optional
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    min_kurtosis : float, optional
        The minimal plausible value of kurtosis. Default: -1.
    max_kurtosis : float, optional
        The maximal plausible value of kurtosis. Default: 3.
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: maps get stored in the same directory as the last DWI file
        in `data_files`.
    b0_threshold : float


    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    Maps that are calculated: FA, MD, AD, RD, MK, AK, RK

    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files,
                                            bvec_files, mask=mask,
                                            b0_threshold=b0_threshold)

    params = avs_dki_df(gtab, data, mask=mask)

    MD = params[..., 0]
    MK = params[..., 1]
    S0 = params[..., 2]

    maps = [MD, MK, S0]
    names = ['MD', 'MK', 'S0']

    if out_dir is None:
        if isinstance(data_files, list):
            out_dir = op.join(op.split(data_files[0])[0], 'dki')
        else:
            out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'mdki_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths


def fit_dkimicro(data_files, bval_files, bvec_files, mask=None,
                 min_kurtosis=-1, max_kurtosis=3, out_dir=None,
                 b0_threshold=50):
    """
    Fit the DKI model, save files with derived maps

    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.
    mask : ndarray, optional
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    min_kurtosis : float, optional
        The minimal plausible value of kurtosis. Default: -1.
    max_kurtosis : float, optional
        The maximal plausible value of kurtosis. Default: 3.
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: maps get stored in the same directory as the last DWI file
        in `data_files`.
    b0_threshold : float


    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    Maps that are calculated: FA, MD, AD, RD, MK, AK, RK

    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files,
                                            bvec_files, mask=mask,
                                            b0_threshold=b0_threshold)

    dkimodel = dki_micro.KurtosisMicrostructureModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    AWF = dkifit.awf
    T = dkifit.tortuosity
    Da = dkifit.axonal_diffusivity
    hRD = dkifit.hindered_rd
    hAD = dkifit.hindered_ad
    evals = dkifit.hindered_evals
    hMD = (evals[..., 0] + evals[..., 1] + evals[..., 2]) / 3.0
    params = dkifit.model_params

    maps = [AWF, T, hAD, hRD, hMD, Da, params]
    names = ['AWF', 'T', 'hAD', 'hRD', 'hMD', 'Da', 'params']

    if out_dir is None:
        if isinstance(data_files, list):
            out_dir = op.join(op.split(data_files[0])[0], 'dki')
        else:
            out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dkimicro_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths


def predict(params_file, gtab, S0_file=None, out_dir=None):
    """
    Create a signal prediction from DKI params

    params_file : str
        Full path to a file with parameters saved from a DKI fit

    gtab : GradientTable object
        The gradient table to predict for

    S0_file : str
        Full path to a nifti file that contains S0 measurements to incorporate
        into the prediction. If the file contains 4D data, the volumes that
        contain the S0 data must be the same as the gtab.b0s_mask.


    """
    if out_dir is None:
        out_dir = op.join(op.split(params_file)[0])

    if S0_file is None:
        S0 = 100
    else:
        S0 = nib.load(S0_file).get_fdata()
        # If the S0 data is 4D, we assume it comes from an acquisition that had
        # B0 measurements in the same volumes described in the gtab:
        if len(S0.shape) == 4:
            S0 = np.mean(S0[..., gtab.b0s_mask], -1)
        # Otherwise, we assume that it's already a 3D volume, and do nothing

    img = nib.load(params_file)
    params = img.get_fdata()
    pred = dki.dki_prediction(params, gtab, S0=S0)
    fname = op.join(out_dir, 'dki_prediction.nii.gz')
    nib.save(nib.Nifti1Image(pred, img.affine), fname)

    return fname
