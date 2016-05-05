import os
import os.path as op
from distutils.version import LooseVersion

import numpy as np
import nibabel as nib

from dipy.reconst import dki
from ._fixes import dki_prediction

import AFQ.utils.models as ut

import dipy

if LooseVersion(dipy.__version__) < '0.12':
    # Monkey patch the fix in:
    dki.dki_prediction = dki_prediction


def fit_dki(data_files, bval_files, bvec_files, mask=None, min_kurtosis=-1,
            max_kurtosis=3, out_dir=None):
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
        Default: maps get stored in the same directory as the last DWI file in
        `data_files`.

    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    Maps that are calculated: FA, MD, AD, RD, MK, AK, RK

    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files, bvec_files)
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
        out_dir = op.join(op.split(data_files[0])[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.get_affine()
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dki_%s.nii.gz' % n)
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
        S0 = nib.load(S0_file).get_data()
        # If the S0 data is 4D, we assume it comes from an acquisition that had
        # B0 measurements in the same volumes described in the gtab:
        if len(S0.shape) == 4:
            S0 = np.mean(S0[..., gtab.b0s_mask], -1)
        # Otherwise, we assume that it's already a 3D volume, and do nothing

    img = nib.load(params_file)
    params = img.get_data()
    pred = dki.dki_prediction(params, gtab, S0=S0)
    fname = op.join(out_dir, 'dki_prediction.nii.gz')
    nib.save(nib.Nifti1Image(pred, img.affine), fname)

    return fname
