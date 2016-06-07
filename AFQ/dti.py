import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.core.geometry import vector_norm
from dipy.reconst import dti

import AFQ.utils.models as ut


def fit_dti(data_files, bval_files, bvec_files, mask=None,
            out_dir=None):
    """
    Fit the DTI model using default settings, save files with derived maps

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
    Maps that are calculated: FA, MD, AD, RD

    """
    img, data, gtab, mask, out_dir = ut.prepare_data(data_files,
                                                     bval_files,
                                                     bvec_files,
                                                     mask=mask,
                                                     out_dir=out_dir)
    dtimodel = dti.TensorModel(gtab)
    dtifit = dtimodel.fit(data, mask=mask)

    FA = dtifit.fa
    MD = dtifit.md
    AD = dtifit.ad
    RD = dtifit.rd
    params = dtifit.model_params

    maps = [FA, MD, AD, RD, params]
    names = ['FA', 'MD', 'AD', 'RD', 'params']

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.get_affine()
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dti_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths


def predict(params_file, gtab, S0_file=None, out_dir=None):
    """
    Create a signal prediction from DTI params

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
    pred = dti.tensor_prediction(params, gtab, S0=S0)
    fname = op.join(out_dir, 'dti_prediction.nii.gz')
    nib.save(nib.Nifti1Image(pred, img.affine), fname)

    return fname


def tensor_odf(evals, evecs, sphere):
    """
    Calculate the tensor Orientation Distribution Function

    Parameters

    """
    odf = np.zeros((evals.shape[:3] + (sphere.vertices.shape[0],)))
    mask = np.where((evals[..., 0] > 0) &
                    (evals[..., 1] > 0) &
                    (evals[..., 2] > 0))

    lower = 4 * np.pi * np.sqrt(np.prod(evals[mask], -1))
    projection = np.dot(sphere.vertices, evecs[mask])
    projection /= np.sqrt(evals[mask])
    odf[mask] = ((vector_norm(projection) ** -3) / lower).T
    return odf
