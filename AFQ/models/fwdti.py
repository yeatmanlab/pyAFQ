import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import fwdti

import AFQ.utils.models as ut


__all__ = ["fit_fwdti"]


def _fit(data, gtab, mask):
    fwmodel = fwdti.FreeWaterTensorModel(gtab)
    fwfit = fwmodel.fit(data, mask=mask)
    return fwfit


def fit_fwdti(data_files, bval_files, bvec_files, mask=None, out_dir=None,
              b0_threshold=50):
    """
    Fit the free water DTI model [1]_, save files with derived maps

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
        Default: maps get stored in the same directory as the last DWI file
        in `data_files`.
    b0_threshold : float


    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    ..[1] R. Neto Henriques, A. Rokem, E. Garyfallidis, S. St-Jean, E.T.
          Peterson, M. Correia (2017). [Re] Optimization of a free water
          elimination two-compartment model for diffusion tensor imaging.
          *ReScience*

    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files,
                                            bvec_files, mask=mask,
                                            b0_threshold=b0_threshold)

    fwfit = _fit(data, gtab, mask)

    FA = fwfit.fa
    MD = fwfit.md
    AD = fwfit.ad
    RD = fwfit.rd
    fwvf = fwfit.f
    params = fwfit.model_params

    maps = [FA, MD, AD, RD, fwvf, params]
    names = ['FA', 'MD', 'AD', 'RD', 'FWVF', 'params']

    if out_dir is None:
        if isinstance(data_files, list):
            out_dir = op.join(op.split(data_files[0])[0], 'fwdti')
        else:
            out_dir = op.join(op.split(data_files)[0], 'fwdti')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'fwdti_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths
