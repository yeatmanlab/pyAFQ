import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import csdeconv as csd
from dipy.core import gradients as dpg

import AFQ.utils.models as ut


def fit_csd(data_files, bval_files, bvec_files, response=None,
            sh_order=8, lambda_=1, tau=0.1, mask=None, out_dir=None):
    """
    Fit the CSD model save file with SH coefficients

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
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files, bvec_files)
    if response is None:
        response, ratio = csd.auto_response(gtab, data, roi_radius=10,
                                            fa_thr=0.7)
    else:
        response, ratio = response

    csdmodel = csd.ConstrainedSphericalDeconvModel(gtab, response,
                                                   sh_order=sh_order)
    csdfit = csdmodel.fit(data, mask=mask)
    if out_dir is None:
        out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.get_affine()
    fname = op.join(out_dir, 'csd_sh_coeff.nii.gz')
    nib.save(nib.Nifti1Image(csdfit.shm_coeff, aff), fname)
    return fname
