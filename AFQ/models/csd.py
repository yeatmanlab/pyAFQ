import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import csdeconv as csd
from dipy.reconst import mcsd
from dipy.reconst import shm
import dipy.data as dpd
import AFQ.utils.models as ut

# Monkey patch fixed spherical harmonics for conda and fixed solve_qp from
# DIPY dev:
from AFQ._fixes import spherical_harmonics, solve_qp
shm.spherical_harmonics = spherical_harmonics
mcsd.solve_qp = solve_qp

__all__ = ["fit_csd"]


def _model(gtab, data, response=None, sh_order=None, msmt=False):
    """
    Helper function that defines a CSD model.
    """
    if sh_order is None:
        ndata = np.sum(~gtab.b0s_mask)
        # See dipy.reconst.shm.calculate_max_order
        L1 = (-3 + np.sqrt(1 + 8 * ndata)) / 2.0
        sh_order = int(L1)
        if np.mod(sh_order, 2) != 0:
            sh_order = sh_order - 1
        if sh_order > 8:
            sh_order = 8

    if msmt:
        my_model = mcsd.MultiShellDeconvModel
        if response is None:
            mask_wm, mask_gm, mask_csf =\
                mcsd.mask_for_response_msmt(gtab, data)
            response_wm, response_gm, response_csf =\
                mcsd.response_from_mask_msmt(gtab, data,
                                             mask_wm, mask_gm, mask_csf)
            response = np.array([response_wm, response_gm, response_csf])
    else:
        my_model = csd.ConstrainedSphericalDeconvModel
        if response is None:
            response, _ = csd.auto_response_ssst(gtab, data, roi_radii=10,
                                                 fa_thr=0.7)

    csdmodel = my_model(gtab, response, sh_order=sh_order)
    return csdmodel


def _fit(gtab, data, mask, response=None, sh_order=None, lambda_=1, tau=0.1,
         msmt=False):
    """
    Helper function that does the core of fitting a model to data.
    """
    return _model(gtab, data, response, sh_order, msmt).fit(data, mask=mask)


def fit_csd(data_files, bval_files, bvec_files, mask=None, response=None,
            b0_threshold=50, sh_order=None, lambda_=1, tau=0.1, out_dir=None,
            msmt=False):
    """
    Fit the CSD model and save file with SH coefficients.

    Parameters
    ----------
    data_files : str or list.
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list.
        Equivalent to `data_files`.
    bvec_files : str or list.
        Equivalent to `data_files`.
    mask : ndarray, optional.
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    response: tuple, optional.
        The response function to be used by CSD, as a tuple with two elements.
        The first is the eigen-values as an (3,) ndarray and the second is
        the signal value for the response function without diffusion-weighting
        (i.e. S0). If not provided, auto_response will be used to calculate
        these values.
    b0_threshold : float,optional.
      The value of diffusion-weighting under which we consider it to be
      equivalent to 0. Default:50
    sh_order : int, optional.
        default: infer the number of parameters from the number of data
        volumes, but no larger than 8.
    lambda_ : float, optional.
        weight given to the constrained-positivity regularization part of
        the deconvolution equation. Default: 1
    tau : float, optional.
        threshold controlling the amplitude below which the corresponding
        fODF is assumed to be zero.  Ideally, tau should be set to
        zero. However, to improve the stability of the algorithm, tau is
        set to tau*100 % of the mean fODF amplitude (here, 10% by default)
        (see [1]_). Default: 0.1
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: file with coefficients gets stored in the same directory as
        the first DWI file in `data_files`.
    msmt : bool, optional
        If False, standard single-shell CSD will be used. Otherwise,


    Returns
    -------
    fname : the full path to the file containing the SH coefficients.
    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files, bvec_files,
                                            b0_threshold=b0_threshold,
                                            mask=mask)

    csdfit = _fit(gtab, data, mask, response=response, sh_order=sh_order,
                  lambda_=lambda_, tau=tau, msmt=msmt)

    if out_dir is None:
        out_dir = op.join(op.split(data_files)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.affine
    fname = op.join(out_dir, 'csd_sh_coeff.nii.gz')
    nib.save(nib.Nifti1Image(csdfit.shm_coeff, aff), fname)
    return fname
