import os
import os.path as op

import nibabel as nib

from dipy.reconst import csdeconv as csd
from dipy.reconst import shm
import AFQ.utils.models as ut

# Monkey patch fixed spherical harmonics for conda
from AFQ._fixes import spherical_harmonics
shm.spherical_harmonics = spherical_harmonics


def fit_csd(data_files, bval_files, bvec_files, mask=None, response=None,
            sh_order=8, lambda_=1, tau=0.1, out_dir=None):
    """
    Fit the CSD model and save file with SH coefficients.

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
        Default: file with coefficients gets stored in the same directory as
        the first DWI file in `data_files`.

    Returns
    -------
    fname : the full path to the file containing the SH coefficients.
    """
    img, data, gtab, mask = ut.prepare_data(data_files, bval_files, bvec_files)
    if response is None:
        response, ratio = csd.auto_response(gtab, data, roi_radius=10,
                                            fa_thr=0.7)

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
