import nibabel as nib
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
import dipy.core.gradients as dpg

import pimms
from AFQ.tasks.utils import as_file, get_fname, has_args


@pimms.calc("data", "gtab", "img")
def get_data_gtab(subses_dict, bval_file, bvec_file, b0_threshold, min_bval,
                  max_bval, filter_b=True, patch2self=False):
    img = nib.load(subses_dict["dwi_file"])
    data = img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    if filter_b and (min_bval is not None):
        valid_b = np.logical_or(
            (bvals >= min_bval), (bvals <= b0_threshold))
        data = data[..., valid_b]
        bvals = bvals[valid_b]
        bvecs = bvecs[valid_b]
    if filter_b and (max_bval is not None):
        valid_b = np.logical_or(
            (bvals <= max_bval), (bvals <= b0_threshold))
        data = data[..., valid_b]
        bvals = bvals[valid_b]
        bvecs = bvecs[valid_b]
    gtab = dpg.gradient_table(
        bvals, bvecs,
        b0_threshold=b0_threshold)
    if patch2self:
        from dipy.denoise.patch2self import patch2self
        data = patch2self(data, bvals, b0_threshold=b0_threshold)
    return data, gtab, img


@pimms.calc("b0_file")
@as_file('_b0.nii.gz')
def b0(subses_dict, data, gtab, img):
    mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
    mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
    meta = dict(b0_threshold=gtab.b0_threshold,
                source=subses_dict['dwi_file'])
    return mean_b0_img, meta


@pimms.calc("masked_b0_file")
@as_file('_maskedb0.nii.gz')
def b0_mask(subses_dict, b0_file, brain_mask_file):
    img = nib.load(b0_file)
    brain_mask = nib.load(brain_mask_file).get_fdata().astype(bool)

    masked_data = img.get_fdata()
    masked_data[~brain_mask] = 0

    masked_b0_img = nib.Nifti1Image(masked_data, img.affine)
    meta = dict(
        source=get_fname(subses_dict, '_b0.nii.gz'),
        masked=True)
    return masked_b0_img, meta


@pimms.calc("dwi_img", "dwi_affine")
def load_dwi(subses_dict):
    img = nib.load(subses_dict["dwi_file"])
    return img, img.affine


data_tasks = [get_data_gtab, b0, b0_mask, load_dwi]
