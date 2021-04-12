import nibabel as nib
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
import dipy.core.gradients as dpg

from pydra import mark
from AFQ.tasks.utils import *

# data tuple: ses, subject, dwi_file, results_dir


@mark.task
@mark.annotate(
    {"return": {"data": object, "gtab": object, "img": object}}
)
def get_data_gtab(subses_tuple, bval_file, bvec_file, b0_threshold, min_bval,
                  max_bval, filter_b=True, patch2self=False):
    img = nib.load(subses_tuple["dwi_file"])
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
    gtab = dpg.gradient_table(bvals, bvecs,
                                b0_threshold=b0_threshold)
    if patch2self:
        from dipy.denoise.patch2self import patch2self
        data = patch2self(data, bvals, b0_threshold=b0_threshold)
    return data, gtab, img


@mark.task
@mark.annotate(
    {"return": {"b0_file": str}}
)
@as_file('_b0.nii.gz')
def b0(subses_tuple, data, gtab, img):
    mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
    mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
    meta = dict(b0_threshold=gtab.b0_threshold,
                source=subses_tuple['dwi_file'])
    return mean_b0_img, meta


@mark.task
@mark.annotate(
    {"return": {"masked_b0_file": str}}
)
@as_file('_maskedb0.nii.gz')
def b0_mask(subses_tuple, masked_b0_img):
    meta = dict(
        source=get_fname(subses_tuple, '_b0.nii.gz'),
        masked=True)
    return masked_b0_img, meta


@mark.task
@mark.annotate(
    {"return": {"brain_mask_file": str}}
)
@as_file('_brain_mask.nii.gz')
def brain_mask(subses_tuple, brain_mask_definition):
    brain_mask, brain_affine, meta =\
        brain_mask_definition.get_for_subses(subses_tuple)
    brain_mask_img = nib.Nifti1Image(
        brain_mask.astype(int),
        brain_affine)
    return brain_mask_img, meta
