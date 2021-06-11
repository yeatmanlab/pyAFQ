import nibabel as nib
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
import dipy.core.gradients as dpg

import pimms

import dipy.reconst.dki as dpy_dki
import dipy.reconst.dti as dpy_dti
from dipy.reconst import shm
from dipy.reconst.dki_micro import axonal_water_fraction

from AFQ.tasks.decorators import as_file, as_model, as_dt_deriv
from AFQ.tasks.utils import get_fname, with_name

from AFQ.models.dti import noise_from_b0
from AFQ.models.csd import _fit as csd_fit_model
from AFQ.models.dki import _fit as dki_fit_model
from AFQ.models.dti import _fit as dti_fit_model


DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


outputs = [
    "data", "gtab", "img", "b0_file", "masked_b0_file", "dti_tf",
    "dti_params_file", "dki_tf", "dki_params_file", "csd_params_file",
    "pmap_file", "dti_fa_file", "dti_cfa_file", "dti_pdd_file",
    "dti_md_file", "dki_fa_file", "dki_md_file", "dki_awf_file",
    "dki_mk_file", "brain_mask_file"]


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


@pimms.calc("dti_tf")
def dti_fit(dti_params_file, gtab):
    dti_params = nib.load(dti_params_file).get_fdata()
    tm = dpy_dti.TensorModel(gtab)
    dti_tf = dpy_dti.TensorFit(tm, dti_params)
    return dti_tf


@as_file(suffix='_model-DTI_diffmodel.nii.gz')
@as_model
def dti(subses_dict, dwi_affine, brain_mask_file, data, gtab,
        bval_file, bvec_file, b0_threshold, robust_tensor_fitting=False):
    mask =\
        nib.load(brain_mask_file).get_fdata()
    if robust_tensor_fitting:
        bvals, _ = read_bvals_bvecs(
            bval_file, bvec_file)
        sigma = noise_from_b0(
            data, gtab, bvals,
            mask=mask, b0_threshold=b0_threshold)
    else:
        sigma = None
    dtf = dti_fit_model(
        gtab, data,
        mask=mask, sigma=sigma)
    meta = dict(
        Parameters=dict(
            FitMethod="WLS"),
        OutlierRejection=False,
        ModelURL=f"{DIPY_GH}reconst/dti.py")
    return dtf.model_params, meta


dti_params = pimms.calc("dti_params_file")(dti)


@pimms.calc("dki_tf")
def dki_fit(dki_params_file, gtab):
    dki_params = nib.load(dki_params_file).get_fdata()
    tm = dpy_dki.DiffusionKurtosisModel(gtab)
    dki_tf = dpy_dki.DiffusionKurtosisFit(tm, dki_params)
    return dki_tf


@as_file(suffix='_model-DKI_diffmodel.nii.gz')
@as_model
def dki(subses_dict, dwi_affine, brain_mask_file, gtab, data):
    mask =\
        nib.load(brain_mask_file).get_fdata()
    dkf = dki_fit_model(
        gtab, data,
        mask=mask)
    meta = dict(
        Parameters=dict(
            FitMethod="WLS"),
        OutlierRejection=False,
        ModelURL=f"{DIPY_GH}reconst/dki.py")
    return dkf.model_params, meta


dki_params = pimms.calc("dki_params_file")(dki)


@as_file(suffix='_model-CSD_diffmodel.nii.gz')
@as_model
def csd(subses_dict, dwi_affine,
        brain_mask_file, gtab, data,
        tracking_params,
        csd_response=None, csd_sh_order=None,
        csd_lambda_=1, csd_tau=0.1):
    msmt = (tracking_params["odf_model"] == "MSMT")
    mask =\
        nib.load(brain_mask_file).get_fdata()
    csdf = csd_fit_model(
        gtab, data,
        mask=mask,
        response=csd_response, sh_order=csd_sh_order,
        lambda_=csd_lambda_, tau=csd_tau,
        msmt=msmt)
    meta = dict(
        SphericalHarmonicDegree=csd_sh_order,
        ResponseFunctionTensor=csd_response,
        lambda_=csd_lambda_,
        tau=csd_tau)
    meta["SphericalHarmonicBasis"] = "DESCOTEAUX"
    if msmt:
        model_file = "mcsd.py"
    else:
        model_file = "csdeconv.py"
    meta["ModelURL"] = f"{DIPY_GH}reconst/{model_file}"
    return csdf.shm_coeff, meta


csd_params = pimms.calc("csd_params_file")(csd)


@pimms.calc("pmap_file")
@as_file(suffix='_model-CSD_APM.nii.gz')
def anisotropic_power_map(subses_dict, csd_params_file):
    sh_coeff = nib.load(csd_params_file)
    pmap = shm.anisotropic_power(sh_coeff.get_fdata())
    pmap = nib.Nifti1Image(pmap, sh_coeff.affine)
    return pmap, dict(CSDParamsFile=csd_params_file)


@pimms.calc("dti_fa_file")
@as_file(suffix='_model-DTI_FA.nii.gz')
@as_dt_deriv(tf_name='DTI')
def dti_fa(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.fa


@pimms.calc("dti_cfa_file")
@as_file(suffix='_model-DTI_desc-DEC_FA.nii.gz')
@as_dt_deriv(tf_name='DTI')
def dti_cfa(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.color_fa


@pimms.calc("dti_pdd_file")
@as_file(suffix='_model-DTI_PDD.nii.gz')
@as_dt_deriv(tf_name='DTI')
def dti_pdd(subses_dict, dwi_affine, dti_params_file, dti_tf):
    pdd = dti_tf.directions.squeeze()
    # Invert the x coordinates:
    pdd[..., 0] = pdd[..., 0] * -1
    return pdd


@pimms.calc("dti_md_file")
@as_file('_model-DTI_MD.nii.gz')
@as_dt_deriv('DTI')
def dti_md(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.md


@pimms.calc("dki_fa_file")
@as_file('_model-DKI_FA.nii.gz')
@as_dt_deriv('DKI')
def dki_fa(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.fa


@pimms.calc("dki_md_file")
@as_file('_model-DKI_MD.nii.gz')
@as_dt_deriv('DKI')
def dki_md(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.md


@pimms.calc("dki_awf_file")
@as_file('_model-DKI_AWF.nii.gz')
@as_dt_deriv('DKI')
def dki_awf(subses_dict, dwi_affine, dki_params_file, dki_tf,
            sphere='repulsion100', gtol=1e-2):
    dki_params = nib.load(dki_params_file).get_fdata()
    awf = axonal_water_fraction(dki_params, sphere=sphere, gtol=gtol)
    return awf


@pimms.calc("dki_mk_file")
@as_file('_model-DKI_MK.nii.gz')
@as_dt_deriv('DKI')
def dki_mk(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.mk()


def get_data_plan(brain_mask_definition):
    data_tasks = with_name([
        get_data_gtab, b0, b0_mask,
        dti_fit, dki_fit, anisotropic_power_map,
        dti_fa, dti_cfa, dti_pdd, dti_md, dki_fa, dki_md, dki_awf, dki_mk,
        dti_params, dki_params, csd_params])
    data_tasks["brain_mask_res"] = \
        pimms.calc("brain_mask_file")(
            as_file('_brain_mask.nii.gz')(
                brain_mask_definition.get_mask_getter(in_data=True)))
    return pimms.plan(**data_tasks)
