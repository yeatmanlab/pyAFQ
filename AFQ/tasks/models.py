import nibabel as nib

import pimms

import dipy.reconst.dki as dpy_dki
import dipy.reconst.dti as dpy_dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import shm
from dipy.reconst.dki_micro import axonal_water_fraction

from AFQ.tasks.utils import as_file, as_model, as_tf_deriv, with_name
from AFQ.models.dti import noise_from_b0
from AFQ.models.csd import _fit as csd_fit_model
from AFQ.models.dki import _fit as dki_fit_model
from AFQ.models.dti import _fit as dti_fit_model

DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


@pimms.calc("dti_tf")
def dti_fit(dti_params_file, data_data):
    dti_params = nib.load(dti_params_file).get_fdata()
    tm = dpy_dti.TensorModel(data_data["gtab"])
    dti_tf = dpy_dti.TensorFit(tm, dti_params)
    return dti_tf


@as_file(suffix='_model-DTI_diffmodel.nii.gz')
@as_model
def dti(subses_dict, data_data,
        bval_file, bvec_file, b0_threshold, robust_tensor_fitting=False):
    if robust_tensor_fitting:
        bvals, _ = read_bvals_bvecs(
            bval_file, bvec_file)
        sigma = noise_from_b0(
            data_data["data"], data_data["gtab"], bvals,
            mask=data_data["brain_mask_file"], b0_threshold=b0_threshold)
    else:
        sigma = None
    dtf = dti_fit_model(
        data_data["gtab"], data_data["data"],
        mask=data_data["brain_mask_file"], sigma=sigma)
    meta = dict(
        Parameters=dict(
            FitMethod="WLS"),
        OutlierRejection=False,
        ModelURL=f"{DIPY_GH}reconst/dti.py")
    return dtf.model_params, meta


dti_params = pimms.calc("dti_params_file")(dti)


@pimms.calc("dki_tf")
def dki_fit(dki_params_file, data_data):
    dki_params = nib.load(dki_params_file).get_fdata()
    tm = dpy_dki.DiffusionKurtosisModel(data_data["gtab"])
    dki_tf = dpy_dki.DiffusionKurtosisFit(tm, dki_params)
    return dki_tf


@as_file(suffix='_model-DKI_diffmodel.nii.gz')
@as_model
def dki(subses_dict, dwi_affine, data_data):
    dkf = dki_fit_model(
        data_data["gtab"], data_data["data"],
        mask=data_data["brain_mask_file"])
    meta = dict(
        Parameters=dict(
            FitMethod="WLS"),
        OutlierRejection=False,
        ModelURL=f"{DIPY_GH}reconst/dki.py")
    return dkf.model_params, meta


dki_params = pimms.calc("dki_params_file")(dki)


@as_file(suffix='_model-CSD_diffmodel.nii.gz')
@as_model  # TODO: expose csd_fit_kwargs at api.AFQ level
def csd(subses_dict, dwi_affine,
        data_data, csd_fit_kwargs={}, msmt=False):
    csdf = csd_fit_model(
        data_data["gtab"], data_data["data"],
        mask=data_data["brain_mask_file"],
        msmt=msmt, **csd_fit_kwargs)
    meta = csd_fit_kwargs.copy()
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
@as_tf_deriv(tf_name='DTI')
def dti_fa(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.fa


@pimms.calc("dti_cfa_file")
@as_file(suffix='_model-DTI_desc-DEC_FA.nii.gz')
@as_tf_deriv(tf_name='DTI')
def dti_cfa(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.color_fa


@pimms.calc("dti_pdd_file")
@as_file(suffix='_model-DTI_PDD.nii.gz')
@as_tf_deriv(tf_name='DTI')
def dti_pdd(subses_dict, dwi_affine, dti_params_file, dti_tf):
    pdd = dti_tf.directions.squeeze()
    # Invert the x coordinates:
    pdd[..., 0] = pdd[..., 0] * -1
    return pdd


@pimms.calc("dti_md_file")
@as_file('_model-DTI_MD.nii.gz')
@as_tf_deriv('DTI')
def dti_md(subses_dict, dwi_affine, dti_params_file, dti_tf):
    return dti_tf.md


@pimms.calc("dki_fa_file")
@as_file('_model-DKI_FA.nii.gz')
@as_tf_deriv('DKI')
def dki_fa(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.fa


@pimms.calc("dki_md_file")
@as_file('_model-DKI_MD.nii.gz')
@as_tf_deriv('DKI')
def dki_md(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.md


@pimms.calc("dki_awf_file")
@as_file('_model-DKI_AWF.nii.gz')
@as_tf_deriv('DKI')
def dki_awf(subses_dict, dwi_affine, dki_params_file, dki_tf,
            sphere='repulsion100', gtol=1e-2):
    dki_params = nib.load(dki_params_file).get_fdata()
    awf = axonal_water_fraction(dki_params, sphere=sphere, gtol=gtol)
    return awf


@pimms.calc("dki_mk_file")
@as_file('_model-DKI_MK.nii.gz')
@as_tf_deriv('DKI')
def dki_mk(subses_dict, dwi_affine, dki_params_file, dki_tf):
    return dki_tf.mk()


def get_model_plan():
    model_tasks = with_name([
        dti_fit, dki_fit, anisotropic_power_map,
        dti_fa, dti_cfa, dti_pdd, dti_md, dki_fa, dki_md, dki_awf, dki_mk,
        dti_params, dki_params, csd_params])
    return pimms.plan(**model_tasks)
