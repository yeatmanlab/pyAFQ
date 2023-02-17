import nibabel as nib
import numpy as np

from dipy.io.gradients import read_bvals_bvecs
import dipy.core.gradients as dpg

import pimms

import dipy.reconst.dki as dpy_dki
import dipy.reconst.dti as dpy_dti
import dipy.reconst.fwdti as dpy_fwdti
from dipy.reconst import shm
from dipy.reconst.dki_micro import axonal_water_fraction
from AFQ.definitions.image import ImageDefinition

from AFQ.tasks.decorators import as_file, as_img, as_fit_deriv
from AFQ.tasks.utils import get_fname, with_name
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd

from AFQ.definitions.utils import Definition
from AFQ.definitions.image import B0Image

from AFQ.models.dti import noise_from_b0
from AFQ.models.csd import _fit as csd_fit_model
from AFQ.models.csd import CsdNanResponseError
from AFQ.models.dki import _fit as dki_fit_model
from AFQ.models.dti import _fit as dti_fit_model
from AFQ.models.fwdti import _fit as fwdti_fit_model


DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


@pimms.calc("data", "gtab", "img")
def get_data_gtab(dwi, bval, bvec, min_bval=None,
                  max_bval=None, filter_b=True, b0_threshold=50):
    """
    DWI data as an ndarray for selected b values,
    A DIPY GradientTable with all the gradient information,
    and unaltered DWI data in a Nifti1Image.

    Parameters
    ----------
    min_bval : float, optional
        Minimum b value you want to use
        from the dataset (other than b0), inclusive.
        If None, there is no minimum limit. Default: None
    max_bval : float, optional
        Maximum b value you want to use
        from the dataset (other than b0), inclusive.
        If None, there is no maximum limit. Default: None
    filter_b : bool, optional
        Whether to filter the DWI data based on min or max bvals.
        Default: True
    b0_threshold : int, optional
        The value of b under which
        it is considered to be b0. Default: 50.
    """
    img = nib.load(dwi)
    data = img.get_fdata()
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
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
    return data, gtab, img


@pimms.calc("b0")
@as_file('_desc-b0_dwi.nii.gz')
def b0(dwi, data, gtab, img):
    """
    full path to a nifti file containing the mean b0
    """
    mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
    mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
    meta = dict(b0_threshold=gtab.b0_threshold,
                source=dwi)
    return mean_b0_img, meta


@pimms.calc("masked_b0")
@as_file('_desc-maskedb0_dwi.nii.gz')
def b0_mask(base_fname, b0, brain_mask):
    """
    full path to a nifti file containing the
    mean b0 after applying the brain mask
    """
    img = nib.load(b0)
    brain_mask = nib.load(brain_mask).get_fdata().astype(bool)

    masked_data = img.get_fdata()
    masked_data[~brain_mask] = 0

    masked_b0_img = nib.Nifti1Image(masked_data, img.affine)
    meta = dict(
        source=b0,
        masked=True)
    return masked_b0_img, meta


@pimms.calc("dti_tf")
def dti_fit(dti_params, gtab):
    """DTI TensorFit object"""
    dti_params = nib.load(dti_params).get_fdata()
    tm = dpy_dti.TensorModel(gtab)
    return dpy_dti.TensorFit(tm, dti_params)


@as_file(suffix='_model-DTI_desc-diffmodel_dwi.nii.gz')
@as_img
def dti(brain_mask, data, gtab,
        bval, bvec, b0_threshold=50, robust_tensor_fitting=False):
    """
    full path to a nifti file containing parameters
    for the DTI fit

    Parameters
    ----------
    robust_tensor_fitting : bool, optional
        Whether to use robust_tensor_fitting when
        doing dti. Only applies to dti.
        Default: False
    b0_threshold : int, optional
        The value of b under which
        it is considered to be b0. Default: 50.
    """
    mask =\
        nib.load(brain_mask).get_fdata()
    if robust_tensor_fitting:
        bvals, _ = read_bvals_bvecs(
            bval, bvec)
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
        OutlierRejection=robust_tensor_fitting,
        ModelURL=f"{DIPY_GH}reconst/dti.py")
    return dtf.model_params, meta


dti_params = pimms.calc("dti_params")(dti)


@pimms.calc("fwdti_tf")
def fwdti_fit(fwdti_params, gtab):
    """Free-water DTI TensorFit object"""
    fwdti_params = nib.load(fwdti_params).get_fdata()
    fwtm = dpy_fwdti.FreeWaterTensorModel(gtab)
    return dpy_fwdti.FreeWaterTensorFit(fwtm, fwdti_params)


@as_file(suffix='_model-FWDTI_desc-diffmodel_dwi.nii.gz')
@as_img
def fwdti(brain_mask, data, gtab):
    """
    Full path to a nifti file containing parameters
    for the free-water DTI fit.
    """
    mask =\
        nib.load(brain_mask).get_fdata()
    dtf = fwdti_fit_model(
        data, gtab,
        mask=mask)
    meta = dict(
        Parameters=dict(
            FitMethod="NLS"),
        ModelURL=f"{DIPY_GH}reconst/fwdti.py")
    return dtf.model_params, meta


fwdti_params = pimms.calc("fwdti_params")(fwdti)


@pimms.calc("dki_tf")
def dki_fit(dki_params, gtab):
    """DKI DiffusionKurtosisFit object"""
    dki_params = nib.load(dki_params).get_fdata()
    tm = dpy_dki.DiffusionKurtosisModel(gtab)
    return dpy_dki.DiffusionKurtosisFit(tm, dki_params)


@as_file(suffix='_model-DKI_desc-diffmodel_dwi.nii.gz')
@as_img
def dki(brain_mask, gtab, data):
    """
    full path to a nifti file containing
    parameters for the DKI fit
    """
    mask =\
        nib.load(brain_mask).get_fdata()
    dkf = dki_fit_model(
        gtab, data,
        mask=mask)
    meta = dict(
        Parameters=dict(
            FitMethod="WLS"),
        OutlierRejection=False,
        ModelURL=f"{DIPY_GH}reconst/dki.py")
    return dkf.model_params, meta


dki_params = pimms.calc("dki_params")(dki)


@as_file(suffix='_model-CSD_desc-diffmodel_dwi.nii.gz')
@as_img
def csd(dwi, brain_mask, gtab, data,
        csd_response=None, csd_sh_order=None,
        csd_lambda_=1, csd_tau=0.1):
    """
    full path to a nifti file containing
    parameters for the CSD fit

    Parameters
    ----------
    csd_response : tuple or None, optional.
        The response function to be used by CSD, as a tuple with two elements.
        The first is the eigen-values as an (3,) ndarray and the second is
        the signal value for the response function without diffusion-weighting
        (i.e. S0). If not provided, auto_response will be used to calculate
        these values.
        Default: None
    csd_sh_order : int or None, optional.
        default: infer the number of parameters from the number of data
        volumes, but no larger than 8.
        Default: None
    csd_lambda_ : float, optional.
        weight given to the constrained-positivity regularization part of
        the deconvolution equation. Default: 1
    csd_tau : float, optional.
        threshold controlling the amplitude below which the corresponding
        fODF is assumed to be zero.  Ideally, tau should be set to
        zero. However, to improve the stability of the algorithm, tau is
        set to tau*100 percent of the mean fODF amplitude (here, 10 percent
        by default)
        (see [1]_). Default: 0.1

    References
    ----------
    .. [1] Tournier, J.D., et al. NeuroImage 2007. Robust determination of
            the fibre orientation distribution in diffusion MRI:
            Non-negativity constrained super-resolved spherical
            deconvolution
    """
    mask =\
        nib.load(brain_mask).get_fdata()
    try:
        csdf = csd_fit_model(
            gtab, data,
            mask=mask,
            response=csd_response, sh_order=csd_sh_order,
            lambda_=csd_lambda_, tau=csd_tau)
    except CsdNanResponseError as e:
        raise CsdNanResponseError(
            'Could not compute CSD response function for file: '
            f'{dwi}.') from e

    meta = dict(
        SphericalHarmonicDegree=csd_sh_order,
        ResponseFunctionTensor=csd_response,
        lambda_=csd_lambda_,
        tau=csd_tau)
    meta["SphericalHarmonicBasis"] = "DESCOTEAUX"
    meta["ModelURL"] = f"{DIPY_GH}reconst/csdeconv.py"
    return csdf.shm_coeff, meta


csd_params = pimms.calc("csd_params")(csd)


@pimms.calc("pmap")
@as_file(suffix='_model-CSD_desc-APM_dwi.nii.gz')
def anisotropic_power_map(csd_params):
    """
    full path to a nifti file containing
    the anisotropic power map
    """
    sh_coeff = nib.load(csd_params)
    pmap = shm.anisotropic_power(sh_coeff.get_fdata())
    pmap = nib.Nifti1Image(pmap, sh_coeff.affine)
    return pmap, dict(CSDParamsFile=csd_params)


@pimms.calc("fwdti_fa")
@as_file(suffix='_model-FWDTI_desc-FA_dwi.nii.gz')
@as_fit_deriv('FWDTI')
def fwdti_fa(fwdti_tf):
    """
    full path to a nifti file containing the Free-water DTI fractional
    anisotropy
    """
    return fwdti_tf.fa


@pimms.calc("fwdti_md")
@as_file(suffix='_model-FWDTI_desc-MD_dwi.nii.gz')
@as_fit_deriv('FWDTI')
def fwdti_md(fwdti_tf):
    """
    full path to a nifti file containing the Free-water DTI mean diffusivity
    """
    return fwdti_tf.md


@pimms.calc("fwdti_fwf")
@as_file(suffix='_model-FWDTI_desc-FWF_dwi.nii.gz')
@as_fit_deriv('FWDTI')
def fwdti_fwf(fwdti_tf):
    """
    full path to a nifti file containing the Free-water DTI free water fraction
    """
    return fwdti_tf.f


@pimms.calc("dti_fa")
@as_file(suffix='_model-DTI_desc-FA_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_fa(dti_tf):
    """
    full path to a nifti file containing
    the DTI fractional anisotropy
    """
    return dti_tf.fa


@pimms.calc("dti_lt0", "dti_lt1", "dti_lt2", "dti_lt3", "dti_lt4", "dti_lt5")
def dti_lt(dti_tf, dwi_affine):
    """
    Image of first element in the DTI tensor,
    Image of second element in the DTI tensor,
    Image of third element in the DTI tensor,
    Image of fourth element in the DTI tensor,
    Image of fifth element in the DTI tensor,
    Image of sixth element in the DTI tensor
    """
    dti_lt_dict = {}
    for ii in range(6):
        dti_lt_dict[f"dti_lt{ii}"] = nib.Nifti1Image(
            dti_tf.lower_triangular()[..., ii],
            dwi_affine)
    return dti_lt_dict


@pimms.calc("dti_cfa")
@as_file(suffix='_model-DTI_desc-CFA_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_cfa(dti_tf):
    """
    full path to a nifti file containing
    the DTI color fractional anisotropy
    """
    return dti_tf.color_fa


@pimms.calc("dti_pdd")
@as_file(suffix='_model-DTI_desc-PDD_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_pdd(dti_tf):
    """
    full path to a nifti file containing
    the DTI principal diffusion direction
    """
    pdd = dti_tf.directions.squeeze()
    # Invert the x coordinates:
    pdd[..., 0] = pdd[..., 0] * -1
    return pdd


@pimms.calc("dti_md")
@as_file('_model-DTI_desc-MD_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_md(dti_tf):
    """
    full path to a nifti file containing
    the DTI mean diffusivity
    """
    return dti_tf.md


@pimms.calc("dti_ga")
@as_file(suffix='_model-DTI_desc-GA_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_ga(dti_tf):
    """
    full path to a nifti file containing
    the DTI geodesic anisotropy
    """
    return dti_tf.ga


@pimms.calc("dti_rd")
@as_file(suffix='_model-DTI_desc-RD_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_rd(dti_tf):
    """
    full path to a nifti file containing
    the DTI radial diffusivity
    """
    return dti_tf.rd


@pimms.calc("dti_ad")
@as_file(suffix='_model-DTI_desc-AD_dwi.nii.gz')
@as_fit_deriv('DTI')
def dti_ad(dti_tf):
    """
    full path to a nifti file containing
    the DTI axial diffusivity
    """
    return dti_tf.ad


@pimms.calc(
    "dki_kt0", "dki_kt1", "dki_kt2", "dki_kt3", "dki_kt4",
    "dki_kt5", "dki_kt6", "dki_kt7", "dki_kt8", "dki_kt9",
    "dki_kt10", "dki_kt11", "dki_kt12", "dki_kt13", "dki_kt14")
def dki_kt(dki_tf, dwi_affine):
    """
    Image of first element in the DKI kurtosis model,
    Image of second element in the DKI kurtosis model,
    Image of third element in the DKI kurtosis model,
    Image of fourth element in the DKI kurtosis model,
    Image of fifth element in the DKI kurtosis model,
    Image of sixth element in the DKI kurtosis model,
    Image of seventh element in the DKI kurtosis model,
    Image of eighth element in the DKI kurtosis model,
    Image of ninth element in the DKI kurtosis model,
    Image of tenth element in the DKI kurtosis model,
    Image of eleventh element in the DKI kurtosis model,
    Image of twelf element in the DKI kurtosis model,
    Image of thirteenth element in the DKI kurtosis model,
    Image of fourteenth element in the DKI kurtosis model,
    Image of fifteenth element in the DKI kurtosis model
    """
    dki_kt_dict = {}
    for ii in range(15):
        dki_kt_dict[f"dki_kt{ii}"] = nib.Nifti1Image(
            dki_tf.kt[..., ii],
            dwi_affine)
    return dki_kt_dict


@pimms.calc("dki_lt0", "dki_lt1", "dki_lt2", "dki_lt3", "dki_lt4", "dki_lt5")
def dki_lt(dki_tf, dwi_affine):
    """
    Image of first element in the DTI tensor from DKI,
    Image of second element in the DTI tensor from DKI,
    Image of third element in the DTI tensor from DKI,
    Image of fourth element in the DTI tensor from DKI,
    Image of fifth element in the DTI tensor from DKI,
    Image of sixth element in the DTI tensor from DKI
    """
    dki_lt_dict = {}
    for ii in range(6):
        dki_lt_dict[f"dki_lt{ii}"] = nib.Nifti1Image(
            dki_tf.lower_triangular()[..., ii],
            dwi_affine)
    return dki_lt_dict


@pimms.calc("dki_fa")
@as_file('_model-DKI_desc-FA_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_fa(dki_tf):
    """
    full path to a nifti file containing
    the DKI fractional anisotropy
    """
    return dki_tf.fa


@pimms.calc("dki_md")
@as_file('_model-DKI_desc-MD_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_md(dki_tf):
    """
    full path to a nifti file containing
    the DKI mean diffusivity
    """
    return dki_tf.md


@pimms.calc("dki_awf")
@as_file('_model-DKI_desc-AWF_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_awf(dki_params,
            sphere='repulsion100', gtol=1e-2):
    """
    full path to a nifti file containing
    the DKI axonal water fraction

    Parameters
    ----------
    sphere : Sphere class instance, optional
        The sphere providing sample directions for the initial
        search of the maximal value of kurtosis.
        Default: 'repulsion100'
    gtol : float, optional
        This input is to refine kurtosis maxima under the precision of
        the directions sampled on the sphere class instance.
        The gradient of the convergence procedure must be less than gtol
        before successful termination.
        If gtol is None, fiber direction is directly taken from the initial
        sampled directions of the given sphere object.
        Default: 1e-2
    """
    dki_params = nib.load(dki_params).get_fdata()
    return axonal_water_fraction(dki_params, sphere=sphere, gtol=gtol)


@pimms.calc("dki_mk")
@as_file('_model-DKI_desc-MK_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_mk(dki_tf):
    """
    full path to a nifti file containing
    the DKI mean kurtosis file
    """
    return dki_tf.mk()


@pimms.calc("dki_ga")
@as_file(suffix='_model-DKI_desc-GA_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_ga(dki_tf):
    """
    full path to a nifti file containing
    the DKI geodesic anisotropy
    """
    return dki_tf.ga


@pimms.calc("dki_rd")
@as_file(suffix='_model-DKI_desc-RD_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_rd(dki_tf):
    """
    full path to a nifti file containing
    the DKI radial diffusivity
    """
    return dki_tf.rd


@pimms.calc("dki_ad")
@as_file(suffix='_model-DKI_desc-AD_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_ad(dki_tf):
    """
    full path to a nifti file containing
    the DKI axial diffusivity
    """
    return dki_tf.ad


@pimms.calc("dki_rk")
@as_file(suffix='_model-DKI_desc-RK_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_rk(dki_tf):
    """
    full path to a nifti file containing
    the DKI radial kurtosis
    """
    return dki_tf.rk


@pimms.calc("dki_ak")
@as_file(suffix='_model-DKI_desc-AK_dwi.nii.gz')
@as_fit_deriv('DKI')
def dki_ak(dki_tf):
    """
    full path to a nifti file containing
    the DKI axial kurtosis file
    """
    return dki_tf.ak


@pimms.calc("brain_mask")
@as_file('_desc-brain_mask.nii.gz')
def brain_mask(base_fname, dwi, b0,
               bids_info, brain_mask_definition=None):
    """
    full path to a nifti file containing
    the brain mask

    Parameters
    ----------
    brain_mask_definition : instance from `AFQ.definitions.image`, optional
        This will be used to create
        the brain mask, which gets applied before registration to a
        template.
        If you want no brain mask to be applied, use FullImage.
        If None, use B0Image()
        Default: None
    """
    if brain_mask_definition is None:
        brain_mask_definition = B0Image()
    if not isinstance(brain_mask_definition, Definition):
        raise TypeError(
            "brain_mask_definition must be a Definition")
    if bids_info is not None:
        brain_mask_definition.find_path(
            bids_info["bids_layout"],
            dwi,
            bids_info["subject"],
            bids_info["session"])
    return brain_mask_definition.get_image_direct(
        dwi, bids_info, b0, data_imap=None)


@pimms.calc("bundle_dict", "reg_template")
def get_bundle_dict(base_fname, dwi, segmentation_params,
                    brain_mask, bids_info, b0,
                    bundle_info=None, reg_template_spec="mni_T1"):
    """
    Dictionary defining the different bundles to be segmented,
    and a Nifti1Image containing the template for registration

    Parameters
    ----------
    bundle_info : list of strings, dict, or BundleDict, optional
        List of bundle names to include in segmentation,
        or a bundle dictionary (see BundleDict for inspiration),
        or a BundleDict. See `Defining Custom Bundle Dictionaries`
        in the `usage` section of pyAFQ's documentation for details.
        If None, will get all appropriate bundles for the chosen
        segmentation algorithm.
        Default: None
    reg_template_spec : str, or Nifti1Image, optional
        The target image data for registration.
        Can either be a Nifti1Image, a path to a Nifti1Image, or
        if "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1",
        image data will be loaded automatically.
        If "hcp_atlas" is used, slr registration will be used
        and reg_subject should be "subject_sls".
        Default: "mni_T1"
    """
    if not isinstance(reg_template_spec, str)\
            and not isinstance(reg_template_spec, nib.Nifti1Image):
        raise TypeError(
            "reg_template must be a str or Nifti1Image")

    if bundle_info is not None and not ((
            isinstance(bundle_info, list)
            and isinstance(bundle_info[0], str)) or (
                isinstance(bundle_info, dict)) or (
                    isinstance(bundle_info, abd.BundleDict))):
        raise TypeError((
            "bundle_info must be a list of strings,"
            " a dict, or a BundleDict"))

    if bundle_info is None:
        if segmentation_params["seg_algo"] == "reco" or\
                segmentation_params["seg_algo"] == "reco16":
            bundle_info = abd.RECO_BUNDLES_16
        elif segmentation_params["seg_algo"] == "reco80":
            bundle_info = abd.RECO_BUNDLES_80
        else:
            bundle_info = abd.BUNDLES

    use_brain_mask = True
    brain_mask = nib.load(brain_mask).get_fdata()
    if np.all(brain_mask == 1.0):
        use_brain_mask = False
    if isinstance(reg_template_spec, nib.Nifti1Image):
        reg_template = reg_template_spec
    else:
        img_l = reg_template_spec.lower()
        if img_l == "mni_t2":
            reg_template = afd.read_mni_template(
                mask=use_brain_mask, weight="T2w")
        elif img_l == "mni_t1":
            reg_template = afd.read_mni_template(
                mask=use_brain_mask, weight="T1w")
        elif img_l == "dti_fa_template":
            reg_template = afd.read_ukbb_fa_template(mask=use_brain_mask)
        elif img_l == "hcp_atlas":
            reg_template = afd.read_mni_template(mask=use_brain_mask)
        elif img_l == "pediatric":
            reg_template = afd.read_pediatric_templates()[
                "UNCNeo-withCerebellum-for-babyAFQ"]
        else:
            reg_template = nib.load(reg_template_spec)

    if isinstance(bundle_info, abd.BundleDict):
        bundle_dict = bundle_info
    else:
        bundle_dict = abd.BundleDict(
            bundle_info,
            seg_algo=segmentation_params["seg_algo"],
            resample_to=reg_template)

    def roi_scalar_to_info(roi):
        if not isinstance(roi, ImageDefinition):
            return roi
        if bids_info is not None:
            roi.find_path(
                bids_info["bids_layout"],
                dwi,
                bids_info["subject"],
                bids_info["session"])
        roi_img, _ = roi.get_image_direct(
            dwi, bids_info, b0, data_imap=None)
        return roi_img
    for b_name, b_info in bundle_dict._dict.items():
        if "space" in b_info and b_info["space"] == "subject":
            bundle_dict.apply_to_rois(b_name, roi_scalar_to_info)
            bundle_dict._resample_roi(b_name)
    return bundle_dict, reg_template


def get_data_plan(kwargs):
    if "scalars" in kwargs and not (
        isinstance(kwargs["scalars"], list) and isinstance(
            kwargs["scalars"][0], (str, Definition))):
        raise TypeError(
            "scalars must be a list of "
            "strings/scalar definitions")

    data_tasks = with_name([
        get_data_gtab, b0, b0_mask, brain_mask,
        dti_fit, dki_fit, fwdti_fit, anisotropic_power_map,
        dti_fa, dti_lt, dti_cfa, dti_pdd, dti_md, dki_kt, dki_lt, dki_fa,
        fwdti_fa, fwdti_md, fwdti_fwf,
        dki_md, dki_awf, dki_mk, dti_ga, dti_rd, dti_ad, dki_ga, dki_rd,
        dki_ad, dki_rk, dki_ak, dti_params, dki_params, fwdti_params,
        csd_params, get_bundle_dict])

    if "scalars" not in kwargs:
        kwargs["scalars"] = ["dti_fa", "dti_md"]
    else:
        scalars = []
        for scalar in kwargs["scalars"]:
            if isinstance(scalar, str):
                scalars.append(scalar.lower())
            else:
                scalars.append(scalar)
        kwargs["scalars"] = scalars
    return pimms.plan(**data_tasks)
