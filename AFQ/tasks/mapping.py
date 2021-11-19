import nibabel as nib
import os.path as op
import os
import numpy as np
import logging

import pimms
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_fname, with_name
import AFQ.data as afd
import AFQ.utils.volume as auv
from AFQ.definitions.mapping import SynMap
from AFQ.definitions.utils import Definition
import AFQ.api.bundle_dict as abd

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space


logger = logging.getLogger('AFQ.api.mapping')


@pimms.calc("b0_warped_file")
@as_file('_b0_in_MNI.nii.gz')
def export_registered_b0(subses_dict, data_imap, mapping, reg_template):
    """
    full path to a nifti file containing
    b0 transformed to template space
    """
    mean_b0 = nib.load(data_imap["b0_file"]).get_fdata()
    warped_b0 = mapping.transform(mean_b0)
    warped_b0 = nib.Nifti1Image(warped_b0, reg_template.affine)
    return warped_b0, dict(b0InSubject=data_imap["b0_file"])


@pimms.calc("template_xform_file")
@as_file('_template_xform.nii.gz')
def template_xform(subses_dict, dwi_affine, mapping, reg_template):
    """
    full path to a nifti file containing
    registration template transformed to subject space
    """
    template_xform = mapping.transform_inverse(reg_template.get_fdata())
    template_xform = nib.Nifti1Image(template_xform, dwi_affine)
    return template_xform, dict()


@pimms.calc("rois_file")
def export_rois(subses_dict, bundle_dict, mapping, dwi_affine):
    """
    dictionary of full paths to Nifti1Image files of ROIs
    transformed to subject space
    """
    rois_dir = op.join(subses_dict['results_dir'], 'ROIs')
    os.makedirs(rois_dir, exist_ok=True)
    roi_files = {}
    for bundle in bundle_dict:
        roi_files[bundle] = []
        for ii, roi in enumerate(bundle_dict[bundle]['ROIs']):
            if bundle_dict[bundle]['rules'][ii]:
                inclusion = 'include'
            else:
                inclusion = 'exclude'

            fname = op.split(
                get_fname(
                    subses_dict,
                    f'_desc-ROI-{bundle}-{ii + 1}-{inclusion}.nii.gz'))

            fname = op.join(rois_dir, fname[1])
            if not op.exists(fname):
                warped_roi = auv.transform_inverse_roi(
                    roi,
                    mapping,
                    bundle_name=bundle)

                # Cast to float32, so that it can be read in by MI-Brain:
                logger.info(f"Saving {fname}")
                nib.save(
                    nib.Nifti1Image(
                        warped_roi.astype(np.float32),
                        dwi_affine), fname)
                meta = dict()
                meta_fname = fname.split('.')[0] + '.json'
                afd.write_json(meta_fname, meta)
            roi_files[bundle].append(fname)
    return {'rois_file': roi_files}


@pimms.calc("mapping")
def mapping(subses_dict, reg_subject, reg_template, bids_info,
            mapping_definition=SynMap()):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        Default: SynMap()
    """
    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    if bids_info is not None:
        mapping_definition.find_path(
            bids_info["bids_layout"],
            subses_dict["dwi_file"],
            bids_info["subject"],
            bids_info["session"])
    return mapping_definition.get_for_subses(
        subses_dict, reg_subject, reg_template)


@pimms.calc("mapping")
def sls_mapping(subses_dict, reg_subject, reg_template, bids_info,
                tractography_imap, mapping_definition=SynMap()):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        Default: SynMap()
    """
    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    if bids_info is not None:
        mapping_definition.find_path(
            bids_info["bids_layout"],
            subses_dict["dwi_file"],
            bids_info["subject"],
            bids_info["session"])
    streamlines_file = tractography_imap["streamlines_file"]
    tg = load_tractogram(
        streamlines_file, reg_subject,
        Space.VOX, bbox_valid_check=False)
    tg.to_rasmm()

    atlas_fname = op.join(
        afd.afq_home,
        'hcp_atlas_16_bundles',
        'Atlas_in_MNI_Space_16_bundles',
        'whole_brain',
        'whole_brain_MNI.trk')
    if not op.exists(atlas_fname):
        afd.fetch_hcp_atlas_16_bundles()
    hcp_atlas = load_tractogram(
        atlas_fname,
        'same', bbox_valid_check=False)
    return mapping_definition.get_for_subses(
        subses_dict, reg_subject, reg_template,
        subject_sls=tg.streamlines,
        template_sls=hcp_atlas.streamlines)


@pimms.calc("reg_subject")
def get_reg_subject(data_imap, bids_info, subses_dict,
                    reg_template,
                    reg_subject_spec="power_map"):
    """
    Nifti1Image which represents this subject
    when registering the subject to the template

    Parameters
    ----------
    reg_subject : str, Nifti1Image, dict, optional
        The source image data to be registered.
        Can either be a Nifti1Image, a scalar definition, or
        if "b0", "dti_fa_subject", "subject_sls", or "power_map,"
        image data will be loaded automatically.
        If "subject_sls" is used, slr registration will be used
        and reg_template should be "hcp_atlas".
        Default: "power_map"
    """
    if not isinstance(reg_subject_spec, str)\
            and not isinstance(reg_subject_spec, nib.Nifti1Image)\
            and not isinstance(reg_subject_spec, dict):
        raise TypeError(
            "reg_subject must be a str, dict, or Nifti1Image")

    filename_dict = {
        "b0": data_imap["b0_file"],
        "power_map": data_imap["pmap_file"],
        "dti_fa_subject": data_imap["dti_fa_file"],
        "subject_sls": data_imap["b0_file"],
    }
    bm = nib.load(data_imap["brain_mask_file"])

    if bids_info is not None and isinstance(reg_subject_spec, Definition):
        reg_subject_spec.find_path(
            bids_info["bids_layout"],
            subses_dict["dwi_file"],
            bids_info["subject"],
            bids_info["session"])
        scalar_data, _ = reg_subject_spec.get_data(
            subses_dict, subses_dict["dwi_affine"], reg_template, None)
        img = nib.Nifti1Image(scalar_data, bm.affine)
    else:
        if reg_subject_spec in filename_dict:
            reg_subject_spec = filename_dict[reg_subject_spec]
        img = nib.load(reg_subject_spec)
    bm = bm.get_fdata().astype(bool)
    masked_data = img.get_fdata()
    masked_data[~bm] = 0
    img = nib.Nifti1Image(masked_data, img.affine)
    return img


@pimms.calc("bundle_dict", "reg_template")
def get_bundle_dict(segmentation_params, data_imap, bundle_info=None,
                    reg_template_spec="mni_T1"):
    """
    Dictionary defining the different bundles to be segmented,
    and a Nifti1Image containing the template for registration

    Parameters
    ----------
    bundle_info : list of strings, dict, or BundleDict, optional
        List of bundle names to include in segmentation,
        or a bundle dictionary (see BundleDict for inspiration),
        or a BundleDict.
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
            "bundle_info must a list of strings,"
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
    brain_mask = nib.load(data_imap["brain_mask_file"]).get_fdata()
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
        else:
            reg_template = nib.load(reg_template_spec)

    if isinstance(bundle_info, abd.BundleDict):
        bundle_dict = bundle_info
    else:
        bundle_dict = abd.BundleDict(
            bundle_info,
            seg_algo=segmentation_params["seg_algo"],
            resample_to=reg_template)

    return bundle_dict, reg_template


def get_mapping_plan(kwargs, use_sls=False):
    mapping_tasks = with_name([
        export_registered_b0, template_xform, export_rois, mapping,
        get_reg_subject, get_bundle_dict])

    bids_info = kwargs["bids_info"]
    # add custom scalars
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, Definition):
            scalar.find_path(
                bids_info["bids_layout"],
                kwargs["subses_dict"]["dwi_file"],
                bids_info["subject"],
                bids_info["session"]
            )
            mapping_tasks["{scalar.name}_file_res"] =\
                pimms.calc(f"{scalar.name}_file")(scalar.get_for_subses())

    if use_sls:
        mapping_tasks["mapping_res"] = sls_mapping

    return pimms.plan(**mapping_tasks)
