import nibabel as nib
import os.path as op
import os
import numpy as np
import logging

import pimms
from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import with_name, str_to_desc
import AFQ.data.fetch as afd
from AFQ.utils.path import drop_extension, write_json
from AFQ.definitions.mapping import SynMap
from AFQ.definitions.utils import Definition
from AFQ.definitions.image import ImageDefinition

from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space


logger = logging.getLogger('AFQ')


@pimms.calc("b0_warped")
@as_file('_space-template_desc-b0_dwi.nii.gz')
def export_registered_b0(data_imap, mapping):
    """
    full path to a nifti file containing
    b0 transformed to template space
    """
    mean_b0 = nib.load(data_imap["b0"]).get_fdata()
    warped_b0 = mapping.transform(mean_b0)
    warped_b0 = nib.Nifti1Image(warped_b0, data_imap["reg_template"].affine)
    return warped_b0, dict(b0InSubject=data_imap["b0"])


@pimms.calc("template_xform")
@as_file('_space-subject_desc-template_dwi.nii.gz')
def template_xform(mapping, data_imap):
    """
    full path to a nifti file containing
    registration template transformed to subject space
    """
    template_xform = mapping.transform_inverse(
        data_imap["reg_template"].get_fdata())
    template_xform = nib.Nifti1Image(template_xform, data_imap["dwi_affine"])
    return template_xform, dict()


@pimms.calc("rois")
def export_rois(base_fname, results_dir, data_imap, mapping):
    """
    dictionary of full paths to Nifti1Image files of ROIs
    transformed to subject space
    """
    bundle_dict = data_imap["bundle_dict"]
    rois_dir = op.join(results_dir, 'ROIs')
    os.makedirs(rois_dir, exist_ok=True)
    roi_files = {}
    base_roi_fname = op.join(rois_dir, op.split(base_fname)[1])
    for bundle_name in bundle_dict:
        roi_files[bundle_name] = []
        for roi_fname in bundle_dict.transform_rois(
                bundle_name, mapping, data_imap["dwi_affine"],
                base_fname=base_roi_fname):
            logger.info(f"Saving {roi_fname}")
            roi_files[bundle_name].append(roi_fname)
            meta = {}
            meta_fname = f'{drop_extension(roi_fname)}.json'
            write_json(meta_fname, meta)
    return {'rois': roi_files}


@pimms.calc("mapping")
def mapping(base_fname, dwi_path, reg_subject, data_imap, bids_info,
            mapping_definition=None):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        If None, use SynMap()
        Default: None
    """
    reg_template = data_imap["reg_template"]
    if mapping_definition is None:
        mapping_definition = SynMap()
    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    if bids_info is not None:
        mapping_definition.find_path(
            bids_info["bids_layout"],
            dwi_path,
            bids_info["subject"],
            bids_info["session"])
    return mapping_definition.get_for_subses(
        base_fname, data_imap["dwi"], bids_info,
        reg_subject, reg_template)


@pimms.calc("mapping")
def sls_mapping(base_fname, dwi_path, reg_subject, data_imap, bids_info,
                tractography_imap, mapping_definition=None):
    """
    mapping from subject to template space.

    Parameters
    ----------
    mapping_definition : instance of `AFQ.definitions.mapping`, optional
        This defines how to either create a mapping from
        each subject space to template space or load a mapping from
        another software. If creating a map, will register reg_subject and
        reg_template.
        If None, use SynMap()
        Default: None
    """
    reg_template = data_imap["reg_template"]
    if mapping_definition is None:
        mapping_definition = SynMap()
    if not isinstance(mapping_definition, Definition):
        raise TypeError(
            "mapping must be a mapping defined"
            + " in `AFQ.definitions.mapping`")
    if bids_info is not None:
        mapping_definition.find_path(
            bids_info["bids_layout"],
            dwi_path,
            bids_info["subject"],
            bids_info["session"])
    streamlines_file = tractography_imap["streamlines"]
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
        base_fname, data_imap["dwi"], bids_info,
        reg_subject, reg_template,
        subject_sls=tg.streamlines,
        template_sls=hcp_atlas.streamlines)


@pimms.calc("reg_subject")
def get_reg_subject(data_imap,
                    reg_subject_spec="power_map"):
    """
    Nifti1Image which represents this subject
    when registering the subject to the template

    Parameters
    ----------
    reg_subject_spec : str, instance of `AFQ.definitions.ImageDefinition`, optional  # noqa
        The source image data to be registered.
        Can either be a Nifti1Image, an ImageFile, or str.
        if "b0", "dti_fa_subject", "subject_sls", or "power_map,"
        image data will be loaded automatically.
        If "subject_sls" is used, slr registration will be used
        and reg_template should be "hcp_atlas".
        Default: "power_map"
    """
    if not isinstance(reg_subject_spec, str)\
            and not isinstance(reg_subject_spec, nib.Nifti1Image):
        # Note the ImageDefinition case is handled in get_mapping_plan
        raise TypeError(
            "reg_subject must be a str, ImageDefinition, or Nifti1Image")

    filename_dict = {
        "b0": "b0",
        "power_map": "csd_pmap",
        "dti_fa_subject": "dti_fa",
        "subject_sls": "b0",
    }
    bm = nib.load(data_imap["brain_mask"])

    if reg_subject_spec in filename_dict:
        reg_subject_spec = data_imap[filename_dict[reg_subject_spec]]
    if isinstance(reg_subject_spec, str):
        img = nib.load(reg_subject_spec)
    bm = bm.get_fdata().astype(bool)
    masked_data = img.get_fdata()
    masked_data[~bm] = 0
    img = nib.Nifti1Image(masked_data, img.affine)
    return img


def get_mapping_plan(kwargs, use_sls=False):
    mapping_tasks = with_name([
        export_registered_b0, template_xform, export_rois, mapping,
        get_reg_subject])

    bids_info = kwargs.get("bids_info", None)
    # add custom scalars
    for scalar in kwargs["scalars"]:
        if isinstance(scalar, Definition):
            if bids_info is None:
                scalar.find_path(
                    None,
                    kwargs["dwi_path"],
                    None,
                    None)
                scalar_found = True
            else:
                scalar_found = scalar.find_path(
                    bids_info["bids_layout"],
                    kwargs["dwi_path"],
                    bids_info["subject"],
                    bids_info["session"],
                    required=False)
            if scalar_found != False:
                mapping_tasks[f"{scalar.get_name()}_res"] =\
                    pimms.calc(f"{scalar.get_name()}")(
                        as_file((
                            f'_desc-{str_to_desc(scalar.get_name())}'
                            '_dwi.nii.gz'))(
                                scalar.get_image_getter("mapping")))

    if use_sls:
        mapping_tasks["mapping_res"] = sls_mapping

    reg_ss = kwargs.get("reg_subject_spec", None)
    if isinstance(reg_ss, ImageDefinition):
        reg_ss.find_path(
            bids_info["bids_layout"],
            kwargs["dwi_path"],
            bids_info["subject"],
            bids_info["session"])
        del kwargs["reg_subject_spec"]
        mapping_tasks["reg_subject_spec_res"] = pimms.calc("reg_subject_spec")(
            as_file((
                f'_desc-{str_to_desc(reg_ss.get_name())}'
                '_dwi.nii.gz'))(reg_ss.get_image_getter("mapping")))

    return pimms.plan(**mapping_tasks)
