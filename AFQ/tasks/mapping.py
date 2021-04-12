import nibabel as nib
import os.path as op
import os
import numpy as np

from pydra import mark
from AFQ.tasks.utils import *
import AFQ.data as afd
import AFQ.utils.volume as auv


@mark.task
@mark.annotate(
    {"return": {"b0_warped_file": str}}
)
@as_file('_b0_in_MNI.nii.gz')
def export_registered_b0(subses_tuple, b0_file, mapping, reg_template):
    mean_b0 = nib.load(b0_file).get_fdata()
    warped_b0 = mapping.transform(mean_b0)
    warped_b0 = nib.Nifti1Image(warped_b0, reg_template.affine)
    return warped_b0, dict(b0InSubject=b0_file)


@mark.task
@mark.annotate(
    {"return": {"template_xform_file": str}}
)
@as_file('_template_xform.nii.gz')
def template_xform(subses_tuple, dwi_affine, mapping, reg_template):
    template_xform = mapping.transform_inverse(reg_template.get_fdata())
    template_xform = nib.Nifti1Image(template_xform, dwi_affine)
    return template_xform, dict()


@mark.task
@mark.annotate(
    {"return": {"roi_files": list}}
)
def export_rois(subses_tuple, bundle_dict, mapping, dwi_affine):
    rois_dir = op.join(subses_tuple['results_dir'], 'ROIs')
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
                    subses_tuple,
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
    return roi_files


@mark.task
@mark.annotate(
    {"return": {"mapping": object}}
)
def mapping(subses_tuple, mapping_definition):
    return mapping_definition.get_for_subses(subses_tuple)
