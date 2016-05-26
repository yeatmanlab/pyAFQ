import numpy as np
import scipy.ndimage as ndim

import nibabel as nib

import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.data as afd

# Set the default set as a constant:
AFQ_BUNDLES = ["ATR", "CGC", "CST", "FA", "FP", "HCC", "IFO", "ILF",
               "SLF", "ARC", "UNC"]


def patch_up_roi(roi):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed
    """
    return ndim.binary_fill_holes(ndim.binary_dilation(roi).astype(int))

def _register_to_template(fdata, gtab):
    MNI_T2 = dpd.read_mni_template()
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()
    dwi = nib.load(fdata)
    dwi_affine = dwi.get_affine()
    dwi_data = dwi.get_data()
    mean_b0 = np.mean(dwi_data[..., gtab.b0s_mask], -1)
    warped_b0, mapping = reg.syn_registration(mean_b0, MNI_T2_data,
                                              moving_affine=dwi_affine,
                                              static_affine=MNI_T2_affine,
                                              step_length=0.1,
                                              sigma_diff=2.0,
                                              metric='CC',
                                              dim=3,
                                              level_iters=[10, 10, 5],
                                              prealign=None)
    return mapping

def segment(fdata, fbval, fbvec, streamlines, bundles=AFQ_BUNDLES,
            mapping=None):
    img, data, gtab, mask = ut.prepare_data(fdata, fbval, fbvec)
    xform_sl = [s for s in dtu.move_streamlines(streamlines,
                                                np.linalg.inv(img.affine))]
    if mapping is None:
        mapping = _register_to_template(fdata, gtab)

    afq_templates = afd.read_templates()
    # For the arcuate, we need to rename a few of these and duplicate SLF ROI:
    afq_templates['ARC_roi1_L'] = afq_templates['SLF_roi1_L']
    afq_templates['ARC_roi1_R'] = afq_templates['SLF_roi1_R']
    afq_templates['ARC_roi2_L'] = afq_templates['SLFt_roi2_L']
    afq_templates['ARC_roi2_R'] = afq_templates['SLFt_roi2_R']
    fiber_groups = {}
    for hemi in ["R", "L"]:
        for bundle in bundles:
            ROIs = [bundle + "_roi%s_"%(i+1) + hemi for i in range(2)]
            select_sl = xform_sl
            for ROI in ROIs:
                data = afq_templates[ROI].get_data()
                warped_ROI = patch_up_roi(mapping.transform_inverse(data,
                                                interpolation='nearest'))

                select_sl = dts.select_by_rois(select_sl,
                                               [warped_ROI.astype(bool)],
                                               [True])
            fiber_groups[bundle + "_" + hemi] = select_sl

    return fiber_groups
