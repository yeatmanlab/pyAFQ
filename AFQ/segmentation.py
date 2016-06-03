import numpy as np
import scipy.ndimage as ndim

import nibabel as nib

import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.data as afd

# Set the default set as a module-wide constant:
AFQ_BUNDLES = ["ATR", "CGC", "CST",
                #"FA", "FP",
                "HCC", "IFO", "ILF",
               "SLF", "ARC", "UNC"]

# Read in the standard templates:
AFQ_TEMPLATES = afd.read_templates()
# For the arcuate, we need to rename a few of these and duplicate the SLF ROI:
AFQ_TEMPLATES['ARC_roi1_L'] = AFQ_TEMPLATES['SLF_roi1_L']
AFQ_TEMPLATES['ARC_roi1_R'] = AFQ_TEMPLATES['SLF_roi1_R']
AFQ_TEMPLATES['ARC_roi2_L'] = AFQ_TEMPLATES['SLFt_roi2_L']
AFQ_TEMPLATES['ARC_roi2_R'] = AFQ_TEMPLATES['SLFt_roi2_R']


def patch_up_roi(roi):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed

    Returns
    -------
    ROI after dilation and hole-filling
    """
    return ndim.binary_fill_holes(ndim.binary_dilation(roi).astype(int))


def segment(fdata, fbval, fbvec, streamlines, bundles=AFQ_BUNDLES,
            templates=AFQ_TEMPLATES, reg_template=None, mapping=None,
            as_generator=True, **reg_kwargs):
    """

    generate : bool
        Whether to generate the streamlines here, or return generators.

    reg_template : template to use for registration (defaults to the MNI T2)
    """
    img, data, gtab, mask = ut.prepare_data(fdata, fbval, fbvec)
    xform_sl = [s for s in dtu.move_streamlines(streamlines,
                                                np.linalg.inv(img.affine))]

    if reg_template is None:
        reg_template = dpd.read_mni_template()

    if mapping is None:
        mapping = reg.syn_register_dwi(fdata, gtab, template=reg_template,
                                       **reg_kwargs)

    if isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
        mapping = reg.read_mapping(mapping, img, reg_template)

    fiber_groups = {}
    for hemi in ["R", "L"]:
        for bundle in bundles:
            ROIs = [bundle + "_roi%s_"%(i+1) + hemi for i in range(2)]
            select_sl = xform_sl
            for ROI in ROIs:
                data = templates[ROI].get_data()
                warped_ROI = patch_up_roi(mapping.transform_inverse(data,
                                                interpolation='nearest'))

                select_sl = dts.select_by_rois(select_sl,
                                               [warped_ROI.astype(bool)],
                                               [True])
            if as_generator:
                fiber_groups[bundle + "_" + hemi] = select_sl
            else:
                fiber_groups[bundle + "_" + hemi] = list(select_sl)

    return fiber_groups
