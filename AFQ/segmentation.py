import scipy.ndimage as ndim

import nibabel as nib

import dipy.data as dpd

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.data as afd

# These are the "include" and "exclude" rules we use per default
seg_rules = {"callosum_occipital":
             {"include": ["L_Occipital", "R_Occipital", "Callosum_midsag"],
              "exclude": []}}


def segment(fdata, fbval, fbvec, tracks, rules=seg_rules):
    img, data, gtab, mask = ut.prepare_data(data_files,
                                            bval_files,
                                            bvec_files)
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

    afq_templates = afd.read_templates()

    for bundle in seg_rules:
        include_rois = afq_templates[seg_rules[bundle]["include"]]
        exclude_rois = afq_templates[eg_rules[bundle]["exclude"]]
        warped_includes = []
        for roi in include_rois:
            roi_data = nib.load(roi).get_data()
            warped_includes.append(
                mapping.transform_inverse(
                    ndim.binary_dilation(roi_data).astype(int),
                    interpolation='nearest'))
