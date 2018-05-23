import os.path as op
import pandas as pd
import numpy as np
import nibabel as nib

import AFQ.utils.streamlines as aus
import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg
import dipy.data as dpd

import dipy.tracking.streamline as dts
from dipy.segment.mask import median_otsu
import dipy.core.gradients as dpg


input_folder = '/Users/arokem/tmp/input/'

fdata = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.nii.gz")
fbval = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.bvals")
fbvec = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.bvecs")

img = nib.load(fdata)
data = img.get_data()
gtab = dpg.gradient_table(fbval, fbvec)

mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)

print("Calculating brain-mask")
if not op.exists('./brain_mask.nii.gz'):
    _, brain_mask = median_otsu(mean_b0, median_radius=4, numpass=4)
    nib.save(nib.Nifti1Image(brain_mask.astype(int),
                             img.affine), './brain_mask.nii.gz')
else:
    brain_mas = nib.load('./brain_mask.nii.gz').get_data().astype(bool)

print("Calculating DTI...")
if not op.exists('./dti_FA.nii.gz'):
    dti_params = dti.fit_dti(fdata, fbval, fbvec,
                             out_dir='.', mask=brain_mask)
else:
    dti_params = {'FA': './dti_FA.nii.gz',
                  'MD': './dti_MD.nii.gz',
                  'RD': './dti_RD.nii.gz',
                  'AD': './dti_AD.nii.gz',
                  'params': './dti_params.nii.gz'}

print("Registering to template...")
MNI_T2_img = dpd.read_mni_template()
if not op.exists('mapping.nii.gz'):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(fbval, fbvec)
    mapping = reg.syn_register_dwi(fdata, gtab)
    reg.write_mapping(mapping, './mapping.nii.gz')
else:
    mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)

print("Tracking...")
if not op.exists('dti_streamlines.trk'):
    FA = nib.load(dti_params["FA"]).get_data()
    wm_mask = np.zeros_like(FA)
    wm_mask[FA > 0.2] = 1
    step_size = 1
    min_length_mm = 50
    streamlines = dts.Streamlines(
        aft.track(dti_params['params'],
                  directions="det",
                  seed_mask=wm_mask,
                  seeds=2,
                  stop_mask=FA,
                  stop_threshold=0.2,
                  step_size=step_size,
                  min_length=min_length_mm / step_size))
    aus.write_trk('./dti_streamlines.trk', streamlines, affine=img.affine)
else:
    tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
    streamlines = tg.apply_affine(np.linalg.inv(img.affine)).streamlines

print("We're looking at: %s streamlines" % len(streamlines))

templates = afd.read_templates()
templates['ARC_roi1_L'] = templates['SLF_roi1_L']
templates['ARC_roi1_R'] = templates['SLF_roi1_R']
templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
templates['ARC_roi2_R'] = templates['SLFt_roi2_R']


bundle_names = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC"]

bundles = {}
for name in bundle_names:
    for hemi in ['_R', '_L']:
        bundles[name + hemi] = {
            'ROIs': [templates[name + '_roi1' + hemi],
                     templates[name + '_roi2' + hemi]],
            'rules': [True, True],
            'prob_map': templates[name + hemi + '_prob_map'],
            'cross_midline': False}


bundles["FP"] = {'ROIs': [templates["FP_L"],
                          templates["FP_R"]],
                 'rules': [True, True],
                 'prob_map': templates['FP_prob_map'],
                 'cross_midline': True}
bundles["FA"] = {'ROIs': [templates["FA_L"],
                          templates["FA_R"]],
                 'rules': [True, True],
                 'prob_map': templates['FA_prob_map'],
                 'cross_midline': True}

print("Segmenting fiber groups...")
fiber_groups = seg.segment(fdata,
                           fbval,
                           fbvec,
                           streamlines,
                           bundles,
                           reg_template=MNI_T2_img,
                           mapping=mapping,
                           affine=img.affine,
                           clean_threshold=6,
                           prob_threshold=5)


print("Getting tract profiles")
n_points = 100

dfs = []
for bundle in fiber_groups:
    print("Getting profile for: %s" % bundle)
    if len(fiber_groups[bundle]) > 0:
        bundle_df = pd.DataFrame(data={
            'tractID': [bundle] * n_points,
            'nodeID': np.arange(1, n_points + 1)})

        for stat_key in dti_params.keys():
            if stat_key == 'params':
                pass
            else:
                stat_data = nib.load(dti_params[stat_key]).get_data()
                fgarray = seg._resample_bundle(fiber_groups[bundle], n_points)
                weights = seg.gaussian_weights(fgarray)
                profile = seg.calculate_tract_profile(stat_data,
                                                      fgarray,
                                                      weights=weights)
                bundle_df[stat_key] = profile
        dfs.append(bundle_df)
    else:
        print("There are no fibers in %s" % bundle)

result = pd.concat(dfs)
