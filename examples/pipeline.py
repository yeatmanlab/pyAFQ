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
import AFQ.api as api
import dipy.data as dpd



input_folder = '/Users/arokem/tmp/input/'

hardi_fdata = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.nii.gz")
hardi_fbval = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.bvals")
hardi_fbvec = op.join(input_folder, "Ax DTI 30 DIRECTIONAL_aligned_trilin.bvecs")

img = nib.load(hardi_fdata)

print("Calculating brain mask...")


print("Calculating DTI...")
if not op.exists('./dti_FA.nii.gz'):
    dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir='.')
else:
    dti_params = {'FA': './dti_FA.nii.gz',
                  'MD': './dti_MD.nii.gz',
                  'RD': './dti_RD.nii.gz',
                  'AD': './dti_AD.nii.gz',
                  'params': './dti_params.nii.gz'}

print("Tracking...")
if not op.exists('dti_streamlines.trk'):
    FA = nib.load(dti_params["FA"]).get_data()
    step_size = 1
    min_length_mm = 50
    streamlines = dts.Streamlines(aft.track(dti_params['params'], seeds=2, stop_mask=FA,
                                           stop_threshold=0.2, step_size=step_size,
                                           min_length=min_length_mm/step_size))
    aus.write_trk('./dti_streamlines.trk', streamlines, affine=img.affine)
else:
    tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
    streamlines = tg.apply_affine(np.linalg.inv(img.affine)).streamlines

# Use only a small portion of the streamlines, for expedience:
streamlines = streamlines[::10]

print("Splitting into tracks that cross midline, and those that don't")
midsaggital_roi = np.zeros(img.shape[:3])
midsaggital_roi[midsaggital_roi.shape[0]//2, :, :] = 1

streamlines_midsag = seg.select_streamlines(streamlines, midsaggital_roi, include=True)
streamlines_no_midsag = seg.select_streamlines(streamlines, midsaggital_roi, include=False)

templates = afd.read_templates()
templates['ARC_roi1_L'] = templates['SLF_roi1_L']
templates['ARC_roi1_R'] = templates['SLF_roi1_R']
templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
templates['ARC_roi2_R'] = templates['SLFt_roi2_R']


bundle_names = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC"]

bundles = {}
for name in bundle_names:
    for hemi in ['_R', '_L']:
        bundles[name + hemi] = {'ROIs': [templates[name + '_roi1' + hemi],
                                         templates[name + '_roi2' + hemi]],
                                'rules': [True, True]}


print("Registering to template...")
MNI_T2_img = dpd.read_mni_template()
if not op.exists('mapping.nii.gz'):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    mapping = reg.syn_register_dwi(hardi_fdata, gtab)
    reg.write_mapping(mapping, './mapping.nii.gz')
else:
    mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)

print("Segmenting fiber groups...")
fiber_groups_no_midsag = seg.segment_by_inclusion(hardi_fdata,
                                                  hardi_fbval,
                                                  hardi_fbvec,
                                                  streamlines_no_midsag,
                                                  bundles,
                                                  reg_template=MNI_T2_img,
                                                  mapping=mapping,
                                                  affine=img.affine)

bundles_midsag = {}
bundles_midsag["FP"] = {'ROIs': [templates["FP_L"],
                                 templates["FP_R"]],
                 'rules':[True, True]}
bundles_midsag["FA"] = {'ROIs': [templates["FA_L"],
                                 templates["FA_R"]],
                        'rules':[True, True]}

fiber_groups_midsag = seg.segment_by_inclusion(hardi_fdata,
                                               hardi_fbval,
                                               hardi_fbvec,
                                               streamlines_midsag,
                                               bundles_midsag,
                                               reg_template=MNI_T2_img,
                                               mapping=mapping,
                                               affine=img.affine)

fiber_groups = {**fiber_groups_no_midsag, **fiber_groups_midsag}

print("Getting tract profiles")
n_points = 100

dfs = []
for bundle in bundles:
    print("Getting profile for: %s"%bundle)
    if len(fiber_groups[bundle]) > 0:
        bundle_df = pd.DataFrame(data={'tractID': [bundle] * n_points, 'nodeID': np.arange(1, n_points + 1)})

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
        print("There are no fibers in %s"%bundle)

result = pd.concat(dfs)