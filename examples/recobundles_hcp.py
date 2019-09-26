"""
=========================================
Plotting tract profiles using RecoBundles
=========================================

An example of tracking and segmenting two tracts with RecoBundles, and
plotting their tract profiles for FA (calculated with DTI).
"""

import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.stateful_tractogram import Space
import dipy.core.gradients as dpg
from dipy.segment.mask import median_otsu

import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg

subject = "100206"
afd.fetch_hcp([subject])
afd.afq_home

dwi_dir = op.join(afd.afq_home, 'HCP', 'derivatives',
                  'dmriprep', f'sub-{subject}', 'sess-01/dwi')

hardi_fdata = op.join(dwi_dir, f"sub-{subject}_dwi.nii.gz")
hardi_fbval = op.join(dwi_dir, f"sub-{subject}_dwi.bval")
hardi_fbvec = op.join(dwi_dir, f"sub-{subject}_dwi.bvec")

img = nib.load(hardi_fdata)
gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)

print("Calculating brain-mask")
if not op.exists('./brain_mask.nii.gz'):
    mean_b0 = np.mean(img.get_data()[..., gtab.b0s_mask], -1)
    _, brain_mask = median_otsu(mean_b0, median_radius=4,
                                numpass=1, autocrop=False,
                                vol_idx=None, dilate=10)
    be_img = nib.Nifti1Image(brain_mask.astype(int),
                             img.affine)
    nib.save(be_img, './brain_mask.nii.gz')


print("Calculating DTI...")
if not op.exists('./dti_FA.nii.gz'):
    dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir='.', b0_threshold=50,
                             mask='./brain_mask.nii.gz')
else:
    dti_params = {'FA': './dti_FA.nii.gz',
                  'params': './dti_params.nii.gz'}

FA_img = nib.load(dti_params['FA'])
FA_data = FA_img.get_fdata()

print("Registering to template...")
MNI_T2_img = dpd.read_mni_template()
if not op.exists('mapping.nii.gz'):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    warped_hardi, mapping = reg.syn_register_dwi(hardi_fdata, gtab,
                                                 template=MNI_T2_img)
    reg.write_mapping(mapping, './mapping.nii.gz')
else:
    mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)

from AFQ import api

bundle_names = ['CST',
                'C',
                'F',
                'UF',
                'MCP',
                'AF',
                'CCMid',
                'AF',
                'CC_ForcepsMajor',
                'CC_ForcepsMinor',
                'IFOF']
bundles = api.make_bundle_dict(bundle_names=bundle_names, seg_algo="reco")

# bundle_names = api.BUNDLES
# bundles = api.make_bundle_dict(bundle_names=bundle_names)

from AFQ import csd

print("Calculating CSD...")
if not op.exists('./csd_sh_coeff.nii.gz'):
    csd_params = csd.fit_csd(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir='.', b0_threshold=50,
                             mask='./brain_mask.nii.gz')
else:
    csd_params = './csd_sh_coeff.nii.gz'

print("Tracking...")
if not op.exists('csd_streamlines_reco.trk'):
    seed_roi = np.zeros(img.shape[:-1])
    seed_roi[FA_data > 0.4] = 1
    # for name in bundle_names:
    #     for hemi in ['_R', '_L']:
    #         sl_xform = dts.Streamlines(
    #             dtu.transform_tracking_output(bundles[name + hemi]['sl'],
    #             MNI_T2_img.affine))

    #         delta = dts.values_from_volume(mapping.backward,
    #                                        sl_xform, np.eye(4))
    #         sl_xform = [sum(d, s) for d, s in zip(delta, sl_xform)]

    #         sl_xform = dts.Streamlines(
    #             dtu.transform_tracking_output(sl_xform,
    #             np.linalg.inv(MNI_T2_img.affine)))

    #         sft = StatefulTractogram(sl_xform, img, Space.RASMM)
    #         save_tractogram(sft, './%s%s_atlas.trk' % (name, hemi))

    #         sl_xform = dts.Streamlines(
    #             dtu.transform_tracking_output(sl_xform,
    #                                  np.linalg.inv(img.affine)))

    #         for sl in sl_xform:
    #             sl_as_idx = sl.astype(int)
    #             seed_roi[sl_as_idx[:, 0],
    #                      sl_as_idx[:, 1],
    #                      sl_as_idx[:, 2]] = 1

    nib.save(nib.Nifti1Image(seed_roi, img.affine), 'seed_roi.nii.gz')
    streamlines = aft.track(csd_params, seed_mask=seed_roi,
                            directions='det', stop_mask=FA_data,
                            stop_threshold=0.1)
    print(len(streamlines))
    sft = StatefulTractogram(streamlines, img, Space.RASMM)
    save_tractogram(sft, './csd_streamlines_reco.trk',
                    bbox_valid_check=False)
else:
    tg = load_tractogram('./csd_streamlines_reco.trk', img)
    streamlines = tg.streamlines

print("Segmenting fiber groups...")
segmentation = seg.Segmentation(algo='reco',
                                model_clust_thr=40,
                                reduction_thr=40)
segmentation.segment(bundles, streamlines)
fiber_groups = segmentation.fiber_groups

streamlines = dts.Streamlines(
    dtu.transform_tracking_output(streamlines,
                                  np.linalg.inv(img.affine)))

segmentation = seg.Segmentation()
segmentation.segment(bundles, streamlines, fdata=hardi_fdata,
                     fbval=hardi_fbval, fbvec=hardi_fbvec)
fiber_groups = segmentation.fiber_groups


for kk in fiber_groups:
    print(kk, len(fiber_groups[kk]))
    sft = StatefulTractogram(fiber_groups[kk], img, Space.RASMM)
    save_tractogram(sft, './%s_reco.trk'%kk,
                    bbox_valid_check=False)

print("Cleaning fiber groups...")
for bundle in bundles:
    fiber_groups[bundle] = seg.clean_fiber_group(fiber_groups[bundle])

for kk in fiber_groups:
    print(kk, len(fiber_groups[kk]))

    sft = StatefulTractogram(
        dtu.transform_tracking_output(fiber_groups[kk], img.affine),
        img, Space.RASMM)

    save_tractogram(sft, './%s_afq.trk'%kk,
                    bbox_valid_check=False)



# print("Extracting tract profiles...")
# for name in bundle_names:
#     for hemi in ["_R", "_L"]:
#         fig, ax = plt.subplots(1)
#         streamlines = dts.Streamlines(
#             dtu.transform_tracking_output(
#                 [s for s in fiber_groups[name + hemi] if s.shape[0] > 100],
#                 np.linalg.inv(img.affine)))

#         weights = gaussian_weights(streamlines)
#         profile = afq_profile(FA_data,
#                               streamlines,
#                               np.eye(4),
#                               weights=weights)
#         ax.plot(profile)
#         ax.set_title(name + hemi)

# plt.show()
