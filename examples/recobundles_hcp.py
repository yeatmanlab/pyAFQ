import warnings
import os.path as op
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

warnings.warn(
    "This example requires a reasonably large amount of memory and takes approximately 3-4 hours to run")

subject = 100307

afd.fetch_hcp([subject])
afd.afq_home

dwi_dir = op.join(afd.afq_home, 'HCP', 'derivatives',
                'dmriprep', f'sub-{subject}', 'sess-01/dwi')

hardi_fdata = op.join(dwi_dir, f"sub-{subject}_dwi.nii.gz")
hardi_fbval = op.join(dwi_dir, f"sub-{subject}_dwi.bval")
hardi_fbvec = op.join(dwi_dir, f"sub-{subject}_dwi.bvec")

img = nib.load(hardi_fdata)
gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)

print("Calculating white matter mask")
if not op.exists('./wm_mask.nii.gz'):
    anat_dir = op.join(afd.afq_home, 'HCP', 'derivatives',
                       'dmriprep', f'sub-{subject}', 'sess-01/anat')
    wm_labels = [250, 251, 252, 253, 254, 255, 41, 2, 16, 77]
    seg_img = nib.load(op.join(anat_dir, f"sub-{subject}_aparc+aseg.nii.gz"))
    seg_data_orig = seg_img.get_fdata()
    # For different sets of labels, extract all the voxels that
    # have any of these values:
    wm_mask = np.sum(np.concatenate([(seg_data_orig == l)[..., None]
                                    for l in wm_labels], -1), -1)

    # Resample to DWI data:
    dwi_data = img.get_fdata()
    wm_mask = np.round(reg.resample(wm_mask,
                                    dwi_data[..., 0],
                                    seg_img.affine,
                                    img.affine)).astype(int)

    wm_img = nib.Nifti1Image(wm_mask.astype(int),
                             img.affine)
    nib.save(wm_img, './wm_mask.nii.gz')
else:
    wm_img = nib.load('./wm_mask.nii.gz')
    wm_mask = wm_img.get_data()

print("Calculating DTI...")
if not op.exists('./dti_FA.nii.gz'):
    dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                            out_dir='.', b0_threshold=50,
                            mask='./wm_mask.nii.gz')
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

from AFQ import csd

print("Calculating CSD...")
if not op.exists('./csd_sh_coeff.nii.gz'):
    csd_params = csd.fit_csd(hardi_fdata, hardi_fbval, hardi_fbvec,
                            out_dir='.', b0_threshold=50,
                            mask='./wm_mask.nii.gz')
else:
    csd_params = './csd_sh_coeff.nii.gz'

print("Tracking...")
if not op.exists('csd_streamlines_reco.trk'):
    seed_roi = np.zeros(img.shape[:-1])
    seed_roi[wm_mask > 0] = 1
    # Filter down in addition, to make the whole thing a bit more zippy:
    seed_roi[FA_data < 0.4] = 0  # Comment out or adjust in real use.
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
                                model_clust_thr=20,
                                reduction_thr=20)
segmentation.segment(bundles, streamlines)
fiber_groups = segmentation.fiber_groups

for kk in fiber_groups:
    print(kk, len(fiber_groups[kk]))
    sft = StatefulTractogram(fiber_groups[kk], img, Space.RASMM)
    save_tractogram(sft, './%s_reco.trk'%kk,
                    bbox_valid_check=False)


