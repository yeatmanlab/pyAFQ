"""
==========================
Plotting tract profiles
==========================

An example of tracking and segmenting two tracts, and plotting their tract
profiles for FA (calculated with DTI).

"""
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import dipy.data as dpd
from dipy.data import fetcher
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.io.streamline import save_tractogram
from dipy.stats.analysis import afq_profile, gaussian_weights

import AFQ.utils.streamlines as aus
import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg
from AFQ.utils.volume import patch_up_roi

dpd.fetch_stanford_hardi()

hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")

img = nib.load(hardi_fdata)

print("Calculating DTI...")
if not op.exists('./dti_FA.nii.gz'):
    dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir='.')
else:
    dti_params = {'FA': './dti_FA.nii.gz',
                  'params': './dti_params.nii.gz'}

FA_img = nib.load(dti_params['FA'])
FA_data = FA_img.get_fdata()

templates = afd.read_templates()
bundle_names = ["CST", "ILF"]

bundles = {}
for name in bundle_names:
    for hemi in ['_R', '_L']:
        bundles[name + hemi] = {
            'ROIs': [templates[name + '_roi1' + hemi],
                     templates[name + '_roi2' + hemi]],
            'rules': [True, True],
            'prob_map': templates[name + hemi + '_prob_map'],
            'cross_midline': False}

print("Registering to template...")
MNI_T2_img = dpd.read_mni_template()
if not op.exists('mapping.nii.gz'):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    warped_hardi, mapping = reg.syn_register_dwi(hardi_fdata, gtab)
    reg.write_mapping(mapping, './mapping.nii.gz')
else:
    mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)


print("Tracking...")
if not op.exists('dti_streamlines.trk'):
    seed_roi = np.zeros(img.shape[:-1])
    for name in bundle_names:
        for hemi in ['_R', '_L']:
            for roi in bundles[name + hemi]['ROIs']:
                warped_roi = patch_up_roi(
                    (mapping.transform_inverse(
                        roi.get_data().astype(np.float32),
                     interpolation='linear')) > 0)
                # Add voxels that aren't there yet:
                seed_roi = np.logical_or(seed_roi, warped_roi)

    streamlines = aft.track(dti_params['params'], seed_mask=seed_roi,
                            stop_mask=FA_data > 0.2)

    save_tractogram('./dti_streamlines.trk', streamlines, np.eye(4))
else:
    tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
    streamlines = tg.streamlines

streamlines = dts.Streamlines(dtu.transform_tracking_output(
    [s for s in streamlines if s.shape[0] > 100],
    np.linalg.inv(img.affine)))

print("Segmenting fiber groups...")
segmentation = seg.Segmentation()
segmentation.segment(hardi_fdata, hardi_fbval, hardi_fbvec, bundles,
                     streamlines, mapping=mapping, reg_template=MNI_T2_img)
fiber_groups = segmentation.fiber_groups

print("Cleaning fiber groups...")
for bundle in bundles:
    fiber_groups[bundle] = seg.clean_fiber_group(fiber_groups[bundle])

print("Extracting tract profiles...")
for bundle in bundles:
    fig, ax = plt.subplots(1)
    weights = gaussian_weights(fiber_groups[bundle])
    profile = afq_profile(FA_data, fiber_groups[bundle],
                              np.eye(4), weights=weights)
    ax.plot(profile)
    ax.set_title(bundle)

plt.show()
