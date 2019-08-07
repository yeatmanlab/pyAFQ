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
from dipy.data import fetcher
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.io.streamline import save_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr


import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg
import AFQ.csd as csd

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


print("Calculating CSD...")
if not op.exists('./csd_sh_coeff.nii.gz'):
    dti_params = csd.fit_csd(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir='.')
else:
    csd_params = './csd_sh_coeff.nii.gz'

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

templates = afd.read_templates()
bundle_names = ["CST", "AF"]

bundles = {}
uid = 1
bundle_dict = afd.read_hcp_atlas_16_bundles()
bundles["whole_brain"] = bundle_dict["whole_brain"]

for name in bundle_names:
    for hemi in ["_R", "_L"]:
        bundles[name + hemi] = bundle_dict[name + hemi]
        bundles[name + hemi]['uid'] = uid
        uid += 1

print("Tracking...")
if not op.exists('csd_streamlines.trk'):
    seed_roi = np.zeros(img.shape[:-1])
    for name in bundle_names:
        for hemi in ['_R', '_L']:
            sl_xform = dts.Streamlines(
                dtu.move_streamlines(bundles[name + hemi]['sl'],
                                     MNI_T2_img.affine))

            delta = dts.values_from_volume(mapping.backward,
                                           sl_xform)
            sl_xform = [sum(d, s) for d, s in zip(delta, sl_xform)]

            sl_xform = dts.Streamlines(
                dtu.move_streamlines(sl_xform,
                                     np.linalg.inv(MNI_T2_img.affine)))

            save_tractogram('./%s%s_atlas.trk' % (name, hemi),
                            sl_xform, np.eye(4))

            sl_xform = dts.Streamlines(
                dtu.move_streamlines(sl_xform,
                                     np.linalg.inv(img.affine)))

            for sl in sl_xform:
                sl_as_idx = sl.astype(int)
                seed_roi[sl_as_idx[:, 0],
                         sl_as_idx[:, 1],
                         sl_as_idx[:, 2]] = 1

    nib.save(nib.Nifti1Image(seed_roi, img.affine), 'seed_roi.nii.gz')
    streamlines = aft.track(csd_params, seed_mask=seed_roi,
                            stop_mask=FA_data, stop_threshold=0.1)

    save_tractogram('./dti_streamlines.trk', streamlines, np.eye(4))
else:
    tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
    streamlines = tg.streamlines

fiber_groups = {}
# We start with whole-brain SLR:
atlas = bundle_dict['whole_brain']
moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, streamlines, x0='affine', verbose=False, progressive=True)

# We generate our instance of RB with the moved streamlines:
rb = RecoBundles(moved, verbose=False)

# Next we'll iterate over bundles, registering each one:
bundle_list = list(bundles.keys())
bundle_list.remove('whole_brain')

for bundle in bundle_list:
    model_sl = bundle_dict[bundle]['sl']
    _, rec_labels = rb.recognize(model_bundle=model_sl,
                                 model_clust_thr=20.,
                                 reduction_thr=40,
                                 reduction_distance='mam',
                                 slr=True,
                                 slr_metric='asymmetric',
                                 pruning_distance='mam')

    # Use the streamlines in the original space:
    recognized_sl = streamlines[rec_labels]
    standard_sl = bundle_dict[bundle]['centroid']
    oriented_sl = dts.orient_by_streamline(recognized_sl, standard_sl)
    fiber_groups[bundle] = oriented_sl


for kk in fiber_groups:
    print(kk, len(fiber_groups[kk]))

print("Extracting tract profiles...")
for name in bundle_names:
    for hemi in ["_R", "_L"]:
        fig, ax = plt.subplots(1)
        streamlines = dts.Streamlines(
            dtu.move_streamlines(
                [s for s in fiber_groups[name + hemi] if s.shape[0] > 100],
                np.linalg.inv(img.affine)))

        weights = seg.gaussian_weights(streamlines)
        profile = seg.calculate_tract_profile(FA_data,
                                              streamlines,
                                              weights=weights)
        ax.plot(profile)
        ax.set_title(name + hemi)
        save_tractogram('./%s%s_subject.trk' % (name, hemi),
                        streamlines, img.affine)


plt.show()
