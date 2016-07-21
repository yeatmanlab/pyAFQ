"""
==========================
Plotting tract profiles
==========================

An example of tracking and segmenting two tracts, and plotting their tract
profiles for FA (calculated with DTI).

"""


import os.path as op
import matplotlib.pyplot as plt
import nibabel as nib
import dipy.data as dpd
from dipy.data import fetcher

import AFQ.utils.streamlines as aus
import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg


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


print("Tracking...")
if not op.exists('dti_streamlines.trk'):
    streamlines = list(aft.track(dti_params['params']))
    aus.write_trk('./dti_streamlines.trk', streamlines, affine=img.affine)
else:
    streamlines = aus.read_trk('./dti_streamlines.trk')


# Use only a small portion of the streamlines, for expedience:
streamlines = streamlines[::100]

templates = afd.read_templates()
bundle_names = ["CST", "ILF"]

bundles = {}
for name in bundle_names:
    for hemi in ['_R', '_L']:
        bundles[name + hemi] = {'ROIs': [templates[name + '_roi1' + hemi],
                                         templates[name + '_roi1' + hemi]],
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
fiber_groups = seg.segment(hardi_fdata,
                           hardi_fbval,
                           hardi_fbvec,
                           streamlines,
                           bundles,
                           reg_template=MNI_T2_img,
                           mapping=mapping,
                           as_generator=False)

FA_img = nib.load(dti_params['FA'])
FA_data = FA_img.get_data()

print("Extracting tract profiles...")
for bundle in bundles:
    fig, ax = plt.subplots(1)
    w = seg.gaussian_weights(fiber_groups[bundle])
    profile = seg.calculate_tract_profile(FA_data, fiber_groups[bundle],
                                          weights=w)
    ax.plot(profile)
    ax.set_title(bundle)
