import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import logging
import nibabel as nib
import dipy.data as dpd
from dipy.data import fetcher
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts

import AFQ.utils.streamlines as aus
import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.dti as dti
import AFQ.segmentation as seg
import AFQ.utils.volume as auv

logging.basicConfig(level=logging.INFO)

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

print("Registering to template...")

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

MNI_T2_img = dpd.read_mni_template()
if not op.exists('mapping.nii.gz'):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    mapping = reg.syn_register_dwi(hardi_fdata, gtab)[1]
    reg.write_mapping(mapping, './mapping.nii.gz')
else:
    mapping = reg.read_mapping('./mapping.nii.gz', img, MNI_T2_img)

def mask_from_bundle_ROI(bundles, mapping):
    """
    Creates Mask from ROIs in bundles

    Parameters
    ----------
    bundles : dict
        The format is something like::

            {'name': {'ROIs':[img1, img2],
            'rules':[True, True],
            'prob_map': img3,
            'cross_midline': False}}
    mapping : DiffeomorphicMap object
        A mapping between DWI space and a template.

    Returns
    -------
    A single mask which is the logical or of all ROIs in bundles,
    transformed to be used as a seed mask
    """

    roi1 = bundles['CST_L']['ROIs'][0].get_fdata()
    roi2 = bundles['CST_L']['ROIs'][1].get_fdata()

    warped_roi1 = auv.patch_up_roi(
        (mapping.transform_inverse(
            roi1,
            interpolation='linear')) > 0)
    warped_roi2 = auv.patch_up_roi(
        (mapping.transform_inverse(
            roi2,
            interpolation='linear')) > 0)

    return np.logical_or(warped_roi1, warped_roi2)

print("Generating Seed Masks...")
seed_masks = mask_from_bundle_ROI(bundles, mapping)

print("Getting Tracks...")
if not op.exists('dti_streamlines.trk'):
    streamlines = list(aft.track(dti_params['params'], seed_mask=None))
    aus.write_trk('./dti_streamlines.trk', streamlines, affine=img.affine)
else:
    tg = nib.streamlines.load('./dti_streamlines.trk').tractogram
    streamlines = tg.apply_affine(np.linalg.inv(img.affine)).streamlines

streamlines = dts.Streamlines(dtu.move_streamlines(
    [s for s in streamlines if s.shape[0] > 100],
    np.linalg.inv(img.affine)))

from fury import actor, window
from fury.colormap import line_colors
scene = window.Scene()
scene.add(actor.line(streamlines, line_colors(streamlines)))
scene.add(actor.contour_from_roi(seed_masks, img.affine, [0, 1, 1], 0.5))
window.show(scene)

