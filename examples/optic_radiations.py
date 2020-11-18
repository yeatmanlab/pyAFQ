"""
=============================
Plotting the Optic Radiations
=============================


"""
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import dipy.data as dpd
from dipy.data import fetcher
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.stateful_tractogram import Space
from dipy.reconst import shm

import AFQ.data as afd
import AFQ.tractography as aft
import AFQ.registration as reg
import AFQ.models.dti as dti
import AFQ.models.csd as csd
import AFQ.segmentation as seg
from AFQ.utils.volume import patch_up_roi

import logging
logging.basicConfig(level=logging.INFO)

# Target directory for this example's output files
working_dir = "./optic_radiations"

##########################################################################
# Get example data:
# -------------------------

dpd.fetch_stanford_hardi()
hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
img = nib.load(hardi_fdata)

##########################################################################
# Calculate DTI:
# -------------------------

print("Calculating DTI...")
if not op.exists(op.join(working_dir, 'dti_FA.nii.gz')):
    dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
                             out_dir=working_dir)
else:
    dti_params = {'FA': op.join(working_dir, 'dti_FA.nii.gz'),
                  'params': op.join(working_dir, 'dti_params.nii.gz')}

FA_img = nib.load(dti_params['FA'])
FA_data = FA_img.get_fdata()

##########################################################################
# Calculate CSD:
# -------------------------
print("Calculating CSD...")
if not op.exists(op.join(working_dir, 'csd_sh_coeff.nii.gz')):
    sh_coeff = csd.fit_csd(hardi_fdata, hardi_fbval, hardi_fbvec,
                           sh_order=4, out_dir=working_dir)
else:
    sh_coeff = op.join(working_dir, "csd_sh_coeff.nii.gz")

apm = shm.anisotropic_power(nib.load(sh_coeff).get_fdata())

##########################################################################
# Register the individual data to a template:
# -------------------------------------------
# For the purpose of bundle segmentation, the individual brain is registered to
# the MNI T1 template. The waypoint ROIs used in segmentation are then each
# brought into each subject's native space to test streamlines for whether they
# fulfill the segmentation criteria.
#
# .. note::
#
#     To find the right place for the waypoint ROIs, we calculate a non-linear
#     transformation between the individual's brain DWI measurement (the b0
#     measurements) and the MNI T1 template.
#     Before calculating this non-linear warping, we perform a pre-alignment
#     using an affine transformation.

print("Registering to template...")

MNI_T1w_img = afd.read_mni_template(weight="T1w")

if not op.exists(op.join(working_dir, 'mapping.nii.gz')):
    import dipy.core.gradients as dpg
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    # Prealign using affine registration
    _, prealign = reg.affine_registration(
        apm,
        MNI_T1w_img.get_fdata(),
        img.affine,
        MNI_T1w_img.affine)

    # Then register using a non-linear registration using the affine for
    # prealignment
    warped_hardi, mapping = reg.syn_register_dwi(hardi_fdata, gtab,
                                                 prealign=prealign)
    reg.write_mapping(mapping, op.join(working_dir, 'mapping.nii.gz'))
else:
    mapping = reg.read_mapping(op.join(working_dir, 'mapping.nii.gz'),
                               img, MNI_T1w_img)


##########################################################################
# Bundle specification
# -------------------------------------------

roi_folder = op.join(op.expanduser('~'), "AFQ_Data", "visual")
waypoint_folder = op.join(roi_folder, "waypoint")

waypoint_roi_fnames = [
    "left_OR_1.nii.gz",
    "left_OR_2.nii.gz",
    "left_OP_MNI.nii.gz",
    "left_TP_MNI.nii.gz",
    "left_pos_thal_MNI.nii.gz",
    "right_OR_1.nii.gz",
    "right_OR_2.nii.gz",
    "right_pos_thal_MNI.nii.gz",
    "right_OP_MNI.nii.gz",
    "right_TP_MNI.nii.gz"]

waypoint_rois = {}

for fname in waypoint_roi_fnames:
    waypoint_rois[fname.split('.')[0]] = afd.read_resample_roi(
        op.join(waypoint_folder, fname))

bundles = {
    "L_OR": {
        "ROIs": [waypoint_rois["left_OR_1"],
                 waypoint_rois["left_OR_2"],
                 waypoint_rois["left_OP_MNI"],
                 waypoint_rois["left_TP_MNI"],
                 waypoint_rois["left_pos_thal_MNI"]],
        "rules": [True, True, False, False, False],
        "cross_midline": False,
        "uid": 1
        },
    "R_OR": {
        "ROIs": [waypoint_rois["right_OR_1"],
                 waypoint_rois["right_OR_2"],
                 waypoint_rois["right_OP_MNI"],
                 waypoint_rois["right_TP_MNI"],
                 waypoint_rois["right_pos_thal_MNI"]],
        "rules": [True, True, False, False, False],
        "cross_midline": False,
        "uid": 2
        }
    }

endpoint_folder = op.join(roi_folder, "endpoint")

# bundles = {
#     "L_OR": {
#         "ROIs": [nib.load(op.join(endpoint_folder,
#                                        'left_thal_MNI.nii.gz')),
#                  nib.load(op.join(endpoint_folder,
#                                      'left_V1_MNI.nii.gz')),
#                  waypoint_rois["left_OP_MNI"],
#                  waypoint_rois["left_TP_MNI"],
#                  waypoint_rois["left_pos_thal_MNI"]],
#         "rules": [True, True, False, False, False],
#         "cross_midline": False,
#         "uid": 1
#         },
#     "R_OR": {
#         "ROIs": [nib.load(op.join(endpoint_folder,
#                                        'right_thal_MNI.nii.gz')),
#                  nib.load(op.join(endpoint_folder,
#                                      'right_V1_MNI.nii.gz')),
#                  waypoint_rois["right_OP_MNI"],
#                  waypoint_rois["right_TP_MNI"],
#                  waypoint_rois["right_pos_thal_MNI"]],
#         "rules": [True, True, False, False, False],
#         "cross_midline": False,
#         "uid": 2
#         }
#     }




##########################################################################
# Endpoints
# ----------
endpoint_folder = op.join(roi_folder, "endpoint")

endpoint_spec = {
    "L_OR": {
        "startpoint": nib.load(op.join(endpoint_folder,
                                       'left_thal_MNI.nii.gz')),
        "endpoint": nib.load(op.join(endpoint_folder,
                                     'left_V1_MNI.nii.gz'))},
    "R_OR": {
        "startpoint": nib.load(op.join(endpoint_folder,
                                       'right_thal_MNI.nii.gz')),
        "endpoint": nib.load(op.join(endpoint_folder,
                                     'right_V1_MNI.nii.gz'))}}

##########################################################################
# Tracking
# --------

from dipy.data import get_fnames
f_pve_csf, f_pve_gm, f_pve_wm = get_fnames('stanford_pve_maps')

pve_csf = nib.load(f_pve_csf)
pve_gm = nib.load(f_pve_gm)
pve_wm = nib.load(f_pve_wm)

print("Tracking...")
if not op.exists(op.join(working_dir, 'pft_streamlines.trk')):
    seed_roi = np.zeros(img.shape[:-1])
    for bundle in bundles:
        for idx, roi in enumerate(bundles[bundle]['ROIs']):
            warped_roi = patch_up_roi(
                mapping.transform_inverse(
                    roi.get_fdata().astype(np.float32),
                    interpolation='linear'),
                bundle_name=bundle)
            print(roi)
            nib.save(nib.Nifti1Image(warped_roi.astype(float), img.affine),
                     op.join(working_dir, f"{bundle}_{idx+1}.nii.gz"))

            # Add voxels that aren't there yet:
            if bundles[bundle]['rules'][idx]:
                seed_roi = np.logical_or(seed_roi, warped_roi)

        for ii, pp in enumerate(endpoint_spec[bundle].keys()):
            roi = endpoint_spec[bundle][pp]
            roi = reg.resample(roi.get_fdata(),
                               MNI_T1w_img,
                               roi.affine,
                               MNI_T1w_img.affine)

            warped_roi = patch_up_roi(
                mapping.transform_inverse(
                    roi.astype(np.float32),
                    interpolation='linear'),
                bundle_name=bundle)

            nib.save(nib.Nifti1Image(warped_roi.astype(float), img.affine),
                     op.join(working_dir, f"{bundle}_{pp}.nii.gz"))

    nib.save(nib.Nifti1Image(seed_roi.astype(float), img.affine),
             op.join(working_dir, 'seed_roi.nii.gz'))

    sft = aft.track(sh_coeff,
                    seed_mask=seed_roi,
                    n_seeds=5,
                    tracker="pft",
                    stop_mask=(pve_wm, pve_gm, pve_csf),
                    stop_threshold="ACT",
                    directions="prob",
                    odf_model="CSD")

    save_tractogram(sft, op.join(working_dir, 'pft_streamlines.trk'),
                    bbox_valid_check=False)
else:
    sft = load_tractogram(op.join(working_dir, 'pft_streamlines.trk'), img)

sft.to_vox()


##########################################################################
# Segmentation
# ------------

print("Segmenting fiber groups...")
segmentation = seg.Segmentation(return_idx=True,
                                dist_to_atlas=5)
segmentation.segment(bundles,
                     sft,
                     fdata=hardi_fdata,
                     fbval=hardi_fbval,
                     fbvec=hardi_fbvec,
                     mapping=mapping,
                     reg_template=MNI_T1w_img,
                     endpoint_dict=endpoint_spec)

fiber_groups = segmentation.fiber_groups


##########################################################################
# Cleaning
# --------

print("Cleaning fiber groups...")
for bundle in bundles:
    print(f"Cleaning {bundle}")
    print(f"Before cleaning: {len(fiber_groups[bundle]['sl'])} streamlines")
    new_fibers, idx_in_bundle = seg.clean_bundle(
        fiber_groups[bundle]['sl'],
        return_idx=True)
    print(f"Afer cleaning: {len(new_fibers)} streamlines")
    new_fibers = fiber_groups[bundle]['sl']
    idx_in_global = fiber_groups[bundle]['idx'][idx_in_bundle]
    np.save(op.join(working_dir, f'{bundle}_idx.npy'), idx_in_global)
    sft = StatefulTractogram(new_fibers.streamlines,
                             img,
                             Space.VOX)
    sft.to_rasmm()
    save_tractogram(sft, op.join(working_dir, f'{bundle}_afq.trk'),
                    bbox_valid_check=False)


##########################################################################
# Bundle profiles
# ---------------

print("Extracting tract profiles...")
for bundle in bundles:
    sft = load_tractogram(op.join(working_dir, f'{bundle}_afq.trk'),
                          img, to_space=Space.VOX)
    fig, ax = plt.subplots(1)
    weights = gaussian_weights(sft.streamlines)
    profile = afq_profile(FA_data, sft.streamlines,
                          np.eye(4), weights=weights)
    ax.plot(profile)
    ax.set_title(bundle)

plt.show()

##########################################################################
# References:
# -------------------------
# .. [Yeatman2012] Jason D Yeatman, Robert F Dougherty, Nathaniel J Myall,
#                  Brian A Wandell, Heidi M Feldman, "Tract profiles of
#                  white matter properties: automating fiber-tract
#                  quantification", PloS One, 7: e49790
#
# .. [Yeatman2014] Jason D Yeatman, Brian A Wandell, Aviv Mezer Feldman,
#                  "Lifespan maturation and degeneration of human brain white
#                  matter", Nature Communications 5: 4932
