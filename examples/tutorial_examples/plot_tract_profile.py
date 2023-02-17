"""
==========================
Plotting tract profiles
==========================

An example of tracking and segmenting two tracts, and plotting their tract
profiles for FA (calculated with DTI). This example uses the Yeatman et al.
waypoint ROI approach, first described in [Yeatman2012]_ and further elaborated
in [Yeatman2014]_.

This example goes into the nitty gritty of the underlying methods. This example
is provided as a way to understand what goes on under the hood, but for most
practical uses of pyAFQ, we recommend against using the methods directly,
instead directing users to use the `GroupAFQ` API, demonstrated in other
examples.

"""
# import os.path as op
# import matplotlib.pyplot as plt
# import numpy as np
# import nibabel as nib
# import dipy.data as dpd
# from dipy.data import fetcher
# from dipy.io.streamline import save_tractogram, load_tractogram
# from dipy.stats.analysis import afq_profile, gaussian_weights
# from dipy.io.stateful_tractogram import StatefulTractogram
# from dipy.io.stateful_tractogram import Space
# from dipy.align import affine_registration

# import AFQ.api.bundle_dict as abd
# import AFQ.data.fetch as afd
# import AFQ.tractography as aft
# import AFQ.registration as reg
# import AFQ.models.dti as dti
# import AFQ.segmentation as seg
# from AFQ.utils.volume import transform_inverse_roi

# import logging
# logging.basicConfig(level=logging.INFO)

# # Target directory for this example's output files
# working_dir = "./tract_profile"

# ##########################################################################
# # Get example data:
# # -------------------------

# dpd.fetch_stanford_hardi()
# hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
# hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
# hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
# hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
# img = nib.load(hardi_fdata)

# ##########################################################################
# # Calculate DTI:
# # -------------------------

# print("Calculating DTI...")
# if not op.exists(op.join(working_dir, 'dti_FA.nii.gz')):
#     dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
#                              out_dir=working_dir)
# else:
#     dti_params = {'FA': op.join(working_dir, 'dti_FA.nii.gz'),
#                   'params': op.join(working_dir, 'dti_params.nii.gz')}

# FA_img = nib.load(dti_params['FA'])
# FA_data = FA_img.get_fdata()

# ##########################################################################
# # Register the individual data to a template:
# # -------------------------------------------
# # For the purpose of bundle segmentation, the individual brain is registered to
# # the MNI T2 template. The waypoint ROIs used in segmentation are then each
# # brought into each subject's native space to test streamlines for whether they
# # fulfill the segmentation criteria.
# #
# # .. note::
# #
# #     To find the right place for the waypoint ROIs, we calculate a non-linear
# #     transformation between the individual's brain DWI measurement (the b0
# #     measurements) and the MNI T2 template.
# #     Before calculating this non-linear warping, we perform a pre-alignment
# #     using an affine transformation.

# print("Registering to template...")
# MNI_T2_img = afd.read_mni_template()

# if not op.exists(op.join(working_dir, 'mapping.nii.gz')):
#     import dipy.core.gradients as dpg
#     gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
#     b0 = np.mean(img.get_fdata()[..., gtab.b0s_mask], -1)
#     # Prealign using affine registration
#     _, prealign = affine_registration(
#         b0,
#         MNI_T2_img.get_fdata(),
#         img.affine,
#         MNI_T2_img.affine)

#     # Then register using a non-linear registration using the affine for
#     # prealignment
#     warped_hardi, mapping = reg.syn_register_dwi(hardi_fdata, gtab,
#                                                  prealign=prealign)
#     reg.write_mapping(mapping, op.join(working_dir, 'mapping.nii.gz'))
# else:
#     mapping = reg.read_mapping(op.join(working_dir, 'mapping.nii.gz'),
#                                img, MNI_T2_img)


# ##########################################################################
# # Read in bundle specification
# # -------------------------------------------
# # The waypoint ROIs, in addition to bundle probability maps are stored in this
# # data structure. The templates are first resampled into the MNI space, before
# # they are brought into the subject's individual native space.
# # For speed, we only segment two bundles here.

# bundles = abd.BundleDict(
#     ["CST_L", "CST_R", "ARC_L", "ARC_R"],
#     resample_to=MNI_T2_img)


# ##########################################################################
# # Tracking
# # --------
# # Streamlines are generate using DTI and a deterministic tractography
# # algorithm. For speed, we seed only within the waypoint ROIs for each bundle.

# print("Tracking...")
# if not op.exists(op.join(working_dir, 'dti_streamlines.trk')):
#     seed_roi = np.zeros(img.shape[:-1])
#     for bundle in bundles:
#         for idx, roi in enumerate(bundles[bundle]['include']):
#             warped_roi = transform_inverse_roi(
#                 roi,
#                 mapping,
#                 bundle_name=bundle)

#             nib.save(
#                 nib.Nifti1Image(warped_roi.astype(float), img.affine),
#                 op.join(working_dir, f"{bundle}_{idx+1}.nii.gz"))
#             # Add voxels that aren't there yet:
#             seed_roi = np.logical_or(seed_roi, warped_roi)
#     nib.save(nib.Nifti1Image(
#         seed_roi.astype(float), img.affine),
#         op.join(working_dir, 'seed_roi.nii.gz'))
#     sft = aft.track(dti_params['params'], seed_mask=seed_roi,
#                     stop_mask=FA_data, stop_threshold=0.1,
#                     directions="det", odf_model="dti")
#     save_tractogram(sft, op.join(working_dir, 'dti_streamlines.trk'),
#                     bbox_valid_check=False)
# else:
#     sft = load_tractogram(op.join(working_dir, 'dti_streamlines.trk'), img)

# sft.to_vox()

# ##########################################################################
# # Segmentation
# # --------
# # In this stage, streamlines are tested for several criteria: whether the
# # probability that they belong to a bundle is larger than a threshold (set to
# # 0,per default), whether they pass through inclusion ROIs and whether they do
# # not pass through exclusion ROIs.

# print("Segmenting fiber groups...")
# segmentation = seg.Segmentation(return_idx=True)
# segmentation.segment(bundles,
#                      sft,
#                      fdata=hardi_fdata,
#                      fbval=hardi_fbval,
#                      fbvec=hardi_fbvec,
#                      mapping=mapping,
#                      reg_template=MNI_T2_img)

# fiber_groups = segmentation.fiber_groups


# ##########################################################################
# # Cleaning
# # --------
# # Each fiber group is cleaned to exclude streamlines that are outliers in terms
# # of their trajector and/or length.

# print("Cleaning fiber groups...")
# for bundle in bundles:
#     print(f"Cleaning {bundle}")
#     print(f"Before cleaning: {len(fiber_groups[bundle]['sl'])} streamlines")
#     new_fibers, idx_in_bundle = seg.clean_bundle(
#         fiber_groups[bundle]['sl'],
#         return_idx=True)
#     print(f"Afer cleaning: {len(new_fibers)} streamlines")

#     idx_in_global = fiber_groups[bundle]['idx'][idx_in_bundle]
#     np.save(op.join(working_dir, f'{bundle}_idx.npy'), idx_in_global)
#     sft = StatefulTractogram(new_fibers.streamlines,
#                              img,
#                              Space.VOX)
#     sft.to_rasmm()
#     save_tractogram(sft, op.join(working_dir, f'{bundle}_afq.trk'),
#                     bbox_valid_check=False)


# ##########################################################################
# # Bundle profiles
# # ---------------
# # Streamlines are represented in the original diffusion space (`Space.VOX`) and
# # scalar properties along the length of each bundle are queried from this
# # scalar data. Here, the contribution of each streamline is weighted according
# # to how representative this streamline is of the bundle overall.

# print("Extracting tract profiles...")
# for bundle in bundles:
#     sft = load_tractogram(op.join(working_dir, f'{bundle}_afq.trk'),
#                           img, to_space=Space.VOX)
#     fig, ax = plt.subplots(1)
#     weights = gaussian_weights(sft.streamlines)
#     profile = afq_profile(FA_data, sft.streamlines,
#                           np.eye(4), weights=weights)
#     ax.plot(profile)
#     ax.set_title(bundle)

# plt.show()

# ##########################################################################
# # References:
# # -------------------------
# # .. [Yeatman2012] Jason D Yeatman, Robert F Dougherty, Nathaniel J Myall,
# #                  Brian A Wandell, Heidi M Feldman, "Tract profiles of
# #                  white matter properties: automating fiber-tract
# #                  quantification", PloS One, 7: e49790
# #
# # .. [Yeatman2014] Jason D Yeatman, Brian A Wandell, Aviv Mezer Feldman,
# #                  "Lifespan maturation and degeneration of human brain white
# #                  matter", Nature Communications 5: 4932
