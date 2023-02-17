# """
# ==============================
# Plotting Novel Tract Profiles:
# ==============================

# The following is an example of tractometry for a novel bundle and plotting the
# resulting FA tract profile. We will run tractometry for the *anterior forceps*
# using waypoint ROIs.

# **AFQ Waypoint ROI Tractometry:**

# .. note::

#   This example uses the Yeatman et al. waypoint ROI approach, first
#   described in [Yeatman2012]_ and further elaborated in [Yeatman2014]_.

#   The waypoint ROIs used in this example are from the anterior frontal lobe of
#   the corpus callosum (AntFrontal). The waypoint ROIs are from the human corpus
#   callosum templates:

#     https://figshare.com/articles/Templates_for_Automated_Fiber_Quantification_of_corpus_callosum_from_Diffusion_MRI_data/3381523

# """
# import os.path as op
# import matplotlib.pyplot as plt
# import plotly
# import numpy as np
# import nibabel as nib
# import dipy.data as dpd
# from dipy.data import fetcher
# from dipy.io.streamline import save_tractogram, load_tractogram
# from dipy.stats.analysis import afq_profile, gaussian_weights
# from dipy.io.stateful_tractogram import StatefulTractogram
# from dipy.io.stateful_tractogram import Space
# from dipy.align import affine_registration

# import AFQ.data.fetch as afd
# import AFQ.tractography as aft
# import AFQ.registration as reg
# import AFQ.models.dti as dti
# import AFQ.segmentation as seg
# from AFQ.utils.streamlines import SegmentedSFT
# from AFQ.utils.volume import transform_inverse_roi, density_map
# from AFQ.viz.plot import show_anatomical_slices
# from AFQ.viz.plotly_backend import visualize_bundles, visualize_volume

# import logging
# import sys

# # Ensure segmentation logging information is included in this example's output
# root = logging.getLogger()
# root.setLevel(logging.ERROR)
# logging.getLogger('AFQ').setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# root.addHandler(handler)

# # Target directory for this example's output files
# working_dir = "./callosal_tract_profile"


# ##########################################################################
# # Get example data:
# # -----------------
# # **Diffusion dataset**
# #
# # .. note::
# #   The diffusion data used in this example are from the Stanford High Angular
# #   Resolution Diffusion Imaging (HARDI) dataset:
# #
# #     https://purl.stanford.edu/ng782rw8378


# print("Fetching data...")

# # If does not already exist `fetch_stanford_hardi` will download the first
# # subject's session from the HARDI data into fetcher.dipy_home:
# # `~/.dipy/stanford_hardi`
# dpd.fetch_stanford_hardi()

# # Reference to data file locations
# hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
# hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
# hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
# hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")

# print(f'Loading data file: {hardi_fdata}')
# img = nib.load(hardi_fdata)

# # Display output from this step
# img_data = img.get_fdata()

# # working with space and time data
# img_data = img_data[..., int(img_data.shape[3] / 2)]

# show_anatomical_slices(img_data, 'HARDI 150 DWI')
# print(f'bvals: f{np.loadtxt(hardi_fbval)}')
# print(f'bvec: f{np.loadtxt(hardi_fbvec)}')


# ##########################################################################
# # Calculate DTI:
# # --------------
# # Fit the DTI model using default settings, save files with derived maps.
# #
# # By default following DTI measurements are calculated:
# #
# # - Fractional anisotropy (FA),
# #
# # - Mean diffusivity (MD),
# #
# # - Axial diffusivity (AD),
# #
# # - and Radial diffusivity (RD)
# #
# # In this example we will only use FA.
# #
# # .. note::
# #   By default:
# #
# #   - All b-values less than or equal to 50 are considered to be
# #     without diffusion weighting.
# #
# #   - No binary masks are applied; therefore all voxels are processed.
# #
# # .. note::
# #   The diffusion tensor imaging parameters contain the associated eigenvalues
# #   and eigenvectors from eigen decomposition on the diffusion tensor.


# print("Calculating DTI...")

# if not op.exists(op.join(working_dir, 'dti_FA.nii.gz')):
#     dti_params = dti.fit_dti(hardi_fdata, hardi_fbval, hardi_fbvec,
#                              out_dir=working_dir)
# else:
#     dti_params = {'FA': op.join(working_dir, 'dti_FA.nii.gz'),
#                   'params': op.join(working_dir, 'dti_params.nii.gz')}

# print(f"Loading {dti_params['FA']}")
# FA_img = nib.load(dti_params['FA'])
# FA_data = FA_img.get_fdata()

# show_anatomical_slices(FA_data, 'Fractional Anisotropy (FA)')

# ##########################################################################
# # Register the individual data to a template:
# # -------------------------------------------
# # For the purpose of bundle segmentation, the individual brain is registered
# # to the MNI T2 template. The waypoint ROIs used in segmentation are then each
# # brought into each subject's native space to test streamlines for whether they
# # fulfill the segmentation criteria.
# #
# # .. note::
# #     To find the right place for the waypoint ROIs, we calculate a non-linear
# #     transformation between the individual's brain and the MNI T2 template.
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

# mapping_img = nib.load(op.join(working_dir, 'mapping.nii.gz'))
# mapping_img_data = mapping_img.get_fdata()

# # Working with diffeomorphic map data
# mapping_img_data = mapping_img_data[..., 0, 0]
# show_anatomical_slices(mapping_img_data, 'Registration Displacement Mapping')

# # plot the transformational map of MNI T2 onto subject space
# warped_MNI_T2_data = mapping.transform_inverse(MNI_T2_img.get_fdata())
# warped_MNI_T2_img = nib.Nifti1Image(warped_MNI_T2_data.astype(float),
#                                     img.affine)

# nib.save(warped_MNI_T2_img, op.join(working_dir, 'warped_MNI_T2.nii.gz'))

# show_anatomical_slices(warped_MNI_T2_img.get_fdata(), 'Warped MNI T2')

# ##########################################################################
# # Create novel bundle specification:
# # ----------------------------------
# # The bundles specify meta-data for the segmentation.
# #
# # The following keys are required for a `bundle` entry:
# #
# # - `ROIs`
# #
# #   lists the ROI templates.
# #
# # .. note::
# #   Order of the `ROIs` matters and may result in different tract profiles.
# #   Given a sequence of waypoint ROIs the endpoints should appear first. Which
# #   endpoint appears first should be consistent with the directionality of
# #   other bundles defintions. Any intermediate waypoints ROIs should respect
# #   this ordering.
# #
# # - `rules`
# #
# #   label each ROI as inclusionary `True` or exclusionary `False`.
# #
# # - `cross_midline`
# #
# #   whether or not the streamline crosses the midline
# #
# # .. note::
# #
# #   It is also possible to utilize probablity maps to further refine the
# #   segmentation. If `prob_map` key is not specified the probablities will
# #   all be ones and same shape as the ROI.
# #
# # .. note::
# #
# #   If using multiple bundles it is recommended to add an optional identifier
# #   key such as `uid`. This key is not used by segmentation, but can be helpful
# #   for debugging or quality control reports.
# #
# # To identify candidate streamlines for the anterior forceps we will use three
# # waypoint ROI templates:
# #
# # 1. Left anterior frontal,
# #
# # 2. Right anterior frontal,
# #
# # 3. and midsaggital.
# #
# # The templates are first resampled into the MNI space, before
# # they are brought into the subject's individual native space.


# print("Fetching callosum ROI templates...")

# callosal_templates = afd.read_callosum_templates(resample_to=MNI_T2_img)

# show_anatomical_slices(callosal_templates["L_AntFrontal"].get_fdata(),
#                        'Left anterior frontal ROI')
# show_anatomical_slices(callosal_templates["R_AntFrontal"].get_fdata(),
#                        'Right anterior frontal ROI')
# show_anatomical_slices(callosal_templates["Callosum_midsag"].get_fdata(),
#                        'Midsagittal ROI')

# print("Creating callosal bundle specification...")

# # bundle dict
# bundles = {}

# # anterior frontal ROIs
# bundles["AntFrontal"] = {
#     'include': [
#         callosal_templates["L_AntFrontal"],
#         callosal_templates["R_AntFrontal"],
#         callosal_templates["Callosum_midsag"]],
#     'cross_midline': True
# }

# ##########################################################################
# # Tracking:
# # ---------
# # Streamlines are generate using DTI and a deterministic tractography
# # algorithm. For speed, we seed only within the waypoint ROIs for each bundle.
# #
# # .. note::
# #   By default tractography:
# #
# #   - Will identify streamlines with lengths between 10 mm and 1 m, with
# #     turning angles of less than 30 degrees.
# #
# #   - Is seeded with a single seed in each voxel on each dimension
# #
# #   - Each step is 0.5 mm
# #
# # .. note::
# #   In this example tractography results in a large number of candidate
# #   streamlines for the anterior forceps, but not many streamlines anywhere
# #   else.


# print("Tracking...")

# if not op.exists(op.join(working_dir, 'dti_streamlines.trk')):
#     seed_roi = np.zeros(img.shape[:-1])
#     for bundle in bundles:
#         for idx, roi in enumerate(bundles[bundle]["include"]):
#             warped_roi = transform_inverse_roi(
#                 roi,
#                 mapping,
#                 bundle_name=bundle)

#             nib.save(
#                 nib.Nifti1Image(warped_roi.astype(float), img.affine),
#                 op.join(working_dir, f"{bundle}_{idx+1}.nii.gz"))

#             warped_roi_img = nib.load(op.join(
#                 working_dir,
#                 f"{bundle}_{idx+1}.nii.gz"))
#             show_anatomical_slices(
#                 warped_roi_img.get_fdata(),
#                 f'warped {bundle}_{idx+1} ROI')

#             # Add voxels that aren't there yet:
#             seed_roi = np.logical_or(seed_roi, warped_roi)

#     seed_roi_img = nib.Nifti1Image(seed_roi.astype(float), img.affine)
#     nib.save(seed_roi_img, op.join(working_dir, 'seed_roi.nii.gz'))

#     show_anatomical_slices(seed_roi_img.get_fdata(), 'Seed ROI')

#     tractogram = aft.track(dti_params['params'], seed_mask=seed_roi,
#                            stop_mask=FA_data, stop_threshold=0.1,
#                            directions="det", odf_model="dti")
#     save_tractogram(tractogram, op.join(working_dir, 'dti_streamlines.trk'),
#                     bbox_valid_check=False)

#     tractogram_img = density_map(tractogram, n_sls=1000)
#     nib.save(tractogram_img, op.join(working_dir,
#                                      'afq_dti_density_map.nii.gz'))
# else:
#     tractogram = load_tractogram(op.join(working_dir, 'dti_streamlines.trk'),
#                                  img)

# tractogram.to_vox()

# ##########################################################################
# # Segmentation:
# # -------------
# # In this stage, streamlines are tested for several criteria: whether the
# # probability that they belong to a bundle is larger than a threshold (set to
# # 0, per default), whether they pass through inclusion ROIs and whether they do
# # not pass through exclusion ROIs.
# #
# # .. note::
# #   By default segmentation:
# #
# #   - uses Streamlinear Registration algorithm
# #
# #   - does not clip streamlines to be between ROIs
# #
# #   - All b-values less than or equal to 50 are considered to be
# #     without diffusion weighting.
# #
# # Segmentation will result in a `fiber_group` for each bundle, which
# # contains the following keys:
# #
# #    - `sl`
# #
# #       StatefulTractogram encompassing the selected streamlines
# #
# #    - `idx`
# #
# #       indexes to selected streamlines
# #
# # .. note::
# #    Currently it is not possible to define endpoint filters for novel bundles,
# #    but this is something we expect to address. However we can run
# #    segmentation by ignoring endpoint filters. This means that additional
# #    streamlines may be included that would otherwise be excluded.


# tractogram_img = nib.load(op.join(working_dir, 'afq_dti_density_map.nii.gz'))
# show_anatomical_slices(tractogram_img.get_fdata(), 'DTI Density Map')

# print("Segmenting fiber groups...")

# segmentation = seg.Segmentation(return_idx=True,
#                                 filter_by_endpoints=False)

# segmentation.segment(bundles,
#                      tractogram,
#                      fdata=hardi_fdata,
#                      fbval=hardi_fbval,
#                      fbvec=hardi_fbvec,
#                      mapping=mapping,
#                      reg_template=MNI_T2_img)

# fiber_groups = segmentation.fiber_groups

# for bundle in bundles:
#     tractogram = StatefulTractogram(fiber_groups[bundle]['sl'].streamlines,
#                                     img,
#                                     Space.VOX)
#     tractogram.to_rasmm()
#     save_tractogram(tractogram, op.join(working_dir, f'afq_{bundle}_seg.trk'),
#                     bbox_valid_check=False)

#     tractogram_img = density_map(tractogram, n_sls=1000)
#     nib.save(tractogram_img, op.join(working_dir,
#                                      f'afq_{bundle}_seg_density_map.nii.gz'))
#     show_anatomical_slices(tractogram_img.get_fdata(),
#                            f'Segmented {bundle} Density Map')

# ##########################################################################
# # Cleaning:
# # ---------
# # Each fiber group is cleaned to exclude streamlines that are outliers in terms
# # of their trajectory and/or length.
# #
# # .. note::
# #   By default cleaning
# #
# #   - resamples streamlines to 100 points
# #
# #   - given there are more than 20 streamlines cleaining will make maximum 5
# #     attempts to prune streamlines that are:
# #
# #      - greater than 5 standard deviations from the mean Mahalanobis distance,
# #        or
# #
# #      - greather than 4 standard deviations from the mean length
# #


# print("Cleaning fiber groups...")
# for bundle in bundles:
#     print(f"Cleaning {bundle}...")
#     print(f"Before cleaning: {len(fiber_groups[bundle]['sl'])} streamlines")
#     new_fibers, idx_in_bundle = seg.clean_bundle(
#         fiber_groups[bundle]['sl'],
#         return_idx=True)
#     print(f"After cleaning: {len(new_fibers)} streamlines")

#     idx_in_global = fiber_groups[bundle]['idx'][idx_in_bundle]
#     np.save(op.join(working_dir, f'{bundle}_idx.npy'), idx_in_global)

#     tractogram = StatefulTractogram(new_fibers.streamlines,
#                                     img,
#                                     Space.VOX)
#     tractogram.to_rasmm()
#     save_tractogram(tractogram, op.join(working_dir, f'afq_{bundle}.trk'),
#                     bbox_valid_check=False)

#     tractogram_img = density_map(tractogram, n_sls=1000)
#     nib.save(tractogram_img, op.join(working_dir,
#                                      f'afq_{bundle}_density_map.nii.gz'))
#     show_anatomical_slices(tractogram_img.get_fdata(),
#                            f'Cleaned {bundle} Density Map')

# ##########################################################################
# # Visualizing a bundle and tract profile:
# # ---------------------------------------

# bundle_to_viz = SegmentedSFT({"AntFrontal": load_tractogram(
#     op.join(working_dir, f'afq_AntFrontal.trk'),
#     img, to_space=Space.VOX)}, Space.VOX)

# plotly.io.show(visualize_bundles(bundle_to_viz,
#                                  figure=visualize_volume(warped_MNI_T2_data),
#                                  shade_by_volume=FA_data,
#                                  bundle_dict=bundles))

# ##########################################################################
# # Bundle profiles:
# # ----------------
# # Streamlines are represented in the original diffusion space (`Space.VOX`) and
# # scalar properties along the length of each bundle are queried from this
# # scalar data. Here, the contribution of each streamline is weighted according
# # to how representative this streamline is of the bundle overall.
# #
# # .. note::
# #   As a sanity check the anterior forceps the tract profile is relatively
# #   symmetric?

# print("Extracting tract profiles...")
# for bundle in bundles:
#     print(f"Extracting {bundle}...")
#     tractogram = load_tractogram(op.join(working_dir, f'afq_{bundle}.trk'),
#                                  img, to_space=Space.VOX)
#     fig, ax = plt.subplots(1)
#     weights = gaussian_weights(tractogram.streamlines)
#     profile = afq_profile(FA_data, tractogram.streamlines,
#                           np.eye(4), weights=weights)
#     ax.plot(profile)
#     ax.set_title(bundle)

# plt.show()
# plt.savefig(op.join(working_dir, 'AntFrontal_tractprofile.png'))

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
