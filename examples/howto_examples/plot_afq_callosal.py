"""
==========================
Callosal bundles using AFQ API
==========================
An example using the AFQ API to find callosal bundles using the templates from:
http://hdl.handle.net/1773/34926
"""
# import os.path as op
# import matplotlib.pyplot as plt
# import nibabel as nib

# import plotly

# from AFQ.api.group import GroupAFQ
# import AFQ.api.bundle_dict as abd
# from AFQ.definitions.image import RoiImage
# import AFQ.data.fetch as afd

# ##########################################################################
# # Get some example data
# # ---------------------
# #
# # Retrieves `Stanford HARDI dataset <https://purl.stanford.edu/ng782rw8378>`_.
# #

# afd.organize_stanford_data(clear_previous_afq=True)

# ##########################################################################
# # Set tractography parameters (optional)
# # ---------------------
# # We make this tracking_params which we will pass to the GroupAFQ object
# # which specifies that we want 100,000 seeds randomly distributed
# # in the ROIs of every bundle.
# #
# # We only do this to make this example faster and consume less space.

# tracking_params = dict(seed_mask=RoiImage(),
#                        n_seeds=10000,
#                        random_seeds=True,
#                        rng_seed=42)

# ##########################################################################
# # Set segmentation parameters (optional)
# # ---------------------
# # We make this segmentation_params which we will pass to the GroupAFQ object
# # which specifies that we want to clip the extracted tract profiles
# # to only be between the two ROIs.
# #
# # We do this because tract profiles become less reliable as the bundles
# # approach the gray matter-white matter boundary. On some of the non-callosal
# # bundles, ROIs are not in a good position to clip edges. In these cases,
# # one can remove the first and last nodes in a tract profile.

# segmentation_params = {"clip_edges": True}

# ##########################################################################
# # Initialize a GroupAFQ object:
# # -------------------------
# #
# # We specify bundle_info as the callosal bundles only
# # (`abd.CALLOSUM_BUNDLES`). If we want to segment both the callosum
# # and the other bundles, we would pass `abd.CALLOSUM_BUNDLES + abd.BUNDLES`
# # instead. This would tell the GroupAFQ object to use bundles from both
# # the standard and callosal templates.

# myafq = GroupAFQ(
#     bids_path=op.join(afd.afq_home, 'stanford_hardi'),
#     preproc_pipeline='vistasoft',
#     bundle_info=abd.CALLOSUM_BUNDLES,
#     tracking_params=tracking_params,
#     segmentation_params=segmentation_params,
#     viz_backend_spec='plotly_no_gif')

# # Calling export all produces all of the outputs of processing, including
# # tractography, scalar maps, tract profiles and visualizations:
# myafq.export_all()


# ##########################################################################
# # Create Group Density Maps:
# # -------------------------
# #
# # pyAFQ can make density maps of streamline counts per subject/session
# # by calling `myafq.export("density_map")`. When using `GroupAFQ`, you can also
# # combine these into one file by calling `myafq.export_group_density()`.
# group_density = myafq.export_group_density()
# group_density = nib.load(group_density).get_fdata()
# fig, ax = plt.subplots(1)
# ax.matshow(
#     group_density[:, :, group_density.shape[-1] // 2, 0],
#     cmap='viridis')
# ax.axis("off")


# ##########################################################################
# # Visualizing bundles and tract profiles:
# # ---------------------------------------
# # This would run the script and visualize the bundles using the plotly
# # interactive visualization, which should automatically open in a
# # new browser window.
# bundle_html = myafq.export("all_bundles_figure")
# plotly.io.show(bundle_html["01"][0])
