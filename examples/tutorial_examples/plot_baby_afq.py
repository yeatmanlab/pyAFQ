# """
# ==============================
# Pediatric Profiles:
# ==============================

# The following is an example of tractometry for pediatric bundles.

# .. note::

#   This example and resulting pyAFQ support for pediatric bundles was
#   inspired by and largely due to the work of Grotheer et al. [Grotheer2021]_.

#   This example uses the Yeatman et al. waypoint ROI approach, first
#   described in [Yeatman2012]_ and further elaborated in [Yeatman2014]_.

# """
# import os.path as op
# import plotly

# from AFQ.api.group import GroupAFQ
# import AFQ.api.bundle_dict as abd
# import AFQ.data.fetch as afd

# from AFQ.definitions.image import RoiImage, ImageFile

# ##########################################################################
# # Initialize a GroupAFQ object:
# # -------------------------
# #
# # .. note::
# #   This example assumes:
# #
# #   - A single subject and session
# #
# #   - Subject's session data has been downloaded into
# #
# #     `AFQ_data\\study\\derivatives`
# #
# #   - Subject's session data has been made BIDS compliant
# #
# #     see https://bids.neuroimaging.io
# #
# # .. note::
# #
# #   While it is possible to run tractography and segmentation for pediatric
# #   dMRI data with pyAFQ, we recommend using a custom tractography pipeline
# #   and only using pyAFQ for segmentation as shown in:
# #
# #   - https://github.com/bloomdt-uw/babyafq/blob/main/mrtrix_pipeline.ipynb
# #   - https://github.com/bloomdt-uw/babyafq/blob/main/pybabyafq.ipynb


# myafq = GroupAFQ(  # ==== BIDS parameters ====
#     bids_path=op.join(afd.afq_home, "study"),
#     preproc_pipeline="derivatives",
#     # ===== Registration parameters ====
#     min_bval=1000,
#     max_bval=1000,
#     reg_template=afd.read_pediatric_templates(
#     )["UNCNeo-withCerebellum-for-babyAFQ"],
#     reg_subject="b0",
#     brain_mask_definition=ImageFile(
#         suffix="brainmask", filters={"scope": "derivatives"}),
#     # ==== Bundle parameters ====
#     bundle_info=abd.PediatricBundleDict(),
#     # ==== Compute parameters ====
#     force_recompute=True,
#     # ==== Tracking parameters ====
#     tracking_params={
#         "seed_mask": RoiImage(),
#         "stop_threshold": 0.1},
#     # ==== Segmentation parameters ====
#     segmentation_params={
#         "filter_by_endpoints": False},
# )

# ##########################################################################
# # Visualizing bundles:
# # --------------------

# plotly.io.show(myafq.export("all_bundles_figure").values()[0][0])

# ##########################################################################
# # References:
# # -------------------------
# # .. [Grotheer2021] Mareike Grotheer, Mona Rosenke, Hua Wu, Holly Kular,
# #                   Francesca R. Querdasi, Vaidehi Natu, Jason D. Yeatman,
# #                   alanit Grill-Spector, "Catch me if you can: Least
# #                   myelinated white matter develops fastest during early
# #                   infancy", bioRxiv
# #
# # .. [Yeatman2012] Jason D Yeatman, Robert F Dougherty, Nathaniel J Myall,
# #                  Brian A Wandell, Heidi M Feldman, "Tract profiles of
# #                  white matter properties: automating fiber-tract
# #                  quantification", PloS One, 7: e49790
# #
# # .. [Yeatman2014] Jason D Yeatman, Brian A Wandell, Aviv Mezer Feldman,
# #                  "Lifespan maturation and degeneration of human brain white
# #                  matter", Nature Communications 5: 4932
