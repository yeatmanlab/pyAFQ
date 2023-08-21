"""
================================
Delineating cerebellar peduncles
================================

The cerebellar peduncles are white matter tracts that connect the cerebellum to
the brainstem and cortex. In this example, we show how to delineate the
cerebellar peduncles using a subject from the Healthy Brain Network dataset.

This how-to will focus on the definition of the Cerebellar Peduncles (CP) based
on [1]_, [2].

"""

import AFQ.data.fetch as afd
from AFQ.api import make_bundle_dict

cp_rois = afd.read_cp_templates()
cp_bundles = abd.BundleDict(
            {
                "ICP_L": {
                    "include": [
                        cp_rois["ICP_L_inferior_prob"],
                        cp_rois["ICP_L_superior_prob"],
                    ],
                    "cross_midline": False,
                },
                "ICP_R": {
                    "include": [
                        cp_rois["ICP_R_inferior_prob"],
                        cp_rois["ICP_R_superior_prob"],
                    ],
                    "cross_midline": False,
                },
                "MCP_L": {
                    "include": [
                        cp_rois["MCP_L_inferior_prob"],
                        cp_rois["MCP_R_superior_prob"],
                    ],
                    "exclude": [
                        cp_rois["SCP_L_inter_prob"],
                    ],
                    "cross_midline": True,
                },
                "MCP_R": {
                    "include": [
                        cp_rois["MCP_R_inferior_prob"],
                        cp_rois["MCP_L_superior_prob"],
                    ],
                    "exclude": [
                        cp_rois["SCP_R_inter_prob"],
                    ],
                    "cross_midline": True,
                },
                "SCP_L": {
                    "include": [
                        cp_rois["SCP_L_inferior_prob"],
                        cp_rois["SCP_L_inter_prob"],
                        cp_rois["SCP_R_superior_prob"],
                    ],
                    "exclude": [
                        cp_rois["SCP_L_superior_prob"],
                    ],
                    "cross_midline": True,
                },
                "SCP_R": {
                    "include": [
                        cp_rois["SCP_R_inferior_prob"],
                        cp_rois["SCP_R_inter_prob"],
                        cp_rois["SCP_L_superior_prob"],
                    ],
                    "exclude": [
                        cp_rois["SCP_R_superior_prob"],
                    ],
                    "cross_midline": True,
                },
            }
        )

cp_afq = GroupAFQ(
            bids_path=bids_path,
            preproc_pipeline="qsiprep",
            brain_mask_definition=brain_mask_definition,
            # mapping_definition=mapping_definition,
            tracking_params={
                "n_seeds": 4,
                "directions": "prob",
                "odf_model": "CSD",
                "seed_mask": RoiImage(),
            },
            bundle_info=cp_bundles,
        )


"""

References
----------
.. [1] S. Jossinger, A. Sares, A. Zislis, D. Sury, V. Gracco, M. Ben-Shachar (2022)
       White matter correlates of sensorimotor synchronization in persistent
       developmental stuttering, Journal of Communication Disorders, 95.

.. [2] S. Jossinger, M. Yablonski, O. Amir, M. Ben-Shachar (2023). The
       contributions of the cerebellar peduncles and the frontal aslant tract
       in mediating speech fluency. Neurobiology of Language 2023;
       doi: https://doi.org/10.1162/nol_a_00098

"""