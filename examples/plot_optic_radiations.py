"""
=============================
Plotting the Optic Radiations
=============================

pyAFQ is designed to be customizable. This example shows how
you can customize it to define a new bundle based
on both waypoint ROIs of your design, as well as endpoint
ROIs of your design.

For now, this is a hypothetical example, as we do not yet
provide these ROIs as part of the software.
"""

import os.path as op
from AFQ import api
import AFQ.data as afd
from AFQ.definitions.mask import LabelledMaskFile, RoiMask

afd.organize_stanford_data(clear_previous_afq=True)

or_rois = afd.read_or_templates()

bundles = api.BundleDict({
    "L_OR": {
        "ROIs": [or_rois["left_OR_1"],
                 or_rois["left_OR_2"],
                 or_rois["left_OP_MNI"],
                 or_rois["left_TP_MNI"],
                 or_rois["left_pos_thal_MNI"]],
        "rules": [True, True, False, False, False],
        "cross_midline": False,
        "uid": 1
        },
    "R_OR": {
        "ROIs": [or_rois["right_OR_1"],
                 or_rois["right_OR_2"],
                 or_rois["right_OP_MNI"],
                 or_rois["right_TP_MNI"],
                 or_rois["right_pos_thal_MNI"]],
        "rules": [True, True, False, False, False],
        "cross_midline": False,
        "uid": 2
        }
    }
)

endpoint_info = {
    "L_OR": {
        "startpoint": or_rois['left_thal_MNI'],
        "endpoint": or_rois['left_V1_MNI']},
    "R_OR": {
        "startpoint": or_rois['right_thal_MNI'],
        "endpoint": or_rois['right_V1_MNI']}}

brain_mask = LabelledMaskFile("seg",
                              {"scope": "freesurfer"},
                              exclusive_labels=[0])

my_afq = api.AFQ(
    bids_path=op.join(afd.afq_home,
                     'stanford_hardi'),
    brain_mask=brain_mask,
    tracking_params={"n_seeds": 3,
                     "directions": "prob",
                     "odf_model": "CSD",
                     "seed_mask": RoiMask()},
    segmentation_params=dict(endpoint_info=endpoint_info),
    bundle_info=bundles)

my_afq.export_all()
