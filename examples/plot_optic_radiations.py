"""
=============================
Plotting the Optic Radiations
=============================

pyAFQ is designed to be customizable. This example shows how
you can customize it to define a new bundle based
on both waypoint ROIs of your design, as well as endpoint
ROIs of your design.

In these example, we run pyAFQ with both the custom ROIs and
the default waypoint ROIs.
"""

import os.path as op
from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.definitions.image import LabelledImageFile, RoiImage
import AFQ.utils.streamlines as aus

afd.organize_stanford_data(clear_previous_afq=True)

or_rois = afd.read_or_templates()

bundles = abd.BundleDict({
    "L_OR": {
        "include": [
            or_rois["left_OR_1"],
            or_rois["left_OR_2"]],
        "exclude": [
            or_rois["left_OP_MNI"],
            or_rois["left_TP_MNI"],
            or_rois["left_pos_thal_MNI"]],
        "start": or_rois['left_thal_MNI'],
        "end": or_rois['left_V1_MNI'],
        "cross_midline": False,
    },
    "R_OR": {
        "include": [
            or_rois["right_OR_1"],
            or_rois["right_OR_2"]],
        "exclude": [
            or_rois["right_OP_MNI"],
            or_rois["right_TP_MNI"],
            or_rois["right_pos_thal_MNI"]],
        "start": or_rois['right_thal_MNI'],
        "end": or_rois['right_V1_MNI'],
        "cross_midline": False
    }
})

# combine custom ROIs with default BundleDict ROIs
bundles = bundles + abd.BundleDict()

brain_mask_definition = LabelledImageFile(
    suffix="seg",
    filters={"scope": "freesurfer"},
    exclusive_labels=[0])

my_afq = GroupAFQ(
    bids_path=op.join(
        afd.afq_home,
        'stanford_hardi'),
    brain_mask_definition=brain_mask_definition,
    tracking_params={"n_seeds": 3,
                     "directions": "prob",
                     "odf_model": "CSD",
                     "seed_mask": RoiImage()},
    bundle_info=bundles)

my_afq.export_all()

if len(aus.SegmentedSFT.fromfile(my_afq.export("clean_bundles")[
        "01"]).get_bundle("L_OR").streamlines) > 1:
    #  create bundle montage and bundle combination
    #  across subject/session in MNI
    my_afq.montage("L_OR", (1, 1), "Axial")
    my_afq.combine_bundle("L_OR")
else:
    raise ValueError("No L_OR found")
