"""
====================================
How to segment out only some bundles
====================================

The pyAFQ software can be configured to find all of its default set of white
matter pathways, or bundles. Alternatively, it can be configured to find only
some bundles. This example shows how to track and recognize only certain
bundles that you are interested in, though note that ARC, pARC and VOF are all
also part of the set of bundles that are segmented per default.

"""

import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.api.group import GroupAFQ
from AFQ.definitions.image import RoiImage
import AFQ.utils.streamlines as aus

import os.path as op

afd.organize_stanford_data(clear_previous_afq="track")

bundle_names = [
    "Left Arcuate", "Right Arcuate",
    "Left Posterior Arcuate", "Right Posterior Arcuate",
    "Left Vertical Occipital", "Right Vertical Occipital"]
bundle_dict = abd.default18_bd()[bundle_names]


myafq = GroupAFQ(
    op.join(afd.afq_home, 'stanford_hardi'),
    bundle_info=bundle_dict,
    preproc_pipeline='vistasoft',
    tracking_params={
        "n_seeds": 50000,
        "random_seeds": True,
        "seed_mask": RoiImage(use_waypoints=True, use_endpoints=True),
    }
)

for b_name in bundle_names:
    b_len = len(aus.SegmentedSFT.fromfile(myafq.export("bundles")[
        "01"]).get_bundle(b_name))

    if b_len < 1:
        raise ValueError(f"{b_name} not found")

myafq.export("all_bundles_figure")["01"][0]
