import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.api.group import GroupAFQ
from AFQ.definitions.image import RoiImage
import AFQ.utils.streamlines as aus

import os.path as op

afd.organize_stanford_data(clear_previous_afq=False)

bundle_names = ["ARC_L", "ARC_R", "pARC_L", "pARC_R", "VOF_L", "VOF_R"]

myafq = GroupAFQ(
    op.join(afd.afq_home, 'stanford_hardi'),
    bundle_info=abd.BundleDict(bundle_names),
    preproc_pipeline='vistasoft',
    tracking_params={
        "n_seeds": 50000,
        "random_seeds": True,
        "seed_mask": RoiImage(use_waypoints=True, use_endpoints=True),
    },
    clean_params={"distance_threshold": 3,
                  "length_threshold": 5, "clean_rounds": 20}
)


for b_name in bundle_names:
    b_len = len(aus.SegmentedSFT.fromfile(myafq.export("clean_bundles")[
        "01"]).get_bundle(b_name))

    if b_len < 1:
        raise ValueError(f"{b_name} not found")

myafq.export("all_bundles_figure")["01"][0]
