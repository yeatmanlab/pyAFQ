"""
===============================================================
How to add new bundles into pyAFQ (Acoustic Radiations Example)
===============================================================

pyAFQ is designed to be customizable and extensible. This example shows how you
can customize it to define a new bundle based on a definition of waypoint and
endpoint ROIs of your design. In this case, we add the acoustic radiations.

We start by importing some of the components that we need for this example and
fixing the random seed for reproducibility

"""

import os.path as op
import plotly
import numpy as np

from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.definitions.image import ImageFile, RoiImage
np.random.seed(1234)


#############################################################################
# Get dMRI data
# ---------------
# We will analyze one subject from the Healthy Brain Network Processed Open
# Diffusion Derivatives dataset (HBN-POD2) [1]_, [2]_. We'll use a fetcher to
# get preprocessed dMRI data for one of the >2,000 subjects in that study. The
# data gets organized into a BIDS-compatible format in the `~/AFQ_data/HBN`
# folder:

study_dir = afd.fetch_hbn_preproc(["NDARAA948VFH"])[1]

#############################################################################
# Define custom `BundleDict` object
# --------------------------------
# The `BundleDict` object holds information about "include" and "exclude" ROIs,
# as well as endpoint ROIS, and whether the bundle crosses the midline. In this
# case, the ROIs are all defined in the MNI template space that is used as the
# default template space in pyAFQ, but, in principle, other template spaces
# could be used.
#
# The ROIs for the case can be downloaded using a custom fetcher which saves
# the ROIs to a folder and creates a dictionary of paths to the ROIs:

ar_rois = afd.read_ar_templates()

bundles = abd.BundleDict({
    "Left Acoustic Radiation": {
        "start": ar_rois["AAL_Thal_L"],
        "end": ar_rois["AAL_TempSup_L"],
        "cross_midline": False,
    },
    "Right Acoustic Radiation": {
        "start": ar_rois["AAL_Thal_R"],
        "end": ar_rois["AAL_TempSup_R"],
        "cross_midline": False
    }
})


#############################################################################
# Define GroupAFQ object
# ----------------------
# HBN POD2 have been processed with qsiprep [3]_. This means that a brain mask
# has already been computer for them. As you can see in other examples, these
# data also have a mapping calculated for them, which can also be incorporated
# into processing. However, in this case, we will let pyAFQ calculate its own
# SyN-based mapping so that the `combine_bundle` method can be used below to
# create a montage visualization.
#
# For tractography, we use CSD-based probabilistic tractography seeding
# extensively (`n_seeds=4` means 81 seeds per voxel!), but only within the ROIs
# and not throughout the white matter. This is controlled by passing
# `"seed_mask": RoiImage()` in the `tracking_params` dict. The custom bundles
# are passed as `bundle_info=bundles`. The call to `my_afq.export_all()`
# initiates the pipeline.

brain_mask_definition = ImageFile(
    suffix="mask",
    filters={'desc': 'brain',
             'space': 'T1w',
             'scope': 'qsiprep'})

my_afq = GroupAFQ(
    bids_path=study_dir,
    preproc_pipeline="qsiprep",
    participant_labels=["NDARAA948VFH"],
    output_dir=op.join(study_dir, "derivatives", "afq_ar"),
    brain_mask_definition=brain_mask_definition,
    tracking_params={"n_seeds": 4,
                     "directions": "prob",
                     "odf_model": "CSD",
                     "seed_mask": RoiImage(use_endpoints=True)},
    bundle_info=bundles)

my_afq.export_all()

#############################################################################
# Interactive bundle visualization
# --------------------------------
# Another way to examine the outputs is to export the individual bundle
# figures, which show the streamlines, as well as the ROIs used to define the
# bundle. This is an html file, which contains an interactive figure that can
# be navigated, zoomed, rotated, etc.

bundle_html = my_afq.export("indiv_bundles_figures")
plotly.io.show(bundle_html["NDARAA948VFH"]["Left Acoustic Radiation"])

#############################################################################
# References
# ----------
# .. [1] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [2] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and quality
#     controlled resource for pediatric brain white-matter research. Scientific
#     Data. 2022;9(1):1-27.
#
# .. [3] Cieslak M, Cook PA, He X, et al. QSIPrep: an integrative platform for
#     preprocessing and reconstructing diffusion MRI data. Nat Methods.
#     2021;18(7):775-778.
