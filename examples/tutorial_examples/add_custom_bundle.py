"""
=============================
Adding new bundles into pyAFQ (SLF 1/2/3 Example)
=============================

pyAFQ is designed to be customizable and extensible. This example shows how you
can customize it to define a new bundle based on a definition of waypoint and
endpoint ROIs of your design.

In this case, we add the optic radiations, based on work by Caffara et al. [1]_,
[2]_. The optic radiations (OR) are the primary projection of visual information
from the lateral geniculate nucleus of the thalamus to the primary visual
cortex. Studying the optic radiations with dMRI provides a linkage between white
matter tissue properties, visual perception and behavior, and physiological
responses of the visual cortex to visual stimulation.

We start by importing some of the components that we need for this example and
fixing the random seed for reproducibility

"""

import os.path as op
import plotly
import numpy as np
import shutil

from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd
from AFQ.definitions.image import ImageFile, RoiImage
import AFQ.utils.streamlines as aus
np.random.seed(1234)


#############################################################################
# Get dMRI data
# ---------------
# We will analyze one subject from the Healthy Brain Network Processed Open
# Diffusion Derivatives dataset (HBN-POD2) [3]_, [4]_. We'll use a fetcher to
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
# This example shows how to use ROIs that are already saved to disk

template_dir = '/Users/jyeatman/git/pyAFQ/AFQ/data/templates/SLF123/'

bundles = abd.BundleDict({
    "L_SLF1": {
        "include": [
            template_dir + 'SFgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,
    },
    "L_SLF2": {
        "include": [
            template_dir + 'MFgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,
    },
    "L_SLF3": {
        "include": [
            template_dir + 'PrgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,
    }
    

})


#############################################################################
# Custom bundle definitions such as the OR, and the standard BundleDict can be
# combined through addition. To get both the OR and the standard bundles, we
# would execute the following code::
#
#     bundles = bundles + abd.BundleDict()
#
# In this case, we will skip this and generate just the OR.


#############################################################################
# Define GroupAFQ object
# ----------------------
# HBN POD2 have been processed with qsiprep [5]_. This means that a brain mask
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
    output_dir=op.join(study_dir, "derivatives", "afq_slf"),
    brain_mask_definition=brain_mask_definition,
    tracking_params={"n_seeds": 4,
                     "directions": "prob",
                     "odf_model": "CSD",
                     "seed_mask": RoiImage()},
    clean_params={"clean_rounds":20},
    bundle_info=bundles)

# Redo everying related to bundle recognition. This is useful when changing the bundles
my_afq.clobber(dependent_on='track') 

my_afq.export_all()

#############################################################################
# Visualize a montage
# ----------------------
# One way to examine the output of the pyAFQ pipeline is by creating a montage
# of images of a particular bundle across a group of participants (or, in this
# case, the one participant that was analyzed).
#
# .. note::
#
#   The montage file is copied to the present working directory so that it gets
#   properly rendered into the web-page containing this example. It is not
#   necessary to do this when running this type of analysis.

my_afq.combine_bundle("L_SLF1")
montage = my_afq.montage("L_SLF1", (1, 1), "Axial")
shutil.copy(montage[0], op.split(montage[0])[-1])

#############################################################################
# Interactive bundle visualization
# --------------------------------
# Another way to examine the outputs is to export the individual bundle
# figures, which show the streamlines, as well as the ROIs used to define the
# bundle. This is an html file, which contains an interactive figure that can
# be navigated, zoomed, rotated, etc.

bundle_html = my_afq.export("indiv_bundles_figures")
plotly.io.show(bundle_html["NDARAA948VFH"]["L_SLF1"])

#############################################################################
# References
# ----------
# .. [1] Romi Sagi1, J.S.H. Taylor, Kyriaki Neophytou, Tamar Cohen, 
# Brenda Rapp, Kathleen Rastle, Michal Ben-Shachar.
# White matter associations with spelling performance
#
# .. [2] Caffarra S, Kanopka K, Kruper J, Richie-Halford A, Roy E, Rokem A,
#     Yeatman JD. Development of the alpha rhythm is linked to visual white
#     matter pathways and visual detection performance. bioRxiv.
#     doi:10.1101/2022.09.03.506461
#
# .. [3] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [4] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and quality
#     controlled resource for pediatric brain white-matter research. Scientific
#     Data. 2022;9(1):1-27.
#
# .. [5] Cieslak M, Cook PA, He X, et al. QSIPrep: an integrative platform for
#     preprocessing and reconstructing diffusion MRI data. Nat Methods.
#     2021;18(7):775-778.
