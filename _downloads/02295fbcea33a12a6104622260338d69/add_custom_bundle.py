"""
=====================================================
How to add new bundles into pyAFQ (SLF 1/2/3 Example)
=====================================================

pyAFQ is designed to be customizable and extensible. This example shows how you
can customize it to define a new bundle based on a definition of waypoint and
endpoint ROIs of your design.

In this case, we add sub-bundles of the superior longitudinal fasciculus,
based on work by Sami et al [1]_.

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
import wget
import os
np.random.seed(1234)


#############################################################################
# Get dMRI data
# ---------------
# We will analyze eight subject from the Healthy Brain Network Processed Open
# Diffusion Derivatives dataset (HBN-POD2) [2]_, [3]_. We'll use a fetcher to
# get preprocessed dMRI data for eight of the >2,000 subjects in that study. The
# data gets organized into a BIDS-compatible format in the `~/AFQ_data/HBN`
# folder. These 12 subjects have very high quality data.
# The fether returns this directory as study_dir:

_, study_dir = afd.fetch_hbn_preproc([
    'NDARKP893TWU',
    'NDAREP505XAD',
    'NDARKT540ZW0',
    'NDARAG340ERT',
    'NDAREM757NBG',
    'NDARLL894HC3',
    'NDARFY525TL2',
    'NDARKV461KGZ',
    'NDARUC851WHU',
    'NDARMJ333WJM',
    'NDARJG687YYX',
    'NDARJA157YB3',
])

#############################################################################
# Get ROIs and save to disk
# --------------------------------
# The goal of this tutorial is to demostrate how to segment new pathways based
# on ROIs that are saved to disk. In principle, ROIs can be a) files created by
# the user and saved to the local disk, b) files stored somewhere on the internet
# (as is the case here) or c) Files that are accessed with a fetcher. In this
# example we download these files from the MATLAB AFQ website, but this code could
# be commented out and paths could be used to local ROIs on disk

roi_urls = ['https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/MFgL.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/MFgR.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/PaL.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/PaR.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/PrgL.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/PrgR.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/SFgL.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/SFgR.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/SLFt_roi2_L.nii.gz',
            'https://github.com/yeatmanlab/AFQ/raw/c762ca4c393f2105d4f444c44d9e4b4702f0a646/SLF123/ROIs/SLFt_roi2_R.nii.gz']

# We proceed to download the files. First, we define and create the directory
# for the template ROIs. In the code below, ``op.expanduser("~")`` expands the
# user's home directory into the full path and ``op.join`` joins these paths,
# to make the path `~/AFQ_data/SLF_ROIs/`

template_dir = op.join(
    op.expanduser("~"),
    'AFQ_data/SLF_ROIs/')
os.makedirs(template_dir, exist_ok=True)

# The `wget` Python library works like the `wget` unix command and downloads
# each file into the directory created just above.

for roi_url in roi_urls:
    wget.download(roi_url, template_dir)


#############################################################################
# Define custom `BundleDict` object
# ---------------------------------
# A `BundleDict` is a custom object that holds information about "include" and
# "exclude" ROIs, as well as endpoint ROIs, and whether the bundle crosses the
# midline. In this case, the ROIs are all defined in the MNI template space that
# is used as the default template space in pyAFQ, but, in principle, other
# template spaces could be used. In this example, we provide paths to the ROIs
# to populate the `BundleDict`, but we could also provide already-loaded nifti
# objects, as demonstrated in other examples.

bundles = abd.BundleDict({
    "L_SLF1": {
        "include": [
            template_dir + 'SFgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,

        "mahal": {
            "clean_rounds": 20,
            "length_threshold": 4,
            "distance_threshold": 2}
    },
    "L_SLF2": {
        "include": [
            template_dir + 'MFgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,

        "mahal": {
            "clean_rounds": 20,
            "length_threshold": 4,
            "distance_threshold": 2}
    },
    "L_SLF3": {
        "include": [
            template_dir + 'PrgL.nii.gz',
            template_dir + 'PaL.nii.gz'],
        "exclude": [
            template_dir + 'SLFt_roi2_L.nii.gz'],

        "cross_midline": False,

        "mahal": {
            "clean_rounds": 20,
            "length_threshold": 4,
            "distance_threshold": 2}
    }
})

#############################################################################
# Custom bundle definitions such as the SLF or OR, and the standard BundleDict
# can be combined through addition. To get both the SLF and the standard
# bundles, we would execute the following code::
#
#     bundles = bundles + abd.default18_bd()
#
# In this case, we will skip this and generate just the SLF.

#############################################################################
# Define GroupAFQ object
# ----------------------
# HBN POD2 have been processed with qsiprep [4]_. This means that a brain mask
# has already been computed for them.
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
    segmentation_params={"parallel_segmentation": {"engine": "serial"}},
    bundle_info=bundles)

# If you want to redo different stages you can use the `clobber` method.
# The options for dependent_on are 'track' (to start over from tractography)
# or 'recog' to start over from bundle recognition. For example, to redo everying
# related  to bundle recognition: `my_afq.clobber(dependent_on='recog')`.
# This is useful when changing something about how the bundles are recognized.
# For example, the cleaning parameters.

my_afq.clobber(dependent_on='recog')

my_afq.export_all()

#############################################################################
# Visualize a montage
# ----------------------
# One way to examine the output of the pyAFQ pipeline is by creating a montage
# of images of a particular bundle across a group of participants. In the montage function
# the first input refers to a key in the bundlediect and the second gives the layout
# of the figure (eg. 3 rows 4 columns) and finally is the view.

montage = my_afq.group_montage(
    "L_SLF1", (3, 4), "Sagittal", "left", slice_pos=0.5)
montage = my_afq.group_montage(
    "L_SLF2", (3, 4), "Sagittal", "left", slice_pos=0.5)
montage = my_afq.group_montage(
    "L_SLF3", (3, 4), "Sagittal", "left", slice_pos=0.5)

#############################################################################
# Interactive bundle visualization
# --------------------------------
# Another way to examine the outputs is to export the individual bundle
# figures, which show the streamlines, as well as the ROIs used to define the
# bundle. This is an html file, which contains an interactive figure that can
# be navigated, zoomed, rotated, etc.

bundle_html = my_afq.export("all_bundles_figure")

#############################################################################
# References
# ----------
# .. [1] Romi Sagi, J.S.H. Taylor, Kyriaki Neophytou, Tamar Cohen,
#     Brenda Rapp, Kathleen Rastle, Michal Ben-Shachar.
#     White matter associations with spelling performance
#     https://doi.org/10.21203/rs.3.rs-3282349/v1
#
# .. [2] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [3] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and quality
#     controlled resource for pediatric brain white-matter research. Scientific
#     Data. 2022;9(1):1-27.
#
# .. [4] Cieslak M, Cook PA, He X, et al. QSIPrep: an integrative platform for
#     preprocessing and reconstructing diffusion MRI data. Nat Methods.
#     2021;18(7):775-778.
