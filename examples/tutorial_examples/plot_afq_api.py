"""
=========================================================
1 Using the pyAFQ Application Programming Interface (API)
=========================================================

In this first tutorial, we will introduce the use of the pyAFQ Application
Programming Interface (API). This API is designed to be flexible and
extensible, and allows users to customize the pipeline to their needs.

The pipeline executes the following steps:

1. Tractography
2. Registration to a template.
3. Bundle segmentation + cleaning
4. Tissue property modeling
5. Tract profile calculation
6. Bundle visualization


We will not go into these in a lot more detail here, but you can read more
about them in other examples and tutorials.
"""

##########################################################################
# Importing libraries that we will use
# ------------------------------------
# The tutorial will show-case some of the functionality and outputs of
# pyAFQ. We will use the following libraries in this tutorial:
#
# - `AFQ.api.group.GroupAFQ` to run the pipeline. As you will see below, this
#   object is the workhorse of pyAFQ, and used to define the pipeline and
#   execute it.
# - `AFQ.data.fetch` is pyAFQ's data management module. Here, we use it to
#   download example data and to locate the data in the user's home directory.
# - `os.path` is used to specify file paths
# - `matplotlib.pyplot` used to visualize the results with 2D plots and
#   figures.
# - `nibabel` is used to load resulting data derivatives.
# - `plotly` is used to visualize the results with 3D web-based visualizations.
# - `pandas` is used to read the results into a table.
#
#
from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd

import os.path as op
import matplotlib.pyplot as plt
import nibabel as nib
import plotly
import pandas as pd


##########################################################################
# Download example data and organize it
# -------------------------------------
# Before we can start running the software we need to download some data.
# We will use an example dataset that contains high angular resolution
# diffusion MR imaging (HARDI) from one subject. This data was acquired in
# the Stanford Center for Cognitive and Neurobiological Imaging (CNI). You can
# learn more about this data in [Rokem2015]_.
#
# The following code downloads the data and organizes it in a BIDS compliant
# format, and places it in the `AFQ_data` directory, which is typically
# located in your home directory under:
#
#   ``~/AFQ_data/stanford_hardi/``
#
# The `clear_previous_afq` argument is used to remove the outputs from previous
# runs of pyAFQ that may be stored in the AFQ_data/stanford_hardi/ BIDS
# directory. Set it to None if you want to use all the results of previous
# runs of pyAFQ. Here, we set it to clear all of the the outputs past the
# tractography stage (which tends to be time-consuming).

afd.organize_stanford_data(clear_previous_afq="track")

##########################################################################
# Set tractography parameters
# ---------------------------
# The pyAFQ API allows us to define the parameters used for tractography.
# If you do not set these parameters, a reasonable set of defaults is used:
# probabilistic tractography using constrained spherical decovolution with one
# seed in every white-matter voxel. Here, we create an alternative setting,
# which uses csd-based probabilistic tractography, but with only 25,000 seeds
# randomly distributed in the white matter. This is a much smaller number of
# seeds than the default, and will result in a much faster run-time. However,
# it will also result in less accurate tractography, and may result in missing
# some of the smaller tracts. In this case, we only do this to make this
# example faster and consume less space.

tracking_params = dict(n_seeds=25000,
                       random_seeds=True,
                       rng_seed=42,
                       trx=True)

##########################################################################
# Initialize a GroupAFQ object:
# -------------------------
#
# The following code creates the GroupAFQ object, which manages all of the
# data transformations and computations conducted by the software, based on
# its initial configuration, which we set up below.
#
# .. note::
#
#    The first time intializing the GroupAFQ object will download necessary
#    waypoint regions of interest (ROIs) templates into AFQ data directory:
#
# - Human corpus callosum templates: ``AFQ_data/callosum_templates/``
#
#   see https://digital.lib.washington.edu/researchworks/handle/1773/34926
#
# - Tract probability maps: ``AFQ_data/templates/``
#
#   see https://figshare.com/articles/Tract_probability_maps_for_automated_fiber_quantification/6270434  # noqa
#
# These waypoints ROIs will used to identify the desired white matter tracts.
#
# This will also create an output folder for the corresponding AFQ derivatives
# in the AFQ data directory: ``AFQ_data/stanford_hardi/derivatives/afq/``
#
# To initialize this object we will pass in the path location to our BIDS
# compliant data.
#
# .. note::
#
#    As noted above, the Stanford HARDI data contains anatomical and
#    diffusion weighted imaging (dwi) data. In this example, we are interested
#    in the vistasoft dwi. For our dataset the `dmriprep` is optional, but
#    we have included it to make the initialization more explicit.
#
# .. note::
#
#    We will also be using plotly to generate an interactive visualization.
#    So we will specify plotly_no_gif as the visualization backend.

myafq = GroupAFQ(
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    preproc_pipeline='vistasoft',
    tracking_params=tracking_params,
    viz_backend_spec='plotly_no_gif')

##########################################################################
# Reading in DTI FA (Diffusion Tensor Imaging Fractional Anisotropy)
# ------------------------------------------------------------------
# The GroupAFQ object holds a table with file names to various data derivatives.
#
# For example, the file where the FA computed from DTI is stored can be
# retrieved by inspecting the ``dti_fa`` property. The measures are stored
# in a series, and since we only have one subject and one session we will
# access the first (and only) file name from the example data.
#
# .. note::
#
#    The AFQ API computes quantities lazily. This means that DTI parameters
#    are not computed until they are required. This means that the first
#    line below is the one that requires time.
#
# We will then use `nibabel` to load the deriviative file and retrieve the
# data array.

FA_fname = myafq.export("dti_fa")["01"]
FA_img = nib.load(FA_fname)
FA = FA_img.get_fdata()

##########################################################################
# Visualize the result with Matplotlib
# -------------------------------------
# At this point `FA` is an array, and we can use standard Python tools to
# visualize it or perform additional computations with it.
#
# In this case we are going to take an axial slice halfway through the
# FA data array and plot using a sequential color map.
#
# .. note::
#
#    The data array is structured as a xyz coordinate system.

fig, ax = plt.subplots(1)
ax.matshow(FA[:, :, FA.shape[-1] // 2], cmap='viridis')
ax.axis("off")

##########################################################################
# Visualizing bundles and tract profiles:
# ---------------------------------------
# The pyAFQ API provides several ways to visualize bundles and profiles.
#
# First, we will run a function that exports an html file that contains
# an interactive visualization of the bundles that are segmented.
#
# .. note::
#    By default we resample a 100 points within a bundle, however to reduce
#    processing time we will only resample 50 points.
#
# Once it is done running, it should pop a browser window open and let you
# interact with the bundles.
#
# .. note::
#    Running the code below triggers the full pipeline of operations
#    leading to the computation of the tract profiles. Therefore, it
#    takes a little while to run (about 40 minutes, typically).
#
# .. note::
#    You can hide or show a bundle by clicking the legend, or select a
#    single bundle by double clicking the legend. The interactive
#    visualization will also all you to pan, zoom, and rotate.

bundle_html = myafq.export("all_bundles_figure")
plotly.io.show(bundle_html["01"][0])

##########################################################################
# We can also visualize the tract profiles in all of the bundles. These
# plots show both FA (left) and MD (right) layed out anatomically.
# To make this plots, it is required that you install with
# `pip install pyAFQ[plot]` so that you have the necessary dependencies.
#

fig_files = myafq.export("tract_profile_plots")["01"]

##########################################################################
# .. figure:: {{ fig_files[0] }}
#

##########################################################################
# We can check the number of streamlines per bundle, to make sure
# every bundle is found with a reasonable amount of streamlines.

bundle_counts = pd.read_csv(myafq.export("sl_counts")["01"], index_col=[0])
for ind in bundle_counts.index:
    #  few streamlines are found for these bundles in this subject
    if ind == "FP" or "VOF" in ind:
        threshold = 0
    else:
        threshold = 20
    if bundle_counts["n_streamlines"][ind] < threshold:
        raise ValueError((
            "Small number of streamlines found "
            f"for bundle(s):\n{bundle_counts}"))


##########################################################################
# References
# ----------
#
# .. [Rokem2015] Ariel Rokem, Jason D Yeatman, Franco Pestilli, Kendrick
#    N Kay, Aviv Mezer, Stefan van der Walt, Brian A Wandell. Evaluating the
#    accuracy of diffusion MRI models in white matter. PLoS One, 10(4),
#    e0123272, 2015.