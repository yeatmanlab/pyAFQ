"""
==========================
AFQ API
==========================

An example using the AFQ API


"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import plotly
import pandas as pd

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd

##########################################################################
# Get some example data
# ---------------------
#
# Retrieves High angular resolution diffusion imaging (HARDI) dataset from
# Stanford's Vista Lab
#
#   see https://purl.stanford.edu/ng782rw8378 for details on dataset.
#
# The data for the first subject and first session are downloaded locally
# (by default into the users home directory) under:
#
#   ``.dipy/stanford_hardi/``
#
# Anatomical data (``anat``) and Diffusion-weighted imaging data (``dwi``) are
# then extracted, formatted to be BIDS compliant, and placed in the AFQ
# data directory (by default in the users home directory) under:
#
#   ``AFQ_data/stanford_hardi/``
#
# This data represents the required preprocessed diffusion data necessary for
# intializing the GroupAFQ object (which we will do next)
#
# The clear_previous_afq is used to remove any previous runs of the afq object
# stored in the ~/AFQ_data/stanford_hardi/ BIDS directory. Set it to None if
# you want to use the results of previous runs.

afd.organize_stanford_data(clear_previous_afq="track")

##########################################################################
# Set tractography parameters (optional)
# ---------------------
# We make this tracking_params which we will pass to the GroupAFQ object
# which specifies that we want 25,000 seeds randomly distributed
# in the white matter.
#
# We only do this to make this example faster and consume less space.

tracking_params = dict(n_seeds=25000,
                       random_seeds=True,
                       rng_seed=42,
                       trx=True)

##########################################################################
# Initialize a GroupAFQ object:
# -------------------------
#
# Creates a GroupAFQ object, that encapsulates tractometry. This object can be
# used to manage the entire AFQ pipeline, including:
#
# - Tractography
# - Registration
# - Segmentation
# - Cleaning
# - Profiling
# - Visualization
#
# In this example we will load the subjects session data from the previous step
# using the default AFQ parameters.
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
# Calculating DTI FA (Diffusion Tensor Imaging Fractional Anisotropy)
# ------------------------------------------------------------------
# The GroupAFQ object has a method called `export`, which allows the user
# to calculate various derived quantities from the data.
#
# For example, FA can be computed using the DTI model, by explicitly
# calling `myafq.export("dti_fa")`. This triggers the computation of DTI
# parameters for all subjects in the dataset, and stores the results in
# the AFQ derivatives directory. In addition, it calculates the FA
# from these parameters and stores it in a different file in the same
# directory.
#
# .. note::
#
#    The AFQ API computes quantities lazily. This means that DTI parameters
#    are not computed until they are required. This means that the first
#    line below is the one that requires time.
#
# The result of the call to `export` is a dictionary, with the subject
# IDs as keys, and the filenames of the corresponding files as values.
# This means that to extract the filename corresponding to the FA of the first
# subject, we can do:

FA_fname = myafq.export("dti_fa")["01"]

# We will then use `nibabel` to load the deriviative file and retrieve the
# data array.

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
# Recognizing the bundles and calculating act profiles:
# -----------------------------------------------------
# Typically, users of pyAFQ are interested in calculating not only an overall
# map of the FA, but also the major white matter pathways (or bundles) and
# tract profiles of tissue properties along their length. To trigger the
# entire pyAFQ pipeline, users can call the `export_all` method:
#
# .. note::
#    Running the code below triggers the full pipeline of operations
#    leading to the computation of the tract profiles. Therefore, it
#    takes a little while to run (about 40 minutes, typically).

myafq.export_all()

##########################################################################
# Visualizing the bundles and calculating act profiles:
# -----------------------------------------------------
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
