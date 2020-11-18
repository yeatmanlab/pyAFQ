"""
==========================
RecoBundles80 using AFQ API
==========================

An example using the AFQ API to run recobundles with the 80 bundle atlas.


"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import plotly

from AFQ import api
import AFQ.data as afd

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
# intializing the AFQ object (which we will do next)

afd.organize_stanford_data()

##########################################################################
# Initialize an AFQ object:
# -------------------------
#
# Creates an AFQ object, that encapsulates tractometry. This object can be
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
#    The first time intializing the AFQ object will download necessary
#    waypoint regions of interest (ROIs) templates into AFQ data directory:
#
# - Human corpus callosum templates: ``AFQ_data/callosum_templates/``
#
#   see https://digital.lib.washington.edu/researchworks/handle/1773/34926
#
# - Tract probability maps: ``AFQ_data/templates/``
#
#   see https://figshare.com/articles/Tract_probability_maps_for_automated_fiber_quantification/6270434
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
#    We specify seg_algo as reco80 in segmentation_params. This tells the AFQ
#    object to perform RecoBundles using the 80 bundles atlas in the
#    segmentation step.

myafq = api.AFQ(bids_path=op.join(afd.afq_home,
                                  'stanford_hardi'),
                dmriprep='vistasoft',
                segmentation_params={"seg_algo": "reco80"})

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

bundle_html = myafq.viz_bundles(export=True, n_points=50)
plotly.io.show(bundle_html[0])
