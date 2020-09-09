"""
==========================
AFQ API
==========================

An example using the AFQ API


"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import plotly

from AFQ import api
import AFQ.data as afd


##########################################################################
# Get some example data
# ---------------------

afd.organize_stanford_data()
base_dir = op.join(op.expanduser('~'), 'AFQ_data', 'stanford_hardi')

##########################################################################
# Initialize an AFQ object:
# ------------------------

myafq = api.AFQ(bids_path=op.join(afd.afq_home,
                                  'stanford_hardi'),
                dmriprep='vistasoft',
                viz_backend="plotly")

##########################################################################
# Reading in DTI FA
# -----------------
# The AFQ object holds a table with file-names to various data derivatives.
# For example, this is where the FA computed from DTI is stored.
#
# .. note::
#
#    The AFQ API computes quantities lazily. This means that DTI parameters
#    are  not computed until they are required. This means that the first
#    line below is the one that requires time.

FA_fname = myafq.dti_fa[0]
FA = nib.load(FA_fname).get_fdata()


##########################################################################
# Visualize the result with Matplotlib
# -------------------------------------
# At this point `FA` is an array, and we can use standard Python tools to
# visualize it or perform additional computations with it:
fig, ax = plt.subplots(1)
ax.matshow(FA[:, :, FA.shape[-1] // 2], cmap='viridis')
ax.axis("off")


##########################################################################
# Visualizing bundles and tract profiles:
# ---------------------------------------
# The pyAFQ API provides several ways to visualize bundles and profiles.
# First, we will run a function that exports an html file that contains
# an interactive visualization of the bundles that are segmented. Once
# it is done running, it should pop a browser window open and let you
# interact with the bundles.
#
# .. note::
#    Running the code below triggers the full pipeline of operations
#    leading to the computation of the tract profiles. Therefore, it
#    takes a little while to run (about 40 minutes, typically).

bundle_html = myafq.viz_bundles(export=True, n_points=50)
plotly.io.show(bundle_html[0])

##########################################################################
# We can also visualize the tract profiles in all of the bundles:
#

myafq.plot_tract_profiles()
fig_files = myafq.data_frame['tract_profiles_viz'][0]

##########################################################################
# .. figure:: {{ fig_files[0] }}
#
