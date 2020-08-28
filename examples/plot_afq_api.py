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
# Getting tract profiles:
# -------------------------
# Running the code below triggers the full pipeline of operations leading
# to the computation of the tract profiles. Therefore, it takes a little
# while to run (about 40 minutes in a recent experiment run on a laptop).

# df = pd.read_csv(myafq.tract_profiles[0])
# for bundle in df['bundle'].unique():
#     fig, ax = plt.subplots(1)
#     ax.plot(df[(df['bundle'] == bundle)]['dti_fa'])
#     ax.set_title(bundle)

# plt.show()


##########################################################################
# Visualizations:
# -------------------------
#

bundle_html = myafq.viz_bundles(export=True)
plotly.io.show(bundle_html[0])

