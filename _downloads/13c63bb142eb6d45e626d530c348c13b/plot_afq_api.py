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

myafq = api.AFQ(op.join(afd.afq_home,
                        'stanford_hardi',
                        'derivatives',
                        'dmriprep'),
                sub_prefix='sub')

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
# The computation below is quite time-consuming. For this reason, we have
# commented it out for now. If you want to run it through, uncomment then next
# few lines and go get a cup of coffee:
#
# .. note::
#     Note that because of a quirk in the way that brain segmentation was
#     generated in this data, some bundles will not be detected using the
#     detected using the default values. For example, the corticospinal tracts
#     on both sides will not be properly detected because the waypoint ROIs for
#     these tracts fall into parts of the white matter that were not segmented.

# df = pd.read_csv(myafq.tract_profiles[0])
# for bundle in df['bundle'].unique():
#     fig, ax = plt.subplots(1)
#     ax.plot(df[(df['scalar'] == "dti_fa")
#             & (df['bundle'] == bundle)]['profiles'])
#     ax.set_title(bundle)

plt.show()
