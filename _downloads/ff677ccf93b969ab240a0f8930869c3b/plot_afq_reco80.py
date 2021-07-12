"""
==========================
RecoBundles80 using AFQ API
==========================

An example using the AFQ API to run recobundles with the
`80 bundle atlas <https://figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883>`_.

"""
import os.path as op

import plotly

from AFQ import api
import AFQ.data as afd

##########################################################################
# Get some example data
# ---------------------
#
# Retrieves `Stanford HARDI dataset <https://purl.stanford.edu/ng782rw8378>`_.
#

afd.organize_stanford_data(clear_previous_afq=True)

##########################################################################
# Set tractography parameters (optional)
# ---------------------
# We make this tracking_params which we will pass to the AFQ object
# which specifies that we want 50,000 seeds randomly distributed
# in the white matter.
#
# We only do this to make this example faster and consume less space.

tracking_params = dict(n_seeds=50000,
                       random_seeds=True,
                       rng_seed=42)

##########################################################################
# Initialize an AFQ object:
# -------------------------
#
# We specify seg_algo as reco80 in segmentation_params. This tells the AFQ
# object to perform RecoBundles using the 80 bundles atlas in the
# segmentation step.

myafq = api.AFQ(bids_path=op.join(afd.afq_home,
                                  'stanford_hardi'),
                dmriprep='vistasoft',
                segmentation_params={"seg_algo": "reco80"},
                tracking_params=tracking_params)

##########################################################################
# Visualizing bundles and tract profiles:
# ---------------------------------------
# This would run the script and visualize the bundles using the plotly
# interactive visualization, which should automatically open in a
# new browser window.
bundle_html = myafq.all_bundles_figure
plotly.io.show(bundle_html["01"])
