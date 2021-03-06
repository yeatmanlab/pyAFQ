"""
==========================
Callosal bundles using AFQ API
==========================
An example using the AFQ API to find callosal bundles using the templates from:
http://hdl.handle.net/1773/34926
"""
import os.path as op

import plotly

from AFQ import api
from AFQ.definitions.mask import RoiMask
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
# which specifies that we want 100,000 seeds randomly distributed
# in the ROIs of every bundle.
#
# We only do this to make this example faster and consume less space.

tracking_params = dict(seed_mask=RoiMask(),
                       n_seeds=10000,
                       random_seeds=True,
                       rng_seed=42)

##########################################################################
# Initialize an AFQ object:
# -------------------------
#
# We specify bundle_info as the default bundles list (api.BUNDLES) plus the
# callosal bundle list. This tells the AFQ object to use bundles from both
# the standard and callosal templates.

myafq = api.AFQ(bids_path=op.join(afd.afq_home,
                                  'stanford_hardi'),
                dmriprep='vistasoft',
                bundle_info=api.BUNDLES + api.CALLOSUM_BUNDLES,
                tracking_params=tracking_params)

##########################################################################
# Visualizing bundles and tract profiles:
# ---------------------------------------
# This would run the script and visualize the bundles using the plotly
# interactive visualization, which should automatically open in a
# new browser window.
bundle_html = myafq.viz_bundles(export=True, n_points=50)
plotly.io.show(bundle_html[0])
