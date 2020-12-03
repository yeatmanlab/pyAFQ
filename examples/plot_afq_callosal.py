"""
==========================
Calossal bundles using AFQ API
==========================
An example using the AFQ API to find calossal bundles using the templates from:
http://hdl.handle.net/1773/34926
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

afd.organize_stanford_data()

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
                bundle_info=api.BUNDLES + api.CALLOSAL_BUNDLES)

##########################################################################
# Visualizing bundles and tract profiles:
# ---------------------------------------
# This would run the script and visualize the bundles using the plotly
# interactive visualization, which should automatically open in a
# new browser window.
bundle_html = myafq.viz_bundles(export=True, n_points=50)
plotly.io.show(bundle_html[0])
