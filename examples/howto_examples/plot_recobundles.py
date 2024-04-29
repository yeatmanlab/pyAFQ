"""
=========================================
Using RecoBundles for bundle recognition
=========================================

For bundle recognition, pyAFQ defaults to use the waypoint ROI approach
described in [Yeatman2012]_. However, as an alternative approach, pyAFQ also
supports using the RecoBundles algorithm [Garyfallidis2017]_, which uses an
atlas of bundles in streamlines. This example shows how to
use RecoBundles for bundle recognition.

The code closely resembles the code used in :ref:`sphx_glr_tutorial_examples_plot_001-plot_afq_api.py`.

"""

import os.path as op
import AFQ.data.fetch as afd
from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd

afd.organize_stanford_data(clear_previous_afq="track")

tracking_params = dict(n_seeds=25000,
                       random_seeds=True,
                       rng_seed=42)


##########################################################################
# Defining the segmentation params
# --------------------------------
# We also refer to bundle recognition as the "segmentation" of the tractogram.
# Parameters of this process are set through a dictionary input to the
# `segmentation_params` argument of the GroupAFQ object. In this case, we
# use `abd.reco_bd(16)`, which tells pyAFQ to use the RecoBundles
# algorithm for bundle recognition.

myafq = GroupAFQ(
    output_dir=op.join(afd.afq_home, 'stanford_hardi', 'derivatives',
                       'recobundles'),
    bids_path=op.join(afd.afq_home, 'stanford_hardi'),
    # Set the algorithm to use RecoBundles for bundle recognition:
    bundle_info=abd.reco_bd(16),
    preproc_pipeline='vistasoft',
    tracking_params=tracking_params,
    viz_backend_spec='plotly_no_gif')

fig_files = myafq.export_all()

##########################################################################
# References:
# -------------------------
#  .. [Garyfallidis2017] Garyfallidis, Eleftherios, Marc-Alexandre Côté,
#                      Francois Rheault, Jasmeen Sidhu, Janice Hau, Laurent
#                      Petit, David Fortin, Stephen Cunanne, and Maxime
#                      Descoteaux. 2017.“Recognition of White Matter Bundles
#                      Using Local and Global Streamline-Based Registration and
#                      Clustering.”NeuroImage 170: 283-295.
