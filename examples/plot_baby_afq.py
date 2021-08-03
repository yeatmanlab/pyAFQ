"""
==============================
Pediatric Profiles:
==============================

The following is an example of tractometry for pediatric bundles.

TODO cite @mareike

.. note::

  This example uses the Yeatman et al. waypoint ROI approach, first
  described in [Yeatman2012]_ and further elaborated in [Yeatman2014]_.

"""
import os.path as op
import plotly

from AFQ import api
import AFQ.data as afd

from AFQ.definitions.mask import RoiMask, MaskFile

##########################################################################
# Get example data:
# -----------------
# **Diffusion dataset**
#
# .. note::
#   The diffusion data used in this example are from the Developing Human
#   Connectome project (dHCP):
#
#   http://www.developingconnectome.org/project/
#
#   Acknowledgement: These results were obtained using data made available from
#   the Developing Human Connectome Project funded by the European Research
#   Council under the European Union’s Seventh Framework Programme
#   (FP/2007-2013) / ERC Grant Agreement no. [319456]
#
# .. note::
#   This example assumes:
#
#   - A single subject and session
#
#   - Subject's session data has been downloaded into
#
#     `AFQ_data\\dHCP\\derivatives`
#
#   - Subject's session data has been made BIDS compliant
#
#     see https://bids.neuroimaging.io
#

myafq = api.AFQ(# ==== BIDS parameters ====
                bids_path=op.join(afd.afq_home, "dHCP"),
                dmriprep="derivatives",
                # ===== Registration parameters ====
                min_bval=1000,
                max_bval=1000,
                reg_template=afd.read_pediatric_templates()["UNCNeo-withCerebellum-for-babyAFQ"],
                reg_subject="b0",
                brain_mask=MaskFile("brainmask",
                                    {"scope": "derivatives"}),
                # ==== Bundle parameters ====
                bundle_info=api.PediatricBundleDict(),
                # ==== Compute parameters ====
                force_recompute=True,
                # ==== Tracking parameters ====
                tracking_params={"seed_mask": RoiMask(),
                                 "stop_threshold": 0.1},
                # ==== Segmentation parameters ====
                segmentation_params={
                    "filter_by_endpoints": False},
                )

##########################################################################
# Visualizing bundles:
# --------------------

plotly.io.show(myafq.all_bundles_figure[list(myafq.all_bundles_figure)[0]])

##########################################################################
# References:
# -------------------------
# .. [Yeatman2012] Jason D Yeatman, Robert F Dougherty, Nathaniel J Myall,
#                  Brian A Wandell, Heidi M Feldman, "Tract profiles of
#                  white matter properties: automating fiber-tract
#                  quantification", PloS One, 7: e49790
#
# .. [Yeatman2014] Jason D Yeatman, Brian A Wandell, Aviv Mezer Feldman,
#                  "Lifespan maturation and degeneration of human brain white
#                  matter", Nature Communications 5: 4932
