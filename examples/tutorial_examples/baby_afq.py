"""
=============================================
BabyAFQ : tractometry for pediatric dMRI data
=============================================

The following is an example of tractometry for pediatric bundles. This example
and resulting pyAFQ support for pediatric bundles was inspired by and largely
due to the work of Grotheer et al. [Grotheer2021]_, as implemented in
[Grotheer2023]_.

"""
import os.path as op
import plotly
import wget
import zipfile

from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd

"""


The data for this example, provided by Kalanit Grill Spector's Stanford Vision
and Perception Neuroscience Lab is available to download on Figshare. You can
download it from there and unzip it into ~/AFQ_Data/baby_example/ (Note that
this is 2.69GB of data, so it can take a while to download). This data has
been previously published in Grotheer et al. (2022).

"""

print("Downloading processed pediatric data; this could take a while...")
data_folder = op.join(op.expanduser('~'), "AFQ_data/")
wget.download("https://figshare.com/ndownloader/files/38053692",
              op.join(data_folder, "baby_example.zip"))

with zipfile.ZipFile(op.join(data_folder, "baby_example.zip"), 'r') as zip_ref:
    zip_ref.extractall(op.join(data_folder, "baby_example"))

"""

In this case, a tractography has already been run with the excellent MRtrix
software. When you first run the following code, it will download the infant
templates into your ~/AFQ_data/pediatric_templates folder, organizing them
there in the way that pyAFQ expects to find them.

"""

##########################################################################
# Initialize a GroupAFQ object:
# -------------------------
#
# .. note::
#
#   While it is possible to run tractography and segmentation for pediatric
#   dMRI data with pyAFQ, we recommend using a custom tractography pipeline
#   and only using pyAFQ for segmentation as shown in:
#
#   - https://github.com/bloomdt-uw/babyafq/blob/main/mrtrix_pipeline.ipynb
#   - https://github.com/bloomdt-uw/babyafq/blob/main/pybabyafq.ipynb


myafq = GroupAFQ(
    bids_path=op.join(op.expanduser('~'), "AFQ_data/baby_example/example_bids_subject"),
    preproc_pipeline="vistasoft",
    reg_template_spec=afd.read_pediatric_templates()["UNCNeo-withCerebellum-for-babyAFQ"],
    reg_subject_spec="b0",
    bundle_info=abd.PediatricBundleDict(),
    import_tract={
        "suffix": "tractography", "scope": "mrtrix"},
    segmentation_params={
        "filter_by_endpoints": False},
    clean_params={
        'distance_threshold': 4},
)

##########################################################################
# Visualizing bundles:
# --------------------
viz = myafq.export("all_bundles_figure")
plotly.io.show(viz["01"])

##########################################################################
# References:
# -------------------------
# .. [Grotheer2021] Grotheer, Mareike, Mona Rosenke, Hua Wu, Holly Kular,
#                   Francesca R. Querdasi, Vaidehi S. Natu, Jason D. Yeatman,
#                   and Kalanit Grill-Spector. "White matter myelination during
#                   early infancy is linked to spatial gradients and myelin
#                   content at birth." Nature communications 13: 997.
#
# .. [Grotheer2023] Grotheer, Mareike, David Bloom, John Kruper,
#                   Adam Richie-Halford, Stephanie Zika,
#                   Vicente A. Aguilera González, Jason D. Yeatman,
#                   Kalanit Grill-Spector, and Ariel Rokem. "Human white matter
#                   myelinates faster in utero than ex utero." Proceedings
#                   of the National Academy of Sciences 120: e2303491120.