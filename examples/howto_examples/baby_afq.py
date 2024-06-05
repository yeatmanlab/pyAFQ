"""
=============================================
BabyAFQ : tractometry for infant dMRI data
=============================================

The following is an example of tractometry for infant bundles. This example and
resulting pyAFQ support for pediatric bundles was inspired by and largely due
to the work of Grotheer et al. [Grotheer2022]_, as implemented in
[Grotheer2023]_.

.. note::
    Because it is time and disk-space consuming, this example
    is not run when the pyAFQ documentation is built. To run this example
    yourself, you can download the contents of this file as an
    executable `.py` file or as a Jupyter notebook from the links at the bottom
    of the page.

"""
import os.path as op
import plotly
import wget
import zipfile

from AFQ.api.group import GroupAFQ
import AFQ.api.bundle_dict as abd
import AFQ.data.fetch as afd

##########################################################################
# Baby dMRI data
# -------------------------
#
# Infant MRI data are quite different from data acquired in grownup
# participants, and even from children that are just a few years older.
# First, there is the rather obvious difference in size. Baby brains are
# approximately 25% the size of grownup brains at birth. But there are also
# often less known differences in brain tissue properties. For example, the
# myelin content of white matter is much lower in infants than in grownups.
# This is important because the diffusion signal that we measure with dMRI is
# sensitive to the myelin content, and it means that the dMRI signal differs
# quite a bit in newborn infants. For the purpose of delineating the major
# white matter pathways, it is also important to know that their shape,
# location and curvature is different in infants than in grownups. For example,
# the arcuate fasciculus is much more curved in infants than in grownups.
# Because of this, we use a different set of templates for infant brains than
# for grownup brains. These templates were created and validated by
# Mareike Grotheer and colleagues in [Grotheer2022]_. They are available for
# download as part of the pyAFQ software, as we will show below.
#
# In this example, we will demonstrate the use of pyAFQ on data from one
# infant. The data, provided by Kalanit Grill Spector's
# `Stanford Vision and Perception Neuroscience Lab <http://vpnl.stanford.edu/>`,
# and was previously published in [Grotheer2022]_.
# The data is available to download on
# `Figshare <https://figshare.com/articles/dataset/Example_babyAFQ_BIDS_subject/21440739>`.
# You can download it from there and unzip it into ~/AFQ_Data/baby_example/
# (Note that this is 2.69GB of data, so it can take a while to download). Or
# you can download it and unzip it using the following block of code.

data_folder = op.join(op.expanduser('~'), "AFQ_data/")
baby_zip = op.join(data_folder, "baby_example.zip")
if not op.exists(baby_zip):
    print("Downloading processed pediatric data; this could take a while...")
    wget.download(
        "https://figshare.com/ndownloader/files/38053692",
        baby_zip)

with zipfile.ZipFile(baby_zip, 'r') as zip_ref:
    zip_ref.extractall(op.join(data_folder, "baby_example"))

##########################################################################
# Initialize a GroupAFQ object:
# -----------------------------
#
# Now that the data is downloaded and organized in a BIDS-compliant structure,
# we can start running pyAFQ on it. We start by initializing a GroupAFQ object.
# This object manages all of the data transformations and computations
# conducted by the software, based on its initial configuration, which we set
# up below.
#
# A few special things to note here:
#
# 1. The data were preprocessed using the `vistasoft` pipeline, so we set
#    `preproc_pipeline = "vistasoft"`.
# 2. We use the UNC neonatal template, which can be read on a call to the
#    `read_pediatric_templates` function in `AFQ.data.fetch`.
# 3. We use the `baby_bd` to define the bundles that we want to
#    segment. This dictionary is different from the default behavior in that it
#    uses the waypoint ROIs from [Grotheer2022]_.
# 4. In this case, tractography has already been run using
#    `MRTRIX <https://www.mrtrix.org/>`, and is accessed using the
#    `import_tract` key-word argument.

myafq = GroupAFQ(
    bids_path=op.join(op.expanduser('~'),
                      "AFQ_data/baby_example/example_bids_subject"),
    preproc_pipeline="vistasoft",
    reg_template_spec=afd.read_pediatric_templates(
    )["UNCNeo-withCerebellum-for-babyAFQ"],
    reg_subject_spec="b0",
    bundle_info=abd.baby_bd(),
    import_tract={
        "suffix": "tractography", "scope": "mrtrix"},
)

##########################################################################
# Running the pipeline
# --------------------
#
# A call to the `export` function will trigger the pyAFQ pipeline. This will
# run tractography, bundle segmentation, and bundle cleaning. The results will
# be saved in the `~/AFQ_data/baby_example/derivatives/afq` folder. This can
# take a while to run, depending on your computer.
# In this case, we call `export` with the `all_bundles_figure` option. This is
# because visualizations are created after most other parts of the pipeline
# have been run. This means that when this call is done, you should have many
# of the derivative results in the output folder, including the tractography,
# segmentation, and tract profile results, as well as the visualizations.

viz = myafq.export("all_bundles_figure")

##########################################################################
# Viewing the results
# -------------------
# One way to view the results is to open the file named
# `sub-01_ses-01_dwi_space-RASMM_model-probCSD_algo-AFQ_desc-viz_dwi.html`
# in your browser. This is a visualization of the tractography and segmentation
# results for all of the bundles. You can navigate this visualization by
# clicking on the different bundles in the legend on the right side of the
# screen. You can also zoom in and out using the mouse wheel, and rotate the
# view by clicking and dragging with the mouse. You can also view the FA tract
# profiles in a plot on the left side of the page.
#
# If the baby bundles appear dark in the html visualization due to low FA values, you
# can reduce the upper limit of the range in the `sbv_lims_bundles` option when
# building your GroupAFQ object (e.g. `GroupAFQ(..., sbv_lims_bundles=[0, 0.5])`).


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
