"""
==========================
Free water DTI
==========================

The free-water DTI model [1, 2]_ fits a two compartment model to dMRI data
with more than one non-zero shell. One compartment is a spherical compartment
with the diffusivity of water, which accounts for free water in the tissue.
The other compartment is the standard diffusion tensor.

In this example, we will compare the results of the fwDTI model and the standard DTI model.

"""
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nib

from AFQ.api.group import GroupAFQ
import AFQ.data.fetch as afd

from AFQ.definitions.mapping import ItkMap
from AFQ.definitions.image import ImageFile, RoiImage
import AFQ.api.bundle_dict as abd

import pandas as pd

#############################################################################
# Get some data
# --------------
# In this example, we will look at one subject from the Healthy Brain Network
# Processed Open Diffusion Derivatives dataset (HBN-POD2) [3, 4]_. The data in
# this study were collected with a multi-shell sequence, meaning that most
# subjects in this study have data with more than one non-zero b-value. This
# means that we can fit the fwDTI model to their data.
#
# We'll use a fetcher to get preprocessd dMRI data for one of the >2,000
# subjects in that study. The data gets organized into a BIDS-compatible
# format in the `~/AFQ_data/HBN` folder.

study_dir = afd.fetch_hbn_preproc(["NDARAA948VFH"])[1]

#############################################################################
# Define an AFQ object
# --------------------
# In addition to preprocessd dMRI data, HBN-POD2 contains brain mask and mapping
# information for each subject. We can use this information in our pipeline, by
# inserting this information as `mapping_definition` and `brain_mask_definition`
# inputs to the `GroupAFQ` class initializer. When initializing this object, we
# will also ask for the fwDTI scalars to be computed. For expedience, we will
# limit our investigation to the bilateral arcuate fasciculus and track only
# around that bundle. If you would like to do this for all bundles, you would
# remove the `bundle_dict` and `tracking_params` inputs to the initializer that
# are provided below.

brain_mask_definition = ImageFile(
    suffix="mask",
    filters={'desc': 'brain',
             'space': 'T1w',
             'scope': 'qsiprep'})

mapping_definition = ItkMap(
    warp_suffix='xfm',
    warp_filters={'from': 'MNI152NLin2009cAsym',
                  'to': 'T1w',
                  'scope': 'qsiprep'})


bundle_names = ["ARC_L", "ARC_R"]
bundle_dict = abd.BundleDict(bundle_names)

myafq = GroupAFQ(
    bids_path=study_dir,
    preproc_pipeline='qsiprep',
    output_dir=op.join(study_dir, "derivatives", "afq_fwdti"),
    bundle_info=bundle_dict,
    tracking_params={
        "n_seeds": 50000,
        "random_seeds": True,
        "seed_mask": RoiImage(use_waypoints=True, use_endpoints=True),
    },
    mapping_definition=mapping_definition,
    brain_mask_definition=brain_mask_definition,
    scalars=["fwdti_fa", "fwdti_md", "fwdti_fwf", "dti_fa", "dti_md"])

#############################################################################
# Compare fwDTI and DTI maps
# -------------------------
# First, we take a look at the maps for the FA and MD calculated using the two
# models

fwFA = nib.load(myafq.export("fwdti_fa")["NDARAA948VFH"]).get_fdata()
FA = nib.load(myafq.export("dti_fa")["NDARAA948VFH"]).get_fdata()

fig, ax = plt.subplots(1, 2)
ax[0].matshow(FA[:, :, FA.shape[-1] // 2], cmap='gray')
ax[0].axis("off")

ax[1].matshow(fwFA[:, :, fwFA.shape[-1] // 2], cmap='gray')
ax[1].axis("off")


fwMD = nib.load(myafq.export("fwdti_md")["NDARAA948VFH"]).get_fdata()
MD = nib.load(myafq.export("dti_md")["NDARAA948VFH"]).get_fdata()

fig, ax = plt.subplots(1, 2)
ax[0].matshow(MD[:, :, MD.shape[-1] // 2], cmap='gray', vmax=0.005)
ax[0].axis("off")

ax[1].matshow(fwMD[:, :, fwMD.shape[-1] // 2], cmap='gray', vmax=0.005)
ax[1].axis("off")


#############################################################################
#  Free-water fraction map
# -------------------------
# In addition to the standard tensor scalars, provided by the fwDTI model, this
# model also computes a free-water fraction, which is a number between 0 and 1
# that assesses the fraction of the voxel signal that is explained by the free
# water compartment.

fwf = nib.load(myafq.export("fwdti_fwf")["NDARAA948VFH"]).get_fdata()
fig, ax = plt.subplots()
ax.matshow(fwf[:, :, fwf.shape[-1] // 2], cmap='gray')
ax.axis("off")

#############################################################################
#  Comparing bundle profiles
# -------------------------
# Exporting the profiles will create a CSV file that contains information about
# node-by-node values of the scalars computed with both models. Here, we read in
# this information with Pandas and plot a comparison. As you can see, when free
# water is accounted for with the fwDTI model, FA along the bundle is higher and
# MD is lower than that estimated with the standard DTI model.

profiles_csv = myafq.export("profiles")['NDARAA948VFH']
profiles = pd.read_csv(profiles_csv)

fig, ax = plt.subplots(3, 2)
for ii, bundle in enumerate(["ARC_L", "ARC_R"]):
    ax[0, ii].plot(profiles[profiles["tractID"] == bundle]["fwdti_fa"],
                   label="fwDTI")
    ax[0, ii].plot(profiles[profiles["tractID"] == bundle]["dti_fa"],
                   label="DTI")
    ax[0, ii].set_ylabel("FA")
    ax[0, ii].legend()
    ax[1, ii].plot(profiles[profiles["tractID"] == bundle]["fwdti_md"],
                   label="fwDTI")
    ax[1, ii].plot(profiles[profiles["tractID"] == bundle]["dti_md"],
                   label="DTI")
    ax[1, ii].set_ylabel("MD")
    ax[1, ii].legend()
    ax[2, ii].plot(profiles[profiles["tractID"] == bundle]["fwdti_fwf"])
    ax[2, ii].set_ylabel("Free water fraction")
    ax[2, ii].set_xlabel("Distance along the bundle (A => P)")


#############################################################################
# References
# ----------
#
# .. [1] Hoy AR, Koay CG, Kecskemeti SR, Alexander AL. Optimization of a free
#     water elimination two-compartment model for diffusion tensor imaging.
#     Neuroimage. 2014;103:323-333.
#
# .. [2] Henriques RN, Rokem A, Garyfallidis E, St-Jean S, Peterson ET, Correia
#     MM. [Re] Optimization of a free water elimination two-compartment model
#     for diffusion tensor imaging. bioRxiv. February 2017:108795.
#     doi:10.1101/108795
#
# .. [3] Alexander LM, Escalera J, Ai L, et al. An open resource for
#     transdiagnostic research in pediatric mental health and learning
#     disorders. Sci Data. 2017;4:170181.
#
# .. [4] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and
#     quality controlled resource for pediatric brain white-matter research.
#     Scientific Data. 2022;9(1):1-27.
