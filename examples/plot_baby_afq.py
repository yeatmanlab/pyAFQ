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
import logging
import sys
import time
import os.path as op
import plotly
from dipy.data.fetcher import _make_fetcher
import nibabel as nib

from AFQ import api
import AFQ.registration as reg
from AFQ.mask import RoiMask, MaskFile


# Ensure segmentation logging information is included in this example's output
root = logging.getLogger()
root.setLevel(logging.ERROR)
logging.getLogger('AFQ.Segmentation').setLevel(logging.INFO)
logging.getLogger('AFQ.tractography').setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
root.addHandler(handler)

# Target directory for this example's output files
working_dir = "./baby_afq"

afq_home = op.join(op.expanduser('~'), 'AFQ_data')

baseurl = "https://ndownloader.figshare.com/files/"

##########################################################################
# Pediatric templates:
# --------------------
#
# Templates include the UNC 0-1-2 Infant Atlases and waypoint ROIs matched
# to the atlas.
#
# Information on:
#
# - atlas: https://www.nitrc.org/projects/pediatricatlas
#
# - pediatric templates: https://github.com/yeatmanlab/AFQ/tree/babyAFQ
#
# Templates downloaded from:
#
# - https://figshare.com/articles/dataset/ROIs_probabilistic_maps_and_transform_data_for_pediatric_automated_fiber_quantification/13027487
#

print("Fetching pediatric templates...")

# NOTE Normally logic in data.py, but not sure really want to keep adding to
#      file for each novel dataset...
# TODO coding: duplicating pattern for figshare
# TODO coding: scrape files from figshare is there an interface?
#      make process easier (manually inspect each file)
# TODO coding: design pattern: different files for marshalling data from
#      different datasets
# TODO coding: three arrays for fnames, md5, and remote_fnames doesn't seem
#      like best data structures
# TODO coding: group ROIs together in smaller structures
#      (NOTE going to do this in bundle dict; and not all files are ROIs)
# TODO coding: duplicate md5 hash for CST ROIs. same file?
pediatric_fnames = [
    "ATR_roi1_L.nii.gz", "ATR_roi1_R.nii.gz",
    "ATR_roi2_L.nii.gz", "ATR_roi2_R.nii.gz",
    "ATR_roi3_L.nii.gz", "ATR_roi3_R.nii.gz",
    "CGC_roi1_L.nii.gz", "CGC_roi1_R.nii.gz",
    "CGC_roi2_L.nii.gz", "CGC_roi2_R.nii.gz",
    "CGC_roi3_L.nii.gz", "CGC_roi3_R.nii.gz",
    "CST_roi1_L.nii.gz", "CST_roi1_R.nii.gz",
    "CST_roi2_L.nii.gz", "CST_roi2_R.nii.gz",
    "FA_L.nii.gz", "FA_R.nii.gz",
    "FP_L.nii.gz", "FP_R.nii.gz",
    "HCC_roi1_L.nii.gz", "HCC_roi1_R.nii.gz",
    "HCC_roi2_L.nii.gz", "HCC_roi2_R.nii.gz",
    "IFO_roi1_L.nii.gz", "IFO_roi1_R.nii.gz",
    "IFO_roi2_L.nii.gz", "IFO_roi2_R.nii.gz",
    "IFO_roi3_L.nii.gz", "IFO_roi3_R.nii.gz",
    "ILF_roi1_L.nii.gz", "ILF_roi1_R.nii.gz",
    "ILF_roi2_L.nii.gz", "ILF_roi2_R.nii.gz",
    "LH_Parietal.nii.gz", "RH_Parietal.nii.gz",
    "MdLF_roi1_L.nii.gz", "MdLF_roi1_R.nii.gz",
    "SLF_roi1_L.nii.gz", "SLF_roi1_R.nii.gz",
    "SLF_roi2_L.nii.gz", "SLF_roi2_R.nii.gz",
    "SLFt_roi2_L.nii.gz", "SLFt_roi2_R.nii.gz",
    "SLFt_roi3_L.nii.gz", "SLFt_roi3_R.nii.gz",
    "UNC_roi1_L.nii.gz", "UNC_roi1_R.nii.gz",
    "UNC_roi2_L.nii.gz", "UNC_roi2_R.nii.gz",
    "UNC_roi3_L.nii.gz", "UNC_roi3_R.nii.gz",
    "VOF_box_L.nii.gz", "VOF_box_R.nii.gz",
    "UNCNeo-withCerebellum-for-babyAFQ.nii.gz",
    "UNCNeo_JHU_tracts_prob-for-babyAFQ.nii.gz",
    "mid-saggital.nii.gz"
]

pediatric_md5_hashes = [
    "2efe0deb19ac9e175404bf0cb29d9dbd", "c2e07cd50699527bd7b0cbbe88703c56",
    "76b36d8d6759df58131644281ed16bd2", "645102225bad33da30bafd41d24b3ab0",
    "45ec94d42fdc9448afa6760c656920e9", "54e3cb1b8c242be279f1410d8bb3c383",
    "1ee9f7e8b21ef8ceee81d5a7a178ef33", "4f11097f7ae317aa8d612511be79e2f1",
    "1c4c0823c23b676d6d35004d93b9c695", "d4830d558cc8f707ebec912b32d197a5",
    "c405e0dbd9a4091c77b3d1ad200229b4", "ec0aeccc6661d2ee5ed79259383cdcee",
    "2802cd227b550f6e85df0fec1d515c29", "385addb999dc6d76957d2a35c4ee74bb",
    "b79f01829bd95682faaf545c72b1d52c", "b79f01829bd95682faaf545c72b1d52c",
    "e49ba370edca96734d9376f551d413db", "f59e9e69e06325198f70047cd63c3bdc",
    "ae3bd2931f95adae0280a8f75cd3ca9b", "c409a0036b8c2dd4d03d11fbc6bfbdcd",
    "c2597a474ea5ec9e3126c35fd238f6b2", "67af59c934147c9f9ff6e0b76c4cc6eb",
    "72d0bbc0b6162e9291fdc450c103a1f0", "51f5041be63ad0ac10d1ac09e3bf1a8e",
    "6200f5cdc1402dce46dedd505468b147", "83cb5bf6b9b1eda63c8643871e84a6d4",
    "2a5d8d309b1256d6e48958e112201c2c", "ba24d0915fdff215a403480d0a7745c9",
    "1001e833d1344274d543ffd02a66af80", "03e20c5eebcd4d439c4ffb36d26a10d9",
    "6200f5cdc1402dce46dedd505468b147", "83cb5bf6b9b1eda63c8643871e84a6d4",
    "a6ae325cce2dc4bb21b52ee4c6ca844f", "a96a31df8f96ccf1c5f871a18e4a2d72",
    "65b7378ca85689a5f23b1b84a6ed78f0", "ce0d0ea696ef51c671aa118e11192e2d",
    "ce4918086ca1e829698576bcf95db62b", "96168d2eff74ec2acf9065f499732526",
    "6b20ba319d44472ec21d6ece937605bb", "26b1cf6ec8bd365dde42e3efe9beeac2",
    "0b3ccf06564d973bfcfff9a87e74f8b5", "84f3426033a2b225b0920b2622864375",
    "5351b3cb7efa9aa8e83e266342809ebe", "4e8a34aaba4e0f22a6149f38620dc39d",
    "682c08f66e8c2cf9e4e60f5ce308c76c", "9077affd4f3a8a1d6b44678cde4b3bf4",
    "5adf36f00669cc547d5eb978acf46266", "66a8002688ffdf3943722361da90ec6a",
    "efb5ae138df92019541861f9aa6a4d57", "757ec61078b2e9f9a073871b3216ff7a",
    "ff1e238c52a21f8cc5d44ac614d9627f", "cf16dd2767c6ab2d3fceb2890f6c3e41",
    "6016621e244b60b9c69fd44b055e4a03", "fd495a2c94b6b13bfb4cd63e293d3fc0",
    "bf81a23d80f55e5f1eb0c16717193105",
    "6f8bf8f70216788d14d9a49a3c664b16",
    "19df0297d6a2ac21da5e432645d63174",
]

pediatric_remote_fnames = [
    "24880625", "24880628", "24880631", "24880634", "24880637", "24880640",
    "24880643", "24880646", "24880649", "24880652", "24880655", "24880661",
    "24880664", "24880667", "24880670", "24880673",
    "24880676", "24880679",
    "24880685", "24880688",
    "24880691", "24880694", "24880697", "24880700",
    "24880703", "24880706", "24880712", "24880715", "24880718", "24880721",
    "24880724", "24880727", "24880730", "24880733",
    "24880736", "24880748",
    "24880739", "24880742",
    "24880754", "24880757", "24880760", "24880763",
    "24880769", "24880772", "24880775", "24880778",
    "24880781", "24880787", "24880790", "24880793", "24880796", "24880802",
    "24880805", "24880808",
    "24880616",
    "24880613",
    "24986396"
]

fetch_pediatric_templates = _make_fetcher("fetch_pediatric_templates",
                                          op.join(afq_home,
                                                  'pediatric_templates'),
                                          baseurl, pediatric_remote_fnames,
                                          pediatric_fnames,
                                          md5_list=pediatric_md5_hashes,
                                          doc="Download pediatric templates")


# TODO coding: this duplicates read_xxx_template pattern
# TODO coding: unit testing
def read_pediatric_templates(resample_to=False):
    """Load pediatric AFQ templates from file

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    files, folder = fetch_pediatric_templates()

    print('Loading pediatric templates...')
    tic = time.perf_counter()

    pediatric_templates = {}
    for f in files:
        img = nib.load(op.join(folder, f))
        if resample_to:
            if isinstance(resample_to, str):
                resample_to = nib.load(resample_to)
            img = nib.Nifti1Image(reg.resample(img.get_fdata(),
                                               resample_to,
                                               img.affine,
                                               resample_to.affine),
                                  resample_to.affine)
        pediatric_templates[f.split('.')[0]] = img

    toc = time.perf_counter()
    print(f'pediatric templates loaded in {toc - tic:0.4f} seconds')

    # For the arcuate (AF/ARC), reuse the SLF ROIs
    pediatric_templates['ARC_roi1_L'] = pediatric_templates['SLF_roi1_L']
    pediatric_templates['ARC_roi1_R'] = pediatric_templates['SLF_roi1_R']
    pediatric_templates['ARC_roi2_L'] = pediatric_templates['SLFt_roi2_L']
    pediatric_templates['ARC_roi2_R'] = pediatric_templates['SLFt_roi2_R']
    pediatric_templates['ARC_roi3_L'] = pediatric_templates['SLFt_roi3_L']
    pediatric_templates['ARC_roi3_R'] = pediatric_templates['SLFt_roi3_R']

    # For the middle longitudinal fasciculus (MdLF) reuse ILF ROI
    pediatric_templates['MdLF_roi2_L'] = pediatric_templates['ILF_roi1_L']
    pediatric_templates['MdLF_roi2_R'] = pediatric_templates['ILF_roi1_R']

    return pediatric_templates


pediatric_templates = read_pediatric_templates()

##########################################################################
# Create pediatric bundle specification:
# ----------------------------------
# The bundles specify meta-data for the segmentation.
#
# .. note
#   Pediatric bundles differ from adult bundles:
#
#   - A third ROI has been introduced for curvy tracts: ARC, ATR, CGC, IFO,
#     and UCI
#
#   - ILF posterior ROI has been split into two to separate ILF and mdLF
#
#   - Addition of pAF and VOF ROIs
#
#   - SLF ROIs are restricted to parietal cortex
#
#
# The following keys are required for a `bundle` entry:
#
# - `ROIs`
#
#   lists the ROI templates.
#
# .. note::
#   Order of the `ROIs` matters and may result in different tract profiles
#
# - `rules`
#
#   label each ROI as inclusionary `True` or exclusionary `False`.
#
# - `cross_midline`
#
#   whether or not the streamline crosses the midline
#
# - `prob_map`
#
#   probablity maps to further refine ROIs
#
# - `uid`
#
#   unique bundle identifier. not used by segmentation, but helpful for
#   debugging or quality control reports.

print("Creating pediatric bundle specification...")

# TODO coding: replicating api.make_bundle_dict; find better way
pediatric_bundle_names = [
    "ARC",  # 'Arcuate Fasciculus'
    "ATR",  # 'Thalamic Radiation'
    "CGC",  # 'Cingulum Cingulate'
    "CST",  # 'Corticospinal'
    "FA",   # 'Forceps Minor'
    "FP",   # 'Forceps Major'

    # NOTE @mareike: do not include HCC
    # "HCC",  # 'Cingulum Hippocampus'

    "IFO",  # 'Inferior Fronto-occipital'

    "ILF",  # 'Inferior Longitudinal Fasciculus'
    "MdLF",  # 'Middle Longitudinal Fasciculus'

    "SLF",  # 'Superior Longitudinal Fasciculus'
    "UNC",  # 'Uncinate Fasciculus'

    # NOTE @arokem do not include pAF and VOF_box (for now)
    # "pAF",  # 'Posterior Arcuate Fasciculus'
    # "VOF_box"  # 'Vertical Occipital Fasciculus'
]


# NOTE @arokem start with waypoint ROIs, eventually will consider RECO bundles
def make_pediatric_bundle_dict(bundle_names=pediatric_bundle_names,
                               seg_algo="afq",
                               resample_to=False):
    # pediatric probability maps
    prob_map_order = ["ATR_L", "ATR_R", "CST_L", "CST_R", "CGC_L", "CGC_R",
                      "HCC_L", "HCC_R", "FP", "FA", "IFO_L", "IFO_R", "ILF_L",
                      "ILF_R", "SLF_L", "SLF_R", "UNC_L", "UNC_R",
                      "ARC_L", "ARC_R", "MdLF_L", "MdLF_R"]

    prob_maps = pediatric_templates['UNCNeo_JHU_tracts_prob-for-babyAFQ']
    prob_map_data = prob_maps.get_fdata()

    # pediatric bundle dict
    pediatric_bundles = {}

    # each bundles gets a digit identifier (to be stored in the tractogram)
    uid = 1

    for name in pediatric_bundle_names:
        # ROIs that cross the mid-line
        if name in ["FA", "FP"]:
            pediatric_bundles[name] = {
                'ROIs': [pediatric_templates[name + "_L"],
                         pediatric_templates[name + "_R"],
                         pediatric_templates["mid-saggital"]],
                'rules': [True, True, True],
                'cross_midline': True,
                'prob_map': prob_map_data[...,
                                          prob_map_order.index(name)],
                'uid': uid}
            uid += 1
        # SLF is a special case, because it has an exclusion ROI:
        elif name == "SLF":
            for hemi in ['_R', '_L']:
                pediatric_bundles[name + hemi] = {
                    'ROIs': [pediatric_templates[name + '_roi1' + hemi],
                             pediatric_templates[name + '_roi2' + hemi],
                             pediatric_templates["SLFt_roi2" + hemi]],
                    'rules': [True, True, False],
                    'cross_midline': False,
                    'prob_map': prob_map_data[...,
                                              prob_map_order.index(name + hemi)],
                    'uid': uid}
                uid += 1
        # Third ROI for curvy tracts
        elif name in ["ARC", "ATR", "CGC", "IFO", "UNC"]:
            for hemi in ['_R', '_L']:
                pediatric_bundles[name + hemi] = {
                    'ROIs': [pediatric_templates[name + '_roi1' + hemi],
                             pediatric_templates[name + '_roi2' + hemi],
                             pediatric_templates[name + '_roi3' + hemi]],
                    'rules': [True, True, True],
                    'cross_midline': False,
                    'prob_map': prob_map_data[...,
                                              prob_map_order.index(name + hemi)],
                    'uid': uid}
                uid += 1
        # TODO confirm with @mareike no probabilty map for MdLF (only 20 not 22)
        elif name == "MdLF":
            for hemi in ['_R', '_L']:
                pediatric_bundles[name + hemi] = {
                    'ROIs': [pediatric_templates[name + '_roi1' + hemi],
                             pediatric_templates[name + '_roi2' + hemi]],
                    'rules': [True, True],
                    'cross_midline': False,
                    'uid': uid}
                uid += 1
        # Default: two ROIs within hemisphere
        else:
            for hemi in ['_R', '_L']:
                pediatric_bundles[name + hemi] = {
                    'ROIs': [pediatric_templates[name + '_roi1' + hemi],
                             pediatric_templates[name + '_roi2' + hemi]],
                    'rules': [True, True],
                    'cross_midline': False,
                    'prob_map': prob_map_data[...,
                                              prob_map_order.index(name + hemi)],
                    'uid': uid}
                uid += 1

    return pediatric_bundles


# monkey patch
api.make_bundle_dict = make_pediatric_bundle_dict

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

print("Validating data...")

dhcp_home = op.join(afq_home, 'dHCP')

myafq = api.AFQ(# ==== BIDS parameters ====
                bids_path=dhcp_home,
                dmriprep='dHCP neonatal dMRI pipeline',
                # bids_filters={"suffix": "dwi"}, # default
                # custom_tractography_bids_filters=None, # default
                # ===== Registration parameters ====
                # b0_threshold=50, # default
                # min_bval=None, # default
                min_bval=1000,  # override
                # max_bval=None, # default
                max_bval=1000,  # override
                reg_template=pediatric_templates['UNCNeo-withCerebellum-for-babyAFQ'],
                # reg_subject="power_map", # default
                reg_subject="b0",  # override
                # brain_mask=B0Mask(), # default
                brain_mask=MaskFile("brainmask",
                                    {"scope": "dHCP neonatal dMRI pipeline"}),
                # use_prealign=True, # default
                # ==== Bundle parameters ====
                bundle_names=pediatric_bundle_names,
                # scalars=["dti_fa", "dti_md"], # default
                # ==== Compute parameters ====
                # dask_it=False,
                # force_recompute=False, # default
                force_recompute=True,  # override
                # ==== Tracking parameters ====
                # defaults:
                # seed_mask=None, seed_threshold=0, n_seeds=1,
                # random_seeds=False, rng_seed=None, stop_mask=None,
                # stop_threshold=0, step_size=0.5, min_length=10,
                # max_length=1000, odf_model="DTI", tracker="local"

                # tracking_params=None, # default
                # tracking_params={"seed_threshold": 0.15},  # override
                tracking_params={"seed_mask": RoiMask(),
                                 "stop_threshold": 0.1},
                # params_file, directions="det", max_angle=30., sphere=None,
                # ==== Segmentation parameters ====
                # defaults:
                # nb_points=False, seg_algo='AFQ', reg_algo=None,
                # clip_edges=False, progressive=True, greater_than=50,
                # rm_small_clusters=50, model_clust_thr=5, reduction_thr=20,
                # refine=False, pruning_thr=5, b0_threshold=50,
                # prob_threshold=0, rng=None, return_idx=False,
                # filter_by_endpoints=True, dist_to_aal=4,
                # save_intermediates=None

                # segmentation_params=None, # default
                # TODO endpoints? -- turn off for now
                segmentation_params={"filter_by_endpoints": False},
                # ==== Cleaning parameters ====
                # defaults:
                # n_points=100, clean_rounds=5, distance_threshold=5,
                # length_threshold=4, min_sl=20, stat='mean',
                # return_idx=False

                # clean_params=None # default
                # ==== Visualiation parameters ====
                # virtual_frame_buffer=False, # default
                # viz_backend="plotly_no_gif", # default
                )

# export AFQ artifacts for quality control
myafq.export_all()

##########################################################################
# Visualizing bundles:
# --------------------

plotly.io.show(myafq.viz_bundles(export=True, n_points=50)[0])

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
