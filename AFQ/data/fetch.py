from dipy.align import resample
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature
from dipy.io.streamline import (
    load_tractogram, save_tractogram, StatefulTractogram, Space)
from dipy.data.fetcher import _make_fetcher
import dipy.data as dpd
from AFQ.utils.path import drop_extension

import os
import os.path as op
import json
from glob import glob
import shutil

import numpy as np
import pandas as pd
import logging
import time
from tqdm import tqdm

import warnings
import nibabel as nib
import boto3
from botocore import UNSIGNED
from botocore.client import Config


# capture templateflow resource warning and log
default_warning_format = warnings.formatwarning
try:
    warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}'
    logging.captureWarnings(True)
    pywarnings_logger = logging.getLogger('py.warnings')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    pywarnings_logger.addHandler(console_handler)

    warnings.filterwarnings(
        "default", category=ResourceWarning,
        module="templateflow")

    from templateflow import api as tflow
finally:
    logging.captureWarnings(False)
    warnings.formatwarning = default_warning_format


__all__ = ["fetch_callosum_templates", "read_callosum_templates",
           "fetch_or_templates", "read_or_templates",
           "fetch_templates", "read_templates",
           "fetch_stanford_hardi_tractography",
           "read_stanford_hardi_tractography",
           "organize_stanford_data",
           "fetch_stanford_hardi_lv1"]


afq_home = op.join(op.expanduser('~'), 'AFQ_data')

baseurl = "https://ndownloader.figshare.com/files/"


def _make_reusable_fetcher(name, folder, baseurl, remote_fnames, local_fnames,
                           doc="", md5_list=None, **make_fetcher_kwargs):
    def fetcher():
        all_files_downloaded = True
        for fname in local_fnames:
            if not op.exists(op.join(folder, fname)):
                all_files_downloaded = False
        if all_files_downloaded:
            files = {}
            for i, (f, n), in enumerate(zip(remote_fnames, local_fnames)):
                files[n] = (baseurl + f, md5_list[i] if
                            md5_list is not None else None)
            return files, folder
        else:
            return _make_fetcher(
                name, folder, baseurl, remote_fnames, local_fnames,
                doc=doc, **make_fetcher_kwargs)()
    fetcher.__name__ = name
    fetcher.__doc__ = doc
    return fetcher


def _fetcher_to_template(fetcher, as_img=False, resample_to=False):
    if isinstance(resample_to, str):
        resample_to = nib.load(resample_to)
    files, folder = fetcher()
    template_dict = {}
    for f in files:
        img = op.join(folder, f)
        if as_img:
            img = nib.load(img)
        if resample_to:
            img = nib.Nifti1Image(resample(img.get_fdata(),
                                           resample_to,
                                           img.affine,
                                           resample_to.affine).get_fdata(),
                                  resample_to.affine)
        template_dict[drop_extension(f)] = img
    return template_dict


callosum_fnames = ["Callosum_midsag.nii.gz",
                   "L_AntFrontal.nii.gz",
                   "L_Motor.nii.gz",
                   "L_Occipital.nii.gz",
                   "L_Orbital.nii.gz",
                   "L_PostParietal.nii.gz",
                   "L_SupFrontal.nii.gz",
                   "L_SupParietal.nii.gz",
                   "L_Temporal.nii.gz",
                   "R_AntFrontal.nii.gz",
                   "R_Motor.nii.gz",
                   "R_Occipital.nii.gz",
                   "R_Orbital.nii.gz",
                   "R_PostParietal.nii.gz",
                   "R_SupFrontal.nii.gz",
                   "R_SupParietal.nii.gz",
                   "R_Temporal.nii.gz"]

callosum_remote_fnames = ["5273794", "5273797", "5273800", "5273803",
                          "5273806", "5273809", "5273812", "5273815",
                          "5273821", "5273818", "5273824", "5273827",
                          "5273830", "5273833", "5273836", "5273839",
                          "5273842"]

callosum_md5_hashes = ["709fa90baadeacd64f1d62b5049a4125",
                       "987c6169de807c4e93dc2cbd7a25d506",
                       "0da114123d0b0097b96fe450a459550b",
                       "6d845bd10504f67f1dc17f9000076d7e",
                       "e16c7873ef4b08d26b77ef746dab8237",
                       "47193fd4df1ea17367817466de798b90",
                       "7e78bf9671e6945f4b2f5e7c30595a3c",
                       "8adbb947377ff7b484c88d8c0ffc2125",
                       "0fd981a4d0847e0642ff96e84fe44e47",
                       "87c4855efa406d8fb004cffb8259180e",
                       "c7969bcf5f2343fd9ce9c49b336cf14c",
                       "bb4372b88991932150205ffb22aa6cb7",
                       "d198d4e7db18ddc7236cf143ecb8342e",
                       "d0f6edef64b0c710c92e634496085dda",
                       "85eaee44665f244db5adae2e259833f6",
                       "25f24eb22879a05d12bda007c81ea55a",
                       "2664e0b8c2d9c59f13649a89bfcce399"]

fetch_callosum_templates = _make_reusable_fetcher(
    "fetch_callosum_templates",
    op.join(afq_home,
            'callosum_templates'),
    baseurl, callosum_remote_fnames,
    callosum_fnames,
    md5_list=callosum_md5_hashes,
    doc="Download AFQ callosum templates")


def read_callosum_templates(as_img=True, resample_to=False):
    """Load AFQ callosum templates from file

    Parameters
    ----------
    as_img : bool, optional
        If True, values are `Nifti1Image`. Otherwise, values are
        paths to Nifti files. Default: True
    resample_to : str or nibabel image class instance, optional
        A template image to resample to. Typically, this should be the
        template to which individual-level data are registered. Defaults to
        the MNI template. Default: False

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ')

    logger.debug('loading callosum templates')
    tic = time.perf_counter()

    template_dict = _fetcher_to_template(
        fetch_callosum_templates,
        as_img=as_img,
        resample_to=resample_to)

    toc = time.perf_counter()
    logger.debug(f'callosum templates loaded in {toc - tic:0.4f} seconds')

    return template_dict


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
# - https://figshare.com/articles/dataset/ROIs_probabilistic_maps_and_transform_data_for_pediatric_automated_fiber_quantification/13027487  # noqa
#

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
    "24880664", "24880667", "24880670", "24880673", "24880676", "24880679",
    "24880685", "24880688", "24880691", "24880694", "24880697", "24880700",
    "24880703", "24880706", "24880712", "24880715", "24880718", "24880721",
    "24880724", "24880727", "24880730", "24880733", "24880736", "24880748",
    "24880739", "24880742", "24880754", "24880757", "24880760", "24880763",
    "24880769", "24880772", "24880775", "24880778", "24880781", "24880787",
    "24880790", "24880793", "24880796", "24880802", "24880805", "24880808",
    "24880616", "24880613", "24986396"
]

fetch_pediatric_templates = _make_reusable_fetcher(
    'fetch_pediatric_templates',
    op.join(afq_home, 'pediatric_templates'),
    'https://ndownloader.figshare.com/files/',
    pediatric_remote_fnames,
    pediatric_fnames,
    md5_list=pediatric_md5_hashes,
    doc='Download pediatric templates'
)


def read_pediatric_templates(as_img=True, resample_to=False):
    """
    Load pediatric pyAFQ templates.

    Used to create pediatric `bundle_dict`.

    Parameters
    ----------
    as_img : bool, optional
        If True, values are `Nifti1Image`. Otherwise, values are
        paths to Nifti files. Default: True
    resample_to : str or nibabel image class instance, optional
        A template image to resample to. Typically, this should be the
        template to which individual-level data are registered. Defaults to
        the MNI template. Default: False

    Returns
    -------
    dict :
        keys = names of template ROIs, and
        values = `Nifti1Image` from each of the ROI nifti files.
    """

    print('Loading pediatric templates...', flush=True)
    pediatric_templates = _fetcher_to_template(
        fetch_pediatric_templates,
        as_img=as_img,
        resample_to=resample_to)

    # For the arcuate (AF/ARC), reuse the SLF ROIs
    pediatric_templates['ARC_roi1_L'] = pediatric_templates['SLF_roi1_L']
    pediatric_templates['ARC_roi1_R'] = pediatric_templates['SLF_roi1_R']
    pediatric_templates['ARC_roi2_L'] = pediatric_templates['SLFt_roi2_L']
    pediatric_templates['ARC_roi2_R'] = pediatric_templates['SLFt_roi2_R']
    pediatric_templates['ARC_roi3_L'] = pediatric_templates['SLFt_roi3_L']
    pediatric_templates['ARC_roi3_R'] = pediatric_templates['SLFt_roi3_R']

    # For the middle longitudinal fasciculus (MdLF) reuse ILF ROI
    pediatric_templates['MdLF_roi2_L'] = pediatric_templates['ILF_roi2_L']
    pediatric_templates['MdLF_roi2_R'] = pediatric_templates['ILF_roi2_R']

    return pediatric_templates


def read_resample_roi(roi, resample_to=None, threshold=False):
    """
    Reads an roi from file-name/img and resamples it to conform with
    another file-name/img.

    Parameters
    ----------
    roi : str or nibabel image class instance.
        Should contain a binary volume with 1s in the region of interest and
        0s elsewhere.

    resample_to : str or nibabel image class instance, optional
        A template image to resample to. Typically, this should be the
        template to which individual-level data are registered. Defaults to
        the MNI template.

    threshold: bool or float
        If set to False (default), resampled result is returned. Otherwise,
        the resampled result is thresholded at this value and binarized.
        This is not applied if the input ROI is already in the space of the
        output.

    Returns
    -------
    nibabel image class instance that contains the binary ROI resampled into
    the requested space.
    """
    if isinstance(roi, str):
        roi = nib.load(roi)

    if resample_to is None:
        resample_to = read_mni_template()

    if isinstance(resample_to, str):
        resample_to = nib.load(resample_to)

    if resample_to is False or np.allclose(resample_to.affine, roi.affine):
        return roi

    as_array = resample(
        roi.get_fdata(),
        resample_to,
        roi.affine,
        resample_to.affine).get_fdata()
    if threshold:
        as_array = (as_array > threshold).astype(int)

    img = nib.Nifti1Image(
        as_array,
        resample_to.affine)

    return img


template_fnames = ["ATR_roi1_L.nii.gz",
                   "ATR_roi1_R.nii.gz",
                   "ATR_roi2_L.nii.gz",
                   "ATR_roi2_R.nii.gz",
                   "ATR_L_prob_map.nii.gz",
                   "ATR_R_prob_map.nii.gz",
                   "CGC_roi1_L.nii.gz",
                   "CGC_roi1_R.nii.gz",
                   "CGC_roi2_L.nii.gz",
                   "CGC_roi2_R.nii.gz",
                   "CGC_L_prob_map.nii.gz",
                   "CGC_R_prob_map.nii.gz",
                   "CST_roi1_L.nii.gz",
                   "CST_roi1_R.nii.gz",
                   "CST_roi2_L.nii.gz",
                   "CST_roi2_R.nii.gz",
                   "CST_L_prob_map.nii.gz",
                   "CST_R_prob_map.nii.gz",
                   "FA_L.nii.gz",
                   "FA_R.nii.gz",
                   "FA_prob_map.nii.gz",
                   "FP_L.nii.gz",
                   "FP_R.nii.gz",
                   "FP_prob_map.nii.gz",
                   "HCC_roi1_L.nii.gz",
                   "HCC_roi1_R.nii.gz",
                   "HCC_roi2_L.nii.gz",
                   "HCC_roi2_R.nii.gz",
                   "HCC_L_prob_map.nii.gz",
                   "HCC_R_prob_map.nii.gz",
                   "IFO_roi1_L.nii.gz",
                   "IFO_roi1_R.nii.gz",
                   "IFO_roi2_L.nii.gz",
                   "IFO_roi2_R.nii.gz",
                   "IFO_L_prob_map.nii.gz",
                   "IFO_R_prob_map.nii.gz",
                   "ILF_roi1_L.nii.gz",
                   "ILF_roi1_R.nii.gz",
                   "ILF_roi2_L.nii.gz",
                   "ILF_roi2_R.nii.gz",
                   "ILF_L_prob_map.nii.gz",
                   "ILF_R_prob_map.nii.gz",
                   "SLF_roi1_L.nii.gz",
                   "SLF_roi1_R.nii.gz",
                   "SLF_roi2_L.nii.gz",
                   "SLF_roi2_R.nii.gz",
                   "SLFt_roi2_L.nii.gz",
                   "SLFt_roi2_R.nii.gz",
                   "SLF_L_prob_map.nii.gz",
                   "SLF_R_prob_map.nii.gz",
                   "UNC_roi1_L.nii.gz",
                   "UNC_roi1_R.nii.gz",
                   "UNC_roi2_L.nii.gz",
                   "UNC_roi2_R.nii.gz",
                   "UNC_L_prob_map.nii.gz",
                   "UNC_R_prob_map.nii.gz",
                   "ARC_L_prob_map.nii.gz",
                   "ARC_R_prob_map.nii.gz"]


template_remote_fnames = ["5273680", "5273683", "5273686", "5273689",
                          "11458274", "11458277",
                          "5273695", "5273692", "5273698", "5273701",
                          "11458268", "11458271",
                          "5273704", "5273707", "5273710", "5273713",
                          "11458262", "11458265",
                          "5273716", "5273719",
                          "11458220",
                          "5273722", "5273725",
                          "11458226",
                          "5273728", "5273731", "5273734", "5273746",
                          "11458259", "11458256",
                          "5273737", "5273740", "5273743", "5273749",
                          "11458250", "11458253",
                          "5273752", "5273755", "5273758", "5273761",
                          "11458244", "11458247",
                          "5273764", "5273767", "5273770", "5273773",
                          "5273776", "5273791",
                          "11458238", "11458241",
                          "5273779", "5273782", "5273785", "5273788",
                          "11458223", "11458229",
                          "11458232", "11458235"]


template_md5_hashes = ["6b7aaed1a2982fd0ea436a223133908b",
                       "fd60d46d4e3cbd906c86e4c9e4fd6e2a",
                       "3aba60b169a35c38640de4ec29d362c8",
                       "12716a5688a1809fbaed1d58d2e68b59",
                       "c5637f471df861d9bbb45604db34770b",
                       "850cc4c04d7241747063fe3cd440b2ce",
                       "8e8973bc7838c8744914d402f52d91ca",
                       "c5fa4e6e685e695c006823b6784d2407",
                       "e1fab77f21d5303ed52285f015e24f0b",
                       "5f89defec3753fd75cd688c7bfb20a36",
                       "a4f3cd65b06fb25f63d5dab7592f00f2",
                       "7e73ab02db30a3ad6bd9e82148c2486e",
                       "f9db3154955a20b67c2dda758800d14c",
                       "73941510c798c1ed1b03e2bd481cd5c7",
                       "660cdc031ee0716d60159c7d933119ea",
                       "660cdc031ee0716d60159c7d933119ea",
                       "fd012bc89f6bed7bd54530195496bac4",
                       "3406906a86e633cc102127cf210a1063",
                       "9040a7953dcbbf131d135c866182d8ef",
                       "a72e17194824fcd838a594a2eb50c72e",
                       "627d7bb2e6d55f8243da815a36d9ff1a",
                       "55adbe9b8279185eedbe342149e1ff90",
                       "5a7412a3cf0fb185eec53d1989df2f7c",
                       "1aa36e83ae7b5555bb19d776ede9c18d",
                       "ba453196ff179b0e31172806e313b52c",
                       "d85c6574526b296935f34bf4f65cd493",
                       "9b81646317f59c7db087f27e2f85679e",
                       "9806e82c250e4604534b96917f87b7e8",
                       "213d3fb1ccd756d878f9b50b765b1c8f",
                       "f1e7e6bc51aa0aa279c54fb3805fb5e3",
                       "0e68a9feaaddcc9b4d667c2f15903368",
                       "d45020a87ee4bb496edd350631d91f6a",
                       "75c2c911826ec4b23159f9bd80e3c039",
                       "55d616ea9e0c646adc1aafa0f5fbe625",
                       "dee83fa6b03cfa5e0f5c965953aa6778",
                       "a13eef7059c98568adfefbab660e434e",
                       "045b7d5c6341997f3f0120c3a4212ad8",
                       "d174b1359ba982b03840436c93b7bbb4",
                       "fff9753f394fc4c73fb2ae40b3b4dde0",
                       "cd278b4dd6ff77481ea9ac16485a5ae2",
                       "7bdf5111265107091c7a2fca9215de30",
                       "7d4a43714504e6e930f922c9bc2a13d5",
                       "af2bcedf47e193686af329b9a8e259da",
                       "9a1122943579d11ba169d3ad87a75625",
                       "627903f7a06627bfd4153dc9245fa390",
                       "1714cd7f989c3435bdd5a2076e6272a0",
                       "1fa2114049707a4e05b53f9d95730375",
                       "b6663067d5ea53c70cb8803948f8adf7",
                       "d3e068997ebc60407bd6e9576e47dede",
                       "27ecfbd1d2f98213e52d73b7d70fe0e7",
                       "fa141bb2d951bec486916acda3652d95",
                       "d391d073e86e28588be9a6d01b2e7a82",
                       "a3e085562e6b8111c7ebc358f9450c8b",
                       "d65c67910807504735e034f7ea92d590",
                       "93cb24a9128db1a6c34a09eaf79fe7f0",
                       "71a7455cb4062dc39b1677c118c7b5a5",
                       "19590c712f1776da1fdba64d4eb7f1f6",
                       "04d5af0feb2c1b5b52a87ccbbf148e4b",
                       "53c277be990d00f7de04f2ea35e74d73"]

fetch_templates = _make_reusable_fetcher(
    "fetch_templates",
    op.join(afq_home, 'templates'),
    baseurl, template_remote_fnames,
    template_fnames, md5_list=template_md5_hashes,
    doc="Download AFQ templates")


def read_templates(as_img=True, resample_to=False):
    """Load AFQ templates from file

    Parameters
    ----------
    as_img : bool, optional
        If True, values are `Nifti1Image`. Otherwise, values are
        paths to Nifti files. Default: True
    resample_to : str or nibabel image class instance, optional
        A template image to resample to. Typically, this should be the
        template to which individual-level data are registered. Defaults to
        the MNI template. Default: False

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ')
    logger.debug('loading AFQ templates')
    tic = time.perf_counter()

    template_dict = _fetcher_to_template(
        fetch_templates,
        as_img=as_img,
        resample_to=resample_to)

    toc = time.perf_counter()
    logger.debug(f'AFQ templates loaded in {toc - tic:0.4f} seconds')

    return template_dict


or_fnames = [
    "left_thal_MNI.nii.gz",
    "left_V1_MNI.nii.gz",
    "right_thal_MNI.nii.gz",
    "right_V1_MNI.nii.gz",
    "left_OP_MNI.nii.gz",
    "left_OR_1.nii.gz",
    "left_OR_2.nii.gz",
    "left_pos_thal_MNI.nii.gz",
    "left_TP_MNI.nii.gz",
    "right_OP_MNI.nii.gz",
    "right_OR_1.nii.gz",
    "right_OR_2.nii.gz",
    "right_pos_thal_MNI.nii.gz",
    "right_TP_MNI.nii.gz",
]

or_remote_fnames = [
    "36514170",
    "26831633",
    "36514173",
    "26831639",
    "26831642",
    "26831645",
    "26831648",
    "26831651",
    "26831654",
    "26831657",
    "26831660",
    "26831663",
    "26831666",
    "26831669",
]

or_md5_hashes = [
    "a45126a727c4b5d843b2f7aae181825f",
    "ad996c67bf5cc59fc3a7b60255873b67",
    "7a75c3ddd25335277a099626dbc946ac",
    "cc88fb4671311404eb9dfa8fa11a59e0",
    "9cff03af586d9dd880750cef3e0bf63f",
    "ff728ba3ffa5d1600bcd19fdef8182c4",
    "4f1978e418a3169609375c28b3eba0fd",
    "fd163893081b520f4594171aeea04f39",
    "bf795d197912b5e074d248d2763c6930",
    "13efde1efe0de52683cbf352ecba457e",
    "75c7bd2092950578e599a2dcb218909f",
    "8f3890fa8c26a568503226757f7e7d6c",
    "f239aa3140809152da8884ff879dde1b",
    "60a748567e4dd81b40ad8967a14cb09e",
]

fetch_or_templates = _make_reusable_fetcher(
    "fetch_or_templates",
    op.join(afq_home,
            'or_templates'),
    baseurl, or_remote_fnames,
    or_fnames,
    md5_list=or_md5_hashes,
    doc="Download AFQ or templates")


def read_or_templates(as_img=True, resample_to=False):
    """Load AFQ OR templates from file

    Parameters
    ----------
    as_img : bool, optional
        If True, values are `Nifti1Image`. Otherwise, values are
        paths to Nifti files. Default: True
    resample_to : str or nibabel image class instance, optional
        A template image to resample to. Typically, this should be the
        template to which individual-level data are registered. Defaults to
        the MNI template. Default: False

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ')

    logger.debug('loading or templates')
    tic = time.perf_counter()

    template_dict = _fetcher_to_template(
        fetch_or_templates,
        as_img=as_img,
        resample_to=resample_to)

    toc = time.perf_counter()
    logger.debug(f'or templates loaded in {toc - tic:0.4f} seconds')

    return template_dict


stanford_hardi_tractography_remote_fnames = ["5325715", "5325718", "25289735"]
stanford_hardi_tractography_hashes = ['6f4bdae702031a48d1cd3811e7a42ef9',
                                      'f20854b4f710577c58bd01072cfb4de6',
                                      '294bfd1831861e8635eef8834ff18892']
stanford_hardi_tractography_fnames = [
    'mapping.nii.gz',
    'tractography_subsampled.trk',
    'full_segmented_cleaned_tractography.trk']

fetch_stanford_hardi_tractography = _make_reusable_fetcher(
    "fetch_stanford_hardi_tractography",
    op.join(afq_home,
            'stanford_hardi_tractography'),
    baseurl,
    stanford_hardi_tractography_remote_fnames,
    stanford_hardi_tractography_fnames,
    md5_list=stanford_hardi_tractography_hashes,
    doc="""Download Stanford HARDI tractography and mapping. For testing
           purposes""")


def read_stanford_hardi_tractography():
    """
    Reads a minimal tractography from the Stanford dataset.
    """
    files, folder = fetch_stanford_hardi_tractography()
    files_dict = {}
    files_dict['mapping.nii.gz'] = nib.load(
        op.join(afq_home,
                'stanford_hardi_tractography',
                'mapping.nii.gz'))

    # We need the original data as reference
    dwi_img, gtab = dpd.read_stanford_hardi()

    files_dict['tractography_subsampled.trk'] = load_tractogram(
        op.join(afq_home,
                'stanford_hardi_tractography',
                'tractography_subsampled.trk'),
        dwi_img,
        bbox_valid_check=False,
        trk_header_check=False).streamlines

    files_dict['full_segmented_cleaned_tractography.trk'] = load_tractogram(
        op.join(
            afq_home,
            'stanford_hardi_tractography',
            'full_segmented_cleaned_tractography.trk'),
        dwi_img).streamlines

    return files_dict


def to_bids_description(path, fname='dataset_description.json',
                        BIDSVersion="1.4.0", **kwargs):
    """Dumps a dict into a bids description at the given location"""
    kwargs.update({"BIDSVersion": BIDSVersion})
    desc_file = op.join(path, fname)
    with open(desc_file, 'w') as outfile:
        json.dump(kwargs, outfile)


def organize_cfin_data(path=None):
    """
    Create the expected file-system structure for the
    CFIN multi b-value diffusion data-set.
    """

    dpd.fetch_cfin_multib()
    if path is None:
        os.makedirs(afq_home, exist_ok=True)
        path = afq_home

    bids_path = op.join(path, 'cfin_multib',)
    derivatives_path = op.join(bids_path, 'derivatives')
    dmriprep_folder = op.join(derivatives_path, 'dmriprep')

    if not op.exists(derivatives_path):
        anat_folder = op.join(dmriprep_folder, 'sub-01', 'ses-01', 'anat')
        os.makedirs(anat_folder, exist_ok=True)
        dwi_folder = op.join(dmriprep_folder, 'sub-01', 'ses-01', 'dwi')
        os.makedirs(dwi_folder, exist_ok=True)
        t1_img = dpd.read_cfin_t1()
        nib.save(t1_img, op.join(anat_folder, 'sub-01_ses-01_T1w.nii.gz'))
        dwi_img, gtab = dpd.read_cfin_dwi()
        nib.save(dwi_img, op.join(dwi_folder, 'sub-01_ses-01_dwi.nii.gz'))
        np.savetxt(op.join(dwi_folder, 'sub-01_ses-01_dwi.bvec'), gtab.bvecs)
        np.savetxt(op.join(dwi_folder, 'sub-01_ses-01_dwi.bval'), gtab.bvals)

    to_bids_description(
        bids_path,
        **{"BIDSVersion": "1.0.0",
           "Name": "CFIN",
           "Subjects": ["sub-01"]})

    to_bids_description(
        dmriprep_folder,
        **{"Name": "CFIN",
           "PipelineDescription": {"Name": "dipy"}})


def organize_stanford_data(path=None, clear_previous_afq=False):
    """
    If necessary, downloads the Stanford HARDI dataset into DIPY directory and
    creates a BIDS compliant file-system structure in AFQ data directory:


    ~/AFQ_data/
    └── stanford_hardi
    ├── dataset_description.json
    └── derivatives
        ├── freesurfer
        │   ├── dataset_description.json
        │   └── sub-01
        │       └── ses-01
        │           └── anat
        │               ├── sub-01_ses-01_T1w.nii.gz
        │               └── sub-01_ses-01_seg.nii.gz
        └── vistasoft
            ├── dataset_description.json
            └── sub-01
                └── ses-01
                    └── dwi
                        ├── sub-01_ses-01_dwi.bval
                        ├── sub-01_ses-01_dwi.bvec
                        └── sub-01_ses-01_dwi.nii.gz

    If clear_previous_afq is True and there is an afq folder in derivatives,
    it will be removed.
    """
    logger = logging.getLogger('AFQ')

    # fetches data for first subject and session
    logger.info('fetching Stanford HARDI data')
    dpd.fetch_stanford_hardi()

    if path is None:
        if not op.exists(afq_home):
            logger.info(f'creating AFQ home directory: {afq_home}')
        os.makedirs(afq_home, exist_ok=True)
        path = afq_home

    bids_path = op.join(path, 'stanford_hardi',)
    derivatives_path = op.join(bids_path, 'derivatives')
    dmriprep_folder = op.join(derivatives_path, 'vistasoft')
    freesurfer_folder = op.join(derivatives_path, 'freesurfer')

    if clear_previous_afq:
        afq_folder = op.join(derivatives_path, 'afq')
        if op.exists(afq_folder):
            shutil.rmtree(afq_folder)

    if not op.exists(derivatives_path):
        logger.info(f'creating derivatives directory: {derivatives_path}')

        # anatomical data
        anat_folder = op.join(freesurfer_folder, 'sub-01', 'ses-01', 'anat')
        os.makedirs(anat_folder, exist_ok=True)

        t1_img = dpd.read_stanford_t1()
        nib.save(t1_img, op.join(anat_folder, 'sub-01_ses-01_T1w.nii.gz'))

        seg_img = dpd.read_stanford_labels()[-1]
        nib.save(seg_img, op.join(anat_folder,
                                  'sub-01_ses-01_seg.nii.gz'))

        # diffusion-weighted imaging data
        dwi_folder = op.join(dmriprep_folder, 'sub-01', 'ses-01', 'dwi')
        os.makedirs(dwi_folder, exist_ok=True)

        dwi_img, gtab = dpd.read_stanford_hardi()
        nib.save(dwi_img, op.join(dwi_folder, 'sub-01_ses-01_dwi.nii.gz'))
        np.savetxt(op.join(dwi_folder, 'sub-01_ses-01_dwi.bvec'), gtab.bvecs)
        np.savetxt(op.join(dwi_folder, 'sub-01_ses-01_dwi.bval'), gtab.bvals)
    else:
        logger.info('Dataset is already in place. If you want to fetch it '
                    + 'again please first remove the folder '
                    + derivatives_path)

    # Dump out the description of the dataset
    to_bids_description(bids_path,
                        **{"Name": "Stanford HARDI", "Subjects": ["sub-01"]})

    # And descriptions of the pipelines in the derivatives:
    to_bids_description(dmriprep_folder,
                        **{"Name": "Stanford HARDI",
                           "PipelineDescription": {"Name": "vistasoft"}})
    to_bids_description(freesurfer_folder,
                        **{"Name": "Stanford HARDI",
                           "PipelineDescription": {"Name": "freesurfer"}})


fetch_stanford_hardi_lv1 = _make_reusable_fetcher(
    "fetch_stanford_hardi_lv1",
    op.join(afq_home,
            'stanford_hardi',
            'derivatives/freesurfer/sub-01/ses-01/anat'),
    'https://stacks.stanford.edu/file/druid:ng782rw8378/',
    ["SUB1_LV1.nii.gz"],
    ["sub-01_ses-01_desc-LV1_anat.nii.gz"],
    md5_list=["e403c602e53e5491414f86af5f08a913"],
    doc="Download the LV1 segmentation for the Standord Hardi subject",
    unzip=False)


fetch_hcp_atlas_16_bundles = _make_reusable_fetcher(
    "fetch_hcp_atlas_16_bundles",
    op.join(afq_home,
            'hcp_atlas_16_bundles'),
    'https://ndownloader.figshare.com/files/',
    ["11921522"],
    ["atlas_16_bundles.zip"],
    md5_list=["b071f3e851f21ba1749c02fc6beb3118"],
    doc="Download minimal Recobundles atlas",
    unzip=True)


fetch_hcp_atlas_80_bundles = _make_reusable_fetcher(
    "fetch_hcp_atlas_80_bundles",
    op.join(afq_home,
            'hcp_atlas_80_bundles'),
    'https://ndownloader.figshare.com/files/',
    ["13638644"],
    ["Atlas_80_Bundles.zip"],
    md5_list=["78331d527a10ec000d4f33bac472e099"],
    doc="Download 80-bundle Recobundles atlas",
    unzip=True)


def read_hcp_atlas(n_bundles=16, as_file=False):
    """
    as_file : bool, optional
        If True, values are paths to sls. Otherwise, the sl
        are located and the centroids calculated. Default: False

    n_bundles : int
        16 or 80, which selects among the two different
        atlases:

        https://figshare.com/articles/Simple_model_bundle_atlas_for_RecoBundles/6483614  # noqa

        https://figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883  # noqa
    """
    bundle_dict = {}
    if n_bundles == 16:
        _, folder = fetch_hcp_atlas_16_bundles()
        atlas_folder = "Atlas_in_MNI_Space_16_bundles"
    elif n_bundles == 80:
        _, folder = fetch_hcp_atlas_80_bundles()
        atlas_folder = "Atlas_80_Bundles"

    whole_brain = op.join(
        folder,
        atlas_folder,
        'whole_brain',
        'whole_brain_MNI.trk')
    if not as_file:
        whole_brain = load_tractogram(
            whole_brain,
            'same', bbox_valid_check=False).streamlines

    bundle_dict['whole_brain'] = whole_brain
    bundle_files = glob(
        op.join(
            folder,
            atlas_folder,
            "bundles", "*.trk"))
    centroid_folder = op.join(
        folder,
        atlas_folder,
        "centroid")
    os.makedirs(centroid_folder, exist_ok=True)
    for bundle_file in bundle_files:
        bundle = drop_extension(op.split(bundle_file)[-1])
        centroid_file = op.join(centroid_folder, f"{bundle}.trk")
        bundle_dict[bundle] = {}
        if not op.exists(centroid_file):
            bundle_sl = load_tractogram(
                bundle_file,
                'same',
                bbox_valid_check=False)
            feature = ResampleFeature(nb_points=100)
            metric = AveragePointwiseEuclideanMetric(feature)
            qb = QuickBundles(np.inf, metric=metric)
            cluster = [qb.cluster(bundle_sl.streamlines).centroids[0]]
            save_tractogram(
                StatefulTractogram(
                    cluster, bundle_sl, Space.RASMM),
                centroid_file,
                bbox_valid_check=False)
        if not as_file:
            bundle_dict[bundle]['sl'] = load_tractogram(
                bundle_file,
                'same',
                bbox_valid_check=False).streamlines
            bundle_dict[bundle]['centroid'] = load_tractogram(
                centroid_file,
                "same", bbox_valid_check=False).streamlines
        else:
            bundle_dict[bundle]['sl'] = bundle_file
            bundle_dict[bundle]['centroid'] = centroid_file

    # For some reason, this file-name has a 0 in it, instead of an O:
    bundle_dict["IFOF_R"] = bundle_dict["IF0F_R"]
    # In the 80-bundle case, there are two files, and both have identical
    # content, so this is fine:
    del bundle_dict["IF0F_R"]
    return bundle_dict


fetch_aal_atlas = _make_reusable_fetcher(
    "fetch_aal_atlas",
    op.join(afq_home,
            'aal_atlas'),
    'https://ndownloader.figshare.com/files/',
    ["28416852",
     "28416855"],
    ["MNI_AAL_AndMore.nii.gz",
     "MNI_AAL.txt"],
    md5_list=["69395b75a16f00294a80eb9428bf7855",
              "59fd3284b17de2fbe411ca1c7afe8c65"],
    doc="Download the AAL atlas",
    unzip=False)


def read_aal_atlas(resample_to=None):
    """
    Reads the AAL atlas [1]_.

    Parameters
    ----------
    template : nib.Nifti1Image class instance, optional
        If provided, this is the template used and AAL atlas should be
        registered and aligned to this template


    .. [1] Tzourio-Mazoyer N, Landeau B, Papathanassiou D, Crivello F, Etard O,
           Delcroix N, Mazoyer B, Joliot M. (2002). Automated anatomical
           labeling of activations in SPM using a macroscopic anatomical
           parcellation of the MNI MRI single-subject brain. Neuroimage. 2002;
           15(1):273-89.
    """
    file_dict, folder = fetch_aal_atlas()
    out_dict = {}
    for f in file_dict:
        if f.endswith('.txt'):
            out_dict['labels'] = pd.read_csv(op.join(folder, f))
        else:
            out_dict['atlas'] = nib.load(op.join(folder, f))
    if resample_to is not None:
        data = out_dict['atlas'].get_fdata()
        oo = []
        for ii in range(data.shape[-1]):
            oo.append(resample(
                data[..., ii],
                resample_to,
                out_dict['atlas'].affine,
                resample_to.affine).get_fdata())
        out_dict['atlas'] = nib.Nifti1Image(np.stack(oo, -1),
                                            resample_to.affine)
    return out_dict


def aal_to_regions(regions, atlas=None):
    """
    Queries for large regions containing multiple AAL ROIs

    Parameters
    ----------
    regions : string or list of strings
        The name of the requested region. This can either be an AAL-defined ROI
        name (e.g, 'Occipital_Sup_L') or one of:
        {'leftfrontal' | 'leftoccipital' | 'lefttemporal' | 'leftparietal'
        | 'leftanttemporal' | 'leftparietal' | 'leftanttemporal'
        | 'leftuncinatefront' | 'leftifoffront' | 'leftinfparietal'
        | 'cerebellum' | 'leftarcfrontal' | 'leftarctemp' | 'leftcingpost'}
        each of which there is an equivalent 'right' region for. In addition,
        there are a few bilateral regions: {'occipital' | 'temporal'}, which
        encompass both the right and left region of this name, as well as:
        {'cstinferior' | 'cstsuperior'}

    atlas : 4D array
       Contains the AAL atlas in the correct coordinate frame with additional
       volumes for CST and cingulate ROIs ("AAL and more").

    Returns
    ------
    3D indices to the requested region in the atlas volume

    Notes
    -----
    Several regions can be referred to by multiple names:
           'leftuncinatetemp' = 'leftilftemp'= 'leftanttemporal'
           'rightuncinatetemp' = 'rightilftemp' = 'rightanttemporal'
           'leftslfpar'] = 'leftinfparietal'
           'rightslfpar' = 'rightinfparietal'
           'leftslffrontal' = 'leftarcfrontal'
           'rightslffrontal' = 'rightarcfrontal'
    """
    if atlas is None:
        atlas = read_aal_atlas()['atlas']
    atlas_vals = {'leftfrontal': np.arange(1, 26, 2),
                  # Occipital regions do not include fusiform:
                  'leftoccipital': np.arange(43, 54, 2),
                  # Temporal regions include fusiform:
                  'lefttemporal': np.concatenate([np.arange(37, 42, 2),
                                                  np.array([55]),
                                                  np.arange(79, 90, 2)]),
                  'leftanttemporal': np.array([41, 83, 87]),
                  'leftuncinatefront': np.array([5, 9, 15, 25]),
                  'leftifoffront': np.array([3, 5, 7, 9, 13, 15, 25]),
                  'leftinfparietal': np.array([61, 63, 65]),
                  'cerebellum': np.arange(91, 117),
                  'leftarcfrontal': np.array([1, 11, 13]),
                  'leftarctemp': np.array([79, 81, 85, 89]),
                  'leftthalamus': np.array([77]),
                  'leftventral': np.array([47, 53, 55, 89]),
                  'leftdorsal': np.array([49, 51]),
                  'leftparietal': np.array([59, 61, 63, 65])
                  }

    # Right symmetrical is off by one:
    atlas_vals['rightfrontal'] = atlas_vals['leftfrontal'] + 1
    atlas_vals['rightoccipital'] = atlas_vals['leftoccipital'] + 1
    atlas_vals['righttemporal'] = atlas_vals['lefttemporal'] + 1
    atlas_vals['rightanttemporal'] = atlas_vals['leftanttemporal'] + 1
    atlas_vals['rightuncinatefront'] = atlas_vals['leftuncinatefront'] + 1
    atlas_vals['rightifoffront'] = atlas_vals['leftifoffront'] + 1
    atlas_vals['rightinfparietal'] = atlas_vals['leftinfparietal'] + 1
    atlas_vals['rightarcfrontal'] = atlas_vals['leftarcfrontal'] + 1
    atlas_vals['rightarctemp'] = atlas_vals['leftarctemp'] + 1
    atlas_vals['rightthalamus'] = atlas_vals['leftthalamus'] + 1
    atlas_vals['rightventral'] = atlas_vals['leftventral'] + 1
    atlas_vals['rightdorsal'] = atlas_vals['leftdorsal'] + 1
    atlas_vals['rightparietal'] = atlas_vals['leftparietal'] + 1

    # Multiply named regions:
    atlas_vals['leftuncinatetemp'] = atlas_vals['leftilftemp'] =\
        atlas_vals['leftanttemporal']
    atlas_vals['rightuncinatetemp'] = atlas_vals['rightilftemp'] =\
        atlas_vals['rightanttemporal']
    atlas_vals['leftslfpar'] = atlas_vals['leftinfparietal']
    atlas_vals['rightslfpar'] = atlas_vals['rightinfparietal']
    atlas_vals['leftslffrontal'] = atlas_vals['leftarcfrontal']
    atlas_vals['rightslffrontal'] = atlas_vals['rightarcfrontal']

    # Bilateral regions:
    atlas_vals['occipital'] = np.union1d(atlas_vals['leftoccipital'],
                                         atlas_vals['rightoccipital'])
    atlas_vals['temporal'] = np.union1d(atlas_vals['lefttemporal'],
                                        atlas_vals['righttemporal'])

    if isinstance(regions, str):
        regions = [regions]

    idxes = []
    for region in regions:
        region = region.lower()  # Just to be sure
        if region in atlas_vals.keys():
            vol_idx = 0
            vals = atlas_vals[region]
        elif region == 'cstinferior':
            vol_idx = 1
            vals = np.array([1])
        elif region == 'cstsuperior':
            vol_idx = 2
            vals = np.array([1])
        elif region == 'leftcingpost':
            vol_idx = 3
            vals = np.array([1])
        elif region == 'rightcingpost':
            vol_idx = 4
            vals = np.array([1])

        # Broadcast vals, to test for equality over all three dimensions:
        is_in = atlas[..., vol_idx] == vals[:, None, None, None]
        # Then collapse the 4th dimension (each val), to get the 3D array:
        is_in = np.sum(is_in, 0)
        idxes.append(np.array(np.where(is_in)).T)

    return np.concatenate(idxes, axis=0)


def bundles_to_aal(bundles, atlas=None):
    """
    Given a sequence of AFQ bundle names, give back a sequence of lists
    with [target0, target1] being each NX3 arrays of the endpoint indices
    for the first and last node of the streamlines in this bundle.
    """
    if atlas is None:
        atlas = read_aal_atlas()['atlas']

    endpoint_dict = {
        "ATR_L": [['leftfrontal'], ['leftthalamus']],
        "ATR_R": [['rightfrontal'], ['rightthalamus']],
        "CST_L": [['cstinferior'], ['cstsuperior']],
        "CST_R": [['cstinferior'], ['cstsuperior']],
        "CGC_L": [['leftcingpost'], None],
        "CGC_R": [['rightcingpost'], None],
        "HCC_L": [None, None],
        "HCC_R": [None, None],
        "FP": [['rightoccipital'], ['leftoccipital']],
        "FA": [['rightfrontal'], ['leftfrontal']],
        "IFO_L": [['leftoccipital'], ['leftifoffront']],
        "IFO_R": [['rightoccipital'], ['rightifoffront']],
        "ILF_L": [['leftoccipital'], ['leftilftemp']],
        "ILF_R": [['rightoccipital'], ['rightilftemp']],
        "SLF_L": [['leftslffrontal'], ['leftinfparietal']],
        "SLF_R": [['rightslffrontal'], ['rightinfparietal']],
        "UNC_L": [['leftanttemporal'], ['leftuncinatefront']],
        "UNC_R": [['rightanttemporal'], ['rightuncinatefront']],
        "ARC_L": [['leftfrontal'], ['leftarctemp']],
        "ARC_R": [['rightfrontal'], ['rightarctemp']],
        "AntFrontal": [None, None],
        "Motor": [None, None],
        "Occipital": [None, None],
        "Orbital": [None, None],
        "PostParietal": [None, None],
        "SupFrontal": [None, None],
        "SupParietal": [None, None],
        "Temporal": [None, None],
        "pARC_L": [['leftparietal'], None],
        "pARC_R": [['rightparietal'], None],
        "VOF_L": [['leftdorsal'], ['leftventral']],
        "VOF_R": [['rightdorsal'], ['rightventral']]}

    targets = {}

    for bundle in bundles:
        if bundle in endpoint_dict:
            for region_name, region in zip(
                    ["start", "end"], endpoint_dict[bundle]):
                if region is None:
                    targets[bundle + "_" + region_name] = region
                else:
                    region_list = aal_to_regions(
                        region, atlas=atlas.get_fdata())
                    aal_roi = np.zeros(atlas.get_fdata().shape[:3])
                    aal_roi[region_list[:, 0],
                            region_list[:, 1],
                            region_list[:, 2]] = 1
                    targets[bundle + "_" + region_name] = nib.Nifti1Image(
                        aal_roi, atlas.affine)
        else:
            logger = logging.getLogger('AFQ')
            logger.warning(f"Segmentation end points undefined for {bundle},"
                           + " continuing without end points")
            targets[bundle + "_start"] = None
            targets[bundle + "_end"] = None

    return targets


def _apply_mask(template_img, resolution=1):
    """
    Helper function, gets MNI brain mask and applies it to template_img.

    Parameters
    ----------
    template_img : nib.Nifti1Image
        Unmasked template
    resolution : int, optional
        Resolution of mask. Default: 1

    Returns
    -------
    Masked template as nib.Nifti1Image
    """
    mask_img = nib.load(str(tflow.get('MNI152NLin2009cAsym',
                                      resolution=resolution,
                                      desc='brain',
                                      suffix='mask')))

    template_data = template_img.get_fdata()
    mask_data = mask_img.get_fdata()

    if mask_data.shape != template_data.shape:
        mask_img = nib.Nifti1Image(
            resample(
                mask_data,
                template_data,
                mask_img.affine,
                template_img.affine).get_fdata(),
            template_img.affine)
        mask_data = mask_img.get_fdata()

    out_data = template_data * mask_data
    return nib.Nifti1Image(out_data, template_img.affine)


def read_mni_template(resolution=1, mask=True, weight="T2w"):
    """

    Reads the MNI T1w or T2w template

    Parameters
    ----------
    resolution : int, optional.
        Either 1 or 2, the resolution in mm of the voxels. Default: 1.

    mask : bool, optional
        Whether to mask the data with a brain-mask before returning the image.
        Default : True

    weight: str, optional
        Which relaxation technique to use.
        Should be either "T2w" or "T1w".
        Default : "T2w"

    Returns
    -------
    nib.Nifti1Image class instance
    containing masked or unmasked T1w or template.

    """
    template_img = nib.load(str(tflow.get('MNI152NLin2009cAsym',
                                          desc=None,
                                          resolution=resolution,
                                          suffix=weight,
                                          extension='nii.gz')))
    if not mask:
        return template_img
    else:
        return _apply_mask(template_img, resolution)


fetch_biobank_templates = \
    _make_reusable_fetcher(
        "fetch_biobank_templates",
        op.join(afq_home,
                'biobank_templates'),
        "http://biobank.ctsu.ox.ac.uk/showcase/showcase/docs/",
        ["bmri_group_means.zip"],
        ["bmri_group_means.zip"],
        data_size="1.1 GB",
        doc="Download UK Biobank templates",
        unzip=True)


def read_ukbb_fa_template(mask=True):
    """
    Reads the UK Biobank FA template

    Parameters
    ----------
    mask : bool, optional
        Whether to mask the data with a brain-mask before returning the image.
        Default : True

    Returns
    -------
    nib.Nifti1Image class instance containing the FA template.

    """
    fa_folder = op.join(
        afq_home,
        'biobank_templates',
        'UKBiobank_BrainImaging_GroupMeanTemplates'
    )
    fa_path = op.join(
        fa_folder,
        'dti_FA.nii.gz'
    )

    if not op.exists(fa_path):
        logger = logging.getLogger('AFQ')
        logger.warning(
            "Downloading brain MRI group mean statistics from UK Biobank. "
            + "This download is approximately 1.1 GB. "
            + "It is currently necessary to access the FA template.")

        files, folder = fetch_biobank_templates()

        # remove zip
        for filename in files:
            os.remove(op.join(folder, filename))

        # remove non-FA related directories
        for filename in os.listdir(fa_folder):
            full_path = op.join(fa_folder, filename)
            if full_path != fa_path:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                else:
                    shutil.rmtree(full_path)

    template_img = nib.load(fa_path)

    if not mask:
        return template_img
    else:
        return _apply_mask(template_img, 1)


def fetch_hcp(subjects,
              hcp_bucket='hcp-openaccess',
              profile_name="hcp",
              path=None,
              study='HCP_1200',
              aws_access_key_id=None,
              aws_secret_access_key=None):
    """
    Fetch HCP diffusion data and arrange it in a manner that resembles the
    BIDS [1]_ specification.

    Parameters
    ----------
    subjects : list
        Each item is an integer, identifying one of the HCP subjects
    hcp_bucket : string, optional
        The name of the HCP S3 bucket. Default: "hcp-openaccess"
    profile_name : string, optional
        The name of the AWS profile used for access. Default: "hcp"
    path : string, optional
        Path to save files into. Default: '~/AFQ_data'
    study : string, optional
        Which HCP study to grab. Default: 'HCP_1200'
    aws_access_key_id : string, optional
        AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
        set to False.
    aws_secret_access_key : string, optional
        AWS credentials to HCP AWS S3. Will only be used if `profile_name` is
        set to False.

    Returns
    -------
    dict with remote and local names of these files,
    path to BIDS derivative dataset

    Notes
    -----
    To use this function with its default setting, you need to have a
    file '~/.aws/credentials', that includes a section:

    [hcp]
    AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
    AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX

    The keys are credentials that you can get from HCP
    (see https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)  # noqa

    Local filenames are changed to match our expected conventions.

    .. [1] Gorgolewski et al. (2016). The brain imaging data structure,
           a format for organizing and describing outputs of neuroimaging
           experiments. Scientific Data, 3::160044. DOI: 10.1038/sdata.2016.44.
    """
    if profile_name:
        boto3.setup_default_session(profile_name=profile_name)
    elif aws_access_key_id is not None and aws_secret_access_key is not None:
        boto3.setup_default_session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)
    else:
        raise ValueError("Must provide either a `profile_name` or ",
                         "both `aws_access_key_id` and ",
                         "`aws_secret_access_key` as input to 'fetch_hcp'")

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(hcp_bucket)

    if path is None:
        if not op.exists(afq_home):
            os.mkdir(afq_home)
        my_path = afq_home
    else:
        my_path = path

    base_dir = op.join(my_path, study, 'derivatives', 'dmriprep')

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    data_files = {}
    for subject in subjects:
        # We make a single session folder per subject for this case, because
        # AFQ api expects session structure:
        sub_dir = op.join(base_dir, f'sub-{subject}')
        sess_dir = op.join(sub_dir, "ses-01")
        if not os.path.exists(sub_dir):
            os.makedirs(os.path.join(sess_dir, 'dwi'), exist_ok=True)
            os.makedirs(os.path.join(sess_dir, 'anat'), exist_ok=True)
        data_files[op.join(sess_dir, 'dwi', f'sub-{subject}_dwi.bval')] =\
            f'{study}/{subject}/T1w/Diffusion/bvals'
        data_files[op.join(sess_dir, 'dwi', f'sub-{subject}_dwi.bvec')] =\
            f'{study}/{subject}/T1w/Diffusion/bvecs'
        data_files[op.join(sess_dir, 'dwi', f'sub-{subject}_dwi.nii.gz')] =\
            f'{study}/{subject}/T1w/Diffusion/data.nii.gz'
        data_files[op.join(sess_dir, 'anat', f'sub-{subject}_T1w.nii.gz')] =\
            f'{study}/{subject}/T1w/T1w_acpc_dc.nii.gz'
        data_files[op.join(sess_dir, 'anat',
                           f'sub-{subject}_aparc+aseg_seg.nii.gz')] =\
            f'{study}/{subject}/T1w/aparc+aseg.nii.gz'

    download_files = {}
    for k in data_files.keys():
        if not op.exists(k):
            download_files[k] = data_files[k]
    if len(download_files.keys()):
        with tqdm(total=len(download_files.keys())) as pbar:
            for k in download_files.keys():
                pbar.set_description_str(f"Downloading {k}")
                bucket.download_file(data_files[k], k)
                pbar.update()

    # Create the BIDS dataset description file text
    hcp_acknowledgements = """Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.""",  # noqa
    to_bids_description(op.join(my_path, study),
                        **{"Name": study,
                           "Acknowledgements": hcp_acknowledgements,
                           "Subjects": subjects})

    # Create the BIDS derivatives description file text
    to_bids_description(base_dir,
                        **{"Name": study,
                           "Acknowledgements": hcp_acknowledgements,
                           "PipelineDescription": {'Name': 'dmriprep'}})

    return data_files, op.join(my_path, study)


def fetch_hbn_preproc(subjects, path=None):
    """
    Fetches data from the Healthy Brain Network POD2 study [1, 2]_.

    Parameters
    ----------
    subjects : list
        Identifiers of the subjects to download.
        For example: ["NDARAA112DMH", "NDARAA117NEJ"].
    path : string, optional
        Path to save files into. Default: '~/AFQ_data'

    Returns
    -------
    dict with remote and local names of these files,
    path to BIDS derivative dataset

    Notes
    -----

    .. [1] Alexander LM, Escalera J, Ai L, et al. An open resource for
        transdiagnostic research in pediatric mental health and learning
        disorders. Sci Data. 2017;4:170181.

    .. [2] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and
        quality controlled resource for pediatric brain white-matter research.
        Scientific Data. 2022;9(1):1-27.

    """
    # Anonymous access:
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    if path is None:
        if not op.exists(afq_home):
            os.mkdir(afq_home)
        my_path = afq_home
    else:
        my_path = path

    base_dir = op.join(my_path, "HBN", 'derivatives', 'qsiprep')

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    data_files = {}

    for subject in subjects:
        initial_query = client.list_objects(
            Bucket="fcp-indi",
            Prefix=f"data/Projects/HBN/BIDS_curated/sub-{subject}/")
        ses = initial_query['Contents'][0]["Key"].split('/')[5]
        query = client.list_objects(
            Bucket="fcp-indi",
            Prefix=f"data/Projects/HBN/BIDS_curated/derivatives/qsiprep/sub-{subject}/")  # noqa
        file_list = [kk["Key"] for kk in query["Contents"]]
        sub_dir = op.join(base_dir, f'sub-{subject}')
        ses_dir = op.join(sub_dir, ses)
        if not os.path.exists(sub_dir):
            os.makedirs(os.path.join(sub_dir, 'anat'), exist_ok=True)
            os.makedirs(os.path.join(sub_dir, 'figures'), exist_ok=True)
            os.makedirs(os.path.join(ses_dir, 'dwi'), exist_ok=True)
            os.makedirs(os.path.join(ses_dir, 'anat'), exist_ok=True)
        for remote in file_list:
            full = remote.split("Projects")[-1][1:].replace("/BIDS_curated", "")
            local = op.join(afq_home, full)
            data_files[local] = remote

    download_files = {}
    for k in data_files.keys():
        if not op.exists(k):
            download_files[k] = data_files[k]
    if len(download_files.keys()):
        with tqdm(total=len(download_files.keys())) as pbar:
            for k in download_files.keys():
                pbar.set_description_str(f"Downloading {k}")
                client.download_file("fcp-indi", download_files[k], k)
                pbar.update()

    # Create the BIDS dataset description file text
    hbn_acknowledgements = """XXX""",  # noqa
    to_bids_description(op.join(my_path, "HBN"),
                        **{"Name": "HBN",
                           "Acknowledgements": hbn_acknowledgements,
                           "Subjects": subjects})

    # Create the BIDS derivatives description file text
    to_bids_description(base_dir,
                        **{"Name": "HBN",
                           "Acknowledgements": hbn_acknowledgements,
                           "PipelineDescription": {'Name': 'qsiprep'}})

    return data_files, op.join(my_path, "HBN")


def fetch_hbn_afq(subjects, path=None):
    """
    Fetches AFQ derivatives for Healthy Brain Network POD2 study [1, 2]_.

    Parameters
    ----------
    subjects : list
        Identifiers of the subjects to download.
        For example: ["NDARAA112DMH", "NDARAA117NEJ"].
    path : string, optional
        Path to save files into. Default: '~/AFQ_data'

    Returns
    -------
    dict with remote and local names of these files,
    path to BIDS derivative dataset

    Notes
    -----

    .. [1] Alexander LM, Escalera J, Ai L, et al. An open resource for
        transdiagnostic research in pediatric mental health and learning
        disorders. Sci Data. 2017;4:170181.

    .. [2] Richie-Halford A, Cieslak M, Ai L, et al. An analysis-ready and
        quality controlled resource for pediatric brain white-matter research.
        Scientific Data. 2022;9(1):1-27.

    """
    # Anonymous access:
    client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    if path is None:
        if not op.exists(afq_home):
            os.mkdir(afq_home)
        my_path = afq_home
    else:
        my_path = path


    base_dir = op.join(my_path, "HBN", 'derivatives', 'afq')

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    data_files = {}

    for subject in subjects:
        initial_query = client.list_objects(
            Bucket="fcp-indi",
            Prefix=f"data/Projects/HBN/BIDS_curated/sub-{subject}/")
        ses = initial_query['Contents'][0]["Key"].split('/')[5]
        query = client.list_objects(
            Bucket="fcp-indi",
            Prefix=f"data/Projects/HBN/BIDS_curated/derivatives/afq/sub-{subject}/")  # noqa
        file_list = [kk["Key"] for kk in query["Contents"]]
        sub_dir = op.join(base_dir, f'sub-{subject}')
        ses_dir = op.join(sub_dir, ses)

        for deriv_dir in ["bundles",
                        "clean_bundles",
                        "ROIs",
                        "tract_profile_plots",
                        "viz_bundles"]:
            this_deriv = os.path.join(ses_dir, deriv_dir)
            if not os.path.exists(this_deriv):
                os.makedirs(this_deriv, exist_ok=True)
        for remote in file_list:
            full = remote.split("Projects")[-1][1:].replace("/BIDS_curated", "")
            local = op.join(afq_home, full)
            data_files[local] = remote

    download_files = {}
    for k in data_files.keys():
        if not op.exists(k):
            download_files[k] = data_files[k]
    if len(download_files.keys()):
        with tqdm(total=len(download_files.keys())) as pbar:
            for k in download_files.keys():
                pbar.set_description_str(f"Downloading {k}")
                client.download_file("fcp-indi", download_files[k], k)
                pbar.update()

    # Create the BIDS dataset description file text
    hbn_acknowledgements = """XXX""",  # noqa
    to_bids_description(op.join(my_path, "HBN"),
                        **{"Name": "HBN",
                           "Acknowledgements": hbn_acknowledgements,
                           "Subjects": subjects})

    # Create the BIDS derivatives description file text
    to_bids_description(base_dir,
                        **{"Name": "HBN",
                           "PipelineDescription": {'Name': 'afq'}})

    return data_files, op.join(my_path, "HBN")
