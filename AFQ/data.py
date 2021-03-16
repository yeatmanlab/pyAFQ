from dipy.align import resample
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import (AveragePointwiseEuclideanMetric,
                                 ResampleFeature)
from dipy.io.streamline import load_tractogram, load_trk
from dipy.data.fetcher import _make_fetcher
import dipy.data as dpd
from io import BytesIO
import gzip
import os
import os.path as op
import json
from glob import glob
import shutil

import boto3
import s3fs

import numpy as np
import pandas as pd
import logging
import time

from bids import BIDSLayout
import bids.config as bids_config
try:
    bids_config.set_option('extension_initial_dot', True)
except ValueError:
    pass
from botocore import UNSIGNED
from botocore.client import Config
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from pathlib import Path
from tqdm.auto import tqdm
import nibabel as nib

# capture templateflow resource warning and log
import warnings
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
           "fetch_templates", "read_templates", "fetch_hcp",
           "fetch_stanford_hardi_tractography",
           "read_stanford_hardi_tractography",
           "organize_stanford_data"]


BUNDLE_RECO_2_AFQ = \
    {
        "AF_L": "ARC_L", "AF_R": "ARC_R",
        "UF_L": "UNC_L", "UF_R": "UNC_R",
        "IFOF_L": "IFO_L", "IFOF_R": "IFO_R",
        "CST_L": "CST_L", "CST_R": "CST_R",
        "ILF_L": "ILF_L", "ILF_R": "ILF_R",
        "SLF_L": "SLF_L", "SLF_R": "SLF_R"
    }

BUNDLE_MAT_2_PYTHON = \
    {'Right Corticospinal': 'CST_R', 'Left Corticospinal': 'CST_L',
     'RightCorticospinal': 'CST_R', 'LeftCorticospinal': 'CST_L',
     'Right Uncinate': 'UNC_R', 'Left Uncinate': 'UNC_L',
     'RightUncinate': 'UNC_R', 'LeftUncinate': 'UNC_L',
     'Left IFOF': 'IFO_L', 'Right IFOF': 'IFO_R',
     'LeftIFOF': 'IFO_L', 'RightIFOF': 'IFO_R',
     'Right Arcuate': 'ARC_R', 'Left Arcuate': 'ARC_L',
     'RightArcuate': 'ARC_R', 'LeftArcuate': 'ARC_L',
     'Right Thalamic Radiation': 'ATR_R', 'Left Thalamic Radiation': 'ATR_L',
     'RightThalamicRadiation': 'ATR_R', 'LeftThalamicRadiation': 'ATR_L',
     'Right Cingulum Cingulate': 'CGC_R', 'Left Cingulum Cingulate': 'CGC_L',
     'RightCingulumCingulate': 'CGC_R', 'LeftCingulumCingulate': 'CGC_L',
     'Right Cingulum Hippocampus': 'HCC_R',
     'Left Cingulum Hippocampus': 'HCC_L',
     'RightCingulumHippocampus': 'HCC_R',
     'LeftCingulumHippocampus': 'HCC_L',
     'Callosum Forceps Major': 'FP', 'Callosum Forceps Minor': 'FA',
     'CallosumForcepsMajor': 'FP', 'CallosumForcepsMinor': 'FA',
     'Right ILF': 'ILF_R', 'Left ILF': 'ILF_L',
     'RightILF': 'ILF_R', 'LeftILF': 'ILF_L',
     'Right SLF': 'SLF_R', 'Left SLF': 'SLF_L',
     'RightSLF': 'SLF_R', 'LeftSLF': 'SLF_L'}

afq_home = op.join(op.expanduser('~'), 'AFQ_data')

baseurl = "https://ndownloader.figshare.com/files/"

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

fetch_callosum_templates = _make_fetcher("fetch_callosum_templates",
                                         op.join(afq_home,
                                                 'callosum_templates'),
                                         baseurl, callosum_remote_fnames,
                                         callosum_fnames,
                                         md5_list=callosum_md5_hashes,
                                         doc="Download AFQ callosum templates")


def read_callosum_templates(resample_to=False):
    """Load AFQ callosum templates from file

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ.data')

    files, folder = fetch_callosum_templates()

    logger.debug('loading callosum templates')
    tic = time.perf_counter()

    template_dict = {}
    for f in files:
        img = nib.load(op.join(folder, f))
        if resample_to:
            if isinstance(resample_to, str):
                resample_to = nib.load(resample_to)
            img = nib.Nifti1Image(resample(img.get_fdata(),
                                           resample_to,
                                           img.affine,
                                           resample_to.affine).get_fdata(),
                                  resample_to.affine)
        template_dict[f.split('.')[0]] = img

    toc = time.perf_counter()
    logger.debug(f'callosum templates loaded in {toc - tic:0.4f} seconds')

    return template_dict


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

    if np.allclose(resample_to.affine, roi.affine):
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

fetch_templates = _make_fetcher("fetch_templates",
                                op.join(afq_home, 'templates'),
                                baseurl, template_remote_fnames,
                                template_fnames, md5_list=template_md5_hashes,
                                doc="Download AFQ templates")


def read_templates(resample_to=False):
    """Load AFQ templates from file

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ.data')

    files, folder = fetch_templates()

    logger.debug('loading AFQ templates')
    tic = time.perf_counter()

    template_dict = {}
    for f in files:
        img = nib.load(op.join(folder, f))
        if resample_to:
            if isinstance(resample_to, str):
                resample_to = nib.load(resample_to)
            img = nib.Nifti1Image(
                resample(
                    img.get_fdata(),
                    resample_to,
                    img.affine,
                    resample_to.affine).get_fdata(),
                resample_to.affine)
        template_dict[f.split('.')[0]] = img

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
    "26831630",
    "26831633",
    "26831636",
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
    "c18f3f82c26f334dc26b96d21f026dd1",
    "ad996c67bf5cc59fc3a7b60255873b67",
    "786fb4ba915599f746950acd980e5b03",
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

fetch_or_templates = _make_fetcher("fetch_or_templates",
                                   op.join(afq_home,
                                           'or_templates'),
                                   baseurl, or_remote_fnames,
                                   or_fnames,
                                   md5_list=or_md5_hashes,
                                   doc="Download AFQ or templates")


def read_or_templates(resample_to=False):
    """Load AFQ OR templates from file

    Returns
    -------
    dict with: keys: names of template ROIs and values: nibabel Nifti1Image
    objects from each of the ROI nifti files.
    """
    logger = logging.getLogger('AFQ.data')

    files, folder = fetch_or_templates()

    logger.debug('loading or templates')
    tic = time.perf_counter()

    template_dict = {}
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
        template_dict[f.split('.')[0]] = img

    toc = time.perf_counter()
    logger.debug(f'or templates loaded in {toc - tic:0.4f} seconds')

    return template_dict



# +----------------------------------------------------+
# | Begin S3BIDSStudy classes and supporting functions |
# +----------------------------------------------------+
def get_s3_client(anon=True):
    """Return a boto3 s3 client

    Global boto clients are not thread safe so we use this function
    to return independent session clients for different threads.

    Parameters
    ----------
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    s3_client : boto3.client('s3')
    """
    session = boto3.session.Session()
    if anon:
        s3_client = session.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
    else:
        s3_client = session.client('s3')

    return s3_client


def _ls_s3fs(s3_prefix, anon=True):
    """Returns a dict of list of files using s3fs

    The files are divided between subject directories/files and
    non-subject directories/files.

    Parameters
    ----------
    s3_prefix : str
        AWS S3 key for the study or site "directory" that contains all
        of the subjects

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    subjects : dict
    """
    fs = s3fs.S3FileSystem(anon=anon)
    site_files = fs.ls(s3_prefix, detail=False)

    # Just need BIDSLayout for the `parse_file_entities` method
    # so we can pass dev/null as the argument
    layout = BIDSLayout(os.devnull, validate=False)

    entities = [
        layout.parse_file_entities(f) for f in site_files
    ]

    files = {
        'subjects': [
            f for f, e in zip(site_files, entities)
            if e.get('subject') is not None
        ],
        'other': [
            f for f, e in zip(site_files, entities)
            if e.get('subject') is None
        ]
    }

    return files


def _get_matching_s3_keys(bucket, prefix='', suffix='', anon=True):
    """Generate all the matching keys in an S3 bucket.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket

    prefix : str, optional
        Only fetch keys that start with this prefix

    suffix : str, optional
        Only fetch keys that end with this suffix

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Yields
    ------
    key : list
        S3 keys that match the prefix and suffix
    """
    s3 = get_s3_client(anon=anon)
    kwargs = {'Bucket': bucket, 'MaxKeys': 1000}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str) and prefix:
        kwargs['Prefix'] = prefix

    while True:
        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp['Contents']
        except KeyError:
            return

        for obj in contents:
            key = obj['Key']
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        except KeyError:
            break


def _download_from_s3(fname, bucket, key, overwrite=False, anon=True):
    """Download object from S3 to local file

    Parameters
    ----------
    fname : str
        File path to which to download the object

    bucket : str
        S3 bucket name

    key : str
        S3 key for the object to download

    overwrite : bool
        If True, overwrite file if it already exists.
        If False, skip download and return. Default: False

    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True
    """
    # Create the directory and file if necessary
    fs = s3fs.S3FileSystem(anon=anon)
    if overwrite or not op.exists(fname):
        Path(op.dirname(fname)).mkdir(parents=True, exist_ok=True)
        fs.get("/".join([bucket, key]), fname)


class S3BIDSSubject:
    """A single study subject hosted on AWS S3"""

    def __init__(self, subject_id, study):
        """Initialize a Subject instance

        Parameters
        ----------
        subject_id : str
            Subject-ID for this subject

        study : AFQ.data.S3BIDSStudy
            The S3BIDSStudy for which this subject was a participant
        """
        logging.getLogger("botocore").setLevel(logging.WARNING)

        if not isinstance(subject_id, str):
            raise TypeError('subject_id must be a string.')

        if not isinstance(study, S3BIDSStudy):
            raise TypeError('study must be an instance of S3BIDSStudy.')

        self._subject_id = subject_id
        self._study = study
        self._get_s3_keys()
        self._files = {'raw': {}, 'derivatives': {}}

    @property
    def subject_id(self):
        """An identifier string for the subject"""
        return self._subject_id

    @property
    def study(self):
        """The study in which this subject participated"""
        return self._study

    @property
    def s3_keys(self):
        """A dict of S3 keys for this subject's data

        The S3 keys are divided between "raw" data and derivatives
        """
        return self._s3_keys

    @property
    def files(self):
        """Local files for this subject's dMRI data

        Before the call to subject.download(), this is None.
        Afterward, the files are stored in a dict with keys
        for each Amazon S3 key and values corresponding to
        the local file.
        """
        return self._files

    def __repr__(self):
        return (f'{type(self).__name__}(subject_id={self.subject_id}, '
                f'study_id={self.study.study_id}')

    def _get_s3_keys(self):
        """Get all required S3 keys for this subject

        Returns
        -------
        s3_keys : dict
            S3 keys organized into "raw" and "derivatives" lists
        """
        prefixes = {
            'raw': '/'.join([self.study.s3_prefix,
                             self.subject_id]).lstrip('/'),
            'derivatives': {
                dt: '/'.join([
                    *dt.split('/')[1:],  # removes bucket name
                    self.subject_id
                ]).lstrip('/') for dt in self.study.derivative_types
            },
        }

        s3_keys = {
            'raw': list(set(_get_matching_s3_keys(
                bucket=self.study.bucket,
                prefix=prefixes['raw'],
                anon=self.study.anon,
            ))),
            'derivatives': {
                dt: list(set(_get_matching_s3_keys(
                    bucket=self.study.bucket,
                    prefix=prefixes['derivatives'][dt],
                    anon=self.study.anon,
                ))) for dt in self.study.derivative_types
            }
        }

        self._s3_keys = s3_keys

    def download(self, directory, include_derivs=False, overwrite=False,
                 suffix=None, pbar=True, pbar_idx=0):
        """Download files from S3

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ['dmriprep', 'afq']). Default: False

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        suffix : str
            Suffix, including extension, of file(s) to download.
            Default: None

        pbar : bool
            If True, include download progress bar. Default: True

        pbar_idx : int
            Progress bar index for multithreaded progress bars. Default: 0
        """
        if not isinstance(directory, str):
            raise TypeError('directory must be a string.')

        if not (isinstance(include_derivs, bool)
                or isinstance(include_derivs, str)
                or all(isinstance(s, str) for s in include_derivs)):
            raise TypeError('include_derivs must be a boolean, a '
                            'string, or a sequence of strings.')

        if not isinstance(overwrite, bool):
            raise TypeError('overwrite must be a boolean.')

        if (suffix is not None) and not(isinstance(suffix, str)):
            raise TypeError('suffix must be a string.')

        if not isinstance(pbar, bool):
            raise TypeError('pbar must be a boolean.')

        if not isinstance(pbar_idx, int):
            raise TypeError('pbar_idx must be an integer.')

        def split_key(key):
            if self.study.s3_prefix:
                return key.split(self.study.s3_prefix)[-1]
            else:
                return key

        # Filter out keys that do not end with suffix
        if suffix is not None:
            s3_keys_raw = [
                s3key for s3key in self.s3_keys['raw']
                if s3key.endswith(suffix)
            ]
            s3_keys_deriv = {
                dt: [
                    s3key for s3key in s3keys if s3key.endswith(suffix)
                ] for dt, s3keys in self.s3_keys['derivatives'].items()
            }
        else:
            s3_keys_raw = self.s3_keys['raw']
            s3_keys_deriv = self.s3_keys['derivatives']

        files = {
            'raw': [
                op.abspath(op.join(
                    directory,
                    split_key(key).lstrip('/')
                )) for key in s3_keys_raw
            ],
            'derivatives': {
                dt: [
                    op.abspath(op.join(
                        directory,
                        split_key(s3key).lstrip('/')
                    )) for s3key in s3keys
                ] for dt, s3keys in s3_keys_deriv.items()
            }
        }

        raw_zip = list(zip(s3_keys_raw, files['raw']))

        # Populate files parameter
        self._files["raw"].update({k: f for k, f in raw_zip})

        # Generate list of (key, file) tuples
        download_pairs = [(k, f) for k, f in raw_zip]

        deriv_zips = {
            dt: list(zip(
                s3keys, files['derivatives'][dt]
            )) for dt, s3keys in s3_keys_deriv.items()
        }

        deriv_pairs = []
        for dt in files['derivatives'].keys():
            if include_derivs is True:
                # In this case, include all derivatives files
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }
            elif include_derivs is False:
                pass
            elif (isinstance(include_derivs, str)
                  # In this case, filter only derivatives S3 keys that
                  # include the `include_derivs` string as a substring
                  and include_derivs in dt):
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }
            elif (all(isinstance(s, str) for s in include_derivs)
                  and any([deriv in dt for deriv in include_derivs])):
                # In this case, filter only derivatives S3 keys that
                # include any of the `include_derivs` strings as a
                # substring
                deriv_pairs += [(k, f) for k, f in deriv_zips[dt]]
                self._files['derivatives'][dt] = {
                    k: f for k, f in deriv_zips[dt]
                }

        if include_derivs is not False:
            download_pairs += deriv_pairs

        # Now iterate through the list and download each item
        if pbar:
            progress = tqdm(desc=f'Download {self.subject_id}',
                            position=pbar_idx,
                            total=len(download_pairs) + 1)

        for (key, fname) in download_pairs:
            _download_from_s3(fname=fname,
                              bucket=self.study.bucket,
                              key=key,
                              overwrite=overwrite,
                              anon=self.study.anon)

            if pbar:
                progress.update()

        if pbar:
            progress.update()
            progress.close()


class HBNSubject(S3BIDSSubject):
    """A subject in the HBN study

    See Also
    --------
    AFQ.data.S3BIDSSubject
    """

    def __init__(self, subject_id, study, site=None):
        """Initialize a Subject instance

        Parameters
        ----------
        subject_id : str
            Subject-ID for this subject

        study : AFQ.data.S3BIDSStudy
            The S3BIDSStudy for which this subject was a participant

        site : str, optional
            Site-ID for the site from which this subject's data was collected
        """
        if not (site is None or isinstance(site, str)):
            raise TypeError('site must be a string or None.')

        self._site = site

        super().__init__(
            subject_id=subject_id,
            study=study
        )

    @property
    def site(self):
        """The site at which this subject was a participant"""
        return self._site

    def __repr__(self):
        return (f'{type(self).__name__}(subject_id={self.subject_id}, '
                f'study_id={self.study.study_id}, site={self.site}')

    def _get_s3_keys(self):
        """Get all required S3 keys for this subject

        Returns
        -------
        s3_keys : dict
            S3 keys organized into "raw" and "derivatives" lists
        """
        prefixes = {
            'raw': '/'.join([self.study.s3_prefix,
                             self.subject_id]).lstrip('/'),
            'derivatives': '/'.join([
                self.study.s3_prefix,
                'derivatives',
                self.subject_id
            ]).lstrip('/')
        }

        s3_keys = {
            datatype: list(set(_get_matching_s3_keys(
                bucket=self.study.bucket,
                prefix=prefix,
                anon=self.study.anon,
            ))) for datatype, prefix in prefixes.items()
        }

        def get_deriv_type(s3_key):
            after_sub = s3_key.split('/' + self.subject_id + '/')[-1]
            deriv_type = after_sub.split('/')[0]

        deriv_keys = {
            dt: [
                s3key for s3key in s3_keys['derivatives']
                if dt == get_deriv_type(s3key)
            ] for dt in self.study.derivative_types
        }

        s3_keys['derivatives'] = deriv_keys
        self._s3_keys = s3_keys


class S3BIDSStudy:
    """A BIDS-compliant study hosted on AWS S3"""

    def __init__(self, study_id, bucket, s3_prefix, subjects=None,
                 anon=True, use_participants_tsv=False, random_seed=None,
                 _subject_class=S3BIDSSubject):
        """Initialize an S3BIDSStudy instance

        Parameters
        ----------
        study_id : str
            An identifier string for the study

        bucket : str
            The S3 bucket that contains the study data

        s3_prefix : str
            The S3 prefix common to all of the study objects on S3

        subjects : str, sequence(str), int, or None
            If int, retrieve S3 keys for the first `subjects` subjects.
            If "all", retrieve all subjects. If str or sequence of
            strings, retrieve S3 keys for the specified subjects. If sequence
            of ints, then for each int n retrieve S3 keys for the nth subject.
            If None, retrieve S3 keys for the first subject. Default: None

        anon : bool
            Whether to use anonymous connection (public buckets only).
            If False, uses the key/secret given, or boto’s credential
            resolver (client_kwargs, environment, variables, config
            files, EC2 IAM server, in that order). Default: True

        use_participants_tsv : bool
            If True, use the particpants tsv files to retrieve subject
            identifiers. This is faster but may not catch all subjects.
            Sometimes the tsv files are outdated. Default: False

        random_seed : int or None
            Random seed for selection of subjects if `subjects` is an
            integer. Use the same random seed for reproducibility.
            Default: None

        _subject_class : object
            The subject class to be used for this study. This parameter
            has a leading underscore because you probably don't want
            to change it. If you do change it, you must provide a
            class that quacks like AFQ.data.S3BIDSSubject. Default:
            S3BIDSSubject
        """
        logging.getLogger("botocore").setLevel(logging.WARNING)

        if not isinstance(study_id, str):
            raise TypeError('`study_id` must be a string.')

        if not isinstance(bucket, str):
            raise TypeError('`bucket` must be a string.')

        if not isinstance(s3_prefix, str):
            raise TypeError('`s3_prefix` must be a string.')

        if not (subjects is None
                or isinstance(subjects, int)
                or isinstance(subjects, str)
                or all(isinstance(s, str) for s in subjects)
                or all(isinstance(s, int) for s in subjects)):
            raise TypeError('`subjects` must be an int, string, '
                            'sequence of strings, or a sequence of ints.')

        if not isinstance(anon, bool):
            raise TypeError('`anon` must be of type bool.')

        if isinstance(subjects, int) and subjects < 1:
            raise ValueError('If `subjects` is an int, it must be '
                             'greater than 0.')

        if not isinstance(use_participants_tsv, bool):
            raise TypeError('`use_participants_tsv` must be boolean.')

        if not (random_seed is None or isinstance(random_seed, int)):
            raise TypeError("`random_seed` must be an integer.")

        self._study_id = study_id
        self._bucket = bucket
        self._s3_prefix = s3_prefix
        self._use_participants_tsv = use_participants_tsv
        self._random_seed = random_seed
        self._anon = anon
        self._subject_class = _subject_class
        self._local_directories = []

        # Get a list of all subjects in the study
        self._all_subjects = self._list_all_subjects()
        self._derivative_types = self._get_derivative_types()
        self._non_subject_s3_keys = self._get_non_subject_s3_keys()

        # Convert `subjects` into a sequence of subjectID strings
        if subjects is None or isinstance(subjects, int) \
                or (isinstance(subjects, list)
                    and isinstance(subjects[0], int)):
            # if subjects is an int, get that many random subjects
            prng = np.random.RandomState(random_seed)
            randomized_subjects = sorted(self._all_subjects.copy())
            prng.shuffle(randomized_subjects)

            if subjects is None:
                subjects = randomized_subjects[0]
            elif isinstance(subjects, int):
                subjects = randomized_subjects[:subjects]
            else:
                subjects = [randomized_subjects[i] for i in subjects]

            if isinstance(subjects, str):
                subjects = [subjects]
        elif subjects == 'all':
            # if "all," retrieve all subjects
            subjects = sorted(self._all_subjects)
        elif isinstance(subjects, str):
            # if a string, just get that one subject
            subjects = [subjects]
        # The last case for subjects is what we want. No transformation needed.

        if not set(subjects) <= set(self._all_subjects):
            raise ValueError(
                f'The following subjects could not be found in the study: '
                f'{set(subjects) - set(self._all_subjects)}'
            )

        subs = [
            delayed(self._get_subject)(s) for s in set(subjects)
        ]

        print('Retrieving subject S3 keys')
        with ProgressBar():
            subjects = list(compute(*subs, scheduler='threads'))

        self._subjects = subjects

    @property
    def study_id(self):
        """An identifier string for the study"""
        return self._study_id

    @property
    def bucket(self):
        """The S3 bucket that contains the study data"""
        return self._bucket

    @property
    def s3_prefix(self):
        """The S3 prefix common to all of the study objects on S3"""
        return self._s3_prefix

    @property
    def subjects(self):
        """A list of Subject instances for each requested subject"""
        return self._subjects

    @property
    def anon(self):
        """Is this study using an anonymous S3 connection?"""
        return self._anon

    @property
    def derivative_types(self):
        """A list of derivative pipelines available in this study"""
        return self._derivative_types

    @property
    def non_sub_s3_keys(self):
        """A dict of S3 keys that are not in subject directories"""
        return self._non_subject_s3_keys

    @property
    def local_directories(self):
        """A list of local directories to which this study has been downloaded"""
        return self._local_directories

    @property
    def use_participants_tsv(self):
        """Did we use a participants.tsv file to populate the list of
        study subjects."""
        return self._use_participants_tsv

    @property
    def random_seed(self):
        """The random seed used to retrieve study subjects"""
        return self._random_seed

    def __repr__(self):
        return (f'{type(self).__name__}(study_id={self.study_id}, '
                f'bucket={self.bucket}, s3_prefix={self.s3_prefix})')

    def _get_subject(self, subject_id):
        """Return a Subject instance from a subject-ID"""
        return self._subject_class(subject_id=subject_id,
                                   study=self)

    def _get_derivative_types(self):
        """Return a list of available derivatives pipelines

        Returns
        -------
        list
            list of available derivatives pipelines
        """
        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        derivatives_prefix = '/'.join([s3_prefix, 'derivatives'])
        if derivatives_prefix in nonsub_keys:
            return _ls_s3fs(
                s3_prefix=derivatives_prefix,
                anon=self.anon
            )['other']
        else:
            return []

    def _get_non_subject_s3_keys(self):
        """Return a list of 'non-subject' files

        In this context, a 'non-subject' file is any file
        or directory that is not a subject ID folder

        Returns
        -------
        dict
            dict with keys 'raw' and 'derivatives' and whose values
            are lists of S3 keys for non-subject files
        """
        non_sub_s3_keys = {}

        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")

        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        nonsub_keys = [k for k in nonsub_keys
                       if not k.endswith('derivatives')]

        nonsub_deriv_keys = []
        for dt in self.derivative_types:
            nonsub_deriv_keys.append(_ls_s3fs(
                s3_prefix=dt,
                anon=self.anon
            )['other'])

        non_sub_s3_keys = {
            'raw': nonsub_keys,
            'derivatives': nonsub_deriv_keys,
        }

        return non_sub_s3_keys

    def _list_all_subjects(self):
        """Return list of subjects

        Returns
        -------
        list
            list of participant_ids
        """
        if self._use_participants_tsv:
            tsv_key = "/".join([self.s3_prefix,
                                "participants.tsv"]).lstrip("/")
            s3 = get_s3_client(anon=self.anon)

            def get_subs_from_tsv_key(s3_key):
                response = s3.get_object(
                    Bucket=self.bucket,
                    Key=s3_key
                )

                return set(pd.read_csv(
                    response.get('Body')
                ).participant_id.values)

            subject_set = get_subs_from_tsv_key(tsv_key)
            subjects = list(subject_set)
        else:
            s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
            sub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                                anon=self.anon)['subjects']

            # Just need BIDSLayout for the `parse_file_entities`
            # method so we can pass dev/null as the argument
            layout = BIDSLayout(os.devnull, validate=False)
            subjects = []
            for key in sub_keys:
                entities = layout.parse_file_entities(key)
                subjects.append('sub-' + entities.get('subject'))

        return list(set(subjects))

    def _download_non_sub_keys(self, directory,
                               select=("dataset_description.json",),
                               filenames=None):
        fs = s3fs.S3FileSystem(anon=self.anon)
        if filenames is None:
            filenames = self.non_sub_s3_keys['raw']
        for fn in filenames:
            if select == "all" or any([s in fn for s in select]):
                Path(directory).mkdir(parents=True, exist_ok=True)
                fs.get(fn, op.join(directory, op.basename(fn)))

    def _download_derivative_descriptions(self, include_derivs, directory):
        for derivative in self.derivative_types:
            if include_derivs is True \
                or (isinstance(include_derivs, str)
                    and include_derivs == op.basename(derivative)) \
                or (isinstance(include_derivs, list)
                    and all(isinstance(s, str) for s in include_derivs)
                    and any([deriv in derivative for
                             deriv in include_derivs])):
                filenames = \
                    _ls_s3fs(s3_prefix=derivative, anon=self.anon)['other']
                deriv_directory = op.join(
                    directory, *derivative.split('/')[-2:])
                self._download_non_sub_keys(
                    deriv_directory,
                    select=("dataset_description.json",),
                    filenames=filenames)

    def download(self, directory,
                 include_modality_agnostic=("dataset_description.json",),
                 include_derivs=False,
                 include_derivs_dataset_description=True,
                 suffix=None,
                 overwrite=False, pbar=True):
        """Download files for each subject in the study

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_modality_agnostic : bool, "all" or any subset of ["dataset_description.json", "CHANGES", "README", "LICENSE"]
            If True or "all", download all keys in self.non_sub_s3_keys
            also. If a subset of ["dataset_description.json", "CHANGES",
            "README", "LICENSE"], download only those files. This is
            useful if the non_sub_s3_keys contain files common to all
            subjects that should be inherited. Default: ("dataset_description.json",)

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ["dmriprep", "afq"]). Default: False

        include_derivs_dataset_description : bool
            Used only if include_derivs is not False. If True,
            dataset_description.json downloaded for each derivative.

        suffix : str
            Suffix, including extension, of file(s) to download.
            Default: None

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        pbar : bool
            If True, include progress bar. Default: True

        See Also
        --------
        AFQ.data.S3BIDSSubject.download
        """
        self._local_directories.append(directory)
        self._local_directories = list(set(self._local_directories))

        if include_modality_agnostic is True or include_modality_agnostic == "all":
            self._download_non_sub_keys(directory, select="all")
        elif include_modality_agnostic is not False:
            # Subset selection
            valid_set = {"dataset_description.json",
                         "CHANGES",
                         "README",
                         "LICENSE"}
            if not set(include_modality_agnostic) <= valid_set:
                raise ValueError(
                    "include_modality_agnostic must be either a boolean, 'all', "
                    "or a subset of {valid_set}".format(valid_set=valid_set)
                )

            self._download_non_sub_keys(
                directory, select=include_modality_agnostic)

        # download dataset_description.json for derivatives
        if (include_derivs is not False) \
                and include_derivs_dataset_description:
            self._download_derivative_descriptions(
                include_derivs, directory)

        results = [delayed(sub.download)(
            directory=directory,
            include_derivs=include_derivs,
            suffix=suffix,
            overwrite=overwrite,
            pbar=pbar,
            pbar_idx=idx,
        ) for idx, sub in enumerate(self.subjects)]

        compute(*results, scheduler='threads')


class HBNSite(S3BIDSStudy):
    """An HBN study site

    See Also
    --------
    AFQ.data.S3BIDSStudy
    """

    def __init__(self, site, study_id='HBN', bucket='fcp-indi',
                 s3_prefix='data/Projects/HBN/MRI',
                 subjects=None, use_participants_tsv=False,
                 random_seed=None):
        """Initialize the HBN site

        Parameters
        ----------
        site : ["Site-SI", "Site-RU", "Site-CBIC", "Site-CUNY"]
            The HBN site

        study_id : str
            An identifier string for the site

        bucket : str
            The S3 bucket that contains the study data

        s3_prefix : str
            The S3 prefix common to all of the study objects on S3

        subjects : str, sequence(str), int, or None
            If int, retrieve S3 keys for the first `subjects` subjects.
            If "all", retrieve all subjects. If str or sequence of
            strings, retrieve S3 keys for the specified subjects. If
            None, retrieve S3 keys for the first subject. Default: None

        use_participants_tsv : bool
            If True, use the particpants tsv files to retrieve subject
            identifiers. This is faster but may not catch all subjects.
            Sometimes the tsv files are outdated. Default: False

        random_seed : int or None
            Random seed for selection of subjects if `subjects` is an
            integer. Use the same random seed for reproducibility.
            Default: None
        """
        valid_sites = ["Site-SI", "Site-RU", "Site-CBIC", "Site-CUNY"]
        if site not in valid_sites:
            raise ValueError(
                "site must be one of {}.".format(valid_sites)
            )

        self._site = site

        super().__init__(
            study_id=study_id,
            bucket=bucket,
            s3_prefix='/'.join([s3_prefix, site]),
            subjects=subjects,
            use_participants_tsv=use_participants_tsv,
            random_seed=random_seed,
            _subject_class=HBNSubject
        )

    @property
    def site(self):
        """The HBN site"""
        return self._site

    def _get_subject(self, subject_id):
        """Return a Subject instance from a subject-ID"""
        return self._subject_class(subject_id=subject_id,
                                   study=self,
                                   site=self.site)

    def _get_derivative_types(self):
        """Return a list of available derivatives pipelines

        The HBN dataset is not BIDS compliant so to go a list
        of available derivatives, we must peak inside every
        directory in `derivatives/sub-XXXX/`

        Returns
        -------
        list
            list of available derivatives pipelines
        """
        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")
        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        derivatives_prefix = '/'.join([s3_prefix, 'derivatives'])

        if any([derivatives_prefix in key for key in nonsub_keys]):
            deriv_subs = _ls_s3fs(
                s3_prefix=derivatives_prefix,
                anon=self.anon
            )['subjects']

            deriv_types = []
            for sub_key in deriv_subs:
                deriv_types += [
                    s.split(sub_key)[-1].lstrip('/')
                    for s in _ls_s3fs(
                        s3_prefix=sub_key,
                        anon=self.anon
                    )['subjects']
                ]

            return list(set(deriv_types))
        else:
            return []

    def _get_non_subject_s3_keys(self):
        """Return a list of 'non-subject' files

        In this context, a 'non-subject' file is any file
        or directory that is not a subject ID folder. This method
        is different from AFQ.data.S3BIDSStudy because the HBN
        dataset is not BIDS compliant

        Returns
        -------
        dict
            dict with keys 'raw' and 'derivatives' and whose values
            are lists of S3 keys for non-subject files

        See Also
        --------
        AFQ.data.S3BIDSStudy._get_non_subject_s3_keys
        """
        non_sub_s3_keys = {}

        s3_prefix = '/'.join([self.bucket, self.s3_prefix]).rstrip("/")

        nonsub_keys = _ls_s3fs(s3_prefix=s3_prefix,
                               anon=self.anon)['other']
        nonsub_keys = [k for k in nonsub_keys
                       if not k.endswith('derivatives')]

        nonsub_deriv_keys = _ls_s3fs(
            s3_prefix='/'.join([
                self.bucket,
                self.s3_prefix,
                'derivatives'
            ]),
            anon=self.anon
        )['other']

        non_sub_s3_keys = {
            'raw': nonsub_keys,
            'derivatives': nonsub_deriv_keys,
        }

        return non_sub_s3_keys

    def download(self, directory, include_modality_agnostic=False,
                 include_derivs=False, overwrite=False, pbar=True):
        """Download files for each subject in the study

        Parameters
        ----------
        directory : str
            Directory to which to download subject files

        include_modality_agnostic : bool, "all" or any subset of ["dataset_description.json", "CHANGES", "README", "LICENSE"]
            If True or "all", download all keys in self.non_sub_s3_keys
            also. If a subset of ["dataset_description.json", "CHANGES",
            "README", "LICENSE"], download only those files. This is
            useful if the non_sub_s3_keys contain files common to all
            subjects that should be inherited. Default: False

        include_derivs : bool or str
            If True, download all derivatives files. If False, do not.
            If a string or sequence of strings is passed, this will
            only download derivatives that match the string(s) (e.g.
            ["dmriprep", "afq"]). Default: False

        overwrite : bool
            If True, overwrite files for each subject. Default: False

        pbar : bool
            If True, include progress bar. Default: True

        See Also
        --------
        AFQ.data.S3BIDSSubject.download
        """
        super().download(
            directory=directory,
            include_modality_agnostic=include_modality_agnostic,
            include_derivs=include_derivs,
            overwrite=overwrite,
            pbar=pbar
        )

        to_bids_description(
            directory,
            **{"BIDSVersion": "1.0.0",
               "Name": "HBN Study, " + self.site,
               "DatasetType": "raw",
               "Subjects": [s.subject_id for s in self.subjects]})


# +--------------------------------------------------+
# | End S3BIDSStudy classes and supporting functions |
# +--------------------------------------------------+


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

    for k in data_files.keys():
        if not op.exists(k):
            bucket.download_file(data_files[k], k)
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


stanford_hardi_tractography_remote_fnames = ["5325715", "5325718", "25289735"]
stanford_hardi_tractography_hashes = ['6f4bdae702031a48d1cd3811e7a42ef9',
                                      'f20854b4f710577c58bd01072cfb4de6',
                                      '294bfd1831861e8635eef8834ff18892']
stanford_hardi_tractography_fnames = ['mapping.nii.gz',
                                      'tractography_subsampled.trk',
                                      'full_segmented_cleaned_tractography.trk']

fetch_stanford_hardi_tractography = _make_fetcher(
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

    files_dict['tractography_subsampled.trk'] = load_trk(
        op.join(afq_home,
                'stanford_hardi_tractography',
                'tractography_subsampled.trk'),
        dwi_img,
        bbox_valid_check=False,
        trk_header_check=False).streamlines

    files_dict['full_segmented_cleaned_tractography.trk'] = load_trk(
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
    logger = logging.getLogger('AFQ.data')

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


fetch_hcp_atlas_16_bundles = _make_fetcher(
    "fetch_hcp_atlas_16_bundles",
    op.join(afq_home,
            'hcp_atlas_16_bundles'),
    'https://ndownloader.figshare.com/files/',
    ["11921522"],
    ["atlas_16_bundles.zip"],
    md5_list=["b071f3e851f21ba1749c02fc6beb3118"],
    doc="Download minimal Recobundles atlas",
    unzip=True)


fetch_hcp_atlas_80_bundles = _make_fetcher(
    "fetch_hcp_atlas_80_bundles",
    op.join(afq_home,
            'hcp_atlas_80_bundles'),
    'https://ndownloader.figshare.com/files/',
    ["13638644"],
    ["Atlas_80_Bundles.zip"],
    md5_list=["78331d527a10ec000d4f33bac472e099"],
    doc="Download 80-bundle Recobundles atlas",
    unzip=True)


def read_hcp_atlas(n_bundles=16):
    """
    n_bundles : int
        16 or 80, which selects among the two different
        atlases:

        https://figshare.com/articles/Simple_model_bundle_atlas_for_RecoBundles/6483614  #noqa

        https://figshare.com/articles/Advanced_Atlas_of_80_Bundles_in_MNI_space/7375883  #noqa
    """
    bundle_dict = {}
    if n_bundles == 16:
        _, folder = fetch_hcp_atlas_16_bundles()
        atlas_folder = "Atlas_in_MNI_Space_16_bundles"
    elif n_bundles == 80:
        _, folder = fetch_hcp_atlas_80_bundles()
        atlas_folder = "Atlas_80_Bundles"

    whole_brain = load_tractogram(
        op.join(
            folder,
            atlas_folder,
            'whole_brain',
            'whole_brain_MNI.trk'),
            'same', bbox_valid_check=False).streamlines

    bundle_dict['whole_brain'] = whole_brain
    bundle_files = glob(
        op.join(
            folder,
            atlas_folder,
            "bundles", "*.trk"))
    for bundle_file in bundle_files:
        bundle = op.splitext(op.split(bundle_file)[-1])[0]
        bundle_dict[bundle] = {}
        bundle_dict[bundle]['sl'] = load_tractogram(
            bundle_file,
            'same',
            bbox_valid_check=False).streamlines

        feature = ResampleFeature(nb_points=100)
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(np.inf, metric=metric)
        cluster = qb.cluster(bundle_dict[bundle]['sl'])
        bundle_dict[bundle]['centroid'] = cluster.centroids[0]

    # For some reason, this file-name has a 0 in it, instead of an O:
    bundle_dict["IFOF_R"] = bundle_dict["IF0F_R"]
    # In the 80-bundle case, there are two files, and both have identical
    # content, so this is fine:
    del bundle_dict["IF0F_R"]
    return bundle_dict


fetch_aal_atlas = _make_fetcher(
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
                  'leftparietal': np.array([57, 67, 2]),
                  'leftanttemporal': np.array([41, 83, 87]),
                  'leftuncinatefront': np.array([5, 9, 15, 25]),
                  'leftifoffront': np.array([3, 5, 7, 9, 13, 15, 25]),
                  'leftinfparietal': np.array([61, 63, 65]),
                  'cerebellum': np.arange(91, 117),
                  'leftarcfrontal': np.array([1, 11, 13]),
                  'leftarctemp': np.array([79, 81, 85, 89]),
                  }

    # Right symmetrical is off by one:
    atlas_vals['rightfrontal'] = atlas_vals['leftfrontal'] + 1
    atlas_vals['rightoccipital'] = atlas_vals['leftoccipital'] + 1
    atlas_vals['righttemporal'] = atlas_vals['lefttemporal'] + 1
    atlas_vals['rightparietal'] = atlas_vals['leftparietal'] + 1
    atlas_vals['rightanttemporal'] = atlas_vals['leftanttemporal'] + 1
    atlas_vals['rightuncinatefront'] = atlas_vals['leftuncinatefront'] + 1
    atlas_vals['rightifoffront'] = atlas_vals['leftifoffront'] + 1
    atlas_vals['rightinfparietal'] = atlas_vals['leftinfparietal'] + 1
    atlas_vals['rightarcfrontal'] = atlas_vals['leftarcfrontal'] + 1
    atlas_vals['rightarctemp'] = atlas_vals['leftarctemp'] + 1

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
        "ATR_L": [['leftfrontal'], None],
        "ATR_R": [['rightfrontal'], None],
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
        "Temporal": [None, None]}

    targets = []
    for bundle in bundles:
        targets.append([])

        if (endpoint_dict.get(bundle)):
            for region in endpoint_dict[bundle]:
                if region is None:
                    targets[-1].append(None)
                else:
                    targets[-1].append(aal_to_regions(region, atlas=atlas))
        else:
            logger = logging.getLogger('AFQ.data')
            logger.warning(f"Segmentation end points undefined for {bundle},"
                           + " continuing without end points")
            targets[-1] = [None, None]

    return targets


def s3fs_nifti_write(img, fname, fs=None):
    """
    Write a nifti file straight to S3

    Paramters
    ---------
    img : nib.Nifti1Image class instance
        The image containing data to be written into S3
    fname : string
        Full path (including bucket name and extension) to the S3 location
        where the file is to be saved.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system
    """
    if fs is None:
        fs = s3fs.S3FileSystem()

    bio = BytesIO()
    file_map = img.make_file_map({'image': bio, 'header': bio})
    img.to_file_map(file_map)
    data = gzip.compress(bio.getvalue())
    with fs.open(fname, 'wb') as ff:
        ff.write(data)


def s3fs_nifti_read(fname, fs=None, anon=False):
    """
    Lazily reads a nifti image from S3.

    Paramters
    ---------
    fname : string
        Full path (including bucket name and extension) to the S3 location
        of the file to be read.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True

    Returns
    -------
    nib.Nifti1Image class instance

    Notes
    -----
    Because the image is lazily loaded, data stored in the file
    is not transferred until `get_fdata` is called.

    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=anon)
    with fs.open(fname) as ff:
        zz = gzip.open(ff)
        rr = zz.read()
        bb = BytesIO(rr)
        fh = nib.FileHolder(fileobj=bb)
        img = nib.Nifti1Image.from_file_map({'header': fh, 'image': fh})
    return img


def write_json(fname, data):
    """
    Write data to JSON file.

    Parameters
    ----------
    fname : str
        Full path to the file to write.

    data : dict
        A dict containing the data to write.

    Returns
    -------
    None
    """
    with open(fname, 'w') as ff:
        json.dump(data, ff, default=lambda obj: "Not Serializable")


def read_json(fname):
    """
    Read data from a JSON file.

    Parameters
    ----------
    fname : str
        Full path to the data-containing file

    Returns
    -------
    dict
    """
    with open(fname, 'w') as ff:
        out = json.load(ff)
    return out


def s3fs_json_read(fname, fs=None, anon=False):
    """
    Reads json directly from S3

    Paramters
    ---------
    fname : str
        Full path (including bucket name and extension) to the file on S3.
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    anon : bool
        Whether to use anonymous connection (public buckets only).
        If False, uses the key/secret given, or boto’s credential
        resolver (client_kwargs, environment, variables, config files,
        EC2 IAM server, in that order). Default: True
    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=anon)
    with fs.open(fname) as ff:
        data = json.load(ff)
    return data


def s3fs_json_write(data, fname, fs=None):
    """
    Writes json from a dict directly into S3

    Parameters
    ----------
    data : dict
        The json to be written out
    fname : str
        Full path (including bucket name and extension) to the file to
        be written out on S3
    fs : an s3fs.S3FileSystem class instance, optional
        A file-system to refer to. Default to create a new file-system.
    """
    if fs is None:
        fs = s3fs.S3FileSystem()
    with fs.open(fname, 'w') as ff:
        json.dump(data, ff)


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
    _make_fetcher(
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
        logger = logging.getLogger('AFQ.data')
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
