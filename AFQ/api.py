# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import logging
import functools
from AFQ.definitions.mask import (B0Mask, ScalarMask, FullMask)
from AFQ.definitions.mapping import (SynMap, FnirtMap, ItkMap)
from AFQ.definitions.utils import Definition
from AFQ.utils.bin import get_default_args
from AFQ.viz.utils import Viz, visualize_tract_profiles
import AFQ.utils.volume as auv
import AFQ.segmentation as seg
import AFQ.utils.streamlines as aus
import dipy.reconst.dki as dpy_dki
import dipy.reconst.dti as dpy_dti
import AFQ.tractography as aft
from AFQ.models.csd import _fit as csd_fit
from AFQ.models.dki import _fit as dki_fit
from AFQ.models.dti import noise_from_b0
from AFQ.models.dti import _fit as dti_fit
import AFQ.data as afd  # TODO: clean up these imports

from AFQ.tasks.data import data_tasks
from AFQ.tasks.models import model_tasks, dti, dki, csd
from AFQ.tasks.mapping import mapping_tasks, get_reg_subject
from AFQ.tasks.tractography import (
    tractography_tasks, custom_tractography, export_stop_mask_pft)
from AFQ.tasks.segmentation import segmentation_tasks
from AFQ.tasks.profile import profile_tasks, gen_scalar_func
from AFQ.tasks.viz import viz_tasks, viz_bundles, viz_indivBundle
from AFQ.tasks.utils import as_file

from .version import version as pyafq_version
import pandas as pd
import dask.dataframe as ddf
import os
import os.path as op
import json
import s3fs
from time import time
import pimms
from pimms.calculation import calc_tr

import numpy as np
import nibabel as nib

import dipy.core.gradients as dpg
import dipy.tracking.utils as dtu
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.gradients import read_bvals_bvecs
from dipy.align import resample
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.reconst import shm
from dipy.reconst.dki_micro import axonal_water_fraction
from dipy.tracking.streamline import set_number_of_points, values_from_volume

import bids
from bids.layout import BIDSLayout
import bids.config as bids_config
try:
    bids_config.set_option('extension_initial_dot', True)
except ValueError:
    pass


logging.basicConfig(level=logging.INFO)


__all__ = ["AFQ", "make_bundle_dict"]


def do_preprocessing():
    raise NotImplementedError


BUNDLES = ["ATR", "CGC", "CST", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]

CALLOSUM_BUNDLES = ["AntFrontal", "Motor", "Occipital", "Orbital",
                    "PostParietal", "SupFrontal", "SupParietal",
                    "Temporal"]

# See: https://www.cmu.edu/dietrich/psychology/cognitiveaxon/documents/yeh_etal_2018.pdf  # noqa

RECO_BUNDLES_16 = [
    'CST', 'C', 'F', 'UF', 'MCP', 'AF', 'CCMid',
    'CC_ForcepsMajor', 'CC_ForcepsMinor', 'IFOF']

RECO_BUNDLES_80 = ["AC", "AF", "AR", "AST", "C", "CB", "CC_ForcepsMajor",
                   "CC_ForcepsMinor", "CC", "CCMid", "CNII", "CNII", "CNIII",
                   "CNIV", "CNV", "CNVII", "CNVIII", "CS", "CST", "CT",
                   "CTT", "DLF", "EMC", "F_L_R", "FPT", "ICP", "IFOF", "ILF",
                   "LL", "MCP", "MdLF", "ML", "MLF", "OPT", "OR", "PC", "PPT",
                   "RST", "SCP", "SLF", "STT", "TPT", "UF", "V", "VOF"]

RECO_UNIQUE = [
    'CCMid', 'CC_ForcepsMajor', 'CC_ForcepsMinor', 'MCP', 'AC', 'PC', 'SCP',
    'V', 'CC', 'F_L_R']

DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


def make_bundle_dict(bundle_names=BUNDLES,
                     seg_algo="afq",
                     resample_to=False):
    """
    Create a bundle dictionary, needed for the segmentation

    Parameters
    ----------
    bundle_names : list, optional
        A list of the bundles to be used in this case. Default: all of them

    seg_algo: One of {"afq", "reco", "reco16", "reco80"}
        The bundle segmentation algorithm to use.
            "afq" : Use waypoint ROIs + probability maps, as described
            in [Yeatman2012]_
            "reco" / "reco16" : Use Recobundles [Garyfallidis2017]_
            with a 16-bundle set.
            "reco80": Use Recobundles with an 80-bundle set.

    resample_to : Nifti1Image, optional
        If set, templates will be resampled to the affine and shape of this
        image.
    """
    logger = logging.getLogger('AFQ.api')
    if seg_algo == "afq":
        if "FP" in bundle_names and "Occipital" in bundle_names:
            logger.warning((
                f"FP and Occipital bundles are co-located, and AFQ"
                f" assigns each streamline to only one bundle."
                f" Only Occipital will be used."))
            bundle_names.remove("FP")
        if "FA" in bundle_names and "Orbital" in bundle_names:
            logger.warning((
                f"FA and Orbital bundles are co-located, and AFQ"
                f" assigns each streamline to only one bundle."
                f" Only Orbital will be used."))
            bundle_names.remove("FA")
        if "FA" in bundle_names and "AntFrontal" in bundle_names:
            logger.warning((
                f"FA and AntFrontal bundles are co-located, and AFQ"
                f" assigns each streamline to only one bundle."
                f" Only AntFrontal will be used."))
            bundle_names.remove("FA")
        templates = afd.read_templates(resample_to=resample_to)
        callosal_templates = afd.read_callosum_templates(
            resample_to=resample_to)
        # For the arcuate, we need to rename a few of these and duplicate the
        # SLF ROI:
        templates['ARC_roi1_L'] = templates['SLF_roi1_L']
        templates['ARC_roi1_R'] = templates['SLF_roi1_R']
        templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
        templates['ARC_roi2_R'] = templates['SLFt_roi2_R']

        afq_bundles = {}
        # Each bundles gets a digit identifier (to be stored in the tractogram)
        uid = 1
        for name in bundle_names:
            # Consider hard coding since we might have different rules for
            # some tracts
            if name in ["FA", "FP"]:
                afq_bundles[name] = {
                    'ROIs': [templates[name + "_L"],
                             templates[name + "_R"],
                             callosal_templates["Callosum_midsag"]],
                    'rules': [True, True, True],
                    'prob_map': templates[name + "_prob_map"],
                    'cross_midline': True,
                    'uid': uid}
                uid += 1
            elif name in CALLOSUM_BUNDLES:
                afq_bundles[name] = {
                    'ROIs': [callosal_templates["L_" + name],
                             callosal_templates["R_" + name],
                             callosal_templates["Callosum_midsag"]],
                    'rules': [True, True, True],
                    'cross_midline': True,
                    'uid': uid}
                uid += 1
            # SLF is a special case, because it has an exclusion ROI:
            elif name == "SLF":
                for hemi in ['_R', '_L']:
                    afq_bundles[name + hemi] = {
                        'ROIs': [templates[name + '_roi1' + hemi],
                                 templates[name + '_roi2' + hemi],
                                 templates["SLFt_roi2" + hemi]],
                        'rules': [True, True, False],
                        'prob_map': templates[name + hemi + '_prob_map'],
                        'cross_midline': False,
                        'uid': uid}
                    uid += 1
            else:
                for hemi in ['_R', '_L']:
                    if (templates.get(name + '_roi1' + hemi)
                            and templates.get(name + '_roi2' + hemi)
                            and templates.get(name + hemi + '_prob_map')):
                        afq_bundles[name + hemi] = {
                            'ROIs': [templates[name + '_roi1' + hemi],
                                     templates[name + '_roi2' + hemi]],
                            'rules': [True, True],
                            'prob_map': templates[name + hemi + '_prob_map'],
                            'cross_midline': False,
                            'uid': uid}
                    else:
                        logger.warning(f"{name} is not in AFQ templates")

                    uid += 1
    elif seg_algo.startswith("reco"):
        if seg_algo.endswith("80"):
            bundle_dict = afd.read_hcp_atlas(80)
        else:
            bundle_dict = afd.read_hcp_atlas(16)

        afq_bundles = {}
        uid = 1
        afq_bundles["whole_brain"] = bundle_dict["whole_brain"]
        for name in bundle_names:
            if name in RECO_UNIQUE:
                afq_bundles[name] = bundle_dict[name]
                afq_bundles[name]['uid'] = uid
                uid += 1
            else:
                for hemi in ["_R", "_L"]:
                    afq_bundles[name + hemi] = bundle_dict[name + hemi]
                    afq_bundles[name + hemi]['uid'] = uid
                    uid += 1
    else:
        raise ValueError("Input: %s is not a valid input`seg_algo`" % seg_algo)

    return afq_bundles


def _getter(attr, subses_dict=False):
    def _getter_helper(self, wf_dict, results, attr):
        for subject in self.subjects:
            if subses_dict:
                results[subject] = wf_dict[subject]['subses_dict'][attr]
            else:
                results[subject] = wf_dict[subject][attr]

    def _this_getter(func):
        @functools.wraps(func)
        def wrapper_getter(self):
            results = {}
            if len(self.sessions) > 1:
                for session in self.sessions:
                    results[session] = {}
                    _getter_helper(
                        self,
                        self.wf_dict[session],
                        results[session],
                        attr)
            else:
                _getter_helper(
                    self,
                    self.wf_dict,
                    results,
                    attr)
            return results
        return wrapper_getter
    return _this_getter


class AFQ(object):
    """
    """

    def __init__(self,
                 bids_path,
                 bids_filters={"suffix": "dwi"},
                 dmriprep="all",
                 custom_tractography_bids_filters=None,
                 b0_threshold=50,
                 patch2self=False,
                 robust_tensor_fitting=False,
                 min_bval=None,
                 max_bval=None,
                 reg_template="mni_T1",
                 reg_subject="power_map",
                 brain_mask=B0Mask(),
                 mapping=SynMap(),
                 profile_weights="gauss",
                 bundle_info=None,
                 dask_it=False,
                 scalars=["dti_fa", "dti_md"],
                 virtual_frame_buffer=False,
                 viz_backend="plotly_no_gif",
                 tracking_params=None,
                 segmentation_params=None,
                 clean_params=None):
        '''
        Initialize an AFQ object.
        Some special notes on parameters:
        In tracking_params, parameters with the suffix mask which are also
        a mask from AFQ.definitions.mask will be handled automatically by the
        api. You can set additional parameters for a given step of the process
        by directly calling the relevant api function. For example,
        to set the sh_order for csd to 4, call:
        myafq._csd(sh_order=4)
        before otherwise generating the csd file.

        Parameters
        ----------
        bids_path : str
            [BIDS] The path to preprocessed diffusion data organized in a BIDS
            dataset. This should contain a BIDS derivative dataset with
            preprocessed dwi/bvals/bvecs.
        bids_filters : dict
            [BIDS] Filter to pass to bids_layout.get when finding DWI files.
            Default: {"suffix": "dwi"}
        dmriprep : str, optional.
            [BIDS] The name of the pipeline used to preprocess the DWI data.
            Default: "all".
        custom_tractography_bids_filters : dict, optional
            [BIDS] BIDS filters for inputing a user made tractography file.
            If None, tractography will be performed automatically.
            Default: None
        b0_threshold : int, optional
            [REGISTRATION] The value of b under which
            it is considered to be b0. Default: 50.
        patch2self : bool, optional
            [REGISTRATION] Whether to use patch2self
            to denoise the dwi data.
            Default: False
        robust_tensor_fitting : bool, optional
            [REGISTRATION] Whether to use robust_tensor_fitting when
            doing dti. Only applies to dti.
            Default: False
        min_bval : float, optional
            [REGISTRATION] Minimum b value you want to use
            from the dataset (other than b0), inclusive.
            If None, there is no minimum limit. Default: None
        max_bval : float, optional
            [REGISTRATION] Maximum b value you want to use
            from the dataset (other than b0), inclusive.
            If None, there is no maximum limit. Default: None
        reg_subject : str, Nifti1Image, dict, optional
            [REGISTRATION] The source image data to be registered.
            Can either be a Nifti1Image, bids filters for a Nifti1Image, or
            if "b0", "dti_fa_subject", "subject_sls", or "power_map,"
            image data will be loaded automatically.
            If "subject_sls" is used, slr registration will be used
            and reg_template should be "hcp_atlas".
            Default: "power_map"
        reg_template : str or Nifti1Image, optional
            [REGISTRATION] The target image data for registration.
            Can either be a Nifti1Image, a path to a Nifti1Image, or
            if "mni_T2", "dti_fa_template", "hcp_atlas", or "mni_T1",
            image data will be loaded automatically.
            If "hcp_atlas" is used, slr registration will be used
            and reg_subject should be "subject_sls".
            Default: "mni_T1"
        brain_mask : instance of class from `AFQ.definitions.mask`, optional
            [REGISTRATION] This will be used to create
            the brain mask, which gets applied before registration to a
            template.
            If None, no brain mask will not be applied,
            and no brain mask will be applied to the template.
            Default: B0Mask()
        mapping : instance of class from `AFQ.definitions.mapping`, optional
            [REGISTRATION]  This defines how to either create a mapping from
            each subject space to template space or load a mapping from
            another software. If creating a map, will register reg_subject and
            reg_template.
            Default: SynMap()
        profile_weights : str, 1D array, 2D array callable, optional
            [PROFILE] How to weight each streamline (1D) or each node (2D)
            when calculating the tract-profiles. If callable, this is a
            function that calculates weights. If None, no weighting will
            be applied. If "gauss", gaussian weights will be used.
            If "median", the median of values at each node will be used
            instead of a mean or weighted mean.
            Default: "gauss"
        bundle_info : list of strings or dict, optional
            [BUNDLES] List of bundle names to include in segmentation,
            or a bundle dictionary (see make_bundle_dict for inspiration).
            If None, will get all appropriate bundles for the chosen
            segmentation algorithm.
            Default: None
        dask_it : bool, optional
            [COMPUTE] Whether to use a dask DataFrame object.
            Default: False
        scalars : list of strings and/or scalar definitions, optional
            [BUNDLES] List of scalars to use.
            Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf",
            "dki_mk". Can also be a scalar from AFQ.definitions.scalar.
            Default: ["dti_fa", "dti_md"]
        virtual_frame_buffer : bool, optional
            [VIZ] Whether to use a virtual fram buffer. This is neccessary if
            generating GIFs in a headless environment. Default: False
        viz_backend : str, optional
            [VIZ] Which visualization backend to use.
            See Visualization Backends page in documentation for details:
            https://yeatmanlab.github.io/pyAFQ/usage/viz_backend.html
            One of {"fury", "plotly", "plotly_no_gif"}.
            Default: "plotly_no_gif"
        segmentation_params : dict, optional
            The parameters for segmentation.
            Default: use the default behavior of the seg.Segmentation object.
        tracking_params: dict, optional
            The parameters for tracking. Default: use the default behavior of
            the aft.track function. Seed mask and seed threshold, if not
            specified, are replaced with scalar masks from scalar[0]
            thresholded to 0.2. The ``seed_mask`` and ``stop_mask`` items of
            this dict may be ``AFQ.definitions.mask.MaskFile`` instances.
            If ``tracker`` is set to "pft" then ``stop_mask`` should be
            an instance of ``AFQ.definitions.mask.PFTMask``.
        clean_params: dict, optional
            The parameters for cleaning.
            Default: use the default behavior of the seg.clean_bundle
            function.
        '''
        if not isinstance(bids_path, str):
            raise TypeError("bids_path must be a string")
        if not op.exists(bids_path):
            raise ValueError("bids_path not found")
        if not op.exists(op.join(bids_path, "dataset_description.json")):
            raise ValueError("There must be a dataset_description.json"
                             + " in bids_path")
        if not isinstance(bids_filters, dict):
            raise TypeError("bids_filters must be a dict")
        # dmriprep typechecking handled by pyBIDS
        if custom_tractography_bids_filters is not None\
                and not isinstance(custom_tractography_bids_filters, dict):
            raise TypeError(
                "custom_tractography_bids_filters must be"
                + " either a dict or None")
        if not isinstance(b0_threshold, int):
            raise TypeError("b0_threshold must be an int")
        if not isinstance(patch2self, bool):
            raise TypeError("patch2self must be a bool")
        if not isinstance(robust_tensor_fitting, bool):
            raise TypeError("robust_tensor_fitting must be a bool")
        if min_bval is not None and not isinstance(min_bval, int):
            raise TypeError("min_bval must be an int")
        if max_bval is not None and not isinstance(max_bval, int):
            raise TypeError("max_bval must be an int")
        if not isinstance(reg_template, str)\
                and not isinstance(reg_template, nib.Nifti1Image):
            raise TypeError(
                "reg_template must be a str or Nifti1Image")
        if not isinstance(reg_subject, str)\
            and not isinstance(reg_subject, nib.Nifti1Image)\
                and not isinstance(reg_subject, dict):
            raise TypeError(
                "reg_subject must be a str, dict, or Nifti1Image")
        if isinstance(reg_subject, str) and isinstance(reg_template, str)\
                and (reg_subject.lower() == 'subject_sls'
                     or reg_template.lower() == 'hcp_atlas'):
            if reg_template.lower() != 'hcp_atlas':
                raise TypeError(
                    "If reg_subject is 'subject_sls',"
                    + " reg_template must be 'hcp_atlas'")
            if reg_subject.lower() != 'subject_sls':
                raise TypeError(
                    "If reg_template is 'hcp_atlas',"
                    + " reg_subject must be 'subject_sls'")
        if brain_mask is not None and not isinstance(
                brain_mask, Definition):
            raise TypeError(
                "brain_mask must be None or a mask "
                "defined in `AFQ.definitions.mask`")
        if not isinstance(mapping, Definition):
            raise TypeError(
                "mapping must be a mapping defined"
                + " in `AFQ.definitions.mapping`")
        if not (profile_weights is None
                or isinstance(profile_weights, str)
                or callable(profile_weights)
                or hasattr(profile_weights, "__len__")):
            raise TypeError(
                "profile_weights must be string, None, callable, or"
                + "a 1D or 2D array")
        if isinstance(profile_weights, str) and\
                profile_weights != "gauss" and profile_weights != "median":
            raise TypeError(
                "if profile_weights is a string,"
                + " it must be 'gauss' or 'median'")
        if bundle_info is not None and not ((
                isinstance(bundle_info, list)
                and isinstance(bundle_info[0], str)) or (
                    isinstance(bundle_info, dict))):
            raise TypeError(
                "bundle_info must be None, a list of strings, or a dict")
        if not isinstance(dask_it, bool):
            raise TypeError("dask_it must be a bool")
        if scalars is not None and not (
                isinstance(scalars, list)
                and (
                    isinstance(scalars[0], str)
                    or isinstance(scalars[0], Definition))):
            raise TypeError(
                "scalars must be None or a list of "
                "strings/scalar definitions")
        if not isinstance(virtual_frame_buffer, bool):
            raise TypeError("virtual_frame_buffer must be a bool")
        if "fury" not in viz_backend and "plotly" not in viz_backend:
            raise TypeError(
                "viz_backend must contain either 'fury' or 'plotly'")
        if tracking_params is not None\
                and not isinstance(tracking_params, dict):
            raise TypeError(
                "tracking_params must be None or a dict")
        if segmentation_params is not None\
                and not isinstance(segmentation_params, dict):
            raise TypeError(
                "segmentation_params must be None or a dict")
        if clean_params is not None\
                and not isinstance(clean_params, dict):
            raise TypeError(
                "clean_params must be None or a dict")

        self.logger = logging.getLogger('AFQ.api')

        # validate input and fail early
        if not op.exists(bids_path):
            raise ValueError(f'Unable to locate BIDS dataset in: {bids_path}')

        self.max_bval = max_bval
        self.min_bval = min_bval

        self.reg_subject = reg_subject
        self.reg_template = reg_template
        if brain_mask is None:
            self.brain_mask_definition = FullMask()
            self.mask_template = False
        else:
            self.brain_mask_definition = brain_mask
            self.mask_template = True
        self.mapping_definition = mapping

        self.b0_threshold = b0_threshold
        self.patch2self = patch2self
        self.robust_tensor_fitting = robust_tensor_fitting
        self.custom_tractography_bids_filters =\
            custom_tractography_bids_filters

        self.scalars = []
        for scalar in scalars:
            if isinstance(scalar, str):
                self.scalars.append(scalar.lower())
            else:
                self.scalars.append(scalar)

        if virtual_frame_buffer:
            from xvfbwrapper import Xvfb
            self.vdisplay = Xvfb(width=1280, height=1280)
            self.vdisplay.start()
        self.viz = Viz(backend=viz_backend.lower())

        best_scalar = self._get_best_scalar()
        default_tracking_params = get_default_args(aft.track)
        default_tracking_params["seed_mask"] = ScalarMask(
            best_scalar)
        default_tracking_params["stop_mask"] = ScalarMask(
            best_scalar)
        default_tracking_params["seed_threshold"] = 0.2
        default_tracking_params["stop_threshold"] = 0.2
        # Replace the defaults only for kwargs for which a non-default value
        # was given:
        if tracking_params is not None:
            for k in tracking_params:
                default_tracking_params[k] = tracking_params[k]

        self.tracking_params = default_tracking_params
        self.tracking_params["odf_model"] =\
            self.tracking_params["odf_model"].upper()

        default_seg_params = get_default_args(seg.Segmentation.__init__)
        if segmentation_params is not None:
            for k in segmentation_params:
                default_seg_params[k] = segmentation_params[k]

        self.segmentation_params = default_seg_params
        self.seg_algo = self.segmentation_params["seg_algo"].lower()

        default_clean_params = get_default_args(seg.clean_bundle)
        if clean_params is not None:
            for k in clean_params:
                default_clean_params[k] = clean_params[k]

        self.clean_params = default_clean_params
        self.profile_weights = profile_weights
        if isinstance(self.profile_weights, str):
            self.profile_weights = self.profile_weights.lower()

        if bundle_info is None:
            if self.seg_algo == "reco" or self.seg_algo == "reco16":
                bundle_info = RECO_BUNDLES_16
            elif self.seg_algo == "reco80":
                bundle_info = RECO_BUNDLES_80
            else:
                bundle_info = BUNDLES

        # set reg_template and bundle_info:
        self.reg_template_img = self.get_reg_template()
        self.bundle_info = bundle_info
        self.bundle_dict = self.get_bundle_dict()

        if isinstance(
                self.segmentation_params["presegment_bundle_dict"], list):
            self.segmentation_params["presegment_bundle_dict"] =\
                make_bundle_dict(
                    bundle_names=self.segmentation_params[
                        "presegment_bundle_dict"],
                    seg_algo="afq",
                    resample_to=self.reg_template_img)

        # This is where all the outputs will go:
        self.afq_path = op.join(bids_path, 'derivatives', 'afq')

        # Create it as needed:
        os.makedirs(self.afq_path, exist_ok=True)

        bids_layout = BIDSLayout(bids_path, derivatives=True)
        bids_description = bids_layout.description

        # Add required metadata file at top level (inheriting as needed):
        pipeline_description = {
            "Name": bids_description["Name"],
            "BIDSVersion": bids_description["BIDSVersion"],
            "PipelineDescription": {"Name": "pyAFQ",
                                    "Version": pyafq_version}}

        pl_desc_file = op.join(self.afq_path, 'dataset_description.json')

        with open(pl_desc_file, 'w') as outfile:
            json.dump(pipeline_description, outfile)

        self.subjects = bids_layout.get(return_type='id', target='subject')
        if not len(self.subjects):
            raise ValueError(
                "`bids_path` contains no subjects in derivatives folders."
                + " This could be caused by derivatives folders not following"
                + " the BIDS format.")

        sessions = bids_layout.get(return_type='id', target='session')
        if len(sessions):
            self.sessions = sessions
        else:
            self.sessions = [None]

        self.sub_list = []
        self.ses_list = []
        self.dwi_file_list = []
        self.results_dir_list = []

        all_tasks = {}

        for task in [
                *data_tasks,
                *model_tasks,
                *mapping_tasks,
                *tractography_tasks,
                *segmentation_tasks,
                *profile_tasks,
                *viz_tasks]:
            all_tasks[task.function.__name__ + "_res"] = task

        # TODO: put these in their own functions before class definition
        all_tasks["scalar_func_res"] = gen_scalar_func(self.scalars)
        if custom_tractography_bids_filters is not None:
            all_tasks["streamlines_res"] = custom_tractography

        odf_model = self.tracking_params["odf_model"]
        if odf_model == "DTI":
            params_task = pimms.calc("params_file")(dti)
        elif odf_model == "CSD":
            params_task = pimms.calc("params_file")(csd)
        elif odf_model == "DKI":
            params_task = pimms.calc("params_file")(dki)
        else:
            raise TypeError((
                f"The ODF model you gave ({odf_model}) was not recognized"))
        all_tasks["params_file_res"] = params_task

        all_tasks["viz_bundles_res"] =\
            viz_bundles.tr({
                "volume": "b0_file",
                "color_by_volume": best_scalar + "_file"})
        all_tasks["viz_indivBundle_res"] =\
            viz_indivBundle.tr({
                "volume": "b0_file",
                "color_by_volume": best_scalar + "_file"})

        filename_dict = {
            "b0": "b0_file",
            "power_map": "pmap_file",
            "dti_fa_subject": "dti_fa_file",
            "subject_sls": "b0_file",
        }
        if self.reg_subject in filename_dict:
            all_tasks["get_reg_subject_res"] =\
                get_reg_subject.tr({
                    "reg_subject_spec": filename_dict[self.reg_subject]})

        stop_mask = self.tracking_params['stop_mask']
        if self.tracking_params["tracker"] == "pft":
            probseg_funcs = stop_mask.get_mask_getter()
            all_tasks["wm_res"] = pimms.calc("pve_wm")(probseg_funcs[0])
            all_tasks["gm_res"] = pimms.calc("pve_gm")(probseg_funcs[1])
            all_tasks["csf_res"] = pimms.calc("pve_csf")(probseg_funcs[2])
            all_tasks["export_stop_mask_res"] = \
                export_stop_mask_pft
        else:
            if isinstance(stop_mask, Definition):
                all_tasks["export_stop_mask_res"] = pimms.calc("stop_file")(
                    as_file('_stop_mask.nii.gz')(
                        stop_mask.get_mask_getter()))

        if isinstance(self.tracking_params['seed_mask'], Definition):
            all_tasks["export_seed_mask_res"] = pimms.calc("seed_file")(
                as_file('_seed_mask.nii.gz')(
                    self.tracking_params['seed_mask'].get_mask_getter()))

        all_tasks["brain_mask_res"] = \
            pimms.calc("brain_mask_file")(as_file('_brain_mask.nii.gz')(
                self.brain_mask_definition.get_mask_getter()))

        self.wf_dict = {}
        for session in self.sessions:
            if len(self.sessions) > 1:
                self.wf_dict[session] = {}
            for subject in self.subjects:
                results_dir = op.join(self.afq_path, 'sub-' + subject)

                if session is not None:
                    results_dir = op.join(results_dir, 'ses-' + session)

                dwi_bids_filters = {
                    "subject": subject,
                    "session": session,
                    "return_type": "filename",
                    "scope": dmriprep,
                    "extension": "nii.gz",
                    "suffix": "dwi",
                }
                dwi_bids_filters.update(bids_filters)
                dwi_files = bids_layout.get(**dwi_bids_filters)

                if (not len(dwi_files)):
                    self.logger.warning(
                        f"No dwi found for subject {subject} and session "
                        f"{session}. Skipping.")
                    continue

                self.results_dir_list.append(results_dir)
                os.makedirs(results_dir, exist_ok=True)

                dwi_data_file = dwi_files[0]
                self.dwi_file_list.append(dwi_data_file)

                # For bvals and bvecs, use ``get_bval()`` and ``get_bvec()`` to
                # walk up the file tree and inherit the closest bval and bvec
                # files. Maintain input ``bids_filters`` in case user wants to
                # specify acquisition labels, but pop suffix since it is
                # already specified inside ``get_bvec()`` and ``get_bval()``
                suffix = bids_filters.pop("suffix", None)
                bvec_file = bids_layout.get_bvec(
                    dwi_data_file,
                    **bids_filters)
                bval_file = bids_layout.get_bval(
                    dwi_data_file,
                    **bids_filters)
                if suffix is not None:
                    bids_filters["suffix"] = suffix

                if custom_tractography_bids_filters is not None:
                    custom_tract_file = \
                        bids_layout.get(subject=subject, session=session,
                                        extension=[
                                            '.trk',
                                            '.tck',
                                            '.vtk',
                                            '.fib',
                                            '.dpy'],
                                        return_type='filename',
                                        **custom_tractography_bids_filters)[0]
                else:
                    custom_tract_file = None

                if isinstance(self.reg_subject, dict):
                    reg_subject_spec = \
                        bids_layout.get(
                            **self.reg_subject,
                            session=session,
                            subject=subject,
                            return_type='filename'
                        )[0]
                else:
                    reg_subject_spec = self.reg_subject

                for scalar in self.scalars:
                    if isinstance(scalar, Definition):
                        scalar.find_path(
                            bids_layout,
                            dwi_data_file,
                            subject,
                            session
                        )

                if isinstance(
                        self.tracking_params["seed_mask"], Definition):
                    self.tracking_params["seed_mask"].find_path(
                        bids_layout,
                        dwi_data_file,
                        subject,
                        session
                    )

                if isinstance(
                        self.tracking_params["stop_mask"], Definition):
                    self.tracking_params["stop_mask"].find_path(
                        bids_layout,
                        dwi_data_file,
                        subject,
                        session
                    )

                self.brain_mask_definition.find_path(
                    bids_layout,
                    dwi_data_file,
                    subject,
                    session
                )

                self.mapping_definition.find_path(
                    bids_layout,
                    dwi_data_file,
                    subject,
                    session
                )

                self.sub_list.append(subject)
                self.ses_list.append(session)

                subses_dict = {
                    "ses": session,
                    "subject": subject,
                    "dwi_file": dwi_data_file,
                    "results_dir": results_dir}

                subses_plan = pimms.plan(**all_tasks)

                subses_data = subses_plan(
                    subses_dict=subses_dict,
                    bval_file=bval_file,
                    bvec_file=bvec_file,
                    b0_threshold=self.b0_threshold,
                    min_bval=self.min_bval,
                    max_bval=self.max_bval,
                    patch2self=self.patch2self,
                    brain_mask_definition=self.brain_mask_definition,
                    custom_tract_file=custom_tract_file,
                    reg_template=self.reg_template_img,
                    reg_subject_spec=reg_subject_spec,
                    bundle_dict=self.bundle_dict,
                    scalars=self.scalars,
                    mapping_definition=self.mapping_definition,
                    robust_tensor_fitting=self.robust_tensor_fitting,
                    profile_weights=self.profile_weights,
                    viz_backend=self.viz,
                    tracking_params=self.tracking_params,
                    segmentation_params=self.segmentation_params,
                    clean_params=self.clean_params)
                if len(self.sessions) > 1:
                    # TODO: subject then session
                    self.wf_dict[session][subject] = subses_data
                else:
                    self.wf_dict[subject] = subses_data

    def _get_best_scalar(self):
        for scalar in self.scalars:
            if isinstance(scalar, str):
                if "fa" in scalar:
                    return scalar
            else:
                if "fa" in scalar.name:
                    return scalar
        return self.scalars[0]

    def get_reg_template(self):
        img_l = self.reg_template.lower()
        if img_l == "mni_t2":
            img = afd.read_mni_template(
                mask=self.mask_template, weight="T2w")
        elif img_l == "mni_t1":
            img = afd.read_mni_template(
                mask=self.mask_template, weight="T1w")
        elif img_l == "dti_fa_template":
            img = afd.read_ukbb_fa_template(mask=self.mask_template)
        elif img_l == "hcp_atlas":
            img = afd.read_mni_template(mask=self.mask_template)
        else:
            img = nib.load(img)

        return img

    def get_bundle_dict(self):
        if isinstance(self.bundle_info, list):
            return make_bundle_dict(
                bundle_names=self.bundle_info,
                seg_algo=self.seg_algo,
                resample_to=self.reg_template_img)
        else:
            if self.seg_algo == "afq":
                _bundle_dict = self.bundle_info.copy()
                for bundle in _bundle_dict:
                    rois = _bundle_dict[bundle]['ROIs']
                    for ii, roi in enumerate(rois):
                        _bundle_dict[bundle]['ROIs'][ii] =\
                            afd.read_resample_roi(
                                roi, resample_to=self.reg_template_img)
                return _bundle_dict
            else:
                return self.bundle_info.copy()

    @_getter("results_dir", subses_dict=True)
    def get_results_dir(self):
        pass
    results_dir = property(get_results_dir, get_results_dir)

    @_getter("dwi_file", subses_dict=True)
    def get_dwi_file(self):
        pass
    dwi_file = property(get_dwi_file, get_dwi_file)

    @_getter("gtab")
    def get_gtab(self):
        pass
    gtab = property(get_gtab, get_gtab)

    @_getter("dwi_affine")
    def get_dwi_affine(self):
        pass
    dwi_affine = property(get_dwi_affine, get_dwi_affine)

    @_getter("dwi_img")
    def get_dwi_img(self):
        pass
    dwi_img = property(get_dwi_img, get_dwi_img)

    @_getter("b0_file")
    def get_b0(self):
        pass
    b0 = property(get_b0, get_b0)

    @_getter("masked_b0_file")
    def get_masked_b0(self):
        pass
    masked_b0_file = property(get_masked_b0, get_masked_b0)

    @_getter("brain_mask_file")
    def get_brain_mask(self):
        pass
    brain_mask = property(get_brain_mask, get_brain_mask)

    @_getter("dti_params_file")
    def get_dti(self):
        pass
    dti = property(get_dti, get_dti)

    @_getter("dti_fa_file")
    def get_dti_fa(self):
        pass
    dti_fa = property(get_dti_fa, get_dti_fa)

    @_getter("dti_cfa_file")
    def get_dti_cfa(self):
        pass
    dti_cfa = property(get_dti_cfa, get_dti_cfa)

    @_getter("dti_pdd_file")
    def get_dti_pdd(self):
        pass
    dti_pdd = property(get_dti_pdd, get_dti_pdd)

    @_getter("dti_md_file")
    def get_dti_md(self):
        pass
    dti_md = property(get_dti_md, get_dti_md)

    @_getter("dki_params_file")
    def get_dki(self):
        pass
    dki = property(get_dki, get_dki)

    @_getter("dki_mk_file")
    def get_dki_mk(self):
        pass
    dki_mk = property(get_dki_mk, get_dki_mk)

    @_getter("dki_fa_file")
    def get_dki_fa(self):
        pass
    dki_fa = property(get_dki_fa, get_dki_fa)

    @_getter("dki_md_file")
    def get_dki_md(self):
        pass
    dki_md = property(get_dki_md, get_dki_md)

    @_getter("dki_awf_file")
    def get_dki_awf(self):
        pass
    dki_awf = property(get_dki_awf, get_dki_awf)

    @_getter("mapping")
    def get_mapping(self):
        pass
    mapping = property(get_mapping, get_mapping)

    @_getter("streamlines_file")
    def get_streamlines(self):
        pass
    streamlines = property(get_streamlines, get_streamlines)

    @_getter("bundles_file")
    def get_bundles(self):
        pass
    bundles = property(get_bundles, get_bundles)

    @_getter("clean_bundles_file")
    def get_clean_bundles(self):
        pass
    clean_bundles = property(get_clean_bundles, get_clean_bundles)

    @_getter("profiles_file")
    def get_tract_profiles(self):
        pass
    tract_profiles = property(get_tract_profiles, get_tract_profiles)

    @_getter("template_xform_file")
    def get_template_xform(self):
        pass
    template_xform = property(get_template_xform, get_template_xform)

    @_getter("roi_files")
    def export_rois(self):
        pass

    @_getter("seed_file")
    def export_seed_mask(self):
        pass

    @_getter("stop_file")
    def export_stop_mask(self):
        pass

    @_getter("is_bundles_exported")
    def export_bundles(self):
        pass

    @_getter("sl_counts_file")
    def export_sl_counts(self):
        pass

    @_getter("all_bundles_figure_fname")
    def viz_bundles(self):
        pass

    @_getter("indiv_bundles_exported")
    def viz_ROIs(self):
        pass

    @_getter("tract_profiles_files")
    def plot_tract_profiles(self):
        pass

    @_getter("b0_warped_file")
    def export_registered_b0(self):
        pass

    def combine_profiles(self):
        tract_profiles_dict = self.tract_profiles
        if len(self.sessions) > 1:
            tract_profiles_list = []
            for _, subject_dict in tract_profiles_dict.items():
                tract_profiles_list.extend(subject_dict.values())
        else:
            tract_profiles_list = list(tract_profiles_dict.values())
        _df = combine_list_of_profiles(tract_profiles_list)
        out_file = op.abspath(op.join(
            self.afq_path, "tract_profiles.csv"))
        os.makedirs(op.dirname(out_file), exist_ok=True)
        _df.to_csv(out_file, index=False)
        return _df

    def export_all(self):
        """ Exports all the possible outputs"""
        start_time = time()
        if not isinstance(self.mapping_definition, FnirtMap)\
                and not isinstance(self.mapping_definition, ItkMap):
            self.export_registered_b0()
        self.get_template_xform()
        self.export_bundles()
        self.export_sl_counts()
        self.get_tract_profiles()
        if len(self.tract_profiles) > 1:
            self.combine_profiles()
        self.viz_bundles()
        if self.seg_algo == "afq":
            self.viz_ROIs()
            self.export_rois()
        self.logger.info(
            "Time taken for export all: " + str(time() - start_time))

    def upload_to_s3(self, s3fs, remote_path):
        """ Upload entire AFQ derivatives folder to S3"""
        s3fs.put(self.afq_path, remote_path, recursive=True)


def download_and_combine_afq_profiles(bucket,
                                      study_s3_prefix="", deriv_name=None,
                                      out_file=None,
                                      upload=False, session=None,
                                      **kwargs):
    """
    Download and combine tract profiles from different subjects / sessions
    on an s3 bucket into one CSV.
    Parameters
    ----------
    bucket : str
        The S3 bucket that contains the study data.
    study_s3_prefix : str
        The S3 prefix common to all of the study objects on S3.
    out_file : filename, optional
        Filename for the combined output CSV.
    deriv_name : str, optional
        If deriv_name is not None, it should be a string that specifies
        which derivatives folder to download and combine profiles from.
    upload : bool or str, optional
        If True, upload the combined CSV to Amazon S3 at
        bucket/study_s3_prefix/derivatives/afq. If a string,
        assume string is an Amazon S3 URI and upload there.
        Defaut: False
    session : str, optional
        Session to get CSVs from. If None, all sessions are used.
        Default: None
    kwargs : optional
        Optional arguments to pass to S3BIDSStudy.
    Returns
    -------
    Ouput CSV's pandas dataframe.
    """
    if "subjects" not in kwargs:
        kwargs["subjects"] = "all"
    if "anon" not in kwargs:
        kwargs["anon"] = False
    if deriv_name is None:
        deriv_name = True

    with nib.tmpdirs.InTemporaryDirectory() as t_dir:
        remote_study = afd.S3BIDSStudy(
            "get_profiles",
            bucket,
            study_s3_prefix,
            **kwargs)
        remote_study.download(
            t_dir,
            include_modality_agnostic=False,
            include_derivs=deriv_name,
            include_derivs_dataset_description=True,
            suffix="profiles.csv")
        temp_study = BIDSLayout(t_dir, validate=False, derivatives=True)
        if session is None:
            profiles = temp_study.get(
                extension='csv',
                suffix='profiles',
                return_type='filename')
        else:
            profiles = temp_study.get(
                session=session,
                extension='csv',
                suffix='profiles',
                return_type='filename')

        df = combine_list_of_profiles(profiles)
        df.to_csv("tmp.csv")
        if upload is True:
            bids_prefix = "/".join([bucket, study_s3_prefix]).rstrip("/")
            fs = s3fs.S3FileSystem()
            fs.put(
                "tmp.csv",
                "/".join([
                    bids_prefix,
                    "derivatives",
                    "afq",
                    "combined_tract_profiles.csv"
                ]))
        elif isinstance(upload, str):
            fs = s3fs.S3FileSystem()
            fs.put("tmp.csv", upload.replace("s3://", ""))

    if out_file is not None:
        out_file = op.abspath(out_file)
        os.makedirs(op.dirname(out_file), exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def combine_list_of_profiles(profile_fnames):
    """
    Combine tract profiles from different subjects / sessions
    into one CSV.

    Parameters
    ----------
    profile_fnames : list of str
        List of csv filenames.

    Returns
    -------
    Ouput CSV's pandas dataframe.
    """
    dfs = []
    for fname in profile_fnames:
        profiles = pd.read_csv(fname)
        profiles['subjectID'] = fname.split('sub-')[1].split('/')[0]
        if 'ses-' in fname:
            session_name = fname.split('ses-')[1].split('/')[0]
        else:
            session_name = 'unknown'
        profiles['sessionID'] = session_name
        dfs.append(profiles)

    return pd.concat(dfs)
