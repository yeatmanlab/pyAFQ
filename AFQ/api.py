# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import logging
from AFQ.definitions.mask import (B0Mask, ScalarMask, FullMask)
from AFQ.definitions.mapping import (SynMap, FnirtMap, ItkMap, SlrMap)
from AFQ.definitions.utils import Definition
from AFQ.utils.bin import get_default_args
import AFQ.segmentation as seg
import AFQ.tractography as aft
import AFQ.data as afd
from AFQ.viz.utils import Viz
from AFQ.utils.parallel import parfor

from AFQ.tasks.data import get_data_plan
from AFQ.tasks.mapping import get_mapping_plan
from AFQ.tasks.tractography import get_tractography_plan
from AFQ.tasks.segmentation import get_segmentation_plan
from AFQ.tasks.viz import get_viz_plan

from .version import version as pyafq_version
import pandas as pd
import os
import os.path as op
import json
import s3fs
from time import time
import nibabel as nib
from importlib import import_module
from collections.abc import MutableMapping

from bids.layout import BIDSLayout
import bids.config as bids_config
try:
    bids_config.set_option('extension_initial_dot', True)
except ValueError:
    pass


logging.basicConfig(level=logging.INFO)


__all__ = ["AFQ", "BundleDict"]


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
                   "CC_ForcepsMinor", "CC", "CCMid", "CNII", "CNIII",
                   "CNIV", "CNV", "CNVII", "CNVIII", "CS", "CST", "CT",
                   "CTT", "DLF", "EMC", "F_L_R", "FPT", "ICP", "IFOF", "ILF",
                   "LL", "MCP", "MdLF", "ML", "MLF", "OPT", "OR", "PC", "PPT",
                   "RST", "SCP", "SLF", "STT", "TPT", "UF", "V", "VOF"]

RECO_UNIQUE = [
    'CCMid', 'CC_ForcepsMajor', 'CC_ForcepsMinor', 'MCP', 'AC', 'PC', 'SCP',
    'V', 'CC', 'F_L_R']

DIPY_GH = "https://github.com/dipy/dipy/blob/master/dipy/"


class BundleDict(MutableMapping):
    def __init__(self,
                 bundle_info=BUNDLES,
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

        resample_to : Nifti1Image or bool, optional
            If set, templates will be resampled to the affine and shape of this
            image. If False, no resampling will be done.
            Default: False
        """
        if not (isinstance(bundle_info, dict)
                or isinstance(bundle_info, list)):
            raise TypeError((
                f"bundle_info must be a dict or a list,"
                f" currently a {type(bundle_info)}"))

        if isinstance(bundle_info, dict):
            self.bundle_names = list(bundle_info.keys())
            self.bundle_dict_unresampled = bundle_info.copy()
        else:
            self.bundle_names = bundle_info.copy()
            self.bundle_dict_unresampled = None

        self.logger = logging.getLogger('AFQ.api')
        self.seg_algo = seg_algo.lower()
        self.resample_to = resample_to
        self.bundle_dict_resampled = None
        self._dict = None

        if self.seg_algo == "afq":
            if "FP" in self.bundle_names\
                    and "Occipital" in self.bundle_names:
                self.logger.warning((
                    "FP and Occipital bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Occipital will be used."))
                self.bundle_names.remove("FP")
            if "FA" in self.bundle_names\
                    and "Orbital" in self.bundle_names:
                self.logger.warning((
                    "FA and Orbital bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only Orbital will be used."))
                self.bundle_names.remove("FA")
            if "FA" in self.bundle_names\
                    and "AntFrontal" in self.bundle_names:
                self.logger.warning((
                    "FA and AntFrontal bundles are co-located, and AFQ"
                    " assigns each streamline to only one bundle."
                    " Only AntFrontal will be used."))
                self.bundle_names.remove("FA")

    def _list_to_dict(self):
        if self.seg_algo == "afq":
            templates = afd.read_templates(resample_to=self.resample_to)
            callosal_templates = afd.read_callosum_templates(
                resample_to=self.resample_to)
            # For the arcuate, we need to rename a few of these and duplicate
            # the SLF ROI:
            templates['ARC_roi1_L'] = templates['SLF_roi1_L']
            templates['ARC_roi1_R'] = templates['SLF_roi1_R']
            templates['ARC_roi2_L'] = templates['SLFt_roi2_L']
            templates['ARC_roi2_R'] = templates['SLFt_roi2_R']

            afq_bundles = {}
            # Each bundles gets a digit identifier
            # (to be stored in the tractogram)
            uid = 1
            for name in self.bundle_names:
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
                                'prob_map': templates[
                                    name + hemi + '_prob_map'],
                                'cross_midline': False,
                                'uid': uid}
                        else:
                            self.logger.warning(
                                f"{name} is not in AFQ templates")

                        uid += 1
        elif self.seg_algo.startswith("reco"):
            if self.seg_algo.endswith("80"):
                bundle_dict = afd.read_hcp_atlas(80)
            else:
                bundle_dict = afd.read_hcp_atlas(16)

            afq_bundles = {}
            uid = 1
            afq_bundles["whole_brain"] = bundle_dict["whole_brain"]
            for name in self.bundle_names:
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
            raise ValueError(
                "Input: %s is not a valid input`seg_algo`" % self.seg_algo)
        self.bundle_dict_unresampled = afq_bundles

    def _gen_resampled_dict(self):
        if self.bundle_dict_unresampled is None:
            self._list_to_dict()

        if self.seg_algo == "afq" and self.resample_to:
            self.bundle_dict_resampled = self.bundle_dict_unresampled.copy()
            for bundle in self.bundle_dict_unresampled:
                rois = self.bundle_dict_unresampled[bundle]['ROIs']
                for ii, roi in enumerate(rois):
                    self.bundle_dict_resampled[bundle]['ROIs'][ii] =\
                        afd.read_resample_roi(
                            roi, resample_to=self.resample_to)
        else:
            self.bundle_dict_resampled = self.bundle_dict_unresampled

        self._dict = self.bundle_dict_resampled

    def __getattribute__(self, attr):
        if attr == "_dict":
            if object.__getattribute__(self, attr) is None:
                self._gen_resampled_dict()
        return object.__getattribute__(self, attr)

    def __setitem__(self, key, item):
        self._dict[key] = item

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return len(self._dict)

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict)


# this is parallelized below
def _getter_helper(wf_dict, attr_name):
    return wf_dict[attr_name]


# define sections in workflow dictionary
_sub_attrs = [
    "data_imap", "mapping_imap",
    "tractography_imap", "segmentation_imap",
    "subses_dict"]

task_outputs = []
for task_module in ["data", "mapping", "segmentation", "tractography", "viz"]:
    task_outputs.extend(import_module(f"AFQ.tasks.{task_module}").outputs)


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
                 parallel_params={
                     "n_jobs": -1, "engine": "joblib",
                     "backend": "loky"},
                 scalars=["dti_fa", "dti_md"],
                 virtual_frame_buffer=False,
                 viz_backend="plotly_no_gif",
                 tracking_params=None,
                 segmentation_params=None,
                 clean_params=None,
                 **kwargs):
        '''
        Initialize an AFQ object.
        Some special notes on parameters:
        In tracking_params, parameters with the suffix mask which are also
        a mask from AFQ.definitions.mask will be handled automatically by the
        api.

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
            [DATA] The value of b under which
            it is considered to be b0. Default: 50.
        patch2self : bool, optional
            [DATA] Whether to use patch2self
            to denoise the dwi data.
            Default: False
        robust_tensor_fitting : bool, optional
            [DATA] Whether to use robust_tensor_fitting when
            doing dti. Only applies to dti.
            Default: False
        min_bval : float, optional
            [DATA] Minimum b value you want to use
            from the dataset (other than b0), inclusive.
            If None, there is no minimum limit. Default: None
        max_bval : float, optional
            [DATA] Maximum b value you want to use
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
            [REGISTRATION] This defines how to either create a mapping from
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
            or a bundle dictionary (see BundleDict for inspiration).
            If None, will get all appropriate bundles for the chosen
            segmentation algorithm.
            Default: None
        parallel_params : dict, optional
            [COMPUTE] Parameters to pass to parfor in AFQ.utils.parallel,
            to parallelize computations across subjects and sessions.
            Set "n_jobs" to -1 to automatically parallelize as
            the number of cpus.
            Default: {"n_jobs": -1, "engine": "joblib", "backend": "loky"}
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
        kwargs : additional optional parameters
            [KWARGS] You can set additional parameters for any step
            of the process.
            For example, to set the sh_order for csd to 4, do:
            api.AFQ(my_path, sh_order=4)
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
        if not isinstance(parallel_params, dict):
            raise TypeError("parallel_params must be a dict")
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
        self.parallel_params = parallel_params
        self.wf_dict = {}

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
        self.bundle_dict = BundleDict(
            bundle_info,
            seg_algo=self.seg_algo,
            resample_to=self.reg_template_img)

        if not (isinstance(
                self.segmentation_params["presegment_bundle_dict"],
                BundleDict)
                or self.segmentation_params["presegment_bundle_dict"]
                is None):
            self.segmentation_params["presegment_bundle_dict"] =\
                BundleDict(
                    self.segmentation_params["presegment_bundle_dict"],
                    seg_algo="afq",
                    resample_to=self.reg_template_img)

        # This is where all the outputs will go:
        self.afq_path = op.join(bids_path, 'derivatives', 'afq')

        # Create it as needed:
        os.makedirs(self.afq_path, exist_ok=True)

        bids_layout = BIDSLayout(bids_path, derivatives=True)
        bids_description = bids_layout.description

        # check that any files exist in the derivatives folder,
        # not including the dataset_description.json files
        # the second check may be particularly useful in checking
        # that the derivatives folder is well-defined
        if len(bids_layout.get())\
                - len(bids_layout.get(extension="json")) < 1:
            raise ValueError(
                f"No non-json files recognized by pyBIDS in {bids_path}")
        if len(bids_layout.get(scope=dmriprep))\
                - len(bids_layout.get(scope=dmriprep, extension="json")) < 1:
            raise ValueError((
                f"No non-json files recognized by "
                f"pyBIDS in the pipeline: {dmriprep}"))

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

        # do not bother to parallelize if less than 2 subject-sessions
        if len(self.sessions) * len(self.subjects) < 2:
            self.parallel_params["engine"] = "serial"

        # do not parallelize segmentation if parallelizing across
        # subject-sessions
        if self.parallel_params["engine"] != "serial":
            self.segmentation_params["parallel_segmentation"]["engine"] =\
                "serial"

        # construct pimms plans
        if isinstance(mapping, SlrMap):
            plans = {  # if using SLR map, do tractography first
                "data": get_data_plan(self.brain_mask_definition),
                "tractography": get_tractography_plan(
                    self.custom_tractography_bids_filters,
                    self.tracking_params),
                "mapping": get_mapping_plan(
                    self.reg_subject, self.scalars, use_sls=True),
                "segmentation": get_segmentation_plan(),
                "viz": get_viz_plan()}
        else:
            plans = {  # Otherwise, do mapping first
                "data": get_data_plan(self.brain_mask_definition),
                "mapping": get_mapping_plan(self.reg_subject, self.scalars),
                "tractography": get_tractography_plan(
                    self.custom_tractography_bids_filters,
                    self.tracking_params),
                "segmentation": get_segmentation_plan(),
                "viz": get_viz_plan()}

        self.valid_sub_list = []
        self.valid_ses_list = []
        for subject in self.subjects:
            self.wf_dict[subject] = {}
            for session in self.sessions:
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

                os.makedirs(results_dir, exist_ok=True)
                dwi_data_file = dwi_files[0]

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

                self.valid_sub_list.append(subject)
                self.valid_ses_list.append(session)

                img = nib.load(dwi_data_file)
                subses_dict = {
                    "ses": session,
                    "subject": subject,
                    "dwi_file": dwi_data_file,
                    "results_dir": results_dir}

                input_data = dict(
                    subses_dict=subses_dict,
                    dwi_img=img,
                    dwi_affine=img.affine,
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
                    best_scalar=best_scalar,
                    tracking_params=self.tracking_params,
                    segmentation_params=self.segmentation_params,
                    clean_params=self.clean_params,
                    **kwargs)

                # chain together a complete plan from individual plans
                previous_data = {}
                for name, plan in plans.items():
                    previous_data[f"{name}_imap"] = plan(
                        **input_data,
                        **previous_data)
                    last_name = name

                self.wf_dict[subject][str(session)] =\
                    previous_data[f"{last_name}_imap"]

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
            img = nib.load(self.reg_template)

        return img

    def __getattribute__(self, attr):
        # check if normal attr exists first
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:
            pass

        # find what name to use
        first_dict =\
            self.wf_dict[self.valid_sub_list[0]][str(self.valid_ses_list[0])]
        attr_file = attr + "_file"
        attr_name = None
        if attr in first_dict:
            attr_name = attr
            section = None
        elif attr_file in first_dict:
            attr_name = attr_file
            section = None
        else:
            for sub_attr in _sub_attrs:
                if attr in first_dict[sub_attr]:
                    attr_name = attr
                    section = sub_attr
                    break
                elif attr_file in first_dict[sub_attr]:
                    attr_name = attr_file
                    section = sub_attr
                    break

        # attr not found, allow typical AttributeError
        if attr_name is None:
            return object.__getattribute__(self, attr)

        # iterate over subjects / sessions,
        # decide if they need to be calculated or not
        in_list = []
        to_calc_list = []
        results = {}
        for subject in self.valid_sub_list:
            results[subject] = {}
            for session in self.valid_ses_list:
                wf_dict = self.wf_dict[subject][str(session)]
                if section is not None:
                    wf_dict = wf_dict[section]
                if ((self.parallel_params.get("engine", False) != "serial")
                        and (hasattr(wf_dict, "efferents"))
                        and (attr_name not in wf_dict.efferents)):
                    in_list.append((wf_dict))
                    to_calc_list.append((subject, session))
                else:
                    results[subject][session] =\
                        _getter_helper(wf_dict, attr_name)

        # if some need to be calculated, do those in parallel
        if len(to_calc_list) > 0:
            par_results = parfor(
                _getter_helper, in_list,
                func_args=[attr_name],
                **self.parallel_params)

            for i, subses in enumerate(to_calc_list):
                subject, session = subses
                results[subject][session] = par_results[i]

        # If only one session, collapse session dimension
        if len(self.sessions) == 1:
            for subject in self.valid_sub_list:
                results[subject] = results[subject][self.valid_ses_list[0]]

        return results

    def combine_profiles(self):
        tract_profiles_dict = self.profiles
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
            self.b0_warped
        self.template_xform
        self.indiv_bundles
        self.sl_counts
        self.tract_profile_plots
        self.profiles
        if len(self.sessions) * len(self.subjects) > 1:
            self.combine_profiles()
        self.all_bundles_figure
        if self.seg_algo == "afq":
            self.indiv_bundles_figures
            self.rois
        self.logger.info(
            "Time taken for export all: " + str(time() - start_time))

    def upload_to_s3(self, s3fs, remote_path):
        """ Upload entire AFQ derivatives folder to S3"""
        s3fs.put(self.afq_path, remote_path, recursive=True)


# iterate through all attributes, setting methods for each one
for output in task_outputs:
    exec(f"def export_{output}(self):\n    return self.{output}")
    fn = locals()[f"export_{output}"]
    if output[-5:] == "_file":
        setattr(AFQ, f"export_{output[:-5]}", fn)
    else:
        setattr(AFQ, f"export_{output}", fn)


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
