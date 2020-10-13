# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import pandas as pd
import dask.dataframe as ddf
import os
import os.path as op
import json
from time import time

import numpy as np
import nibabel as nib

import dipy.core.gradients as dpg
import dipy.tracking.utils as dtu
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.gradients import read_bvals_bvecs
from dipy.stats.analysis import afq_profile, gaussian_weights

from bids.layout import BIDSLayout

from .version import version as pyafq_version
import AFQ.data as afd
from AFQ.models.dti import _fit as dti_fit
from AFQ.models.dki import _fit as dki_fit
from AFQ.models.csd import _fit as csd_fit
from AFQ.models.csd import fit_anisotropic_power_map
import AFQ.tractography as aft
import dipy.reconst.dti as dpy_dti
import dipy.reconst.dki as dpy_dki
import AFQ.utils.streamlines as aus
import AFQ.segmentation as seg
import AFQ.registration as reg
import AFQ.utils.volume as auv
from AFQ.viz.utils import Viz, visualize_tract_profiles
from AFQ.utils.bin import get_default_args
from AFQ.mask import (B0Mask, ScalarMask, FullMask, check_mask_methods)
import logging
logging.basicConfig(level=logging.INFO)


__all__ = ["AFQ", "make_bundle_dict"]


def do_preprocessing():
    raise NotImplementedError


BUNDLES = ["ATR", "CGC", "CST", "IFO", "ILF", "SLF", "ARC", "UNC",
           "FA", "FP"]

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


def make_bundle_dict(bundle_names=BUNDLES, seg_algo="afq", resample_to=False):
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
    if seg_algo == "afq":
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
                    afq_bundles[name + hemi] = {
                        'ROIs': [templates[name + '_roi1' + hemi],
                                 templates[name + '_roi2' + hemi]],
                        'rules': [True, True],
                        'prob_map': templates[name + hemi + '_prob_map'],
                        'cross_midline': False,
                        'uid': uid}

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


class AFQ(object):
    """
    """

    def __init__(self,
                 bids_path,
                 bids_filters={"suffix": "dwi"},
                 dmriprep="all",
                 custom_tractography_bids_filters=None,
                 b0_threshold=50,
                 min_bval=None,
                 max_bval=None,
                 reg_template="mni_T1",
                 reg_subject="power_map",
                 brain_mask=B0Mask(),
                 bundle_info=None,
                 dask_it=False,
                 force_recompute=False,
                 scalars=["dti_fa", "dti_md"],
                 use_prealign=True,
                 virtual_frame_buffer=False,
                 viz_backend="plotly_no_gif",
                 tracking_params=None,
                 segmentation_params=None,
                 clean_params=None):
        '''
        Initialize an AFQ object.
        Some special notes on parameters:
        In tracking_params, parameters with the suffix mask which are also
        a mask from AFQ.mask will be handled automatically by the api.
        You can set additional parameters for a given step of the process
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
        brain_mask : instance of a class defined in `AFQ.mask`, optional
            [REGISTRATION] This will be used to create
            the brain mask, which gets applied before registration to a
            template.
            If None, no brain mask will not be applied,
            and no brain mask will be applied to the template.
            Default: B0Mask()
        bundle_info : list of strings or dict, optional
            [BUNDLES] List of bundle names to include in segmentation,
            or a bundle dictionary (see make_bundle_dict for inspiration).
            If None, will get all appropriate bundles for the chosen
            segmentation algorithm.
            Default: None
        dask_it : bool, optional
            [COMPUTE] Whether to use a dask DataFrame object.
            Default: False
        force_recompute : bool, optional
            [COMPUTE] Whether to recompute or ignore existing derivatives.
            This parameter can be turned on/off dynamically.
            Default: False
        scalars : list of strings, optional
            [BUNDLES] List of scalars to use.
            Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md"
            Default: ["dti_fa", "dti_md"]
        use_prealign : bool, optional
            [REGISTRATION] Whether to perform pre-alignment before perforiming
            the diffeomorphic mapping in registration. Default: True
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
            The parameters for tracking.
            Default: use the default behavior of the aft.track function.
            Seed mask and seed threshold, if not specified, are replaced
            with scalar masks from scalar[0] thresholded to 0.2.
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
        if brain_mask is not None and not check_mask_methods(brain_mask):
            raise TypeError(
                "brain_mask must be None or a mask defined in `AFQ.mask`")
        if bundle_info is not None and not ((
                isinstance(bundle_info, list)
                and isinstance(bundle_info[0], str)) or (
                    isinstance(bundle_info, dict))):
            raise TypeError(
                "bundle_info must be None, a list of strings, or a dict")
        if not isinstance(dask_it, bool):
            raise TypeError("dask_it must be a bool")
        if not isinstance(force_recompute, bool):
            raise TypeError("force_recompute must be a bool")
        if scalars is not None and not (
                isinstance(scalars, list)
                and isinstance(scalars[0], str)):
            raise TypeError(
                "scalars must be None or a list of strings")
        if not isinstance(use_prealign, bool):
            raise TypeError("use_prealign must be a bool")
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

        self.force_recompute = force_recompute

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

        if isinstance(reg_subject, str) and isinstance(reg_template, str)\
                and (reg_subject.lower() == 'subject_sls'
                     or reg_template.lower() == 'hcp_atlas'):
            if reg_template.lower() != 'hcp_atlas':
                self.logger.error(
                    "If reg_subject is 'subject_sls',"
                    + " reg_template must be 'hcp_atlas'")
            if reg_subject.lower() != 'subject_sls':
                self.logger.error(
                    "If reg_template is 'hcp_atlas',"
                    + " reg_subject must be 'subject_sls'")

            self.reg_algo = 'slr'
        else:
            self.reg_algo = 'syn'
        self.use_prealign = (use_prealign and (self.reg_algo != 'slr'))
        self.b0_threshold = b0_threshold
        self.custom_tractography_bids_filters =\
            custom_tractography_bids_filters

        self.scalars = [scalar.lower() for scalar in scalars]

        if virtual_frame_buffer:
            from xvfbwrapper import Xvfb
            self.vdisplay = Xvfb(width=1280, height=1280)
            self.vdisplay.start()
        self.viz = Viz(backend=viz_backend.lower())

        default_tracking_params = get_default_args(aft.track)
        default_tracking_params["seed_mask"] = ScalarMask(
            self._get_best_scalar())
        default_tracking_params["stop_mask"] = ScalarMask(
            self._get_best_scalar())
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

        if bundle_info is None:
            if self.seg_algo == "reco" or self.seg_algo == "reco16":
                bundle_info = RECO_BUNDLES_16
            elif self.seg_algo == "reco80":
                bundle_info = RECO_BUNDLES_80
            else:
                bundle_info = BUNDLES

        # Create the bundle dict after reg_template has been resolved:
        self.reg_template_img, _ = self._reg_img(self.reg_template, False)
        if isinstance(bundle_info, list):
            self.bundle_dict = make_bundle_dict(
                bundle_names=bundle_info,
                seg_algo=self.seg_algo,
                resample_to=self.reg_template_img)
        else:
            self.bundle_dict = bundle_info

        # Initialize dict to store relevant timing information
        timing_dict = {
            "Tractography": 0,
            "Registration": 0,
            "Segmentation": 0,
            "Cleaning": 0,
            "Visualization": 0
        }

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

        sub_list = []
        ses_list = []
        dwi_file_list = []
        bvec_file_list = []
        bval_file_list = []
        custom_tract_list = []
        reg_subject_list = []
        timing_list = []
        results_dir_list = []
        for subject in self.subjects:
            for session in self.sessions:
                results_dir = op.join(self.afq_path, 'sub-' + subject)

                if session is not None:
                    results_dir = op.join(results_dir, 'ses-' + session)

                results_dir_list.append(results_dir)

                os.makedirs(results_dir_list[-1], exist_ok=True)
                dwi_file_list.append(
                    bids_layout.get(subject=subject, session=session,
                                    extension='nii.gz',
                                    return_type='filename',
                                    scope=dmriprep,
                                    **bids_filters)[0])
                bvec_file_list.append(
                    bids_layout.get(subject=subject, session=session,
                                    extension=['bvec', 'bvecs'],
                                    return_type='filename',
                                    scope=dmriprep,
                                    **bids_filters)[0])
                bval_file_list.append(
                    bids_layout.get(subject=subject, session=session,
                                    extension=['bval', 'bvals'],
                                    return_type='filename',
                                    scope=dmriprep,
                                    **bids_filters)[0])

                if custom_tractography_bids_filters is not None:
                    custom_tract_list.append(
                        bids_layout.get(subject=subject, session=session,
                                        extension=[
                                            '.trk',
                                            '.tck',
                                            '.vtk',
                                            '.fib',
                                            '.dpy'],
                                        return_type='filename',
                                        **custom_tractography_bids_filters)[0])
                else:
                    custom_tract_list.append(None)

                if isinstance(self.reg_subject, dict):
                    reg_subject_list.append(
                        bids_layout.get(subject=subject, session=session,
                                        return_type='filename',
                                        **self.reg_subject)[0])
                else:
                    reg_subject_list.append(None)

                if check_mask_methods(self.tracking_params["seed_mask"]):
                    self.tracking_params["seed_mask"].find_path(
                        bids_layout,
                        subject,
                        session
                    )

                if check_mask_methods(self.tracking_params["stop_mask"]):
                    self.tracking_params["stop_mask"].find_path(
                        bids_layout,
                        subject,
                        session
                    )

                self.brain_mask_definition.find_path(
                    bids_layout,
                    subject,
                    session
                )

                sub_list.append(subject)
                ses_list.append(session)
                timing_list.append(timing_dict.copy())

        self.data_frame = pd.DataFrame(dict(subject=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            custom_tract=custom_tract_list,
                                            reg_subject=reg_subject_list,
                                            ses=ses_list,
                                            timing=timing_list,
                                            results_dir=results_dir_list))

        if dask_it:
            self.data_frame = ddf.from_pandas(self.data_frame,
                                              npartitions=len(sub_list))
        self.set_gtab(b0_threshold)
        self.set_dwi_affine()
        self.set_dwi_img()

    def log_and_save_nii(self, img, fname):
        self.logger.info(f"Saving {fname}")
        nib.save(img, fname)

    def log_and_save_trk(self, sft, fname):
        self.logger.info(f"Saving {fname}")
        save_tractogram(sft, fname, bbox_valid_check=False)

    def _get_data_gtab(self, row, filter_b=True):
        img = nib.load(row['dwi_file'])
        data = img.get_fdata()
        bvals, bvecs = read_bvals_bvecs(row['bval_file'], row['bvec_file'])
        if filter_b and (self.min_bval is not None):
            valid_b = np.logical_or(
                (bvals >= self.min_bval), (bvals <= self.b0_threshold))
            data = data[..., valid_b]
            bvals = bvals[valid_b]
            bvecs = bvecs[valid_b]
        if filter_b and (self.max_bval is not None):
            valid_b = np.logical_or(
                (bvals <= self.max_bval), (bvals <= self.b0_threshold))
            data = data[..., valid_b]
            bvals = bvals[valid_b]
            bvecs = bvecs[valid_b]
        gtab = dpg.gradient_table(bvals, bvecs,
                                  b0_threshold=self.b0_threshold)
        return data, gtab, img

    def _b0(self, row):
        b0_file = self._get_fname(row, '_b0.nii.gz')
        if self.force_recompute or not op.exists(b0_file):
            data, gtab, img = self._get_data_gtab(row, filter_b=False)
            mean_b0 = np.mean(data[..., ~gtab.b0s_mask], -1)
            mean_b0_img = nib.Nifti1Image(mean_b0, img.affine)
            self.log_and_save_nii(mean_b0_img, b0_file)

            meta = dict(b0_threshold=gtab.b0_threshold,
                        source=row['dwi_file'])
            meta_fname = self._get_fname(row, '_b0.json')
            afd.write_json(meta_fname, meta)
        return b0_file

    def _brain_mask(self, row, median_radius=4, numpass=1, autocrop=False,
                    vol_idx=None, dilate=10):
        brain_mask_file = self._get_fname(row, '_brain_mask.nii.gz')
        if self.force_recompute or not op.exists(brain_mask_file):
            brain_mask, brain_affine, meta = \
                self.brain_mask_definition.get_mask(self, row)
            brain_mask_img = nib.Nifti1Image(
                brain_mask.astype(int),
                brain_affine)
            self.log_and_save_nii(brain_mask_img, brain_mask_file)
            meta_fname = self._get_fname(row, '_brain_mask.json')
            afd.write_json(meta_fname, meta)
        return brain_mask_file

    def _dti_fit(self, row):
        dti_params_file = self._dti(row)
        dti_params = nib.load(dti_params_file).get_fdata()
        tm = dpy_dti.TensorModel(row['gtab'])
        tf = dpy_dti.TensorFit(tm, dti_params)
        return tf

    def _dti(self, row):
        dti_params_file = self._get_fname(row, '_model-DTI_diffmodel.nii.gz')
        if self.force_recompute or not op.exists(dti_params_file):
            data, gtab, _ = self._get_data_gtab(row)
            brain_mask_file = self._brain_mask(row)
            mask = nib.load(brain_mask_file).get_fdata()
            dtf = dti_fit(gtab, data, mask=mask)
            self.log_and_save_nii(nib.Nifti1Image(dtf.model_params,
                                                  row['dwi_affine']),
                                  dti_params_file)
            meta_fname = self._get_fname(row, '_model-DTI_diffmodel.json')
            meta = dict(
                Parameters=dict(
                    FitMethod="WLS"),
                OutlierRejection=False,
                ModelURL=f"{DIPY_GH}reconst/dti.py")
            afd.write_json(meta_fname, meta)
        return dti_params_file

    def _dki_fit(self, row):
        dki_params_file = self._dki(row)
        dki_params = nib.load(dki_params_file).get_fdata()
        tm = dpy_dki.DiffusionKurtosisModel(row['gtab'])
        tf = dpy_dki.DiffusionKurtosisFit(tm, dki_params)
        return tf

    def _dki(self, row):
        dki_params_file = self._get_fname(row, '_model-DKI_diffmodel.nii.gz')
        if self.force_recompute or not op.exists(dki_params_file):
            data, gtab, _ = self._get_data_gtab(row)
            brain_mask_file = self._brain_mask(row)
            mask = nib.load(brain_mask_file).get_fdata()
            dkf = dki_fit(gtab, data, mask=mask)
            nib.save(nib.Nifti1Image(dkf.model_params, row['dwi_affine']),
                     dki_params_file)
            meta_fname = self._get_fname(row, '_model-DKI_diffmodel.json')
            meta = dict(
                Parameters=dict(
                    FitMethod="WLS"),
                OutlierRejection=False,
                ModelURL=f"{DIPY_GH}reconst/dki.py")
            afd.write_json(meta_fname, meta)
        return dki_params_file

    def _csd(self, row, response=None, sh_order=None, lambda_=1, tau=0.1,
             msmt=False):
        if msmt:
            model_str = "MSMT"
            model_file = "mcsd.py"
        else:
            model_str = "CSD"
            model_file = "csdeconv.py"
        csd_params_file = self._get_fname(
            row,
            f'_model-{model_str}_diffmodel.nii.gz')
        if self.force_recompute or not op.exists(csd_params_file):
            data, gtab, _ = self._get_data_gtab(row)
            brain_mask_file = self._brain_mask(row)
            mask = nib.load(brain_mask_file).get_fdata()
            csdf = csd_fit(gtab, data, mask=mask,
                           response=response, sh_order=sh_order,
                           lambda_=lambda_, tau=tau, msmt=msmt)
            self.log_and_save_nii(nib.Nifti1Image(csdf.shm_coeff,
                                                  row['dwi_affine']),
                                  csd_params_file)
            meta_fname = self._get_fname(
                row,
                f'_model-{model_str}_diffmodel.json')
            meta = dict(SphericalHarmonicDegree=sh_order,
                        ResponseFunctionTensor=response,
                        SphericalHarmonicBasis="DESCOTEAUX",
                        ModelURL=f"{DIPY_GH}reconst/{model_file}",
                        lambda_=lambda_,
                        tau=tau)
            afd.write_json(meta_fname, meta)
        return csd_params_file

    def _anisotropic_power_map(self, row):
        pmap_file = self._get_fname(
            row, '_model-CSD_APM.nii.gz')
        if self.force_recompute or not op.exists(pmap_file):
            dwi_data, gtab, img = self._get_data_gtab(row)
            mask = self._brain_mask(row)
            pmap = fit_anisotropic_power_map(
                dwi_data, gtab, mask)
            pmap = nib.Nifti1Image(pmap, img.affine)
            self.log_and_save_nii(pmap, pmap_file)
            meta_fname = self._get_fname(row, '_model-CSD_APM.json')
            meta = dict()
            afd.write_json(meta_fname, meta)

        return pmap_file

    def _dti_fa(self, row):
        dti_fa_file = self._get_fname(row, '_model-DTI_FA.nii.gz')
        if self.force_recompute or not op.exists(dti_fa_file):
            tf = self._dti_fit(row)
            fa = tf.fa
            self.log_and_save_nii(nib.Nifti1Image(fa, row['dwi_affine']),
                                  dti_fa_file)
            meta_fname = self._get_fname(row, '_model-DTI_FA.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dti_fa_file

    def _dti_cfa(self, row):
        dti_cfa_file = self._get_fname(row, '_model-DTI_desc-DEC_FA.nii.gz')
        if self.force_recompute or not op.exists(dti_cfa_file):
            tf = self._dti_fit(row)
            cfa = tf.color_fa
            self.log_and_save_nii(nib.Nifti1Image(cfa, row['dwi_affine']),
                                  dti_cfa_file)
            meta_fname = self._get_fname(row, '_model-DTI_desc-DEC_FA.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dti_cfa_file

    def _dti_pdd(self, row):
        dti_pdd_file = self._get_fname(row, '_model-DTI_PDD.nii.gz')
        if self.force_recompute or not op.exists(dti_pdd_file):
            tf = self._dti_fit(row)
            pdd = tf.directions.squeeze()
            # Invert the x coordinates:
            pdd[..., 0] = pdd[..., 0] * -1

            self.log_and_save_nii(nib.Nifti1Image(pdd, row['dwi_affine']),
                                  dti_pdd_file)
            meta_fname = self._get_fname(row, '_model-DTI_PDD.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dti_pdd_file

    def _dti_md(self, row):
        dti_md_file = self._get_fname(row, '_model-DTI_MD.nii.gz')
        if self.force_recompute or not op.exists(dti_md_file):
            tf = self._dti_fit(row)
            md = tf.md
            self.log_and_save_nii(nib.Nifti1Image(md, row['dwi_affine']),
                                  dti_md_file)
            meta_fname = self._get_fname(row, '_model-DTI_MD.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dti_md_file

    def _dki_fa(self, row):
        dki_fa_file = self._get_fname(row, '_model-DKI_FA.nii.gz')
        if self.force_recompute or not op.exists(dki_fa_file):
            tf = self._dki_fit(row)
            fa = tf.fa
            nib.save(nib.Nifti1Image(fa, row['dwi_affine']),
                     dki_fa_file)
            meta_fname = self._get_fname(row, '_model-DKI_FA.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dki_fa_file

    def _dki_md(self, row):
        dki_md_file = self._get_fname(row, '_model-DKI_MD.nii.gz')
        if self.force_recompute or not op.exists(dki_md_file):
            tf = self._dki_fit(row)
            md = tf.md
            nib.save(nib.Nifti1Image(md, row['dwi_affine']),
                     dki_md_file)
            meta_fname = self._get_fname(row, '_model-DKI_MD.json')
            meta = dict()
            afd.write_json(meta_fname, meta)
        return dki_md_file

    # Keep track of functions that compute scalars:
    _scalar_dict = {"dti_fa": _dti_fa,
                    "dti_md": _dti_md,
                    "dki_fa": _dki_fa,
                    "dki_md": _dki_md}

    def _get_best_scalar(self):
        for scalar in self.scalars:
            if "fa" in scalar:
                return scalar
        return self.scalars[0]

    def _reg_img(self, img, mask, row=None):
        if row is not None and row["reg_subject"] is not None:
            img = nib.load(row["reg_subject"])
        elif isinstance(img, str):
            img_l = img.lower()
            if img_l == "mni_t2":
                img = afd.read_mni_template(
                    mask=self.mask_template, weight="T2w")
            elif img_l == "mni_t1":
                img = afd.read_mni_template(
                    mask=self.mask_template, weight="T1w")
            elif img_l == "b0":
                img = nib.load(self._b0(row))
            elif img_l == "dti_fa_subject":
                img = nib.load(self._dti_fa(row))
            elif img_l == "dti_fa_template":
                img = afd.read_ukbb_fa_template(mask=self.mask_template)
            elif img_l == "power_map":
                img = nib.load(self._anisotropic_power_map(row))
            elif img_l == "subject_sls":
                img = nib.load(row['dwi_file'])
                tg = load_tractogram(self._streamlines(row),
                                     img,
                                     Space.VOX,
                                     bbox_valid_check=False)
                return img, tg.streamlines
            elif img_l == "hcp_atlas":
                atlas_fname = op.join(
                    afd.afq_home,
                    'hcp_atlas_16_bundles',
                    'Atlas_in_MNI_Space_16_bundles',
                    'whole_brain',
                    'whole_brain_MNI.trk')
                if not op.exists(atlas_fname):
                    afd.fetch_hcp_atlas_16_bundles()
                img = afd.read_mni_template(mask=self.mask_template)
                hcp_atlas = load_tractogram(
                    atlas_fname,
                    'same', bbox_valid_check=False)

                return img, hcp_atlas.streamlines
            else:
                img = nib.load(img)

        if mask:
            brain_mask_file = self._brain_mask(row)
            brain_mask = nib.load(brain_mask_file).get_fdata().astype(bool)

            masked_data = img.get_fdata()
            masked_data[~brain_mask] = 0

            img = nib.Nifti1Image(masked_data, img.affine)

        return img, None

    def _reg_prealign(self, row):
        prealign_file = self._get_fname(
            row, '_prealign_from-DWI_to-MNI_xfm.npy')
        if self.force_recompute or not op.exists(prealign_file):
            reg_subject_img, _ = self._reg_img(self.reg_subject, True, row)
            _, aff = reg.affine_registration(
                reg_subject_img.get_fdata(),
                self.reg_template_img.get_fdata(),
                reg_subject_img.affine,
                self.reg_template_img.affine)
            np.save(prealign_file, aff)
            meta_fname = self._get_fname(
                row, '_prealign_from-DWI_to-MNI_xfm.json')
            meta = dict(type="rigid")
            afd.write_json(meta_fname, meta)
        return prealign_file

    def _export_registered_b0(self, row):
        if self.use_prealign:
            b0_warped_file = self._get_fname(row, '_b0_in_MNI.nii.gz')
        else:
            b0_warped_file = self._get_fname(
                row, '_b0_in_MNI_without_prealign.nii.gz')

        if self.force_recompute or not op.exists(b0_warped_file):
            b0_file = self._b0(row)
            mean_b0 = nib.load(b0_file).get_fdata()

            if self.use_prealign:
                reg_prealign = np.load(self._reg_prealign(row))
                reg_prealign_inv = np.linalg.inv(reg_prealign)
            else:
                reg_prealign_inv = None

            mapping_file = self._mapping(row)
            mapping = reg.read_mapping(mapping_file, b0_file,
                                       self.reg_template_img,
                                       prealign=reg_prealign_inv)

            warped_b0 = mapping.transform(mean_b0)

            self.log_and_save_nii(nib.Nifti1Image(
                warped_b0, self.reg_template_img.affine), b0_warped_file)

        return b0_warped_file

    def _mapping(self, row):
        mapping_file = self._get_fname(
            row,
            '_mapping_from-DWI_to_MNI_xfm')
        meta_fname = self._get_fname(row, '_mapping_reg')
        if not self.use_prealign:
            mapping_file = mapping_file + '_without_prealign'
            meta_fname = meta_fname + '_without_prealign'
        if self.reg_algo == "slr":
            mapping_file = mapping_file + '.npy'
        else:
            mapping_file = mapping_file + '.nii.gz'
        meta_fname = meta_fname + '.json'

        if self.force_recompute or not op.exists(mapping_file):
            if self.use_prealign:
                reg_prealign = np.load(self._reg_prealign(row))
            else:
                reg_prealign = None

            reg_template_img, reg_template_sls = \
                self._reg_img(self.reg_template, False, row)
            reg_subject_img, reg_subject_sls = \
                self._reg_img(self.reg_subject, True, row)

            start_time = time()
            if self.reg_algo == "slr":
                mapping = reg.slr_registration(
                    reg_subject_sls, reg_template_sls,
                    moving_affine=reg_subject_img.affine,
                    moving_shape=reg_subject_img.shape,
                    static_affine=reg_template_img.affine,
                    static_shape=reg_template_img.shape)
            else:
                _, mapping = reg.syn_registration(
                    reg_subject_img.get_fdata(),
                    reg_template_img.get_fdata(),
                    moving_affine=reg_subject_img.affine,
                    static_affine=reg_template_img.affine,
                    prealign=reg_prealign)

            if self.use_prealign:
                mapping.codomain_world2grid = np.linalg.inv(reg_prealign)

            reg.write_mapping(mapping, mapping_file)
            meta = dict(type="displacementfield")
            afd.write_json(meta_fname, meta)
            row['timing']['Registration'] =\
                row['timing']['Registration'] + time() - start_time

        return mapping_file

    def _streamlines(self, row):
        if self.custom_tractography_bids_filters is not None:
            return row["custom_tract"]

        odf_model = self.tracking_params["odf_model"]

        streamlines_file = self._get_fname(
            row,
            '_tractography.trk',
            include_track=True)

        if self.force_recompute or not op.exists(streamlines_file):
            if odf_model == "DTI":
                params_file = self._dti(row)
            elif odf_model == "CSD":
                params_file = self._csd(row)
            elif odf_model == "MSMT":
                params_file = self._csd(row, msmt=True)
            elif odf_model == "DKI":
                params_file = self._dki(row)

            tracking_params = self.tracking_params.copy()
            if check_mask_methods(self.tracking_params['seed_mask']):
                tracking_params['seed_mask'], _, seed_mask_desc =\
                    self.tracking_params['seed_mask'].get_mask(self, row)
            else:
                seed_mask_desc = dict(source=tracking_params['seed_mask'])

            if check_mask_methods(self.tracking_params['stop_mask']):
                tracking_params['stop_mask'], _, stop_mask_desc =\
                    self.tracking_params['stop_mask'].get_mask(self, row)
            else:
                stop_mask_desc = dict(source=tracking_params['stop_mask'])

            start_time = time()
            sft = aft.track(params_file, **tracking_params)
            sft.to_vox()
            meta_directions = {"det": "deterministic",
                               "prob": "probabilistic"}

            meta = dict(
                TractographyClass="local",
                TractographyMethod=meta_directions[
                    self.tracking_params["directions"]],
                Count=len(sft.streamlines),
                Seeding=dict(
                    ROI=seed_mask_desc,
                    n_seeds=self.tracking_params["n_seeds"],
                    random_seeds=self.tracking_params["random_seeds"]),
                Constraints=dict(ROI=stop_mask_desc),
                Parameters=dict(
                    Units="mm",
                    StepSize=self.tracking_params["step_size"],
                    MinimumLength=self.tracking_params["min_length"],
                    MaximumLength=self.tracking_params["max_length"],
                    Unidirectional=False))

            meta_fname = self._get_fname(
                row,
                '_tractography.json',
                include_track=True)
            afd.write_json(meta_fname, meta)
            self.log_and_save_trk(sft, streamlines_file)
            row['timing']['Tractography'] =\
                row['timing']['Tractography'] + time() - start_time

        return streamlines_file

    def _segment(self, row):
        # We pass `clean_params` here, but do not use it, so we have the
        # same signature as `_clean_bundles`.
        bundles_file = self._get_fname(
            row,
            '_tractography.trk',
            include_track=True,
            include_seg=True)

        if self.force_recompute or not op.exists(bundles_file):
            streamlines_file = self._streamlines(row)

            img = nib.load(row['dwi_file'])
            tg = load_tractogram(streamlines_file, img, Space.VOX)
            if self.use_prealign:
                reg_prealign = np.load(self._reg_prealign(row))
            else:
                reg_prealign = None

            start_time = time()
            segmentation = seg.Segmentation(**self.segmentation_params)
            bundles = segmentation.segment(self.bundle_dict,
                                           tg,
                                           row['dwi_file'],
                                           row['bval_file'],
                                           row['bvec_file'],
                                           reg_template=self.reg_template_img,
                                           mapping=self._mapping(row),
                                           reg_prealign=reg_prealign)

            if self.segmentation_params['return_idx']:
                idx = {bundle: bundles[bundle]['idx'].tolist()
                       for bundle in self.bundle_dict}
                afd.write_json(bundles_file.split('.')[0] + '_idx.json',
                               idx)
                bundles = {bundle: bundles[bundle]['sl']
                           for bundle in self.bundle_dict}

            tgram = aus.bundles_to_tgram(bundles, self.bundle_dict, img)
            self.log_and_save_trk(tgram, bundles_file)
            meta = dict(source=streamlines_file,
                        Parameters=self.segmentation_params)
            meta_fname = bundles_file.split('.')[0] + '.json'
            afd.write_json(meta_fname, meta)
            row['timing']['Segmentation'] =\
                row['timing']['Segmentation'] + time() - start_time

        return bundles_file

    def _clean_bundles(self, row):
        clean_bundles_file = self._get_fname(
            row,
            '-clean_tractography.trk',
            include_track=True,
            include_seg=True)

        if self.force_recompute or not op.exists(clean_bundles_file):
            bundles_file = self._segment(row)

            sft = load_tractogram(bundles_file,
                                  row['dwi_img'],
                                  Space.VOX)

            start_time = time()
            tgram = nib.streamlines.Tractogram([], {'bundle': []})
            if self.clean_params['return_idx']:
                return_idx = {}

            for b in self.bundle_dict.keys():
                if b != "whole_brain":
                    idx = np.where(sft.data_per_streamline['bundle']
                                   == self.bundle_dict[b]['uid'])[0]
                    this_tg = StatefulTractogram(
                        sft.streamlines[idx],
                        row['dwi_img'],
                        Space.VOX)
                    this_tg = seg.clean_bundle(this_tg, **self.clean_params)
                    if self.clean_params['return_idx']:
                        this_tg, this_idx = this_tg
                        idx_file = bundles_file.split('.')[0] + '_idx.json'
                        with open(idx_file) as ff:
                            bundle_idx = json.load(ff)[b]
                        return_idx[b] = \
                            np.array(bundle_idx)[this_idx].tolist()
                    this_tgram = nib.streamlines.Tractogram(
                        this_tg.streamlines,
                        data_per_streamline={
                            'bundle': (len(this_tg)
                                       * [self.bundle_dict[b]['uid']])},
                            affine_to_rasmm=row['dwi_affine'])
                    tgram = aus.add_bundles(tgram, this_tgram)
            self.log_and_save_trk(
                StatefulTractogram(
                    tgram.streamlines,
                    sft,
                    Space.VOX,
                    data_per_streamline=tgram.data_per_streamline),
                clean_bundles_file)

            seg_args = get_default_args(seg.clean_bundle)
            for k in seg_args:
                if callable(seg_args[k]):
                    seg_args[k] = seg_args[k].__name__

            meta = dict(source=bundles_file,
                        Parameters=seg_args)
            meta_fname = clean_bundles_file.split('.')[0] + '.json'
            afd.write_json(meta_fname, meta)

            if self.clean_params['return_idx']:
                afd.write_json(clean_bundles_file.split('.')[0] + '_idx.json',
                               return_idx)
            row['timing']['Cleaning'] =\
                row['timing']['Cleaning'] + time() - start_time

        return clean_bundles_file

    def _tract_profiles(self, row):
        profiles_file = self._get_fname(row, '_profiles.csv')
        if self.force_recompute or not op.exists(profiles_file):
            bundles_file = self._clean_bundles(row)
            keys = []
            vals = []
            for k in self.bundle_dict.keys():
                if k != "whole_brain":
                    keys.append(self.bundle_dict[k]['uid'])
                    vals.append(k)
            reverse_dict = dict(zip(keys, vals))

            bundle_names = []
            node_numbers = []
            profiles = np.empty((len(self.scalars), 0)).tolist()
            this_profile = np.zeros((len(self.scalars), 100))

            trk = nib.streamlines.load(bundles_file)
            for b in np.unique(
                    trk.tractogram.data_per_streamline['bundle']):
                idx = np.where(
                    trk.tractogram.data_per_streamline['bundle'] == b)[0]
                this_sl = trk.streamlines[idx]
                bundle_name = reverse_dict[b]
                for ii, scalar in enumerate(self.scalars):
                    scalar_file = self._scalar_dict[scalar](self, row)
                    scalar_data = nib.load(scalar_file).get_fdata()
                    this_profile[ii] = afq_profile(
                        scalar_data,
                        this_sl,
                        row["dwi_affine"],
                        weights=gaussian_weights(this_sl))
                    profiles[ii].extend(list(this_profile[ii]))
                nodes = list(np.arange(this_profile[0].shape[0]))
                bundle_names.extend([bundle_name] * len(nodes))
                node_numbers.extend(nodes)

            profile_dict = dict()
            profile_dict["tractID"] = bundle_names
            profile_dict["nodeID"] = node_numbers
            for ii, scalar in enumerate(self.scalars):
                profile_dict[scalar] = profiles[ii]

            profile_dframe = pd.DataFrame(profile_dict)
            profile_dframe.to_csv(profiles_file)
            meta = dict(source=bundles_file,
                        parameters=get_default_args(afq_profile))
            meta_fname = profiles_file.split('.')[0] + '.json'
            afd.write_json(meta_fname, meta)

        return profiles_file

    def _template_xform(self, row):
        template_xform_file = self._get_fname(row, "_template_xform.nii.gz")
        if self.force_recompute or not op.exists(template_xform_file):
            if self.use_prealign:
                reg_prealign_inv = np.linalg.inv(
                    np.load(self._reg_prealign(row))
                )
            else:
                reg_prealign_inv = None

            mapping = reg.read_mapping(self._mapping(row),
                                       row['dwi_file'],
                                       self.reg_template_img,
                                       prealign=reg_prealign_inv)

            template_xform = mapping.transform_inverse(
                self.reg_template_img.get_fdata())
            self.log_and_save_nii(nib.Nifti1Image(template_xform,
                                                  row['dwi_affine']),
                                  template_xform_file)

        return template_xform_file

    def _export_rois(self, row):
        if self.use_prealign:
            reg_prealign = np.load(self._reg_prealign(row))
            reg_prealign_inv = np.linalg.inv(reg_prealign)
        else:
            reg_prealign_inv = None

        mapping = reg.read_mapping(self._mapping(row),
                                   row['dwi_file'],
                                   self.reg_template_img,
                                   prealign=reg_prealign_inv)

        rois_dir = op.join(row['results_dir'], 'ROIs')
        os.makedirs(rois_dir, exist_ok=True)
        roi_files = {}
        for bundle in self.bundle_dict:
            roi_files[bundle] = []
            for ii, roi in enumerate(self.bundle_dict[bundle]['ROIs']):

                if self.bundle_dict[bundle]['rules'][ii]:
                    inclusion = 'include'
                else:
                    inclusion = 'exclude'

                fname = op.split(
                    self._get_fname(
                        row,
                        f'_desc-ROI-{bundle}-{ii + 1}-{inclusion}.nii.gz'))

                fname = op.join(rois_dir, fname[1])
                if not op.exists(fname):

                    warped_roi = auv.patch_up_roi(
                        (mapping.transform_inverse(
                            roi.get_fdata(),
                            interpolation='linear')) > 0,
                        bundle_name=bundle).astype(int)

                    # Cast to float32, so that it can be read in by MI-Brain:
                    self.log_and_save_nii(
                        nib.Nifti1Image(warped_roi.astype(np.float32),
                                        row['dwi_affine']),
                        fname)
                    meta = dict()
                    meta_fname = fname.split('.')[0] + '.json'
                    afd.write_json(meta_fname, meta)
                roi_files[bundle].append(fname)
        return roi_files

    def _export_bundles(self, row):
        for func, folder in zip([self._clean_bundles, self._segment],
                                ['clean_bundles', 'bundles']):
            bundles_file = func(row)

            bundles_dir = op.join(row['results_dir'], folder)
            os.makedirs(bundles_dir, exist_ok=True)
            trk = nib.streamlines.load(bundles_file)
            tg = trk.tractogram
            streamlines = tg.streamlines
            for bundle in self.bundle_dict:
                if bundle != "whole_brain":
                    uid = self.bundle_dict[bundle]['uid']
                    idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
                    this_sl = dtu.transform_tracking_output(
                        streamlines[idx],
                        np.linalg.inv(row['dwi_affine']))

                    this_tgm = StatefulTractogram(this_sl, row['dwi_img'],
                                                  Space.VOX)
                    fname = op.split(
                        self._get_fname(
                            row,
                            f'-{bundle}'
                            f'_tractography.trk',
                            include_track=True,
                            include_seg=True))
                    fname = op.join(bundles_dir, fname[1])
                    self.log_and_save_trk(this_tgm, fname)
                    meta = dict(source=bundles_file)
                    meta_fname = fname.split('.')[0] + '.json'
                    afd.write_json(meta_fname, meta)

    def _export_sl_counts(self, row):
        total_file = self._streamlines(row)
        total_sl = len(nib.streamlines.load(total_file).tractogram.streamlines)
        sl_counts = []
        for func, folder in zip([self._clean_bundles, self._segment],
                                ['clean_bundles', 'bundles']):
            sl_counts.append([])
            bundles_file = func(row)
            trk = nib.streamlines.load(bundles_file)
            tg = trk.tractogram
            streamlines = tg.streamlines
            for bundle in self.bundle_dict:
                if bundle != "whole_brain":
                    uid = self.bundle_dict[bundle]['uid']
                    idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
                    this_sl = dtu.transform_tracking_output(
                        streamlines[idx],
                        np.linalg.inv(row['dwi_affine']))

                    this_tgm = StatefulTractogram(this_sl, row['dwi_img'],
                                                  Space.VOX)
                    sl_counts[-1].append(len(this_tgm.streamlines))
            sl_counts[-1].append(total_sl)

        bundle_list = list(self.bundle_dict.keys())
        if 'whole_brain' in bundle_list:
            bundle_list.remove('whole_brain')
        bundle_list.append("Total")

        sl_counts = pd.DataFrame(
            data=dict(bundle=bundle_list,
                      n_streamlines=sl_counts[1],
                      n_streamlines_cleaned=sl_counts[0]))

        sl_count_file = self._get_fname(
            row,
            '_sl_count.csv',
            include_track=True,
            include_seg=True)

        sl_counts.to_csv(sl_count_file)

    def _viz_prepare_vol(self, row, vol, xform, mapping):
        if vol in self.scalars:
            vol = nib.load(
                self._scalar_dict[vol](self, row)).get_fdata()
        if isinstance(vol, str):
            vol = nib.load(vol).get_fdata()
        if xform:
            vol = mapping.transform_inverse(vol)
        return vol

    def _viz_prepare_vols(self, row,
                          volume,
                          xform_volume,
                          color_by_volume,
                          xform_color_by_volume):
        if volume is None:
            volume = self._b0(row)
        if color_by_volume is None:
            color_by_volume = self._get_best_scalar()

        if xform_volume or xform_color_by_volume:
            if self.use_prealign:
                reg_prealign = np.load(self._reg_prealign(row))
                reg_prealign_inv = np.linalg.inv(reg_prealign)
            else:
                reg_prealign_inv = None

            mapping = reg.read_mapping(self._mapping(row),
                                       row['dwi_file'],
                                       self.reg_template_img,
                                       prealign=reg_prealign_inv)
        else:
            mapping = None

        volume = self._viz_prepare_vol(
            row,
            volume,
            xform_volume,
            mapping)

        color_by_volume = self._viz_prepare_vol(
            row,
            color_by_volume,
            xform_color_by_volume,
            mapping)

        return volume, color_by_volume

    def _viz_bundles(self, row,
                     export=False,
                     inline=False,
                     interactive=False,
                     volume=None,
                     xform_volume=False,
                     color_by_volume=None,
                     xform_color_by_volume=False,
                     n_points=40):
        bundles_file = self._clean_bundles(row)

        start_time = time()
        volume, color_by_volume = self._viz_prepare_vols(
            row,
            volume=volume,
            xform_volume=xform_volume,
            color_by_volume=color_by_volume,
            xform_color_by_volume=xform_color_by_volume
        )

        figure = self.viz.visualize_volume(volume,
                                           interact=False,
                                           inline=False)

        figure = self.viz.visualize_bundles(bundles_file,
                                            color_by_volume=color_by_volume,
                                            bundle_dict=self.bundle_dict,
                                            n_points=n_points,
                                            interact=interactive,
                                            inline=inline,
                                            figure=figure)

        if export:
            if "no_gif" not in self.viz.backend:
                fname = self._get_fname(
                    row,
                    '_viz.gif',
                    include_track=True,
                    include_seg=True)

                self.viz.create_gif(figure, fname)
            if "plotly" in self.viz.backend:
                fname = self._get_fname(
                    row,
                    '_viz.html',
                    include_track=True,
                    include_seg=True)

                figure.write_html(fname)
        row['timing']['Visualization'] =\
            row['timing']['Visualization'] + time() - start_time
        return figure

    def _viz_ROIs(self, row,
                  bundle_names=None,
                  export=False,
                  inline=False,
                  interactive=False,
                  volume=None,
                  xform_volume=False,
                  color_by_volume=None,
                  xform_color_by_volume=False,
                  n_points=40):
        bundles_file = self._clean_bundles(row)

        start_time = time()
        volume, color_by_volume = self._viz_prepare_vols(
            row,
            volume=volume,
            xform_volume=xform_volume,
            color_by_volume=color_by_volume,
            xform_color_by_volume=xform_color_by_volume
        )

        if bundle_names is None:
            bundle_names = self.bundle_dict.keys()

        for bundle_name in bundle_names:
            self.logger.info(f"Generating {bundle_name} visualization...")
            uid = self.bundle_dict[bundle_name]['uid']
            figure = self.viz.visualize_volume(volume,
                                               interact=False,
                                               inline=False)
            try:
                figure = self.viz.visualize_bundles(
                    bundles_file,
                    color_by_volume=color_by_volume,
                    bundle_dict=self.bundle_dict,
                    bundle=uid,
                    n_points=n_points,
                    interact=False,
                    inline=False,
                    figure=figure)
            except ValueError:
                self.logger.info("No streamlines found to visualize for "
                                 + bundle_name)

            roi_files = self._export_rois(row)
            for i, roi in enumerate(roi_files[bundle_name]):
                if i == len(roi_files[bundle_name]) - 1:  # show on last ROI
                    figure = self.viz.visualize_roi(
                        roi,
                        name=f"{bundle_name} ROI {i}",
                        inline=inline,
                        interact=interactive,
                        opacity=0.75,
                        figure=figure)
                else:
                    figure = self.viz.visualize_roi(
                        roi,
                        name=f"{bundle_name} ROI {i}",
                        inline=False,
                        interact=False,
                        figure=figure)

            if export:
                roi_dir = op.join(row['results_dir'], 'viz_bundles')
                os.makedirs(roi_dir, exist_ok=True)
                if "no_gif" not in self.viz.backend:
                    fname = op.split(
                        self._get_fname(
                            row,
                            f'_{bundle_name}'
                            f'_viz.gif',
                            include_track=True,
                            include_seg=True))

                    fname = op.join(roi_dir, fname[1])
                    self.viz.create_gif(figure, fname)
                if "plotly" in self.viz.backend:
                    roi_dir = op.join(row['results_dir'], 'viz_bundles')
                    os.makedirs(roi_dir, exist_ok=True)
                    fname = op.split(
                        self._get_fname(
                            row,
                            f'_{bundle_name}'
                            f'_viz.html',
                            include_track=True,
                            include_seg=True))

                    fname = op.join(roi_dir, fname[1])
                    figure.write_html(fname)
        row['timing']['Visualization'] =\
            row['timing']['Visualization'] + time() - start_time
        return figure

    def _plot_tract_profiles(self, row):
        start_time = time()
        fnames = []
        for scalar in self.scalars:
            fname = self._get_fname(
                row,
                f'_{scalar}_profile_plots',
                include_track=True,
                include_seg=True)

            visualize_tract_profiles(self._tract_profiles(row),
                                     scalar=scalar,
                                     file_name=fname,
                                     n_boot=100)
            fnames.append(fname)
        row['timing']['Visualization'] =\
            row['timing']['Visualization'] + time() - start_time

        return fnames

    def _export_timing(self, row):
        df = pd.DataFrame.from_dict(
            row["timing"],
            'index',
            columns=['Time (s)'])
        timing_fname = self._get_fname(row, "_desc-timing.csv", True, True)
        df.to_csv(timing_fname, index=True, index_label='step')

    def _get_affine(self, fname):
        return nib.load(fname).affine

    def _get_fname(self, row, suffix, include_track=False, include_seg=False):
        split_fdwi = op.split(row['dwi_file'])
        fname = op.join(row['results_dir'], split_fdwi[1].split('.')[0])

        if include_track:
            odf_model = self.tracking_params['odf_model']
            directions = self.tracking_params['directions']
            fname = fname + (
                f'_space-RASMM_model-{odf_model}'
                f'_desc-{directions}'
            )
        if include_seg:
            seg_algo = self.segmentation_params['seg_algo']
            fname = fname + f'-{seg_algo}'

        return fname + suffix

    def set_gtab(self, b0_threshold):
        self.data_frame['gtab'] = self.data_frame.apply(
            lambda x: dpg.gradient_table(x['bval_file'], x['bvec_file'],
                                         b0_threshold=b0_threshold),
            axis=1)

    def get_gtab(self):
        return self.data_frame['gtab']

    gtab = property(get_gtab, set_gtab)

    def set_dwi_affine(self):
        self.data_frame['dwi_affine'] =\
            self.data_frame['dwi_file'].apply(self._get_affine)

    def get_dwi_affine(self):
        return self.data_frame['dwi_affine']

    dwi_affine = property(get_dwi_affine, set_dwi_affine)

    def set_dwi_img(self):
        self.data_frame['dwi_img'] =\
            self.data_frame['dwi_file'].apply(nib.load)

    def get_dwi_img(self):
        return self.data_frame['dwi_img']

    dwi_img = property(get_dwi_img, set_dwi_img)

    def __getitem__(self, k):
        return self.data_frame.__getitem__(k)

    def set_b0(self):
        if 'b0_file' not in self.data_frame.columns:
            self.data_frame['b0_file'] =\
                self.data_frame.apply(self._b0,
                                      axis=1)

    def get_b0(self):
        self.set_b0()
        return self.data_frame['b0_file']

    b0 = property(get_b0, set_b0)

    def set_brain_mask(self):
        if 'brain_mask_file' not in self.data_frame.columns:
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(self._brain_mask,
                                      axis=1)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dti(self):
        if 'dti_params_file' not in self.data_frame.columns:
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(self._dti,
                                      axis=1)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_dti_fa(self):
        if 'dti_fa_file' not in self.data_frame.columns:
            self.data_frame['dti_fa_file'] =\
                self.data_frame.apply(self._dti_fa,
                                      axis=1)

    def get_dti_fa(self):
        self.set_dti_fa()
        return self.data_frame['dti_fa_file']

    dti_fa = property(get_dti_fa, set_dti_fa)

    def set_dti_cfa(self):
        if 'dti_cfa_file' not in self.data_frame.columns:
            self.data_frame['dti_cfa_file'] =\
                self.data_frame.apply(self._dti_cfa,
                                      axis=1)

    def get_dti_cfa(self):
        self.set_dti_cfa()
        return self.data_frame['dti_cfa_file']

    dti_cfa = property(get_dti_cfa, set_dti_cfa)

    def set_dti_pdd(self):
        if 'dti_pdd_file' not in self.data_frame.columns:
            self.data_frame['dti_pdd_file'] =\
                self.data_frame.apply(self._dti_pdd,
                                      axis=1)

    def get_dti_pdd(self):
        self.set_dti_pdd()
        return self.data_frame['dti_pdd_file']

    dti_pdd = property(get_dti_pdd, set_dti_pdd)

    def set_dti_md(self):
        if 'dti_md_file' not in self.data_frame.columns:
            self.data_frame['dti_md_file'] =\
                self.data_frame.apply(self._dti_md,
                                      axis=1)

    def get_dti_md(self):
        self.set_dti_md()
        return self.data_frame['dti_md_file']

    dti_md = property(get_dti_md, set_dti_md)

    def set_dki(self):
        if 'dki_params_file' not in self.data_frame.columns:
            self.data_frame['dki_params_file'] =\
                self.data_frame.apply(self._dki,
                                      axis=1)

    def get_dki(self):
        self.set_dki()
        return self.data_frame['dki_params_file']

    dki = property(get_dki, set_dki)

    def set_dki_fa(self):
        if 'dki_fa_file' not in self.data_frame.columns:
            self.data_frame['dki_fa_file'] =\
                self.data_frame.apply(self._dki_fa,
                                      axis=1)

    def get_dki_fa(self):
        self.set_dki_fa()
        return self.data_frame['dki_fa_file']

    dki_fa = property(get_dki_fa, set_dki_fa)

    def set_dki_md(self):
        if 'dki_md_file' not in self.data_frame.columns:
            self.data_frame['dki_md_file'] =\
                self.data_frame.apply(self._dki_md,
                                      axis=1)

    def get_dki_md(self):
        self.set_dki_md()
        return self.data_frame['dki_md_file']

    dki_md = property(get_dki_md, set_dki_md)

    def set_mapping(self):
        if 'mapping' not in self.data_frame.columns:
            self.data_frame['mapping'] =\
                self.data_frame.apply(self._mapping,
                                      axis=1)

    def get_mapping(self):
        self.set_mapping()
        return self.data_frame['mapping']

    mapping = property(get_mapping, set_mapping)

    def set_streamlines(self):
        if 'streamlines_file' not in self.data_frame.columns:
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(self._streamlines, axis=1)

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']

    streamlines = property(get_streamlines, set_streamlines)

    def set_bundles(self):
        if 'bundles_file' not in self.data_frame.columns:
            self.data_frame['bundles_file'] =\
                self.data_frame.apply(self._segment, axis=1)

    def get_bundles(self):
        self.set_bundles()
        return self.data_frame['bundles_file']

    bundles = property(get_bundles, set_bundles)

    def set_clean_bundles(self):
        if 'clean_bundles_file' not in self.data_frame.columns:
            if self.seg_algo == "reco":
                self.set_bundles()
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame['bundles_file']
            else:
                self.data_frame['clean_bundles_file'] =\
                    self.data_frame.apply(self._clean_bundles, axis=1)

    def get_clean_bundles(self):
        self.set_clean_bundles()
        return self.data_frame['clean_bundles_file']

    clean_bundles = property(get_clean_bundles, set_clean_bundles)

    def set_tract_profiles(self):
        if 'tract_profiles_file' not in self.data_frame.columns:
            self.data_frame['tract_profiles_file'] =\
                self.data_frame.apply(self._tract_profiles, axis=1)

    def get_tract_profiles(self):
        self.set_tract_profiles()
        return self.data_frame['tract_profiles_file']

    tract_profiles = property(get_tract_profiles, set_tract_profiles)

    def set_template_xform(self):
        if 'template_xform_file' not in self.data_frame.columns:
            self.data_frame['template_xform_file'] = \
                self.data_frame.apply(self._template_xform,
                                      axis=1)

    def get_template_xform(self):
        self.set_template_xform()
        return self.data_frame['template_xform_file']

    template_xform = property(get_template_xform, set_template_xform)

    def export_rois(self):
        return self.data_frame.apply(self._export_rois, axis=1)

    def export_bundles(self):
        self.data_frame.apply(self._export_bundles, axis=1)

    def export_sl_counts(self):
        self.data_frame.apply(self._export_sl_counts, axis=1)

    def viz_bundles(self,
                    export=False,
                    volume=None,
                    xform_volume=False,
                    color_by_volume=None,
                    xform_color_by_volume=False,
                    n_points=40,
                    inline=False,
                    interactive=False):
        return self.data_frame.apply(
            self._viz_bundles, axis=1,
            export=export,
            volume=volume,
            xform_volume=xform_volume,
            color_by_volume=color_by_volume,
            xform_color_by_volume=xform_color_by_volume,
            n_points=n_points,
            inline=inline,
            interactive=interactive)

    def viz_ROIs(self,
                 bundle_names=None,
                 export=False,
                 volume=None,
                 xform_volume=False,
                 color_by_volume=None,
                 xform_color_by_volume=False,
                 n_points=40,
                 inline=False,
                 interactive=False):
        return self.data_frame.apply(
            self._viz_ROIs,
            axis=1,
            bundle_names=bundle_names,
            export=export,
            inline=inline,
            interactive=interactive,
            volume=volume,
            xform_volume=xform_volume,
            color_by_volume=color_by_volume,
            xform_color_by_volume=xform_color_by_volume,
            n_points=n_points)

    def plot_tract_profiles(self):
        if 'tract_profiles_viz' not in self.data_frame.columns:
            self.data_frame['tract_profiles_viz'] =\
                self.data_frame.apply(self._plot_tract_profiles, axis=1)

    def export_registered_b0(self):
        self.data_frame.apply(self._export_registered_b0, axis=1)

    def combine_profiles(self):
        return combine_list_of_profiles(
            self.tract_profiles,
            op.join(self.afq_path, 'tract_profiles.csv'))

    def export_timing(self):
        self.data_frame.apply(self._export_timing, axis=1)

    def export_all(self):
        """ Exports all the possible outputs"""
        self.export_registered_b0()
        self.get_template_xform()
        self.export_bundles()
        self.export_sl_counts()
        self.get_tract_profiles()
        self.viz_bundles(export=True)
        if self.seg_algo == "afq":
            self.viz_ROIs(export=True)
            self.export_rois()
        if len(self.tract_profiles) > 1:
            self.combine_profiles()
        self.export_timing()

    def upload_to_s3(self, s3fs, remote_path):
        """ Upload entire AFQ derivatives folder to S3"""
        s3fs.put(self.afq_path, remote_path, recursive=True)


def download_and_combine_afq_profiles(out_file, bucket, study_s3_prefix,
                                      upload=None, session=None):
    """
    Download and combine tract profiles from different subjects / sessions
    on an s3 bucket into one CSV.
    Parameters
    ----------
    outfile : filename
        Filename for the combined output CSV.
    bucket : str
        The S3 bucket that contains the study data.
    study_s3_prefix : str
        The S3 prefix common to all of the study objects on S3.
    upload : s3fs, optional
        If not None,
        Upload the combined CSV after combining using the s3fs object.
        Uploads to bucket/study_s3_prefix/derivatives/afq
        Defaut: None
    session : str, optional
        Session to get CSVs from. If None, all sessions are used.
        Default: None
    Returns
    -------
    Ouput CSV's pandas dataframe.
    """
    with nib.tmpdirs.InTemporaryDirectory() as t_dir:
        remote_study = afd.S3BIDSStudy(
            "get_profiles",
            bucket,
            study_s3_prefix,
            subjects="all",
            anon=False)
        remote_study.download(
            t_dir,
            include_modality_agnostic=False,
            include_derivs="afq",
            include_derivs_dataset_description=False,
            suffix="profiles.csv")
        temp_study = BIDSLayout(t_dir, validate=False)
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
        df = combine_list_of_profiles(profiles, out_file)
    if upload is not None:
        upload.put(
            out_file,
            os.join(
                bucket,
                study_s3_prefix,
                "derivatives/afq",
                "combined_tract_profiles.csv"
            ))
    return df


def combine_list_of_profiles(profile_fnames, out_file):
    """
    Combine tract profiles from different subjects / sessions
    into one CSV.
    Parameters
    ----------
    profile_fnames : list of str
        List of csv filenames.
    outfile : filename
        Filename for the combined output CSV.
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

    df = pd.concat(dfs)
    os.makedirs(op.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index=False)
    return df
