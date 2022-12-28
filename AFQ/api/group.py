# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
import warnings
import tempfile

from AFQ.definitions.mapping import SynMap
warnings.simplefilter(action='ignore', category=FutureWarning)  # noqa

import logging
import AFQ.data.s3bids as afs
from AFQ.api.participant import ParticipantAFQ
from AFQ.api.utils import (
    check_attribute, AFQclass_doc,
    export_all_helper, valid_exports_string)
import AFQ.utils.streamlines as aus

from dipy.utils.parallel import paramap
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy.tracking.streamlinespeed as dps
import dipy.tracking.streamline as dts
from dipy.io.streamline import save_tractogram

from AFQ.version import version as pyafq_version
from AFQ.viz.utils import trim
import pandas as pd
import numpy as np
import os
import os.path as op
from tqdm import tqdm
import json
import s3fs
from time import time
import nibabel as nib
from PIL import Image

from bids.layout import BIDSLayout, BIDSLayoutIndexer
try:
    import afqbrowser as afqb
    using_afqb = True
except ImportError:
    using_afqb = False


__all__ = ["GroupAFQ"]


logger = logging.getLogger('AFQ')
logger.setLevel(logging.INFO)


# get rid of unnecessary columns in df
def clean_pandas_df(df):
    df = df.reset_index(drop=True)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


class GroupAFQ(object):
    f"""{AFQclass_doc}"""

    def __init__(self,
                 bids_path,
                 bids_filters={"suffix": "dwi"},
                 preproc_pipeline="all",
                 participant_labels=None,
                 output_dir=None,
                 parallel_params={"engine": "serial"},
                 bids_layout_kwargs={},
                 **kwargs):
        '''
        Initialize a GroupAFQ object from a BIDS dataset.

        Parameters
        ----------
        bids_path : str
            The path to preprocessed diffusion data organized in a BIDS
            dataset. This should contain a BIDS derivative dataset with
            preprocessed dwi/bvals/bvecs.
        bids_filters : dict
            Filter to pass to bids_layout.get when finding DWI files.
            Default: {"suffix": "dwi"}
        preproc_pipeline : str, optional.
            The name of the pipeline used to preprocess the DWI data.
            Default: "all".
        participant_labels : list or None, optional
            List of participant labels (subject IDs) to perform
            processing on. If None, all subjects are used.
            Default: None
        output_dir : str or None, optional
            Path to output directory. If None, outputs are put
            in a AFQ pipeline folder in the derivatives folder of
            the BIDS directory. pyAFQ will use existing derivatives
            from the output directory if they exist, instead of recalculating
            them (this means you need to clear the output folder if you want
            to recalculate a derivative).
            Default: None
        parallel_params : dict, optional
            Parameters to pass to paramap in AFQ.utils.parallel,
            to parallelize computations across subjects and sessions.
            Set "n_jobs" to -1 to automatically parallelize as
            the number of cpus. Here is an example for how to do
            multiprocessing with 4 cpus:
            {"n_jobs": 4, "engine": "joblib", "backend": "loky"}
            Default: {"engine": "serial"}
        bids_layout_kwargs: dict, optional
            Additional arguments to give to BIDSLayout from pybids.
            For large datasets, try:
            {"validate": False, "index_metadata": False}
            Default: {}
        kwargs : additional optional parameters
            You can set additional parameters for any step
            of the process. See :ref:`usage/kwargs` for more details.

        Examples
        --------
        api.GroupAFQ(my_path, csd_sh_order=4)
        api.GroupAFQ(
            my_path,
            reg_template_spec="mni_t2", reg_subject_spec="b0")
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
        # preproc_pipeline typechecking handled by pyBIDS
        if participant_labels is not None\
                and not isinstance(participant_labels, list):
            raise TypeError(
                "participant_labels must be either a list or None")
        if output_dir is not None\
                and not isinstance(output_dir, str):
            raise TypeError(
                "output_dir must be either a str or None")
        if not isinstance(parallel_params, dict):
            raise TypeError("parallel_params must be a dict")
        if not isinstance(bids_layout_kwargs, dict):
            raise TypeError("bids_layout_kwargs must be a dict")

        self.logger = logger

        self.parallel_params = parallel_params
        self.wf_dict = {}

        # validate input and fail early
        if not op.exists(bids_path):
            raise ValueError(f'Unable to locate BIDS dataset in: {bids_path}')

        # This is where all the outputs will go:
        if output_dir is None:
            self.afq_path = op.join(bids_path, 'derivatives', 'afq')
            self.afqb_path = op.join(bids_path, 'derivatives', 'afq_browser')
        else:
            self.afq_path = output_dir
            self.afqb_path = op.join(output_dir, 'afq_browser')

        # Create it as needed:
        os.makedirs(self.afq_path, exist_ok=True)

        bids_indexer = BIDSLayoutIndexer(**bids_layout_kwargs)
        bids_layout = BIDSLayout(
            bids_path, derivatives=True, indexer=bids_indexer)
        bids_description = bids_layout.description

        # check that any files exist in the derivatives folder,
        # not including the dataset_description.json files
        # the second check may be particularly useful in checking
        # that the derivatives folder is well-defined
        if len(bids_layout.get())\
                - len(bids_layout.get(extension="json")) < 1:
            raise ValueError(
                f"No non-json files recognized by pyBIDS in {bids_path}")
        if len(bids_layout.get(scope=preproc_pipeline))\
                - len(bids_layout.get(
                    scope=preproc_pipeline,
                    extension="json")) < 1:
            raise ValueError((
                f"No non-json files recognized by "
                f"pyBIDS in the pipeline: {preproc_pipeline}"))

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

        if participant_labels is not None:
            filtered_subjects = []
            subjects_found_printed = False
            for subjectID in participant_labels:
                subjectID = str(subjectID)
                if subjectID not in self.subjects:
                    self.logger.warning((
                        f"Subject {subjectID} specified in "
                        f"`participant_labels` but not found "
                        f"in BIDS derivatives folders"))
                    if not subjects_found_printed:
                        subjects_found_printed = True
                        self.logger.warning((
                            f"Only these subjects found in BIDS "
                            f"derivatives folders: {self.subjects}"))
                else:
                    filtered_subjects.append(subjectID)
            self.subjects = filtered_subjects
            if not len(self.subjects):
                raise ValueError(
                    "No subjects specified in `participant_labels` "
                    + " found in BIDS derivatives folders."
                    + " See above warnings.")

        sessions = bids_layout.get(return_type='id', target='session')
        self.sessions = sessions if len(sessions) else [None]

        # do not bother to parallelize if less than 2 subject-sessions
        if len(self.sessions) * len(self.subjects) < 2:
            self.parallel_params["engine"] = "serial"

        # do not parallelize segmentation if parallelizing across
        # subject-sessions
        if self.parallel_params["engine"] != "serial":
            if "segmentation_params" not in kwargs:
                kwargs["segmentation_params"] = {}
            if "parallel_segmentation" not in kwargs["segmentation_params"]:
                kwargs["segmentation_params"]["parallel_segmentation"] = {}
            kwargs["segmentation_params"]["parallel_segmentation"]["engine"] =\
                "serial"

        self.valid_sub_list = []
        self.valid_ses_list = []
        self.pAFQ_list = []
        for subject in self.subjects:
            self.wf_dict[subject] = {}
            for session in self.sessions:
                this_kwargs = kwargs.copy()
                results_dir = op.join(self.afq_path, 'sub-' + subject)

                if session is not None:
                    results_dir = op.join(results_dir, 'ses-' + session)

                dwi_bids_filters = {
                    "subject": subject,
                    "session": session,
                    "return_type": "filename",
                    "scope": preproc_pipeline,
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

                self.valid_sub_list.append(subject)
                self.valid_ses_list.append(str(session))

                this_pAFQ = ParticipantAFQ(
                    dwi_data_file,
                    bval_file, bvec_file,
                    results_dir,
                    bids_info={
                        "bids_layout": bids_layout,
                        "subject": subject,
                        "session": session},
                    **this_kwargs)
                self.wf_dict[subject][str(session)] = this_pAFQ.wf_dict
                self.pAFQ_list.append(this_pAFQ)

    def combine_profiles(self):
        tract_profiles_dict = self.export("profiles")
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
        _df = clean_pandas_df(_df)
        _df.to_csv(out_file, index=False)
        return _df

    def get_streamlines_json(self):
        sls_json_fname = op.abspath(op.join(
            self.afq_path, "afqb_streamlines.json"))
        if not op.exists(sls_json_fname):
            subses_info = []

            def load_next_subject():
                subses_idx = len(subses_info)
                sub = self.valid_sub_list[subses_idx]
                ses = self.valid_ses_list[subses_idx]
                this_bundles_file = self.export(
                    "clean_bundles", collapse=False)[sub][ses]
                this_mapping = self.export("mapping", collapse=False)[sub][ses]
                this_img = nib.load(self.export(
                    "dwi", collapse=False)[sub][ses])
                seg_sft = aus.SegmentedSFT.fromfile(
                    this_bundles_file,
                    this_img)
                seg_sft.sft.to_rasmm()
                subses_info.append((seg_sft, this_mapping))

            bundle_dict = self.export("bundle_dict", collapse=False)[
                self.valid_sub_list[0]][self.valid_ses_list[0]]

            sls_dict = {}
            load_next_subject()  # load first subject
            for b in bundle_dict.keys():
                if b != "whole_brain":
                    for i in range(len(self.valid_sub_list)):
                        seg_sft, mapping = subses_info[i]
                        idx = seg_sft.bundle_idxs[b]
                        # use the first subses that works
                        # otherwise try each successive subses
                        if len(idx) == 0:
                            # break if we run out of subses
                            if i + 1 >= len(self.valid_sub_list):
                                break
                            # load subses if not already loaded
                            if i + 1 >= len(subses_info):
                                load_next_subject()
                            continue
                        if len(idx) > 100:
                            idx = np.random.choice(
                                idx, size=100, replace=False)
                        these_sls = seg_sft.sft.streamlines[idx]
                        these_sls = dps.set_number_of_points(these_sls, 100)
                        tg = StatefulTractogram(
                            these_sls,
                            seg_sft.sft,
                            Space.RASMM)
                        delta = dts.values_from_volume(
                            mapping.forward,
                            tg.streamlines, np.eye(4))
                        moved_sl = dts.Streamlines(
                            [d + s for d, s in zip(delta, tg.streamlines)])
                        moved_sl = np.asarray(moved_sl)
                        median_sl = np.median(moved_sl, axis=0)
                        sls_dict[b] = {"coreFiber": median_sl.tolist()}
                        for ii, sl_idx in enumerate(idx):
                            sls_dict[b][str(sl_idx)] = moved_sl[ii].tolist()
                        break

            with open(sls_json_fname, 'w') as fp:
                json.dump(sls_dict, fp)
        return sls_json_fname

    def export(self, attr_name="help", collapse=True):
        f"""
        Export a specific output. To print a list of available outputs,
        call export without arguments.
        {valid_exports_string}

        Parameters
        ----------
        attr_name : str
            Name of the output to export. Default: "help"
        collapse : bool
            Whether to collapse session dimension if there is only 1 session.
            Default: True

        Returns
        -------
        output : dict
            The specific output as a dictionary. Keys are subjects.
            Values are dictionaries with keys of sessions
            if multiple sessions are used. Otherwise, values are
            the output.
            None if called without arguments.
        """
        section = check_attribute(attr_name)

        # iterate over subjects / sessions,
        # decide if they need to be calculated or not
        in_list = []
        to_calc_list = []
        results = {}
        for ii, subject in enumerate(self.valid_sub_list):
            if subject not in results:
                results[subject] = {}
            session = self.valid_ses_list[ii]
            wf_dict = self.wf_dict[subject][str(session)]
            if section is not None:
                wf_dict = wf_dict[section]
            if ((self.parallel_params.get("engine", False) != "serial")
                    and (hasattr(wf_dict, "efferents"))
                    and (attr_name not in wf_dict.efferents)):
                in_list.append((wf_dict))
                to_calc_list.append((subject, session))
            else:
                results[subject][session] = wf_dict[attr_name]

        # if some need to be calculated, do those in parallel
        if to_calc_list:
            par_results = paramap(
                lambda wf, attr: wf[attr], in_list,
                func_args=[attr_name],
                **self.parallel_params)

            for i, subses in enumerate(to_calc_list):
                subject, session = subses
                results[subject][session] = par_results[i]

        # If only one session, collapse session dimension
        if len(self.sessions) == 1 and collapse:
            for subject in self.valid_sub_list:
                results[subject] = results[subject][self.valid_ses_list[0]]

        return results

    def export_all(self, viz=True, afqbrowser=True, xforms=True,
                   indiv=True):
        """ Exports all the possible outputs

        Parameters
        ----------
        viz : bool
            Whether to output visualizations. This includes tract profile
            plots, a figure containing all bundles, and, if using the AFQ
            segmentation algorithm, individual bundle figures.
            Default: True
        afqbrowser : bool
            Whether to output an AFQ-Browser from this AFQ instance.
            Default: True
        xforms : bool
            Whether to output the reg_template image in subject space and,
            depending on if it is possible based on the mapping used, to
            output the b0 in template space.
            Default: True
        indiv : bool
            Whether to output individual bundles in their own files, in
            addition to the one file containing all bundles. If using
            the AFQ segmentation algorithm, individual ROIs are also
            output.
            Default: True
        """
        start_time = time()
        seg_params = self.export("segmentation_params", collapse=False)[
            self.valid_sub_list[0]][self.valid_ses_list[0]]
        seg_algo = seg_params.get("seg_algo", "AFQ")

        export_all_helper(self, seg_algo, xforms, indiv, viz)

        self.combine_profiles()
        if afqbrowser:
            self.assemble_AFQ_browser()
        self.logger.info(
            f"Time taken for export all: {str(time() - start_time)}")

    def cmd_outputs(self, cmd="rm", dependent_on=None, exceptions=[],
                    suffix=""):
        """
        Perform some command some or all outputs of pyafq.
        This is useful if you change a parameter and need
        to recalculate derivatives that depend on it.
        Some examples: cp, mv, rm .
        -r will be automtically added when necessary.

        Parameters
        ----------
        cmd : str
            Command to run on outputs. Default: 'rm'
        dependent_on : str or None
            Which derivatives to perform command on .
            If None, perform on all.
            If "track", perform on all derivatives that depend on the
            tractography.
            If "recog", perform on all derivatives that depend on the
            bundle recognition.
            Default: None
        exceptions : list of str
            Name outputs that the command should not be applied to.
            Default: []
        suffix : str
            Parts of command that are used after the filename.
            Default: ""

        Example
        -------
        # This command would move all derivatives that are
        # dependent on the tractography into 'my_other_folder'
        myafq.cmd_outputs(
            "cp",
            dependent_on="track",
            suffix="~/my_other_folder/")
        """
        for pAFQ in self.pAFQ_list:
            pAFQ.cmd_outputs(cmd, dependent_on, exceptions, suffix=suffix)

    clobber = cmd_outputs  # alias for default of cmd_outputs

    def montage(self, bundle_name, size, view, slice_pos=None):
        """
        Generate montage file(s) of a given bundle at a given angle.

        Parameters
        ----------
        bundle_name : str
            Name of bundle to visualize, should be the same as in the
            bundle dictionary.
        size : tuple of int
            The number of columns and rows for each file.
        view : str
            Which view to display. Can be one of Sagittal, Coronal, or Axial.
        slice_pos : float, or None
            If float, indicates the fractional position along the
            perpendicular axis to the slice. Currently only works with plotly.
            If None, no slice is displayed.

        Returns
        -------
        list of filenames of montage images
        """
        if view not in ["Sagittal", "Coronal", "Axial"]:
            raise ValueError(
                "View must be one of: Sagittal, Coronal, or Axial")

        tdir = tempfile.gettempdir()

        best_scalar = self.export("best_scalar", collapse=False)[
            self.valid_sub_list[0]][self.valid_ses_list[0]]
        bundle_dict = self.export("bundle_dict", collapse=False)[
            self.valid_sub_list[0]][self.valid_ses_list[0]]

        viz_backend_dict = self.export("viz_backend", collapse=False)
        b0_backend_dict = self.export("b0", collapse=False)
        dwi_affine_dict = self.export("dwi_affine", collapse=False)
        clean_bundles_dict = self.export("clean_bundles", collapse=False)
        best_scalar_dict = self.export(best_scalar, collapse=False)

        all_fnames = []
        self.logger.info("Generating Montage...")
        for ii in tqdm(range(len(self.valid_ses_list))):
            this_sub = self.valid_sub_list[ii]
            this_ses = self.valid_ses_list[ii]
            viz_backend = viz_backend_dict[this_sub][this_ses]
            b0 = b0_backend_dict[this_sub][this_ses]
            dwi_affine = dwi_affine_dict[this_sub][this_ses]
            clean_bundles = clean_bundles_dict[this_sub][this_ses]
            best_scalar = best_scalar_dict[this_sub][this_ses]

            flip_axes = [False, False, False]
            for i in range(3):
                flip_axes[i] = (dwi_affine[i, i] < 0)

            if slice_pos is not None:
                slice_kwargs = {}
                if view == "Sagittal":
                    slice_kwargs["x_pos"] = slice_pos
                    slice_kwargs["y_pos"] = None
                    slice_kwargs["z_pos"] = None
                elif view == "Coronal":
                    slice_kwargs["x_pos"] = None
                    slice_kwargs["y_pos"] = slice_pos
                    slice_kwargs["z_pos"] = None
                elif view == "Axial":
                    slice_kwargs["x_pos"] = None
                    slice_kwargs["y_pos"] = None
                    slice_kwargs["z_pos"] = slice_pos

                figure = viz_backend.visualize_volume(
                    b0,
                    flip_axes=flip_axes,
                    interact=False,
                    inline=False,
                    **slice_kwargs)
            else:
                figure = None

            figure = viz_backend.visualize_bundles(
                clean_bundles,
                shade_by_volume=best_scalar,
                bundle_dict=bundle_dict,
                flip_axes=flip_axes,
                bundle=bundle_name,
                interact=False,
                inline=False,
                figure=figure)

            eye = {}
            view_up = {}
            if view == "Sagittal":
                eye["x"] = 1
                eye["y"] = 0
                eye["z"] = 0
                view_up["x"] = 0
                view_up["y"] = 1
                view_up["z"] = 0
            elif view == "Coronal":
                eye["x"] = 0
                eye["y"] = 1
                eye["z"] = 0
                view_up["x"] = 0
                view_up["y"] = 0
                view_up["z"] = 1
            elif view == "Axial":
                eye["x"] = 0
                eye["y"] = 0
                eye["z"] = 1
                view_up["x"] = 1
                view_up["y"] = 0
                view_up["z"] = 0

            this_fname = tdir + f"/t{ii}.png"
            if "plotly" in viz_backend.backend:

                figure.update_layout(scene_camera=dict(
                    projection=dict(type="orthographic"),
                    up=view_up,
                    eye=eye,
                    center=dict(x=0, y=0, z=0)))
                figure.write_image(this_fname)

                # temporary fix for memory leak
                import plotly.io as pio
                pio.kaleido.scope._shutdown_kaleido()
            else:
                from dipy.viz import window
                direc = np.fromiter(eye.values(), dtype=int)
                data_shape = np.asarray(nib.load(b0).get_fdata().shape)
                figure.set_camera(
                    position=direc * data_shape,
                    focal_point=data_shape // 2,
                    view_up=tuple(view_up.values()))
                figure.zoom(0.5)
                window.snapshot(figure, fname=this_fname, size=(600, 600))

        def _save_file(curr_img, curr_file_num):
            save_path = op.abspath(op.join(
                self.afq_path,
                (f"bundle-{bundle_name}_view-{view}"
                    f"_idx-{curr_file_num}_montage.png")))
            curr_img.save(save_path)
            all_fnames.append(save_path)

        this_img_trimmed = {}
        max_height = 0
        max_width = 0
        for ii in range(len(self.valid_ses_list)):
            this_img = Image.open(tdir + f"/t{ii}.png")
            try:
                this_img_trimmed[ii] = trim(trim(this_img))
            except IndexError:  # this_img is a picture of nothing
                this_img_trimmed[ii] = this_img

            if this_img_trimmed[ii].size[0] > max_width:
                max_width = this_img_trimmed[ii].size[0]
            if this_img_trimmed[ii].size[1] > max_height:
                max_height = this_img_trimmed[ii].size[1]

        curr_img = Image.new(
            'RGB',
            (max_width * size[0], max_height * size[1]),
            color="white")
        curr_file_num = 0
        for ii in range(len(self.valid_ses_list)):
            x_pos = ii % size[0]
            _ii = ii // size[0]
            y_pos = _ii % size[1]
            _ii = _ii // size[1]
            file_num = _ii

            if file_num != curr_file_num:
                _save_file(curr_img, curr_file_num)
                curr_img = Image.new(
                    'RGB',
                    (max_width * size[0], max_height * size[1]),
                    color="white")
                curr_file_num = file_num
            curr_img.paste(
                this_img_trimmed[ii],
                (x_pos * max_width, y_pos * max_height))

        _save_file(curr_img, curr_file_num)
        return all_fnames

    def combine_bundle(self, bundle_name):
        """
        Transforms a given bundle to reg_template space for all subjects
        then merges them to one trk file.
        Useful for visualizing the variability in the bundle across subjects.
        Note: currently only implemented using built-in SynMap

        Parameters
        ----------
        bundle_name : str
        Name of the bundle to transform, should be one of the bundles in
        bundle_dict.
        """
        reference_wf_dict = self.wf_dict[
            self.valid_sub_list[0]][self.valid_ses_list[0]]
        if "mapping_definition" in reference_wf_dict:
            mapping_definition = reference_wf_dict["mapping_definition"]
            if mapping_definition is not None and not isinstance(
                    mapping_definition, SynMap):
                raise NotImplementedError((
                    "combine_bundle not implemented for mapping_definition "
                    "other than SynMap"))

        reg_template = self.export("reg_template", collapse=False)[
            self.valid_sub_list[0]][self.valid_ses_list[0]]
        clean_bundles_dict = self.export("clean_bundles", collapse=False)
        mapping_dict = self.export("mapping", collapse=False)

        sls_mni = []
        self.logger.info("Combining Bundles...")
        for ii in tqdm(range(len(self.valid_ses_list))):
            this_sub = self.valid_sub_list[ii]
            this_ses = self.valid_ses_list[ii]
            seg_sft = aus.SegmentedSFT.fromfile(clean_bundles_dict[
                this_sub][this_ses])
            seg_sft.sft.to_vox()
            sls = seg_sft.get_bundle(bundle_name).streamlines
            mapping = mapping_dict[this_sub][this_ses]

            if len(sls) > 0:
                delta = dts.values_from_volume(
                    mapping.forward,
                    sls, np.eye(4))
                sls_mni.extend([d + s for d, s in zip(delta, sls)])

        moved_sft = StatefulTractogram(
            sls_mni,
            reg_template,
            Space.VOX)
        save_tractogram(
            moved_sft,
            op.abspath(op.join(
                self.afq_path,
                f"bundle-{bundle_name}_subjects-all_MNI.trk")),
            bbox_valid_check=False)

    def upload_to_s3(self, s3fs, remote_path):
        """ Upload entire AFQ derivatives folder to S3"""
        s3fs.put(self.afq_path, remote_path, recursive=True)
        if op.exists(self.afqb_path):
            s3fs.put(self.afqb_path, remote_path, recursive=True)

    def export_group_density(self, boolify=True):
        """
        Generate a group density map by combining single subject density maps.

        Parameters
        ----------
        boolify : bool
            Whether to turn subject streamline count images into booleans
            before adding them into the group density map.

        Return
        ------
        Path to density nifti file.
        """
        densities = self.export("density_maps", collapse=False)
        ex_density_init = nib.load(densities[
            self.valid_sub_list[0]][
                self.valid_ses_list[0]])  # for shape and header

        group_density = np.zeros_like(ex_density_init.get_fdata())
        self.logger.info("Generating Group Density...")
        for ii in tqdm(range(len(self.valid_ses_list))):
            this_sub = self.valid_sub_list[ii]
            this_ses = self.valid_ses_list[ii]
            this_density = nib.load(densities[this_sub][this_ses]).get_fdata()
            if boolify:
                this_density = this_density.astype(bool)

            group_density = group_density + this_density
        group_density = group_density / len(self.valid_sub_list)
        group_density = nib.Nifti1Image(
            group_density,
            ex_density_init.affine,
            header=ex_density_init.header
        )

        out_fname = op.abspath(op.join(
            self.afq_path,
            f"desc-density_subjects-all_space-MNI_dwi.nii.gz"))
        nib.save(group_density, out_fname)
        return out_fname

    def assemble_AFQ_browser(self, output_path=None, metadata=None,
                             page_title="AFQ Browser", page_subtitle="",
                             page_title_link="", page_subtitle_link=""):
        """
        Assembles an instance of the AFQ-Browser from this AFQ instance.
        First, we generate the combined tract profile if it is not already
        generated. This includes running the full AFQ pipeline if it has not
        already run. The combined tract profile is one of the outputs of
        export_all.
        Second, we generate a streamlines.json file from the bundle
        recognized in the first subject's first session.
        Third, we call AFQ-Browser's assemble to assemble an AFQ-Browser
        instance in output_path.

        Parameters
        ----------
        output_path : str
            Path to location to create this instance of the browser in.
            Called "target" in AFQ Browser API. If None,
            bids_path/derivatives/afq_browser is used.
            Default: None
        metadata : str
            Path to subject metadata csv file. If None, an metadata file
            containing only subject ID is created. This file requires a
            "subjectID" column to work.
            Default: None
        page_title : str
            Page title. If None, prompt is sent to command line.
            Default: "AFQ Browser"
        page_subtitle : str
            Page subtitle. If None, prompt is sent to command line.
            Default: ""
        page_title_link : str
            Title hyperlink (including http(s)://).
            If None, prompt is sent to command line.
            Default: ""
        page_subtitle_link : str
            Subtitle hyperlink (including http(s)://).
            If None, prompt is sent to command line.
            Default: ""
        """
        if not using_afqb:
            self.logger.warning((
                "AFQ Browser is not installed, so AFQ Browswer instance "
                "cannot be assembled. AFQ Browser can be installed with: "
                "`pip install pyAFQ[afqbrowser]` or "
                "`pip install AFQ-Browser>=0.3`"))
            return

        if output_path is None:
            output_path = self.afqb_path
        os.makedirs(self.afqb_path, exist_ok=True)

        # generate combined profiles csv
        self.combine_profiles()

        # generate streamlines.json file
        sls_json_fname = self.get_streamlines_json()

        afqb.assemble(
            op.abspath(op.join(self.afq_path, "tract_profiles.csv")),
            target=output_path,
            metadata=metadata,
            streamlines=sls_json_fname,
            title=page_title,
            subtitle=page_subtitle,
            link=page_title_link,
            sublink=page_subtitle_link)


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
        remote_study = afs.S3BIDSStudy(
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
        df.to_csv("tmp.csv", index=False)
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
        df = clean_pandas_df(df)
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

    return clean_pandas_df(pd.concat(dfs))
