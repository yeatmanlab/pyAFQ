import nibabel as nib
import os.path as op
import os
from time import time
import logging

from AFQ.definitions.mapping import SlrMap
from AFQ.api.utils import (
    check_attribute, AFQclass_doc,
    export_all_helper, valid_exports_string)

from AFQ.tasks.data import get_data_plan
from AFQ.tasks.mapping import get_mapping_plan
from AFQ.tasks.tractography import get_tractography_plan
from AFQ.tasks.segmentation import get_segmentation_plan
from AFQ.tasks.viz import get_viz_plan
from AFQ.utils.path import drop_extension
from AFQ.data.s3bids import read_json


__all__ = ["ParticipantAFQ"]


class ParticipantAFQ(object):
    f"""{AFQclass_doc}"""

    def __init__(self,
                 dwi_data_file,
                 bval_file, bvec_file,
                 output_dir,
                 bids_info=None,
                 **kwargs):
        """
        Initialize a ParticipantAFQ object from a BIDS dataset.

        Parameters
        ----------
        dwi_data_file : str
            Path to DWI data file.
        bval_file : str
            Path to bval file.
        bvec_file : str
            Path to bvec file.
        output_dir : str
            Path to output directory.
        bids_info : dict or None, optional
            This is used by GroupAFQ to provide information about
            the BIDS layout to each participant.
        kwargs : additional optional parameters
            You can set additional parameters for any step
            of the process. See :ref:`usage/kwargs` for more details.

        Examples
        --------
        api.ParticipantAFQ(
            dwi_data_file, bval_file, bvec_file, output_dir,
            csd_sh_order=4)
        api.ParticipantAFQ(
            dwi_data_file, bval_file, bvec_file, output_dir,
            reg_template_spec="mni_t2", reg_subject_spec="b0")

        Notes
        -----
        In tracking_params, parameters with the suffix mask which are also
        an image from AFQ.definitions.image will be handled automatically by
        the api.

        It is recommended that you leave the bids_info parameter as None,
        and instead pass in the paths to the files you want to use directly.
        """
        if not isinstance(output_dir, str):
            raise TypeError(
                "output_dir must be a str")
        if not isinstance(dwi_data_file, str):
            raise TypeError(
                "dwi_data_file must be a str")
        if not isinstance(bval_file, str):
            raise TypeError(
                "bval_file must be a str")
        if not isinstance(bvec_file, str):
            raise TypeError(
                "bvec_file must be a str")
        if not op.exists(output_dir):
            raise ValueError(
                f"output_dir does not exist: {output_dir}")
        if "tractography_params" in kwargs:
            raise ValueError((
                "unrecognized parameter tractography_params, "
                "did you mean tracking_params ?"))

        self.logger = logging.getLogger('AFQ')
        self.output_dir = output_dir

        self.kwargs = dict(
            dwi=dwi_data_file,
            bval=bval_file,
            bvec=bvec_file,
            results_dir=output_dir,
            dwi_affine=nib.load(dwi_data_file).affine,
            bids_info=bids_info,
            base_fname=op.join(
                output_dir,
                drop_extension(op.basename(dwi_data_file))),
            **kwargs)
        self.make_workflow()

    def make_workflow(self):
        # construct pimms plans
        if "mapping_definition" in self.kwargs and isinstance(
                self.kwargs["mapping_definition"], SlrMap):
            plans = {  # if using SLR map, do tractography first
                "data": get_data_plan(self.kwargs),
                "tractography": get_tractography_plan(self.kwargs),
                "mapping": get_mapping_plan(self.kwargs, use_sls=True),
                "segmentation": get_segmentation_plan(self.kwargs),
                "viz": get_viz_plan(self.kwargs)}
        else:
            plans = {  # Otherwise, do mapping first
                "data": get_data_plan(self.kwargs),
                "mapping": get_mapping_plan(self.kwargs),
                "tractography": get_tractography_plan(self.kwargs),
                "segmentation": get_segmentation_plan(self.kwargs),
                "viz": get_viz_plan(self.kwargs)}

        # chain together a complete plan from individual plans
        previous_data = {}
        for name, plan in plans.items():
            previous_data[f"{name}_imap"] = plan(
                **self.kwargs,
                **previous_data)

        self.wf_dict =\
            previous_data[f"{name}_imap"]

    def export(self, attr_name="help"):
        """
        Export a specific output. To print a list of available outputs,
        call export without arguments.

        Parameters
        ----------
        attr_name : str
            Name of the output to export. Default: "help"

        Returns
        -------
        output : any
            The specific output, or None if called without arguments.
        """
        section = check_attribute(attr_name)

        if section is None:
            return self.wf_dict[attr_name]
        return self.wf_dict[section][attr_name]

    def export_all(self, viz=True, xforms=True,
                   indiv=True):
        f""" Exports all the possible outputs
        {valid_exports_string}

        Parameters
        ----------
        viz : bool
            Whether to output visualizations. This includes tract profile
            plots, a figure containing all bundles, and, if using the AFQ
            segmentation algorithm, individual bundle figures.
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
        seg_algo = self.export("segmentation_params").get("seg_algo", "AFQ")
        export_all_helper(self, seg_algo, xforms, indiv, viz)
        self.logger.info(
            f"Time taken for export all: {time() - start_time}")

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
        """
        if dependent_on is None:
            dependent_on_list = ["trk", "rec", "dwi"]
        elif dependent_on.lower() == "track":
            dependent_on_list = ["trk", "rec"]
        elif dependent_on.lower() == "recog":
            dependent_on_list = ["rec"]
        else:
            raise ValueError((
                "dependent_on must be one of "
                "None, 'track', or 'recog'."))

        exception_file_names = []
        for exception in exceptions:
            file_name = self.export(exception)
            if isinstance(file_name, str):
                exception_file_names.append(file_name)
            else:
                self.logger.warn((
                    f"The exception '{exception}' does not correspond"
                    " to a filename and will be ignored."))

        for filename in os.listdir(self.output_dir):
            if filename in exception_file_names:
                continue
            full_path = os.path.join(self.output_dir, filename)
            if os.path.isfile(full_path) or os.path.islink(full_path):
                if not full_path.startswith(self.export("base_fname")):
                    continue
                if not filename.endswith("json"):
                    sidecar_file = f'{drop_extension(full_path)}.json'
                    if op.exists(sidecar_file):
                        sidecar_info = read_json(sidecar_file)
                        if "dependent" in sidecar_info\
                            and sidecar_info["dependent"]\
                                in dependent_on_list:
                            os.system(f"{cmd} {full_path} {suffix}")
                            os.system(f"{cmd} {sidecar_file} {suffix}")
                    else:
                        os.system(f"{cmd} {full_path} {suffix}")
            elif os.path.isdir(full_path):
                # other than ROIs, folders are dependent on everything
                if dependent_on is None or filename != "ROIs":
                    os.system(f"{cmd} -r {full_path} {suffix}")

        # do not assume previous calculations are still valid
        # after file operations
        self.make_workflow()

    clobber = cmd_outputs  # alias for default of cmd_outputs
