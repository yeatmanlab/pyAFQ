import nibabel as nib
import os.path as op
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

        self.logger = logging.getLogger('AFQ.api')

        kwargs = dict(
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

        # construct pimms plans
        if "mapping_definition" in kwargs and isinstance(
                kwargs["mapping_definition"], SlrMap):
            plans = {  # if using SLR map, do tractography first
                "data": get_data_plan(kwargs),
                "tractography": get_tractography_plan(kwargs),
                "mapping": get_mapping_plan(kwargs, use_sls=True),
                "segmentation": get_segmentation_plan(kwargs),
                "viz": get_viz_plan(kwargs)}
        else:
            plans = {  # Otherwise, do mapping first
                "data": get_data_plan(kwargs),
                "mapping": get_mapping_plan(kwargs),
                "tractography": get_tractography_plan(kwargs),
                "segmentation": get_segmentation_plan(kwargs),
                "viz": get_viz_plan(kwargs)}

        # chain together a complete plan from individual plans
        previous_data = {}
        for name, plan in plans.items():
            previous_data[f"{name}_imap"] = plan(
                **kwargs,
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
