import nibabel as nib
import os.path as op
from time import time
import logging
from tqdm import tqdm
import numpy as np
import tempfile
import math

from PIL import Image, ImageDraw, ImageFont

from AFQ.definitions.mapping import SlrMap
from AFQ.api.utils import (
    check_attribute, AFQclass_doc,
    export_all_helper, valid_exports_string)

from AFQ.tasks.data import get_data_plan
from AFQ.tasks.mapping import get_mapping_plan
from AFQ.tasks.tractography import get_tractography_plan
from AFQ.tasks.segmentation import get_segmentation_plan
from AFQ.tasks.viz import get_viz_plan
from AFQ.tasks.utils import get_base_fname
from AFQ.utils.path import apply_cmd_to_afq_derivs
from AFQ.viz.utils import BEST_BUNDLE_ORIENTATIONS, trim, get_eye


__all__ = ["ParticipantAFQ"]


class ParticipantAFQ(object):
    f"""{AFQclass_doc}"""

    def __init__(self,
                 dwi_data_file,
                 bval_file, bvec_file,
                 output_dir,
                 _bids_info=None,
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
        _bids_info : dict or None, optional
            This should be left as None in most cases. It
            is used by GroupAFQ to provide information about
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
            dwi_path=dwi_data_file,
            bval=bval_file,
            bvec=bvec_file,
            results_dir=output_dir,
            bids_info=_bids_info,
            base_fname=get_base_fname(output_dir, dwi_data_file),
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

    def export_up_to(self, attr_name="help"):
        f"""
        Export all derivatives necessary for a specific output.
        To print a list of available outputs,
        call export_up_to without arguments.
        {valid_exports_string}

        Parameters
        ----------
        attr_name : str
            Name of the output to export up to. Default: "help"
        """
        section = check_attribute(attr_name)
        wf_dict = self.wf_dict
        if section is not None:
            wf_dict = wf_dict[section]
        for dependent in wf_dict.plan.dependencies[attr_name]:
            self.export(dependent)

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

    def participant_montage(self, images_per_row=2):
        """
        Generate montage of all bundles for a given subject.

        Parameters
        ----------
        images_per_row : int
            Number of bundle images per row in output file.
            Default: 2

        Returns
        -------
        filename of montage images
        """
        tdir = tempfile.gettempdir()

        all_fnames = []
        bundle_dict = self.export("bundle_dict")
        self.logger.info("Generating Montage...")
        viz_backend = self.export("viz_backend")
        best_scalar = self.export(self.export("best_scalar"))
        size = (images_per_row, math.ceil(len(bundle_dict) / images_per_row))
        for ii, bundle_name in enumerate(tqdm(bundle_dict)):
            flip_axes = [False, False, False]
            for i in range(3):
                flip_axes[i] = (self.export("dwi_affine")[i, i] < 0)

            figure = viz_backend.visualize_volume(
                best_scalar,
                flip_axes=flip_axes,
                interact=False,
                inline=False)
            figure = viz_backend.visualize_bundles(
                self.export("bundles"),
                shade_by_volume=best_scalar,
                color_by_direction=True,
                flip_axes=flip_axes,
                bundle=bundle_name,
                figure=figure,
                interact=False,
                inline=False)

            view, direc = BEST_BUNDLE_ORIENTATIONS.get(
                bundle_name, ("Axial", "Top"))
            eye = get_eye(view, direc)

            this_fname = tdir + f"/t{ii}.png"
            if "plotly" in viz_backend.backend:
                figure.update_layout(
                    scene_camera=dict(
                        projection=dict(type="orthographic"),
                        up={"x": 0, "y": 0, "z": 1},
                        eye=eye,
                        center=dict(x=0, y=0, z=0)),
                    showlegend=False)
                figure.write_image(this_fname, scale=4)

                # temporary fix for memory leak
                import plotly.io as pio
                pio.kaleido.scope._shutdown_kaleido()
            else:
                from dipy.viz import window
                direc = np.fromiter(eye.values(), dtype=int)
                data_shape = np.asarray(
                    nib.load(self.export("b0")).get_fdata().shape)
                figure.set_camera(
                    position=direc * data_shape,
                    focal_point=data_shape // 2,
                    view_up=(0, 0, 1))
                figure.zoom(0.5)
                window.snapshot(figure, fname=this_fname, size=(600, 600))

        def _save_file(curr_img):
            save_path = op.abspath(op.join(
                self.output_dir,
                "bundle_montage.png"))
            curr_img.save(save_path)
            all_fnames.append(save_path)

        this_img_trimmed = {}
        max_height = 0
        max_width = 0
        for ii, bundle_name in enumerate(bundle_dict):
            this_img = Image.open(tdir + f"/t{ii}.png")
            try:
                this_img_trimmed[ii] = trim(this_img)
            except IndexError:  # this_img is a picture of nothing
                this_img_trimmed[ii] = this_img

            text_sz = 70
            width, height = this_img_trimmed[ii].size
            height = height + text_sz
            result = Image.new(
                this_img_trimmed[ii].mode, (width, height),
                color=(255, 255, 255))
            result.paste(this_img_trimmed[ii], (0, text_sz))
            this_img_trimmed[ii] = result

            draw = ImageDraw.Draw(this_img_trimmed[ii])
            draw.text(
                (0, 0), bundle_name, (0, 0, 0),
                font=ImageFont.truetype(
                    "Arial", text_sz))

            if this_img_trimmed[ii].size[0] > max_width:
                max_width = this_img_trimmed[ii].size[0]
            if this_img_trimmed[ii].size[1] > max_height:
                max_height = this_img_trimmed[ii].size[1]

        curr_img = Image.new(
            'RGB',
            (max_width * size[0], max_height * size[1]),
            color="white")

        for ii in range(len(bundle_dict)):
            x_pos = ii % size[0]
            _ii = ii // size[0]
            y_pos = _ii % size[1]
            _ii = _ii // size[1]
            this_img = this_img_trimmed[ii].resize((max_width, max_height))
            curr_img.paste(
                this_img,
                (x_pos * max_width, y_pos * max_height))

        _save_file(curr_img)
        return all_fnames

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
            If "prof", perform on all derivatives that depend on the
            bundle profiling.
            Default: None
        exceptions : list of str
            Name outputs that the command should not be applied to.
            Default: []
        suffix : str
            Parts of command that are used after the filename.
            Default: ""
        """
        exception_file_names = []
        for exception in exceptions:
            file_name = self.export(exception)
            if isinstance(file_name, str):
                exception_file_names.append(file_name)
            else:
                self.logger.warn((
                    f"The exception '{exception}' does not correspond"
                    " to a filename and will be ignored."))

        apply_cmd_to_afq_derivs(
            self.output_dir,
            self.export("base_fname"),
            cmd=cmd,
            exception_file_names=exception_file_names,
            suffix=suffix,
            dependent_on=dependent_on
        )

        # do not assume previous calculations are still valid
        # after file operations
        self.make_workflow()

    clobber = cmd_outputs  # alias for default of cmd_outputs
