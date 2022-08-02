import nibabel as nib
import logging
import numpy as np
import os
import os.path as op
from time import time
import pandas as pd

import pimms

from dipy.align import resample

from AFQ.tasks.utils import get_fname, with_name, str_to_desc
import AFQ.utils.volume as auv
from AFQ.data.s3bids import write_json
from AFQ.viz.utils import Viz

from plotly.subplots import make_subplots

logger = logging.getLogger('AFQ')


def _viz_prepare_vol(vol, xform, mapping, scalar_dict):
    if vol in scalar_dict.keys():
        vol = scalar_dict[vol]
        if isinstance(vol, str):
            vol = nib.load(vol)
        vol = vol.get_fdata()
    if isinstance(vol, str):
        vol = nib.load(vol).get_fdata()
    if xform:
        vol = mapping.transform_inverse(vol)
    return vol


@pimms.calc("all_bundles_figure")
def viz_bundles(base_fname,
                dwi_affine,
                viz_backend,
                data_imap,
                mapping_imap,
                segmentation_imap,
                tracking_params,
                segmentation_params,
                best_scalar,
                sbv_lims_bundles=[None, None],
                volume_opacity_bundles=0.3,
                n_points_bundles=40):
    """
    figure for the visualizaion of the recognized
    bundles in the subject's brain.

    Parameters
    ----------
    sbv_lims_bundles : ndarray
        Of the form (lower bound, upper bound). Shading based on
        shade_by_volume will only differentiate values within these bounds.
        If lower bound is None, will default to 0.
        If upper bound is None, will default to the maximum value in
        shade_by_volume.
        Default: [None, None]
    volume_opacity_bundles : float, optional
        Opacity of volume slices.
        Default: 0.3
    n_points_bundles : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.
        Default: 40

    Returns
    -------
    List of Figure, String or just the Figure:
    If file can be generated, returns a tuple including the figure and the
    path to the file.
    Otherwise, returns the figure.
    """
    mapping = mapping_imap["mapping"]
    bundle_dict = data_imap["bundle_dict"]
    scalar_dict = segmentation_imap["scalar_dict"]
    profiles_file = segmentation_imap["profiles"]
    volume = data_imap["b0"]
    shade_by_volume = data_imap[best_scalar]
    start_time = time()
    volume = _viz_prepare_vol(volume, False, mapping, scalar_dict)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (dwi_affine[i, i] < 0)

    if "plotly" in viz_backend.backend:
        figure = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]])
    else:
        figure = None

    figure = viz_backend.visualize_volume(
        volume,
        opacity=volume_opacity_bundles,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure)

    figure = viz_backend.visualize_bundles(
        segmentation_imap["clean_bundles"],
        shade_by_volume=shade_by_volume,
        sbv_lims=sbv_lims_bundles,
        bundle_dict=bundle_dict,
        include_profiles=(pd.read_csv(profiles_file), best_scalar),
        n_points=n_points_bundles,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure)

    fname = None
    if "no_gif" not in viz_backend.backend:
        fname = get_fname(
            base_fname, '_desc-viz_dwi.gif',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        viz_backend.create_gif(figure, fname)
    if "plotly" in viz_backend.backend:
        fname = get_fname(
            base_fname, '_desc-viz_dwi.html',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        figure.write_html(fname)
    meta_fname = get_fname(
        base_fname, '_desc-viz_dwi.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    write_json(meta_fname, meta)
    if fname is None:
        return figure
    else:
        return [figure, fname]


@pimms.calc("indiv_bundles_figures")
def viz_indivBundle(base_fname,
                    results_dir,
                    dwi_affine,
                    viz_backend,
                    data_imap,
                    mapping_imap,
                    segmentation_imap,
                    tracking_params,
                    segmentation_params,
                    best_scalar,
                    sbv_lims_indiv=[None, None],
                    volume_opacity_indiv=0.3,
                    n_points_indiv=40):
    """
    list of full paths to html or gif files
    containing visualizaions of individual bundles

    Parameters
    ----------
    sbv_lims_indiv : ndarray
        Of the form (lower bound, upper bound). Shading based on
        shade_by_volume will only differentiate values within these bounds.
        If lower bound is None, will default to 0.
        If upper bound is None, will default to the maximum value in
        shade_by_volume.
        Default: [None, None]
    volume_opacity_indiv : float, optional
        Opacity of volume slices.
        Default: 0.3
    n_points_indiv : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.
        Default: 40
    """
    mapping = mapping_imap["mapping"]
    bundle_dict = data_imap["bundle_dict"]
    reg_template = data_imap["reg_template"]
    scalar_dict = segmentation_imap["scalar_dict"]
    volume = data_imap["b0"]
    shade_by_volume = data_imap[best_scalar]
    profiles = pd.read_csv(segmentation_imap["profiles"])

    start_time = time()
    volume = _viz_prepare_vol(
        volume, False, mapping, scalar_dict)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (dwi_affine[i, i] < 0)

    bundle_names = bundle_dict.keys()

    figures = {}
    for bundle_name in bundle_names:
        logger.info(f"Generating {bundle_name} visualization...")
        figure = viz_backend.visualize_volume(
            volume,
            opacity=volume_opacity_indiv,
            flip_axes=flip_axes,
            interact=False,
            inline=False)
        try:
            figure = viz_backend.visualize_bundles(
                segmentation_imap["clean_bundles"],
                shade_by_volume=shade_by_volume,
                sbv_lims=sbv_lims_indiv,
                bundle_dict=bundle_dict,
                bundle=bundle_name,
                n_points=n_points_indiv,
                flip_axes=flip_axes,
                interact=False,
                inline=False,
                figure=figure)
        except ValueError:
            logger.info(
                "No streamlines found to visualize for "
                + bundle_name)

        if segmentation_params["filter_by_endpoints"]:
            warped_rois = []
            for reg_type in ['start', 'end']:
                if reg_type in bundle_dict[bundle_name]:
                    pp = bundle_dict[bundle_name][reg_type]
                    pp = resample(
                        pp.get_fdata(),
                        reg_template,
                        pp.affine,
                        reg_template.affine).get_fdata()

                    atlas_roi = np.zeros(pp.shape)
                    atlas_roi[np.where(pp > 0)] = 1
                    warped_roi = auv.transform_inverse_roi(
                        atlas_roi,
                        mapping,
                        bundle_name=bundle_name)
                    warped_rois.append(warped_roi)
            for i, roi in enumerate(warped_rois):
                figure = viz_backend.visualize_roi(
                    roi,
                    name=f"{bundle_name} endpoint ROI {i}",
                    flip_axes=flip_axes,
                    inline=False,
                    interact=False,
                    figure=figure)

        for i, roi in enumerate(mapping_imap["rois"][bundle_name]):
            figure = viz_backend.visualize_roi(
                roi,
                name=f"{bundle_name} ROI {i}",
                flip_axes=flip_axes,
                inline=False,
                interact=False,
                figure=figure)

        roi_dir = op.join(results_dir, 'viz_bundles')
        os.makedirs(roi_dir, exist_ok=True)
        figures[bundle_name] = figure
        if "no_gif" not in viz_backend.backend:
            fname = op.split(
                get_fname(
                    base_fname,
                    f'_desc-{str_to_desc(bundle_name)}viz'
                    f'_dwi.gif',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            viz_backend.create_gif(figure, fname)
        if "plotly" in viz_backend.backend:
            roi_dir = op.join(results_dir, 'viz_bundles')
            os.makedirs(roi_dir, exist_ok=True)
            fname = op.split(
                get_fname(
                    base_fname,
                    f'_desc-{str_to_desc(bundle_name)}viz'
                    f'_dwi.html',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            figure.write_html(fname)

            # also do the core visualizations when using the plotly backend
            core_dir = op.join(results_dir, 'viz_core_bundles')
            os.makedirs(core_dir, exist_ok=True)
            indiv_profile = profiles[
                profiles.tractID == bundle_name][best_scalar].to_numpy()
            if len(indiv_profile) > 1:
                fname = op.split(
                    get_fname(
                        base_fname,
                        f'_desc-{str_to_desc(bundle_name)}viz'
                        f'_dwi.html',
                        tracking_params=tracking_params,
                        segmentation_params=segmentation_params))
                fname = op.join(core_dir, fname[1])
                core_fig = make_subplots(
                    rows=1, cols=2,
                    specs=[[{"type": "scene"}, {"type": "scene"}]])
                core_fig = viz_backend.visualize_volume(
                    volume,
                    opacity=volume_opacity_indiv,
                    flip_axes=flip_axes,
                    figure=core_fig,
                    interact=False,
                    inline=False)
                core_fig = viz_backend.visualize_bundles(
                    segmentation_imap["clean_bundles"],
                    shade_by_volume=shade_by_volume,
                    sbv_lims=sbv_lims_indiv,
                    bundle_dict=bundle_dict,
                    bundle=bundle_name,
                    colors={bundle_name: [0.5, 0.5, 0.5]},
                    n_points=n_points_indiv,
                    flip_axes=flip_axes,
                    interact=False,
                    inline=False,
                    figure=core_fig)
                core_fig = viz_backend.single_bundle_viz(
                    indiv_profile,
                    segmentation_imap["clean_bundles"],
                    bundle_name,
                    best_scalar,
                    bundle_dict=bundle_dict,
                    flip_axes=flip_axes,
                    figure=core_fig,
                    include_profile=True)
                core_fig.write_html(fname)
    meta_fname = get_fname(
        base_fname, f'_desc-{str_to_desc(bundle_name)}viz_dwi',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    write_json(meta_fname, meta)
    return {"indiv_bundles_figures": figures}


@pimms.calc("tract_profile_plots")
def plot_tract_profiles(base_fname, scalars, tracking_params,
                        segmentation_params, segmentation_imap):
    """
    list of full paths to png files,
    where files contain plots of the tract profiles
    """
    from AFQ.viz.plot import visualize_tract_profiles
    start_time = time()
    fnames = []
    for scalar in scalars:
        this_scalar = scalar if isinstance(scalar, str) else scalar.get_name()
        fname = get_fname(
            base_fname,
            f'_model-{str_to_desc(this_scalar)}_desc-vizprofile_dwi',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)
        tract_profiles_folder = op.join(
            op.dirname(fname),
            "tract_profile_plots")
        fname = op.join(
            tract_profiles_folder,
            op.basename(fname))
        os.makedirs(op.abspath(tract_profiles_folder), exist_ok=True)

        visualize_tract_profiles(
            segmentation_imap["profiles"],
            scalar=this_scalar,
            file_name=fname,
            n_boot=100)
        fnames.append(fname + ".png")
        meta_fname = fname + ".json"
        meta = dict(Timing=time() - start_time)
        write_json(meta_fname, meta)
    return fnames


@pimms.calc("viz_backend")
def init_viz_backend(viz_backend_spec="plotly_no_gif",
                     virtual_frame_buffer=False):
    """
    An instance of the `AFQ.viz.utils.viz_backend` class.

    Parameters
    ----------
    virtual_frame_buffer : bool, optional
        Whether to use a virtual fram buffer. This is neccessary if
        generating GIFs in a headless environment. Default: False
    viz_backend_spec : str, optional
        Which visualization backend to use.
        See Visualization Backends page in documentation for details:
        https://yeatmanlab.github.io/pyAFQ/usage/viz_backend.html
        One of {"fury", "plotly", "plotly_no_gif"}.
        Default: "plotly_no_gif"
    """
    if not isinstance(virtual_frame_buffer, bool):
        raise TypeError("virtual_frame_buffer must be a bool")
    if "fury" not in viz_backend_spec\
            and "plotly" not in viz_backend_spec:
        raise TypeError(
            "viz_backend_spec must contain either 'fury' or 'plotly'")

    if virtual_frame_buffer:
        from xvfbwrapper import Xvfb
        vdisplay = Xvfb(width=1280, height=1280)
        vdisplay.start()

    return Viz(backend=viz_backend_spec.lower())


def get_viz_plan(kwargs):
    viz_tasks = with_name([
        plot_tract_profiles, viz_bundles, viz_indivBundle, init_viz_backend])
    return pimms.plan(**viz_tasks)
