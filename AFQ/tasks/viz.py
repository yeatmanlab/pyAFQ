import nibabel as nib
import logging
import numpy as np
import os
import os.path as op
from time import time
import pandas as pd

import pimms

from AFQ.tasks.utils import (
    get_fname, with_name, str_to_desc, get_default_args)
from AFQ.tasks.decorators import as_file
import AFQ.utils.volume as auv
from AFQ.viz.utils import Viz
import AFQ.utils.streamlines as aus
from AFQ.utils.path import write_json
import AFQ.nn.profile_roi as anp

from plotly.subplots import make_subplots

from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.tracking.streamline import set_number_of_points, values_from_volume
from dipy.align import resample


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
    vol[np.isnan(vol)] = 0
    return vol


@pimms.calc("profiles")
@as_file('_desc-profiles_dwi.csv', include_track=True, include_seg=True)
def tract_profiles(segmentation_imap,
                   data_imap,
                   nn_bundle_dict=None,
                   profile_weights="gauss",
                   n_points_profile=100):
    """
    full path to a CSV file containing tract profiles

    Parameters
    ----------
    nn_bundle_dict : dict, optional
        Dictionary specify how to use ROIs from HypVINN
        segmentation. If None, no HypVINN segmentation
        will be performed.
        Default: None
    profile_weights : str, 1D array, 2D array callable, optional
        How to weight each streamline (1D) or each node (2D)
        when calculating the tract-profiles. If callable, this is a
        function that calculates weights. If None, no weighting will
        be applied. If "gauss", gaussian weights will be used.
        If "median", the median of values at each node will be used
        instead of a mean or weighted mean.
        Default: "gauss"
    n_points_profile : int, optional
        Number of points to resample each streamline to before
        calculating the tract-profiles.
        Default: 100
    """
    if not (profile_weights is None
            or isinstance(profile_weights, str)
            or callable(profile_weights)
            or hasattr(profile_weights, "__len__")):
        raise TypeError(
            "profile_weights must be string, None, callable, or"
            + "a 1D or 2D array")
    if isinstance(profile_weights, str):
        profile_weights = profile_weights.lower()
    if isinstance(profile_weights, str) and\
            profile_weights != "gauss" and profile_weights != "median":
        raise TypeError(
            "if profile_weights is a string,"
            + " it must be 'gauss' or 'median'")

    scalar_dict = segmentation_imap["scalar_dict"]
    bundle_names = []
    node_numbers = []
    profiles = np.empty((len(scalar_dict), 0)).tolist()
    this_profile = np.zeros((len(scalar_dict), n_points_profile))
    reference = nib.load(scalar_dict[list(scalar_dict.keys())[0]])
    nodes = list(np.arange(n_points_profile))
    meta = dict(parameters=get_default_args(afq_profile))

    if len(data_imap["bundle_dict"]) > 0:
        seg_sft = aus.SegmentedSFT.fromfile(
            segmentation_imap["bundles"],
            reference=reference)
        seg_sft.sft.to_rasmm()
        meta["bundles_source"] = segmentation_imap["bundles"]
        for bundle_name in seg_sft.bundle_names:
            this_sl = seg_sft.get_bundle(bundle_name).streamlines
            if len(this_sl) == 0:
                continue
            if profile_weights == "gauss":
                # calculate only once per bundle
                bundle_profile_weights = gaussian_weights(
                    this_sl,
                    n_points=n_points_profile)
            for ii, (scalar, scalar_file) in enumerate(scalar_dict.items()):
                if isinstance(scalar_file, str):
                    scalar_file = nib.load(scalar_file)
                scalar_data = scalar_file.get_fdata()
                if isinstance(profile_weights, str):
                    if profile_weights == "gauss":
                        this_prof_weights = bundle_profile_weights
                    elif profile_weights == "median":
                        # weights bundle to only return the mean
                        def _median_weight(bundle):
                            fgarray = set_number_of_points(
                                bundle, n_points_profile)
                            values = np.array(
                                values_from_volume(
                                    scalar_data,
                                    fgarray,
                                    data_imap["dwi_affine"]))
                            weights = np.zeros(values.shape)
                            for ii, jj in enumerate(
                                np.argsort(values, axis=0)[
                                    len(values) // 2, :]):
                                weights[jj, ii] = 1
                            return weights
                        this_prof_weights = _median_weight
                else:
                    this_prof_weights = profile_weights
                this_profile[ii] = afq_profile(
                    scalar_data,
                    this_sl,
                    data_imap["dwi_affine"],
                    weights=this_prof_weights,
                    n_points=n_points_profile)
                profiles[ii].extend(list(this_profile[ii]))

            bundle_names.extend([bundle_name] * len(nodes))
            node_numbers.extend(nodes)

    if nn_bundle_dict is not None:
        hypvinn_seg = nib.load(data_imap["hypvinn_seg"])
        meta["seg_source"] = data_imap["hypvinn_seg"]
        for bundle_name, bundle_info in nn_bundle_dict.items():
            roi = anp.roi_from_segmentation(
                hypvinn_seg,
                bundle_info["label"],
                data_imap["dwi"])
            skel_pts = anp.skeleton_from_roi(
                roi,
                data_imap["dwi"].affine,
                bundle_info["orientation_axis"])
            if len(skel_pts) == 0:
                logger.warning(f"{bundle_name} not found")
            else:
                for ii, sc_data in enumerate(scalar_dict.values()):
                    if isinstance(sc_data, str):
                        sc_data = nib.load(sc_data)
                    sc_data = sc_data.get_fdata()
                    profiles[ii].extend(list(anp.profile_roi(
                        roi, skel_pts, sc_data)))
                bundle_names.extend([bundle_name] * len(nodes))
                node_numbers.extend(nodes)

    profile_dict = dict()
    profile_dict["tractID"] = bundle_names
    profile_dict["nodeID"] = node_numbers
    for ii, scalar in enumerate(scalar_dict.keys()):
        profile_dict[scalar] = profiles[ii]

    profile_dframe = pd.DataFrame(profile_dict)

    return profile_dframe, meta


@pimms.calc("all_bundles_figure")
def viz_bundles(base_fname,
                viz_backend,
                data_imap,
                mapping_imap,
                segmentation_imap,
                tracking_params,
                segmentation_params,
                profiles,
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
    scalar_dict = segmentation_imap["scalar_dict"]
    volume = data_imap["masked_b0"]
    shade_by_volume = data_imap[best_scalar]
    start_time = time()
    volume = _viz_prepare_vol(volume, False, mapping, scalar_dict)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (data_imap["dwi_affine"][i, i] < 0)

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
        segmentation_imap["bundles"],
        shade_by_volume=shade_by_volume,
        sbv_lims=sbv_lims_bundles,
        include_profiles=(pd.read_csv(profiles), best_scalar),
        n_points=n_points_bundles,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure)

    if "nn_bundle_dict" in data_imap and\
            data_imap["nn_bundle_dict"] is not None:
        hypvinn_seg = nib.load(data_imap["hypvinn_seg"])
        for bundle_name, bundle_info in data_imap["nn_bundle_dict"].items():
            roi = anp.roi_from_segmentation(
                hypvinn_seg,
                bundle_info["label"],
                data_imap["dwi"])
            figure = viz_backend.visualize_roi(
                roi,
                name=bundle_name,
                flip_axes=flip_axes,
                inline=False,
                interact=False,
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
                    viz_backend,
                    data_imap,
                    mapping_imap,
                    segmentation_imap,
                    tracking_params,
                    segmentation_params,
                    profiles,
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
    volume = data_imap["masked_b0"]
    shade_by_volume = data_imap[best_scalar]
    profiles = pd.read_csv(profiles)

    start_time = time()
    volume = _viz_prepare_vol(
        volume, False, mapping, scalar_dict)
    shade_by_volume = _viz_prepare_vol(
        shade_by_volume, False, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (data_imap["dwi_affine"][i, i] < 0)

    bundles = aus.SegmentedSFT.fromfile(
        segmentation_imap["bundles"])

    # This dictionary contains a mapping to which ROIs
    # should be used from the bundle dict, based on the
    # name from the segmented SFT file. Currently,
    # This is only different when using bundle sections.
    segmented_bname_to_roi_bname = {}
    for b_name, b_info in bundle_dict.items():
        if "bundlesection" in b_info:
            for sb_name in b_info["bundlesection"]:
                segmented_bname_to_roi_bname[sb_name] = b_name
        else:
            segmented_bname_to_roi_bname[b_name] = b_name

    figures = {}
    for bundle_name in bundles.bundle_names:
        logger.info(f"Generating {bundle_name} visualization...")
        roi_bname = segmented_bname_to_roi_bname[bundle_name]

        figure = viz_backend.visualize_volume(
            volume,
            opacity=volume_opacity_indiv,
            flip_axes=flip_axes,
            interact=False,
            inline=False)
        if len(bundles.get_bundle(bundle_name)) > 0:
            figure = viz_backend.visualize_bundles(
                bundles,
                shade_by_volume=shade_by_volume,
                sbv_lims=sbv_lims_indiv,
                bundle=bundle_name,
                n_points=n_points_indiv,
                flip_axes=flip_axes,
                interact=False,
                inline=False,
                figure=figure)
        else:
            logger.info(
                "No streamlines found to visualize for "
                + bundle_name)

        if segmentation_params["filter_by_endpoints"]:
            warped_rois = []
            for reg_type in ['start', 'end']:
                if reg_type in bundle_dict[
                        roi_bname]:
                    pp = bundle_dict[roi_bname][reg_type]
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
                        bundle_name=roi_bname)
                    warped_rois.append(warped_roi)
            for i, roi in enumerate(warped_rois):
                figure = viz_backend.visualize_roi(
                    roi,
                    name=f"{roi_bname} endpoint ROI {i}",
                    flip_axes=flip_axes,
                    inline=False,
                    interact=False,
                    figure=figure)

        for roi_fname in mapping_imap["rois"][roi_bname]:
            figure = viz_backend.visualize_roi(
                roi_fname,
                name=roi_fname.split("desc-")[1].split("_")[0],
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
                    segmentation_imap["bundles"],
                    shade_by_volume=shade_by_volume,
                    sbv_lims=sbv_lims_indiv,
                    bundle=bundle_name,
                    colors={bundle_name: [0.5, 0.5, 0.5]},
                    n_points=n_points_indiv,
                    flip_axes=flip_axes,
                    interact=False,
                    inline=False,
                    figure=core_fig)
                core_fig = viz_backend.single_bundle_viz(
                    indiv_profile,
                    segmentation_imap["bundles"],
                    bundle_name,
                    best_scalar,
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
def plot_tract_profiles(base_fname, profiles, scalars, tracking_params,
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
            profiles,
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
        plot_tract_profiles, viz_bundles, viz_indivBundle, init_viz_backend,
        tract_profiles])
    return pimms.plan(**viz_tasks)
