import nibabel as nib
import logging
import numpy as np
import os

from pydra import mark

from dipy.align import resample

from AFQ.tasks.utils import *
import AFQ.utils.volume as auv

logger = logging.getLogger('AFQ.api.viz')


def _viz_prepare_vol(vol, xform, mapping, scalar_dict):
    if vol in scalar_dict.keys():
        vol = nib.load(scalar_dict[vol]).get_fdata()
    if isinstance(vol, str):
        vol = nib.load(vol).get_fdata()
    if xform:
        vol = mapping.transform_inverse(vol)
    return vol


@mark.task
@mark.annotate(
    {"return": {"all_bundles_figure_fname": str}}
)
def viz_bundles(subses_tuple,
                dwi_affine,
                viz_backend,
                bundle_dict,
                clean_bundles_file,
                mapping,
                scalar_dict,
                tracking_params,
                segmentation_params,
                volume,
                color_by_volume,
                xform_volume=False,
                cbv_lims=[None, None],
                xform_color_by_volume=False,
                volume_opacity=0.3,
                n_points=40):
    start_time = time()
    volume = _viz_prepare_vol(volume, xform_volume, mapping, scalar_dict)
    color_by_volume = _viz_prepare_vol(
        color_by_volume, xform_color_by_volume, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (dwi_affine[i, i] < 0)

    figure = viz_backend.visualize_volume(
        volume,
        opacity=volume_opacity,
        flip_axes=flip_axes,
        interact=False,
        inline=False)

    figure = viz_backend.visualize_bundles(
        clean_bundles_file,
        color_by_volume=color_by_volume,
        cbv_lims=cbv_lims,
        bundle_dict=bundle_dict,
        n_points=n_points,
        flip_axes=flip_axes,
        interact=False,
        inline=False,
        figure=figure)

    if "no_gif" not in viz_backend.backend:
        fname = get_fname(
            subses_tuple, '_viz.gif',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        viz_backend.create_gif(figure, fname)
    if "plotly" in viz_backend.backend:
        fname = get_fname(
            subses_tuple, '_viz.html',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        figure.write_html(fname)
    meta_fname = get_fname(
        subses_tuple, '_viz.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    afd.write_json(meta_fname, meta)
    return fname


@mark.task
def viz_indivBundle(subses_tuple,
                    dwi_affine,
                    viz_backend,
                    bundle_dict,
                    clean_bundles_file,
                    roi_files,
                    mapping,
                    scalar_dict,
                    tracking_params,
                    segmentation_params,
                    reg_template,
                    volume,
                    color_by_volume,
                    xform_volume=False,
                    cbv_lims=[None, None],
                    xform_color_by_volume=False,
                    volume_opacity=0.3,
                    n_points=40):
    start_time = time()
    volume = _viz_prepare_vol(volume, xform_volume, mapping, scalar_dict)
    color_by_volume = _viz_prepare_vol(
        color_by_volume, xform_color_by_volume, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (dwi_affine[i, i] < 0)

    bundle_names = bundle_dict.keys()

    for bundle_name in bundle_names:
        logger.info(f"Generating {bundle_name} visualization...")
        uid = bundle_dict[bundle_name]['uid']
        figure = viz_backend.visualize_volume(
            volume,
            opacity=volume_opacity,
            flip_axes=flip_axes,
            interact=False,
            inline=False)
        try:
            figure = viz_backend.visualize_bundles(
                clean_bundles_file,
                color_by_volume=color_by_volume,
                cbv_lims=cbv_lims,
                bundle_dict=bundle_dict,
                bundle=uid,
                n_points=n_points,
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
            endpoint_info = segmentation_params["endpoint_info"]
            if endpoint_info is not None:
                start_p = endpoint_info[bundle_name]['startpoint']
                end_p = endpoint_info[bundle_name]['endpoint']
                for ii, pp in enumerate([start_p, end_p]):
                    pp = resample(
                        pp.get_fdata(),
                        reg_template,
                        pp.affine,
                        reg_template).get_fdata()

                    atlas_roi = np.zeros(pp.shape)
                    atlas_roi[np.where(pp > 0)] = 1
                    warped_roi = auv.transform_inverse_roi(
                        atlas_roi,
                        mapping,
                        bundle_name=bundle_name)
                    warped_rois.append(warped_roi)
            else:
                aal_atlas = afd.read_aal_atlas(reg_template)
                atlas = aal_atlas['atlas'].get_fdata()
                aal_targets = afd.bundles_to_aal(
                    [bundle_name], atlas=atlas)[0]
                for targ in aal_targets:
                    if targ is not None:
                        aal_roi = np.zeros(atlas.shape[:3])
                        aal_roi[targ[:, 0],
                                targ[:, 1],
                                targ[:, 2]] = 1
                    warped_roi = auv.transform_inverse_roi(
                        aal_roi,
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

        for i, roi in enumerate(roi_files[bundle_name]):
            figure = viz_backend.visualize_roi(
                roi,
                name=f"{bundle_name} ROI {i}",
                flip_axes=flip_axes,
                inline=False,
                interact=False,
                figure=figure)

        roi_dir = op.join(subses_tuple['results_dir'], 'viz_bundles')
        os.makedirs(roi_dir, exist_ok=True)
        if "no_gif" not in viz_backend.backend:
            fname = op.split(
                get_fname(
                    subses_tuple,
                    f'_{bundle_name}'
                    f'_viz.gif',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            viz_backend.create_gif(figure, fname)
        if "plotly" in viz_backend.backend:
            roi_dir = op.join(subses_tuple['results_dir'], 'viz_bundles')
            os.makedirs(roi_dir, exist_ok=True)
            fname = op.split(
                get_fname(
                    subses_tuple,
                    f'_{bundle_name}'
                    f'_viz.html',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            figure.write_html(fname)
    meta_fname = get_fname(
        subses_tuple, '_indiv_viz.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    afd.write_json(meta_fname, meta)
