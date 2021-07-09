import nibabel as nib
import logging
import numpy as np
import os
import os.path as op
from time import time

import pimms

from dipy.align import resample

from AFQ.tasks.utils import get_fname, with_name
import AFQ.utils.volume as auv
import AFQ.data as afd
from AFQ.viz.utils import visualize_tract_profiles

logger = logging.getLogger('AFQ.api.viz')


outputs = ["all_bundles_figure", "indiv_bundles_figures"]


def _viz_prepare_vol(vol, xform, mapping, scalar_dict):
    if vol in scalar_dict.keys():
        vol = nib.load(scalar_dict[vol]).get_fdata()
    if isinstance(vol, str):
        vol = nib.load(vol).get_fdata()
    if xform:
        vol = mapping.transform_inverse(vol)
    return vol


@pimms.calc("all_bundles_figure")
def viz_bundles(subses_dict,
                dwi_affine,
                viz_backend,
                bundle_dict,
                data_imap,
                mapping_imap,
                segmentation_imap,
                tracking_params,
                segmentation_params,
                best_scalar,
                xform_volume=False,
                cbv_lims=[None, None],
                xform_color_by_volume=False,
                volume_opacity=0.3,
                n_points=40):
    mapping = mapping_imap["mapping"]
    scalar_dict = segmentation_imap["scalar_dict"]
    volume = data_imap["b0_file"]
    color_by_volume = data_imap[best_scalar + "_file"]
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
        segmentation_imap["clean_bundles_file"],
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
            subses_dict, '_viz.gif',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        viz_backend.create_gif(figure, fname)
    if "plotly" in viz_backend.backend:
        fname = get_fname(
            subses_dict, '_viz.html',
            tracking_params=tracking_params,
            segmentation_params=segmentation_params)

        figure.write_html(fname)
    meta_fname = get_fname(
        subses_dict, '_viz.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    afd.write_json(meta_fname, meta)
    return figure


@pimms.calc("indiv_bundles_figures")
def viz_indivBundle(subses_dict,
                    dwi_affine,
                    viz_backend,
                    bundle_dict,
                    data_imap,
                    mapping_imap,
                    segmentation_imap,
                    tracking_params,
                    segmentation_params,
                    reg_template,
                    best_scalar,
                    xform_volume_indiv=False,
                    cbv_lims_indiv=[None, None],
                    xform_color_by_volume_indiv=False,
                    volume_opacity_indiv=0.3,
                    n_points_indiv=40):
    mapping = mapping_imap["mapping"]
    scalar_dict = segmentation_imap["scalar_dict"]
    volume = data_imap["b0_file"]
    color_by_volume = data_imap[best_scalar + "_file"]

    start_time = time()
    volume = _viz_prepare_vol(
        volume, xform_volume_indiv, mapping, scalar_dict)
    color_by_volume = _viz_prepare_vol(
        color_by_volume, xform_color_by_volume_indiv, mapping, scalar_dict)

    flip_axes = [False, False, False]
    for i in range(3):
        flip_axes[i] = (dwi_affine[i, i] < 0)

    bundle_names = bundle_dict.keys()

    for bundle_name in bundle_names:
        logger.info(f"Generating {bundle_name} visualization...")
        uid = bundle_dict[bundle_name]['uid']
        figure = viz_backend.visualize_volume(
            volume,
            opacity=volume_opacity_indiv,
            flip_axes=flip_axes,
            interact=False,
            inline=False)
        try:
            figure = viz_backend.visualize_bundles(
                segmentation_imap["clean_bundles_file"],
                color_by_volume=color_by_volume,
                cbv_lims=cbv_lims_indiv,
                bundle_dict=bundle_dict,
                bundle=uid,
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
            endpoint_info = segmentation_params["endpoint_info"]
            if endpoint_info is not None:
                start_p = endpoint_info[bundle_name]['startpoint']
                end_p = endpoint_info[bundle_name]['endpoint']
                for pp in [start_p, end_p]:
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

        for i, roi in enumerate(mapping_imap["rois_file"][bundle_name]):
            figure = viz_backend.visualize_roi(
                roi,
                name=f"{bundle_name} ROI {i}",
                flip_axes=flip_axes,
                inline=False,
                interact=False,
                figure=figure)

        roi_dir = op.join(subses_dict['results_dir'], 'viz_bundles')
        os.makedirs(roi_dir, exist_ok=True)
        if "no_gif" not in viz_backend.backend:
            fname = op.split(
                get_fname(
                    subses_dict,
                    f'_{bundle_name}'
                    f'_viz.gif',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            viz_backend.create_gif(figure, fname)
        if "plotly" in viz_backend.backend:
            roi_dir = op.join(subses_dict['results_dir'], 'viz_bundles')
            os.makedirs(roi_dir, exist_ok=True)
            fname = op.split(
                get_fname(
                    subses_dict,
                    f'_{bundle_name}'
                    f'_viz.html',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))

            fname = op.join(roi_dir, fname[1])
            figure.write_html(fname)
    meta_fname = get_fname(
        subses_dict, '_vizIndiv.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    afd.write_json(meta_fname, meta)
    return True


@pimms.calc("tract_profile_plots")
def plot_tract_profiles(subses_dict, scalars, tracking_params,
                        segmentation_params, segmentation_imap):
    start_time = time()
    fnames = []
    for scalar in scalars:
        if not isinstance(scalar, str):
            this_scalar = scalar.name
        else:
            this_scalar = scalar
        fname = get_fname(
            subses_dict, f'_{this_scalar}_profile_plots',
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
            segmentation_imap["profiles_file"],
            scalar=this_scalar,
            file_name=fname,
            n_boot=100)
        fnames.append(fname)
    meta_fname = get_fname(
        subses_dict, '_profile_plots.json',
        tracking_params=tracking_params,
        segmentation_params=segmentation_params)
    meta = dict(Timing=time() - start_time)
    afd.write_json(meta_fname, meta)

    return fnames


def get_viz_plan():
    viz_tasks = with_name([
        plot_tract_profiles, viz_bundles, viz_indivBundle])
    return pimms.plan(**viz_tasks)
