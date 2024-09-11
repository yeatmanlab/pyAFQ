import numpy as np
import logging
from time import time

import dipy.tracking.streamline as dts
from dipy.utils.parallel import paramap
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature
from dipy.io.streamline import load_tractogram
from dipy.segment.bundles import RecoBundles
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.recognition.utils as abu
import AFQ.recognition.cleaning as abc
import AFQ.recognition.curvature as abv
import AFQ.recognition.roi as abr

bundle_criterion_order = [
    "prob_map", "cross_midline", "start", "end",
    "length", "primary_axis", "include", "exclude",
    "recobundles", "qb_thresh"]


logger = logging.getLogger('AFQ')


def prob_map(b_sls, bundle_def, preproc_imap, prob_threshold, **kwargs):
    b_sls.initiate_selection("Prob. Map")
    # using entire fgarray here only because it is the first step
    fiber_probabilities = dts.values_from_volume(
        bundle_def["prob_map"].get_fdata(),
        preproc_imap["fgarray"], np.eye(4))
    fiber_probabilities = np.mean(fiber_probabilities, -1)
    b_sls.select(
        fiber_probabilities > prob_threshold,
        "Prob. Map")


def cross_midline(b_sls, bundle_def, preproc_imap, **kwargs):
    b_sls.initiate_selection("Cross Mid.")
    accepted = preproc_imap["crosses"][b_sls.selected_fiber_idxs]
    if not bundle_def["cross_midline"]:
        accepted = np.invert(accepted)
    b_sls.select(accepted, "Cross Mid.")


def start(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("Startpoint")
    abr.clean_by_endpoints(
        b_sls.get_selected_sls(),
        bundle_def["start"],
        0,
        tol=preproc_imap["dist_to_atlas"],
        flip_sls=b_sls.sls_flipped,
        accepted_idxs=accept_idx)
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            b_sls.get_selected_sls(),
            bundle_def["start"],
            -1,
            tol=preproc_imap["dist_to_atlas"])
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = np.logical_xor(
            accepted_idx_flipped, accept_idx)
    b_sls.select(accept_idx, "Startpoint")


def end(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("endpoint")
    abr.clean_by_endpoints(
        b_sls.get_selected_sls(),
        bundle_def["end"],
        -1,
        tol=preproc_imap["dist_to_atlas"],
        flip_sls=b_sls.sls_flipped,
        accepted_idxs=accept_idx)
    if not b_sls.oriented_yet:
        accepted_idx_flipped = abr.clean_by_endpoints(
            b_sls.get_selected_sls(),
            bundle_def["end"],
            0,
            tol=preproc_imap["dist_to_atlas"])
        b_sls.reorient(accepted_idx_flipped)
        accept_idx = np.logical_xor(
            accepted_idx_flipped, accept_idx)
    b_sls.select(accept_idx, "endpoint")


def length(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("length")
    min_len = bundle_def["length"].get(
        "min_len", 0) / preproc_imap["vox_dim"]
    max_len = bundle_def["length"].get(
        "max_len", np.inf) / preproc_imap["vox_dim"]
    for idx, sl in enumerate(b_sls.get_selected_sls()):
        sl_len = np.sum(
            np.linalg.norm(np.diff(sl, axis=0), axis=1))
        if sl_len >= min_len and sl_len <= max_len:
            accept_idx[idx] = 1
    b_sls.select(accept_idx, "length")


def primary_axis(b_sls, bundle_def, **kwargs):
    b_sls.initiate_selection("orientation")
    accept_idx = abc.clean_by_orientation(
        b_sls.get_selected_sls(),
        bundle_def["primary_axis"],
        bundle_def.get(
            "primary_axis_percentage", None))
    b_sls.select(accept_idx, "orientation")


def include(b_sls, bundle_def, preproc_imap, max_includes,
            parallel_segmentation, **kwargs):
    accept_idx = b_sls.initiate_selection("include")
    flip_using_include = len(bundle_def["include"]) > 1\
        and not b_sls.oriented_yet

    if f'inc_addtol' in bundle_def:
        include_roi_tols = []
        for inc_tol in bundle_def["inc_addtol"]:
            include_roi_tols.append((
                inc_tol / preproc_imap["vox_dim"] + preproc_imap["tol"])**2)
    else:
        include_roi_tols = [preproc_imap["tol"]**2] * len(
            bundle_def["include"])

    include_rois = []
    for include_roi in bundle_def["include"]:
        include_rois.append(np.array(
            np.where(include_roi.get_fdata())).T)

    # with parallel segmentation, the first for loop will
    # only collect streamlines and does not need tqdm
    if parallel_segmentation["engine"] != "serial":
        inc_results = paramap(
            abr.check_sl_with_inclusion, b_sls.get_selected_sls(),
            func_args=[
                include_rois, include_roi_tols],
            **parallel_segmentation)

    else:
        inc_results = abr.check_sls_with_inclusion(
            b_sls.get_selected_sls(),
            include_rois,
            include_roi_tols)

    roi_dists = -np.ones(
        (len(b_sls), max_includes),
        dtype=np.int32)
    if flip_using_include:
        to_flip = np.ones_like(accept_idx, dtype=np.bool8)
    for sl_idx, inc_result in enumerate(inc_results):
        sl_accepted, sl_dist = inc_result

        if sl_accepted:
            if len(sl_dist) > 1:
                roi_dists[sl_idx, :len(sl_dist)] = [
                    np.argmin(dist, 0)[0]
                    for dist in sl_dist]
                first_roi_idx = roi_dists[sl_idx, 0]
                last_roi_idx = roi_dists[
                    sl_idx, len(sl_dist) - 1]
                # Only accept SLs that, when cut, are meaningful
                if (len(sl_dist) < 2) or abs(
                        first_roi_idx - last_roi_idx) > 1:
                    # Flip sl if it is close to second ROI
                    # before its close to the first ROI
                    if flip_using_include:
                        to_flip[sl_idx] =\
                            first_roi_idx > last_roi_idx
                        if to_flip[sl_idx]:
                            roi_dists[sl_idx, :len(sl_dist)] =\
                                np.flip(roi_dists[
                                    sl_idx, :len(sl_dist)])
                    accept_idx[sl_idx] = 1
            else:
                accept_idx[sl_idx] = 1
    # see https://github.com/joblib/joblib/issues/945
    if (
        (parallel_segmentation.get(
            "engine", "joblib") != "serial")
        and (parallel_segmentation.get(
            "backend", "loky") == "loky")):
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    b_sls.roi_dists = roi_dists
    if flip_using_include:
        b_sls.reorient(to_flip)
    b_sls.select(accept_idx, "include")


def curvature(b_sls, bundle_def, mapping, img, save_intermediates, **kwargs):
    '''
    Filters streamlines by how well they match
    a curve in orientation and shape but not scale
    '''
    accept_idx = b_sls.initiate_selection("curvature")
    if "sft" in bundle_def["curvature"]:
        ref_sl = bundle_def["curvature"]["sft"]
    else:
        ref_sl = load_tractogram(
            bundle_def["curvature"]["path"], "same",
            bbox_valid_check=False)
    moved_ref_sl = abu.move_streamlines(
        ref_sl, "subject", mapping, img,
        save_intermediates=save_intermediates)
    moved_ref_sl.to_vox()
    moved_ref_sl = moved_ref_sl.streamlines[0]
    moved_ref_curve = abv.sl_curve(
        moved_ref_sl,
        len(moved_ref_sl))
    ref_curve_threshold = np.radians(bundle_def["curvature"].get(
        "thresh", 10))
    cut = bundle_def["curvature"].get("cut", True)
    for idx, sl in enumerate(b_sls.get_selected_sls(
            cut=cut, flip=True)):
        if len(sl) > 1:
            this_sl_curve = abv.sl_curve(sl, len(moved_ref_sl))
            dist = abv.sl_curve_dist(this_sl_curve, moved_ref_curve)
            if dist <= ref_curve_threshold:
                accept_idx[idx] = 1
    b_sls.select(accept_idx, "curvature", cut=cut)


def exclude(b_sls, bundle_def, preproc_imap, **kwargs):
    accept_idx = b_sls.initiate_selection("exclude")
    if f'exc_addtol' in bundle_def:
        exclude_roi_tols = []
        for exc_tol in bundle_def["exc_addtol"]:
            exclude_roi_tols.append((
                exc_tol / preproc_imap["vox_dim"] + preproc_imap["tol"])**2)
    else:
        exclude_roi_tols = [
            preproc_imap["tol"]**2] * len(bundle_def["exclude"])
    exclude_rois = []
    for exclude_roi in bundle_def["exclude"]:
        exclude_rois.append(np.array(
            np.where(exclude_roi.get_fdata())).T)
    for sl_idx, sl in enumerate(b_sls.get_selected_sls()):
        if abr.check_sl_with_exclusion(
                sl, exclude_rois, exclude_roi_tols):
            accept_idx[sl_idx] = 1
    b_sls.select(accept_idx, "exclude")


def recobundles(b_sls, mapping, bundle_def, reg_template, img, refine_reco,
                save_intermediates, rng, rb_recognize_params, **kwargs):
    b_sls.initiate_selection("Recobundles")
    moved_sl = abu.move_streamlines(
        StatefulTractogram(b_sls.get_selected_sls(), img, Space.VOX),
        "template", mapping, reg_template,
        save_intermediates=save_intermediates).streamlines
    rb = RecoBundles(moved_sl, verbose=True, rng=rng)
    _, rec_labels = rb.recognize(
        bundle_def['recobundles']['sl'],
        **rb_recognize_params)
    if refine_reco:
        _, rec_labels = rb.refine(
            bundle_def['recobundles']['sl'], moved_sl[rec_labels],
            **rb_recognize_params)
    if not b_sls.oriented_yet:
        standard_sl = next(iter(bundle_def['recobundles']['centroid']))
        oriented_idx = abu.orient_by_streamline(
            moved_sl[rec_labels],
            standard_sl)
        b_sls.reorient(rec_labels[oriented_idx])
    b_sls.select(rec_labels, "Recobundles")


def qb_thresh(b_sls, bundle_def, preproc_imap, clip_edges, **kwargs):
    b_sls.initiate_selection("qb_thresh")
    cut = clip_edges or ("bundlesection" in bundle_def)
    qbx = QuickBundles(
        bundle_def["qb_thresh"] / preproc_imap["vox_dim"],
        AveragePointwiseEuclideanMetric(
            ResampleFeature(nb_points=12)))
    clusters = qbx.cluster(b_sls.get_selected_sls(
        cut=cut, flip=True))
    cleaned_idx = clusters[np.argmax(
        clusters.clusters_sizes())].indices
    b_sls.select(cleaned_idx, "qb_thresh", cut=cut)


def mahalanobis(b_sls, bundle_def, clip_edges, cleaning_params, **kwargs):
    b_sls.initiate_selection("Mahalanobis")
    clean_params = bundle_def.get("mahal", {})
    clean_params = {
        **cleaning_params,
        **clean_params}
    clean_params["return_idx"] = True
    cut = clip_edges or ("bundlesection" in bundle_def)
    _, cleaned_idx = abc.clean_bundle(
        b_sls.get_selected_sls(cut=cut, flip=True),
        **clean_params)
    b_sls.select(cleaned_idx, "Mahalanobis", cut=cut)


def run_bundle_rec_plan(
        bundle_dict, tg, mapping, img, reg_template, preproc_imap,
        bundle_name, bundle_idx, bundle_to_flip, bundle_roi_dists,
        bundle_decisions,
        **segmentation_params):
    # Warp ROIs
    logger.info(f"Preparing ROIs for {bundle_name}")
    start_time = time()
    bundle_def = dict(bundle_dict.get_b_info(bundle_name))
    bundle_def.update(bundle_dict.transform_rois(
        bundle_name,
        mapping,
        img.affine,
        apply_to_recobundles=True))
    logger.info(f"Time to prep ROIs: {time()-start_time}s")

    b_sls = abu.SlsBeingRecognized(
        tg.streamlines, logger,
        segmentation_params["save_intermediates"],
        bundle_name,
        img, len(bundle_def.get("include", [])))

    inputs = {}
    inputs["b_sls"] = b_sls
    inputs["preproc_imap"] = preproc_imap
    inputs["bundle_def"] = bundle_def
    inputs["max_includes"] = bundle_dict.max_includes
    inputs["mapping"] = mapping
    inputs["img"] = img
    inputs["reg_template"] = reg_template
    for key, value in segmentation_params.items():
        inputs[key] = value

    for criterion in bundle_criterion_order:
        if b_sls and criterion in bundle_def:
            inputs[criterion] = globals()[criterion](**inputs)
    if b_sls:
        mahalanobis(**inputs)

    if b_sls and not b_sls.oriented_yet:
        raise ValueError(
            "pyAFQ was unable to consistently orient streamlines "
            f"in bundle {bundle_name} using the provided ROIs. "
            "This can be fixed by including at least 2 "
            "waypoint ROIs, or by using "
            "endpoint ROIs.")

    if b_sls:
        bundle_to_flip[
            b_sls.selected_fiber_idxs,
            bundle_idx] = b_sls.sls_flipped.copy()
        bundle_decisions[
            b_sls.selected_fiber_idxs,
            bundle_idx] = 1
        if hasattr(b_sls, "roi_dists"):
            bundle_roi_dists[
                b_sls.selected_fiber_idxs,
                bundle_idx
            ] = b_sls.roi_dists.copy()
