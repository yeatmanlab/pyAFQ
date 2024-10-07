import numpy as np
import os.path as op
import os
import logging

import dipy.tracking.streamlinespeed as dps
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.recognition.utils as abu
from AFQ.api.bundle_dict import BundleDict
from AFQ.recognition.criteria import run_bundle_rec_plan
from AFQ.recognition.preprocess import get_preproc_plan


logger = logging.getLogger('AFQ')


def recognize(
        tg,
        img,
        mapping,
        bundle_dict,
        reg_template,
        nb_points=False,
        nb_streamlines=False,
        clip_edges=False,
        parallel_segmentation={"engine": "serial"},
        rb_recognize_params=dict(
            model_clust_thr=1.25,
            reduction_thr=25,
            pruning_thr=12),
        refine_reco=False,
        prob_threshold=0,
        dist_to_waypoint=None,
        rng=None,
        return_idx=False,
        filter_by_endpoints=True,
        dist_to_atlas=4,
        save_intermediates=None,
        cleaning_params={}):
    """
    Segment streamlines into bundles.

    Parameters
    ----------
    tg : str, StatefulTractogram
        Tractogram to segment.
    img : str, nib.Nifti1Image
        Image for reference.
    mapping : MappingDefinition
        Mapping from subject to template.
    bundle_dict : dict or AFQ.api.BundleDict
        Dictionary of bundles to segment.
    reg_template : str, nib.Nifti1Image
        Template image for registration.
    nb_points : int, boolean
        Resample streamlines to nb_points number of points.
        If False, no resampling is done. Default: False
    nb_streamlines : int, boolean
        Subsample streamlines to nb_streamlines.
        If False, no subsampling is don. Default: False
    clip_edges : bool
        Whether to clip the streamlines to be only in between the ROIs.
        Default: False
    parallel_segmentation : dict or AFQ.api.BundleDict
        How to parallelize segmentation across processes when performing
        waypoint ROI segmentation. Set to {"engine": "serial"} to not
        perform parallelization. Some engines may cause errors, depending
        on the system. See ``dipy.utils.parallel.paramap`` for
        details.
        Default: {"engine": "serial"}
    rb_recognize_params : dict
        RecoBundles parameters for the recognize function.
        Default: dict(model_clust_thr=1.25, reduction_thr=25, pruning_thr=12)
    refine_reco : bool
        Whether to refine the RecoBundles segmentation.
        Default: False
    prob_threshold : float.
        Using AFQ Algorithm.
        Initial cleaning of fiber groups is done using probability maps
        from [Hua2008]_. Here, we choose an average probability that
        needs to be exceeded for an individual streamline to be retained.
        Default: 0.
    dist_to_waypoint : float.
        The distance that a streamline node has to be from the waypoint
        ROI in order to be included or excluded.
        If set to None (default), will be calculated as the
        center-to-corner distance of the voxel in the diffusion data.
        If a bundle has inc_addtol or exc_addtol in its bundle_dict, that
        tolerance will be added to this distance.
        For example, if you wanted to increase tolerance for the right
        arcuate waypoint ROIs by 3 each, you could make the following
        modification to your bundle_dict:
        bundle_dict["Right Arcuate"]["inc_addtol"] = [3, 3]
        Additional tolerances can also be negative.
        Default: None.
    rng : RandomState or int
        If None, creates RandomState.
        If int, creates RandomState with seed rng.
        Used in RecoBundles Algorithm.
        Default: None.
    return_idx : bool
        Whether to return the indices in the original streamlines as part
        of the output of segmentation.
        Default: False.
    filter_by_endpoints: bool
        Whether to filter the bundles based on their endpoints.
        Default: True.
    dist_to_atlas : float
        If filter_by_endpoints is True, this is the required distance
        from the endpoints to the atlas ROIs.
        Default: 4
    save_intermediates : str, optional
        The full path to a folder into which intermediate products
        are saved. Default: None, means no saving of intermediates.
    cleaning_params : dict, optional
        Cleaning params to pass to seg.clean_bundle. This will
        override the default parameters of that method. However, this
        can be overriden by setting the cleaning parameters in the
        bundle_dict. Default: {}.

    References
    ----------
    .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
    Tract probability maps in stereotaxic spaces: analyses of white
    matter anatomy and tract-specific quantification. Neuroimage 39:
    336-347
    .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty, Nathaniel J.
    Myall, Brian A. Wandell, and Heidi M. Feldman. 2012. "Tract Profiles of
    White Matter Properties: Automating Fiber-Tract Quantification"
    PloS One 7 (11): e49790.
    .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
    bundles using local and global streamline-based registration and
    clustering, Neuroimage, 2017.
    """
    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    if (save_intermediates is not None) and \
            (not op.exists(save_intermediates)):
        os.makedirs(save_intermediates, exist_ok=True)

    logger.info("Preprocessing Streamlines")
    tg = abu.read_tg(tg, nb_streamlines)

    # If resampling over-write the sft:
    if nb_points:
        tg = StatefulTractogram(
            dps.set_number_of_points(tg.streamlines, nb_points),
            tg, tg.space)

    if not isinstance(bundle_dict, BundleDict):
        bundle_dict = BundleDict(bundle_dict)

    tg.to_vox()
    n_streamlines = len(tg)
    bundle_decisions = np.zeros(
        (n_streamlines, len(bundle_dict)),
        dtype=np.bool8)
    bundle_to_flip = np.zeros(
        (n_streamlines, len(bundle_dict)),
        dtype=np.bool8)
    bundle_roi_dists = -np.ones(
        (
            n_streamlines,
            len(bundle_dict),
            bundle_dict.max_includes),
        dtype=np.uint32)

    fiber_groups = {}
    meta = {}

    preproc_imap = get_preproc_plan(img, tg, dist_to_waypoint, dist_to_atlas)

    logger.info("Assigning Streamlines to Bundles")
    for bundle_idx, bundle_name in enumerate(
            bundle_dict.bundle_names):
        logger.info(f"Finding Streamlines for {bundle_name}")
        run_bundle_rec_plan(
            bundle_dict, tg, mapping, img, reg_template, preproc_imap,
            bundle_name, bundle_idx, bundle_to_flip, bundle_roi_dists,
            bundle_decisions,
            clip_edges=clip_edges,
            parallel_segmentation=parallel_segmentation,
            rb_recognize_params=rb_recognize_params,
            prob_threshold=prob_threshold,
            refine_reco=refine_reco,
            rng=rng,
            return_idx=return_idx,
            filter_by_endpoints=filter_by_endpoints,
            save_intermediates=save_intermediates,
            cleaning_params=cleaning_params)

    if save_intermediates is not None:
        os.makedirs(save_intermediates, exist_ok=True)
        bc_path = op.join(save_intermediates,
                          "sls_bundle_decisions.npy")
        np.save(bc_path, bundle_decisions)

    conflicts = np.sum(np.sum(bundle_decisions, axis=1) > 1)
    if conflicts > 0:
        logger.warning((
            "Conflicts in bundle assignment detected. "
            f"{conflicts} conflicts detected in total out of "
            f"{n_streamlines} total streamlines."
            "Defaulting to whichever bundle appears first"
            "in the bundle_dict."))
    bundle_decisions = np.concatenate((
        bundle_decisions, np.ones((n_streamlines, 1))), axis=1)
    bundle_decisions = np.argmax(bundle_decisions, -1)

    # We do another round through, so that we can:
    # 1. Clip streamlines according to ROIs
    # 2. Re-orient streamlines
    logger.info("Re-orienting streamlines to consistent directions")
    for bundle_idx, bundle in enumerate(bundle_dict.bundle_names):
        logger.info(f"Processing {bundle}")

        select_idx = np.where(bundle_decisions == bundle_idx)[0]

        if len(select_idx) == 0:
            # There's nothing here, set and move to the next bundle:
            if "bundlesection" in bundle_dict.get_b_info(bundle):
                for sb_name in bundle_dict.get_b_info(bundle)[
                        "bundlesection"]:
                    _return_empty(sb_name, return_idx, fiber_groups, img)
            else:
                _return_empty(bundle, return_idx, fiber_groups, img)
            continue

        # Use a list here, because ArraySequence doesn't support item
        # assignment:
        select_sl = list(tg.streamlines[select_idx])
        roi_dists = bundle_roi_dists[select_idx, bundle_idx, :]
        n_includes = len(bundle_dict.get_b_info(
            bundle).get("include", []))
        if clip_edges and n_includes > 1:
            logger.info("Clipping Streamlines by ROI")
            select_sl = abu.cut_sls_by_dist(
                select_sl, roi_dists,
                (0, n_includes - 1), in_place=True)

        to_flip = bundle_to_flip[select_idx, bundle_idx]
        b_def = dict(bundle_dict.get_b_info(bundle_name))
        if "bundlesection" in b_def:
            for sb_name, sb_include_cuts in bundle_dict.get_b_info(
                    bundle)["bundlesection"].items():
                bundlesection_select_sl = abu.cut_sls_by_dist(
                    select_sl, roi_dists,
                    sb_include_cuts, in_place=False)
                _add_bundle_to_fiber_group(
                    sb_name, bundlesection_select_sl, select_idx,
                    to_flip, return_idx, fiber_groups, img)
                _add_bundle_to_meta(sb_name, b_def, meta)
        else:
            _add_bundle_to_fiber_group(
                bundle, select_sl, select_idx, to_flip,
                return_idx, fiber_groups, img)
            _add_bundle_to_meta(bundle, b_def, meta)
    return fiber_groups, meta


# Helper functions for formatting the results
def _return_empty(bundle_name, return_idx, fiber_groups, img):
    """
    Helper function to return an empty dict under
    some conditions.
    """
    if return_idx:
        fiber_groups[bundle_name] = {}
        fiber_groups[bundle_name]['sl'] = StatefulTractogram(
            [], img, Space.VOX)
        fiber_groups[bundle_name]['idx'] = np.array([])
    else:
        fiber_groups[bundle_name] = StatefulTractogram(
            [], img, Space.VOX)


def _add_bundle_to_fiber_group(b_name, sl, idx, to_flip,
                               return_idx, fiber_groups, img):
    """
    Helper function to add a bundle to a fiber group.
    """
    sl = abu.flip_sls(
        sl, to_flip,
        in_place=False)

    sl = StatefulTractogram(
        sl,
        img,
        Space.VOX)

    if return_idx:
        fiber_groups[b_name] = {}
        fiber_groups[b_name]['sl'] = sl
        fiber_groups[b_name]['idx'] = idx
    else:
        fiber_groups[b_name] = sl


def _add_bundle_to_meta(bundle_name, b_def, meta):
    # remove keys that can never be serialized
    for key in [
            'include', 'exclude', 'prob_map',
            'start', 'end', 'curvature']:
        b_def.pop(key, None)
    meta[bundle_name] = b_def
