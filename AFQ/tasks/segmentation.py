import nibabel as nib
import os
import os.path as op
from time import time
import numpy as np
import pandas as pd
import logging

import pimms

from AFQ.tasks.decorators import as_file, as_img
from AFQ.tasks.utils import get_fname, with_name, str_to_desc
import AFQ.segmentation as seg
from AFQ.utils.path import drop_extension
import AFQ.utils.streamlines as aus
from AFQ.tasks.utils import get_default_args
from AFQ.data.s3bids import write_json
import AFQ.api.bundle_dict as abd
import AFQ.utils.streamlines as aus
import AFQ.utils.volume as auv

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.tracking.streamline import set_number_of_points, values_from_volume


logger = logging.getLogger('AFQ')


@pimms.calc("bundles")
@as_file('_tractography.trk', include_track=True, include_seg=True)
def segment(dwi, data_imap, mapping_imap,
            tractography_imap, segmentation_params):
    """
    full path to a trk file containing containting
    segmented streamlines, labeled by bundle

    Parameters
    ----------
    segmentation_params : dict, optional
        The parameters for segmentation.
        Default: use the default behavior of the seg.Segmentation object.
    """
    bundle_dict = data_imap["bundle_dict"]
    reg_template = data_imap["reg_template"]
    streamlines = tractography_imap["streamlines"]
    # We pass `clean_params` here, but do not use it, so we have the
    # same signature as `_clean_bundles`.
    img = nib.load(dwi)
    tg = load_tractogram(
        streamlines, img, Space.VOX,
        bbox_valid_check=False)
    indices_to_remove, _ = tg.remove_invalid_streamlines()
    if len(indices_to_remove) > 0:
        logger.warning(f"{len(indices_to_remove)} invalid streamlines removed")

    start_time = time()
    segmentation = seg.Segmentation(**segmentation_params)
    bundles = segmentation.segment(
        bundle_dict,
        tg,
        dwi,
        data_imap["bval"],
        data_imap["bvec"],
        reg_template=reg_template,
        mapping=mapping_imap["mapping"])

    seg_sft = aus.SegmentedSFT(bundles, Space.VOX)

    if len(seg_sft.sft) < 1:
        raise ValueError("Fatal: No bundles recognized.")

    tgram, meta = seg_sft.get_sft_and_sidecar()

    segmentation_params_out = {
        arg_name: value if isinstance(value, (int, float, bool, str)) or (
            value is None) else str(value)
        for arg_name, value in segmentation_params.items()}

    meta["source"] = streamlines
    meta["Parameters"] = segmentation_params_out
    meta["Timing"] = time() - start_time
    return tgram, meta


@pimms.calc("clean_bundles")
@as_file('_desc-clean_tractography.trk', include_track=True, include_seg=True)
def clean_bundles(bundles, data_imap, clean_params=None):
    """
    full path to a trk file containting segmented
    streamlines, cleaned using the Mahalanobis distance, and labeled by
    bundle

    Parameters
    ----------
    clean_params: dict, optional
        The parameters for cleaning.
        Default: use the default behavior of the seg.clean_bundle
        function.
    """
    bundle_dict = data_imap["bundle_dict"]
    default_clean_params = get_default_args(seg.clean_bundle)
    if clean_params is not None:
        for k in clean_params:
            default_clean_params[k] = clean_params[k]
    clean_params = default_clean_params

    seg_sft = aus.SegmentedSFT.fromfile(bundles)

    start_time = time()

    bundles = {}
    for b in bundle_dict.keys():
        if b != "whole_brain":
            idx = seg_sft.bundle_idxs[b]
            this_tg = seg_sft.get_bundle(b)
            this_tg = seg.clean_bundle(this_tg, **clean_params)
            if clean_params['return_idx']:
                bundles[b] = {}
                bundles[b]['sl'], bundles[b]['idx'] = this_tg
                bundles[b]['idx'] = np.array(
                    idx)[bundles[b]['idx']].tolist()
            else:
                bundles[b] = this_tg

    sft, meta = aus.SegmentedSFT(
        bundles, Space.RASMM).get_sft_and_sidecar()

    seg_args = get_default_args(seg.clean_bundle)
    for k in seg_args:
        if callable(seg_args[k]):
            seg_args[k] = seg_args[k].__name__

    meta["source"] = bundles
    meta["Parameters"] = seg_args
    meta["Timing"] = time() - start_time
    return sft, meta


@pimms.calc("indiv_bundles")
def export_bundles(base_fname, results_dir,
                   clean_bundles, bundles,
                   data_imap, tracking_params,
                   segmentation_params):
    """
    dictionary of paths, where each path is
    a full path to a trk file containing the streamlines of a given bundle,
    cleaned or uncleaned
    """
    bundle_dict = data_imap["bundle_dict"]
    reg_template = data_imap["reg_template"]
    if "presegment_bundle_dict" in segmentation_params and\
        segmentation_params["presegment_bundle_dict"] is not None\
        and not isinstance(
            segmentation_params["presegment_bundle_dict"],
            abd.BundleDict):
        segmentation_params["presegment_bundle_dict"] =\
            abd.BundleDict(
                segmentation_params["presegment_bundle_dict"],
                seg_algo="afq",
                resample_to=reg_template)

    for this_bundles_file, folder in zip([clean_bundles, bundles],
                                         ['clean_bundles', 'bundles']):
        bundles_dir = op.join(results_dir, folder)
        os.makedirs(bundles_dir, exist_ok=True)
        seg_sft = aus.SegmentedSFT.fromfile(this_bundles_file)
        for bundle in bundle_dict:
            if bundle != "whole_brain":
                fname = op.split(
                    get_fname(
                        base_fname,
                        f'_desc-{str_to_desc(bundle)}'
                        f'_tractography.trk',
                        tracking_params=tracking_params,
                        segmentation_params=segmentation_params))
                fname = op.join(bundles_dir, fname[1])
                logger.info(f"Saving {fname}")
                save_tractogram(
                    seg_sft.get_bundle(bundle), fname,
                    bbox_valid_check=False)
                meta = dict(source=this_bundles_file)
                meta_fname = drop_extension(fname) + '.json'
                write_json(meta_fname, meta)
    return True


@pimms.calc("sl_counts")
@as_file('_desc-slCount_dwi.csv', include_track=True, include_seg=True)
def export_sl_counts(data_imap,
                     clean_bundles, bundles):
    """
    full path to a JSON file containing streamline counts
    """
    bundle_dict = data_imap["bundle_dict"]
    sl_counts_clean = []
    sl_counts = []
    bundle_names = list(bundle_dict.keys())
    if "whole_brain" not in bundle_names:
        bundle_names.append("whole_brain")
    bundles_files = [clean_bundles, bundles]
    lists = [sl_counts_clean, sl_counts]

    for bundles_file, count in zip(bundles_files, lists):
        seg_sft = aus.SegmentedSFT.fromfile(bundles_file)

        for bundle in bundle_names:
            if bundle == "whole_brain":
                count.append(len(seg_sft.sft.streamlines))
            else:
                count.append(len(
                    seg_sft.get_bundle(bundle).streamlines))
    counts_df = pd.DataFrame(
        data=dict(
            n_streamlines=sl_counts,
            n_streamlines_clean=sl_counts_clean),
        index=bundle_names)
    return counts_df, dict(sources=bundles_files)


@pimms.calc("median_bundle_lengths")
@as_file(
    '_desc-medianBundleLengths_dwi.csv',
    include_track=True, include_seg=True)
def export_bundle_lengths(data_imap,
                          clean_bundles, bundles):
    """
    full path to a JSON file containing median bundle lengths
    """
    bundle_dict = data_imap["bundle_dict"]
    med_len_clean_counts = []
    med_len_counts = []
    bundle_names = list(bundle_dict.keys())
    if "whole_brain" not in bundle_names:
        bundle_names.append("whole_brain")
    bundles_files = [clean_bundles, bundles]
    lists = [med_len_clean_counts, med_len_counts]

    for bundles_file, lens in zip(bundles_files, lists):
        seg_sft = aus.SegmentedSFT.fromfile(bundles_file)

        for bundle in bundle_names:
            if bundle == "whole_brain":
                lens.append(np.median(
                    seg_sft.sft._tractogram._streamlines._lengths))
            else:
                these_lengths = seg_sft.get_bundle(
                    bundle)._tractogram._streamlines._lengths
                if len(these_lengths) > 0:
                    lens.append(np.median(
                        these_lengths))
                else:
                    lens.append(0)

    counts_df = pd.DataFrame(
        data=dict(
            median_len=med_len_counts,
            median_len_clean=med_len_clean_counts),
        index=bundle_names)
    return counts_df, dict(sources=bundles_files)


@pimms.calc("density_maps")
@as_file('_desc-density_dwi.nii.gz', include_track=True, include_seg=True)
@as_img
def export_density_maps(clean_bundles, dwi, data_imap):
    """
    full path to 4d nifti file containing streamline counts per voxel
    per bundle, where the 4th dimension encodes the bundle
    """
    bundle_dict = data_imap["bundle_dict"]
    seg_sft = aus.SegmentedSFT.fromfile(
        clean_bundles)
    entire_density_map = np.zeros((*nib.load(dwi).shape[:3], len(bundle_dict)))
    for ii, bundle_name in enumerate(bundle_dict.keys()):
        bundle_sl = seg_sft.get_bundle(bundle_name)
        bundle_density = auv.density_map(bundle_sl).get_fdata()
        entire_density_map[..., ii] = bundle_density

    return entire_density_map, dict(
        source=clean_bundles, bundles=list(bundle_dict.keys()))


@pimms.calc("profiles")
@as_file('_desc-profiles_dwi.csv', include_track=True, include_seg=True)
def tract_profiles(clean_bundles, data_imap,
                   scalar_dict, dwi_affine,
                   profile_weights="gauss"):
    """
    full path to a CSV file containing tract profiles

    Parameters
    ----------
    profile_weights : str, 1D array, 2D array callable, optional
        How to weight each streamline (1D) or each node (2D)
        when calculating the tract-profiles. If callable, this is a
        function that calculates weights. If None, no weighting will
        be applied. If "gauss", gaussian weights will be used.
        If "median", the median of values at each node will be used
        instead of a mean or weighted mean.
        Default: "gauss"
    """
    bundle_dict = data_imap["bundle_dict"]
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

    bundle_names = []
    node_numbers = []
    profiles = np.empty((len(scalar_dict), 0)).tolist()
    this_profile = np.zeros((len(scalar_dict), 100))

    seg_sft = aus.SegmentedSFT.fromfile(
        clean_bundles)
    seg_sft.sft.to_rasmm()
    for bundle_name in bundle_dict.keys():
        this_sl = seg_sft.get_bundle(bundle_name).streamlines
        if len(this_sl) == 0:
            continue
        if profile_weights == "gauss":
            # calculate only once per bundle
            bundle_profile_weights = gaussian_weights(this_sl)
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
                        fgarray = set_number_of_points(bundle, 100)
                        values = np.array(
                            values_from_volume(
                                scalar_data,
                                fgarray,
                                dwi_affine))
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
                dwi_affine,
                weights=this_prof_weights)
            profiles[ii].extend(list(this_profile[ii]))
        nodes = list(np.arange(this_profile[0].shape[0]))
        bundle_names.extend([bundle_name] * len(nodes))
        node_numbers.extend(nodes)

    profile_dict = dict()
    profile_dict["tractID"] = bundle_names
    profile_dict["nodeID"] = node_numbers
    for ii, scalar in enumerate(scalar_dict.keys()):
        profile_dict[scalar] = profiles[ii]

    profile_dframe = pd.DataFrame(profile_dict)
    meta = dict(source=clean_bundles,
                parameters=get_default_args(afq_profile))

    return profile_dframe, meta


@pimms.calc("scalar_dict")
def get_scalar_dict(data_imap, mapping_imap, scalars=["dti_fa", "dti_md"]):
    """
    dicionary mapping scalar names
    to their respective file paths

    Parameters
    ----------
    scalars : list of strings and/or scalar definitions, optional
        List of scalars to use.
        Can be any of: "dti_fa", "dti_md", "dki_fa", "dki_md", "dki_awf",
        "dki_mk". Can also be a scalar from AFQ.definitions.image.
        Default: ["dti_fa", "dti_md"]
    """
    # Note: some scalars preprocessing done in plans, before this step
    scalar_dict = {}
    for scalar in scalars:
        if isinstance(scalar, str):
            sc = scalar.lower()
            scalar_dict[sc] = data_imap[f"{sc}"]
        else:
            scalar_dict[scalar.get_name()] = mapping_imap[
                f"{scalar.get_name()}"]
    return {"scalar_dict": scalar_dict}


def get_segmentation_plan(kwargs):
    if "segmentation_params" in kwargs\
            and not isinstance(kwargs["segmentation_params"], dict):
        raise TypeError(
            "segmentation_params a dict")
    segmentation_tasks = with_name([
        get_scalar_dict,
        export_sl_counts,
        export_bundle_lengths,
        export_bundles,
        export_density_maps,
        clean_bundles,
        segment,
        tract_profiles])

    default_seg_params = get_default_args(seg.Segmentation.__init__)
    if "segmentation_params" in kwargs:
        for k in kwargs["segmentation_params"]:
            default_seg_params[k] = kwargs["segmentation_params"][k]

    kwargs["segmentation_params"] = default_seg_params
    return pimms.plan(**segmentation_tasks)
