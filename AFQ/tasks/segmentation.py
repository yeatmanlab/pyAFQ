import nibabel as nib
import os
import os.path as op
from time import time
import numpy as np
import pandas as pd
import logging

import pimms

from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import (
    get_fname, with_name, str_to_desc, get_default_args)
import AFQ.segmentation as seg
from AFQ.utils.path import drop_extension, write_json
import AFQ.utils.streamlines as aus
import AFQ.utils.volume as auv

try:
    from trx.io import load as load_trx
    from trx.io import save as save_trx
    from trx.trx_file_memmap import TrxFile
    has_trx = True
except ModuleNotFoundError:
    has_trx = False

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import Space


logger = logging.getLogger('AFQ')


@pimms.calc("bundles")
@as_file('_tractography', include_track=True, include_seg=True)
def segment(data_imap, mapping_imap,
            tractography_imap, segmentation_params):
    """
    full path to a trk/trx file containing containing
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
    if streamlines.endswith(".trk") or streamlines.endswith(".tck"):
        tg = load_tractogram(
            streamlines, data_imap["dwi"], Space.VOX,
            bbox_valid_check=False)
        is_trx = False
    elif streamlines.endswith(".trx"):
        is_trx = True
        trx = load_trx(streamlines, data_imap["dwi"])
        trx.streamlines._data = trx.streamlines._data.astype(np.float32)
        tg = trx.to_sft()

    indices_to_remove, _ = tg.remove_invalid_streamlines()
    if len(indices_to_remove) > 0:
        logger.warning(f"{len(indices_to_remove)} invalid streamlines removed")

    start_time = time()
    segmentation = seg.Segmentation(**segmentation_params)
    bundles = segmentation.segment(
        bundle_dict,
        tg,
        mapping_imap["mapping"],
        data_imap["dwi"],
        reg_template=reg_template)

    seg_sft = aus.SegmentedSFT(bundles, Space.VOX)

    if len(seg_sft.sft) < 1:
        raise ValueError("Fatal: No bundles recognized.")

    if is_trx:
        seg_sft.sft.dtype_dict = {'positions': np.float16,
                                  'offsets': np.uint32}
        tgram = TrxFile.from_sft(seg_sft.sft)
        tgram.groups = seg_sft.bundle_idxs
        meta = {}

    else:
        tgram, meta = seg_sft.get_sft_and_sidecar()

    seg_params_out = {}
    for arg_name, value in segmentation_params.items():
        if isinstance(value, (int, float, bool, str)):
            seg_params_out[arg_name] = value
        elif isinstance(value, (list, tuple)):
            seg_params_out[arg_name] = [str(v) for v in value]
        elif isinstance(value, dict):
            for k, v in value.items():
                seg_params_out[k] = str(v)
        else:
            seg_params_out[arg_name] = str(value)

    meta["source"] = streamlines
    meta["Recognition Parameters"] = seg_params_out
    meta["Timing"] = time() - start_time
    return tgram, meta


@pimms.calc("indiv_bundles")
def export_bundles(base_fname, results_dir,
                   bundles,
                   tracking_params,
                   segmentation_params):
    """
    dictionary of paths, where each path is
    a full path to a trk file containing the streamlines of a given bundle.
    """
    is_trx = tracking_params.get("trx", False)
    if is_trx:
        extension = ".trx"
    else:
        extension = ".trk"

    bundles_dir = op.join(results_dir, "bundles")
    os.makedirs(bundles_dir, exist_ok=True)
    seg_sft = aus.SegmentedSFT.fromfile(bundles)
    for bundle in seg_sft.bundle_names:
        if bundle != "whole_brain":
            fname = op.split(
                get_fname(
                    base_fname,
                    f'_desc-{str_to_desc(bundle)}'
                    f'_tractography{extension}',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params))
            fname = op.join(bundles_dir, fname[1])
            bundle_sft = seg_sft.get_bundle(bundle)
            if len(bundle_sft) > 0:
                logger.info(f"Saving {fname}")
                if is_trx:
                    seg_sft.sft.dtype_dict = {
                        'positions': np.float16,
                        'offsets': np.uint32}
                    trxfile = TrxFile.from_sft(bundle_sft)
                    save_trx(trxfile, fname)
                else:
                    save_tractogram(
                        bundle_sft, fname,
                        bbox_valid_check=False)
            else:
                logger.info(f"No bundle to save for {bundle}")
            meta = dict(source=bundles)
            meta_fname = drop_extension(fname) + '.json'
            write_json(meta_fname, meta)
    return bundles_dir


@pimms.calc("sl_counts")
@as_file('_desc-slCount_dwi.csv', include_track=True, include_seg=True)
def export_sl_counts(bundles):
    """
    full path to a JSON file containing streamline counts
    """
    sl_counts = []
    seg_sft = aus.SegmentedSFT.fromfile(bundles)

    for bundle in seg_sft.bundle_names:
        sl_counts.append(len(
            seg_sft.get_bundle(bundle).streamlines))
    sl_counts.append(len(seg_sft.sft.streamlines))

    counts_df = pd.DataFrame(
        data=dict(
            n_streamlines=sl_counts),
        index=seg_sft.bundle_names + ["Total Recognized"])
    return counts_df, dict(source=bundles)


@pimms.calc("median_bundle_lengths")
@as_file(
    '_desc-medianBundleLengths_dwi.csv',
    include_track=True, include_seg=True)
def export_bundle_lengths(bundles):
    """
    full path to a JSON file containing median bundle lengths
    """
    med_len_counts = []
    seg_sft = aus.SegmentedSFT.fromfile(bundles)

    for bundle in seg_sft.bundle_names:
        these_lengths = seg_sft.get_bundle(
            bundle)._tractogram._streamlines._lengths
        if len(these_lengths) > 0:
            med_len_counts.append(np.median(
                these_lengths))
        else:
            med_len_counts.append(0)
    med_len_counts.append(np.median(
        seg_sft.sft._tractogram._streamlines._lengths))

    counts_df = pd.DataFrame(
        data=dict(
            median_len=med_len_counts),
        index=seg_sft.bundle_names + ["Total Recognized"])
    return counts_df, dict(source=bundles)


@pimms.calc("density_maps")
@as_file('_desc-density_dwi.nii.gz', include_track=True, include_seg=True)
def export_density_maps(bundles, data_imap):
    """
    full path to 4d nifti file containing streamline counts per voxel
    per bundle, where the 4th dimension encodes the bundle
    """
    seg_sft = aus.SegmentedSFT.fromfile(
        bundles)
    entire_density_map = np.zeros((
        *data_imap["data"].shape[:3],
        len(seg_sft.bundle_names)))
    for ii, bundle_name in enumerate(seg_sft.bundle_names):
        bundle_sl = seg_sft.get_bundle(bundle_name)
        bundle_density = auv.density_map(bundle_sl).get_fdata()
        entire_density_map[..., ii] = bundle_density

    return nib.Nifti1Image(
        entire_density_map, data_imap["dwi_affine"]), dict(
            source=bundles, bundles=list(seg_sft.bundle_names))


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
        elif f"{scalar.get_name()}" in mapping_imap:
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
        segment])

    default_seg_params = get_default_args(seg.Segmentation.__init__)
    if "segmentation_params" in kwargs:
        for k in kwargs["segmentation_params"]:
            default_seg_params[k] = kwargs["segmentation_params"][k]

    kwargs["segmentation_params"] = default_seg_params
    return pimms.plan(**segmentation_tasks)
