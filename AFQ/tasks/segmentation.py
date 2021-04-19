import nibabel as nib
import os
import os.path as op
from time import time
import json
import numpy as np
import pandas as pd
import logging

import pimms

from AFQ.tasks.utils import as_file, get_fname
import AFQ.segmentation as seg
import AFQ.utils.streamlines as aus
from AFQ.utils.bin import get_default_args
import AFQ.data as afd

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy.tracking.utils as dtu


logger = logging.getLogger('AFQ.api.seg')


@pimms.calc("bundles_file")
@as_file('_tractography.trk', include_track=True, include_seg=True)
def segment(subses_dict, streamlines_file, bundle_dict,
            bval_file, bvec_file, reg_template, mapping,
            tracking_params, segmentation_params):
    # We pass `clean_params` here, but do not use it, so we have the
    # same signature as `_clean_bundles`.
    img = nib.load(subses_dict['dwi_file'])
    tg = load_tractogram(
        streamlines_file, img, Space.VOX,
        bbox_valid_check=False)
    tg.remove_invalid_streamlines()

    start_time = time()
    segmentation = seg.Segmentation(**segmentation_params)
    bundles = segmentation.segment(
        bundle_dict,
        tg,
        subses_dict['dwi_file'],
        bval_file,
        bvec_file,
        reg_template=reg_template,
        mapping=mapping)

    if segmentation_params['return_idx']:
        idx = {bundle: bundles[bundle]['idx'].tolist()
               for bundle in bundle_dict}
        bundles = {bundle: bundles[bundle]['sl']
                   for bundle in bundle_dict}

    tgram = aus.bundles_to_tgram(bundles, bundle_dict, img)

    segmentation_params_out = {}
    for arg_name, value in segmentation_params.items():
        if isinstance(value, (int, float, bool, str))\
                or (value is None):
            segmentation_params_out[arg_name] = value
        else:
            segmentation_params_out[arg_name] = str(value)
    meta = dict(source=streamlines_file,
                Parameters=segmentation_params_out)
    if segmentation_params['return_idx']:
        meta["idx"] = idx
    meta["Timing"] = time() - start_time
    return tgram, meta


@pimms.calc("clean_bundles_file")
@as_file('-clean_tractography.trk', include_track=True, include_seg=True)
def clean_bundles(subses_dict, bundles_file, bundle_dict, clean_params,
                  tracking_params, segmentation_params):
    img = nib.load(subses_dict['dwi_file'])
    sft = load_tractogram(
        bundles_file,
        img,
        Space.VOX)
    img = nib.load(subses_dict['dwi_file'])

    start_time = time()
    tgram = nib.streamlines.Tractogram([], {'bundle': []})
    if clean_params['return_idx']:
        return_idx = {}

    for b in bundle_dict.keys():
        if b != "whole_brain":
            idx = np.where(
                sft.data_per_streamline['bundle'] == bundle_dict[b]['uid'])[0]
            this_tg = StatefulTractogram(
                sft.streamlines[idx],
                img,
                Space.VOX)
            this_tg = seg.clean_bundle(this_tg, **clean_params)
            if clean_params['return_idx']:
                this_tg, this_idx = this_tg
                idx_file = bundles_file.split('.')[0] + '.json'
                with open(idx_file) as ff:
                    bundle_idx = json.load(ff)["idx"][b]
                return_idx[b] = np.array(
                    bundle_idx)[this_idx].tolist()
            this_tgram = nib.streamlines.Tractogram(
                this_tg.streamlines,
                data_per_streamline={
                    'bundle': (
                        len(this_tg) * [bundle_dict[b]['uid']])},
                    affine_to_rasmm=img.affine)
            tgram = aus.add_bundles(tgram, this_tgram)

    sft = StatefulTractogram(
        tgram.streamlines,
        sft,
        Space.VOX,
        data_per_streamline=tgram.data_per_streamline)

    seg_args = get_default_args(seg.clean_bundle)
    for k in seg_args:
        if callable(seg_args[k]):
            seg_args[k] = seg_args[k].__name__

    meta = dict(source=bundles_file,
                Parameters=seg_args)

    if clean_params['return_idx']:
        meta["idx"] = return_idx

    meta["Timing"] = time() - start_time

    return sft, meta


@pimms.calc("is_bundles_exported")
def export_bundles(subses_dict, clean_bundles_file, bundles_file,
                   bundle_dict, tracking_params, segmentation_params):
    img = nib.load(subses_dict['dwi_file'])
    for bundles_file, folder in zip([clean_bundles_file, bundles_file],
                                    ['clean_bundles', 'bundles']):
        bundles_dir = op.join(subses_dict['results_dir'], folder)
        os.makedirs(bundles_dir, exist_ok=True)
        trk = nib.streamlines.load(bundles_file)
        tg = trk.tractogram
        streamlines = tg.streamlines
        for bundle in bundle_dict:
            if bundle != "whole_brain":
                uid = bundle_dict[bundle]['uid']
                idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
                this_sl = dtu.transform_tracking_output(
                    streamlines[idx],
                    np.linalg.inv(img.affine))

                this_tgm = StatefulTractogram(
                    this_sl, img, Space.VOX)
                fname = op.split(
                    get_fname(
                        subses_dict,
                        f'-{bundle}'
                        f'_tractography.trk',
                        tracking_params=tracking_params,
                        segmentation_params=segmentation_params))
                fname = op.join(bundles_dir, fname[1])
                logger.info(f"Saving {fname}")
                save_tractogram(
                    this_tgm, fname, bbox_valid_check=False)
                meta = dict(source=bundles_file)
                meta_fname = fname.split('.')[0] + '.json'
                afd.write_json(meta_fname, meta)
    return True


@pimms.calc("sl_counts_file")
@as_file('_sl_count.csv', include_track=True, include_seg=True)
def export_sl_counts(subses_dict, bundle_dict,
                     clean_bundles_file, bundles_file,
                     tracking_params, segmentation_params):
    img = nib.load(subses_dict['dwi_file'])
    sl_counts_clean = []
    sl_counts = []
    bundles = list(bundle_dict.keys())
    if "whole_brain" not in bundles:
        bundles.append("whole_brain")
    bundles_files = [clean_bundles_file, bundles_file]
    lists = [sl_counts_clean, sl_counts]

    for bundles_file, count in zip(bundles_files, lists):
        tg = load_tractogram(bundles_file, img)
        bundles = aus.tgram_to_bundles(
            tg, bundle_dict, img)

        for bundle in bundles:
            if bundle == "whole_brain":
                count.append(len(tg.streamlines))
            else:
                count.append(len(
                    bundles[bundle].streamlines))
    counts_df = pd.DataFrame(
        data=dict(
            n_streamlines=sl_counts,
            n_streamlines_clean=sl_counts_clean),
        index=bundles)
    return counts_df, dict(sources=bundles_files)


segmentation_tasks = [
    export_sl_counts,
    export_bundles,
    clean_bundles,
    segment]
