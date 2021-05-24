import nibabel as nib
import os
import os.path as op
from time import time
import json
import numpy as np
import pandas as pd
import logging

import pimms

from AFQ.tasks.decorators import as_file
from AFQ.tasks.utils import get_fname, with_name
import AFQ.segmentation as seg
import AFQ.utils.streamlines as aus
from AFQ.utils.bin import get_default_args
import AFQ.data as afd

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import dipy.tracking.utils as dtu
from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.tracking.streamline import set_number_of_points, values_from_volume


logger = logging.getLogger('AFQ.api.seg')


outputs = [
    "bundles_file", "clean_bundles_file", "indiv_bundles", "sl_counts_file",
    "profiles_file", "scalar_dict"]


@pimms.calc("bundles_file")
@as_file('_tractography.trk', include_track=True, include_seg=True)
def segment(subses_dict, bundle_dict, data_imap, reg_template, mapping_imap,
            tractography_imap, tracking_params, segmentation_params):
    streamlines_file = tractography_imap["streamlines_file"]
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
        data_imap["bval_file"],
        data_imap["bvec_file"],
        reg_template=reg_template,
        mapping=mapping_imap["mapping"])

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


@pimms.calc("indiv_bundles")
def export_bundles(subses_dict, clean_bundles_file, bundles_file,
                   bundle_dict, tracking_params, segmentation_params):
    img = nib.load(subses_dict['dwi_file'])
    for this_bundles_file, folder in zip([clean_bundles_file, bundles_file],
                                         ['clean_bundles', 'bundles']):
        bundles_dir = op.join(subses_dict['results_dir'], folder)
        os.makedirs(bundles_dir, exist_ok=True)
        trk = nib.streamlines.load(this_bundles_file)
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
                meta = dict(source=this_bundles_file)
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


@pimms.calc("profiles_file")
@as_file('_profiles.csv', include_track=True, include_seg=True)
def tract_profiles(subses_dict, clean_bundles_file, bundle_dict,
                   scalar_dict, profile_weights, dwi_affine,
                   tracking_params, segmentation_params):
    keys = []
    vals = []
    for k in bundle_dict.keys():
        if k != "whole_brain":
            keys.append(bundle_dict[k]['uid'])
            vals.append(k)
    reverse_dict = dict(zip(keys, vals))

    bundle_names = []
    node_numbers = []
    profiles = np.empty((len(scalar_dict), 0)).tolist()
    this_profile = np.zeros((len(scalar_dict), 100))

    trk = nib.streamlines.load(clean_bundles_file)
    for b in np.unique(
            trk.tractogram.data_per_streamline['bundle']):
        idx = np.where(
            trk.tractogram.data_per_streamline['bundle'] == b)[0]
        this_sl = trk.streamlines[idx]
        bundle_name = reverse_dict[b]
        for ii, (scalar, scalar_file) in enumerate(scalar_dict.items()):
            scalar_data = nib.load(scalar_file).get_fdata()
            if isinstance(profile_weights, str):
                if profile_weights == "gauss":
                    this_prof_weights = gaussian_weights(this_sl)
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
    meta = dict(source=clean_bundles_file,
                parameters=get_default_args(afq_profile))

    return profile_dframe, meta


@pimms.calc("scalar_dict")
def get_scalar_dict(scalars, data_imap, mapping_imap):
    scalar_dict = {}
    for scalar in scalars:
        if isinstance(scalar, str):
            sc = scalar.lower()
            scalar_dict[sc] = data_imap[f"{sc}_file"]
        else:
            scalar_dict[scalar.name] = mapping_imap[f"{scalar.name}_file"]
    return {"scalar_dict": scalar_dict}


def get_segmentation_plan():
    segmentation_tasks = with_name([
        get_scalar_dict,
        export_sl_counts,
        export_bundles,
        clean_bundles,
        segment,
        tract_profiles])
    return pimms.plan(**segmentation_tasks)
