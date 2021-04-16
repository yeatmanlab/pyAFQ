import functools
import os.path as op
import logging
import inspect
from time import time
import numpy as np

import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
import AFQ.data as afd


__all__ = ["as_file", "as_model", "as_tf_deriv", "as_img", "get_fname"]

logger = logging.getLogger('AFQ.api')


def get_fname(subses_dict, suffix,
              tracking_params=None, segmentation_params=None):
    split_fdwi = op.split(subses_dict["dwi_file"])
    fname = op.join(subses_dict["results_dir"], split_fdwi[1].split('.')[0])

    if tracking_params is not None and 'odf_model' in tracking_params:
        odf_model = tracking_params['odf_model']
        directions = tracking_params['directions']
        fname = fname + (
            f'_space-RASMM_model-{odf_model}'
            f'_desc-{directions}'
        )
    if segmentation_params is not None and 'seg_algo' in segmentation_params:
        seg_algo = segmentation_params['seg_algo']
        fname = fname + f'-{seg_algo}'

    return fname + suffix


# replaces *args and **kwargs with specific parameters from og_func
# so that pimms can see original parameter names after wrapping
def has_args(og_func):
    def _has_args(func):
        header = "def wrapper_has_args_func("
        content = "):\n    return func("
        for name, param in inspect.signature(og_func).parameters.items():
            if param.default is param.empty:
                header = header + f"{name}, "
                content = content + f"{name}, "
            else:
                if isinstance(param.default, str):
                    header = header + f"{name}='{param.default}', "
                    content = content + f"{name}='{param.default}', "
                else:
                    header = header + f"{name}={param.default}, "
                    content = content + f"{name}={param.default}, "
        header = header[:-2]
        content = content[:-2]
        content = content + ")"

        wrapper_has_args = header + content
        scope = {"func": func}
        exec(wrapper_has_args, scope)
        return scope['wrapper_has_args_func']
    return _has_args


# first arg must be the subses_dict
# to include tracking or segmentation params,
# they must be last and in that order.
def as_file(suffix, include_track=False, include_seg=False):
    def _as_file(func):
        @functools.wraps(func)
        @has_args(func)
        def wrapper_as_file(subses_dict, *args, **kwargs):
            tracking_params = None
            segmentation_params = None
            if include_track and include_seg:
                tracking_params = args[-2]
                segmentation_params = args[-1]
            elif include_track:
                tracking_params = args[-1]
            elif include_seg:
                segmentation_params = args[-1]
            this_file = get_fname(
                subses_dict, suffix,
                tracking_params=tracking_params,
                segmentation_params=segmentation_params)
            if not op.exists(this_file):
                img_trk_or_csv, meta = func(subses_dict, *args, **kwargs)

                logger.info(f"Saving {this_file}")
                if isinstance(img_trk_or_csv, nib.Nifti1Image):
                    nib.save(img_trk_or_csv, this_file)
                elif isinstance(img_trk_or_csv, StatefulTractogram):
                    save_tractogram(
                        img_trk_or_csv, this_file, bbox_valid_check=False)
                else:
                    img_trk_or_csv.to_csv(this_file)
                meta_fname = get_fname(
                    subses_dict, suffix.split('.')[0] + '.json',
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params)
                afd.write_json(meta_fname, meta)
            return this_file
        return wrapper_as_file
    return _as_file


# second arg must be dwi_affine
# third arg must be a mask to load
def as_model(func):
    @functools.wraps(func)
    @has_args(func)
    def wrapper_as_model(farg, dwi_affine, mask,
                         *args, **kwargs):
        mask =\
            nib.load(mask).get_fdata()
        start_time = time()
        model, meta = func(farg, dwi_affine, mask, *args, **kwargs)
        meta['timing'] = time() - start_time
        model_img = nib.Nifti1Image(
            model,
            dwi_affine)
        return model_img, meta
    return wrapper_as_model


# second arg must be dwi_affine
# third arg must be params file
def as_tf_deriv(tf_name):
    def _as_tf_deriv(func):
        @functools.wraps(func)
        @has_args(func)
        def wrapper_as_tf_deriv(farg, dwi_affine, params,
                                *args, **kwargs):
            img = nib.Nifti1Image(func(
                farg, dwi_affine, params, *args, **kwargs), dwi_affine)
            return img, {f"{tf_name}ParamsFile": params}
        return wrapper_as_tf_deriv
    return _as_tf_deriv


# second arg must be affine
def as_img(func):
    @functools.wraps(func)
    @has_args(func)
    def wrapper_as_img(farg, affine, *args, **kwargs):
        data, meta = func(
            farg, affine, *args, **kwargs)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        return img, meta
    return wrapper_as_img
