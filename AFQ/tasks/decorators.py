import functools
import inspect
import logging
import os.path as op
from time import time

import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
import AFQ.data as afd

import numpy as np

from AFQ.tasks.utils import get_fname


__all__ = ["as_file", "as_model", "as_dt_deriv", "as_img"]


logger = logging.getLogger('AFQ.api')


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


# get argument values, given names, from list of args
def get_args(func, names, args, inside=False):
    vals = []
    for name in names:
        for ii, arg_name in enumerate(list(
                inspect.signature(func).parameters.keys())):
            if inside:
                if name in arg_name:
                    vals.append(args[ii])
                    break
            else:
                if arg_name == name:
                    vals.append(args[ii])
                    break

    if len(vals) < len(names):
        raise NameError(f"{names} requested but {vals} found in get_args")
    return vals


# subses_dict must be in arguments
# if including tracking or segmentation params in fname,
# also include them in args
def as_file(suffix, include_track=False, include_seg=False):
    def _as_file(func):
        @functools.wraps(func)
        @has_args(func)
        def wrapper_as_file(*args, **kwargs):
            subses_dict = get_args(
                func, ["subses_dict"], args)[0]
            if include_track:
                tracking_params = get_args(
                    func, ["tracking_params"], args)[0]
            else:
                tracking_params = None
            if include_seg:
                segmentation_params = get_args(
                    func, ["segmentation_params"], args)[0]
            else:
                segmentation_params = None
            this_file = get_fname(
                subses_dict, suffix,
                tracking_params=tracking_params,
                segmentation_params=segmentation_params)
            if not op.exists(this_file):
                img_trk_or_csv, meta = func(*args, **kwargs)

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


# args must include dwi_affine
def as_model(func):
    @functools.wraps(func)
    @has_args(func)
    def wrapper_as_model(*args, **kwargs):
        dwi_affine = get_args(
            func, ["dwi_affine"], args)[0]
        start_time = time()
        model, meta = func(*args, **kwargs)
        meta['timing'] = time() - start_time
        model_img = nib.Nifti1Image(
            model,
            dwi_affine)
        return model_img, meta
    return wrapper_as_model


# args must include dwi_affine
# args must include arg with name that contains "param"
def as_dt_deriv(tf_name):
    def _as_dt_deriv(func):
        @functools.wraps(func)
        @has_args(func)
        def wrapper_as_dt_deriv(*args, **kwargs):
            dwi_affine = get_args(
                func, ["dwi_affine"], args)[0]
            params = get_args(
                func, ["param"], args, inside=True)[0]
            img = nib.Nifti1Image(func(*args, **kwargs), dwi_affine)
            return img, {f"{tf_name}ParamsFile": params}
        return wrapper_as_dt_deriv
    return _as_dt_deriv


# args must include arg with name that contains affine
def as_img(func):
    @functools.wraps(func)
    @has_args(func)
    def wrapper_as_img(*args, **kwargs):
        affine = get_args(
            func, ["affine"], args, inside=True)[0]
        data, meta = func(*args, **kwargs)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        return img, meta
    return wrapper_as_img
