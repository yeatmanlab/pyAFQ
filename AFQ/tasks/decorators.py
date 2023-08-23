import functools
import inspect
import logging
import os.path as op
from time import time

import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram

from trx.trx_file_memmap import TrxFile
from trx.io import save as save_trx

import numpy as np

from AFQ.tasks.utils import get_fname
from AFQ.utils.path import drop_extension, write_json


# These should only be used with pimms.calc
__all__ = ["as_file", "as_fit_deriv", "as_img"]


logger = logging.getLogger('AFQ')
logger.setLevel(logging.INFO)


# get args and kwargs from function
def get_args_and_kwargs(func):
    param_dict = inspect.signature(func).parameters
    param_list = func.__code__.co_varnames[
        :func.__code__.co_argcount]
    is_param_kwarg = {
        name: name in param_dict and param_dict[name].default is
        not param_dict[name].empty for name in param_list}
    return param_list, is_param_kwarg, param_dict


# replaces *args and **kwargs with specific parameters from og_func
# so that pimms can see original parameter names after wrapping
# also adds on any args the decorator requires
# these will be extracted with extract_added_args
def has_args(og_func, needed_args):
    def _has_args(func):
        header = "def wrapper_has_args_func("
        content = "):\n    return func("
        found_args = []
        param_list, is_param_kwarg, param_dict = get_args_and_kwargs(og_func)

        # add func args
        for name in param_list:
            if not is_param_kwarg[name]:
                header += f"{name}, "
                content += f"{name}, "
                found_args.append(name)

        # add decorator args
        for arg in needed_args:
            if arg not in found_args:
                header += f"{arg}, "
                content += f"{arg}, "

        # add func kwargs
        for name in param_list:
            if is_param_kwarg[name]:
                default = param_dict[name].default
                if isinstance(default, str):
                    header += f"{name}='{default}', "
                else:
                    header += f"{name}={default}, "
                content += f"{name}={name}, "

        header = header[:-2]
        content = content[:-2]
        content = f"{content})"

        wrapper_has_args = header + content
        scope = {"func": func}
        exec(wrapper_has_args, scope)
        return scope['wrapper_has_args_func']
    return _has_args


# from function where needed args (like base_fname) are added,
# return length of args before added args, and the added args
def extract_added_args(func, names, args, includes=None):
    vals = []
    param_list, is_param_kwarg, _ = get_args_and_kwargs(func)
    arg_list = [param for param in param_list if not is_param_kwarg[param]]
    extra_count = 0
    for jj, name in enumerate(names):
        if includes is not None and not includes[jj]:
            vals.append(None)
            continue

        found = False
        for ii, arg_name in enumerate(arg_list):
            if arg_name == name:
                vals.append(args[ii])
                found = True
                break
        if not found:
            vals.append(args[len(arg_list) + extra_count])
            extra_count = extra_count + 1

    return len(arg_list), *vals


def as_file(suffix, include_track=False, include_seg=False):
    """
    return img and meta as saved file path, with json,
    and only run if not already found
    """
    def _as_file(func):
        needed_args = ["base_fname"]
        if include_track:
            needed_args.append("tracking_params")
        if include_seg:
            needed_args.append("segmentation_params")

        @functools.wraps(func)
        @has_args(func, needed_args)
        def wrapper_as_file(*args, **kwargs):
            og_arg_count, base_fname, tracking_params, segmentation_params =\
                extract_added_args(
                    func,
                    ["base_fname", "tracking_params", "segmentation_params"],
                    args,
                    includes=[True, include_track, include_seg])
            this_file = get_fname(
                base_fname, suffix,
                tracking_params=tracking_params,
                segmentation_params=segmentation_params)
            exists = (op.exists(this_file)
                      or op.exists(this_file + ".trk")
                      or op.exists(this_file + ".trx"))
            if not exists:
                gen, meta = func(*args[:og_arg_count], **kwargs)

                logger.info(f"Saving {this_file}")
                if isinstance(gen, nib.Nifti1Image):
                    nib.save(gen, this_file)
                elif isinstance(gen, StatefulTractogram):
                    is_trx = False
                    this_file = this_file + ".trk"
                    save_tractogram(
                        gen, this_file, bbox_valid_check=False)
                elif isinstance(gen, np.ndarray):
                    np.save(this_file, gen)
                elif isinstance(gen, TrxFile):
                    is_trx = True
                    this_file = this_file + ".trx"
                    save_trx(gen, this_file)
                else:
                    gen.to_csv(this_file)

                if include_seg:
                    meta["dependent"] = "rec"
                elif include_track:
                    if is_trx:
                        meta["dependent"] = "trx"
                    else:
                        meta["dependent"] = "trk"
                else:
                    meta["dependent"] = "dwi"

                meta_fname = get_fname(
                    base_fname, f"{drop_extension(suffix)}.json",
                    tracking_params=tracking_params,
                    segmentation_params=segmentation_params)
                write_json(meta_fname, meta)
            return this_file
        return wrapper_as_file
    return _as_file


def as_fit_deriv(tf_name):
    """
    return data as nibabel image, meta with params information
    """
    def _as_fit_deriv(func):
        needed_args = ["dwi_affine", f"{tf_name.lower()}_params"]

        @functools.wraps(func)
        @has_args(func, needed_args)
        def wrapper_as_fit_deriv(*args, **kwargs):
            og_arg_count, dwi_affine, params = extract_added_args(
                func, needed_args, args)
            img = nib.Nifti1Image(
                func(*args[:og_arg_count], **kwargs), dwi_affine)
            return img, {f"{tf_name}ParamsFile": params}
        return wrapper_as_fit_deriv
    return _as_fit_deriv


def as_img(func):
    """
    return data, meta as nibabel image, meta with timing
    """
    needed_args = ["dwi_affine"]

    @functools.wraps(func)
    @has_args(func, needed_args)
    def wrapper_as_img(*args, **kwargs):
        og_arg_count, affine = extract_added_args(
            func, needed_args, args)
        start_time = time()
        data, meta = func(*args[:og_arg_count], **kwargs)
        meta['timing'] = time() - start_time
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        return img, meta
    return wrapper_as_img
