import functools
import os.path as op
import logging
from time import time

import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
import AFQ.data as afd


__all__ = ["as_file", "as_model", "as_tf_deriv", "get_fname"]

logger = logging.getLogger('AFQ.api')


def get_fname(subses_tuple, suffix,
              tracking_params=None, segmentation_params=None):
    split_fdwi = op.split(subses_tuple["dwi_file"])
    fname = op.join(subses_tuple["results_dir"], split_fdwi[1].split('.')[0])

    if tracking_params is not None:
        odf_model = tracking_params['odf_model']
        directions = tracking_params['directions']
        fname = fname + (
            f'_space-RASMM_model-{odf_model}'
            f'_desc-{directions}'
        )
    if segmentation_params is not None:
        seg_algo = segmentation_params['seg_algo']
        fname = fname + f'-{seg_algo}'

    return fname + suffix


# first arg must be the subses_tuple
# to include tracking or segmentation params,
# have them as optional arguments
def as_file(suffix):
    def _as_file(func):
        @functools.wraps(func)
        def wrapper_as_file(subses_tuple, *args, **kwargs):
            if "tracking_params" in kwargs:
                tracking_params = kwargs["tracking_params"]
            else:
                tracking_params = None
            if "segmentation_params" in kwargs:
                segmentation_params = kwargs["segmentation_params"]
            else:
                segmentation_params = None
            this_file = get_fname(
                subses_tuple, suffix,
                tracking_params=tracking_params,
                segmentation_params=segmentation_params)
            if not op.exists(this_file):
                img_trk_or_csv, meta = func(subses_tuple, *args, **kwargs)

                logger.info(f"Saving {this_file}")
                if isinstance(img_trk_or_csv, nib.Nifti1Image):
                    nib.save(img_trk_or_csv, this_file)
                elif isinstance(img_trk_or_csv, StatefulTractogram):
                    save_tractogram(
                        img_trk_or_csv, this_file, bbox_valid_check=False)
                else:
                    img_trk_or_csv.to_csv(this_file)
                meta_fname = get_fname(
                    subses_tuple, suffix.split('.')[0] + '.json',
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
        def wrapper_as_tf_deriv(farg, dwi_affine, params,
                                *args, **kwargs):
            img = nib.Nifti1Image(func(
                farg, dwi_affine, params, *args, **kwargs), dwi_affine)
            return img, {f"{tf_name}ParamsFile": params}
        return wrapper_as_tf_deriv
    return _as_tf_deriv

# TODO: as_img wrapper takes data and converts to either sub or templ space image.
# as_tf_deriv, as_model would go on top of this as well
