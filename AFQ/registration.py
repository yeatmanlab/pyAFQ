"""
Registration tools
"""
import os
import os.path as op
import numpy as np
import nibabel as nib
from dipy.align.imwarp import DiffeomorphicMap

from dipy.align import (syn_registration, center_of_mass, translation,
                        rigid, affine, register_series, )

import dipy.core.gradients as dpg
from dipy.align.streamlinear import whole_brain_slr

import AFQ.utils.models as mut

__all__ = ["syn_register_dwi", "write_mapping", "read_mapping",
           "register_dwi", "slr_registration"]


def reduce_shape(shape):
    """
    Reduce dimension in shape to 3 if possible
    """
    try:
        return shape[:3]
    except TypeError:
        return shape


def syn_register_dwi(dwi, gtab, template=None, **syn_kwargs):
    """
    Register DWI data to a template.

    Parameters
    -----------
    dwi : nifti image or str
        Image containing DWI data, or full path to a nifti file with DWI.
    gtab : GradientTable or list of strings
        The gradients associated with the DWI data, or a string with [fbcal, ]
    template : nifti image or str, optional

    syn_kwargs : key-word arguments for :func:`syn_registration`

    Returns
    -------
    DiffeomorphicMap object
    """
    if template is None:
        import AFQ.data.fetch as afd
        template = afd.read_mni_template()
    if isinstance(template, str):
        template = nib.load(template)

    template_data = template.get_fdata()
    template_affine = template.affine

    if isinstance(dwi, str):
        dwi = nib.load(dwi)

    if not isinstance(gtab, dpg.GradientTable):
        gtab = dpg.gradient_table(*gtab)

    dwi_affine = dwi.affine
    dwi_data = dwi.get_fdata()
    mean_b0 = np.mean(dwi_data[..., gtab.b0s_mask], -1)
    warped_b0, mapping = syn_registration(mean_b0, template_data,
                                          moving_affine=dwi_affine,
                                          static_affine=template_affine,
                                          **syn_kwargs)
    return warped_b0, mapping


def write_mapping(mapping, domain_img, codomain_img,
                  forward_fname, backward_fname):
    """
    Write out a DiffeomorphicMap mapping to a file

    Parameters
    ----------
    mapping : a DiffeomorphicMap object
    domain_img : Nifti1Image
    codomain_img : Nifti1Image
    forward_fname : str
        Full path to store the forward displacement field.
    backward_fname : str
        Full path to store the backward displacement field.
    """
    is_inverse = mapping.is_inverse
    mapping = mapping.get_simplified_transform()
    if is_inverse:
        forward = mapping.backward
        backward = mapping.forward
    else:
        forward = mapping.forward
        backward = mapping.backward
    nib.save(nib.Nifti1Image(forward.astype(np.float32),
                             domain_img.affine),
             forward_fname)
    nib.save(nib.Nifti1Image(backward.astype(np.float32),
                             codomain_img.affine),
             backward_fname)


def read_mapping(forward, backward):
    """
    Read a syn registration mapping from a nifti file,
    Or an affine mapping from a numpy file.

    Parameters
    ----------
    forward : str, Nifti1Image
        If string, file must be the image described below.
        If image, contains the forward mapping displacement field in
        each voxel Shape (x, y, z, 3)

    backward : str, Nifti1Image
        If string, file must be the image described below.
        If image, contains the forward mapping displacement field in
        each voxel Shape (x, y, z, 3)

    domain_img : str or Nifti1Image

    codomain_img : str or Nifti1Image

    Returns
    -------
    A :class:`DiffeomorphicMap` object
    """
    if isinstance(forward, str):
        forward = nib.load(forward)
    if isinstance(backward, str):
        backward = nib.load(backward)
    forward = forward.get_fdata()
    backward = backward.get_fdata()

    mapping = DiffeomorphicMap(
        3, forward.shape[:3], None,
        forward.shape[:3], None,
        backward.shape[:3], None)
    mapping.forward = forward.astype(np.float32)
    mapping.backward = backward.astype(np.float32)

    return mapping


def register_dwi(data_files, bval_files, bvec_files,
                 b0_ref=0,
                 pipeline=[center_of_mass, translation, rigid, affine],
                 out_dir=None):
    """
    Register a DWI data-set

    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.


    """
    img, data, gtab, mask = mut.prepare_data(data_files,
                                             bval_files,
                                             bvec_files)
    if np.sum(gtab.b0s_mask) > 1:
        # First, register the b0s into one image:
        b0_img = nib.Nifti1Image(data[..., gtab.b0s_mask], img.affine)
        trans_b0 = register_series(b0_img, ref=b0_ref, pipeline=pipeline)
        ref_data = np.mean(trans_b0, -1)
    else:
        ref_data = data[..., gtab.b0s_mask]

    # Construct a series out of the DWI and the registered mean B0:
    series = nib.Nifti1Image(np.concatenate([ref_data,
                                             data[...,
                                                  ~gtab.b0s_mask]], -1),
                             img.affine)

    transformed_list, affine_list = register_series(series, ref=0,
                                                    pipeline=pipeline)
    reg_img = nib.Nifti1Image(np.array(transformed_list),
                              img.affine)

    if out_dir is None:
        out_dir = op.join(op.split(data_files)[0], 'registered')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    path = op.join(out_dir, 'registered.nii.gz')
    nib.save(reg_img, path)
    return path


def slr_registration(moving_data, static_data,
                     moving_affine=None, static_affine=None,
                     moving_shape=None, static_shape=None, **kwargs):
    """Register a source image (moving) to a target image (static).

    Parameters
    ----------
    moving : ndarray
        The source tractography data to be registered
    moving_affine : ndarray
        The affine associated with the moving (source) data.
    moving_shape : ndarray
        The shape of the space associated with the static (target) data.
    static : ndarray
        The target tractography data for registration
    static_affine : ndarray
        The affine associated with the static (target) data.
    static_shape : ndarray
        The shape of the space associated with the static (target) data.

    **kwargs:
        kwargs are passed into whole_brain_slr

    Returns
    -------
    AffineMap
    """
    from AFQ.definitions.mapping import ConformedAffineMapping

    _, transform, _, _ = whole_brain_slr(
        static_data, moving_data, x0='affine', verbose=False, **kwargs)

    return ConformedAffineMapping(
        transform,
        codomain_grid_shape=reduce_shape(static_shape),
        codomain_grid2world=static_affine,
        domain_grid_shape=reduce_shape(moving_shape),
        domain_grid2world=moving_affine)
