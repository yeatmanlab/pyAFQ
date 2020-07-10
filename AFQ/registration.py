"""
Registration tools
"""
import os
import os.path as op
import numpy as np
import nibabel as nib
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import (SymmetricDiffeomorphicRegistration,
                               DiffeomorphicMap)

from dipy.align.imaffine import (transform_centers_of_mass,
                                 MutualInformationMetric,
                                 AffineRegistration, AffineMap)

from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)

import dipy.core.gradients as dpg
import dipy.data as dpd
from dipy.align.streamlinear import \
    StreamlineLinearRegistration, whole_brain_slr
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.utils import transform_tracking_output
from dipy.io.streamline import load_tractogram, load_trk

import AFQ.utils.models as mut
import AFQ.utils.streamlines as sut
from AFQ._fixes import ConformedAffineMap

syn_metric_dict = {'CC': CCMetric,
                   'EM': EMMetric,
                   'SSD': SSDMetric}

__all__ = ["syn_registration", "syn_register_dwi", "write_mapping",
           "read_mapping", "resample", "c_of_mass", "translation", "rigid",
           "affine", "affine_registration", "register_series", "register_dwi",
           "streamline_registration"]


def syn_registration(moving, static,
                     moving_affine=None,
                     static_affine=None,
                     step_length=0.25,
                     metric='CC',
                     dim=3,
                     level_iters=[10, 10, 5],
                     sigma_diff=2.0,
                     radius=4,
                     prealign=None):
    """Register a source image (moving) to a target image (static).

    Parameters
    ----------
    moving : ndarray
        The source image data to be registered
    moving_affine : array, shape (4,4)
        The affine matrix associated with the moving (source) data.
    static : ndarray
        The target image data for registration
    static_affine : array, shape (4,4)
        The affine matrix associated with the static (target) data
    metric : string, optional
        The metric to be optimized. One of `CC`, `EM`, `SSD`,
        Default: CCMetric.
    dim: int (either 2 or 3), optional
       The dimensions of the image domain. Default: 3
    level_iters : list of int, optional
        the number of iterations at each level of the Gaussian Pyramid (the
        length of the list defines the number of pyramid levels to be
        used).
    sigma_diff, radius : float
        Parameters for initialization of the metric.

    Returns
    -------
    warped_moving : ndarray
        The data in `moving`, warped towards the `static` data.
    forward : ndarray (..., 3)
        The vector field describing the forward warping from the source to the
        target.
    backward : ndarray (..., 3)
        The vector field describing the backward warping from the target to the
        source.
    """
    use_metric = syn_metric_dict[metric](dim, sigma_diff=sigma_diff,
                                         radius=radius)

    sdr = SymmetricDiffeomorphicRegistration(use_metric, level_iters,
                                             step_length=step_length)

    mapping = sdr.optimize(static, moving,
                           static_grid2world=static_affine,
                           moving_grid2world=moving_affine,
                           prealign=prealign)

    warped_moving = mapping.transform(moving)
    return warped_moving, mapping


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
        import AFQ.data as afd
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


def write_mapping(mapping, fname):
    """
    Write out a syn registration mapping to file

    Parameters
    ----------
    mapping : a DiffeomorphicMap object derived from :func:`syn_registration`
    fname : str
        Full path to the nifti file storing the mapping

    """
    if isinstance(mapping, DiffeomorphicMap):
        mapping_data = np.array([mapping.forward.T, mapping.backward.T]).T
        nib.save(nib.Nifti1Image(mapping_data, mapping.codomain_world2grid),
                 fname)
    else:
        np.save(fname, mapping.affine)


def read_mapping(disp, domain_img, codomain_img, prealign=None):
    """
    Read a syn registration mapping from a nifti file

    Parameters
    ----------
    disp : str, Nifti1Image, or ndarray
        If string, file must of an image or ndarray.
        If image, contains the mapping displacement field in each voxel
        Shape (x, y, z, 3, 2)
        If ndarray, contains affine transformation used for mapping

    domain_img : str or Nifti1Image

    codomain_img : str or Nifti1Image

    Returns
    -------
    A :class:`DiffeomorphicMap` object
    """
    if isinstance(disp, str):
        if "nii.gz" in disp:
            disp = nib.load(disp)
        else:
            disp = np.load(disp)

    if isinstance(domain_img, str):
        domain_img = nib.load(domain_img)

    if isinstance(codomain_img, str):
        codomain_img = nib.load(codomain_img)

    if isinstance(disp, nib.Nifti1Image):
        mapping = DiffeomorphicMap(3, disp.shape[:3],
                                   disp_grid2world=np.linalg.inv(disp.affine),
                                   domain_shape=domain_img.shape[:3],
                                   domain_grid2world=domain_img.affine,
                                   codomain_shape=codomain_img.shape,
                                   codomain_grid2world=codomain_img.affine,
                                   prealign=prealign)

        disp_data = disp.get_fdata().astype(np.float32)
        mapping.forward = disp_data[..., 0]
        mapping.backward = disp_data[..., 1]
        mapping.is_inverse = True
    else:
        mapping = ConformedAffineMap(disp,
                                     domain_grid_shape=reduce_shape(
                                         domain_img.shape),
                                     domain_grid2world=domain_img.affine,
                                     codomain_grid_shape=reduce_shape(
                                         codomain_img.shape),
                                     codomain_grid2world=codomain_img.affine)

    return mapping


def resample(moving, static, moving_affine, static_affine):
    """Resample an image from one space to another.

    Parameters
    ----------
    moving : array
       The image to be resampled

    static : array

    moving_affine
    static_affine

    Returns
    -------
    resampled : the moving array resampled into the static array's space.
    """
    identity = np.eye(4)
    affine_map = AffineMap(identity,
                           static.shape, static_affine,
                           moving.shape, moving_affine)
    resampled = affine_map.transform(moving)
    return resampled


# Affine registration pipeline:
affine_metric_dict = {'MI': MutualInformationMetric}


def c_of_mass(moving, static, static_affine, moving_affine,
              reg, starting_affine, params0=None):
    transform = transform_centers_of_mass(static, static_affine,
                                          moving, moving_affine)
    transformed = transform.transform(moving)
    return transformed, transform.affine


def translation(moving, static, static_affine, moving_affine,
                reg, starting_affine, params0=None):
    transform = TranslationTransform3D()
    translation = reg.optimize(static, moving, transform, params0,
                               static_affine, moving_affine,
                               starting_affine=starting_affine)

    return translation.transform(moving), translation.affine


def rigid(moving, static, static_affine, moving_affine,
          reg, starting_affine, params0=None):
    transform = RigidTransform3D()
    rigid = reg.optimize(static, moving, transform, params0,
                         static_affine, moving_affine,
                         starting_affine=starting_affine)
    return rigid.transform(moving), rigid.affine


def affine(moving, static, static_affine, moving_affine,
           reg, starting_affine, params0=None):
    transform = AffineTransform3D()
    affine = reg.optimize(static, moving, transform, params0,
                          static_affine, moving_affine,
                          starting_affine=starting_affine)

    return affine.transform(moving), affine.affine


def affine_registration(moving, static,
                        moving_affine=None,
                        static_affine=None,
                        nbins=32,
                        sampling_prop=None,
                        metric='MI',
                        pipeline=[c_of_mass, translation, rigid, affine],
                        level_iters=[10000, 1000, 100],
                        sigmas=[5.0, 2.5, 0.0],
                        factors=[4, 2, 1],
                        params0=None):
    """
    Find the affine transformation between two 3D images.

    Parameters
    ----------

    """
    # Define the Affine registration object we'll use with the chosen metric:
    use_metric = affine_metric_dict[metric](nbins, sampling_prop)
    affreg = AffineRegistration(metric=use_metric,
                                level_iters=level_iters,
                                sigmas=sigmas,
                                factors=factors)

    # Bootstrap this thing with the identity:
    starting_affine = np.eye(4)
    # Go through the selected transformation:
    for func in pipeline:
        transformed, starting_affine = func(moving, static,
                                            static_affine,
                                            moving_affine,
                                            affreg, starting_affine,
                                            params0)
    return transformed, starting_affine


def register_series(series, ref, pipeline):
    """Register a series to a reference image.

    Parameters
    ----------
    series : Nifti1Image object
       The data is 4D with the last dimension separating different 3D volumes
    ref : Nifti1Image or integer or iterable

    Returns
    -------
    transformed_list, affine_list
    """
    if isinstance(ref, nib.Nifti1Image):
        static = ref
        static_data = static.get_fdata()
        s_aff = static.affine
        moving = series
        moving_data = moving.get_fdata()
        m_aff = moving.affine

    elif isinstance(ref, int) or np.iterable(ref):
        data = series.get_fdata()
        idxer = np.zeros(data.shape[-1]).astype(bool)
        idxer[ref] = True
        static_data = data[..., idxer]
        if len(static_data.shape) > 3:
            static_data = np.mean(static_data, -1)
        moving_data = data[..., ~idxer]
        m_aff = s_aff = series.affine

    affine_list = []
    transformed_list = []
    for ii in range(moving_data.shape[-1]):
        this_moving = moving_data[..., ii]
        transformed, affine = affine_registration(this_moving, static_data,
                                                  moving_affine=m_aff,
                                                  static_affine=s_aff,
                                                  pipeline=pipeline)
        transformed_list.append(transformed)
        affine_list.append(affine)

    return transformed_list, affine_list


def register_dwi(data_files, bval_files, bvec_files,
                 b0_ref=0,
                 pipeline=[c_of_mass, translation, rigid, affine],
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
        trans_b0 = register_series(b0_img, ref=0, pipeline=pipeline)
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


def streamline_registration(moving, static, n_points=100,
                            native_resampled=False):
    """
    Register two collections of streamlines ('bundles') to each other

    Parameters
    ----------
    moving, static : lists of 3 by n, or str
        The two bundles to be registered. Given either as lists of arrays with
        3D coordinates, or strings containing full paths to these files.

    n_points : int, optional
        How many points to resample to. Default: 100.

    native_resampled : bool, optional
        Whether to return the moving bundle in the original space, but
        resampled in the static space to n_points.

    Returns
    -------
    aligned : list
        Streamlines from the moving group, moved to be closely matched to
        the static group.

    matrix : array (4, 4)
        The affine transformation that takes us from 'moving' to 'static'
    """
    # Load the streamlines, if you were given a file-name
    if isinstance(moving, str):
        moving = load_trk(moving, 'same', bbox_valid_check=False).streamlines
    if isinstance(static, str):
        static = load_trk(static, 'same', bbox_valid_check=False).streamlines

    srr = StreamlineLinearRegistration()
    srm = srr.optimize(static=set_number_of_points(static, n_points),
                       moving=set_number_of_points(moving, n_points))

    aligned = srm.transform(moving)
    if native_resampled:
        aligned = set_number_of_points(aligned, n_points)
        aligned = transform_tracking_output(aligned, np.linalg.inv(srm.matrix))

    return aligned, srm.matrix


def reduce_shape(shape):
    """
    Reduce dimension in shape to 3 if possible
    """
    try:
        return shape[:3]
    except:
        return shape


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
    _, transform, _, _ = whole_brain_slr(
        static_data, moving_data, x0='affine', verbose=False, **kwargs)

    return ConformedAffineMap(transform,
                              codomain_grid_shape=reduce_shape(static_shape),
                              codomain_grid2world=static_affine,
                              domain_grid_shape=reduce_shape(moving_shape),
                              domain_grid2world=moving_affine)
