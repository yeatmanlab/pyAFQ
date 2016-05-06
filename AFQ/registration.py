"""
Registration tools
"""
import numpy as np
import nibabel as nib
import dipy.core.gradients as dpg
from dipy.align.metrics import CCMetric, EMMetric, SSDMetric
from dipy.align.imwarp import SymmetricDiffeomorphicRegistration

from dipy.align.imaffine import (transform_centers_of_mass,
                                 AffineMap,
                                 MutualInformationMetric,
                                 AffineRegistration)

from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D,
                                   AffineTransform3D)


syn_metric_dict = {'CC': CCMetric,
                   'EM': EMMetric,
                   'SSD': SSDMetric}


def syn_registration(moving, static,
                     moving_affine=None,
                     static_affine=None,
                     step_length=0.25,
                     metric='CC',
                     dim=3,
                     level_iters=[10, 10, 5],
                     sigma_diff=2.0,
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
    use_metric = syn_metric_dict[metric](dim, sigma_diff=sigma_diff)

    sdr = SymmetricDiffeomorphicRegistration(use_metric, level_iters,
                                             step_length=step_length)

    mapping = sdr.optimize(static, moving,
                           static_grid2world=static_affine,
                           moving_grid2world=moving_affine,
                           prealign=prealign)

    warped_moving = mapping.transform(moving)
    return warped_moving, mapping


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
    """ Register a series to a reference image.

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
        static_data = static.get_data()
        saff = static.get_affine()
        moving = series
        moving_data = moving.get_data()
        maff = moving.get_affine()

    elif isinstance(ref, int) or np.iterable(ref):
        data = series.get_data()
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
