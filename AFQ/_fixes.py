import numpy as np
import logging
import time
import numbers
import os
import nibabel as nib
import dipy

from scipy.special import lpmv, gammaln

from tqdm import tqdm
from dipy.align import Bunch
from dipy.tracking.local_tracking import LocalTracking
from dipy.align.imaffine import AffineMap
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.vtk import save_vtk_streamlines, load_vtk_streamlines
from dipy.io.dpy import Dpy
import random

import sys
import math


class ConformedAffineMap(AffineMap):
    """
    Modifies AffineMap API to match DiffeomorphicMap API.
    Important for SLR maps API to be indistinguishable from SYN maps API.
    """

    def transform(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform_inverse(*args, **kwargs)

    def transform_inverse(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform(*args, **kwargs)


def spherical_harmonics(m, n, theta, phi):
    """
    An implementation of spherical harmonics that overcomes conda compilation
    issues. See: https://github.com/nipy/dipy/issues/852
    """
    x = np.cos(phi)
    val = lpmv(m, n, x).astype(complex)
    val *= np.sqrt((2 * n + 1) / 4.0 / np.pi)
    val *= np.exp(0.5 * (gammaln(n - m + 1) - gammaln(n + m + 1)))
    val = val * np.exp(1j * m * theta)
    return val


TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)


class VerboseLocalTracking(LocalTracking):
    def __init__(self, *args, min_length=10, max_length=1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def _generate_streamlines(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        for s in tqdm(self.seeds):
            s = np.dot(lin, s) + offset
            # Set the random seed in numpy and random
            if self.random_seed is not None:
                s_random_seed = hash(np.abs((np.sum(s)) + self.random_seed)) \
                    % (2**32 - 1)
                random.seed(s_random_seed)
                np.random.seed(s_random_seed)
            directions = self.direction_getter.initial_direction(s)
            if directions.size == 0 and self.return_all:
                # only the seed position
                yield [s]
            directions = directions[:self.max_cross]
            for first_step in directions:
                stepsF, tissue_class = self._tracker(s, first_step, F)
                if not (self.return_all
                        or tissue_class == TissueTypes.ENDPOINT
                        or tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                first_step = -first_step
                stepsB, tissue_class = self._tracker(s, first_step, B)
                if not (self.return_all
                        or tissue_class == TissueTypes.ENDPOINT
                        or tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB - 1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)

                len_sl = len(streamline)
                if len_sl < self.min_length * self.step_size \
                        or len_sl > self.max_length * self.step_size:
                    continue
                else:
                    yield streamline


def in_place_norm(vec, axis=-1, keepdims=False, delvec=True):
    """ Return Vectors with Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    -------------
    vec : array_like
        Vectors to norm. Squared in the process of calculating the norm.
    axis : int, optional
        Axis over which to norm. By default norm over last axis. If `axis` is
        None, `vec` is flattened then normed. Default is -1.
    keepdims : bool, optional
        If True, the output will have the same number of dimensions as `vec`,
        with shape 1 on `axis`. Default is False.
    delvec : bool, optional
        If True, vec is deleted as soon as possible.
        If False, vec is not deleted, but still squared. Default is True.

    Returns
    ---------
    norm : array
        Euclidean norms of vectors.

    Examples
    --------
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec)
    array([ 17.,  85.])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, keepdims=True)
    array([[ 17.],
           [ 85.]])
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> in_place_norm(vec, axis=0)
    array([  8.,  39.,  77.])
    """
    vec = np.asarray(vec)

    if keepdims:
        ndim = vec.ndim
        shape = vec.shape

    np.square(vec, out=vec)
    vec_norm = vec.sum(axis)
    if delvec:
        del vec
    try:
        np.sqrt(vec_norm, out=vec_norm)
    except TypeError:
        vec_norm = vec_norm.astype(float)
        np.sqrt(vec_norm, out=vec_norm)

    if keepdims:
        if axis is None:
            shape = [1] * ndim
        else:
            shape = list(shape)
            shape[axis] = 1
        vec_norm = vec_norm.reshape(shape)

    return vec_norm


def tensor_odf(evals, evecs, sphere, num_batches=100):
    """
    Calculate the tensor Orientation Distribution Function

    Parameters
    ----------
    evals : array (4D)
        Eigenvalues of a tensor. Shape (x, y, z, 3).
    evecs : array (5D)
        Eigenvectors of a tensor. Shape (x, y, z, 3, 3)
    sphere : sphere object
        The ODF will be calculated in each vertex of this sphere.
    num_batches : int
        Split the calculation into batches. This reduces memory usage.
        If memory use is not an issue, set to 1.
        If set to -1, there will be 1 batch per vertex in the sphere.
        Default: 100
    """
    num_vertices = sphere.vertices.shape[0]
    if num_batches == -1:
        num_batches = num_vertices
    batch_size = math.ceil(num_vertices / num_batches)
    batches = range(num_batches)

    mask = np.where((evals[..., 0] > 0)
                    & (evals[..., 1] > 0)
                    & (evals[..., 2] > 0))
    evecs = evecs[mask]

    proj_norm = np.zeros((num_vertices, evecs.shape[0]))

    it = tqdm(batches) if num_batches != 1 else batches
    for i in it:
        start = i * batch_size
        end = (i + 1) * batch_size
        if end > num_vertices:
            end = num_vertices

        proj = np.dot(sphere.vertices[start:end], evecs)
        proj /= np.sqrt(evals[mask])
        proj_norm[start:end, :] = in_place_norm(proj)

    proj_norm **= -3
    proj_norm /= 4 * np.pi * np.sqrt(np.prod(evals[mask], -1))

    odf = np.zeros((evals.shape[:3] + (sphere.vertices.shape[0],)))
    odf[mask] = proj_norm.T
    return odf


# The functions below are from dipy master
def is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order):
    """Validate basic data type and value of spatial attribute.
    Does not ensure that voxel_sizes and voxel_order are self-coherent with
    the affine.
    Only verify the following:
        - affine is of the right type (float) and dimension (4,4)
        - affine contain values in the rotation part
        - dimensions is of right type (int) and length (3)
        - voxel_sizes is of right type (float) and length (3)
        - voxel_order is of right type (str) and length (3)
    The listed parameters are what is expected, provide something else and this
    function should fail (cover common mistakes).
    Parameters
    ----------
    affine: ndarray (4,4)
        Tranformation of VOX to RASMM
    dimensions: ndarray (3,), int16
        Volume shape for each axis
    voxel_sizes:  ndarray (3,), float32
        Size of voxel for each axis
    voxel_order: string
        Typically 'RAS' or 'LPS'
    Returns
    -------
    output : bool
        Does the input represent a valid 'state' of spatial attribute
    """
    all_valid = True
    only_3d_warning = False

    if not affine.shape == (4, 4):
        all_valid = False
        logging.warning('Transformation matrix must be 4x4')

    if not affine[0:3, 0:3].any():
        all_valid = False
        logging.warning('Rotation matrix cannot be all zeros')

    if not len(dimensions) >= 3:
        all_valid = False
        only_3d_warning = True

    for i in dimensions:
        if not isinstance(i, numbers.Integral):
            all_valid = False
            logging.warning('Dimensions must be int.')
        if i <= 0:
            all_valid = False
            logging.warning('Dimensions must be above 0.')

    if not len(voxel_sizes) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_sizes:
        if not isinstance(i, numbers.Number):
            all_valid = False
            logging.warning('Voxel size must be int/float.')
        if i <= 0:
            all_valid = False
            logging.warning('Voxel size must be above 0.')

    if not len(voxel_order) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_order:
        if not isinstance(i, str):
            all_valid = False
            logging.warning('Voxel order must be string/char.')
        if i not in ['R', 'A', 'S', 'L', 'P', 'I']:
            all_valid = False
            logging.warning('Voxel order does not follow convention.')

    if only_3d_warning:
        logging.warning('Only 3D (and above) reference are considered valid.')

    return all_valid


def get_reference_info(reference):
    """ Will compare the spatial attribute of 2 references
    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict)
        Reference that provides the spatial attribute.
    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float32, tranformation of VOX to RASMM
        - dimensions ndarray (3,), int16, volume shape for each axis
        - voxel_sizes  ndarray (3,), float32, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """

    is_nifti = False
    is_trk = False
    is_sft = False
    if isinstance(reference, str):
        try:
            header = nib.load(reference).header
            is_nifti = True
        except nib.filebasedimages.ImageFileError:
            pass
        try:
            header = nib.streamlines.load(reference, lazy_load=True).header
            _, extension = os.path.splitext(reference)
            if extension == '.trk':
                is_trk = True
        except ValueError:
            pass
    elif isinstance(reference, nib.nifti1.Nifti1Image):
        header = reference.header
        is_nifti = True
    elif isinstance(reference, nib.streamlines.trk.TrkFile):
        header = reference.header
        is_trk = True
    elif isinstance(reference, nib.nifti1.Nifti1Header):
        header = reference
        is_nifti = True
    elif isinstance(reference, dict) and 'magic_number' in reference:
        header = reference
        is_trk = True
    elif isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
        is_sft = True

    if is_nifti:
        affine = header.get_best_affine()
        dimensions = header['dim'][1:4]
        voxel_sizes = header['pixdim'][1:4]

        if not affine[0:3, 0:3].any():
            raise ValueError('Invalid affine, contains only zeros.'
                             'Cannot determine voxel order from transformation')
        voxel_order = ''.join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header['voxel_to_rasmm']
        dimensions = header['dimensions']
        voxel_sizes = header['voxel_sizes']
        voxel_order = header['voxel_order']
    elif is_sft:
        affine, dimensions, voxel_sizes, voxel_order = reference.space_attributes
    else:
        raise TypeError('Input reference is not one of the supported format')

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode('utf-8')

    # Run this function to logging the warning from it
    is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order)

    return affine, dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """ Will compare the spatial attribute of 2 references
    Parameters
    ----------
    reference_1 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    reference_2 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    Returns
    -------
    output : bool
        Does all the spatial attribute match
    """

    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = get_reference_info(
        reference_1)
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = get_reference_info(
        reference_2)

    identical_header = True
    if not np.allclose(affine_1, affine_2, rtol=1e-03, atol=1e-03):
        logging.error('Affine not equal')
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logging.error('Dimensions not equal')
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2, rtol=1e-03, atol=1e-03):
        logging.error('Voxel_size not equal')
        identical_header = False

    if voxel_order_1 != voxel_order_2:
        logging.error('Voxel_order not equal')
        identical_header = False

    return identical_header


def load_tractogram(filename, reference, to_space=Space.RASMM,
                    to_origin=Origin.NIFTI, bbox_valid_check=True,
                    trk_header_check=True):
    """ Load the stateful tractogram from any format (trk, tck, vtk, fib, dpy)
    Parameters
    ----------
    filename : string
        Filename with valid extension
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), or 'same' if the input is a trk file.
        Reference that provides the spatial attribute.
        Typically a nifti-related object from the native diffusion used for
        streamlines generation
    to_space : Enum (dipy.io.stateful_tractogram.Space)
        Space to which the streamlines will be transformed after loading
    to_origin : Enum (dipy.io.stateful_tractogram.Origin)
        Origin to which the streamlines will be transformed after loading
            NIFTI standard, default (center of the voxel)
            TRACKVIS standard (corner of the voxel)
    bbox_valid_check : bool
        Verification for negative voxel coordinates or values above the
        volume dimensions. Default is True, to enforce valid file.
    trk_header_check : bool
        Verification that the reference has the same header as the spatial
        attributes as the input tractogram when a Trk is loaded
    Returns
    -------
    output : StatefulTractogram
        The tractogram to load (must have been saved properly)
    """
    _, extension = os.path.splitext(filename)
    if extension not in ['.trk', '.tck', '.vtk', '.fib', '.dpy']:
        logging.error('Output filename is not one of the supported format.')
        return False

    if to_space not in Space:
        logging.error('Space MUST be one of the 3 choices (Enum).')
        return False

    if reference == 'same':
        if extension == '.trk':
            reference = filename
        else:
            logging.error('Reference must be provided, "same" is only '
                          'available for Trk file.')
            return False

    if trk_header_check and extension == '.trk':
        if not is_header_compatible(filename, reference):
            logging.error('Trk file header does not match the provided '
                          'reference.')
            return False

    timer = time.time()
    data_per_point = None
    data_per_streamline = None
    if extension in ['.trk', '.tck']:
        tractogram_obj = nib.streamlines.load(filename).tractogram
        streamlines = tractogram_obj.streamlines
        if extension == '.trk':
            data_per_point = tractogram_obj.data_per_point
            data_per_streamline = tractogram_obj.data_per_streamline

    elif extension in ['.vtk', '.fib']:
        streamlines = load_vtk_streamlines(filename)
    elif extension in ['.dpy']:
        dpy_obj = Dpy(filename, mode='r')
        streamlines = list(dpy_obj.read_tracks())
        dpy_obj.close()
    logging.debug('Load %s with %s streamlines in %s seconds.',
                  filename, len(streamlines), round(time.time() - timer, 3))

    sft = StatefulTractogram(streamlines, reference, Space.RASMM,
                             origin=Origin.NIFTI,
                             data_per_point=data_per_point,
                             data_per_streamline=data_per_streamline)

    sft.to_space(to_space)
    sft.to_origin(to_origin)

    if bbox_valid_check and not sft.is_bbox_in_vox_valid():
        raise ValueError('Bounding box is not valid in voxel space, cannot '
                         'load a valid file if some coordinates are invalid.\n'
                         'Please set bbox_valid_check to False and then use '
                         'the function remove_invalid_streamlines to discard '
                         'invalid streamlines.')

    return sft
