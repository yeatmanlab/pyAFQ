import numpy as np
import nibabel as nib
import dipy.reconst.shm as shm

import dipy.data as dpd
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
from dipy.tracking.local import ThresholdTissueClassifier, LocalTracking

from AFQ.dti import tensor_odf


def track(params_file, directions="det", max_angle=30., sphere=None,
          seed_mask=None, n_seeds=1, random_seeds=False,
          stop_mask=None, stop_threshold=0,
          step_size=0.5, min_length=30, max_length=250,
          verbose=False):
    """
    Tractography

    Parameters
    ----------
    params_file : str, nibabel img.
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.
    directions : str
        How tracking directions are determined.
        One of: {"deterministic" | "probablistic"}
    max_angle : float, optional.
        The maximum turning angle in each step. Default: 30
    sphere : Sphere object, optional.
        The discretization of direction getting. default:
        dipy.data.default_sphere.
    seed_mask : array, optional.
        Binary mask describing the ROI within which we seed for tracking.
        Default to the entire volume.
    n_seeds : int or 2D array, optional.
        The seeding density: if this is an int, it is is how many seeds in each
        voxel on each dimension (for example, 2 => [2, 2, 2]). If this is a 2D
        array, these are the coordinates of the seeds. Unless random_seeds is
        set to True, in which case this is the total number of random seeds
        to generate within the mask.
    random_seeds : bool
        Whether to generate a total of n_seeds random seeds in the mask.
        Default: XXX.
    stop_mask : array, optional.
        A floating point value that determines a stopping criterion (e.g. FA).
        Default to no stopping (all ones).
    stop_threshold : float, optional.
        A value of the stop_mask below which tracking is terminated. Default to
        0 (this means that if no stop_mask is passed, we will stop only at
        the edge of the image)
    step_size : float, optional.
        The size (in mm) of a step of tractography. Default: 1.0
    min_length: int, optional
        The miminal length (mm) in a streamline. Default: 10
    max_length: int, optional
        The miminal length (mm) in a streamline. Default: 250
    verbose: bool, optional
        If true, details about what the script is doing is printed. Default: False

    Returns
    -------
    list of streamlines ()
    """

    if verbose:
        print("    Loading Image...")
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    model_params = params_img.get_data()
    affine = params_img.affine

    if verbose:
        print("    Generating Seeds...")
    if isinstance(n_seeds, int):
        if seed_mask is None:
            seed_mask = np.ones(params_img.shape[:3])
        if random_seeds:
            seeds = dtu.random_seeds_from_mask(seed_mask, seeds_count=n_seeds,
                                               seed_count_per_voxel=False,
                                               affine=affine)
        else:
            seeds = dtu.seeds_from_mask(seed_mask,
                                        density=n_seeds,
                                        affine=affine)
    else:
        # If user provided an array, we'll use n_seeds as the seeds:
        seeds = n_seeds
    if sphere is None:
        sphere = dpd.default_sphere

    if verbose:
        print("    Getting Directions...")
    if directions == "det":
        dg = DeterministicMaximumDirectionGetter
    elif directions == "prob":
        dg = ProbabilisticDirectionGetter

    # These are models that have ODFs (there might be others in the future...)
    if model_params.shape[-1] == 12 or model_params.shape[-1] == 27:
        model = "ODF"
    # Could this be an SHM model? If the max order is a whole even number, it
    # might be:
    elif shm.calculate_max_order(model_params.shape[-1]) % 2 == 0:
        model = "SHM"

    if model == "SHM":
        dg = dg.from_shcoeff(model_params, max_angle=max_angle, sphere=sphere)

    elif model == "ODF":
        evals = model_params[..., :3]
        evecs = model_params[..., 3:12].reshape(params_img.shape[:3] + (3, 3))
        odf = tensor_odf(evals, evecs, sphere)
        dg = dg.from_pmf(odf, max_angle=max_angle, sphere=sphere)

    if stop_mask is None:
        stop_mask = np.ones(params_img.shape[:3])

    threshold_classifier = ThresholdTissueClassifier(stop_mask,
                                                     stop_threshold)
    if verbose:
        print("    Tracking...")
    return _local_tracking(seeds, dg, threshold_classifier, affine,
                           step_size=step_size, min_length=min_length,
                           max_length=max_length)


def _local_tracking(seeds, dg, threshold_classifier, affine,
                    step_size=0.5, min_length=10, max_length=250):
    """
    Helper function
    """
    if len(seeds.shape) == 1:
        seeds = seeds[None, ...]
    tracker = ParallelLocalTracking(dg,
                            threshold_classifier,
                            seeds,
                            affine,
                            step_size=step_size)
    
    streamlines = dts.Streamlines(tracker)
    streamlines = streamlines[streamlines._lengths * step_size > min_length]
    streamlines = streamlines[streamlines._lengths * step_size < max_length]
    return streamlines

from tqdm import tqdm
from dipy.align import Bunch
import random
TissueTypes = Bunch(OUTSIDEIMAGE=-1, INVALIDPOINT=0, TRACKPOINT=1, ENDPOINT=2)

class ParallelLocalTracking(LocalTracking):
    # this function is copied from https://github.com/nipy/dipy
    # and modified for parallelization / progress bar
    def _generate_streamlines(self):
        """A streamline generator"""

        # Get inverse transform (lin/offset) for seeds
        inv_A = np.linalg.inv(self.affine)
        lin = inv_A[:3, :3]
        offset = inv_A[:3, 3]

        F = np.empty((self.max_length + 1, 3), dtype=float)
        B = F.copy()
        pbar = tqdm(total=self.seeds.shape[0])
        for s in self.seeds:
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
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                first_step = -first_step
                stepsB, tissue_class = self._tracker(s, first_step, B)
                if not (self.return_all or
                        tissue_class == TissueTypes.ENDPOINT or
                        tissue_class == TissueTypes.OUTSIDEIMAGE):
                    continue
                if stepsB == 1:
                    streamline = F[:stepsF].copy()
                else:
                    parts = (B[stepsB - 1:0:-1], F[:stepsF])
                    streamline = np.concatenate(parts, axis=0)
                yield streamline
            pbar.update(1)
        pbar.close()

## testing

import numbers
from operator import mul
from functools import reduce

import numpy as np

MEGABYTE = 1024 * 1024


def is_array_sequence(obj):
    """ Return True if `obj` is an array sequence. """
    try:
        return obj.is_array_sequence
    except AttributeError:
        return False


def is_ndarray_of_int_or_bool(obj):
    return (isinstance(obj, np.ndarray) and
            (np.issubdtype(obj.dtype, np.integer) or
            np.issubdtype(obj.dtype, np.bool_)))


def _safe_resize(a, shape):
    """ Resize an ndarray safely, using minimal memory """
    try:
        a.resize(shape)
    except ValueError:
        a = a.copy()
        a.resize(shape, refcheck=False)
    return a


class _BuildCache(object):
    def __init__(self, arr_seq, common_shape, dtype):
        self.offsets = list(arr_seq._offsets)
        self.lengths = list(arr_seq._lengths)
        self.next_offset = arr_seq._get_next_offset()
        self.bytes_per_buf = arr_seq._buffer_size * MEGABYTE
        # Use the passed dtype only if null data array
        self.dtype = dtype if arr_seq._data.size == 0 else arr_seq._data.dtype
        if arr_seq.common_shape != () and common_shape != arr_seq.common_shape:
            raise ValueError(
                "All dimensions, except the first one, must match exactly")
        self.common_shape = common_shape
        n_in_row = reduce(mul, common_shape, 1)
        bytes_per_row = n_in_row * dtype.itemsize
        self.rows_per_buf = max(1, self.bytes_per_buf // bytes_per_row)

    def update_seq(self, arr_seq):
        arr_seq._offsets = np.array(self.offsets)
        arr_seq._lengths = np.array(self.lengths)

class ArraySequence(object):
    """ Sequence of ndarrays having variable first dimension sizes.

    This is a container that can store multiple ndarrays where each ndarray
    might have a different first dimension size but a *common* size for the
    remaining dimensions.

    More generally, an instance of :class:`ArraySequence` of length $N$ is
    composed of $N$ ndarrays of shape $(d_1, d_2, ... d_D)$ where $d_1$
    can vary in length between arrays but $(d_2, ..., d_D)$ have to be the
    same for every ndarray.
    """

    def __init__(self, iterable=None, buffer_size=4):
        """ Initialize array sequence instance

        Parameters
        ----------
        iterable : None or iterable or :class:`ArraySequence`, optional
            If None, create an empty :class:`ArraySequence` object.
            If iterable, create a :class:`ArraySequence` object initialized
            from array-like objects yielded by the iterable.
            If :class:`ArraySequence`, create a view (no memory is allocated).
            For an actual copy use :meth:`.copy` instead.
        buffer_size : float, optional
            Size (in Mb) for memory allocation when `iterable` is a generator.
        """
        # Create new empty `ArraySequence` object.
        self._is_view = False
        self._data = np.array([])
        self._offsets = np.array([], dtype=np.intp)
        self._lengths = np.array([], dtype=np.intp)
        self._buffer_size = buffer_size
        self._build_cache = None

        if iterable is None:
            return

        if is_array_sequence(iterable):
            # Create a view.
            self._data = iterable._data
            self._offsets = iterable._offsets
            self._lengths = iterable._lengths
            self._is_view = True
            return

        self.extend(iterable)

    @property
    def is_array_sequence(self):
        return True

    @property
    def common_shape(self):
        """ Matching shape of the elements in this array sequence. """
        return self._data.shape[1:]

    @property
    def total_nb_rows(self):
        """ Total number of rows in this array sequence. """
        return np.sum(self._lengths)

    @property
    def data(self):
        """ Elements in this array sequence. """
        return self._data

    def _get_next_offset(self):
        """ Offset in ``self._data`` at which to write next rowelement """
        if len(self._offsets) == 0:
            return 0
        imax = np.argmax(self._offsets)
        return self._offsets[imax] + self._lengths[imax]

    def append(self, element, cache_build=False):
        """ Appends `element` to this array sequence.

        Append can be a lot faster if it knows that it is appending several
        elements instead of a single element.  In that case it can cache the
        parameters it uses between append operations, in a "build cache".  To
        tell append to do this, use ``cache_build=True``.  If you use
        ``cache_build=True``, you need to finalize the append operations with
        :meth:`finalize_append`.

        Parameters
        ----------
        element : ndarray
            Element to append. The shape must match already inserted elements
            shape except for the first dimension.
        cache_build : {False, True}
            Whether to save the build cache from this append routine.  If True,
            append can assume it is the only player updating `self`, and the
            caller must finalize `self` after all append operations, with
            ``self.finalize_append()``.

        Returns
        -------
        None

        Notes
        -----
        If you need to add multiple elements you should consider
        `ArraySequence.extend`.
        """
        element = np.asarray(element)
        if element.size == 0:
            return
        el_shape = element.shape
        n_items, common_shape = el_shape[0], el_shape[1:]
        build_cache = self._build_cache
        in_cached_build = build_cache is not None
        if not in_cached_build:  # One shot append, not part of sequence
            build_cache = _BuildCache(self, common_shape, element.dtype)
        next_offset = build_cache.next_offset
        req_rows = next_offset + n_items
        if self._data.shape[0] < req_rows:
            self._resize_data_to(req_rows, build_cache)
        self._data[next_offset:req_rows] = element
        build_cache.offsets.append(next_offset)
        build_cache.lengths.append(n_items)
        build_cache.next_offset = req_rows
        if in_cached_build:
            return
        if cache_build:
            self._build_cache = build_cache
        else:
            build_cache.update_seq(self)

    def finalize_append(self):
        """ Finalize process of appending several elements to `self`

        :meth:`append` can be a lot faster if it knows that it is appending
        several elements instead of a single element.  To tell the append
        method this is the case, use ``cache_build=True``.  This method
        finalizes the series of append operations after a call to
        :meth:`append` with ``cache_build=True``.
        """
        if self._build_cache is None:
            return
        self._build_cache.update_seq(self)
        self._build_cache = None
        self.shrink_data()

    def _resize_data_to(self, n_rows, build_cache):
        """ Resize data array if required """
        # Calculate new data shape, rounding up to nearest buffer size
        n_bufs = np.ceil(n_rows / build_cache.rows_per_buf)
        extended_n_rows = int(n_bufs * build_cache.rows_per_buf)
        new_shape = (extended_n_rows,) + build_cache.common_shape
        if self._data.size == 0:
            self._data = np.empty(new_shape, dtype=build_cache.dtype)
        else:
            self._data = _safe_resize(self._data, new_shape)

    def shrink_data(self):
        self._data.resize((self._get_next_offset(),) + self.common_shape,
                          refcheck=False)

    def extend(self, elements):
        """ Appends all `elements` to this array sequence.

        Parameters
        ----------
        elements : iterable of ndarrays or :class:`ArraySequence` object
            If iterable of ndarrays, each ndarray will be concatenated along
            the first dimension then appended to the data of this
            ArraySequence.
            If :class:`ArraySequence` object, its data are simply appended to
            the data of this ArraySequence.

        Returns
        -------
        None

        Notes
        -----
        The shape of the elements to be added must match the one of the data of
        this :class:`ArraySequence` except for the first dimension.
        """
        # If possible try pre-allocating memory.
        try:
            print (elements)
            iter_len = len(elements)
        except TypeError:
            print ("Excepted")
            pass
        else:  # We do know the iterable length
            print ("Elsed")
            if iter_len == 0:
                return
            e0 = np.asarray(elements[0])
            n_elements = np.sum([len(e) for e in elements])
            self._build_cache = _BuildCache(self, e0.shape[1:], e0.dtype)
            self._resize_data_to(self._get_next_offset() + n_elements,
                                 self._build_cache)

        for e in elements:
            self.append(e, cache_build=True)

        self.finalize_append()

    def copy(self):
        """ Creates a copy of this :class:`ArraySequence` object.

        Returns
        -------
        seq_copy : :class:`ArraySequence` instance
            Copy of `self`.

        Notes
        -----
        We do not simply deepcopy this object because we have a chance to use
        less memory. For example, if the array sequence being copied is the
        result of a slicing operation on an array sequence.
        """
        seq = self.__class__()
        total_lengths = np.sum(self._lengths)
        seq._data = np.empty((total_lengths,) + self._data.shape[1:],
                             dtype=self._data.dtype)

        next_offset = 0
        offsets = []
        for offset, length in zip(self._offsets, self._lengths):
            offsets.append(next_offset)
            chunk = self._data[offset:offset + length]
            seq._data[next_offset:next_offset + length] = chunk
            next_offset += length

        seq._offsets = np.asarray(offsets)
        seq._lengths = self._lengths.copy()

        return seq

    def __getitem__(self, idx):
        """ Get sequence(s) through standard or advanced numpy indexing.

        Parameters
        ----------
        idx : int or slice or list or ndarray
            If int, index of the element to retrieve.
            If slice, use slicing to retrieve elements.
            If list, indices of the elements to retrieve.
            If ndarray with dtype int, indices of the elements to retrieve.
            If ndarray with dtype bool, only retrieve selected elements.

        Returns
        -------
        ndarray or :class:`ArraySequence`
            If `idx` is an int, returns the selected sequence.
            Otherwise, returns a :class:`ArraySequence` object which is a view
            of the selected sequences.
        """
        if isinstance(idx, (numbers.Integral, np.integer)):
            start = self._offsets[idx]
            return self._data[start:start + self._lengths[idx]]

        seq = self.__class__()
        seq._is_view = True
        if isinstance(idx, tuple):
            off_idx = idx[0]
            seq._data = self._data.__getitem__((slice(None),) + idx[1:])
        else:
            off_idx = idx
            seq._data = self._data

        if isinstance(off_idx, slice):  # Standard list slicing
            seq._offsets = self._offsets[off_idx]
            seq._lengths = self._lengths[off_idx]
            return seq

        if isinstance(off_idx, list) or is_ndarray_of_int_or_bool(off_idx):
            # Fancy indexing
            seq._offsets = self._offsets[off_idx]
            seq._lengths = self._lengths[off_idx]
            return seq

        raise TypeError("Index must be either an int, a slice, a list of int"
                        " or a ndarray of bool! Not " + str(type(idx)))

    def __iter__(self):
        if len(self._lengths) != len(self._offsets):
            raise ValueError("ArraySequence object corrupted:"
                             " len(self._lengths) != len(self._offsets)")

        for offset, lengths in zip(self._offsets, self._lengths):
            yield self._data[offset: offset + lengths]

    def __len__(self):
        return len(self._offsets)

    def __repr__(self):
        if len(self) > np.get_printoptions()['threshold']:
            # Show only the first and last edgeitems.
            edgeitems = np.get_printoptions()['edgeitems']
            data = str(list(self[:edgeitems]))[:-1]
            data += ", ..., "
            data += str(list(self[-edgeitems:]))[1:]
        else:
            data = str(list(self))

        return "{name}({data})".format(name=self.__class__.__name__,
                                       data=data)

    def save(self, filename):
        """ Saves this :class:`ArraySequence` object to a .npz file. """
        np.savez(filename,
                 data=self._data,
                 offsets=self._offsets,
                 lengths=self._lengths)

    @classmethod
    def load(cls, filename):
        """ Loads a :class:`ArraySequence` object from a .npz file. """
        content = np.load(filename)
        seq = cls()
        seq._data = content["data"]
        seq._offsets = content["offsets"]
        seq._lengths = content["lengths"]
        return seq