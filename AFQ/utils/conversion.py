import numpy as np

from tqdm import tqdm
import scipy.io

from dipy.io.stateful_tractogram import StatefulTractogram, Space
import nibabel as nib

from AFQ.data import BUNDLE_MAT_2_PYTHON


class MatlabFileTracking():
    """
    Helper class.
    Acts the same as a tracking class from DIPY,
    in that it yields a streamline for each call to __iter__.
    Initialized with an opened h5py matlab file
    and the location of the streamlines in that h5py file.
    """

    def __init__(self, fg_ref):
        self.fg_ref = fg_ref

    def __iter__(self):
        for i in tqdm(range(self.fg_ref.shape[0])):
            yield np.transpose(self.fg_ref[i, 0])


def matlab_tractography(mat_file, img):
    """
    Converts a matlab tractography file to a stateful tractogram.

    Parameters
    ----------
    mat_file : str
        Path to a matlab tractography file.
    img : Nifti1Image or str
        Path to an img file to be loaded with nibabel or an img
        to serve as the reference for the stateful tractogram.

    Returns
    -------
    DIPY :class:`StatefulTractogram` in RASMM space.
    """
    mat_file = scipy.io.loadmat(mat_file)
    if isinstance(img, str):
        img = nib.load(img)

    tracker = MatlabFileTracking(mat_file['fg']['fibers'][0][0])
    return StatefulTractogram(tracker, img, Space.RASMM)


def matlab_mori_groups(mat_file, img):
    """
    Converts a matlab Mori groups file to a dictionary of fiber groups.
    This dictionary is structured the same way as the results of pyAFQ
    segmentation. The keys are bundle names and the values are 
    :class:`StatefulTractogram` instances.
    If you want to merge this dictionary into one :class:`StatefulTractogram`,
    use :func:`bundles_to_tgram`.

    Parameters
    ----------
    mat_file : str
        Path to a matlab Mori groups file.
    img : Nifti1Image or str
        Path to an img file to be loaded with nibabel or an img
        to serve as the reference for the stateful tractogram.

    Returns
    -------
    Dictionary where keys are the pyAFQ bundle names and values are
    DIPY :class:`StatefulTractogram` instances in RASMM space.
    """
    mat_file = scipy.io.loadmat(mat_file)
    if isinstance(img, str):
        img = nib.load(img)

    fiber_groups = {}
    for i in range(mat_file["fg"]["name"].shape[1]):
        name = mat_file["fg"]["name"][0][i][0]
        if name in BUNDLE_MAT_2_PYTHON.keys():
            bundle_ref = mat_file["fg"]["fibers"][0][i]
            tracker = MatlabFileTracking(bundle_ref)
            fiber_groups[BUNDLE_MAT_2_PYTHON[name]] =\
                StatefulTractogram(tracker, img, Space.RASMM)

    return fiber_groups
