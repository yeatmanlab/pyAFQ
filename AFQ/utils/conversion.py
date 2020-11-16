import numpy as np

from tqdm import tqdm
import h5py

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

    def __init__(self, mat_file, fg_ref):
        self.mat_file = mat_file
        self.fg_ref = fg_ref

    def __iter__(self):
        for i in tqdm(range(self.fg_ref.shape[1])):
            yield self.mat_file[self.fg_ref[0, i]][:]


def matlab_tractography(mat_file, img_file):
    """
    Converts a matlab tractography file to a stateful tractogram.

    Parameters
    ----------
    mat_file : str
        Path to a matlab tractography file.
    img_file : str
        Path to an img file to be loaded with nibabel and serve as
        the reference for the stateful tractogram.

    Returns
    -------
    Dipy StatefulTractogram in RASMM space.
    """
    mat_file = h5py.File(mat_file, 'r')
    reference = nib.load(img_file)

    fg_ref = mat_file['fg']['fibers'][:]
    tracker = MatlabFileTracking(mat_file, fg_ref)
    return StatefulTractogram(tracker, reference, Space.RASMM)


def matlab_mori_groups(mat_file, img_file):
    """
    Converts a matlab mori groups file to a dictionary of fiber groups.
    This dictionary is structured the same way as the results of pyAFQ
    segmentation. The keys are bundle names and the values are stateful
    tractograms.
    If you want to merge this dictionary into one stateful tractogram,
    use bundles_to_tgram.

    Parameters
    ----------
    mat_file : str
        Path to a matlab mori groups file.
    img_file : str
        Path to an img file to be loaded with nibabel and serve as
        the reference for the stateful tractogram.

    Returns
    -------
    Dictionary where keys are the pyAFQ bundle names and values are
    Dipy StatefulTractograms in RASMM space.
    """
    mat_file = h5py.File(mat_file, 'r')
    reference = nib.load(img_file)

    fiber_groups = {}
    for i in range(mat_file["fg"]["name"].shape[0]):
        name_data = mat_file[mat_file["fg"]["name"][i][0]]
        name = ''.join(chr(i) for i in name_data[:])
        if name in BUNDLE_MAT_2_PYTHON.keys():
            bundle_ref = mat_file[mat_file["fg"]["fibers"][i][0]]
            tracker = MatlabFileTracking(mat_file, bundle_ref)
            fiber_groups[BUNDLE_MAT_2_PYTHON[name]] =\
                StatefulTractogram(tracker, reference, Space.RASMM)

    return fiber_groups
