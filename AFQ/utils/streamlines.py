import numpy as np
import nibabel as nib
from nibabel import trackvis
from dipy.tracking.utils import move_streamlines


def add_bundles(t1, t2):
    """
    Combine two bundles, using the second bundles affine
    """
    return nib.streamlines.Tractogram(
        list(t1.streamlines) + list(t2.streamlines),
        {'bundle': (list(t1.data_per_streamline['bundle']) +
                    list(t2.data_per_streamline['bundle']))},
        affine_to_rasmm=t2.affine_to_rasmm)


def read_trk(fname):
    """
    Read from a .trk file, return streamlines and header

    Parameters
    ----------
    fname : str
        Full path to a trk file containing

    Returns
    -------
    list : list of streamlines (3D coordinates)

    Notes
    -----
    We assume that all streamlines are provided with the "rasmm" points_space.
    That is, they have been transformed to the space reported by the affine
    associated with the image from whence it came, and saved with this affine
    (e.g., using `write_trk`).

    """
    streams, hdr = trackvis.read(fname, points_space="rasmm")
    return [s[0] for s in streams]


def write_trk(fname, streamlines, affine=None, shape=None):
    """
    Write out a .trk file

    Parameters
    ----------
    fname : str
        Full path to save the file into
    streamlines : list
        A list of arrays of 3D coordinates
    affine : array (4,4), optional.
        An affine transformation associated with the streamlines. Defaults to
        identity.
    shape : 3-element tuple, optional
        Spatial dimensions of an image associated with the streamlines.
        Defaults to not be set in the file header.
    """
    if affine is None:
        affine = np.eye(4)

    zooms = np.sqrt((affine * affine).sum(0))
    streamlines = move_streamlines(streamlines, affine)
    data = ((s, None, None) for s in streamlines)

    voxel_order = nib.orientations.aff2axcodes(affine)
    voxel_order = "".join(voxel_order)

    hdr = nib.trackvis.empty_header()
    hdr['voxel_size'] = zooms[:3]
    hdr['voxel_order'] = voxel_order
    hdr['vox_to_ras'] = affine
    if shape is not None:
        hdr['dim'] = shape
    trackvis.write(fname, data, hdr, points_space="rasmm")
