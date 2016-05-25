import numpy as np
import nibabel as nib
from nibabel import trackvis
from dipy.tracking.utils import move_streamlines

def read_trk(fname, points_space='rasmm'):
    """
    Read from a .trk file, return streamlines and header
    """
    streams, hdr = trackvis.read(fname, points_space=points_space)
    affine = hdr['vox_to_ras']
    return move_streamlines([s[0] for s in streams], affine)


def write_trk(fname, streamlines, affine=None, shape=None):
    """
    Write out a .trk file

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
