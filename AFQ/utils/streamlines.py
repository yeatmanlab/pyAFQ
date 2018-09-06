import numpy as np
import nibabel as nib
from nibabel import trackvis
import dipy.tracking.utils as dtu


def add_bundles(t1, t2):
    """
    Combine two bundles, using the second bundles' affine and
    data_per_streamline keys.

    Parameters
    ----------
    t1, t2 : nib.streamlines.Tractogram class instances
    """
    data_per_streamline = {k: (list(t1.data_per_streamline[k]) +
                               list(t2.data_per_streamline[k]))
                           for k in t2.data_per_streamline.keys()}
    return nib.streamlines.Tractogram(
        list(t1.streamlines) + list(t2.streamlines),
        data_per_streamline,
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
    streamlines = dtu.move_streamlines(streamlines, affine)
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


def bundles_to_tgram(bundles, bundle_dict, affine):
    """
    Create a nibabel trk Tractogram object from bundles and their
    specification.

    Parameters
    ----------
    bundles: dict
        Each item in the dict is the streamlines of a particular bundle.
    bundle_dict: dict
        A bundle specification dictionary. Each item includes in particular a
        `uid` key that is a unique integer for that bundle.
    affine : array
        The affine_to_rasmm input to `nib.streamlines.Tractogram`
    """
    tgram = nib.streamlines.Tractogram([], {'bundle': []})
    for b in bundles:
        print("Segmenting: %s" % b)
        this_sl = list(bundles[b])
        this_tgram = nib.streamlines.Tractogram(
            this_sl,
            data_per_streamline={
                'bundle': (len(this_sl) *
                           [bundle_dict[b]['uid']])},
                affine_to_rasmm=affine)
        tgram = add_bundles(tgram, this_tgram)
    return tgram


def tgram_to_bundles(tgram, bundle_dict):
    """
    Convert a nib.streamlines.Tractogram object to a dict with items
    holding the streamlines in each bundle.

    Parameters
    ----------
    tgram : nib.streamlines.Tractogram class instance.

        Requires a data_per_streamline['bundle'][bundle_name]['uid'] attribute.

    bundle_dict: dict
        A bundle specification dictionary. Each item includes in particular a
        `uid` key that is a unique integer for that bundle.
    """
    bundles = {}
    for b in bundle_dict.keys():
        uid = bundle_dict[b]['uid']
        idx = np.where(tgram.data_per_streamline['bundle'] == uid)[0]
        # sl = list(dtu.move_streamlines(tgram.streamlines[idx],
        #                                np.linalg.inv(tgram.affine_to_rasmm)))
        sl = list(tgram.streamlines[idx])

        bundles[b] = sl
    return bundles


def split_streamline(streamlines, sl_to_split, split_idx):
    """
    Given a Streamlines object, split one of the underlying streamlines

    Parameters
    ----------
    streamlines : a Streamlines class instance
        The group of streamlines, one of which is being split.
    sl_to_split : int
        The index of the streamline that is being split
    split_idx : int
        Where is the streamline being split
    """
    this_sl = streamlines[sl_to_split]

    streamlines._lengths = np.concatenate([
        streamlines._lengths[:sl_to_split],
        np.array([split_idx]),
        np.array([this_sl.shape[0] - split_idx]),
        streamlines._lengths[sl_to_split + 1:]])

    streamlines._offsets = np.concatenate([
        np.array([0]),
        np.cumsum(streamlines._lengths[:-1])])

    return streamlines
