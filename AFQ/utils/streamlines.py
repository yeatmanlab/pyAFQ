import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space


def add_bundles(t1, t2):
    """
    Combine two bundles, using the second bundles' affine and
    data_per_streamline keys.
     Parameters
    ----------
    t1, t2 : nib.streamlines.Tractogram class instances
    """
    data_per_streamline = {k: (list(t1.data_per_streamline[k])
                               + list(t2.data_per_streamline[k]))
                           for k in t2.data_per_streamline.keys()}
    return nib.streamlines.Tractogram(
        list(t1.streamlines) + list(t2.streamlines),
        data_per_streamline,
        affine_to_rasmm=t2.affine_to_rasmm)


def bundles_to_tgram(bundles, bundle_dict, reference):
    """
    Create a StatefulTractogram object from bundles and their
    specification.

    Parameters
    ----------
    bundles: dict
        Each key in the dict is a bundle name and each value in the dict
        is the stateful tractogram of a particular bundle.
    bundle_dict: dict
        A bundle specification dictionary. Each key is a bundle name, and each
        value is another dictionary specifying bundle properties. In this
        value dictionary, there must be one `uid` key whose value is a
        unique integer for that bundle.
    reference : Nifti
        The affine_to_rasmm input to `nib.streamlines.Tractogram`
    """
    tgram = nib.streamlines.Tractogram([], {'bundle': []})
    for b in bundles:
        this_sl = bundles[b].streamlines
        this_tgram = nib.streamlines.Tractogram(
            this_sl,
            data_per_streamline={
                'bundle': (len(this_sl)
                           * [bundle_dict[b]['uid']])},
                affine_to_rasmm=reference.affine)
        tgram = add_bundles(tgram, this_tgram)
    return StatefulTractogram(tgram.streamlines, reference, Space.VOX,
                              data_per_streamline=tgram.data_per_streamline)


def tgram_to_bundles(tgram, bundle_dict, reference):
    """
    Convert a StatefulTractogram object to a dict with StatefulTractogram
    objects for each bundle.

    Parameters
    ----------
    tgram : StatefulTractogram class instance.
        Requires a data_per_streamline['bundle'][bundle_name]['uid'] attribute.

    bundle_dict: dict
        A bundle specification dictionary. Each item includes in particular a
        `uid` key that is a unique integer for that bundle.
    """
    bundles = {}
    for bb in bundle_dict.keys():
        if not bb == 'whole_brain':
            uid = bundle_dict[bb]['uid']
            idx = np.where(tgram.data_per_streamline['bundle'] == uid)[0]
            bundles[bb] = StatefulTractogram(
                tgram.streamlines[idx].copy(), reference, Space.VOX)
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
