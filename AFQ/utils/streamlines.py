import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space
import struct


def add_bundles(t1, t2):
    """
    Combine two bundles, using the second bundles' affine and
    data_per_streamline keys.
     Parameters
    ----------
    t1, t2 : nib.streamlines.Tractogram class instances
    """
    data_per_streamline = {}
    for k in t2.data_per_streamline.keys():
        l1 = np.array(t1.data_per_streamline[k])
        l2 = np.array(t2.data_per_streamline[k])
        if len(l1) == 0:
            data_per_streamline[k] = l2
        elif len(l2) == 0:
            data_per_streamline[k] = l1
        else:
            l1_len = l1.shape[1]
            l2_len = l2.shape[1]
            if l1_len > l2_len:
                nanfill = np.full((l2.shape[0], l1_len - l2_len), np.nan)
                l2 = np.concatenate((l2, nanfill), axis=1)
            if l2_len > l1_len:
                nanfill = np.full((l1.shape[0], l2_len - l1_len), np.nan)
                l1 = np.concatenate((l1, nanfill), axis=1)
            data_per_streamline[k] = np.concatenate((l1, l2), axis=0)
    return nib.streamlines.Tractogram(
        list(t1.streamlines) + list(t2.streamlines),
        data_per_streamline,
        affine_to_rasmm=t2.affine_to_rasmm)


# this converts an arbitrary string (bundle_name) to
# a unique, compact array of floats, to be used in data_per_streamline
def bname_to_uid(bundle_name):
    base_idx = 0
    arr_idx = 0
    uid = [0]
    for c in bundle_name:
        c_int = ord(c.upper())
        if c_int >= 65 and c_int <= 90:  # if c is A-Z
            c_int = c_int - 65
            uid[arr_idx] = uid[arr_idx] + c_int * 26**base_idx
            base_idx = base_idx + 1
            if base_idx > 5:
                arr_idx = arr_idx + 1
                uid.append(0)
                base_idx = 0
    for i in range(len(uid)):
        uid[i] = struct.unpack('f', int(uid[i]).to_bytes(4, "little"))[0]
    uid = np.asarray(uid).reshape(1, -1)
    return uid


def bname_to_idx(bundle_name, sft):
    uid = bname_to_uid(bundle_name)[0]
    idxs = []
    for idx, data in enumerate(sft.data_per_streamline['bundle']):
        data = data[~np.isnan(data)]
        if np.allclose(data, uid, rtol=1e-5, atol=0):
            idxs.append(idx)
    return idxs


def bundles_to_tgram(bundles, reference):
    """
    Create a StatefulTractogram object from bundles and their
    specification.

    Parameters
    ----------
    bundles: dict
        Each key in the dict is a bundle name and each value in the dict
        is the stateful tractogram of a particular bundle.
    reference : Nifti
        The affine_to_rasmm input to `nib.streamlines.Tractogram`
    """
    tgram = nib.streamlines.Tractogram([], {'bundle': []})
    for b in bundles:
        this_sl = bundles[b].streamlines
        this_tgram = nib.streamlines.Tractogram(
            this_sl,
            data_per_streamline={
                'bundle': np.repeat(
                    bname_to_uid(b), len(this_sl), axis=0)},
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
        Requires a data_per_streamline['bundle'] attribute.

    bundle_dict: dict
        A bundle specification dictionary.

    reference : Nifti
        The affine_to_rasmm to specify the StatefulTractogram
    """
    bundles = {}
    for bb in bundle_dict.keys():
        if not bb == 'whole_brain':
            idx = bname_to_idx(bb, tgram)
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
