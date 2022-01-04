import numpy as np
import nibabel as nib
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from AFQ.s3bids import read_json


class SegmentedSFT():
    def __init__(self, bundles):
        reference = None
        self.bundle_names = []
        sls = []
        idxs = {}
        this_tracking_idxs = []
        idx_count = 0
        for b_name in bundles:
            if 'idx' in bundles[b_name]:
                this_sls = bundles[b_name]['sl']
                this_tracking_idxs.extend(bundles[b_name]['idx'])
            else:
                this_sls = bundles[b_name]
            if reference is None:
                reference = this_sls.reference
            this_sls = list(this_sls.streamlines)
            sls.extend(this_sls)
            new_idx_count = idx_count + len(this_sls)
            idxs[b_name] = list(range(idx_count, new_idx_count))
            idx_count = new_idx_count
            self.bundle_names.append(b_name)

        self.sft = StatefulTractogram(sls, reference, Space.VOX)
        self.bundle_idxs = idxs
        if len(this_tracking_idxs) > 1:
            self.this_tracking_idxs = this_tracking_idxs
        else:
            self.this_tracking_idxs = None

    def get_sft_and_sidecar(self):
        sidecar_info = {}
        sidecar_info["bundle_idx"] = {}
        for bundle_name in self.bundle_names:
            sidecar_info["bundle_idx"][f"{bundle_name}"] = self.idxs[bundle_name]
        if self.this_tracking_idxs is not None:
            sidecar_info["tracking_idx"] = self.this_tracking_idxs
        return self.sft, sidecar_info

    def get_bundle(self, b_name):
        return self.sft[self.bundle_idxs[b_name]]

    @classmethod
    def fromfile(cls, trk_file, reference, sidecar_file=None):
        if sidecar_file is None:
            # assume json sidecar has the same name as trk_file,
            # but with json suffix
            sidecar_file = trk_file.split('.')[0] + '.json'
        sidecar_info = read_json(sidecar_file)
        tgram = nib.streamlines.load(trk_file)
        bundles = {}
        for b_name, b_idxs in sidecar_info["bundle_idx"].items():
            if not b_name == "whole_brain":
                bundles[b_name] = StatefulTractogram(
                    tgram.streamlines[b_idxs].copy(), reference, Space.VOX)
        return cls(bundles, reference)


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
