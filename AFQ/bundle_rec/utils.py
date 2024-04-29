import numpy as np
import os.path as op
from time import time

import logging

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
import dipy.tracking.streamlinespeed as dps
import dipy.tracking.streamline as dts
from dipy.tracking.distances import bundles_distances_mdf


from AFQ.definitions.mapping import ConformedFnirtMapping


logger = logging.getLogger('AFQ')


def flip_sls(select_sl, idx_to_flip, in_place=False):
    """
    Helper function to flip streamlines
    """
    if in_place:
        flipped_sl = select_sl
    else:
        flipped_sl = [None] * len(select_sl)
    for ii, sl in enumerate(select_sl):
        if idx_to_flip[ii]:
            flipped_sl[ii] = sl[::-1]
        else:
            flipped_sl[ii] = sl
    return flipped_sl


def cut_sls_by_dist(select_sl, roi_dists, roi_idxs,
                    in_place=False):
    """
    Helper function to cut streamlines according to which points
    are closest to certain rois.

    Parameters
    ----------
    select_sl, streamlines to cut
    roi_dists, distances from a given streamline to a given inclusion roi
    roi_idxs, two indices into the list of inclusion rois to use for the cut
    in_place, whether to modify select_sl
    """
    if in_place:
        cut_sl = select_sl
    else:
        cut_sl = [None] * len(select_sl)

    for idx, this_sl in enumerate(select_sl):
        if roi_idxs[0] == -1:
            min0 = 0
        else:
            min0 = int(roi_dists[idx, roi_idxs[0]])
        if roi_idxs[1] == -1:
            min1 = len(this_sl)
        else:
            min1 = int(roi_dists[idx, roi_idxs[1]])

        # handle if sls not flipped
        if min0 > min1:
            min0, min1 = min1, min0

        # If the point that is closest to the first ROI
        # is the same as the point closest to the second ROI,
        # include the surrounding points to make a streamline.
        if min0 == min1:
            min1 = min1 + 1
            min0 = min0 - 1

        cut_sl[idx] = this_sl[min0:min1]

    return cut_sl


def read_tg(tg, nb_streamlines=None):
    if nb_streamlines and len(tg) > nb_streamlines:
        tg = StatefulTractogram.from_sft(
            dts.select_random_set_of_streamlines(
                tg.streamlines,
                nb_streamlines
            ),
            tg)
    return tg


def orient_by_streamline(sls, template_sl):
    DM = bundles_distances_mdf(
        sls,
        [template_sl, template_sl[::-1]])
    return DM[:, 0] > DM[:, 1]


def move_streamlines(tg, to, mapping, img, save_intermediates=None):
    """Move streamlines to or from template space.

    to : str
        Either "template" or "subject".
    mapping : ConformedMapping
        Mapping to use to move streamlines.
    img : Nifti1Image
        Space to move streamlines to.
    """
    tg_og_space = tg.space
    if isinstance(mapping, ConformedFnirtMapping):
        if to != "subject":
            raise ValueError(
                "Attempted to transform streamlines to template using "
                "unsupported mapping. "
                "Use something other than Fnirt.")
        tg.to_vox()
        moved_sl = []
        for sl in tg.streamlines:
            moved_sl.append(mapping.transform_inverse_pts(sl))
    else:
        tg.to_rasmm()
        if to == "template":
            volume = mapping.forward
        else:
            volume = mapping.backward
        delta = dts.values_from_volume(
            volume,
            tg.streamlines, np.eye(4))
        moved_sl = dts.Streamlines(
            [d + s for d, s in zip(delta, tg.streamlines)])
    moved_sft = StatefulTractogram(
        moved_sl,
        img,
        Space.RASMM)
    if save_intermediates is not None:
        save_tractogram(
            moved_sft,
            op.join(save_intermediates,
                    f'sls_in_{to}.trk'),
            bbox_valid_check=False)
    tg.to_space(tg_og_space)
    return moved_sft


def resample_tg(tg, n_points):
    # reformat for dipy's set_number_of_points
    if isinstance(tg, np.ndarray):
        if len(tg.shape) > 2:
            streamlines = tg.tolist()
            streamlines = [np.asarray(item) for item in streamlines]
    elif hasattr(tg, "streamlines"):
        streamlines = tg.streamlines
    else:
        streamlines = tg

    return dps.set_number_of_points(streamlines, n_points)


class SlsBeingRecognized:
    def __init__(self, sls, logger, save_intermediates, b_name, ref,
                 n_roi_dists):
        self.oriented_yet = False
        self.selected_fiber_idxs = np.arange(len(sls), dtype=np.uint32)
        self.sls_flipped = np.zeros(len(sls), dtype=np.bool8)
        self.logger = logger
        self.start_time = -1
        self.save_intermediates = save_intermediates
        self.b_name = b_name
        self.ref_sls = sls
        self.ref = ref
        self.n_roi_dists = n_roi_dists

    def initiate_selection(self, clean_name):
        self.start_time = time()
        self.logger.info(f"Filtering by {clean_name}")
        return np.zeros(len(self.selected_fiber_idxs), dtype=np.bool8)

    def select(self, idx, clean_name, cut=False):
        self.selected_fiber_idxs = self.selected_fiber_idxs[idx]
        self.sls_flipped = self.sls_flipped[idx]
        if hasattr(self, "roi_dists"):
            self.roi_dists = self.roi_dists[idx]
        time_taken = time() - self.start_time
        self.logger.info(
            f"After filtering by {clean_name} (time: {time_taken}s), "
            f"{len(self)} streamlines remain.")
        if self.save_intermediates is not None:
            save_tractogram(
                StatefulTractogram(
                    self.get_selected_sls(cut=cut),
                    self.ref, Space.VOX),
                op.join(self.save_intermediates,
                        f'sls_after_{clean_name}_for_{self.b_name}.trk'),
                bbox_valid_check=False)

    def get_selected_sls(self, cut=False, flip=False):
        selected_sls = self.ref_sls[self.selected_fiber_idxs]
        if cut and hasattr(self, "roi_dists") and self.n_roi_dists > 1:
            selected_sls = cut_sls_by_dist(
                selected_sls, self.roi_dists,
                (0, self.n_roi_dists - 1),
                in_place=False)
        if flip:
            selected_sls = flip_sls(
                selected_sls, self.sls_flipped,
                in_place=False)
        return selected_sls

    def reorient(self, idx):
        if self.oriented_yet:
            raise RuntimeError((
                "Attempted to oriented streamlines "
                "that were already oriented. "
                "This is a bug in the implementation of a "
                "bundle recognition procedure. "))
        self.oriented_yet = True
        self.sls_flipped[idx] = True

    def __bool__(self):
        return len(self) > 0

    def __len__(self):
        return len(self.selected_fiber_idxs)
