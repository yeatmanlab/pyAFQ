import os.path as op
import os
import logging
from time import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from scipy.ndimage import binary_dilation

import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps
from dipy.segment.bundles import RecoBundles
from dipy.stats.analysis import gaussian_weights
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram, load_tractogram
from dipy.utils.parallel import paramap
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.featurespeed import ResampleFeature

import AFQ.utils.models as ut
import AFQ.data.fetch as afd
from AFQ.data.utils import BUNDLE_RECO_2_AFQ
from AFQ.api.bundle_dict import BundleDict
from AFQ.definitions.mapping import ConformedFnirtMapping
from AFQ._fixes import gaussian_weights

__all__ = ["Segmentation", "clean_bundle", "clean_by_endpoints"]


logger = logging.getLogger('AFQ')


def _resample_tg(tg, n_points):
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


class _SlsBeingRecognized:
    def __init__(self, sls, logger, save_intermediates, b_name, ref,
                 n_roi_dists):
        self.oriented_yet = False
        self.selected_fiber_idxs = np.arange(len(sls), dtype=np.uint32)
        self.sls_flipped = np.zeros(len(sls), dtype=np.bool8)
        self.bundle_vote = np.full(len(sls), -np.inf, dtype=np.float32)
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
        self.bundle_vote = self.bundle_vote[idx]
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
            selected_sls = _cut_sls_by_dist(
                selected_sls, self.roi_dists,
                (0, self.n_roi_dists - 1),
                in_place=False)
        if flip:
            selected_sls = _flip_sls(
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


class Segmentation:
    def __init__(self,
                 nb_points=False,
                 nb_streamlines=False,
                 seg_algo='AFQ',
                 clip_edges=False,
                 parallel_segmentation={"engine": "serial"},
                 progressive=True,
                 greater_than=50,
                 rm_small_clusters=50,
                 model_clust_thr=1.25,
                 reduction_thr=25,
                 refine=False,
                 pruning_thr=12,
                 b0_threshold=50,
                 prob_threshold=0,
                 roi_dist_tie_break=False,
                 dist_to_waypoint=None,
                 rng=None,
                 return_idx=False,
                 presegment_bundle_dict=None,
                 presegment_kwargs={},
                 filter_by_endpoints=True,
                 dist_to_atlas=4,
                 save_intermediates=None,
                 cleaning_params={}):
        """
        Segment streamlines into bundles.

        Parameters
        ----------
        nb_points : int, boolean
            Resample streamlines to nb_points number of points.
            If False, no resampling is done. Default: False
        nb_streamlines : int, boolean
            Subsample streamlines to nb_streamlines.
            If False, no subsampling is don. Default: False
        seg_algo : string
            Algorithm for segmentation (case-insensitive):
            'AFQ': Segment streamlines into bundles,
                based on inclusion/exclusion ROIs.
            'Reco': Segment streamlines using the RecoBundles algorithm
            [Garyfallidis2017].
            Default: 'AFQ'
        clip_edges : bool
            Whether to clip the streamlines to be only in between the ROIs.
            Default: False
        parallel_segmentation : dict or AFQ.api.BundleDict
            How to parallelize segmentation across processes when performing
            waypoint ROI segmentation. Set to {"engine": "serial"} to not
            perform parallelization. Some engines may cause errors, depending
            on the system. See ``dipy.utils.parallel.paramap`` for
            details.
            Default: {"engine": "serial"}
        rm_small_clusters : int
            Using RecoBundles Algorithm.
            Remove clusters that have less than this value
                during whole brain SLR.
            Default: 50
        model_clust_thr : int
            Parameter passed on to recognize for Recobundles.
            See Recobundles documentation.
            Default: 1.25
        reduction_thr : int
            Parameter passed on to recognize for Recobundles.
            See Recobundles documentation.
            Default: 25
        refine : bool
            Parameter passed on to recognize for Recobundles.
            See Recobundles documentation.
            Default: False
        pruning_thr : int
            Parameter passed on to recognize for Recobundles.
            See Recobundles documentation.
            Default: 12
        progressive : boolean, optional
            Using RecoBundles Algorithm.
            Whether or not to use progressive technique
                during whole brain SLR.
            Default: True.
        greater_than : int, optional
            Using RecoBundles Algorithm.
            Keep streamlines that have length greater than this value
                during whole brain SLR.
            Default: 50.
        b0_threshold : float.
            Using AFQ Algorithm.
            All b-values with values less than or equal to `bo_threshold` are
            considered as b0s i.e. without diffusion weighting.
            Default: 50.
        prob_threshold : float.
            Using AFQ Algorithm.
            Initial cleaning of fiber groups is done using probability maps
            from [Hua2008]_. Here, we choose an average probability that
            needs to be exceeded for an individual streamline to be retained.
            Default: 0.
        roi_dist_tie_break : bool.
            Whether to use distance from nearest ROI as a tie breaker when a
            streamline qualifies as a part of multiple bundles. If False,
            probability maps are used.
            Default : False.
        dist_to_waypoint : float.
            The distance that a streamline node has to be from the waypoint
            ROI in order to be included or excluded.
            If set to None (default), will be calculated as the
            center-to-corner distance of the voxel in the diffusion data.
            If a bundle has inc_addtol or exc_addtol in its bundle_dict, that
            tolerance will be added to this distance.
            For example, if you wanted to increase tolerance for the right
            arcuate waypoint ROIs by 3 each, you could make the following
            modification to your bundle_dict:
            bundle_dict["Right Arcuate"]["inc_addtol"] = [3, 3]
            Additional tolerances can also be negative.
        rng : RandomState or int
            If None, creates RandomState.
            If int, creates RandomState with seed rng.
            Used in RecoBundles Algorithm.
            Default: None.
        return_idx : bool
            Whether to return the indices in the original streamlines as part
            of the output of segmentation.
        presegment_bundle_dict : dict or None
            If not None, presegment by ROIs before performing
            RecoBundles. Only used if seg_algo starts with 'Reco'.
            Meta-data for the segmentation. The format is something like::
                {'bundle_name': {
                    'include':[img1, img2],
                    'prob_map': img3,
                    'cross_midline': False,
                    'start': img4,
                    'end': img5}}
            Default: None
        presegment_kwargs : dict
            Optional arguments for initializing the segmentation for the
            presegmentation. Only used if presegment_bundle_dict is not None.
            Default: {}
        filter_by_endpoints: bool
            Whether to filter the bundles based on their endpoints.
            Applies only when `seg_algo == 'AFQ'`.
            Default: True.
        dist_to_atlas : float
            If filter_by_endpoints is True, this is the required distance
            from the endpoints to the atlas ROIs.
        save_intermediates : str, optional
            The full path to a folder into which intermediate products
            are saved. Default: None, means no saving of intermediates.
        cleaning_params : dict, optional
            Cleaning params to pass to seg.clean_bundle. This will
            override the default parameters of that method. However, this
            can be overriden by setting the cleaning parameters in the
            bundle_dict. Default: {}.

        References
        ----------
        .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
        Tract probability maps in stereotaxic spaces: analyses of white
        matter anatomy and tract-specific quantification. Neuroimage 39:
        336-347
        """
        self.logger = logger
        self.nb_points = nb_points
        self.nb_streamlines = nb_streamlines

        if rng is None:
            self.rng = np.random.RandomState()
        elif isinstance(rng, int):
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = rng

        self.seg_algo = seg_algo.lower()
        self.prob_threshold = prob_threshold
        self.roi_dist_tie_break = roi_dist_tie_break
        self.dist_to_waypoint = dist_to_waypoint
        self.b0_threshold = b0_threshold
        self.progressive = progressive
        self.greater_than = greater_than
        self.rm_small_clusters = rm_small_clusters
        self.model_clust_thr = model_clust_thr
        self.reduction_thr = reduction_thr
        self.refine = refine
        self.pruning_thr = pruning_thr
        self.return_idx = return_idx
        self.presegment_bundle_dict = presegment_bundle_dict
        self.presegment_kwargs = presegment_kwargs
        self.filter_by_endpoints = filter_by_endpoints
        self.dist_to_atlas = dist_to_atlas
        self.parallel_segmentation = parallel_segmentation
        self.cleaning_params = cleaning_params

        if (save_intermediates is not None) and \
                (not op.exists(save_intermediates)):
            os.makedirs(save_intermediates, exist_ok=True)
        self.save_intermediates = save_intermediates
        self.clip_edges = clip_edges

    def _read_tg(self, tg=None):
        if tg is None:
            tg = self.tg
        else:
            self.tg = tg
        self._tg_orig_space = self.tg.space

        if self.nb_streamlines and len(self.tg) > self.nb_streamlines:
            self.tg = StatefulTractogram.from_sft(
                dts.select_random_set_of_streamlines(
                    self.tg.streamlines,
                    self.nb_streamlines
                ),
                self.tg
            )

        return tg

    def segment(self, bundle_dict, tg, mapping, img,
                reg_prealign=None,
                reg_template=None, reset_tg_space=False):
        """
        Segment streamlines into bundles based on either waypoint ROIs
        [Yeatman2012]_ or RecoBundles [Garyfallidis2017]_.
        Parameters
        ----------
        bundle_dict: dict or AFQ.api.BundleDict
            Meta-data for the segmentation. The format is something like::
                {'bundle_name': {
                    'include':[img1, img2],
                    'prob_map': img3,
                    'cross_midline': False,
                    'start': img4,
                    'end': img5}}
        tg : StatefulTractogram
            Bundles to segment
        mapping : DiffeomorphicMap, or equivalent interface
            A mapping between DWI space and a template.
        img : Nifti1Image
            Image to use as reference.
        reg_prealign : array, optional.
            The linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.
            Default: None.
        reg_template : str or nib.Nifti1Image, optional.
            Template to use for registration. Default: MNI T2.
        reset_tg_space : bool, optional
            Whether to reset the space of the input tractogram after
            segmentation is complete. Default: False.

        Returns
        -------
        dict : Where keys are bundle names, values are tractograms of
            these bundles.

        References
        ----------
        .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty, Nathaniel J.
        Myall, Brian A. Wandell, and Heidi M. Feldman. 2012. "Tract Profiles of
        White Matter Properties: Automating Fiber-Tract Quantification"
        PloS One 7 (11): e49790.
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
        """
        self.img_affine = img.affine
        self.img = img
        self.logger.info("Preprocessing Streamlines")
        tg = self._read_tg(tg)

        # These are calculated as-needed
        self._fg_array = None
        self._crosses = None

        # If resampling over-write the sft:
        if self.nb_points:
            self.tg = StatefulTractogram(
                dps.set_number_of_points(self.tg.streamlines, self.nb_points),
                self.tg, self.tg.space)

        if reg_template is None:
            reg_template = afd.read_mni_template()

        self.reg_prealign = reg_prealign
        self.reg_template = reg_template
        self.mapping = mapping
        self.bundle_dict = bundle_dict
        if not isinstance(self.bundle_dict, BundleDict):
            self.bundle_dict = BundleDict(self.bundle_dict)

        if self.seg_algo == "afq":
            fiber_groups = self.segment_afq()
        elif self.seg_algo.startswith("reco"):
            fiber_groups = self.segment_reco()
        else:
            raise ValueError(f"The seg_algo input is {self.seg_algo}, which",
                             "is not recognized")
        if reset_tg_space:
            # Return the input to the original space when you are done:
            self.tg.to_space(self._tg_orig_space)

        return fiber_groups

    @property
    def fgarray(self):
        """
        Streamlines resampled to 20 points.
        """
        if self._fg_array is None:
            self.logger.info("Resampling Streamlines...")
            start_time = time()
            self._fg_array = np.array(_resample_tg(self.tg, 20))
            self.logger.info((
                "Streamlines Resampled "
                f"(time: {time()-start_time}s)"))
        return self._fg_array

    @property
    def crosses(self):
        """
        Classify the streamlines by whether they cross the midline.
        Creates a crosses attribute which is an array of booleans. Each boolean
        corresponds to a streamline, and is whether or not that streamline
        crosses the midline.
        """
        if self._crosses is None:
            # What is the x,y,z coordinate of 0,0,0 in the template space?
            zero_coord = np.dot(np.linalg.inv(self.img_affine),
                                np.array([0, 0, 0, 1]))

            self._crosses = np.logical_and(
                np.any(self.fgarray[:, :, 0] > zero_coord[0], axis=1),
                np.any(self.fgarray[:, :, 0] < zero_coord[0], axis=1))
        return self._crosses

    def _return_empty(self, bundle):
        """
        Helper function for segment_afq, to return an empty dict under
        some conditions.
        """

        if self.return_idx:
            self.fiber_groups[bundle] = {}
            self.fiber_groups[bundle]['sl'] = StatefulTractogram(
                [], self.img, Space.VOX)
            self.fiber_groups[bundle]['idx'] = np.array([])
        else:
            self.fiber_groups[bundle] = StatefulTractogram(
                [], self.img, Space.VOX)

    def _add_bundle_to_fiber_group(self, b_name, sl, idx, to_flip):
        """
        Helper function for segment_afq, to add a bundle
        to a fiber group.
        """
        sl = _flip_sls(
            sl, to_flip,
            in_place=False)

        sl = StatefulTractogram(
            sl,
            self.img,
            Space.VOX)

        if self.return_idx:
            self.fiber_groups[b_name] = {}
            self.fiber_groups[b_name]['sl'] = sl
            self.fiber_groups[b_name]['idx'] = idx
        else:
            self.fiber_groups[b_name] = sl

    def _add_bundle_to_meta(self, bundle_name, bundle_info):
        bundle_info = bundle_info.copy()
        # remove keys that can never be serialized
        for key in [
                'include', 'exclude', 'prob_map',
                'start', 'end', 'curvature']:
            bundle_info.pop(key, None)
        self.meta[bundle_name] = bundle_info

    def segment_afq(self, tg=None):
        """
        Assign streamlines to bundles using the waypoint ROI approach
        Parameters
        ----------
        tg : StatefulTractogram class instance
        """
        tg = self._read_tg(tg=tg)
        tg.to_vox()

        n_streamlines = len(tg)

        bundle_votes = np.full(
            (n_streamlines, len(self.bundle_dict)),
            -np.inf, dtype=np.float32)
        bundle_to_flip = np.zeros(
            (n_streamlines, len(self.bundle_dict)),
            dtype=np.bool8)

        bundle_roi_dists = -np.ones(
            (
                n_streamlines,
                len(self.bundle_dict),
                self.bundle_dict.max_includes),
            dtype=np.uint32)

        self.fiber_groups = {}
        self.meta = {}

        # We need to calculate the size of a voxel, so we can transform
        # from mm to voxel units:
        R = self.img_affine[0:3, 0:3]
        vox_dim = np.mean(np.diag(np.linalg.cholesky(R.T.dot(R))))

        # Tolerance is set to the square of the distance to the corner
        # because we are using the squared Euclidean distance in calls to
        # `cdist` to make those calls faster.
        if self.dist_to_waypoint is None:
            tol = dts.dist_to_corner(self.img_affine)
        else:
            tol = self.dist_to_waypoint / vox_dim
        dist_to_atlas = int(self.dist_to_atlas / vox_dim)

        self.logger.info("Assigning Streamlines to Bundles")
        for bundle_idx, bundle_name in enumerate(
                self.bundle_dict.bundle_names):
            self.logger.info(f"Finding Streamlines for {bundle_name}")

            # Warp ROIs
            self.logger.info(f"Preparing ROIs for {bundle_name}")
            start_time = time()
            bundle_def = dict(self.bundle_dict.get_b_info(bundle_name))
            bundle_def.update(self.bundle_dict.transform_rois(
                bundle_name,
                self.mapping,
                self.img_affine))
            self.logger.info(f"Time to prep ROIs: {time()-start_time}s")

            if "curvature" in bundle_def:
                self.logger.info(f"Loading curvature...")
                start_time = time()
                if "sft" in bundle_def["curvature"]:
                    ref_sl = bundle_def["curvature"]["sft"]
                else:
                    ref_sl = load_tractogram(
                        bundle_def["curvature"]["path"], "same",
                        bbox_valid_check=False)
                moved_ref_sl = self.move_streamlines(
                    ref_sl, "subject")
                moved_ref_sl.to_vox()
                moved_ref_sl = moved_ref_sl.streamlines[0]
                moved_ref_curve = sl_curve(
                    moved_ref_sl,
                    len(moved_ref_sl))
                self.logger.info((
                    "Time to load curves: "
                    f"{time()-start_time}s"))

            b_sls = _SlsBeingRecognized(
                tg.streamlines, self.logger,
                self.save_intermediates, bundle_name,
                self.img, len(bundle_def.get("include", [])))

            # filter by probability map
            if "prob_map" in bundle_def:
                b_sls.initiate_selection("Prob. Map")
                # using entire fgarray here only because it is the first step
                fiber_probabilities = dts.values_from_volume(
                    bundle_def["prob_map"].get_fdata(),
                    self.fgarray, np.eye(4))
                fiber_probabilities = np.mean(fiber_probabilities, -1)
                if not self.roi_dist_tie_break:
                    b_sls.bundle_vote = fiber_probabilities
                b_sls.select(
                    fiber_probabilities > self.prob_threshold,
                    "Prob. Map")
            elif not self.roi_dist_tie_break:
                b_sls.bundle_vote = np.ones(len(b_sls))

            if b_sls and "cross_midline" in bundle_def:
                b_sls.initiate_selection("Cross Mid.")
                accepted = self.crosses[b_sls.selected_fiber_idxs]
                if not bundle_def["cross_midline"]:
                    accepted = np.invert(accepted)
                b_sls.select(accepted, "Cross Mid.")

            if b_sls and "start" in bundle_def:
                accept_idx = b_sls.initiate_selection("startpoint")
                clean_by_endpoints(
                    b_sls.get_selected_sls(),
                    bundle_def["start"],
                    0,
                    tol=dist_to_atlas,
                    flip_sls=b_sls.sls_flipped,
                    accepted_idxs=accept_idx)
                if not b_sls.oriented_yet:
                    accepted_idx_flipped = clean_by_endpoints(
                        b_sls.get_selected_sls(),
                        bundle_def["start"],
                        -1,
                        tol=dist_to_atlas)
                    b_sls.reorient(accepted_idx_flipped)
                    accept_idx = np.logical_xor(
                        accepted_idx_flipped, accept_idx)
                b_sls.select(accept_idx, "startpoint")

            if b_sls and "end" in bundle_def:
                accept_idx = b_sls.initiate_selection("endpoint")
                cleaned_idx = clean_by_endpoints(
                    b_sls.get_selected_sls(),
                    bundle_def["end"],
                    -1,
                    tol=dist_to_atlas,
                    flip_sls=b_sls.sls_flipped,
                    accepted_idxs=accept_idx)
                if not b_sls.oriented_yet:
                    accepted_idx_flipped = clean_by_endpoints(
                        b_sls.get_selected_sls(),
                        bundle_def["end"],
                        0,
                        tol=dist_to_atlas)
                    b_sls.reorient(accepted_idx_flipped)
                    accept_idx = np.logical_xor(
                        accepted_idx_flipped, accept_idx)
                b_sls.select(accept_idx, "endpoint")

            if b_sls and (
                    ("min_len" in bundle_def) or ("max_len" in bundle_def)):
                accept_idx = b_sls.initiate_selection("length")
                min_len = bundle_def.get("min_len", 0) / vox_dim
                max_len = bundle_def.get("max_len", np.inf) / vox_dim
                for idx, sl in enumerate(b_sls.get_selected_sls()):
                    sl_len = np.sum(
                        np.linalg.norm(np.diff(sl, axis=0), axis=1))
                    if sl_len >= min_len and sl_len <= max_len:
                        accept_idx[idx] = 1
                b_sls.select(accept_idx, "length")

            if b_sls and "primary_axis" in bundle_def:
                b_sls.initiate_selection("orientation")
                accept_idx = clean_by_orientation(
                    b_sls.get_selected_sls(),
                    bundle_def["primary_axis"],
                    bundle_def.get(
                        "primary_axis_percentage", None))
                b_sls.select(accept_idx, "orientation")

            if b_sls and "include" in bundle_def:
                accept_idx = b_sls.initiate_selection("include")
                flip_using_include = len(bundle_def["include"]) > 1\
                    and not b_sls.oriented_yet

                if f'inc_addtol' in bundle_def:
                    include_roi_tols = []
                    for inc_tol in bundle_def["inc_addtol"]:
                        include_roi_tols.append((inc_tol / vox_dim + tol)**2)
                else:
                    include_roi_tols = [tol**2] * len(bundle_def["include"])

                include_rois = []
                for include_roi in bundle_def["include"]:
                    include_rois.append(np.array(
                        np.where(include_roi.get_fdata())).T)

                # with parallel segmentation, the first for loop will
                # only collect streamlines and does not need tqdm
                if self.parallel_segmentation["engine"] != "serial":
                    inc_results = paramap(
                        _check_sl_with_inclusion, b_sls.get_selected_sls(),
                        func_args=[
                            include_rois, include_roi_tols],
                        **self.parallel_segmentation)

                else:
                    inc_results = _check_sls_with_inclusion(
                        b_sls.get_selected_sls(),
                        include_rois,
                        include_roi_tols)

                if self.roi_dist_tie_break:
                    min_dist_coords = np.ones(len(b_sls))
                roi_dists = -np.ones(
                    (len(b_sls), self.bundle_dict.max_includes),
                    dtype=np.int32)
                if flip_using_include:
                    to_flip = np.ones_like(accept_idx, dtype=np.bool8)
                for sl_idx, inc_result in enumerate(inc_results):
                    sl_accepted, sl_dist = inc_result

                    if sl_accepted:
                        if self.roi_dist_tie_break:
                            min_dist_coords[sl_idx] = np.min(sl_dist)

                        if len(sl_dist) > 1:
                            roi_dists[sl_idx, :len(sl_dist)] = [
                                np.argmin(dist, 0)[0]
                                for dist in sl_dist]
                            first_roi_idx = roi_dists[sl_idx, 0]
                            last_roi_idx = roi_dists[
                                sl_idx, len(sl_dist) - 1]
                            # Only accept SLs that, when cut, are meaningful
                            if (len(sl_dist) < 2) or abs(
                                    first_roi_idx - last_roi_idx) > 1:
                                # Flip sl if it is close to second ROI
                                # before its close to the first ROI
                                if flip_using_include:
                                    to_flip[sl_idx] =\
                                        first_roi_idx > last_roi_idx
                                    if to_flip[sl_idx]:
                                        roi_dists[sl_idx, :len(sl_dist)] =\
                                            np.flip(roi_dists[
                                                sl_idx, :len(sl_dist)])
                                accept_idx[sl_idx] = 1
                        else:
                            accept_idx[sl_idx] = 1
                # see https://github.com/joblib/joblib/issues/945
                if (
                    (self.parallel_segmentation.get(
                        "engine", "joblib") != "serial")
                    and (self.parallel_segmentation.get(
                        "backend", "loky") == "loky")):
                    from joblib.externals.loky import get_reusable_executor
                    get_reusable_executor().shutdown(wait=True)
                if self.roi_dist_tie_break:
                    b_sls.bundle_vote = -min_dist_coords
                b_sls.roi_dists = roi_dists
                if flip_using_include:
                    b_sls.reorient(to_flip)
                b_sls.select(accept_idx, "include")

            # Filters streamlines by how well they match
            # a curve in orientation and shape but not scale
            if b_sls and "curvature" in bundle_def:
                accept_idx = b_sls.initiate_selection("curvature")
                ref_curve_threshold = np.radians(bundle_def["curvature"].get(
                    "thresh", 10))
                cut = bundle_def["curvature"].get("cut", True)
                for idx, sl in enumerate(b_sls.get_selected_sls(
                        cut=cut, flip=True)):
                    if len(sl) > 1:
                        this_sl_curve = sl_curve(sl, len(moved_ref_sl))
                        dist = sl_curve_dist(this_sl_curve, moved_ref_curve)
                        if dist <= ref_curve_threshold:
                            accept_idx[idx] = 1
                b_sls.select(accept_idx, "curvature", cut=cut)

            if b_sls and "exclude" in bundle_def:
                accept_idx = b_sls.initiate_selection("exclude")
                if f'exc_addtol' in bundle_def:
                    exclude_roi_tols = []
                    for exc_tol in bundle_def["exc_addtol"]:
                        exclude_roi_tols.append((exc_tol / vox_dim + tol)**2)
                else:
                    exclude_roi_tols = [tol**2] * len(bundle_def["exclude"])
                exclude_rois = []
                for exclude_roi in bundle_def["exclude"]:
                    exclude_rois.append(np.array(
                        np.where(exclude_roi.get_fdata())).T)
                for sl_idx, sl in enumerate(b_sls.get_selected_sls()):
                    if _check_sl_with_exclusion(
                            sl, exclude_rois, exclude_roi_tols):
                        accept_idx[sl_idx] = 1
                b_sls.select(accept_idx, "exclude")

            if b_sls and "qb_thresh" in bundle_def:
                b_sls.initiate_selection("qb_thresh")
                cut = self.clip_edges or ("bundlesection" in bundle_def)
                qbx = QuickBundles(
                    bundle_def["qb_thresh"] / vox_dim,
                    AveragePointwiseEuclideanMetric(
                        ResampleFeature(nb_points=12)))
                clusters = qbx.cluster(b_sls.get_selected_sls(
                    cut=cut, flip=True))
                cleaned_idx = clusters[np.argmax(
                    clusters.clusters_sizes())].indices
                b_sls.select(cleaned_idx, "qb_thresh", cut=cut)

            if b_sls:
                accept_idx = b_sls.initiate_selection("Mahalanobis")
                clean_params = bundle_def.get("mahal", {})
                clean_params = {
                    **self.cleaning_params,
                    **clean_params}
                clean_params["return_idx"] = True
                cut = self.clip_edges or ("bundlesection" in bundle_def)
                _, cleaned_idx = clean_bundle(
                    b_sls.get_selected_sls(cut=cut, flip=True),
                    **clean_params)
                b_sls.select(cleaned_idx, "Mahalanobis", cut=cut)

            if b_sls and not b_sls.oriented_yet:
                raise ValueError(
                    "pyAFQ was unable to consistently orient streamlines "
                    f"in bundle {bundle_name} using the provided ROIs. "
                    "This can be fixed by including at least 2 "
                    "waypoint ROIs, or by using "
                    "endpoint ROIs.")

            if b_sls:
                bundle_votes[
                    b_sls.selected_fiber_idxs,
                    bundle_idx] = b_sls.bundle_vote.copy()
                bundle_to_flip[
                    b_sls.selected_fiber_idxs,
                    bundle_idx] = b_sls.sls_flipped.copy()
                if hasattr(b_sls, "roi_dists"):
                    bundle_roi_dists[
                        b_sls.selected_fiber_idxs,
                        bundle_idx
                    ] = b_sls.roi_dists.copy()

        if self.save_intermediates is not None:
            os.makedirs(self.save_intermediates, exist_ok=True)
            bc_path = op.join(self.save_intermediates,
                              "sls_bundle_votes.npy")
            np.save(bc_path, bundle_votes)

        bundle_choice = np.argmax(bundle_votes, -1)
        bundle_choice[bundle_votes.max(-1) == -np.inf] = -1

        # We do another round through, so that we can orient all the
        # streamlines within a bundle in the same orientation with respect to
        # the ROIs. This order is ARBITRARY but CONSISTENT (going from ROI0
        # to ROI1).
        self.logger.info("Re-orienting streamlines to consistent directions")
        for bundle_idx, bundle in enumerate(self.bundle_dict.bundle_names):
            self.logger.info(f"Processing {bundle}")

            select_idx = np.where(bundle_choice == bundle_idx)[0]

            if len(select_idx) == 0:
                # There's nothing here, set and move to the next bundle:
                if "bundlesection" in self.bundle_dict.get_b_info(bundle):
                    for sb_name in self.bundle_dict.get_b_info(bundle)[
                            "bundlesection"]:
                        self._return_empty(sb_name)
                else:
                    self._return_empty(bundle)
                continue

            # Use a list here, because ArraySequence doesn't support item
            # assignment:
            select_sl = list(tg.streamlines[select_idx])
            roi_dists = bundle_roi_dists[select_idx, bundle_idx, :]
            n_includes = len(self.bundle_dict.get_b_info(
                bundle).get("include", []))
            if self.clip_edges and n_includes > 1:
                self.logger.info("Clipping Streamlines by ROI")
                _cut_sls_by_dist(
                    select_sl, roi_dists,
                    (0, n_includes - 1), in_place=True)

            to_flip = bundle_to_flip[select_idx, bundle_idx]
            if "bundlesection" in self.bundle_dict[bundle]:
                for sb_name, sb_include_cuts in self.bundle_dict.get_b_info(
                        bundle)["bundlesection"].items():
                    bundlesection_select_sl = _cut_sls_by_dist(
                        select_sl, roi_dists,
                        sb_include_cuts, in_place=False)
                    self._add_bundle_to_fiber_group(
                        sb_name, bundlesection_select_sl, select_idx, to_flip)
                    self._add_bundle_to_meta(sb_name, bundle_def)
            else:
                self._add_bundle_to_fiber_group(
                    bundle, select_sl, select_idx, to_flip)
                self._add_bundle_to_meta(bundle, bundle_def)
        return self.fiber_groups, self.meta

    def move_streamlines(self, tg, to="template"):
        """Streamline-based registration of a whole-brain tractogram to
        the MNI whole-brain atlas.

        to : str
            "template" or "subject"
        """
        tg_og_space = tg.space
        if isinstance(self.mapping, ConformedFnirtMapping):
            if to != "subject":
                raise ValueError(
                    "Attempted to transform streamlines to template using "
                    "unsupported mapping. "
                    "Use something other than Fnirt.")
            tg.to_vox()
            moved_sl = []
            for sl in tg.streamlines:
                moved_sl.append(self.mapping.transform_inverse_pts(sl))
        else:
            tg.to_rasmm()
            if to == "template":
                volume = self.mapping.forward
            else:
                volume = self.mapping.backward
            delta = dts.values_from_volume(
                volume,
                tg.streamlines, np.eye(4))
            moved_sl = dts.Streamlines(
                [d + s for d, s in zip(delta, tg.streamlines)])
        if to == "template":
            ref = self.reg_template
        else:
            ref = self.img
        moved_sft = StatefulTractogram(
            moved_sl,
            ref,
            Space.RASMM)
        if self.save_intermediates is not None:
            save_tractogram(
                moved_sft,
                op.join(self.save_intermediates,
                        f'sls_in_{to}.trk'),
                bbox_valid_check=False)
        tg.to_space(tg_og_space)
        return moved_sft

    def segment_reco(self, tg=None):
        """
        Segment streamlines using the RecoBundles algorithm [Garyfallidis2017]
        Parameters
        ----------
        tg : StatefulTractogram class instance
            A whole-brain tractogram to be segmented.
        Returns
        -------
        fiber_groups : dict
            Keys are names of the bundles, values are Streamline objects.
            The streamlines in each object have all been oriented to have the
            same orientation (using `dts.orient_by_streamline`).
        """
        tg = self._read_tg(tg=tg)
        fiber_groups = {}

        # We generate our instance of RB with the moved streamlines:
        self.logger.info("Extracting Bundles")
        # If doing a presegmentation based on ROIs then initialize
        # that segmentation and segment using ROIs, else
        # RecoBundles based on the whole brain tractogram
        if self.presegment_bundle_dict is not None:
            roiseg = Segmentation(**self.presegment_kwargs)
            roiseg.segment(
                self.presegment_bundle_dict,
                self.tg,
                self.mapping,
                self.img,
                reg_template=self.reg_template,
                reg_prealign=self.reg_prealign)
            roiseg_fg = roiseg.fiber_groups
        else:
            moved_sl = self.move_streamlines(tg).streamlines
            rb = RecoBundles(moved_sl, verbose=False, rng=self.rng)
        # Next we'll iterate over bundles, registering each one:
        bundle_list = list(self.bundle_dict.keys())
        if 'whole_brain' in bundle_list:
            bundle_list.remove('whole_brain')

        self.logger.info("Assigning Streamlines to Bundles")
        for bundle in bundle_list:
            self.logger.info(f"Finding streamlines for {bundle}")
            b_info = self.bundle_dict[bundle]
            model_sl = b_info['sl']

            # If doing a presegmentation based on ROIs then initialize rb after
            # Filtering the whole brain tractogram to pass through ROIs
            if self.presegment_bundle_dict is not None:
                afq_bundle_name = BUNDLE_RECO_2_AFQ.get(bundle, bundle)
                if "return_idx" in self.presegment_kwargs\
                        and self.presegment_kwargs["return_idx"]:
                    indiv_tg = roiseg_fg[afq_bundle_name]['sl']
                else:
                    indiv_tg = roiseg_fg[afq_bundle_name]

                if len(indiv_tg.streamlines) < 1:
                    self.logger.warning((
                        f"No streamlines found by waypoint ROI "
                        f"pre-segmentation for {bundle}. Using entire"
                        f" tractography instead."))
                    indiv_tg = tg

                # Now rb should be initialized based on the fiber group coming
                # out of the roi segmentation
                indiv_tg = StatefulTractogram(
                    indiv_tg.streamlines,
                    self.img,
                    Space.VOX)
                indiv_tg.to_rasmm()
                moved_sl = self.move_streamlines(indiv_tg).streamlines
                rb = RecoBundles(
                    moved_sl,
                    verbose=False,
                    rng=self.rng)
            if self.save_intermediates is not None:
                if self.presegment_bundle_dict is not None:
                    moved_fname = f"{bundle}_presegmentation.trk"
                else:
                    moved_fname = "whole_brain.trk"
                moved_sft = StatefulTractogram(
                    moved_sl,
                    self.reg_template,
                    Space.RASMM)
                save_tractogram(
                    moved_sft,
                    op.join(self.save_intermediates,
                            moved_fname),
                    bbox_valid_check=False)
                model_sft = StatefulTractogram(
                    model_sl,
                    self.reg_template,
                    Space.RASMM)
                save_tractogram(
                    model_sft,
                    op.join(self.save_intermediates,
                            f"{bundle}_model.trk"),
                    bbox_valid_check=False)

            # Either whole brain tracgtogram or roi presegmented fiber group
            # goes to rb.recognize
            _, rec_labels = rb.recognize(model_bundle=model_sl,
                                         model_clust_thr=self.model_clust_thr,
                                         reduction_thr=self.reduction_thr,
                                         reduction_distance='mdf',
                                         slr=True,
                                         slr_metric='asymmetric',
                                         pruning_distance='mdf')

            # Use the streamlines in the original space:
            if self.presegment_bundle_dict is None:
                recognized_sl = tg.streamlines[rec_labels]
            else:
                recognized_sl = indiv_tg.streamlines[rec_labels]
            if self.refine and len(recognized_sl) > 0:
                _, rec_labels = rb.refine(model_sl, recognized_sl,
                                          self.model_clust_thr,
                                          reduction_thr=self.reduction_thr,
                                          pruning_thr=self.pruning_thr)
                if self.presegment_bundle_dict is None:
                    recognized_sl = tg.streamlines[rec_labels]
                else:
                    recognized_sl = indiv_tg.streamlines[rec_labels]
            standard_sl = next(iter(b_info['centroid']))
            oriented_sl = dts.orient_by_streamline(recognized_sl, standard_sl)

            self.logger.info(
                f"{len(oriented_sl)} streamlines selected with Recobundles")
            if self.return_idx:
                fiber_groups[bundle] = {}
                fiber_groups[bundle]['idx'] = rec_labels
                fiber_groups[bundle]['sl'] = StatefulTractogram(oriented_sl,
                                                                self.img,
                                                                Space.RASMM)
            else:
                fiber_groups[bundle] = StatefulTractogram(oriented_sl,
                                                          self.img,
                                                          Space.RASMM)
        self.fiber_groups = fiber_groups
        return fiber_groups, {}


def sl_curve(sl, n_points):
    """
    Calculate the direction of the displacement between
    each point along a streamline

    Parameters
    ----------
    sl : 2d array-like
        Streamline to calcualte displacements for.
    n_points : int
        Number of points to resample the streamline to

    Returns
    -------
    2d array of shape (len(sl)-1, 3) with displacements
    between each point in sl normalized to 1.
    """
    # Resample to a standardized number of points
    resampled_sl = dps.set_number_of_points(
        sl,
        n_points)

    # displacement at each point
    resampled_sl_diff = np.diff(resampled_sl, axis=0)

    # normalize this displacement
    resampled_sl_diff = resampled_sl_diff / np.linalg.norm(
        resampled_sl_diff, axis=1)[:, None]

    return resampled_sl_diff


def sl_curve_dist(curve1, curve2):
    """
    Calculate the mean angle using the directions of displacement
    between two streamlines

    Parameters
    ----------
    curve1, curve2 : 2d array-like
        Two curves calculated from sl_curve.

    Returns
    -------
    The mean angle between each curve across all steps, in radians
    """
    return np.mean(np.arccos(np.sum(curve1 * curve2, axis=1)))


def clean_bundle(tg, n_points=100, clean_rounds=5, distance_threshold=3,
                 length_threshold=4, min_sl=20, stat='mean',
                 return_idx=False):
    """
    Clean a segmented fiber group based on the Mahalnobis distance of
    each streamline

    Parameters
    ----------
    tg : StatefulTractogram class instance or ArraySequence
        A whole-brain tractogram to be segmented.
    n_points : int, optional
        Number of points to resample streamlines to.
        Default: 100
    clean_rounds : int, optional.
        Number of rounds of cleaning based on the Mahalanobis distance from
        the mean of extracted bundles. Default: 5
    distance_threshold : float, optional.
        Threshold of cleaning based on the Mahalanobis distance (the units are
        standard deviations). Default: 3.
    length_threshold: float, optional
        Threshold for cleaning based on length (in standard deviations). Length
        of any streamline should not be *more* than this number of stdevs from
        the mean length.
    min_sl : int, optional.
        Number of streamlines in a bundle under which we will
        not bother with cleaning outliers. Default: 20.
    stat : callable or str, optional.
        The statistic of each node relative to which the Mahalanobis is
        calculated. Default: `np.mean` (but can also use median, etc.)
    return_idx : bool
        Whether to return indices in the original streamlines.
        Default: False.
    Returns
    -------
    A StatefulTractogram class instance containing only the streamlines
    that have a Mahalanobis distance smaller than `clean_threshold` from
    the mean of each one of the nodes.
    """
    # Convert string to callable, if that's what you got.
    if isinstance(stat, str):
        stat = getattr(np, stat)

    if hasattr(tg, "streamlines"):
        streamlines = tg.streamlines
    else:
        streamlines = dts.Streamlines(tg)

    # We don't even bother if there aren't enough streamlines:
    if len(streamlines) < min_sl:
        if return_idx:
            return tg, np.arange(len(streamlines))
        else:
            return tg

    # Resample once up-front:
    fgarray = np.asarray(_resample_tg(streamlines, n_points))

    # Keep this around, so you can use it for indexing at the very end:
    idx = np.arange(len(fgarray))
    # get lengths of each streamline
    lengths = np.array([sl.shape[0] for sl in streamlines])
    # We'll only do this for clean_rounds
    rounds_elapsed = 0
    idx_belong = idx
    while (rounds_elapsed < clean_rounds) and (np.sum(idx_belong) > min_sl):
        # Update by selection:
        idx = idx[idx_belong]
        fgarray = fgarray[idx_belong]
        lengths = lengths[idx_belong]
        rounds_elapsed += 1

        # This calculates the Mahalanobis for each streamline/node:
        m_dist = gaussian_weights(
            fgarray, return_mahalnobis=True,
            n_points=n_points, stat=stat)
        logger.debug(f"Shape of fgarray: {np.asarray(fgarray).shape}")
        logger.debug(f"Shape of m_dist: {m_dist.shape}")
        logger.debug(f"Maximum m_dist: {np.max(m_dist)}")
        logger.debug((
            f"Maximum m_dist for each fiber: "
            f"{np.max(m_dist, axis=1)}"))

        length_z = zscore(lengths)
        logger.debug(f"Shape of length_z: {length_z.shape}")
        logger.debug(f"Maximum length_z: {np.max(length_z)}")
        logger.debug((
            "length_z for each fiber: "
            f"{length_z}"))

        if not (
                np.any(m_dist > distance_threshold)
                or np.any(length_z > length_threshold)):
            break
        # Select the fibers that have Mahalanobis smaller than the
        # threshold for all their nodes:
        idx_dist = np.all(m_dist < distance_threshold, axis=-1)
        idx_len = length_z < length_threshold
        idx_belong = np.logical_and(idx_dist, idx_len)

        if np.sum(idx_belong) < min_sl:
            # need to sort and return exactly min_sl:
            idx_belong = np.argsort(np.sum(
                m_dist, axis=-1))[:min_sl].astype(int)
            logger.debug((
                f"At rounds elapsed {rounds_elapsed}, "
                "minimum streamlines reached"))
        else:
            idx_removed = idx_belong == 0
            logger.debug((
                f"Rounds elapsed: {rounds_elapsed}, "
                f"num removed: {np.sum(idx_removed)}"))
            logger.debug(f"Removed indicies: {np.where(idx_removed)[0]}")

    # Select based on the variable that was keeping track of things for us:
    if hasattr(tg, "streamlines"):
        out = StatefulTractogram(tg.streamlines[idx], tg, Space.VOX)
    else:
        out = streamlines[idx]
    if return_idx:
        return out, idx
    else:
        return out

# Helper functions for segmenting using waypoint ROIs
# they are not a part of the class because we do not want
# copies of the class to be parallelized


def _check_sls_with_inclusion(sls, include_rois, include_roi_tols):
    for sl in sls:
        yield _check_sl_with_inclusion(
            sl,
            include_rois,
            include_roi_tols)


def _check_sl_with_inclusion(sl, include_rois,
                             include_roi_tols):
    """
    Helper function to check that a streamline is close to a list of
    inclusion ROIS.
    """
    dist = []
    for ii, roi in enumerate(include_rois):
        # Use squared Euclidean distance, because it's faster:
        dist.append(cdist(sl, roi, 'sqeuclidean'))
        if np.min(dist[-1]) > include_roi_tols[ii]:
            # Too far from one of them:
            return False, []
    # Apparently you checked all the ROIs and it was close to all of them
    return True, dist


def _check_sl_with_exclusion(sl, exclude_rois,
                             exclude_roi_tols):
    """ Helper function to check that a streamline is not too close to a
    list of exclusion ROIs.
    """
    for ii, roi in enumerate(exclude_rois):
        # Use squared Euclidean distance, because it's faster:
        if np.min(cdist(sl, roi, 'sqeuclidean')) < exclude_roi_tols[ii]:
            return False
    # Either there are no exclusion ROIs, or you are not close to any:
    return True


def _flip_sls(select_sl, idx_to_flip, in_place=False):
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


def _cut_sls_by_dist(select_sl, roi_dists, roi_idxs,
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


def clean_by_orientation(streamlines, primary_axis, tol=None):
    """
    Compute the cardinal orientation of each streamline

    Parameters
    ----------
    streamlines : sequence of N by 3 arrays
        Where N is number of nodes in the array, the collection of
        streamlines to filter down to.

    Returns
    -------
    cleaned_idx, indicies of streamlines that passed cleaning,
        logical_and of other two returns
    along_accepted_idx, indices of streamlines that passed
        cleaning along the bundle
    end_accepted_idx, indices of streamlines that passed
        cleaning based on difference between endpoints of bundle
    """
    axis_diff = np.zeros((len(streamlines), 3))
    endpoint_diff = np.zeros((len(streamlines), 3))
    for ii, sl in enumerate(streamlines):
        # endpoint diff is between first and last
        endpoint_diff[ii, :] = np.abs(sl[0, :] - sl[-1, :])
        # axis diff is difference between the nodes, along
        axis_diff[ii, :] = np.sum(np.abs(np.diff(sl, axis=0)), axis=0)

    orientation_along = np.argmax(axis_diff, axis=1)
    along_accepted_idx = orientation_along == primary_axis
    if tol is not None:
        percentage_primary = 100 * axis_diff[:, primary_axis] / np.sum(
            axis_diff, axis=1)
        logger.debug((
            "Maximum primary percentage found: "
            f"{np.max(percentage_primary)}"))
        along_accepted_idx = np.logical_and(
            along_accepted_idx, percentage_primary > tol)

    orientation_end = np.argmax(endpoint_diff, axis=1)
    end_accepted_idx = orientation_end == primary_axis

    cleaned_idx = np.logical_and(
        along_accepted_idx,
        end_accepted_idx)

    return cleaned_idx


def clean_by_endpoints(streamlines, target, target_idx, tol=0,
                       flip_sls=None, accepted_idxs=None):
    """
    Clean a collection of streamlines based on an endpoint ROI.
    Filters down to only include items that have their start or end points
    close to the targets.
    Parameters
    ----------
    streamlines : sequence of N by 3 arrays
        Where N is number of nodes in the array, the collection of
        streamlines to filter down to.
    target: Nifti1Image
        Nifti1Image containing a boolean representation of the ROI.
    target_idx: int.
        Index within each streamline to check if within the target region.
        Typically 0 for startpoint ROIs or -1 for endpoint ROIs.
        If using flip_sls, this becomes (len(sl) - this_idx - 1) % len(sl)
    tol : int, optional
        A distance tolerance (in units that the coordinates
        of the streamlines are represented in). Default: 0, which means that
        the endpoint is exactly in the coordinate of the target ROI.
    flip_sls : 1d array, optional
        Length is len(streamlines), whether to flip the streamline.
    accepted_idxs : 1d array, optional
        Boolean array, where entries correspond to eachs streamline,
        and streamlines that pass cleaning will be set to 1.
    Yields
    -------
    boolean array of streamlines that survive cleaning.
    """
    if accepted_idxs is None:
        accepted_idxs = np.zeros(len(streamlines), dtype=np.bool8)

    if flip_sls is None:
        flip_sls = np.zeros(len(streamlines))
    flip_sls = flip_sls.astype(int)

    roi = target.get_fdata()
    if tol > 0:
        roi = binary_dilation(
            roi,
            iterations=tol)

    for ii, sl in enumerate(streamlines):
        this_idx = target_idx
        if flip_sls[ii]:
            this_idx = (len(sl) - this_idx - 1) % len(sl)
        xx, yy, zz = sl[this_idx].astype(int)
        accepted_idxs[ii] = roi[xx, yy, zz]

    return accepted_idxs
