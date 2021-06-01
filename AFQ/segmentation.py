import os.path as op
import os
import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import zscore

import nibabel as nib
from tqdm.auto import tqdm

import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from dipy.stats.analysis import gaussian_weights
import dipy.core.gradients as dpg
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
from dipy.align import resample

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.utils.volume as auv
import AFQ.data as afd
from AFQ.utils.parallel import parfor

__all__ = ["Segmentation", "clean_bundle", "clean_by_endpoints"]


def _resample_tg(tg, n_points):
    # reformat for dipy's set_number_of_points
    if isinstance(tg, np.ndarray):
        if len(tg.shape) > 2:
            streamlines = tg.tolist()
            streamlines = [np.asarray(item) for item in streamlines]
    else:
        streamlines = tg.streamlines

    return dps.set_number_of_points(streamlines, n_points)


class Segmentation:
    def __init__(self,
                 nb_points=False,
                 nb_streamlines=False,
                 seg_algo='AFQ',
                 reg_algo=None,
                 clip_edges=False,
                 parallel_segmentation={
                     "n_jobs": -1, "engine": "joblib",
                     "backend": "loky"},
                 progressive=True,
                 greater_than=50,
                 rm_small_clusters=50,
                 model_clust_thr=1.25,
                 reduction_thr=25,
                 refine=False,
                 pruning_thr=12,
                 b0_threshold=50,
                 prob_threshold=0,
                 dist_to_waypoint=None,
                 rng=None,
                 return_idx=False,
                 presegment_bundle_dict=None,
                 presegment_kawrgs={},
                 filter_by_endpoints=True,
                 endpoint_info=None,
                 dist_to_atlas=4,
                 save_intermediates=None):
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
        reg_algo : string or None, optional
            Algorithm for streamline registration (case-insensitive):
            'slr' : Use Streamlinear Registration [Garyfallidis2015]_
            'syn' : Use image-based nonlinear registration
            If None, will use SyN if a mapping is provided, slr otherwise.
            If  seg_algo="AFQ", SyN is always used.
            Default: None
        clip_edges : bool
            Whether to clip the streamlines to be only in between the ROIs.
            Default: False
        parallel_segmentation : dict or AFQ.api.BundleDict
            How to parallelize segmentation across processes when performing
            waypoint ROI segmentation. Set to {"engine": "serial"} to not
            perform parallelization. See ``AFQ.utils.parallel.pafor`` for
            details.
            Default: {"n_jobs": -1, "engine": "joblib",
                      "backend": "loky"}
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
        dist_to_waypoint : float.
            The distance that a streamline node has to be from the waypoint
            ROI in order to be included or excluded.
            If set to None (default), will be calculated as the
            center-to-corner distance of the voxel in the diffusion data.
            If a bundle has additional_tolerance in its bundle_dict, that
            tolerance will be added to this distance.
            For example, if you wanted to increase tolerance for the right
            arcuate waypoint ROIs by 3 each, you could make the following
            modification to your bundle_dict:
            bundle_dict["ARC_R"]["additional_tolerances"] = [3, 3]
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
                {'name': {'ROIs':[img1, img2],
                'rules':[True, True]},
                'prob_map': img3,
                'cross_midline': False}
            Default: None
        presegment_kawrgs : dict
            Optional arguments for initializing the segmentation for the
            presegmentation. Only used if presegment_bundle_dict is not None.
            Default: {}
        filter_by_endpoints: bool
            Whether to filter the bundles based on their endpoints relative
            to regions defined in the AAL atlas. Applies only to the waypoint
            approach (XXX for now). Default: True.
        endpoint_info : dict, optional. This overrides use of the
            AAL atlas, which is the default behavior.
            The format for this should be:
            {"bundle1": {"startpoint":img1_1,
                         "endpoint":img1_2},
             "bundle2": {"startpoint":img2_1,
                          "endpoint":img2_2}}
            where the images used are binary masks of the desired
            endpoints.
        dist_to_atlas : float
            If filter_by_endpoints is True, this is the required distance
            from the endpoints to the atlas ROIs.
        save_intermediates : str, optional
            The full path to a folder into which intermediate products
            are saved. Default: None, means no saving of intermediates.

        References
        ----------
        .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
        Tract probability maps in stereotaxic spaces: analyses of white
        matter anatomy and tract-specific quantification. Neuroimage 39:
        336-347
        """
        self.logger = logging.getLogger('AFQ.Segmentation')
        self.nb_points = nb_points
        self.nb_streamlines = nb_streamlines

        if rng is None:
            self.rng = np.random.RandomState()
        elif isinstance(rng, int):
            self.rng = np.random.RandomState(rng)
        else:
            self.rng = rng

        self.seg_algo = seg_algo.lower()
        if reg_algo is not None:
            reg_algo = reg_algo.lower()
        self.reg_algo = reg_algo
        self.prob_threshold = prob_threshold
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
        self.presegment_kawrgs = presegment_kawrgs
        self.filter_by_endpoints = filter_by_endpoints
        self.endpoint_info = endpoint_info
        self.dist_to_atlas = dist_to_atlas
        self.parallel_segmentation = parallel_segmentation

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

    def segment(self, bundle_dict, tg, fdata=None, fbval=None,
                fbvec=None, mapping=None, reg_prealign=None,
                reg_template=None, b0_threshold=50, img_affine=None,
                reset_tg_space=False):
        """
        Segment streamlines into bundles based on either waypoint ROIs
        [Yeatman2012]_ or RecoBundles [Garyfallidis2017]_.
        Parameters
        ----------
        bundle_dict: dict or AFQ.api.BundleDict
            Meta-data for the segmentation. The format is something like::
                {'name': {'ROIs':[img1, img2],
                'rules':[True, True]},
                'prob_map': img3,
                'cross_midline': False}
        tg : StatefulTractogram
            Bundles to segment
        fdata, fbval, fbvec : str
            Full path to data, bvals, bvecs
        mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional.
            A mapping between DWI space and a template. If None, mapping will
            be registered from data used in prepare_img. Default: None.
        reg_prealign : array, optional.
            The linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.
            Default: None.
        reg_template : str or nib.Nifti1Image, optional.
            Template to use for registration. Default: MNI T2.
        img_affine : array, optional.
            The spatial transformation from the measurement to the scanner
            space.
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
        if img_affine is not None:
            if (mapping is None
                or fdata is not None
                or fbval is not None
                    or fbvec is not None):

                self.logger.error(
                    "Provide either the full path to data, bvals, bvecs,"
                    + "or provide the affine of the image and the mapping")

        self.logger.info("Preparing Segmentation Parameters")
        self.img_affine = img_affine
        self.prepare_img(fdata, fbval, fbvec)
        self.logger.info("Preprocessing Streamlines")
        tg = self._read_tg(tg)

        # If resampling over-write the sft:
        if self.nb_points:
            self.tg = StatefulTractogram(
                dps.set_number_of_points(self.tg.streamlines, self.nb_points),
                self.tg, self.tg.space)

        self.prepare_map(mapping, reg_prealign, reg_template)
        self.bundle_dict = bundle_dict

        if self.seg_algo == "afq":
            # We only care about midline crossing if we use AFQ:
            self.cross_streamlines()
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

    def prepare_img(self, fdata, fbval, fbvec):
        """
        Prepare image data from DWI data.
        Parameters
        ----------
        fdata, fbval, fbvec : str
            Full path to data, bvals, bvecs
        """
        if self.img_affine is None:
            self.img, _, _, _ = \
                ut.prepare_data(fdata, fbval, fbvec,
                                b0_threshold=self.b0_threshold)
            self.img_affine = self.img.affine

        self.fdata = fdata
        self.fbval = fbval
        self.fbvec = fbvec

    def prepare_map(self, mapping=None, reg_prealign=None, reg_template=None):
        """
        Set mapping between DWI space and a template.
        Parameters
        ----------
        mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional.
            A mapping between DWI space and a template.
            If None, mapping will be registered from data used in prepare_img.
            Default: None.
        reg_template : str or nib.Nifti1Image, optional.
            Template to use for registration (defaults to the MNI T2)
            Default: None.
        reg_prealign : array, optional.
            The linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.
            Default: None.
        """
        if reg_template is None:
            reg_template = afd.read_mni_template()

        self.reg_prealign = reg_prealign
        self.reg_template = reg_template

        if mapping is None:
            if self.seg_algo == "afq" or self.reg_algo == "syn":
                gtab = dpg.gradient_table(self.fbval, self.fbvec)
                self.mapping = reg.syn_register_dwi(self.fdata, gtab,
                                                    template=reg_template)[1]
            else:
                self.mapping = None
        elif isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
            if reg_prealign is None:
                reg_prealign = np.eye(4)
            if self.img is None:
                self.img, _, _, _ = \
                    ut.prepare_data(self.fdata,
                                    self.fbval,
                                    self.fbvec,
                                    b0_threshold=self.b0_threshold)
            self.mapping = reg.read_mapping(
                mapping,
                self.img,
                reg_template,
                prealign=np.linalg.inv(reg_prealign))
        else:
            self.mapping = mapping

    def cross_streamlines(self, tg=None, template=None, low_coord=10):
        """
        Classify the streamlines by whether they cross the midline.
        Creates a crosses attribute which is an array of booleans. Each boolean
        corresponds to a streamline, and is whether or not that streamline
        crosses the midline.
        Parameters
        ----------
        tg : StatefulTractogram class instance.
        template : nibabel.Nifti1Image class instance
            An affine transformation into a template space.
        """
        if tg is None:
            tg = self.tg
        if template is None:
            template_affine = self.img_affine
        else:
            template_affine = template.affine

        # What is the x,y,z coordinate of 0,0,0 in the template space?
        zero_coord = np.dot(np.linalg.inv(template_affine),
                            np.array([0, 0, 0, 1]))

        self.crosses = np.zeros(len(tg), dtype=bool)
        # already_split = 0
        for sl_idx, sl in enumerate(tg.streamlines):
            if np.any(sl[:, 0] > zero_coord[0]) and \
                    np.any(sl[:, 0] < zero_coord[0]):
                self.crosses[sl_idx] = True
            else:
                self.crosses[sl_idx] = False

    def _get_bundle_info(self, bundle_idx, bundle, vox_dim, tol):
        """
        Get fiber probabilites and ROIs for a given bundle.
        """
        bundle_entry = self.bundle_dict[bundle]
        rules = bundle_entry['rules']
        include_rois = []
        include_roi_tols = []
        exclude_rois = []
        exclude_roi_tols = []
        for rule_idx, rule in enumerate(rules):
            roi = bundle_entry['ROIs'][rule_idx]
            if 'additional_tolerance' in bundle_entry:
                this_tol = (bundle_entry['additional_tolerance'][rule_idx]
                            / vox_dim + tol)**2
            else:
                this_tol = tol**2

            warped_roi = auv.transform_inverse_roi(
                roi,
                self.mapping,
                bundle_name=bundle)

            if rule:
                # include ROI:
                include_roi_tols.append(this_tol)
                include_rois.append(np.array(np.where(warped_roi)).T)
            else:
                # Exclude ROI:
                exclude_roi_tols.append(this_tol)
                exclude_rois.append(np.array(np.where(warped_roi)).T)

            # For debugging purposes, we can save the variable as it is:
            if self.save_intermediates is not None:
                os.makedirs(
                    op.join(self.save_intermediates,
                            'warpedROI_',
                            bundle),
                    exist_ok=True)
                nib.save(
                    nib.Nifti1Image(warped_roi.astype(np.float32),
                                    self.img_affine),
                    op.join(self.save_intermediates,
                            'warpedROI_',
                            bundle,
                            'as_used.nii.gz'))

        # The probability map if doesn't exist is all ones with the same
        # shape as the ROIs:
        if isinstance(roi, str):
            roi = nib.load(roi)
        if isinstance(roi, nib.Nifti1Image):
            roi = roi.get_fdata()
        prob_map = bundle_entry.get(
            'prob_map', np.ones(roi.shape))

        if not isinstance(prob_map, np.ndarray):
            prob_map = prob_map.get_fdata()
        warped_prob_map = \
            self.mapping.transform_inverse(prob_map.copy(),
                                           interpolation='nearest')
        return warped_prob_map, include_rois, exclude_rois,\
            include_roi_tols, exclude_roi_tols

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

    def segment_afq(self, tg=None):
        """
        Assign streamlines to bundles using the waypoint ROI approach
        Parameters
        ----------
        tg : StatefulTractogram class instance
        """
        tg = self._read_tg(tg=tg)
        self.tg.to_vox()

        # For expedience, we approximate each streamline as a 100 point curve.
        # This is only used in extracting the values from the probability map,
        # so will not affect measurement of distance from the waypoint ROIs
        fgarray = np.array(_resample_tg(tg, 100))
        n_streamlines = fgarray.shape[0]

        streamlines_in_bundles = np.zeros(
            (n_streamlines, len(self.bundle_dict)))
        min_dist_coords = np.zeros(
            (n_streamlines, len(self.bundle_dict), 2), dtype=int)
        self.fiber_groups = {}

        if self.return_idx:
            out_idx = np.arange(n_streamlines, dtype=int)

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

        if self.filter_by_endpoints:
            if self.endpoint_info is None:
                aal_atlas = afd.read_aal_atlas(self.reg_template)
                atlas = aal_atlas['atlas']
                if self.save_intermediates is not None:
                    nib.save(
                        atlas,
                        op.join(self.save_intermediates,
                                'atlas_registered_to_template.nii.gz'))

                atlas = atlas.get_fdata()

            dist_to_atlas = self.dist_to_atlas / vox_dim

        self.logger.info("Assigning Streamlines to Bundles")
        for bundle_idx, bundle in enumerate(self.bundle_dict):
            self.logger.info(f"Finding Streamlines for {bundle}")
            warped_prob_map, include_roi, exclude_roi,\
                include_roi_tols, exclude_roi_tols =\
                self._get_bundle_info(bundle_idx, bundle, vox_dim, tol)
            if self.save_intermediates is not None:
                os.makedirs(
                    op.join(self.save_intermediates,
                            'warpedprobmap',
                            bundle),
                    exist_ok=True)
                nib.save(
                    nib.Nifti1Image(warped_prob_map.astype(np.float32),
                                    self.img_affine),
                    op.join(self.save_intermediates,
                            'warpedprobmap',
                            bundle,
                            'as_used.nii.gz'))

            fiber_probabilities = dts.values_from_volume(
                warped_prob_map,
                fgarray, np.eye(4))
            fiber_probabilities = np.mean(fiber_probabilities, -1)
            idx_above_prob = np.where(
                fiber_probabilities > self.prob_threshold)
            self.logger.info((f"{len(idx_above_prob[0])} streamlines exceed"
                              " the probability threshold."))
            crosses_midline = self.bundle_dict[bundle]['cross_midline']

            # with parallel segmentation, the first for loop will
            # only collect streamlines and does not need tqdm
            if self.parallel_segmentation["engine"] != "serial":
                in_list = []
                sl_idxs = idx_above_prob[0]
                parallelizing = True
            else:
                sl_idxs = tqdm(idx_above_prob[0])
                parallelizing = False

            for sl_idx in sl_idxs:
                if fiber_probabilities[sl_idx] > self.prob_threshold:
                    if crosses_midline is not None:
                        if self.crosses[sl_idx]:
                            # This means that the streamline does
                            # cross the midline:
                            if crosses_midline:
                                # This is what we want, keep going
                                pass
                            else:
                                # This is not what we want,
                                # skip to next streamline
                                continue
                sl = tg.streamlines[sl_idx]
                fiber_prob = fiber_probabilities[sl_idx]

                # if parallel, collect the streamlines now
                if parallelizing:
                    in_list.append((sl, fiber_prob, sl_idx))
                else:
                    min_dist_coords[sl_idx, bundle_idx, 0],\
                        min_dist_coords[sl_idx, bundle_idx, 1],\
                        streamlines_in_bundles[sl_idx, bundle_idx] =\
                        _is_streamline_in_ROIs(
                            sl, tol, include_roi,
                            include_roi_tols, exclude_roi,
                            exclude_roi_tols, fiber_prob)

            # collects results from the submitted streamlines
            if parallelizing:
                results = parfor(
                    _is_streamline_in_ROIs_parallel, in_list,
                    func_args=[
                        tol, include_roi, include_roi_tols, exclude_roi,
                        exclude_roi_tols, bundle_idx],
                    **self.parallel_segmentation)
                for result in results:
                    sl_idx, bundle_idx, min_dist_coords_0,\
                        min_dist_coords_1, sl_in_bundles =\
                        result
                    min_dist_coords[sl_idx, bundle_idx, 0] = min_dist_coords_0
                    min_dist_coords[sl_idx, bundle_idx, 1] = min_dist_coords_1
                    streamlines_in_bundles[sl_idx, bundle_idx] = sl_in_bundles

            self.logger.info(
                (f"{np.sum(streamlines_in_bundles[:, bundle_idx] > 0)} "
                 "streamlines selected with waypoint ROIs"))

        # Eliminate any fibers not selected using the waypoint ROIs:
        possible_fibers = np.sum(streamlines_in_bundles, -1) > 0
        tg = StatefulTractogram(tg.streamlines[possible_fibers],
                                self.img,
                                Space.VOX)
        if self.return_idx:
            out_idx = out_idx[possible_fibers]

        streamlines_in_bundles = streamlines_in_bundles[possible_fibers]
        min_dist_coords = min_dist_coords[possible_fibers]
        bundle_choice = np.argmax(streamlines_in_bundles, -1)

        # We do another round through, so that we can orient all the
        # streamlines within a bundle in the same orientation with respect to
        # the ROIs. This order is ARBITRARY but CONSISTENT (going from ROI0
        # to ROI1).
        self.logger.info("Re-orienting streamlines to consistent directions")
        for bundle_idx, bundle in enumerate(self.bundle_dict):
            self.logger.info(f"Processing {bundle}")

            select_idx = np.where(bundle_choice == bundle_idx)

            if len(select_idx[0]) == 0:
                # There's nothing here, set and move to the next bundle:
                self._return_empty(bundle)
                continue

            # Use a list here, because ArraySequence doesn't support item
            # assignment:
            select_sl = list(tg.streamlines[select_idx])
            # Sub-sample min_dist_coords:
            min_dist_coords_bundle = min_dist_coords[select_idx]
            for idx in range(len(select_sl)):
                min0 = min_dist_coords_bundle[idx, bundle_idx, 0]
                min1 = min_dist_coords_bundle[idx, bundle_idx, 1]
                if min0 > min1:
                    select_sl[idx] = select_sl[idx][::-1]

            if self.filter_by_endpoints:
                self.logger.info("Filtering by endpoints")
                self.logger.info("Before filtering "
                                 f"{len(select_sl)} streamlines")
                if self.endpoint_info is not None:
                    # We use definitions of endpoints provided
                    # through this dict:
                    start_p = self.endpoint_info[bundle]['startpoint']
                    end_p = self.endpoint_info[bundle]['endpoint']

                    atlas_idx = []
                    for ii, pp in enumerate([start_p, end_p]):
                        pp = resample(
                            pp.get_fdata(),
                            self.reg_template,
                            pp.affine,
                            self.reg_template.affine).get_fdata()

                        atlas_roi = np.zeros(pp.shape)
                        atlas_roi[np.where(pp > 0)] = 1
                        # Create binary masks and warp these into subject's
                        # DWI space:
                        warped_roi = self.mapping.transform_inverse(
                            atlas_roi,
                            interpolation='nearest')

                        if self.save_intermediates is not None:
                            if ii == 0:
                                point_name = "startpoint"
                            else:
                                point_name = "endpoint"
                            os.makedirs(op.join(
                                self.save_intermediates,
                                'endpoint_ROI',
                                bundle), exist_ok=True)

                            nib.save(
                                nib.Nifti1Image(
                                    warped_roi,
                                    self.img_affine),
                                op.join(self.save_intermediates,
                                        'endpoint_ROI',
                                        bundle,
                                        f'{point_name}_as_used.nii.gz'))

                        atlas_idx.append(
                            np.array(np.where(warped_roi > 0)).T)
                else:
                    # We automatically fallback on AAL, which as its own
                    # set of rules.
                    aal_targets = afd.bundles_to_aal(
                        [bundle], atlas=atlas)[0]
                    atlas_idx = []
                    for targ in aal_targets:
                        if targ is not None:
                            aal_roi = np.zeros(atlas.shape[:3])
                            aal_roi[targ[:, 0],
                                    targ[:, 1],
                                    targ[:, 2]] = 1
                            warped_roi = self.mapping.transform_inverse(
                                aal_roi,
                                interpolation='nearest')
                            atlas_idx.append(
                                np.array(np.where(warped_roi > 0)).T)
                        else:
                            atlas_idx.append(None)

                new_select_sl = clean_by_endpoints(
                    select_sl,
                    atlas_idx[0],
                    atlas_idx[1],
                    tol=dist_to_atlas,
                    return_idx=self.return_idx)

                # Generate immediately:
                new_select_sl = list(new_select_sl)

                # We need to check this again:
                if len(new_select_sl) == 0:
                    self.logger.info("After filtering "
                                     f"{len(new_select_sl)} streamlines")
                    # There's nothing here, set and move to the next bundle:
                    self._return_empty(bundle)
                    continue

                if self.return_idx:
                    temp_select_sl = []
                    temp_select_idx = np.empty(len(new_select_sl), int)
                    for ii, ss in enumerate(new_select_sl):
                        temp_select_sl.append(ss[0])
                        temp_select_idx[ii] = ss[1]
                    select_idx = select_idx[0][temp_select_idx]
                    new_select_sl = temp_select_sl

                select_sl = new_select_sl
                self.logger.info("After filtering "
                                 f"{len(select_sl)} streamlines")

            if self.clip_edges:
                self.logger.info("Clipping Streamlines by ROI")
                for idx in range(len(select_sl)):
                    min0 = min_dist_coords_bundle[idx, bundle_idx, 0]
                    min1 = min_dist_coords_bundle[idx, bundle_idx, 1]

                    # If the point that is closest to the first ROI
                    # is the same as the point closest to the second ROI,
                    # include the surrounding points to make a streamline.
                    if min0 == min1:
                        min1 = min1 + 1
                        min0 = min0 - 1

                    select_sl[idx] = select_sl[idx][min0:min1]

            select_sl = StatefulTractogram(select_sl,
                                           self.img,
                                           Space.RASMM)

            if self.return_idx:
                self.fiber_groups[bundle] = {}
                self.fiber_groups[bundle]['sl'] = select_sl
                self.fiber_groups[bundle]['idx'] = out_idx[select_idx]
            else:
                self.fiber_groups[bundle] = select_sl
        return self.fiber_groups

    def move_streamlines(self, tg, reg_algo='slr'):
        """Streamline-based registration of a whole-brain tractogram to
        the MNI whole-brain atlas.

        registration_algo : str
            "slr" or "syn"
        """
        if reg_algo is None:
            if self.mapping is None:
                reg_algo = 'slr'
            else:
                reg_algo = 'syn'

        if reg_algo == "slr":
            self.logger.info("Registering tractogram with SLR")
            atlas = self.bundle_dict['whole_brain']
            self.moved_sl, _, _, _ = whole_brain_slr(
                atlas, tg.streamlines, x0='affine', verbose=False,
                progressive=self.progressive,
                greater_than=self.greater_than,
                rm_small_clusters=self.rm_small_clusters,
                rng=self.rng)
        elif reg_algo == "syn":
            self.logger.info("Registering tractogram based on syn")
            tg.to_rasmm()
            delta = dts.values_from_volume(
                self.mapping.forward,
                tg.streamlines, np.eye(4))
            self.moved_sl = dts.Streamlines(
                [d + s for d, s in zip(delta, tg.streamlines)])
            tg.to_vox()

        if self.save_intermediates is not None:
            moved_sft = StatefulTractogram(
                self.moved_sl,
                self.reg_template,
                Space.RASMM)
            save_tractogram(
                moved_sft,
                op.join(self.save_intermediates,
                        'sls_in_mni.trk'),
                bbox_valid_check=False)

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
            roiseg = Segmentation(**self.presegment_kawrgs)
            roiseg.segment(
                self.presegment_bundle_dict,
                self.tg,
                self.fdata,
                self.fbval,
                self.fbvec,
                reg_template=self.reg_template,
                mapping=self.mapping,
                reg_prealign=self.reg_prealign)
            roiseg_fg = roiseg.fiber_groups
        else:
            self.move_streamlines(tg, self.reg_algo)
            rb = RecoBundles(self.moved_sl, verbose=False, rng=self.rng)
        # Next we'll iterate over bundles, registering each one:
        bundle_list = list(self.bundle_dict.keys())
        if 'whole_brain' in bundle_list:
            bundle_list.remove('whole_brain')

        self.logger.info("Assigning Streamlines to Bundles")
        for bundle in bundle_list:
            self.logger.info(f"Finding streamlines for {bundle}")
            model_sl = self.bundle_dict[bundle]['sl']

            # If doing a presegmentation based on ROIs then initialize rb after
            # Filtering the whole brain tractogram to pass through ROIs
            if self.presegment_bundle_dict is not None:
                afq_bundle_name = afd.BUNDLE_RECO_2_AFQ.get(bundle, bundle)
                if "return_idx" in self.presegment_kawrgs\
                        and self.presegment_kawrgs["return_idx"]:
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
                self.move_streamlines(indiv_tg, self.reg_algo)
                rb = RecoBundles(
                    self.moved_sl,
                    verbose=False,
                    rng=self.rng)
            if self.save_intermediates is not None:
                if self.presegment_bundle_dict is not None:
                    moved_fname = f"{bundle}_presegmentation.trk"
                else:
                    moved_fname = "whole_brain.trk"
                moved_sft = StatefulTractogram(
                    self.moved_sl,
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
            standard_sl = self.bundle_dict[bundle]['centroid']
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
        return fiber_groups


def clean_bundle(tg, n_points=100, clean_rounds=5, distance_threshold=5,
                 length_threshold=4, min_sl=20, stat='mean',
                 return_idx=False):
    """
    Clean a segmented fiber group based on the Mahalnobis distance of
    each streamline

    Parameters
    ----------
    tg : StatefulTractogram class instance
        A whole-brain tractogram to be segmented.
    n_points : int, optional
        Number of points to resample streamlines to.
        Default: 100
    clean_rounds : int, optional.
        Number of rounds of cleaning based on the Mahalanobis distance from
        the mean of extracted bundles. Default: 5
    distance_threshold : float, optional.
        Threshold of cleaning based on the Mahalanobis distance (the units are
        standard deviations). Default: 5.
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
    # We don't even bother if there aren't enough streamlines:
    if len(tg.streamlines) < min_sl:
        if return_idx:
            return tg, np.arange(len(tg.streamlines))
        else:
            return tg

    # Resample once up-front:
    fgarray = _resample_tg(tg, n_points)

    # Keep this around, so you can use it for indexing at the very end:
    idx = np.arange(len(fgarray))
    # This calculates the Mahalanobis for each streamline/node:
    w = gaussian_weights(fgarray, return_mahalnobis=True, stat=stat)
    lengths = np.array([sl.shape[0] for sl in tg.streamlines])
    # We'll only do this for clean_rounds
    rounds_elapsed = 0
    while ((np.any(w > distance_threshold)
            or np.any(zscore(lengths) > length_threshold))
           and rounds_elapsed < clean_rounds
           and len(tg.streamlines) > min_sl):
        # Select the fibers that have Mahalanobis smaller than the
        # threshold for all their nodes:
        idx_dist = np.where(np.all(w < distance_threshold, axis=-1))[0]
        idx_len = np.where(zscore(lengths) < length_threshold)[0]
        idx_belong = np.intersect1d(idx_dist, idx_len)

        if len(idx_belong) < min_sl:
            # need to sort and return exactly min_sl:
            idx_belong = np.argsort(np.sum(w, axis=-1))[:min_sl]

        idx = idx[idx_belong.astype(int)]
        # Update by selection:
        fgarray = fgarray[idx_belong.astype(int)]
        lengths = lengths[idx_belong.astype(int)]
        # Repeat:
        w = gaussian_weights(fgarray, return_mahalnobis=True)
        rounds_elapsed += 1

    # Select based on the variable that was keeping track of things for us:
    out = StatefulTractogram(tg.streamlines[idx], tg, Space.VOX)
    if return_idx:
        return out, idx
    else:
        return out

# Helper functions for segmenting using waypoint ROIs
# they are not a part of the class because we do not want
# copies of the class to be parallelized


def _check_sl_with_inclusion(sl, include_rois, tol,
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


def _check_sl_with_exclusion(sl, exclude_rois, tol,
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


def _is_streamline_in_ROIs(sl, tol, include_roi,
                           include_roi_tols, exclude_roi,
                           exclude_roi_tols, fiber_prob):
    is_close, dist = \
        _check_sl_with_inclusion(
            sl,
            include_roi,
            tol,
            include_roi_tols)
    if is_close:
        is_far = \
            _check_sl_with_exclusion(
                sl,
                exclude_roi,
                tol,
                exclude_roi_tols)
        if is_far:
            return np.argmin(dist[0], 0)[0],\
                np.argmin(dist[1], 0)[0], fiber_prob
    return 0, 0, 0


def _is_streamline_in_ROIs_parallel(indiv_args, tol, include_roi,
                                    include_roi_tols, exclude_roi,
                                    exclude_roi_tols, bundle_idx):
    sl, fiber_prob, sl_idx = indiv_args
    min_dist_coords_0,\
        min_dist_coords_1,\
        streamlines_in_bundles = \
        _is_streamline_in_ROIs(
            sl, tol, include_roi,
            include_roi_tols, exclude_roi,
            exclude_roi_tols, fiber_prob)
    return (sl_idx, bundle_idx, min_dist_coords_0, min_dist_coords_1,
            streamlines_in_bundles)


def clean_by_endpoints(streamlines, targets0, targets1, tol=None, atlas=None,
                       return_idx=False):
    """
    Clean a collection of streamlines based on their two endpoints
    Filters down to only include items that have their starting points close to
    the targets0 and ending points close to targets1
    Parameters
    ----------
    streamlines : sequence of 3XN_i arrays The collection of streamlines to
        filter down to.
    targets0, target1: sequences or Nx3 arrays or None.
        The targets. Numerical values in the atlas array for targets for the
        first and last node in each streamline respectively, or NX3 arrays with
        each row containing the indices for these locations in the atlas.
        If provided a None, this means no restriction on that end.
    tol : float, optional A distance tolerance (in units that the coordinates
        of the streamlines are represented in). Default: 0, which means that
        the endpoint is exactly in the coordinate of the target ROI.
    atlas : 3D array or Nifti1Image class instance with a 3D array, optional.
        Contains numerical values for ROIs. Default: if not provided, assume
        that targets0 and targets1 are both arrays of indices, and this
        information is not needed.
    Yields
    -------
    Generator of the filtered collection
    """
    if tol is None:
        tol = 0

    # We square the tolerance, because below we are using the squared Euclidean
    # distance which is slightly faster:
    tol = tol ** 2

    # Check whether it's already in the right format:
    sp_is_idx = (targets0 is None
                 or (isinstance(targets0, np.ndarray)
                     and targets0.shape[1] == 3))

    ep_is_idx = (targets1 is None
                 or (isinstance(targets1, np.ndarray)
                     and targets1.shape[1] == 3))

    if atlas is None and not (ep_is_idx and sp_is_idx):
        e_s = "Need to provide endpoint and startpoint as "
        e_s += "indices, or provide an atlas"
        raise ValueError(e_s)

    if sp_is_idx:
        idxes0 = targets0
    else:
        # Otherwise, we'll need to derive it:
        startpoint_roi = np.zeros(atlas.shape, dtype=bool)
        for targ in targets0:
            startpoint_roi[atlas == targ] = 1
        idxes0 = np.array(np.where(startpoint_roi)).T

    if ep_is_idx:
        idxes1 = targets1
    else:
        endpoint_roi = np.zeros(atlas.shape, dtype=bool)
        for targ in targets1:
            endpoint_roi[atlas == targ] = 1
        idxes1 = np.array(np.where(endpoint_roi)).T

    for ii, sl in enumerate(streamlines):
        if targets0 is None:
            # Nothing to check
            dist0ok = True
        else:
            dist0ok = False
            dist0 = np.min(cdist(np.array([sl[0]]), idxes0, 'sqeuclidean'))
            if dist0 <= tol:
                dist0ok = True
        # Only proceed if conditions for one side are fulfilled:
        if dist0ok:
            if targets1 is None:
                # Nothing to check on this end:
                if return_idx:
                    yield sl, ii
                else:
                    yield sl
            else:
                dist2 = np.min(cdist(np.array([sl[-1]]), idxes1,
                                     'sqeuclidean'))
                if dist2 <= tol:
                    if return_idx:
                        yield sl, ii
                    else:
                        yield sl
