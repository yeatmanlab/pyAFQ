import numpy as np
import logging
from scipy.spatial.distance import mahalanobis, cdist

import nibabel as nib
from tqdm.auto import tqdm

import dipy.data as dpd
import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps
from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from dipy.stats.analysis import gaussian_weights
import dipy.core.gradients as dpg

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.utils.volume as auv

__all__ = ["Segmentation"]


def _resample_bundle(streamlines, n_points):
    # reformat for dipy's set_number_of_points
    if isinstance(streamlines, np.ndarray):
        if len(streamlines.shape) > 2:
            streamlines = streamlines.tolist()
            streamlines = [np.asarray(item) for item in streamlines]

    return dps.set_number_of_points(streamlines, n_points)


class Segmentation:
    def __init__(self,
                 nb_points=False,
                 algo='AFQ',
                 progressive=True,
                 greater_than=50,
                 rm_small_clusters=50,
                 model_clust_thr=40.,
                 reduction_thr=40,
                 b0_threshold=0,
                 prob_threshold=0,
                 rng=None,
                 return_idx=False):
        """
        Segment streamlines into bundles.

        Parameters
        ----------
        nb_points : int, boolean
            Resample streamlines to nb_points number of points.
            If False, no resampling is done. Default: False
        algo : string
            Algorithm for segmentation (case-insensitive):
            'AFQ': Segment streamlines into bundles,
                based on inclusion/exclusion ROIs.
            'Reco': Segment streamlines using the RecoBundles algorithm
            [Garyfallidis2017].
            Default: 'AFQ'

        rm_small_clusters : int
            Using RecoBundles Algorithm.
            Remove clusters that have less than this value
                during whole brain SLR.
            Default: 50

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

        b0_theshold : float.
            Using AFQ Algorithm.
            All b-values with values less than or equal to `bo_threshold` are
            considered as b0s i.e. without diffusion weighting.
            Default: 0.

        prob_threshold : float.
            Using AFQ Algorithm.
            Initial cleaning of fiber groups is done using probability maps
            from [Hua2008]_. Here, we choose an average probability that
            needs to be exceeded for an individual streamline to be retained.
            Default: 0.

        rng : RandomState
            If None, creates RandomState. Used in RecoBundles Algorithm.
            Default: None.

        return_idx : bool
            Whether to return the indices in the original streamlines as part
            of the output of segmentation.

        References
        ----------
        .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
        Tract probability maps in stereotaxic spaces: analyses of white
        matter anatomy and tract-specific quantification. Neuroimage 39:
        336-347
        """
        self.logger = logging.getLogger('AFQ.Segmentation')
        self.nb_points = nb_points

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        algo = algo.lower()
        if algo == 'reco':
            self.segment = self._seg_reco
        else:
            self.segment = self._seg_afq

        self.prob_threshold = prob_threshold
        self.b0_threshold = b0_threshold
        self.progressive = progressive
        self.greater_than = greater_than
        self.rm_small_clusters = rm_small_clusters
        self.model_clust_thr = model_clust_thr
        self.reduction_thr = reduction_thr
        self.return_idx = return_idx

    def _seg_reco(self, bundle_dict, streamlines, fdata=None, fbval=None,
                  fbvec=None, mapping=None, reg_prealign=None,
                  reg_template=None):
        """
        Segment streamlines using the RecoBundles algorithm [Garyfallidis2017]

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each array is a streamline, shape (3, N).
            If streamlines is None, will use previously given streamlines.
            Default: None.
        bundle_dict: dict
            The format is something like::

                {'name': {'ROIs':[img1, img2],
                'rules':[True, True]},
                'prob_map': img3,
                'cross_midline': False}

        fdata, fbval, fbvec : str
            Full path to data, bvals, bvecs
        mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional.
            A mapping between DWI space and a template.
            If None, mapping will be registered from data used in prepare_img.
            Default: None.
        reg_prealign : array, optional.
            The linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.
            Default: None.
        reg_template : str or nib.Nifti1Image, optional.
            Template to use for registration (defaults to the MNI T2)
            Default: None.

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
        bundles using local and global streamline-based registration and
        clustering, Neuroimage, 2017.
        """
        self.bundle_dict = bundle_dict
        self.streamlines = streamlines
        return self.segment_reco()

    def _seg_afq(self, bundle_dict, streamlines, fdata=None, fbval=None,
                 fbvec=None, mapping=None, reg_prealign=None,
                 reg_template=None, b0_threshold=0, img_affine=None):
        """
        Segment streamlines into bundles based on inclusion ROIs.

        Prepare image data from DWI data,
        Set mapping between DWI space and a template,
        Get fiber probabilites and ROIs for each bundle,
        And iterate over streamlines and bundles,
        assigning streamlines to fiber groups.

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each array is a streamline, shape (3, N).
            If streamlines is None, will use previously given streamlines.
            Default: None.
        bundle_dict: dict
            The format is something like::

                {'name': {'ROIs':[img1, img2],
                'rules':[True, True]},
                'prob_map': img3,
                'cross_midline': False}

        fdata, fbval, fbvec : str
            Full path to data, bvals, bvecs
        mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional.
            A mapping between DWI space and a template.
            If None, mapping will be registered from data used in prepare_img.
            Default: None.
        reg_prealign : array, optional.
            The linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.
            Default: None.
        reg_template : str or nib.Nifti1Image, optional.
            Template to use for registration (defaults to the MNI T2)
            Default: None.
        img_affine : array, optional.
            The spatial transformation from the measurement to the scanner
            space.

        References
        ----------
        .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty,
        Nathaniel J. Myall, Brian A. Wandell, and Heidi M. Feldman. 2012.
        "Tract Profiles of White Matter Properties: Automating Fiber-Tract
        Quantification" PloS One 7 (11): e49790.
        """
        if img_affine is not None:
            if (mapping is None
                or fdata is not None
                or fbval is not None
                    or fbvec is not None):

                self.logger.error(
                    "Provide either the full path to data, bvals, bvecs,"
                    + "or provide the affine of the image and the mapping")

        self.logger.info("Preparing Segmentation Parameters...")
        self.img_affine = img_affine
        self.prepare_img(fdata, fbval, fbvec)
        self.prepare_map(mapping, reg_prealign, reg_template)
        self.bundle_dict = bundle_dict

        self.logger.info("Preprocessing Streamlines...")
        self.streamlines = streamlines
        if self.nb_points:
            self.resample_streamlines(self.nb_points)
        self.cross_streamlines()

        return self.segment_afq()

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
            reg_template = dpd.read_mni_template()

        if mapping is None:
            gtab = dpg.gradient_table(self.fbval, self.fbvec)
            self.mapping = reg.syn_register_dwi(self.fdata, gtab)[1]
        elif isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
            if reg_prealign is None:
                reg_prealign = np.eye(4)
            if self.img is None:
                self.img, _, _, _ = \
                    ut.prepare_data(self.fdata,
                                    self.fbval,
                                    self.fbvec,
                                    b0_threshold=self.b0_threshold)
            self.mapping = reg.read_mapping(mapping, self.img, reg_template,
                                            prealign=reg_prealign)
        else:
            self.mapping = mapping

    def resample_streamlines(self, nb_points, streamlines=None):
        """
        Resample streamlines to nb_points number of points.

        Parameters
        ----------
        nb_points : int
            Integer representing number of points wanted along the curve.
            Streamlines will be resampled to this number of points.
        streamlines : list of 2D arrays
            Each array is a streamline, shape (3, N).
            If streamlines is None, will use previously given streamlines.
            Default: None.
        """
        if streamlines is None:
            streamlines = self.streamlines

        self.streamlines = np.array(
            dps.set_number_of_points(streamlines, nb_points))

    def cross_streamlines(self, streamlines=None,
                          template=None, low_coord=10):
        """
        Classify the streamlines by whether they cross the midline.
        Creates a crosses attribute which is an array of booleans.
        Each boolean corresponds to a streamline,
        and is whether or not that streamline crosses the midline.

        Parameters
        ----------
        streamlines : list or Streamlines class instance.
        template : nibabel.Nifti1Image class instance
            An affine transformation into a template space.
        low_coords: int, optional.
            How many coordinates below the 0,0,0 point should a streamline be
            to be split if it passes the midline.
            Default: 10
        """
        if streamlines is None:
            streamlines = self.streamlines
        if template is None:
            template = self.img

        # What is the x,y,z coordinate of 0,0,0 in the template space?
        zero_coord = np.dot(np.linalg.inv(template.affine),
                            np.array([0, 0, 0, 1]))

        # cross_below = zero_coord[2] - low_coord
        self.crosses = np.zeros(len(streamlines), dtype=bool)
        # already_split = 0
        for sl_idx, sl in enumerate(streamlines):
            if np.any(sl[:, 0] > zero_coord[0]) and \
                    np.any(sl[:, 0] < zero_coord[0]):
                # if np.any(sl[:, 2] < cross_below):
                #     # This is a streamline that needs to be split where it
                #     # self.crosses the midline:
                #     split_idx = np.argmin(np.abs(sl[:, 0] - zero_coord[0]))
                #     streamlines = aus.split_streamline(
                #         streamlines, sl_idx + already_split, split_idx)
                #     already_split = already_split + 1
                #     # Now that it's been split, neither cross the midline:
                #     self.crosses[sl_idx] = False
                #     self.crosses = np.concatenate([self.crosses[:sl_idx+1],
                #                               np.array([False]),
                #                               self.crosses[sl_idx+1:]])
                # else:
                self.crosses[sl_idx] = True
            else:
                self.crosses[sl_idx] = False

        # Move back to the original space:
        self.streamlines = streamlines

    def _get_bundle_info(self, bundle_idx, bundle):
        """
        Get fiber probabilites and ROIs for a given bundle.
        """
        rules = self.bundle_dict[bundle]['rules']
        include_rois = []
        exclude_rois = []
        for rule_idx, rule in enumerate(rules):
            roi = self.bundle_dict[bundle]['ROIs'][rule_idx]
            if not isinstance(roi, np.ndarray):
                roi = roi.get_fdata()
            warped_roi = auv.patch_up_roi(
                (self.mapping.transform_inverse(
                    roi.astype(np.float32),
                    interpolation='linear')) > 0)

            if rule:
                # include ROI:
                include_rois.append(np.array(np.where(warped_roi)).T)
            else:
                # Exclude ROI:
                exclude_rois.append(np.array(np.where(warped_roi)).T)

        # The probability map if doesn't exist is all ones with the same
        # shape as the ROIs:
        prob_map = self.bundle_dict[bundle].get(
            'prob_map', np.ones(roi.shape))

        if not isinstance(prob_map, np.ndarray):
            prob_map = prob_map.get_fdata()
        warped_prob_map = \
            self.mapping.transform_inverse(prob_map,
                                           interpolation='nearest')
        return warped_prob_map, include_rois, exclude_rois

    def _check_sl_with_inclusion(self, sl, include_rois, tol):
        """
        Helper function to check that a streamline is close to a list of
        inclusion ROIS.
        """
        dist = []
        for roi in include_rois:
            dist.append(cdist(sl, roi, 'sqeuclidean'))
            if np.min(dist[-1]) > tol:
                # Too far from one of them:
                return False, []
        # Apparently you checked all the ROIs and it was close to all of them
        return True, dist

    def _check_sl_with_exclusion(self, sl, exclude_rois, tol):
        """ Helper function to check that a streamline is not too close to a
        list of exclusion ROIs.
        """
        for roi in exclude_rois:
            if np.min(cdist(sl, roi, 'sqeuclidean')) < tol:
                return False
        # Either there are no exclusion ROIs, or you are not close to any:
        return True

    def segment_afq(self, streamlines=None):
        """
        Iterate over streamlines and bundles,
        assigning streamlines to fiber groups.

        Parameters
        ----------
        streamlines : list of 2D arrays
            Each array is a streamline, shape (3, N).
            If streamlines is None, will use previously given streamlines.
            Default: None.
        """
        if streamlines is None:
            streamlines = self.streamlines
        else:
            self.streamlines = streamlines

        # For expedience, we approximate each streamline as a 100 point curve:
        fgarray = _resample_bundle(streamlines, 100)

        n_streamlines = len(streamlines)

        streamlines_in_bundles = np.zeros(
            (n_streamlines, len(self.bundle_dict)))
        min_dist_coords = np.zeros(
            (n_streamlines, len(self.bundle_dict), 2), dtype=int)
        self.fiber_groups = {}

        if self.return_idx:
            out_idx = np.arange(n_streamlines, dtype=int)

        self.logger.info("Assigning Streamlines to Bundles...")
        tol = dts.dist_to_corner(self.img_affine)**2
        for bundle_idx, bundle in enumerate(self.bundle_dict):
            warped_prob_map, include_roi, exclude_roi = \
                self._get_bundle_info(bundle_idx, bundle)
            fiber_probabilities = dts.values_from_volume(
                warped_prob_map,
                fgarray, np.eye(4))
            fiber_probabilities = np.mean(fiber_probabilities, -1)
            crosses_midline = self.bundle_dict[bundle]['cross_midline']
            for sl_idx, sl in enumerate(tqdm(streamlines)):
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

                    is_close, dist = \
                        self._check_sl_with_inclusion(sl,
                                                      include_roi,
                                                      tol)
                    if is_close:
                        is_far = \
                            self._check_sl_with_exclusion(sl,
                                                          exclude_roi,
                                                          tol)
                        if is_far:
                            min_dist_coords[sl_idx, bundle_idx, 0] =\
                                np.argmin(dist[0], 0)[0]
                            min_dist_coords[sl_idx, bundle_idx, 1] =\
                                np.argmin(dist[1], 0)[0]
                            streamlines_in_bundles[sl_idx, bundle_idx] =\
                                fiber_probabilities[sl_idx]

        self.logger.info("Cleaning and Re-Orienting...")
        # Eliminate any fibers not selected using the plane ROIs:
        possible_fibers = np.sum(streamlines_in_bundles, -1) > 0
        streamlines = streamlines[possible_fibers]
        if self.return_idx:
            out_idx = out_idx[possible_fibers]

        streamlines_in_bundles = streamlines_in_bundles[possible_fibers]
        min_dist_coords = min_dist_coords[possible_fibers]
        bundle_choice = np.argmax(streamlines_in_bundles, -1)

        # We do another round through, so that we can orient all the
        # streamlines within a bundle in the same orientation with respect to
        # the ROIs. This order is ARBITRARY but CONSISTENT (going from ROI0
        # to ROI1).
        for bundle_idx, bundle in enumerate(self.bundle_dict):
            select_idx = np.where(bundle_choice == bundle_idx)
            # Use a list here, Streamlines don't support item assignment:
            select_sl = list(streamlines[select_idx])
            if len(select_sl) == 0:
                self.fiber_groups[bundle] = dts.Streamlines([])
                # There's nothing here, move to the next bundle:
                continue

            # Sub-sample min_dist_coords:
            min_dist_coords_bundle = min_dist_coords[select_idx]
            for idx in range(len(select_sl)):
                min0 = min_dist_coords_bundle[idx, bundle_idx, 0]
                min1 = min_dist_coords_bundle[idx, bundle_idx, 1]
                if min0 > min1:
                    select_sl[idx] = select_sl[idx][::-1]
            # Set this to nibabel.Streamlines object for output:
            select_sl = dts.Streamlines(select_sl)
            if self.return_idx:
                self.fiber_groups[bundle] = {}
                self.fiber_groups[bundle]['sl'] = select_sl
                self.fiber_groups[bundle]['idx'] = out_idx[select_idx]
            else:
                self.fiber_groups[bundle] = select_sl
        return self.fiber_groups

    def segment_reco(self, streamlines=None):
        """
        Segment streamlines using the RecoBundles algorithm [Garyfallidis2017]

        Parameters
        ----------
        streamlines : list or Streamlines object.
            A whole-brain tractogram to be segmented.

        Returns
        -------
        fiber_groups : dict
            Keys are names of the bundles, values are Streamline objects.
            The streamlines in each object have all been oriented to have the
            same orientation (using `dts.orient_by_streamline`).
        """
        if streamlines is None:
            streamlines = self.streamlines
        else:
            self.streamlines = streamlines

        fiber_groups = {}
        self.logger.info("Registering Whole-brain with SLR...")
        # We start with whole-brain SLR:
        atlas = self.bundle_dict['whole_brain']
        moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            atlas, streamlines, x0='affine', verbose=False,
            progressive=self.progressive,
            greater_than=self.greater_than,
            rm_small_clusters=self.rm_small_clusters,
            rng=self.rng)

        # We generate our instance of RB with the moved streamlines:
        self.logger.info("Extracting Bundles...")
        rb = RecoBundles(moved, verbose=False, rng=self.rng)

        # Next we'll iterate over bundles, registering each one:
        bundle_list = list(self.bundle_dict.keys())
        bundle_list.remove('whole_brain')

        self.logger.info("Assigning Streamlines to Bundles...")
        for bundle in bundle_list:
            model_sl = self.bundle_dict[bundle]['sl']
            _, rec_labels = rb.recognize(model_bundle=model_sl,
                                         model_clust_thr=self.model_clust_thr,
                                         reduction_thr=self.reduction_thr,
                                         reduction_distance='mdf',
                                         slr=True,
                                         slr_metric='asymmetric',
                                         pruning_distance='mdf')

            # Use the streamlines in the original space:
            recognized_sl = streamlines[rec_labels]
            standard_sl = self.bundle_dict[bundle]['centroid']
            oriented_sl = dts.orient_by_streamline(recognized_sl, standard_sl)
            if self.return_idx:
                fiber_groups[bundle] = {}
                fiber_groups[bundle]['idx'] = rec_labels
                fiber_groups[bundle]['sl'] = oriented_sl
            else:
                fiber_groups[bundle] = oriented_sl
        self.fiber_groups = fiber_groups
        return fiber_groups


def clean_fiber_group(streamlines, n_points=100, clean_rounds=5,
                      clean_threshold=3, min_sl=20, stat=np.mean,
                      return_idx=False):
    """
    Clean a segmented fiber group based on the Mahalnobis distance of
    each streamline

    Parameters
    ----------

    streamlines : nibabel.Streamlines class instance.
        The streamlines constituting a fiber group.
        If streamlines is None, will use previously given streamlines.
        Default: None.

    clean_rounds : int, optional.
        Number of rounds of cleaning based on the Mahalanobis distance from
        the mean of extracted bundles. Default: 5

    clean_threshold : float, optional.
        Threshold of cleaning based on the Mahalanobis distance (the units are
        standard deviations). Default: 3.

    min_sl : int, optional.
        Number of streamlines in a bundle under which we will
        not bother with cleaning outliers. Default: 20.

    stat : callable, optional.
        The statistic of each node relative to which the Mahalanobis is
        calculated. Default: `np.mean` (but can also use median, etc.)

    using_idx : bool
        Whether 'streamlines' contains indices in the original streamlines.
        Default: False.

    Returns
    -------
    A nibabel.Streamlines class instance containing only the streamlines
    that have a Mahalanobis distance smaller than `clean_threshold` from
    the mean of each one of the nodes.
    """

    # We don't even bother if there aren't enough streamlines:
    if len(streamlines) < min_sl:
        return streamlines

    # Resample once up-front:
    fgarray = _resample_bundle(streamlines, n_points)

    # Keep this around, so you can use it for indexing at the very end:
    idx = np.arange(len(fgarray))
    # This calculates the Mahalanobis for each streamline/node:
    w = gaussian_weights(fgarray, return_mahalnobis=True, stat=stat)
    # We'll only do this for clean_rounds
    rounds_elapsed = 0
    while (np.any(w > clean_threshold)
           and rounds_elapsed < clean_rounds
           and len(streamlines) > min_sl):
        # Select the fibers that have Mahalanobis smaller than the
        # threshold for all their nodes:
        idx_belong = np.where(
            np.all(w < clean_threshold, axis=-1))[0]
        idx = idx[idx_belong.astype(int)]
        # Update by selection:
        fgarray = fgarray[idx_belong.astype(int)]
        # Repeat:
        w = gaussian_weights(fgarray, return_mahalnobis=True)
        rounds_elapsed += 1
    # Select based on the variable that was keeping track of things for us:
    out = streamlines[idx]
    if return_idx:
        return out, idx
    else:
        return out
