from distutils.version import LooseVersion

import numpy as np
from scipy.spatial.distance import mahalanobis, cdist
import pandas as pd

import nibabel as nib
from glob import glob
import os.path as op

import dipy
import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps
from dipy.segment.bundles import RecoBundles
from dipy.segment.clustering import qbx_and_merge

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.utils.volume as auv
# import AFQ.utils.streamlines as aus
import AFQ._fixes as fix

if LooseVersion(dipy.__version__) < '0.12':
    # Monkey patch the fix in:
    dts.orient_by_rois = fix.orient_by_rois


__all__ = ["segment"]


def _resample_bundle(streamlines, n_points):
    return np.array(dps.set_number_of_points(streamlines, n_points))


# make dictionary of ROIs from excel sheet
def build_volumetric_atlas_dict(track_name, roi_df):
    rois = roi_df[track_name]
    roi_groups = list(rois.value_counts().index)

    roi_dict = {}
    roi_dict['include'] = {}
    roi_dict['exclude'] = {}

    for i in roi_groups:
        if i > 0:
            roitype = 'include'
        elif i < 0:
            roitype = 'exclude'
        else:
            raise('UHOH')
        setname = 'set' + str(int(i))
        roi_dict[roitype][setname] = {}

        temp = roi_df[rois == i]
        for j, name in enumerate(temp['VOIS']):
            roi_dict[roitype][setname][name] = temp['aparc+aseg'].values[j]
    return roi_dict


# combine the rois into a single "NOT" exclusion and sets of "AND" inclusion
def combine_rois(mydict, apac):
    x, y, z = apac.shape

    include_coords = []
    exclude_coords = []

    for i, iset in enumerate(mydict['include'].keys()):
        include = np.zeros([x, y, z])
        for item in mydict['include'][iset].items():
            include += 1 * (apac == item[-1])
        include_coords.append(np.array(np.where(include > 0)).T)
    for j, jset in enumerate(mydict['exclude'].keys()):
        for item in mydict['exclude'][jset].items():
            exclude_coords.append(np.array(np.where(apac == item[-1])).T)
    return include_coords, exclude_coords


# targeting script to target streamlines with ROIs
def check_targets(sls, include, exclude, tol):
    keep_array = np.zeros(len(sls))
    for i, sl in enumerate(sls):
        is_close_include, dist = _check_sl_with_inclusion(sl, include, tol)
        if is_close_include:
            is_close_exclude = _check_sl_with_exclusion(sl, exclude, tol)
            if is_close_exclude:
                keep_array[i] = 1
    return keep_array


def calculate_volumetric_atlas_score(streamlines, csv_lookup_path,
                                     volumetric_atlas_path,
                                     track_list_path, file_map_path, aff, tol):
    # Here we assume the csv column names match track_list names
    roi_df = pd.read_excel(csv_lookup_path)
    track_list_df = pd.read_excel(track_list_path, index_col='column_name')
    track_list = list(track_list_df[track_list_df.segment_this > 0].index)

    # Build dictionary of include/exclude rois and freesurfer codes from csv
    roi_dict = {}
    for tr in track_list:
        roi_dict[tr] = build_volumetric_atlas_dict(tr, roi_df=roi_df)
    atlas_data = nib.load(volumetric_atlas_path).get_data()

    streamlines_by_bundle = np.zeros([len(streamlines), len(track_list)])
    for i, tr in enumerate(track_list):
        print(tr)
        xfmd_sls = dts.Streamlines(
            dtu.move_streamlines(streamlines, np.linalg.inv(aff)))
        include, exclude = combine_rois(roi_dict[tr], atlas_data)
        streamlines_by_bundle[:, i], test_bundle = check_targets(
            xfmd_sls, include, exclude, tol)

    return streamlines_by_bundle


# make dictionary of bundles from excel sheet
def build_bundle_atlas_dict(bundle_atlas_directory, wb_streamlines, df):
    bundles = glob(bundle_atlas_directory + '/*.trk')
    bundle_dict = {}
    for bpath in bundles:
        basename = op.basename(bpath)
        bname = list(df[df['file_name'] == basename]['column_name'])[0]
        prune = list(df[df['file_name'] == basename]['pruning_thresh'])[0]

        bundle_dict[bname] = {}
        bundle_dict[bname]['path'] = bpath
        bundle_dict[bname]['pruning_threshold'] = prune
    return bundle_dict


def calculate_bundle_atlas_score(bundle_atlas_dict, whole_brain,
                                 template2sub_xfm,
                                 cluster_thr=5, pruning_thr=10,
                                 reduction_thr=10):

    wb_arrayseq = nib.streamlines.ArraySequence(whole_brain)

    cluster_map = qbx_and_merge(whole_brain, thresholds=[
                                40, 25, 20, 10], verbose=False)

    rb = RecoBundles(wb_arrayseq, cluster_map=cluster_map, verbose=False)
    track_list = list(bundle_atlas_dict.keys())
    streamlines_by_bundle = np.zeros([len(whole_brain), len(track_list)])
    for i, b in enumerate(track_list):
        print(b)
        trk = nib.streamlines.trk.TrkFile.load(bundle_atlas_dict[b]['path'])
        b_tg = trk.tractogram

        bundle_xfmd = b_tg.copy().apply_affine(template2sub_xfm).streamlines

        b_atlassp, labels, b_subsp = rb.recognize(model_bundle=bundle_xfmd,
                                                  model_clust_thr=5.,
                                                  reduction_thr=10,
                                                  pruning_thr=pruning_thr)
        for l in labels:
            streamlines_by_bundle[l, i] = 1

    return streamlines_by_bundle


def calculate_tract_profile(img, streamlines, affine=None, n_points=100,
                            weights=None):
    """

    Parameters
    ----------
    img : 3D volume

    streamlines : list of arrays, or array

    weights : 1D array or 2D array (optional)
        Weight each streamline (1D) or each node (2D) when calculating the
        tract-profiles. Must sum to 1 across streamlines (in each node if
        relevant).

    """
    # It's already an array
    if isinstance(streamlines, np.ndarray):
        fgarray = streamlines
    else:
        # It's some other kind of thing (list, Streamlines, etc.).
        # Resample each streamline to the same number of points
        # list => np.array
        # Setting the number of points should happen in a streamline template
        # space, rather than in the subject native space, but for now we do
        # everything as in the Matlab version -- in native space.
        # In the future, an SLR object can be passed here, and then it would
        # move these streamlines into the template space before resampling...
        fgarray = _resample_bundle(streamlines, n_points)
    # ...and move them back to native space before indexing into the volume:
    values = dts.values_from_volume(img, fgarray, affine=affine)

    # We assume that weights *always sum to 1 across streamlines*:
    if weights is None:
        weights = np.ones(values.shape) / values.shape[0]

    tract_profile = np.sum(weights * values, 0)
    return tract_profile


def gaussian_weights(bundle, n_points=100, return_mahalnobis=False):
    """
    Calculate weights for each streamline/node in a bundle, based on a
    Mahalanobis distance from the mean of the bundle, at that node

    Parameters
    ----------
    bundle : array or list
        If this is a list, assume that it is a list of streamline coordinates
        (each entry is a 2D array, of shape n by 3). If this is an array, this
        is a resampled version of the streamlines, with equal number of points
        in each streamline.
    n_points : int, optional
        The number of points to resample to. *If the `bundle` is an array, this
        input is ignored*. Default: 100.

    Returns
    -------
    w : array of shape (n_streamlines, n_points)
        Weights for each node in each streamline, calculated as its relative
        inverse of the Mahalanobis distance, relative to the distribution of
        coordinates at that node position across streamlines.
    """
    if isinstance(bundle, np.ndarray):
        # It's an array, go with it:
        n_points = bundle.shape[1]
    else:
        # It's something else, assume that it needs to be resampled:
        bundle = _resample_bundle(bundle, n_points)
    w = np.zeros((bundle.shape[0], n_points))

    # If there's only one fiber here, it gets the entire weighting:
    if bundle.shape[0] == 1:
        if return_mahalnobis:
            return np.array([np.nan])
        else:
            return np.array([1])

    for node in range(bundle.shape[1]):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines
        # This is a 3-by-3 array:
        node_coords = bundle[:, node]
        c = np.cov(node_coords.T, ddof=0)
        c = np.array([[c[0, 0], c[0, 1], c[0, 2]],
                      [0, c[1, 1], c[1, 2]],
                      [0, 0, c[2, 2]]])
        # Calculate the mean or median of this node as well
        # delta = node_coords - np.mean(node_coords, 0)
        m = np.mean(node_coords, 0)
        # Weights are the inverse of the Mahalanobis distance
        for fn in range(bundle.shape[0]):
            # calculate Mahalanobis for node on fiber[fn]
            w[fn, node] = mahalanobis(node_coords[fn], m, np.linalg.inv(c))
    if return_mahalnobis:
        return w
    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    w = 1 / w
    # Normalize before returning, so that the weights in each node sum to 1:
    return w / np.sum(w, 0)


def split_streamlines(streamlines, template, low_coord=10):
    """
    Classify streamlines and split sl passing the mid-point below some height.
    Parameters
    ----------
    streamlines : list or Streamlines class instance.
    template : nibabel.Nifti1Image class instance
        An affine transformation into a template space.
    low_coords: int
        How many coordinates below the 0,0,0 point should a streamline be to
        be split if it passes the midline.
    Returns
    -------
    streamlines that have been processed, a boolean array of whether they
    cross the midline or not, a boolean array that for those who do not cross
    designates whether they are strictly in the left hemisphere, and a boolean
    that tells us whether the streamline has superior-inferior parts that pass
    below `low_coord` steps below the middle of the image (which should also
    be `low_coord` mms for templates with 1 mm resolution)
    """
    # What is the x,y,z coordinate of 0,0,0 in the template space?
    zero_coord = np.dot(np.linalg.inv(template.affine),
                        np.array([0, 0, 0, 1]))

    # cross_below = zero_coord[2] - low_coord
    crosses = np.zeros(len(streamlines), dtype=bool)
    # already_split = 0
    for sl_idx, sl in enumerate(streamlines):
        if np.any(sl[:, 0] > zero_coord[0]) and \
           np.any(sl[:, 0] < zero_coord[0]):
            # if np.any(sl[:, 2] < cross_below):
            #     # This is a streamline that needs to be split where it
            #     # crosses the midline:
            #     split_idx = np.argmin(np.abs(sl[:, 0] - zero_coord[0]))
            #     streamlines = aus.split_streamline(
            #         streamlines, sl_idx + already_split, split_idx)
            #     already_split = already_split + 1
            #     # Now that it's been split, neither cross the midline:
            #     crosses[sl_idx] = False
            #     crosses = np.concatenate([crosses[:sl_idx+1],
            #                               np.array([False]),
            #                               crosses[sl_idx+1:]])
            # else:
            crosses[sl_idx] = True
        else:
            crosses[sl_idx] = False

    # Move back to the original space:
    return streamlines, crosses


def _check_sl_with_inclusion(sl, include_rois, tol):
    """
    Helper function to check that a streamline is close to a list of
    inclusion ROIS.
    """
    dist = []
    for roi in include_rois:
        dist.append(cdist(sl, roi, 'euclidean'))
        if np.min(dist[-1]) > tol:
            # Too far from one of them:
            return False, []
    # Apparently you checked all the ROIs and it was close to all of them
    return True, dist


def _check_sl_with_exclusion(sl, exclude_rois, tol):
    """ Helper function to check that a streamline is not too close to a list
    of exclusion ROIs.
    """
    for roi in exclude_rois:
        if np.min(cdist(sl, roi, 'euclidean')) < tol:
            return False
    # Either there are no exclusion ROIs, or you are not close to any:
    return True


def segment(fdata, fbval, fbvec, streamlines, bundle_dict, b0_threshold=0,
            reg_template=None, mapping=None, prob_threshold=0,
            **reg_kwargs):
    """
    Segment streamlines into bundles based on inclusion ROIs.

    Parameters
    ----------
    fdata, fbval, fbvec : str
        Full path to data, bvals, bvecs

    streamlines : list of 2D arrays
        Each array is a streamline, shape (3, N).

    bundle_dict: dict
        The format is something like::

            {'name': {'ROIs':[img1, img2],
            'rules':[True, True]},
            'prob_map': img3,
            'cross_midline': False}

    reg_template : str or nib.Nifti1Image, optional.
        Template to use for registration (defaults to the MNI T2)

    mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional
        A mapping between DWI space and a template. Defaults to generate
        this.

    prob_threshold : float.
        Initial cleaning of fiber groups is done using probability maps from
        [Hua2008]_. Here, we choose an average probability that needs to be
        exceeded for an individual streamline to be retained. Default: 0.

    References
    ----------
    .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
       Tract probability maps in stereotaxic spaces: analyses of white
       matter anatomy and tract-specific quantification. Neuroimage 39:
       336-347
    """
    img, _, gtab, _ = ut.prepare_data(fdata, fbval, fbvec,
                                      b0_threshold=b0_threshold)

    tol = dts.dist_to_corner(img.affine)

    if reg_template is None:
        reg_template = dpd.read_mni_template()

    if mapping is None:
        mapping = reg.syn_register_dwi(fdata, gtab, template=reg_template,
                                       **reg_kwargs)

    # Classify the streamlines and split those that: 1) cross the
    # midline, and 2) pass under 10 mm below the mid-point of their
    # representation in the template space:
    xform_sl, crosses = split_streamlines(streamlines, img)

    if isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
        mapping = reg.read_mapping(mapping, img, reg_template)

    fiber_probabilities = np.zeros((len(xform_sl), len(bundle_dict)))

    # For expedience, we approximate each streamline as a 100 point curve:
    fgarray = _resample_bundle(xform_sl, 100)
    streamlines_in_bundles = np.zeros((len(xform_sl), len(bundle_dict)))
    min_dist_coords = np.zeros((len(xform_sl), len(bundle_dict), 2))

    fiber_groups = {}

    for bundle_idx, bundle in enumerate(bundle_dict):
        rules = bundle_dict[bundle]['rules']
        include_rois = []
        exclude_rois = []
        for rule_idx, rule in enumerate(rules):
            roi = bundle_dict[bundle]['ROIs'][rule_idx]
            if not isinstance(roi, np.ndarray):
                roi = roi.get_data()
            warped_roi = auv.patch_up_roi(
                mapping.transform_inverse(
                    roi,
                    interpolation='nearest')).astype(bool)
            if rule:
                # include ROI:
                include_rois.append(np.array(np.where(warped_roi)).T)
            else:
                # Exclude ROI:
                exclude_rois.append(np.array(np.where(warped_roi)).T)

        crosses_midline = bundle_dict[bundle]['cross_midline']

        # The probability map if doesn't exist is all ones with the same
        # shape as the ROIs:
        prob_map = bundle_dict[bundle].get('prob_map', np.ones(roi.shape))

        if not isinstance(prob_map, np.ndarray):
            prob_map = prob_map.get_data()
        warped_prob_map = mapping.transform_inverse(prob_map,
                                                    interpolation='nearest')
        fiber_probabilities = dts.values_from_volume(warped_prob_map,
                                                     fgarray)
        fiber_probabilities = np.mean(fiber_probabilities, -1)

        for sl_idx, sl in enumerate(xform_sl):
            if fiber_probabilities[sl_idx] > prob_threshold:
                if crosses_midline is not None:
                    # This means that the streamline does
                    # cross the midline:
                    if crosses[sl_idx]:
                        if crosses_midline:
                            # This is what we want, keep going
                            pass
                        else:
                            # This is not what we want, skip to next streamline
                            continue

                is_close, dist = _check_sl_with_inclusion(sl, include_rois,
                                                          tol)
                if is_close:
                    is_far = _check_sl_with_exclusion(sl, exclude_rois,
                                                      tol)
                    if is_far:
                        min_dist_coords[sl_idx, bundle_idx, 0] =\
                            np.argmin(dist[0], 0)[0]
                        min_dist_coords[sl_idx, bundle_idx, 1] =\
                            np.argmin(dist[1], 0)[0]
                        streamlines_in_bundles[sl_idx, bundle_idx] =\
                            fiber_probabilities[sl_idx]

    # Eliminate any fibers not selected using the plane ROIs:
    possible_fibers = np.sum(streamlines_in_bundles, -1) > 0
    xform_sl = xform_sl[possible_fibers]
    streamlines_in_bundles = streamlines_in_bundles[possible_fibers]
    min_dist_coords = min_dist_coords[possible_fibers]
    bundle_choice = np.argmax(streamlines_in_bundles, -1)

    # We do another round through, so that we can orient all the
    # streamlines within a bundle in the same orientation with respect to
    # the ROIs. This order is ARBITRARY but CONSISTENT (going from ROI0
    # to ROI1).
    for bundle_idx, bundle in enumerate(bundle_dict):
        select_idx = np.where(bundle_choice == bundle_idx)
        # Use a list here, because Streamlines don't support item assignment:
        select_sl = list(xform_sl[select_idx])
        if len(select_sl) == 0:
            fiber_groups[bundle] = dts.Streamlines([])
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
        fiber_groups[bundle] = select_sl

    return fiber_groups


def clean_fiber_group(streamlines, n_points=100, clean_rounds=5,
                      clean_threshold=6, min_sl=20):
    """
    Clean a segmented fiber group based on the Mahalnobis distance of
    each streamline

    Parameters
    ----------

    streamlines : nibabel.Streamlines class instance. The streamlines
        constituting a fiber group.

    clean_rounds : int, optional. Number of rounds of cleaning based on
        the Mahalanobis distance from the mean of extracted bundles.
        Default: 5

    clean_threshold : float, optional. Threshold of cleaning based on the
        Mahalanobis distance (the units are standard deviations).
        Default: 6.

    min_sl : int Number of streamlines in a bundle under which we will
       not bother with cleaning outliers.

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
    idx = np.arange(fgarray.shape[0])
    # This calculates the Mahalanobis for each streamline/node:
    w = gaussian_weights(fgarray, return_mahalnobis=True)
    # We'll only do this for clean_rounds
    rounds_elapsed = 0
    while (np.any(w > clean_threshold) and
            rounds_elapsed < clean_rounds and
            len(streamlines) > min_sl):
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
    return streamlines[idx]
