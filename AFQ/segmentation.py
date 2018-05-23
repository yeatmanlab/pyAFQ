from distutils.version import LooseVersion

import numpy as np
import scipy.ndimage as ndim
from scipy.spatial.distance import mahalanobis, cdist

import nibabel as nib

import dipy
import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ._fixes as fix

if LooseVersion(dipy.__version__) < '0.12':
    # Monkey patch the fix in:
    dts.orient_by_rois = fix.orient_by_rois


__all__ = ["patch_up_roi", "segment"]


def patch_up_roi(roi):
    """
    After being non-linearly transformed, ROIs tend to have holes in them.
    We perform a couple of computational geometry operations on the ROI to
    fix that up.

    Parameters
    ----------
    roi : 3D binary array
        The ROI after it has been transformed

    Returns
    -------
    ROI after dilation and hole-filling
    """
    return ndim.binary_fill_holes(ndim.binary_dilation(roi).astype(int))


def _resample_bundle(streamlines, n_points):
    return np.array(dps.set_number_of_points(streamlines, n_points))


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
    if (isinstance(streamlines, list) or
            isinstance(streamlines, dts.Streamlines)):
        # Resample each streamline to the same number of points
        # list => np.array
        # Setting the number of points should happen in a streamline template
        # space, rather than in the subject native space, but for now we do
        # everything as in the Matlab version -- in native space.
        # In the future, an SLR object can be passed here, and then it would
        # move these streamlines into the template space before resampling...
        fgarray = _resample_bundle(streamlines, n_points)
    else:
        fgarray = streamlines
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
    if isinstance(bundle, list) or isinstance(bundle, dts.Streamlines):
        # if you got a list, assume that it needs to be resampled:
        bundle = _resample_bundle(bundle, n_points)
    else:
        if bundle.shape[-1] != 3:
            e_s = "Input must be shape (n_streamlines, n_points, 3)"
            raise ValueError(e_s)
        n_points = bundle.shape[1]

    w = np.zeros((bundle.shape[0], n_points))
    # If there's only one fiber here, it gets the entire weighting:
    if bundle.shape[0] == 1:
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


def segment(fdata, fbval, fbvec, streamlines, bundles,
            reg_template=None, mapping=None,
            clean_rounds=5, clean_threshold=6, min_sl=20,
            prob_threshold=0, **reg_kwargs):
    """
    Segment streamlines into bundles based on inclusion ROIs.

    Parameters
    ----------
    fdata, fbval, fbvec : str Full path to data, bvals, bvecs

    streamlines : list of 2D arrays Each array is a streamline, shape
        (3, N).

    bundles: dict The format is something like::

             {'name': {'ROIs':[img, img], 'rules':[True, True]}}

    reg_template : str or nib.Nifti1Image, optional. Template to use for
        registration (defaults to the MNI T2)

    mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional
        A mapping between DWI space and a template. Defaults to generate
        this.

    clean_rounds : int, optional.
        Number of rounds of cleaning based on the Mahalanobis distance from the
        mean of extracted bundles. Default: 5

    clean_threshold : float, optional.
        Threshold of cleaning based on the Mahalanobis distance (the units
        are standard deviations). Default: 6.

    prob_threshold : float.
        Cleaning of fiber groups is done using probability maps
        from [Hua2008]_. Here, we choose an average probability that needs to
        be exceeded for an individual streamline to be retained. Default: 0.

    min_sl : int
       Number of streamlines in a bundle under which we will not bother with
       cleaning outliers.

    References
    ----------
    .. [Hua2008] Hua K, Zhang J, Wakana S, Jiang H, Li X, et al. (2008)
    Tract probability maps in stereotaxic spaces: analyses of white
    matter anatomy and tract-specific quantification. Neuroimage 39:
    336-347
    """
    img, _, gtab, _ = ut.prepare_data(fdata, fbval, fbvec)
    tol = dts.dist_to_corner(img.affine)

    xform_sl = dts.Streamlines(dtu.move_streamlines(streamlines,
                                                    np.linalg.inv(img.affine)))

    if reg_template is None:
        reg_template = dpd.read_mni_template()

    if mapping is None:
        mapping = reg.syn_register_dwi(fdata, gtab, template=reg_template,
                                       **reg_kwargs)

    if isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
        mapping = reg.read_mapping(mapping, img, reg_template)

    fiber_probabilities = np.zeros((len(xform_sl), len(bundles)))

    # For expedience, we approximate each streamline as a 100 point curve:
    fgarray = _resample_bundle(xform_sl, 100)
    streamlines_in_bundles = np.zeros((len(xform_sl), len(bundles)))

    fiber_groups = {}

    for bundle_idx, bundle in enumerate(bundles):
        print(bundle)

        prob_map = bundles[bundle]['prob_map']
        if not isinstance(prob_map, np.ndarray):
            prob_map = prob_map.get_data()
        warped_prob_map = mapping.transform_inverse(prob_map,
                                                    interpolation='nearest')
        fiber_probabilities = dts.values_from_volume(warped_prob_map,
                                                     fgarray)
        fiber_probabilities = np.mean(fiber_probabilities, -1)

        ROI0 = bundles[bundle]['ROIs'][0]
        ROI1 = bundles[bundle]['ROIs'][1]
        if not isinstance(ROI0, np.ndarray):
            ROI0 = ROI0.get_data()

        warped_ROI0 = patch_up_roi(
            mapping.transform_inverse(
                ROI0,
                interpolation='nearest')).astype(bool)

        if not isinstance(ROI1, np.ndarray):
            ROI1 = ROI1.get_data()

        warped_ROI1 = patch_up_roi(
            mapping.transform_inverse(
                ROI1,
                interpolation='nearest')).astype(bool)

        roi_coords0 = np.array(np.where(warped_ROI0)).T
        roi_coords1 = np.array(np.where(warped_ROI1)).T

        # Replace this here, so we can reuse below:
        bundles[bundle]['ROIs'][0] = roi_coords0
        bundles[bundle]['ROIs'][1] = roi_coords1

        crosses_midline = bundles[bundle]['cross_midline']

        for sl_idx, sl in enumerate(xform_sl):
            if fiber_probabilities[sl_idx] > prob_threshold:
                if crosses_midline is not None:
                    if (np.any(sl[:, 0] > img.shape[0] // 2) and
                            np.any(sl[:, 0] < img.shape[0] // 2)):
                        # This means that the streamline does
                        # cross the midline:
                        if crosses_midline:
                            # This is what we want, keep going
                            pass
                        else:
                            # This is not what we want, skip to next streamline
                            continue
                if dts.streamline_near_roi(sl, roi_coords0, tol=tol):
                    if dts.streamline_near_roi(sl, roi_coords1, tol=tol):
                        streamlines_in_bundles[sl_idx, bundle_idx] =\
                            fiber_probabilities[sl_idx]

    # Eliminate any fibers not selected using the plane ROIs:
    possible_fibers = np.sum(streamlines_in_bundles, -1) > 0
    xform_sl = xform_sl[possible_fibers]
    streamlines_in_bundles = streamlines_in_bundles[possible_fibers]
    bundle_choice = np.argmax(streamlines_in_bundles, -1)

    for bundle_idx, bundle in enumerate(bundles):
        print(bundle)
        select_idx = np.where(bundle_choice == bundle_idx)
        # Use a list here, because Streamlines don't support item assignment:
        select_sl = list(xform_sl[select_idx])
        if len(select_sl) == 0:
            fiber_groups[bundle] = dts.Streamlines([])
            # There's nothing here, move to the next bundle:
            continue

        # Next, we reorient each streamline according to
        # an ARBITRARY, but CONSISTENT order:
        roi_coords0 = bundles[bundle]['ROIs'][0]
        roi_coords1 = bundles[bundle]['ROIs'][1]

        for idx in range(len(select_sl)):
            this_sl = select_sl[idx]
            dist0 = cdist(this_sl, roi_coords0, 'euclidean')
            dist1 = cdist(this_sl, roi_coords1, 'euclidean')
            min0 = np.argmin(dist0, 0)[0]
            min1 = np.argmin(dist1, 0)[0]
            if min0 > min1:
                this_sl = this_sl[::-1]
            select_sl[idx] = this_sl
        # We'll use a Streamlines object for the next steps
        # because these objects support indexing with arrays:
        select_sl = dts.Streamlines(select_sl)
        print(len(select_sl))

        # Next, clean using distance from the mean fiber:
        if clean_rounds:
            if len(select_sl) > min_sl:
                w = gaussian_weights(select_sl, n_points=100,
                                     return_mahalnobis=True)
                rounds_elapsed = 0
                while (np.any(w > clean_threshold) and
                       rounds_elapsed < clean_rounds and
                       len(select_sl) > min_sl):
                    idx_belong = np.where(
                        np.all(w < clean_threshold, axis=-1))[0]
                    select_sl = select_sl[idx_belong.astype(int)]
                    w = gaussian_weights(select_sl,
                                         n_points=100,
                                         return_mahalnobis=True)
                    rounds_elapsed += 1

        print(len(select_sl))
        fiber_groups[bundle] = select_sl

    return fiber_groups
