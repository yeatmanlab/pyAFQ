from distutils.version import LooseVersion

import numpy as np
import scipy.ndimage as ndim
from scipy.spatial.distance import mahalanobis

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


def segment(fdata, fbval, fbvec, streamlines, bundles,
            reg_template=None, mapping=None, as_generator=True,
            clip_to_roi=True, **reg_kwargs):
    """
    Segment streamlines into bundles.

    Parameters
    ----------
    fdata, fbval, fbvec : str
        Full path to data, bvals, bvecs

    streamlines : list of 2D arrays
        Each array is a streamline, shape (3, N).

    bundles: dict
        The format is something like::

             {'name': {'ROIs':[img, img], 'rules':[True, True]}}

    reg_template : str or nib.Nifti1Image, optional.
        Template to use for registration (defaults to the MNI T2)

    mapping : DiffeomorphicMap object, str or nib.Nifti1Image, optional
        A mapping between DWI space and a template. Defaults to generate this.

    as_generator : bool, optional
        Whether to generate the streamlines here, or return generators.
        Default: True.

    clip_to_roi : bool, optional
        Whether to clip the streamlines between the ROIs
    """
    img, data, gtab, mask = ut.prepare_data(fdata, fbval, fbvec)
    xform_sl = [s for s in dtu.move_streamlines(streamlines,
                                                np.linalg.inv(img.affine))]

    if reg_template is None:
        reg_template = dpd.read_mni_template()

    if mapping is None:
        mapping = reg.syn_register_dwi(fdata, gtab, template=reg_template,
                                       **reg_kwargs)

    if isinstance(mapping, str) or isinstance(mapping, nib.Nifti1Image):
        mapping = reg.read_mapping(mapping, img, reg_template)

    fiber_groups = {}
    for bundle in bundles:
        select_sl = xform_sl
        for ROI, rule in zip(bundles[bundle]['ROIs'],
                             bundles[bundle]['rules']):
            data = ROI.get_data()
            warped_ROI = patch_up_roi(mapping.transform_inverse(
                data,
                interpolation='nearest'))
            # This function requires lists as inputs:
            select_sl = dts.select_by_rois(select_sl,
                                           [warped_ROI.astype(bool)],
                                           [rule])
        # Next, we reorient each streamline according to an ARBITRARY, but
        # CONSISTENT order. To do this, we use the first ROI for which the rule
        # is True as the first one to pass through, and the last ROI for which
        # the rule is True as the last one to pass through:

        # Indices where the 'rule' is True:
        idx = np.where(bundles[bundle]['rules'])

        orient_ROIs = [bundles[bundle]['ROIs'][idx[0][0]],
                       bundles[bundle]['ROIs'][idx[0][-1]]]

        select_sl = dts.orient_by_rois(select_sl,
                                       orient_ROIs[0].get_data(),
                                       orient_ROIs[1].get_data(),
                                       as_generator=True)
        if clip_to_roi:
            dts.clip()

        #selec
        if as_generator:
            fiber_groups[bundle] = select_sl
        else:
            fiber_groups[bundle] = list(select_sl)

    return fiber_groups


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
    if isinstance(streamlines, list):
        # Resample each streamline to the same number of points
        # list => np.array
        # Setting the number of points should happen in a streamline template
        # space, rather than in the subject native space, but for now we do
        # everything as in the Matlab version -- in native space.
        # In the future, an SLR object can be passed here, and then it would
        # move these streamlines into the template space before resampling...
        fgarray = np.array(dps.set_number_of_points(streamlines, n_points))
    else:
        fgarray = streamlines
    # ...and move them back to native space before indexing into the volume:
    values = dts.values_from_volume(img, fgarray, affine=affine)

    # We assume that weights *always sum to 1 across streamlines*:
    if weights is None:
        weights = np.ones(values.shape) / values.shape[0]

    tract_profile = np.sum(weights * values, 0)
    return tract_profile


def gaussian_weights(bundle, n_points=100):
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
    if isinstance(bundle, list):
        # if you got a list, assume that it needs to be resampled:
        bundle = np.array(dps.set_number_of_points(bundle, n_points))
    else:
        if bundle.shape[-1] != 3:
            e_s = "Input must be shape (n_streamlines, n_points, 3)"
            raise ValueError(e_s)
        n_points = bundle.shape[1]

    w = np.zeros((bundle.shape[0], n_points))
    for node in range(bundle.shape[1]):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines
        # This is a 3-by-3 array:
        node_coords = bundle[:, node]
        c = np.cov(node_coords.T, ddof=0)
        # Calculate the mean or median of this node as well
        # delta = node_coords - np.mean(node_coords, 0)
        m = np.mean(node_coords, 0)
        # Weights are the inverse of the Mahalanobis distance
        for fn in range(bundle.shape[0]):
            # calculate Mahalanobis for node on fiber[fn]
            w[fn, node] = mahalanobis(node_coords[fn], m, c)
    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    w = 1 / w
    # Normalize before returning, so that the weights in each node sum to 1:
    return w / np.sum(w, 0)
