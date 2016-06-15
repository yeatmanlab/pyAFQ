import numpy as np
import scipy.ndimage as ndim
from distutils.version import LooseVersion

import nibabel as nib

import dipy
import dipy.data as dpd
import dipy.tracking.utils as dtu
import dipy.tracking.streamline as dts
import dipy.tracking.streamlinespeed as dps

import AFQ.registration as reg
import AFQ.utils.models as ut
import AFQ.data as afd
import AFQ._fixes as fix

if LooseVersion(dipy.__version__) < '0.12':
    # Monkey patch the fix in:
    dps.orient_by_rois = fix.orient_by_rois


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
            reg_template=None, mapping=None, as_generator=True, **reg_kwargs):
    """

    generate : bool
        Whether to generate the streamlines here, or return generators.

    reg_template : template to use for registration (defaults to the MNI T2)

    bundles: dict
        The format is something like::

             {'name': {'ROIs':[img, img], 'rules':[True, True]}}


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
        for ROI, rule in zip(bundles[bundle]['ROIs'], bundles[bundle]['rules']):
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
                                       copy=False)
        if as_generator:
            fiber_groups[bundle] = select_sl
        else:
            fiber_groups[bundle] = list(select_sl)

    return fiber_groups


def calculate_tract_profile(img, streamlines, affine=None, n_points=100,
                            weighting=None):
    """

    Parameters
    ----------
    img : 3D volume

    streamlines : list of arrays, or array

    weighting : 1D array or 2D array (optional)

    """
    if isinstance(streamlines, list):
        # Resample each streamline to the same number of points
        # list => array
        # Setting the number of points should happen in a streamline template
        # space, rather than in the subject native space, but for now we do
        # everything as in the Matlab version -- in native space.
        # In the future, an SLR object can be passed here, and then it would
        # move these streamlines into the template space before resampling...
        fgarray = np.array(dps.set_number_of_points(streamlines, n_points))

    # ...and move them back to native space before indexing into the volume:
    values = values_from_volume(img, fgarray, affine=affine)

    # We assume that weights *always sum to 1 across streamlines*:
    if weighting is None:
        w = np.ones(values.shape) / values.shape[0]

    tract_profile = np.sum(w * values, 0)

    #tract_profile = np.mean(w * values, 0)

    if weighting is not None:
        return weights, tract_profile
    else:
        return tract_profile


# def gaussian_weights(fgarray):
#     """
#     Let's considder calculating these weights within our fiber groups object so that this object carries the wweights with it
#     """
#     for node in range(fgarray.shape[1]):
#         # Grab all the coordinates at this node
#         n = fgarray[:,node]
#         # This should come back as a 3D covariance matrix with var(x), var(y),
#         # var(z) along the diagonal etc.
#         S = np.cov(n)
#         # Calculate the mean or median of this node as well
#         m = np.mean(n)
#         # Weights are the inverse of the Mahalanobis distance
#         for fn in range(fgarray.shape[0]):
#             # calculate Mahalanobis for node on fiber[fn]
#             w[fn,node] = np.sqrt(np.dot((n[fn]-m).T,np.inv(S),n[fn]-m))
