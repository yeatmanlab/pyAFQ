import numpy as np
import pimms
from time import time
import logging

import dipy.tracking.streamline as dts

import AFQ.recognition.utils as abu


logger = logging.getLogger('AFQ')


@pimms.calc("tol", "dist_to_atlas", "vox_dim")
def tolerance_mm_to_vox(img, dist_to_waypoint, input_dist_to_atlas):
    # We need to calculate the size of a voxel, so we can transform
    # from mm to voxel units:
    R = img.affine[0:3, 0:3]
    vox_dim = np.mean(np.diag(np.linalg.cholesky(R.T.dot(R))))

    # Tolerance is set to the square of the distance to the corner
    # because we are using the squared Euclidean distance in calls to
    # `cdist` to make those calls faster.
    if dist_to_waypoint is None:
        tol = dts.dist_to_corner(img.affine)
    else:
        tol = dist_to_waypoint / vox_dim
    dist_to_atlas = int(input_dist_to_atlas / vox_dim)
    return tol, dist_to_atlas, vox_dim


@pimms.calc("fgarray")
def fgarray(tg):
    """
    Streamlines resampled to 20 points.
    """
    logger.info("Resampling Streamlines...")
    start_time = time()
    fg_array = np.array(abu.resample_tg(tg, 20))
    logger.info((
        "Streamlines Resampled "
        f"(time: {time()-start_time}s)"))
    return fg_array


@pimms.calc("crosses")
def crosses(fgarray, img):
    """
    Classify the streamlines by whether they cross the midline.
    Creates a crosses attribute which is an array of booleans. Each boolean
    corresponds to a streamline, and is whether or not that streamline
    crosses the midline.
    """
    # What is the x,y,z coordinate of 0,0,0 in the template space?
    zero_coord = np.dot(np.linalg.inv(img.affine),
                        np.array([0, 0, 0, 1]))

    return np.logical_and(
        np.any(fgarray[:, :, 0] > zero_coord[0], axis=1),
        np.any(fgarray[:, :, 0] < zero_coord[0], axis=1))


# Things that can be calculated for multiple bundles at once
# (i.e., for a whole tractogram) go here
def get_preproc_plan(img, tg, dist_to_waypoint, dist_to_atlas):
    preproc_plan = pimms.Plan(
        tolerance_mm_to_vox=tolerance_mm_to_vox,
        fgarray=fgarray,
        crosses=crosses)
    return preproc_plan(
        img=img, tg=tg,
        dist_to_waypoint=dist_to_waypoint,
        input_dist_to_atlas=dist_to_atlas)
