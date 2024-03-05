from skimage.morphology import skeletonize_3d
import numpy as np
import nibabel as nib


def _find_longest_true_series_indices(input_list):
    max_length = 0
    current_length = 0
    start_index = 0
    max_start_index = -1
    max_end_index = -1

    for i, value in enumerate(input_list):
        if value:
            if current_length == 0:
                start_index = i
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start_index = start_index
                max_end_index = i - 1
            current_length = 0

    if current_length > max_length:
        max_length = current_length
        max_start_index = start_index
        max_end_index = len(input_list) - 1

    if max_length > 0:
        return max_start_index, max_end_index
    else:
        return 0, -1


def skeleton_from_roi(roi, affine, orientation_axis, jump_threshold=3):
    """
    Skeletonize a region of interest (ROI)

    Parameters
    ----------
    roi : ndarray
        A 3D binary array with the ROI.
    affine : ndarray
        The affine transformation of the ROI.
    orientation_axis : ndarray
        Axis used to orient profile. One of
        "L", "P", "I".
    jump_threshold : int, optional
        The maximum distance between two points in the
        skeleton of the ROI. Used to prune the skeleton.
        Default: 3.
    """
    # First, skeletonize the ROI
    print(np.sum(roi))
    skeleton = skeletonize_3d(roi)
    skel_pts = np.asarray(np.where(skeleton)).T

    # Then, remove any jumps in the skeleton
    # that are less than the jump_threshold
    # (i.e., remove any small branches in the skeleton)
    skel_jumps = np.linalg.norm(
        skel_pts[1:] - skel_pts[:-1], ord=2, axis=1) < jump_threshold
    sidx, eidx = _find_longest_true_series_indices(skel_jumps)
    skel_pts = skel_pts[sidx:eidx]

    # Reorient the skeleton if necessary
    orientation = nib.aff2axcodes(affine)
    if orientation_axis not in ["L", "P", "I"]:
        raise ValueError("Invalid orientation_axis. "
                         "Valid options are 'L', 'P', 'I'.")
    elif orientation_axis == "L":
        if np.logical_xor(
                skel_pts[0, 0] - skel_pts[-1, 0] < 0,
                orientation[0] == "L"):
            skel_pts = skel_pts[::-1]
    elif orientation_axis == "P":
        if np.logical_xor(
                skel_pts[0, 1] - skel_pts[-1, 1] < 0,
                orientation[1] == "P"):
            skel_pts = skel_pts[::-1]
    elif orientation_axis == "I":
        if np.logical_xor(
                skel_pts[0, 2] - skel_pts[-1, 2] < 0,
                orientation[2] == "I"):
            skel_pts = skel_pts[::-1]

    return skel_pts


def profile_roi(
        roi, skel_pts, scalar_data, d_plane_thresh=1):
    """
    Calculate the tract profile of a set of scalars
    within a region of interest (ROI). Finds the maximum
    value of each scalar within a disk
    around each point in the skeleton of the ROI.

    Parameters
    ----------
    roi : ndarray
        A 3D binary array with the ROI.
    skel_pts : ndarray
        A 2D array with the coordinates of the
        skeleton of the ROI.
    scalar_data : ndarray
        A 3D array with the scalar data.
    d_plane_thresh : int, optional
        The maximum distance between a point in the
        ROI and the plane defined by the skeleton of
        the ROI.
        Default: 1.
    """
    pts_len = skel_pts.shape[0]
    roi_rad = int(np.sqrt((np.sum(roi) / pts_len) / np.pi))
    tract_profile = np.full(pts_len, -np.inf)
    for nodeid in range(pts_len):
        min_idx = max(0, nodeid - 1)
        max_idx = min(pts_len, nodeid + 1)
        n_vec = skel_pts[min_idx] - skel_pts[max_idx]
        n_vec = n_vec / np.linalg.norm(n_vec)
        c_pt = skel_pts[nodeid]

        dr = np.zeros((3, 2), dtype=int)
        for dim in range(3):
            dr[dim, 0] = int(c_pt[dim] - roi_rad)
            dr[dim, 1] = int(c_pt[dim] + roi_rad)
        for ii in range(dr[0, 0], dr[0, 1]):
            for jj in range(dr[1, 0], dr[1, 1]):
                for kk in range(dr[2, 0], dr[2, 1]):
                    if roi[ii, jj, kk]:
                        euc_d = c_pt - np.asarray([ii, jj, kk])
                        d_plane = np.abs(np.sum(n_vec * euc_d))
                        d_point = np.sum(euc_d**2)**0.5
                        if d_plane <= d_plane_thresh and d_point <= roi_rad:
                            if scalar_data[ii, jj, kk] >\
                                    tract_profile[nodeid]:
                                tract_profile[nodeid] =\
                                    scalar_data[ii, jj, kk]

    # Interpolate the tract profile to have 100 points
    tract_profile = np.interp(
        np.linspace(0, pts_len - 1, num=100),
        np.linspace(0, pts_len - 1, num=pts_len),
        tract_profile)

    return tract_profile
