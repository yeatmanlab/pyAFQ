from collections import OrderedDict
import os.path as op

import numpy as np
from palettable.tableau import Tableau_20
import imageio as io

import nibabel as nib

import AFQ.utils.volume as auv
import AFQ.registration as reg

tableau_20_rgb = np.array(Tableau_20.colors) / 255 - 0.0001

COLOR_DICT = OrderedDict({"ATR_L": tableau_20_rgb[0],
                          "ATR_R": tableau_20_rgb[1],
                          "CST_L": tableau_20_rgb[2],
                          "CST_R": tableau_20_rgb[3],
                          "CGC_L": tableau_20_rgb[4],
                          "CGC_R": tableau_20_rgb[5],
                          "HCC_L": tableau_20_rgb[6],
                          "HCC_R": tableau_20_rgb[7],
                          "FP": tableau_20_rgb[8],
                          "FA": tableau_20_rgb[9],
                          "IFO_L": tableau_20_rgb[10],
                          "IFO_R": tableau_20_rgb[11],
                          "ILF_L": tableau_20_rgb[12],
                          "ILF_R": tableau_20_rgb[13],
                          "SLF_L": tableau_20_rgb[14],
                          "SLF_R": tableau_20_rgb[15],
                          "UNC_L": tableau_20_rgb[16],
                          "UNC_R": tableau_20_rgb[17],
                          "ARC_L": tableau_20_rgb[18],
                          "ARC_R": tableau_20_rgb[19]})

POSITIONS = OrderedDict({"ATR_L": (1, 0), "ATR_R": (1, 4),
                         "CST_L": (1, 1), "CST_R": (1, 3),
                         "CGC_L": (3, 1), "CGC_R": (3, 3),
                         "HCC_L": (4, 1), "HCC_R": (4, 3),
                         "FP": (4, 2), "FA": (0, 2),
                         "IFO_L": (4, 0), "IFO_R": (4, 4),
                         "ILF_L": (3, 0), "ILF_R": (3, 4),
                         "SLF_L": (2, 1), "SLF_R": (2, 3),
                         "ARC_L": (2, 0), "ARC_R": (2, 4),
                         "UNC_L": (0, 1), "UNC_R": (0, 3)})


def tract_loader(trk, affine):
    """Helper function """
    if isinstance(trk, str):
        trk = nib.streamlines.load(trk)
        tg = trk.tractogram
    else:
        # Assume these are streamlines (as list or Streamlines object):
        tg = nib.streamlines.Tractogram(trk)

    if affine is not None:
        tg = tg.apply_affine(np.linalg.inv(affine))

    return tg, tg.streamlines


def bundle_selector(bundle_dict, colors, b):
    """Helper function """
    b_name = str(b)
    if bundle_dict is None:
        # We'll choose a color from a rotating list:
        if isinstance(colors, list):
            color = colors[np.mod(len(colors), int(b))]
        else:
            color_list = colors.values()
            color = color_list[np.mod(len(colors), int(b))]
    else:
        # We have a mapping from UIDs to bundle names:
        for b_name_iter, b_iter in bundle_dict.items():
            if b_iter['uid'] == b:
                b_name = b_name_iter
                break
        color = colors[b_name]
    return color, b_name


def tract_generator(tg, streamlines, bundle, bundle_dict, colors):
    """Helper function """
    if colors is None:
        # Use the color dict provided
        colors = COLOR_DICT

    if list(tg.data_per_streamline.keys()) == []:
        # There are no bundles in here:
        yield list(streamlines), [0.5, 0.5, 0.5], "all_bundles"

    else:
        # There are bundles:
        if bundle is None:
            # No selection: visualize all of them:

            for b in np.unique(tg.data_per_streamline['bundle']):
                idx = np.where(tg.data_per_streamline['bundle'] == b)[0]
                these_sls = list(streamlines[idx])
                color, b_name = bundle_selector(bundle_dict, colors, b)
                yield these_sls, color, b_name

        else:
            # Select just one to visualize:
            if isinstance(bundle, str):
                # We need to find the UID:
                uid = bundle_dict[bundle]['uid']
            else:
                # It's already a UID:
                uid = bundle

            idx = np.where(tg.data_per_streamline['bundle'] == uid)[0]
            these_sls = list(streamlines[idx])
            color, b_name = bundle_selector(bundle_dict, colors, uid)
            yield these_sls, color, b_name


def gif_from_pngs(tdir, gif_fname, n_frames, png_fname="tgif", add_zeros=False):
    """Helper function """
    if add_zeros:
        fname_suffix10 = "00000"
        fname_suffix100 = "0000"
        fname_suffix1000 = "000"
    else:
        fname_suffix10 = ""
        fname_suffix100 = ""
        fname_suffix1000 = ""
    angles = []
    for i in range(n_frames):
        if i < 10:
            angle_fname = f"{png_fname}{fname_suffix10}{i}.png"
        elif i < 100:
            angle_fname = f"{png_fname}{fname_suffix100}{i}.png"
        else:
            angle_fname = f"{png_fname}{fname_suffix1000}{i}.png"
        angles.append(io.imread(op.join(tdir, angle_fname)))

    io.mimsave(gif_fname, angles)


def prepare_roi(roi, affine_or_mapping, static_img,
                roi_affine, static_affine, reg_template):
    """Helper function """
    if not isinstance(roi, np.ndarray):
        if isinstance(roi, str):
            roi = nib.load(roi).get_fdata()
        else:
            roi = roi.get_fdata()

    if affine_or_mapping is not None:
        if isinstance(affine_or_mapping, np.ndarray):
            # This is an affine:
            if (static_img is None or roi_affine is None
                    or static_affine is None):
                raise ValueError("If using an affine to transform an ROI, "
                                 "need to also specify all of the following",
                                 "inputs: `static_img`, `roi_affine`, ",
                                 "`static_affine`")
            roi = reg.resample(roi, static_img, roi_affine, static_affine)
        else:
            # Assume it is  a mapping:
            if (isinstance(affine_or_mapping, str)
                    or isinstance(affine_or_mapping, nib.Nifti1Image)):
                if reg_template is None or static_img is None:
                    raise ValueError(
                        "If using a mapping to transform an ROI, need to ",
                        "also specify all of the following inputs: ",
                        "`reg_template`, `static_img`")
                affine_or_mapping = reg.read_mapping(affine_or_mapping,
                                                     static_img,
                                                     reg_template)

            roi = auv.patch_up_roi(affine_or_mapping.transform_inverse(
                                   roi,
                                   interpolation='nearest')).astype(bool)
    return roi


def load_volume(volume):
    """Helper function """
    if isinstance(volume, str):
        return nib.load(volume).get_fdata()
    else:
        return volume
