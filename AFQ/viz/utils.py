from collections import OrderedDict
import os.path as op
import logging

import numpy as np
import imageio as io

import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import transform_tracking_output
import dipy.tracking.streamlinespeed as dps
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.align import resample

import AFQ.utils.volume as auv
import AFQ.registration as reg

__all__ = ["Viz"]

viz_logger = logging.getLogger("AFQ.viz")
tableau_20 = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
    (0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
    (1.0, 0.4980392156862745, 0.054901960784313725),
    (1.0, 0.7333333333333333, 0.47058823529411764),
    (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    (0.596078431372549, 0.8745098039215686, 0.5411764705882353),
    (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
    (1.0, 0.596078431372549, 0.5882352941176471),
    (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
    (0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    (0.7686274509803922, 0.611764705882353, 0.5803921568627451),
    (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
    (0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
    (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
    (0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
    (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
    (0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
    (0.6196078431372549, 0.8549019607843137, 0.8980392156862745)]
large_font = 28
medium_font = 24
small_font = 20
marker_size = 200

COLOR_DICT = OrderedDict({
    "ATR_L": tableau_20[0], "C_L": tableau_20[0],
    "ATR_R": tableau_20[1], "C_R": tableau_20[1],
    "CST_L": tableau_20[2],
    "CST_R": tableau_20[3],
    "CGC_L": tableau_20[4], "MCP": tableau_20[4],
    "CGC_R": tableau_20[5], "CCMid": tableau_20[5],
    "HCC_L": tableau_20[6],
    "HCC_R": tableau_20[7],
    "FP": tableau_20[8], "CC_ForcepsMinor": tableau_20[8],
    "FA": tableau_20[9], "CC_ForcepsMajor": tableau_20[9],
    "IFO_L": tableau_20[10], "IFOF_L": tableau_20[10],
    "IFO_R": tableau_20[11], "IFOF_R": tableau_20[11],
    "ILF_L": tableau_20[12], "F_L": tableau_20[12],
    "ILF_R": tableau_20[13], "F_R": tableau_20[13],
    "SLF_L": tableau_20[14],
    "SLF_R": tableau_20[15],
    "UNC_L": tableau_20[16], "UF_L": tableau_20[16],
    "UNC_R": tableau_20[17], "UF_R": tableau_20[17],
    "ARC_L": tableau_20[18], "AF_L": tableau_20[18],
    "ARC_R": tableau_20[19], "AF_R": tableau_20[19],
    "median": tableau_20[6]})

POSITIONS = OrderedDict({
    "ATR_L": (1, 0), "ATR_R": (1, 4), "C_L": (1, 0), "C_R": (1, 4),
    "CST_L": (1, 1), "CST_R": (1, 3),
    "CGC_L": (3, 1), "CGC_R": (3, 3), "MCP": (3, 1), "CCMid": (3, 3),
    "FP": (4, 2), "FA": (0, 2),
    "CC_ForcepsMinor": (4, 2), "CC_ForcepsMajor": (0, 2),
    "IFO_L": (4, 1), "IFO_R": (4, 3), "IFOF_L": (4, 1), "IFOF_R": (4, 3),
    "HCC_L": (4, 0), "HCC_R": (4, 4),
    "ILF_L": (3, 0), "ILF_R": (3, 4), "F_L": (3, 0), "F_R": (3, 4),
    "SLF_L": (2, 1), "SLF_R": (2, 3),
    "ARC_L": (2, 0), "ARC_R": (2, 4), "AF_L": (2, 0), "AF_R": (2, 4),
    "UNC_L": (0, 1), "UNC_R": (0, 3), "UF_L": (0, 1), "UF_R": (0, 3)})

CSV_MAT_2_PYTHON = \
    {'fa': 'dti_fa', 'md': 'dti_md'}

SCALE_MAT_2_PYTHON = \
    {'dti_md': 0.001}

SCALAR_REMOVE_MODEL = \
    {'dti_md': 'MD', 'dki_md': 'MD', 'dki_fa': 'FA', 'dti_fa': 'FA'}

RECO_FLIP = ["IFO_L", "IFO_R", "UNC_L", "ILF_L", "ILF_R"]


def display_string(scalar_name):
    if isinstance(scalar_name, str):
        return scalar_name.replace("_", " ").upper()
    else:
        return [sn.replace("_", " ").upper() for sn in scalar_name]


def gen_color_dict(bundles):
    """
    Helper function.
    Generate a color dict given a list of bundles.
    """
    def incr_color_idx(color_idx):
        return (color_idx + 1) % 20
    custom_color_dict = {}
    color_idx = 0
    for bundle in bundles:
        if bundle not in custom_color_dict.keys():
            if bundle in COLOR_DICT.keys():
                custom_color_dict[bundle] = COLOR_DICT[bundle]
            else:
                other_bundle = list(bundle)
                if bundle[-2:] == '_L':
                    other_bundle[-2:] = '_R'
                elif bundle[-2:] == '_R':
                    other_bundle[-2:] = '_L'
                other_bundle = str(other_bundle)

                if other_bundle == bundle:  # lone bundle
                    custom_color_dict[bundle] = tableau_20[color_idx]
                    color_idx = incr_color_idx(color_idx)
                else:  # right left pair
                    if color_idx % 2 != 0:
                        color_idx = incr_color_idx(color_idx)
                    custom_color_dict[bundle] =\
                        tableau_20[color_idx]
                    custom_color_dict[other_bundle] =\
                        tableau_20[color_idx + 1]
                    color_idx = incr_color_idx(incr_color_idx(color_idx))
    return custom_color_dict


def viz_import_msg_error(module):
    """Alerts user to install the appropriate viz module """
    if module == "plot":
        msg = "To make plots in pyAFQ, you will need to install "
        msg += "the relevant plotting software packages."
        msg += "You can do that by installing pyAFQ with "
        msg += "`pip install AFQ[plot]`"
    else:
        msg = f"To use {module.upper()} visualizations in pyAFQ, you will "
        msg += f"need to have {module.upper()} installed. "
        msg += "You can do that by installing pyAFQ with "
        msg += f"`pip install AFQ[{module.lower()}]`, or by "
        msg += f"separately installing {module.upper()}: "
        msg += f"`pip install {module.lower()}`."
    return msg


def bundle_selector(bundle_dict, colors, b):
    """
    Selects bundle and color
    from the given bundle dictionary and color informaiton
    Helper function

    Parameters
    ----------
    bundle_dict : dict, optional
        Keys are names of bundles and values are dicts that should include
        a key `'uid'` with values as integers for selection from the trk
        metadata. Default: bundles are either not identified, or identified
        only as unique integers in the metadata.

    bundle : str or int, optional
        The name of a bundle to select from among the keys in `bundle_dict`
        or an integer for selection from the trk metadata.

    colors : dict or list
        If this is a dict, keys are bundle names and values are RGB tuples.
        If this is a list, each item is an RGB tuple. Defaults to a list
        with Tableau 20 RGB values

    Returns
    -------
    RGB tuple and str
    """
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
        b_found = False
        for b_name_iter, b_iter in bundle_dict.items():
            if b_iter['uid'] == b:
                b_name = b_name_iter
                b_found = True
                break

        # ignore bundle if it is not in the bundle_dict
        if b_found is False:
            return None, None
        color = colors[b_name]
    return color, b_name


def tract_generator(sft, affine, bundle, bundle_dict, colors, n_points,
                    n_sls_viz=3600, n_sls_min=75):
    """
    Generates bundles of streamlines from the tractogram.
    Only generates from relevant bundle if bundle is set.
    Uses bundle_dict and colors to assign colors if set.
    Otherwise, returns all streamlines.

    Helper function

    Parameters
    ----------
    sft : Stateful Tractogram, str
        A Stateful Tractogram containing streamline information
        or a path to a trk file

    affine : ndarray
       An affine transformation to apply to the streamlines.

    bundle : str or int
        The name of a bundle to select from among the keys in `bundle_dict`
        or an integer for selection from the trk metadata.

    bundle_dict : dict, optional
        Keys are names of bundles and values are dicts that should include
        a key `'uid'` with values as integers for selection from the sft
        metadata. Default: bundles are either not identified, or identified
        only as unique integers in the metadata.

    colors : dict or list
        If this is a dict, keys are bundle names and values are RGB tuples.
        If this is a list, each item is an RGB tuple. Defaults to a list
        with Tableau 20 RGB values

    n_points : int or None
        n_points to resample streamlines to before plotting. If None, no
        resampling is done.

    n_sls_viz : int
        Number of streamlines to randomly select if plotting
        all bundles. Selections will be proportional to the original number of
        streamlines per bundle.
        Default: 3600
    n_sls_min : int
        Minimun number of streamlines to display per bundle.
        Default: 75

    Returns
    -------
    Statefule Tractogram streamlines, RGB numpy array, str
    """
    if colors is None:
        if bundle_dict is None:
            colors = tableau_20
        else:
            colors = gen_color_dict(bundle_dict.keys())

    if isinstance(sft, str):
        viz_logger.info("Loading Stateful Tractogram...")
        sft = load_tractogram(sft, 'same', Space.VOX, bbox_valid_check=False)

    if affine is not None:
        viz_logger.info("Transforming Stateful Tractogram...")
        sft = StatefulTractogram.from_sft(
            transform_tracking_output(sft.streamlines, affine),
            sft,
            data_per_streamline=sft.data_per_streamline)

    streamlines = sft.streamlines
    viz_logger.info("Generating colorful lines from tractography...")

    if list(sft.data_per_streamline.keys()) == []:
        if isinstance(colors, dict):
            colors = list(colors.values())
        # There are no bundles in here:
        if len(streamlines) > n_sls_viz:
            idx = np.arange(len(streamlines))
            idx = np.random.choice(
                idx, size=n_sls_viz, replace=False)
            streamlines = streamlines[idx]
        if n_points is not None:
            streamlines = dps.set_number_of_points(streamlines, n_points)
        yield streamlines, colors[0], "all_bundles", sft.dimensions

    else:
        # There are bundles:
        if bundle_dict is not None:
            bundle_dict = bundle_dict.copy()
            bundle_dict.pop('whole_brain', None)

        if bundle is None:
            # No selection: visualize all of them:

            for b in np.unique(sft.data_per_streamline['bundle']):
                idx = np.where(sft.data_per_streamline['bundle'] == b)[0]
                n_sl_viz = (len(idx) * n_sls_viz) //\
                    len(sft.streamlines)
                n_sl_viz = max(n_sls_min, n_sl_viz)
                if len(idx) > n_sl_viz:
                    idx = np.random.choice(idx, size=n_sl_viz, replace=False)
                these_sls = streamlines[idx]
                if n_points is not None:
                    these_sls = dps.set_number_of_points(these_sls, n_points)
                color, b_name = bundle_selector(bundle_dict, colors, b)
                if color is None:
                    continue
                yield these_sls, color, b_name, sft.dimensions

        else:
            # Select just one to visualize:
            if isinstance(bundle, str):
                # We need to find the UID:
                uid = bundle_dict[bundle]['uid']
            else:
                # It's already a UID:
                uid = bundle

            idx = np.where(sft.data_per_streamline['bundle'] == uid)[0]
            these_sls = streamlines[idx]
            if n_points is not None:
                these_sls = dps.set_number_of_points(these_sls, n_points)
            color, b_name = bundle_selector(bundle_dict, colors, uid)
            yield these_sls, color, b_name, sft.dimensions


def gif_from_pngs(tdir, gif_fname, n_frames,
                  png_fname="tgif", add_zeros=False):
    """
        Helper function
        Stitches together gif from screenshots
    """
    if add_zeros:
        fname_suffix10 = "00000"
        fname_suffix100 = "0000"
        fname_suffix1000 = "000"
    else:
        fname_suffix10 = ""
        fname_suffix100 = ""
        fname_suffix1000 = ""
    angles = []
    n_frame_copies = 60 // n_frames
    for i in range(n_frames):
        if i < 10:
            angle_fname = f"{png_fname}{fname_suffix10}{i}.png"
        elif i < 100:
            angle_fname = f"{png_fname}{fname_suffix100}{i}.png"
        else:
            angle_fname = f"{png_fname}{fname_suffix1000}{i}.png"
        frame = io.imread(op.join(tdir, angle_fname))
        for j in range(n_frame_copies):
            angles.append(frame)

    io.mimsave(gif_fname, angles)


def prepare_roi(roi, affine_or_mapping, static_img,
                roi_affine, static_affine, reg_template):
    """
    Load the ROI
    Possibly perform a transformation on an ROI
    Helper function

    Parameters
    ----------
    roi : str or Nifti1Image
        The ROI information.
        If str, ROI will be loaded using the str as a path.

    affine_or_mapping : ndarray, Nifti1Image, or str
       An affine transformation or mapping to apply to the ROI before
       visualization. Default: no transform.

    static_img: str or Nifti1Image
        Template to resample roi to.

    roi_affine: ndarray

    static_affine: ndarray

    reg_template: str or Nifti1Image
        Template to use for registration.

    Returns
    -------
    ndarray
    """
    viz_logger.info("Preparing ROI...")
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
            roi = resample(roi, static_img, roi_affine,
                           static_affine).get_fdata()
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

            roi = auv.transform_inverse_roi(
                roi,
                affine_or_mapping).astype(bool)
    return roi


def load_volume(volume):
    """
    Load a volume
    Helper function

    Parameters
    ----------
    volume : ndarray or str
        3d volume to load.
        If string, it is used as a file path.
        If it is an ndarray, it is simply returned.

    Returns
    -------
    ndarray
    """
    viz_logger.info("Loading Volume...")
    if isinstance(volume, str):
        return nib.load(volume).get_fdata()
    else:
        return volume


class Viz:
    def __init__(self,
                 backend="fury"):
        """
        Set up visualization preferences.

        Parameters
        ----------
            backend : str, optional
                Should be either "fury" or "plotly".
                Default: "fury"
        """
        self.backend = backend
        if "fury" in backend:
            try:
                import AFQ.viz.fury_backend
            except ImportError:
                raise ImportError(viz_import_msg_error("fury"))
            self.visualize_bundles = AFQ.viz.fury_backend.visualize_bundles
            self.visualize_roi = AFQ.viz.fury_backend.visualize_roi
            self.visualize_volume = AFQ.viz.fury_backend.visualize_volume
            self.create_gif = AFQ.viz.fury_backend.create_gif
        elif "plotly" in backend:
            try:
                import AFQ.viz.plotly_backend
            except ImportError:
                raise ImportError(viz_import_msg_error("plotly"))
            self.visualize_bundles = AFQ.viz.plotly_backend.visualize_bundles
            self.visualize_roi = AFQ.viz.plotly_backend.visualize_roi
            self.visualize_volume = AFQ.viz.plotly_backend.visualize_volume
            self.create_gif = AFQ.viz.plotly_backend.create_gif
            self.single_bundle_viz = AFQ.viz.plotly_backend.single_bundle_viz
        else:
            raise TypeError("Visualization backend contain"
                            + " either 'plotly' or 'fury'. "
                            + "It is currently set to %s"
                            % backend)
