from collections import OrderedDict
import os.path as op
import logging

import numpy as np
import imageio as io
from PIL import Image, ImageChops

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

import nibabel as nib
import dipy.tracking.streamlinespeed as dps
from dipy.align import resample

import AFQ.utils.volume as auv
import AFQ.registration as reg
import AFQ.utils.streamlines as aus

__all__ = ["Viz"]

viz_logger = logging.getLogger("AFQ")
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
    "Left Anterior Thalamic": tableau_20[0], "C_L": tableau_20[0],
    "Right Anterior Thalamic": tableau_20[1], "C_R": tableau_20[1],
    "Left Corticospinal": tableau_20[2],
    "Right Corticospinal": tableau_20[3],
    "Left Cingulum Cingulate": tableau_20[4], "MCP": tableau_20[4],
    "Right Cingulum Cingulate": tableau_20[5], "CCMid": tableau_20[5],
    "Forceps Minor": tableau_20[8], "CC_ForcepsMinor": tableau_20[8],
    "Forceps Major": tableau_20[9], "CC_ForcepsMajor": tableau_20[9],
    "Left Inferior Fronto-occipital": tableau_20[10],
    "IFOF_L": tableau_20[10],
    "Right Inferior Fronto-occipital": tableau_20[11],
    "IFOF_R": tableau_20[11],
    "Left Inferior Longitudinal": tableau_20[12], "F_L": tableau_20[12],
    "Right Inferior Longitudinal": tableau_20[13], "F_R": tableau_20[13],
    "Left Superior Longitudinal": tableau_20[14],
    "Right Superior Longitudinal": tableau_20[15],
    "Left Uncinate": tableau_20[16], "UF_L": tableau_20[16],
    "Right Uncinate": tableau_20[17], "UF_R": tableau_20[17],
    "Left Arcuate": tableau_20[18], "AF_L": tableau_20[18],
    "Right Arcuate": tableau_20[19], "AF_R": tableau_20[19],
    "median": tableau_20[6],
    # Paul Tol's palette for callosal bundles
    "Callosum Orbital": (0.2, 0.13, 0.53),
    "Callosum Anterior Frontal": (0.07, 0.47, 0.2),
    "Callosum Superior Frontal": (0.27, 0.67, 0.6),
    "Callosum Motor": (0.53, 0.8, 0.93),
    "Callosum Superior Parietal": (0.87, 0.8, 0.47),
    "Callosum Posterior Parietal": (0.8, 0.4, 0.47),
    "Callosum Occipital": (0.67, 0.27, 0.6),
    "Callosum Temporal": (0.53, 0.13, 0.33)})

POSITIONS = OrderedDict({
    "Left Anterior Thalamic": (1, 0), "Right Anterior Thalamic": (1, 4),
    "C_L": (1, 0), "C_R": (1, 4),
    "Left Corticospinal": (1, 1), "Right Corticospinal": (1, 3),
    "Left Cingulum Cingulate": (3, 1),
    "Right Cingulum Cingulate": (3, 3),
    "MCP": (3, 1), "CCMid": (3, 3),
    "Forceps Minor": (4, 2), "Forceps Major": (0, 2),
    "CC_ForcepsMinor": (4, 2), "CC_ForcepsMajor": (0, 2),
    "Left Inferior Fronto-occipital": (4, 1),
    "Right Inferior Fronto-occipital": (4, 3),
    "IFOF_L": (4, 1), "IFOF_R": (4, 3),
    "Left Inferior Longitudinal": (3, 0),
    "Right Inferior Longitudinal": (3, 4),
    "F_L": (3, 0), "F_R": (3, 4),
    "Left Superior Longitudinal": (2, 1),
    "Right Superior Longitudinal": (2, 3),
    "Left Arcuate": (2, 0), "Right Arcuate": (2, 4),
    "AF_L": (2, 0), "AF_R": (2, 4),
    "Left Uncinate": (0, 1), "Right Uncinate": (0, 3),
    "UF_L": (0, 1), "UF_R": (0, 3)})

CSV_MAT_2_PYTHON = \
    {'fa': 'dti_fa', 'md': 'dti_md'}

SCALE_MAT_2_PYTHON = \
    {'dti_md': 0.001}

SCALAR_REMOVE_MODEL = \
    {'dti_md': 'MD', 'dki_md': 'MD', 'dki_fa': 'FA', 'dti_fa': 'FA'}

RECO_FLIP = ["IFO_L", "IFO_R", "UNC_L", "ILF_L", "ILF_R"]

BEST_BUNDLE_ORIENTATIONS = {
    "Left Anterior Thalamic": ("Sagittal", "Left"),
    "Right Anterior Thalamic": ("Sagittal", "Right"),
    "Left Corticospinal": ("Sagittal", "Left"),
    "Right Corticospinal": ("Sagittal", "Right"),
    "Left Cingulum Cingulate": ("Sagittal", "Left"),
    "Right Cingulum Cingulate": ("Sagittal", "Right"),
    "Forceps Minor": ("Axial", "Top"),
    "Forceps Major": ("Axial", "Top"),
    "Left Inferior Fronto-Occipital": ("Sagittal", "Left"),
    "Right Inferior Fronto-Occipital": ("Sagittal", "Right"),
    "Left Inferior Longitudinal": ("Sagittal", "Left"),
    "Right Inferior Longitudinal": ("Sagittal", "Right"),
    "Left Superior Longitudinal": ("Axial", "Top"),
    "Right Superior Longitudinal": ("Axial", "Top"),
    "Left Uncinate": ("Axial", "Bottom"),
    "Right Uncinate": ("Axial", "Bottom"),
    "Left Arcuate": ("Sagittal", "Left"),
    "Right Arcuate": ("Sagittal", "Right"),
    "Left Vertical Occipital": ("Coronal", "Back"),
    "Right Vertical Occipital": ("Coronal", "Back"),
    "Left Posterior Arcuate": ("Coronal", "Back"),
    "Right Posterior Arcuate": ("Coronal", "Back")}


class PanelFigure():
    """
    Super useful class for organizing existing images
    into subplots using matplotlib
    """

    def __init__(self, num_rows, num_cols, width, height,
                 panel_label_kwargs={}):
        """
        Initialize PanelFigure.

        Parameters
        ----------
        num_rows : int
            Number of rows in figure
        num_cols : int
            Number of columns in figure
        width : int
            Width of figure in inches
        height : int
            Height of figure in inches
        panel_label_kwargs : dict
            Additional arguments for matplotlib's text method,
            which is used to add panel labels to each subplot
        """
        self.fig = plt.figure(figsize=(width, height))
        self.grid = plt.GridSpec(num_rows, num_cols, hspace=0, wspace=0)
        self.subplot_count = 0
        self.panel_label_kwargs = dict(
            fontfamily="Helvetica-Bold",
            fontsize="xx-large",
            color="white",
            fontweight='bold',
            verticalalignment="top",
            bbox=dict(
                facecolor='none',
                edgecolor='none'))
        self.panel_label_kwargs.update(panel_label_kwargs)

    def add_img(self, fname, x_coord, y_coord, reduct_count=1,
                subplot_label_pos=(0.1, 1.0), legend=None, legend_kwargs={},
                add_panel_label=True):
        """
        Add image from fname into figure as a panel.

        Parameters
        ----------
        fname : str
            path to image file to add to subplot
        x_coord : int or slice
            x coordinate(s) of subplot in matlotlib figure
        y_coord : int or slice
            y coordinate(s) of subplot in matlotlib figure
        reduct_count : int
            number of times to trim whitespace around image
            Default: 1
        subplot_label_pos : tuple of floats
            position of subplot label
            Default: (0.1, 1.0)
        legend : dict
            dictionary of legend items, where keys are labels
            and values are colors
            Default: None
        legend_kwargs : dict
            ADditional arguments for matplotlib's legend method
        add_panel_label : bool
            Whether or not to add a panel label to the subplot
            Default: True
        """
        ax = self.fig.add_subplot(self.grid[y_coord, x_coord])
        im1 = Image.open(fname)
        for _ in range(reduct_count):
            im1 = trim(im1)
        if legend is not None:
            patches = []
            for value, color in legend.items():
                patches.append(mpatches.Patch(
                    color=color,
                    label=value))
            ax.legend(handles=patches, borderaxespad=0., **legend_kwargs)
        if add_panel_label:
            trans = mtransforms.ScaledTranslation(
                10 / 72, -5 / 72, self.fig.dpi_scale_trans)
            ax.text(
                subplot_label_pos[0], subplot_label_pos[1],
                f"{chr(65+self.subplot_count)}",
                transform=ax.transAxes + trans,
                **self.panel_label_kwargs)
        ax.imshow(np.asarray(im1), aspect=1)
        ax.axis('off')
        self.subplot_count = self.subplot_count + 1
        return ax

    def format_and_save_figure(self, fname, trim_final=True):
        """
        Format and save figure to fname.
        Parameters
        ----------
        fname : str
            Path to save figure to
        trim : bool
            Whether or not to trim whitespace around figure.
            Default: True
        """
        self.fig.tight_layout()
        self.fig.savefig(fname, dpi=300)
        if trim_final:
            im1 = Image.open(fname)
            im1 = trim(im1)
            im1.save(fname, dpi=(300, 300))


def get_eye(view, direc):
    direc = direc.lower()
    view = view.lower()

    if view in ["sagital", "saggital"]:
        viz_logger.warning("You don't know how to spell sagggitttal!")
        view = "sagittal"

    if view not in ["sagittal", "coronal", "axial"]:
        raise ValueError(
            "View must be one of: sagittal, coronal, or axial")

    if direc not in ["left", "right", "top", "bottom", "front", "back"]:
        raise ValueError(
            "View must be one of: left, right, top, bottom, front, back")

    eye = {}
    if view == "sagittal":
        if direc == "left":
            eye["x"] = -1
        else:
            eye["x"] = 1
        eye["y"] = 0
        eye["z"] = 0
    elif view == "coronal":
        eye["x"] = 0
        if direc == "front":
            eye["y"] = 1
        else:
            eye["y"] = -1
        eye["z"] = 0
    elif view == "axial":
        eye["x"] = 0
        eye["y"] = 0
        if direc == "top":
            eye["z"] = 1
        else:
            eye["z"] = -1
    return eye


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
                other_bundle = bundle
                if bundle.startswith("Left "):
                    other_bundle = "Right" + other_bundle[5:]
                elif bundle.startswith("Right "):
                    other_bundle = "Left" + other_bundle[4:]
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


def tract_generator(trk_file, bundle, colors, n_points,
                    n_sls_viz=3600, n_sls_min=75):
    """
    Generates bundles of streamlines from the tractogram.
    Only generates from relevant bundle if bundle is set.
    Otherwise, returns all streamlines.

    Helper function

    Parameters
    ----------
    trk_file : str or SegmentedSFT
        Path to a trk file or SegmentedSFT

    bundle : str
        The name of a bundle to select from the trk metadata.

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
    viz_logger.info("Loading Stateful Tractogram...")
    if isinstance(trk_file, str):
        seg_sft = aus.SegmentedSFT.fromfile(trk_file)
    else:
        seg_sft = trk_file

    if colors is None:
        colors = gen_color_dict(seg_sft.bundle_names)

    seg_sft.sft.to_vox()
    streamlines = seg_sft.sft.streamlines
    viz_logger.info("Generating colorful lines from tractography...")

    if len(seg_sft.bundle_names) == 1\
            and seg_sft.bundle_names[0] == "whole_brain":
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
        yield streamlines, colors[0], "all_bundles", seg_sft.sft.dimensions
    else:
        if bundle is None:
            # No selection: visualize all of them:
            for bundle_name in seg_sft.bundle_names:
                idx = seg_sft.bundle_idxs[bundle_name]
                if len(idx) == 0:
                    continue
                n_sl_viz = (len(idx) * n_sls_viz) //\
                    len(streamlines)
                n_sl_viz = max(n_sls_min, n_sl_viz)
                if len(idx) > n_sl_viz:
                    idx = np.random.choice(idx, size=n_sl_viz, replace=False)
                these_sls = streamlines[idx]
                if n_points is not None:
                    these_sls = dps.set_number_of_points(these_sls, n_points)
                if isinstance(colors, dict):
                    color = colors[bundle_name]
                else:
                    color = colors[0]
                yield these_sls, color, bundle_name, seg_sft.sft.dimensions
        else:
            these_sls = seg_sft.get_bundle(bundle).streamlines
            if n_points is not None:
                these_sls = dps.set_number_of_points(these_sls, n_points)
            if isinstance(colors, dict):
                color = colors[bundle]
            else:
                color = colors[0]
            yield these_sls, color, bundle, seg_sft.sft.dimensions


def bbox(img):
    img = np.sum(img, axis=-1)
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return cmin, rmin, cmax, rmax


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    diff.getbbox
    this_bbox = bbox(diff)
    if this_bbox:
        return im.crop(this_bbox)


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
