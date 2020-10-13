from collections import OrderedDict
import os.path as op
import os
import logging
import tempfile

import numpy as np
from scipy.stats import sem
import pandas as pd
import imageio as io
import IPython.display as display
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm

import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import transform_tracking_output
import dipy.tracking.streamlinespeed as dps
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.utils.volume as auv
import AFQ.registration as reg
from AFQ.utils.stats import contrast_index as calc_contrast_index
from AFQ.data import BUNDLE_RECO_2_AFQ, BUNDLE_MAT_2_PYTHON

__all__ = ["Viz", "visualize_tract_profiles", "visualize_gif_inline"]

viz_logger = logging.getLogger("AFQ.viz")
tableau_20_sns = sns.color_palette("tab20")
large_font = 24
medium_font = 20
small_font = 16

COLOR_DICT = OrderedDict({
    "ATR_L": tableau_20_sns[0], "C_L": tableau_20_sns[0],
    "ATR_R": tableau_20_sns[1], "C_R": tableau_20_sns[1],
    "CST_L": tableau_20_sns[2],
    "CST_R": tableau_20_sns[3],
    "CGC_L": tableau_20_sns[4], "MCP": tableau_20_sns[4],
    "CGC_R": tableau_20_sns[5], "CCMid": tableau_20_sns[5],
    "HCC_L": tableau_20_sns[6],
    "HCC_R": tableau_20_sns[7],
    "FP": tableau_20_sns[8], "CC_ForcepsMinor": tableau_20_sns[8],
    "FA": tableau_20_sns[9], "CC_ForcepsMajor": tableau_20_sns[9],
    "IFO_L": tableau_20_sns[10], "IFOF_L": tableau_20_sns[10],
    "IFO_R": tableau_20_sns[11], "IFOF_R": tableau_20_sns[11],
    "ILF_L": tableau_20_sns[12], "F_L": tableau_20_sns[12],
    "ILF_R": tableau_20_sns[13], "F_R": tableau_20_sns[13],
    "SLF_L": tableau_20_sns[14],
    "SLF_R": tableau_20_sns[15],
    "UNC_L": tableau_20_sns[16], "UF_L": tableau_20_sns[16],
    "UNC_R": tableau_20_sns[17], "UF_R": tableau_20_sns[17],
    "ARC_L": tableau_20_sns[18], "AF_L": tableau_20_sns[18],
    "ARC_R": tableau_20_sns[19], "AF_R": tableau_20_sns[19]})

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
                    custom_color_dict[bundle] = tableau_20_sns[color_idx]
                    color_idx = incr_color_idx(color_idx)
                else:  # right left pair
                    if color_idx % 2 != 0:
                        color_idx = incr_color_idx(color_idx)
                    custom_color_dict[bundle] =\
                        tableau_20_sns[color_idx]
                    custom_color_dict[other_bundle] =\
                        tableau_20_sns[color_idx + 1]
                    color_idx = incr_color_idx(incr_color_idx(color_idx))
    return custom_color_dict


def viz_import_msg_error(module):
    """Alerts user to install the appropriate viz module """
    msg = f"To use {module.upper()} visualizations in pyAFQ, you will need "
    msg += f"to have {module.upper()} installed. "
    msg += f"You can do that by installing pyAFQ with "
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
        for b_name_iter, b_iter in bundle_dict.items():
            if b_iter['uid'] == b:
                b_name = b_name_iter
                break
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
            colors = tableau_20_sns
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
        # There are no bundles in here:
        if n_points is not None:
            streamlines = dps.set_number_of_points(streamlines, n_points)
        yield streamlines, colors[0], "all_bundles"

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
                yield these_sls, color, b_name

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
            yield these_sls, color, b_name


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
        else:
            raise TypeError("Visualization backend contain"
                            + " either 'plotly' or 'fury'. "
                            + "It is currently set to %s"
                            % backend)


def visualize_tract_profiles(tract_profiles, scalar="dti_fa", ylim=None,
                             n_boot=1000,
                             file_name=None,
                             positions=POSITIONS):
    """
    Visualize all tract profiles for a scalar in one plot

    Parameters
    ----------
    tract_profiles : string
        Path to CSV containing tract_profiles.

    scalar : string, optional
       Scalar to use in plots. Default: "dti_fa".

    ylim : list of 2 floats, optional
        Minimum and maximum value used for y-axis bounds.
        If None, ylim is not set.
        Default: None

    n_boot : int, optional
        Number of bootstrap resamples for seaborn to use
        to estimate the ci.
        Default: 1000

    file_name : string, optional
        File name to save figure to if not None. Default: None

    positions : dictionary, optional
        Dictionary that maps bundle names to position in plot.
        Default: POSITIONS

    Returns
    -------
        Matplotlib figure and axes.
    """
    csv_comparison = GroupCSVComparison(
        None,
        [tract_profiles],
        ["my_tract_profiles"],
        scalar_bounds={'lb': {}, 'ub': {}})

    df = csv_comparison.tract_profiles(
        scalar=scalar,
        ylim=ylim,
        positions=positions,
        out_file=file_name,
        n_boot=n_boot)

    return df


class BrainAxes():
    '''
    Helper class.
    Creates and handles a grid of axes.
    Each axis corresponds to a bundle.
    Axis placement should roughly correspond
    to the actual bundle placement in the brain.
    '''

    def __init__(self, size=(5, 5), positions=POSITIONS):
        self.size = size
        self.positions = positions
        self.on_grid = np.zeros((5, 5), dtype=bool)
        self.fig, self.axes = plt.subplots(
            self.size[0],
            self.size[1],
            frameon=False)
        plt.subplots_adjust(
            left=None,
            bottom=None,
            right=None,
            top=None,
            wspace=0.4,
            hspace=0.6)
        self.fig.set_size_inches((18, 18))
        self.temp_fig, self.temp_axis = plt.subplots()
        self.temp_axis_owner = None

    def get_axis(self, bundle):
        '''
        Given a bundle, turn on and get an axis.
        If bundle not positioned, will claim the temporary axis.
        '''
        if bundle in self.positions.keys():
            self.on_grid[self.positions[bundle]] = True
            return self.axes[self.positions[bundle]]
        else:
            if self.temp_axis_owner != bundle:
                plt.close(self.temp_fig)
                self.temp_fig, self.temp_axis = plt.subplots()
                self.temp_axis_owner = bundle
            return self.temp_axis

    def plot_line(self, bundle, x, y, data, ylabel, ylim, n_boot, alpha,
                  lineplot_kwargs, plot_subject_lines=True):
        '''
        Given a dataframe data with at least columns x, y,
        and subjectID, plot the mean of subjects with ci of 95
        in alpha and the individual subjects in alpha-0.2
        using sns.lineplot()
        '''
        ax = self.get_axis(bundle)
        if plot_subject_lines:
            sns.set(style="whitegrid", rc={"lines.linewidth": 1})
        else:
            sns.set(style="whitegrid", rc={"lines.linewidth": 6})
        sns.lineplot(
            x=x, y=y,
            data=data,
            estimator='mean', ci=95, n_boot=n_boot,
            legend=False, ax=ax, alpha=alpha,
            style=[True] * len(data.index), **lineplot_kwargs)
        if plot_subject_lines:
            sns.set(style="whitegrid", rc={"lines.linewidth": 0.5})
            sns.lineplot(
                x=x, y=y,
                data=data,
                ci=None, estimator=None, units='subjectID',
                legend=False, ax=ax, alpha=alpha - 0.2,
                style=[True] * len(data.index), **lineplot_kwargs)

        ax.set_title(bundle, fontsize=large_font)
        ax.set_ylabel(ylabel, fontsize=medium_font)
        ax.set_ylim(ylim)

    def format(self, disable_x=True):
        '''
        Call this functions once after all axes that you intend to use
        have been plotted on. Automatically formats brain axes.
        '''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if not self.on_grid[i, j]:
                    self.axes[i, j].axis("off")
                if j != 0 and self.on_grid[i][j - 1]:
                    self.axes[i, j].set_yticklabels([])
                    self.axes[i, j].set_ylabel("")
                if disable_x or (i != 4 and self.on_grid[i + 1][j]):
                    self.axes[i, j].set_xticklabels([])
                    self.axes[i, j].set_xlabel("")
        self.fig.tight_layout()

    def save_temp_fig(self, fname):
        '''
        If using a temporary axis, save it out and clear it.
        Returns True if temporary axis was saved, false if no
        temporary axis was in use
        '''
        if self.temp_axis_owner is None:
            return False
        self.temp_fig.tight_layout()
        self.temp_fig.savefig(fname)
        plt.close(self.temp_fig)
        self.temp_axis_owner = None
        return True

    def is_using_temp_axis(self):
        return (self.temp_axis_owner is not None)

    def close_all(self):
        '''
        Close all associated figures.
        '''
        plt.close(self.temp_fig)
        plt.close(self.fig)


class GroupCSVComparison():
    """
    Compare different CSVs, using:
    tract profiles, contrast indices,
    scan-rescan reliability using Pearson's r.
    """

    def __init__(self, out_folder, csv_fnames, names, is_mats=False,
                 subjects=None,
                 scalar_bounds={'lb': {'dti_fa': 0.2},
                                'ub': {'dti_md': 0.002}},
                 bundles=None,
                 percent_nan_tol=10,
                 percent_edges_removed=10,
                 mat_bundle_converter=BUNDLE_MAT_2_PYTHON,
                 mat_column_converter=CSV_MAT_2_PYTHON,
                 mat_scale_converter=SCALE_MAT_2_PYTHON):
        """
        Load in csv files, converting from matlab if necessary.

        Parameters
        ----------
        out_folder : path, optional
            Folder where outputs of this class's methods will be saved.

        csv_fnames : list of filenames
            Filenames for the two CSVs containing tract profiles to compare.
            Will obtain subject list from the first file.

        names : list of strings
            Name to use to identify each CSV dataset.

        is_mats : bool or list of bools, optional
            Whether or not the csv was generated from Matlab AFQ or pyAFQ.
            Default: False

        subjects : list of num, optional
            List of subjects to consider.
            If None, will use all subjects in first dataset.
            Default: None

        scalar_bounds : dictionary, optional
            A dictionary with a lower bound and upper bound containting a
            series of scalar / threshold pairs used as a white-matter mask
            on the profiles (any values outside of the threshold will be
            marked NaN and not used or set to 0, depending on the case).
            Default: {'lb': {'dti_fa': 0.2}, 'ub': {'dti_md': 0.002}}

        bundles : list of strings, optional
            Bundles to compare.
            If None, use all bundles in the first profile group.
            Default: None

        percent_nan_tol : int, optional
            Percentage of NaNs tolerable. If a profile has less than this
            percentage of NaNs, NaNs are interpolated. If it has more,
            the profile is thrown out.
            Default: 10

        percent_edges_removed : int, optional
            Percentage of nodes to remove from the edges of the profile.
            Scalar values often change dramatically at the boundary between
            the grey matter and the white matter, and these effects can
            dominate plots. However, they are generally not interesting to us,
            and have low intersubject reliability.
            In a profile of 100 nodes, percent_edges_removed=10 would remove
            5 nodes from each edge.
            Default: 10


        mat_bundle_converter : dictionary, optional
            Dictionary that maps matlab bundle names to python bundle names.
            Default: BUNDLE_MAT_2_PYTHON

        mat_column_converter : dictionary, optional
            Dictionary that maps matlab column names to python column names.
            Default: CSV_MAT_2_PYTHON

        mat_scale_converter : dictionary, optional
            Dictionary that maps scalar names to how they should be scaled
            to match pyAFQ's scale for that scalar.
            Default: SCALE_MAT_2_PYTHON
        """
        self.logger = logging.getLogger('AFQ.csv')
        self.out_folder = out_folder
        self.percent_nan_tol = percent_nan_tol

        if isinstance(is_mats, bool):
            is_mats = [is_mats] * len(csv_fnames)

        self.profile_dict = {}
        for i, fname in enumerate(csv_fnames):
            profile = pd.read_csv(fname)
            if 'subjectID' in profile.columns:
                profile['subjectID'] = \
                    profile['subjectID'].apply(
                        lambda x: int(
                            ''.join(c for c in x if c.isdigit())
                        ) if isinstance(x, str) else x)
            else:
                profile['subjectID'] = 0

            if is_mats[i]:
                profile.rename(
                    columns=mat_column_converter, inplace=True)
                profile['tractID'] = \
                    profile['tractID'].apply(
                        lambda x: mat_bundle_converter[x])
                for scalar, scale in mat_scale_converter.items():
                    profile[scalar] = \
                        profile[scalar].apply(lambda x: x * scale)

            for bound, constraint in scalar_bounds.items():
                for scalar, threshold in constraint.items():
                    profile[scalar] = \
                        profile[scalar].apply(
                            lambda x: self._threshold_scalar(
                                bound,
                                threshold,
                                x))

            if percent_edges_removed > 0:
                profile = profile.drop(profile[np.logical_or(
                    (profile["nodeID"] < percent_nan_tol // 2),
                    (profile["nodeID"] >= 100 - (percent_nan_tol // 2))
                )].index)

            self.profile_dict[names[i]] = profile
        if subjects is None:
            self.subjects = self.profile_dict[names[0]]['subjectID'].unique()
        else:
            self.subjects = subjects
        self.prof_len = 100 - (percent_nan_tol // 2) * 2
        if bundles is None:
            self.bundles = self.profile_dict[names[0]]['tractID'].unique()
            self.bundles.sort()
        else:
            self.bundles = bundles
        self.color_dict = gen_color_dict(self.bundles)

    def _threshold_scalar(self, bound, threshold, val):
        """
        Threshold scalars by a lower and upper bound.
        """
        if bound == "lb":
            if val > threshold:
                return val
            else:
                return np.nan
        elif bound == "ub":
            if val < threshold:
                return val
            else:
                return np.nan
        else:
            raise RuntimeError("scalar_bounds dictionary "
                               + " formatted incorrectly. See"
                               + " the default for reference")

    def _get_fname(self, folder, f_name):
        """
        Get file to save to, and generate the folder if it does not exist.
        """
        f_folder = op.join(
            self.out_folder,
            folder)
        os.makedirs(f_folder, exist_ok=True)
        return op.join(f_folder, f_name)

    def _get_profile(self, name, bundle, subject, scalar, repl_nan=True):
        """
        Get a single profile, then handle not found / NaNs
        """
        profile = self.profile_dict[name]
        single_profile = profile[
            (profile['subjectID'] == subject)
            & (profile['tractID'] == bundle)
        ][scalar].to_numpy()
        nans = np.isnan(single_profile)
        percent_nan = (np.sum(nans) * 100) // self.prof_len
        if len(single_profile) < 1:
            self.logger.warning(
                'No scalars found for scalar ' + scalar
                + ' for subject ' + str(subject)
                + ' for bundle ' + bundle
                + ' for CSV ' + name)
            return None

        if percent_nan > 0:
            message = (
                f'{percent_nan}% NaNs found in scalar ' + scalar
                + ' for subject ' + str(subject)
                + ' for bundle ' + bundle
                + ' for CSV ' + name)
            if np.sum(nans) > self.percent_nan_tol:
                self.logger.warn(message + '. Profile ignored. ')
                return None
            else:
                self.logger.info(message + '. NaNs interpolated. ')
                non_nan = np.logical_not(nans)
                single_profile[nans] = np.interp(
                    nans.nonzero()[0],
                    non_nan.nonzero()[0],
                    single_profile[non_nan])

        return single_profile

    def _alpha(self, alpha):
        '''
        Keep alpha in a reasonable range
        Useful when calculating alpha automatically
        '''
        if alpha < 0.3:
            return 0.3
        if alpha > 1:
            return 1
        return alpha

    def _array_to_df(self, arr):
        '''
        Converts a 2xn array to a pandas dataframe with columns x, y
        Useful for plotting with seaborn.
        '''
        df = pd.DataFrame()
        df['x'] = arr[0]
        df['y'] = arr[1]
        return df

    def masked_corr(self, arr):
        '''
        Mask arr for NaNs before calling np.corrcoef
        '''
        mask = np.logical_not(
            np.logical_or(
                np.isnan(arr[0, ...]),
                np.isnan(arr[1, ...])))
        if np.sum(mask) < 2:
            return 0
        return np.corrcoef(arr[:, mask])[0][1]

    def tract_profiles(self, names=None, scalar="dti_fa",
                       ylim=[0.0, 1.0],
                       show_plots=False,
                       positions=POSITIONS,
                       out_file=None,
                       n_boot=1000,
                       plot_subject_lines=True):
        """
        Compare all tract profiles for a scalar from different CSVs.
        Plots tract profiles for all in one plot.
        Bundles taken from positions argument.

        Parameters
        ----------
        names : list of strings, optional
            Names of datasets to plot profiles of.
            If None, all datasets are used.
            Default: None

        scalar : string, optional
            Scalar to use in plots. Default: "dti_fa".

        ylim : list of 2 floats, optional
            Minimum and maximum value used for y-axis bounds.
            If None, ylim is not set.
            Default: [0.0, 1.0]

        out_file : str, optional
            Path to save the figure to.
            If None, use the default naming convention in self.out_folder
            Default: None

        n_boot : int, optional
            Number of bootstrap resamples for seaborn to use
            to estimate the ci.
            Default: 1000

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        positions : dictionary, optional
            Dictionary that maps bundle names to position in plot.
            Default: POSITIONS

        plot_subject_lines : bool, optional
            Whether to plot individual subject lines with a smaller width.
            Default: True
        """
        if not show_plots:
            plt.ioff()
        if names is None:
            names = list(self.profile_dict.keys())
        if out_file is None:
            fname = self._get_fname(
                f"tract_profiles/{scalar}",
                f"{'_'.join(names)}")
        else:
            fname = out_file

        ba = BrainAxes(positions=positions)
        labels = []
        self.logger.info("Calculating means and CIs...")
        for j, bundle in enumerate(tqdm(self.bundles)):
            labels_temp = []
            for i, name in enumerate(names):
                profile = self.profile_dict[name]
                profile = profile[profile['tractID'] == bundle]
                ba.plot_line(
                    bundle, "nodeID", scalar, profile,
                    scalar, ylim, n_boot, self._alpha(0.6 + 0.2 * i),
                    {
                        "dashes": [(2**i, 2**i)],
                        "hue": "tractID",
                        "palette": [self.color_dict[bundle]]},
                    plot_subject_lines=plot_subject_lines)
                if j == 0:
                    line = Line2D(
                        [], [],
                        color=[0, 0, 0])
                    line.set_dashes((2**(i + 1), 2**(i + 1)))
                    if ba.is_using_temp_axis():
                        labels_temp.append(line)
                    else:
                        labels.append(line)
            if ba.is_using_temp_axis():
                ba.temp_fig.legend(
                    labels_temp, names, fontsize=medium_font)
                ba.save_temp_fig(f"{fname}_{bundle}")
        ba.fig.legend(labels, names, loc='center', fontsize=medium_font)
        ba.format()

        ba.fig.savefig(fname)

        if not show_plots:
            ba.close_all()
            plt.ion()

    def _contrast_index_df_maker(self, bundles, names, scalar):
        ci_df = pd.DataFrame(columns=["subjectID", "nodeID", "diff"])
        for subject in self.subjects:
            profiles = [None] * 2
            both_found = True
            for i, name in enumerate(names):
                for j, bundle in enumerate(bundles):
                    profiles[i + j] = self._get_profile(
                        name, bundle, subject, scalar)
                    if profiles[i + j] is None:
                        both_found = False
            if both_found:
                this_contrast_index = \
                    calc_contrast_index(profiles[0], profiles[1])
                for i, diff in enumerate(this_contrast_index):
                    ci_df = ci_df.append({
                        "subjectID": subject,
                        "nodeID": i,
                        "diff": diff},
                        ignore_index=True)
        return ci_df

    def contrast_index(self, names=None, scalar="dti_fa",
                       show_plots=False, n_boot=1000,
                       positions=POSITIONS, plot_subject_lines=True):
        """
        Calculate the contrast index for each bundle in two datasets.

        Parameters
        ----------
        names : list of strings, optional
            Names of datasets to plot profiles of.
            If None, all datasets are used.
            Should be a total of only two datasets.
            Default: None

        scalar : string, optional
            Scalar to use for the contrast index. Default: "dti_fa".

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        n_boot : int, optional
            Number of bootstrap resamples for seaborn to use
            to estimate the ci.
            Default: 1000

        positions : dictionary, optional
            Dictionary that maps bundle names to position in plot.
            Default: POSITIONS

        plot_subject_lines : bool, optional
            Whether to plot individual subject lines with a smaller width.
            Default: True
        """
        if not show_plots:
            plt.ioff()

        if names is None:
            names = list(self.profile_dict.keys())
        if len(names) != 2:
            self.logger.error("To calculate the contrast index, "
                              + "only two dataset names should be given")
            return None

        ba = BrainAxes(positions=positions)
        ci_all_df = np.zeros((len(self.bundles), self.prof_len))
        for j, bundle in enumerate(tqdm(self.bundles)):
            ci_df = self._contrast_index_df_maker(
                [bundle], names, scalar)
            ba.plot_line(
                bundle, "nodeID", "diff", ci_df, "C.I. * 2", (-1, 1),
                n_boot, 1.0, {"color": self.color_dict[bundle]},
                plot_subject_lines=plot_subject_lines)
            ci_all_df[j] = ci_df["diff"]
            ba.save_temp_fig(
                self._get_fname(
                    f"contrast_plots/{scalar}/",
                    f"{names[0]}_vs_{names[1]}_contrast_index_{bundle}"))
        ba.fig.legend([scalar], loc='center', fontsize=medium_font)
        ba.format()
        ba.fig.savefig(
            self._get_fname(
                f"contrast_plots/{scalar}/",
                f"{names[0]}_vs_{names[1]}_contrast_index"))
        if not show_plots:
            ba.close_all()
            plt.ion()
        return ci_all_df

    def lateral_contrast_index(self, name, scalar="dti_fa",
                               show_plots=False, n_boot=1000,
                               positions=POSITIONS, plot_subject_lines=True):
        """
        Calculate the lateral contrast index for each bundle in a given
        dataset, for each dataset in names.

        Parameters
        ----------
        name : string
            Names of dataset to plot profiles of.

        scalar : string, optional
            Scalar to use for the contrast index. Default: "dti_fa".

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        n_boot : int, optional
            Number of bootstrap resamples for seaborn to use
            to estimate the ci.
            Default: 1000

        positions : dictionary, optional
            Dictionary that maps bundle names to position in plot.
            Default: POSITIONS

        plot_subject_lines : bool, optional
            Whether to plot individual subject lines with a smaller width.
            Default: True
        """
        if not show_plots:
            plt.ioff()

        ba = BrainAxes(positions=positions)
        for j, bundle in enumerate(tqdm(self.bundles)):
            other_bundle = list(bundle)
            if other_bundle[-1] == 'L':
                other_bundle[-1] = 'R'
            elif other_bundle[-1] == 'R':
                other_bundle[-1] = 'L'
            else:
                continue
            other_bundle = "".join(other_bundle)
            if other_bundle not in self.bundles:
                continue

            ci_df = self._contrast_index_df_maker(
                [bundle, other_bundle], [name], scalar)
            ba.plot_line(
                bundle, "nodeID", "diff", ci_df, "C.I. * 2", (-1, 1),
                n_boot, 1.0, {"color": self.color_dict[bundle]},
                plot_subject_lines=plot_subject_lines)
            ba.save_temp_fig(
                self._get_fname(
                    f"contrast_plots/{scalar}/",
                    f"{name}_lateral_contrast_index_{bundle}"))

        ba.fig.legend([scalar], loc='center', fontsize=medium_font)
        ba.format()
        ba.fig.savefig(
            self._get_fname(
                f"contrast_plots/{scalar}/",
                f"{name}_lateral_contrast_index"))
        if not show_plots:
            ba.close_all()

        if not show_plots:
            plt.ion()

    def reliability_plots(self, names=None,
                          scalars=["dti_fa", "dti_md"],
                          ylims=[0.0, 1.0],
                          show_plots=False,
                          only_plot_above_thr=None,
                          positions=POSITIONS):
        """
        Plot the scan-rescan reliability using Pearson's r for 2 scalars.

        Parameters
        ----------
        names : list of strings, optional
            Names of datasets to plot profiles of.
            If None, all datasets are used.
            Should be a total of only two datasets.
            Default: None

        scalars : list of strings, optional
            Scalars to correlate. Default: ["dti_fa", "dti_md"].

        ylims : 2-tuple of floats, optional
            Limits of the y-axis. Useful to synchronize axes across graphs.
            Default: [0.0, 1.0].

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        only_plot_above_thr : int or None, optional
            Only plot bundles with intrersubject reliability above this
            threshold on the final reliability bar plots. If None, plot all.
            Default: None

        positions : dictionary, optional
            Dictionary that maps bundle names to position in plot.
            Default: POSITIONS

        Returns
        -------
        Matplotlib figure and axes.
        """
        if not show_plots:
            plt.ioff()
        if names is None:
            names = list(self.profile_dict.keys())
        if len(names) != 2:
            self.logger.error("To plot correlations, "
                              + "only two dataset names should be given")
            return None

        # extract relevant statistics / data from profiles
        all_sub_coef = np.zeros((len(scalars), len(self.bundles)))
        all_sub_means = np.zeros(
            (len(scalars), len(self.bundles), 2, len(self.subjects)))
        all_profile_coef = \
            np.zeros((len(scalars), len(self.bundles), len(self.subjects)))
        all_node_coef = np.zeros(
            (len(scalars), len(self.bundles), self.prof_len))
        for m, scalar in enumerate(scalars):
            for k, bundle in enumerate(self.bundles):
                bundle_profiles =\
                    np.zeros((2, len(self.subjects), self.prof_len))
                for j, name in enumerate(names):
                    for i, subject in enumerate(self.subjects):
                        single_profile = self._get_profile(
                            name, bundle, subject, scalar, repl_nan=False)
                        if single_profile is None:
                            bundle_profiles[j, i] = np.nan
                        else:
                            bundle_profiles[j, i] = single_profile

                all_sub_means[m, k] = np.nanmean(bundle_profiles, axis=2)
                all_sub_coef[m, k] = self.masked_corr(all_sub_means[m, k])

                bundle_coefs = np.zeros(len(self.subjects))
                for i in range(len(self.subjects)):
                    bundle_coefs[i] = \
                        self.masked_corr(bundle_profiles[:, i, :])
                all_profile_coef[m, k] = bundle_coefs

                node_coefs = np.zeros(self.prof_len)
                for i in range(self.prof_len):
                    node_coefs[i] = self.masked_corr(bundle_profiles[:, :, i])
                all_node_coef[m, k] = node_coefs

        # plot histograms of subject pearson r's
        maxi = all_profile_coef.max()
        mini = all_profile_coef.min()
        bins = np.linspace(mini, maxi, 10)
        ba = BrainAxes(positions=positions)
        for k, bundle in enumerate(self.bundles):
            ax = ba.get_axis(bundle)
            for m, scalar in enumerate(scalars):
                bundle_coefs = all_profile_coef[m, k]
                sns.set(style="whitegrid")
                sns.histplot(
                    data=bundle_coefs,
                    bins=bins,
                    alpha=0.5,
                    label=scalar,
                    color=tableau_20_sns[m * 2],
                    ax=ax)
            ax.set_title(bundle, fontsize=large_font)
            ax.set_xlabel("Pearson's r", fontsize=medium_font)
            ax.set_ylabel("Count", fontsize=medium_font)
            ba.temp_fig.legend(scalars, fontsize=medium_font)
            ba.save_temp_fig(
                self._get_fname(
                    f"rel_plots/{'_'.join(scalars)}/verbose",
                    (f"{names[0]}_vs_{names[1]}_profile_r_distributions"
                     f"_{bundle}")))

        ba.fig.legend(scalars, loc='center', fontsize=medium_font)
        ba.format(disable_x=False)
        ba.fig.savefig(
            self._get_fname(
                f"rel_plots/{'_'.join(scalars)}/verbose",
                f"{names[0]}_vs_{names[1]}_profile_r_distributions"))

        if not show_plots:
            ba.close_all()

        # plot node reliability profile
        all_node_coef[np.isnan(all_node_coef)] = 0
        if ylims is None:
            maxi = all_node_coef.max()
            mini = all_node_coef.min()
        else:
            maxi = ylims[1]
            mini = ylims[0]
        ba = BrainAxes(positions=positions)
        for k, bundle in enumerate(self.bundles):
            ax = ba.get_axis(bundle)
            for m, scalar in enumerate(scalars):
                sns.set(style="whitegrid")
                sns.lineplot(
                    data=all_node_coef[m, k],
                    label=scalar,
                    color=tableau_20_sns[m * 2],
                    ax=ax,
                    legend=False,
                    ci=None, estimator=None)
            ax.set_ylim([mini, maxi])
            ax.set_title(bundle, fontsize=large_font)
            ax.set_ylabel("Pearson's r", fontsize=medium_font)
            ba.temp_fig.legend(scalars, fontsize=medium_font)
            ba.save_temp_fig(
                self._get_fname(
                    f"rel_plots/{'_'.join(scalars)}/verbose",
                    (f"{names[0]}_vs_{names[1]}_node_profiles"
                     f"_{bundle}")))

        ba.fig.legend(scalars, loc='center', fontsize=medium_font)
        ba.format()
        ba.fig.savefig(
            self._get_fname(
                f"rel_plots/{'_'.join(scalars)}/verbose",
                f"{names[0]}_vs_{names[1]}_node_profiles"))

        if not show_plots:
            ba.close_all()

        # plot mean profile scatter plots
        for m, scalar in enumerate(scalars):
            maxi = np.nanmax(all_sub_means[m])
            mini = np.nanmin(all_sub_means[m])
            ba = BrainAxes(positions=positions)
            for k, bundle in enumerate(self.bundles):
                this_sub_means = self._array_to_df(all_sub_means[m, k])
                ax = ba.get_axis(bundle)
                sns.set(style="whitegrid")
                sns.scatterplot(
                    data=this_sub_means, x='x', y='y',
                    label=scalar,
                    color=tableau_20_sns[m * 2],
                    legend=False,
                    ax=ax)
                ax.set_title(bundle, fontsize=large_font)
                ax.set_xlabel(names[0], fontsize=medium_font)
                ax.set_ylabel(names[1], fontsize=medium_font)
                ax.set_ylim([mini, maxi])
                ax.set_xlim([mini, maxi])
                ba.temp_fig.legend(
                    [scalar], fontsize=medium_font)
                ba.save_temp_fig(
                    self._get_fname(
                        f"rel_plots/{'_'.join(scalars)}/verbose",
                        (f"{names[0]}_vs_{names[1]}_{scalar}_mean_profiles"
                         f"_{bundle}")))

            ba.fig.legend([scalar], loc='center', fontsize=medium_font)
            ba.format(disable_x=False)
            ba.fig.savefig(
                self._get_fname(
                    f"rel_plots/{'_'.join(scalars)}/verbose",
                    f"{names[0]}_vs_{names[1]}_{scalar}_mean_profiles"))
            if not show_plots:
                ba.close_all()

        # plot bar plots of pearson's r
        fig, axes = plt.subplots(2, 1)
        fig.set_size_inches((8, 8))
        bundle_prof_means = np.nanmean(all_profile_coef, axis=2)
        bundle_prof_stds = sem(all_profile_coef, axis=2, nan_policy='omit')
        if ylims is None:
            maxi = np.maximum(bundle_prof_means.max(), all_sub_coef.max())
            mini = np.minimum(bundle_prof_means.min(), all_sub_coef.min())
        else:
            maxi = ylims[1]
            mini = ylims[0]

        if only_plot_above_thr is not None:
            is_removed_bundle =\
                np.logical_not(
                    np.any(all_sub_coef > only_plot_above_thr, axis=0))
            removal_idx = np.where(is_removed_bundle)[0]
            bundle_prof_means_removed = np.delete(
                bundle_prof_means,
                removal_idx,
                axis=1)
            bundle_prof_stds_removed = np.delete(
                bundle_prof_stds,
                removal_idx,
                axis=1)
            all_sub_coef_removed = np.delete(
                all_sub_coef,
                removal_idx,
                axis=1)
        else:
            is_removed_bundle = [False]*len(self.bundles)

        df_bundle_prof_means = pd.DataFrame(
            columns=['scalar', 'tractID', 'value'])
        updated_bundles = []
        for m, scalar in enumerate(scalars):
            for k, bundle in enumerate(self.bundles):
                if not is_removed_bundle[k]:
                    df_bundle_prof_means = df_bundle_prof_means.append({
                        'scalar': scalar,
                        'tractID': bundle,
                        'value': bundle_prof_means[m, k]}, ignore_index=True)
                    if m == 0:
                        updated_bundles.append(bundle)
            df_bundle_prof_means = df_bundle_prof_means.append({
                'scalar': scalar,
                'tractID': "median",
                'value': np.median(bundle_prof_means_removed[m])},
                ignore_index=True)
            if m == 0:
                updated_bundles.append("median")
        df_all_sub_coef = pd.DataFrame(columns=['scalar', 'tractID', 'value'])
        for m, scalar in enumerate(scalars):
            for k, bundle in enumerate(self.bundles):
                if not is_removed_bundle[k]:
                    df_all_sub_coef = df_all_sub_coef.append({
                        'scalar': scalar,
                        'tractID': bundle,
                        'value': all_sub_coef[m, k]}, ignore_index=True)
            df_all_sub_coef = df_all_sub_coef.append({
                'scalar': scalar,
                'tractID': "median",
                'value': np.median(all_sub_coef_removed[m])},
                ignore_index=True)

        sns.set(style="whitegrid")
        sns.barplot(
            data=df_bundle_prof_means, x='tractID', y='value', hue='scalar',
            palette=tableau_20_sns[:len(scalars) * 2 + 2:2],
            yerr=[*bundle_prof_stds_removed[m], 0],
            ax=axes[0])
        axes[0].legend_.remove()
        sns.barplot(
            data=df_all_sub_coef, x='tractID', y='value', hue='scalar',
            palette=tableau_20_sns[:len(scalars) * 2 + 2:2],
            ax=axes[1])
        axes[1].legend_.remove()

        if len(updated_bundles) > 20:
            xaxis_font_size = small_font - 4
        else:
            xaxis_font_size = small_font

        axes[0].set_title("A", fontsize=large_font)
        axes[0].set_ylabel('Mean of\nPearson\'s r\nof profiles',
                           fontsize=medium_font)
        axes[0].set_ylim([mini, maxi])
        axes[0].set_xlabel("")
        axes[0].set_xticklabels(
            updated_bundles, fontsize=xaxis_font_size)
        axes[1].set_title("B", fontsize=large_font)
        axes[1].set_ylabel('Pearson\'s r\nof mean\nof profiles',
                           fontsize=medium_font)
        axes[1].set_ylim([mini, maxi])
        axes[1].set_xlabel("")
        axes[1].set_xticklabels(
            updated_bundles, fontsize=xaxis_font_size)

        plt.setp(axes[0].get_xticklabels(),
                 rotation=45,
                 horizontalalignment='right')
        plt.setp(axes[1].get_xticklabels(),
                 rotation=45,
                 horizontalalignment='right')

        fig.tight_layout()
        legend_labels = []
        for m, _ in enumerate(scalars):
            legend_labels.append(Line2D(
                [], [],
                color=tableau_20_sns[m * 2]))
        fig.legend(
            legend_labels,
            scalars,
            fontsize=small_font,
            bbox_to_anchor=(1.25, 0.5))
        fig.savefig(self._get_fname(
            f"rel_plots/{'_'.join(scalars)}",
            f"{names[0]}_vs_{names[1]}"))

        if not show_plots:
            plt.close(fig)
            plt.ion()
        return fig, axes


def visualize_gif_inline(fname, use_s3fs=False):
    """Display a gif inline, possible from s3fs """
    if use_s3fs:
        import s3fs
        fs = s3fs.S3FileSystem()
        tdir = tempfile.gettempdir()
        fname_remote = fname
        fname = op.join(tdir, "fig.gif")
        fs.get(fname_remote, fname)

    display.display(display.Image(fname))


def show_anatomical_slices(img_data, title):
    """
    display anatomical slices from midpoint

    based on:
    https://nipy.org/nibabel/coordinate_systems.html
    """

    axial_slice = img_data[:, :, int(img_data.shape[2] / 2)]
    coronal_slice = img_data[:, int(img_data.shape[1] / 2), :]
    sagittal_slice = img_data[int(img_data.shape[0] / 2), :, :]

    fig = plt.figure(constrained_layout=False)
    gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.01, hspace=0.01)
    ax1 = fig.add_subplot(gs[:-1, :])
    ax1.imshow(axial_slice.T, cmap="gray", origin="lower")
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.imshow(coronal_slice.T, cmap="gray", origin="lower")
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.imshow(sagittal_slice.T, cmap="gray", origin="lower")
    ax3.axis('off')

    plt.suptitle(title)
    plt.show()
