from collections import OrderedDict
import os.path as op
import os
import logging
import tempfile

import numpy as np
from scipy.stats import sem
import pandas as pd
from palettable.tableau import Tableau_20
import imageio as io
import IPython.display as display
import matplotlib.pyplot as plt

import nibabel as nib
from dipy.io.streamline import load_tractogram
from dipy.tracking.utils import transform_tracking_output
from dipy.io.stateful_tractogram import StatefulTractogram, Space

import AFQ.utils.volume as auv
import AFQ.registration as reg
from AFQ.utils.stats import contrast_index as calc_contrast_index

__all__ = ["Viz", "visualize_tract_profiles", "visualize_gif_inline"]

viz_logger = logging.getLogger("AFQ.viz")
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
                         "FP": (4, 2), "FA": (0, 2),
                         "IFO_L": (4, 1), "IFO_R": (4, 3),
                         "ILF_L": (3, 0), "ILF_R": (3, 4),
                         "SLF_L": (2, 1), "SLF_R": (2, 3),
                         "ARC_L": (2, 0), "ARC_R": (2, 4),
                         "UNC_L": (0, 1), "UNC_R": (0, 3)})

BUNDLE_MAT_2_PYTHON = \
    {'Right Corticospinal': 'CST_R', 'Left Corticospinal': 'CST_L',
     'Right Uncinate': 'UNC_R', 'Left Uncinate': 'UNC_L',
     'Left IFOF': 'IFO_L', 'Right IFOF': 'IFO_R',
     'Right Arcuate': 'ARC_R', 'Left Arcuate': 'ARC_L',
     'Right Thalamic Radiation': 'ATR_R', 'Left Thalamic Radiation': 'ATR_L',
     'Right Cingulum Cingulate': 'CGC_R', 'Left Cingulum Cingulate': 'CGC_L',
     'Right Cingulum Hippocampus': 'HCC_R',
     'Left Cingulum Hippocampus': 'HCC_L',
     'Callosum Forceps Major': 'FP', 'Callosum Forceps Minor': 'FA',
     'Right ILF': 'ILF_R', 'Left ILF': 'ILF_L',
     'Right SLF': 'SLF_R', 'Left SLF': 'SLF_L'}

CSV_MAT_2_PYTHON = \
    {'fa': 'dti_fa', 'md': 'dti_md',
     'tractID': 'bundle'}

SCALE_MAT_2_PYTHON = \
    {'dti_md': 0.001}


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


def tract_generator(sft, affine, bundle, bundle_dict, colors):
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

    Returns
    -------
    Statefule Tractogram streamlines, RGB numpy array, str
    """

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

    if colors is None:
        # Use the color dict provided
        colors = COLOR_DICT

    if list(sft.data_per_streamline.keys()) == []:
        # There are no bundles in here:
        yield streamlines, [0.5, 0.5, 0.5], "all_bundles"

    else:
        # There are bundles:
        if bundle is None:
            # No selection: visualize all of them:

            for b in np.unique(sft.data_per_streamline['bundle']):
                idx = np.where(sft.data_per_streamline['bundle'] == b)[0]
                these_sls = streamlines[idx]
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
        if backend == "fury":
            try:
                import AFQ.viz.fury_backend
            except ImportError:
                raise ImportError(viz_import_msg_error("fury"))
            self.visualize_bundles = AFQ.viz.fury_backend.visualize_bundles
            self.visualize_roi = AFQ.viz.fury_backend.visualize_roi
            self.visualize_volume = AFQ.viz.fury_backend.visualize_volume
            self.create_gif = AFQ.viz.fury_backend.create_gif
        elif backend == "plotly":
            try:
                import AFQ.viz.plotly_backend
            except ImportError:
                raise ImportError(viz_import_msg_error("plotly"))
            self.visualize_bundles = AFQ.viz.plotly_backend.visualize_bundles
            self.visualize_roi = AFQ.viz.plotly_backend.visualize_roi
            self.visualize_volume = AFQ.viz.plotly_backend.visualize_volume
            self.create_gif = AFQ.viz.plotly_backend.create_gif
        else:
            raise TypeError("Visualization backend should be"
                            + " either 'plotly' or 'fury'. "
                            + "It is currently set to %s"
                            % backend)


def visualize_tract_profiles(tract_profiles, scalar="dti_fa", min_fa=0.0,
                             max_fa=1.0, file_name=None, positions=POSITIONS):
    """
    Visualize all tract profiles for a scalar in one plot

    Parameters
    ----------
    tract_profiles : pandas dataframe
        Pandas dataframe of tract_profiles. For example,
        tract_profiles = pd.read_csv(my_afq.get_tract_profiles()[0])

    scalar : string, optional
       Scalar to use in plots. Default: "dti_fa".

    min_fa : float, optional
        Minimum FA used for y-axis bounds. Default: 0.0

    max_fa : float, optional
        Maximum FA used for y-axis bounds. Default: 1.0

    file_name : string, optional
        File name to save figure to if not None. Default: None

    positions : dictionary, optional
        Dictionary that maps bundle names to position in plot.
        Default: POSITIONS

    Returns
    -------
        Matplotlib figure and axes.
    """

    if (file_name is not None):
        plt.ioff()

    fig, axes = plt.subplots(5, 5)

    for bundle in positions.keys():
        ax = axes[positions[bundle][0], positions[bundle][1]]
        fa = tract_profiles[
            (tract_profiles["bundle"] == bundle)
        ][scalar].values
        ax.plot(fa, 'o-', color=COLOR_DICT[bundle])
        ax.set_ylim([min_fa, max_fa])
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels([0.2, 0.4, 0.6])
        ax.set_xticklabels([])

    fig.set_size_inches((12, 12))

    axes[0, 0].axis("off")
    axes[0, -1].axis("off")
    axes[1, 2].axis("off")
    axes[2, 2].axis("off")
    axes[3, 2].axis("off")
    axes[4, 0].axis("off")
    axes[4, 4].axis("off")

    if (file_name is not None):
        fig.savefig(file_name)
        plt.ion()

    return fig, axes


class LongitudinalCSVComparison():
    """
    Compare different CSVs, using:
    tract profiles, contrast indices,
    scan-rescan reliability using Pearson's r.
    """

    def __init__(self, out_folder, csv_fnames, names, is_mats=False,
                 subjects=None,
                 scalar_bounds={'lb': {'dti_fa': 0.2},
                                'ub': {'dti_md': 0.002}},
                 percent_nan_tol=10,
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

        subjects : list of str, optional
            List of subjects to consider.
            If None, will use all subjects in first dataset.
            Default: None

        scalar_bounds : dictionary, optional
            A dictionary with a lower bound and upper bound containting a
            series of scalar / threshold pairs used as a white-matter mask
            on the profiles (any values outside of the threshold will be
            marked NaN and not used or set to 0, depending on the case).
            Default: {'lb': {'dti_fa': 0.2}, 'ub': {'dti_md': 0.002}}

        percent_nan_tol : int, optional
            Percentage of NaNs tolerable. If a profile has less than this
            percentage of NaNs, NaNs are interpolated. If it has more,
            the profile is thrown out.

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
            profile['subjectID'] = \
                profile['subjectID'].apply(
                    lambda x: int(
                        ''.join(c for c in x if c.isdigit())
                    ) if isinstance(x, str) else x)

            if is_mats[i]:
                profile.rename(
                    columns=mat_column_converter, inplace=True)
                profile['bundle'] = \
                    profile['bundle'].apply(
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

            self.profile_dict[names[i]] = profile
        if subjects is None:
            self.subjects = self.profile_dict[names[0]]['subjectID'].unique()
        else:
            self.subjects = subjects

    def _threshold_scalar(self, bound, threshold, val):
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
        f_folder = op.join(
            self.out_folder,
            folder)
        os.makedirs(f_folder, exist_ok=True)
        return op.join(f_folder, f_name)

    def _get_profile(self, name, bundle, subject, scalar, repl_nan=True):
        profile = self.profile_dict[name]
        single_profile = profile[
            (profile['subjectID'] == subject)
            & (profile['bundle'] == bundle)
        ][scalar].to_numpy()
        nans = np.isnan(single_profile)
        percent_nan = np.sum(nans)
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

    def _get_brain_axes(self, suptitle):
        fig, axes = plt.subplots(5, 5)
        plt.tight_layout()
        fig.set_size_inches((12, 12))
        fig.suptitle(suptitle)
        axes[0, 0].axis("off")
        axes[0, -1].axis("off")
        axes[1, 2].axis("off")
        axes[2, 2].axis("off")
        axes[3, 2].axis("off")
        axes[4, 0].axis("off")
        axes[4, 4].axis("off")
        return fig, axes

    def masked_corr(self, arr):
        mask = np.logical_not(
            np.logical_or(
                np.isnan(arr[0, ...]),
                np.isnan(arr[1, ...])))
        if np.sum(mask) < 2:
            return 0
        return np.corrcoef(arr[:, mask])[0][1]

    def tract_profiles(self, names=None, scalar="dti_fa",
                       min_scalar=0.0, max_scalar=1.0,
                       show_plots=False,
                       positions=POSITIONS):
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

        min_scalar : float, optional
            Minimum value used for y-axis bounds. Default: 0.0

        max_scalar : float, optional
            Maximum value used for y-axis bounds. Default: 1.0

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        positions : dictionary, optional
            Dictionary that maps bundle names to position in plot.
            Default: POSITIONS
        """
        if not show_plots:
            plt.ioff()
        if names is None:
            names = list(self.profile_dict.keys())

        bundles = positions.keys()

        for subject in self.subjects:
            fig, axes = self._get_brain_axes('Subject ' + str(subject))
            for bundle in bundles:
                ax = axes[positions[bundle][0], positions[bundle][1]]
                for name in names:
                    profile = self._get_profile(name, bundle, subject, scalar)
                    if profile is not None:
                        ax.plot(profile)
                ax.set_title(bundle)
                ax.set_ylim([min_scalar, max_scalar])
                y_ticks = np.asarray([0.2, 0.4, 0.6]) * max_scalar
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_ticks)
                ax.set_xticklabels([])

            fig.legend(names, loc='center')
            fig.savefig(
                self._get_fname(
                    f"tract_profiles/{scalar}",
                    f"{'_'.join(names)}_sub-{subject}"))

        if not show_plots:
            plt.ion()

    def contrast_index(self, names=None, scalar="dti_fa",
                       bundles=list(POSITIONS.keys()), show_plots=False):
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

        bundles : list of strings, optional
            Bundles to correlate. Default: list(POSITIONS.keys())

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

        Returns
        -------
        Pandas dataframe of contrast indices
        with subjects as columns and bundles as rows.
        """
        if not show_plots:
            plt.ioff()

        if names is None:
            names = list(self.profile_dict.keys())
        if len(names) != 2:
            self.logger.error("To calculate the contrast index, "
                              + "only two dataset names should be given")
            return None

        contrast_index = pd.DataFrame(index=bundles, columns=self.subjects)
        for subject in self.subjects:
            fig, axes = self._get_brain_axes(
                (f"Contrast Indices by Bundle, "
                    f" {names[0]} vs {names[1]}"))
            for bundle in bundles:
                profiles = [None] * 2
                both_found = True
                for i, name in enumerate(names):
                    profiles[i] = self._get_profile(
                        name, bundle, subject, scalar)
                    if profiles[i] is None:
                        both_found = False
                if both_found:
                    this_contrast_index = \
                        calc_contrast_index(profiles[0], profiles[1])
                    ax = axes[POSITIONS[bundle][0], POSITIONS[bundle][1]]
                    ax.plot(this_contrast_index, label=scalar)
                    ax.set_title(bundle)
                    ax.set_ylim([-1, 1])
                    ax.set_xticklabels([])
                    contrast_index.at[bundle, subject] = \
                        np.nanmean(this_contrast_index)
            fig.legend([scalar], loc='center')
            fig.savefig(
                self._get_fname(
                    f"contrast_plots/{scalar}/",
                    f"{names[0]}_vs_{names[1]}_contrast_index"))

        contrast_index.to_csv(self._get_fname(
            f"contrast_index/{scalar}",
            f"{names[0]}_vs_{names[1]}"))
        if not show_plots:
            plt.ion()
        return contrast_index

    def lateral_contrast_index(self, names=None, scalar="dti_fa",
                               bundles=list(POSITIONS.keys()),
                               show_plots=False):
        """
        Calculate the lateral contrast index for each bundle in a given
        dataset, for each dataset in names.

        Parameters
        ----------
        names : list of strings, optional
            Names of datasets to plot profiles of.
            If None, all datasets are used.
            Default: None

        scalar : string, optional
            Scalar to use for the contrast index. Default: "dti_fa".

        bundles : list of strings, optional
            Bundles to correlate.
            Every other bundle will be laterally correlated.
            There must be an even number of bundles.
            Default: list(POSITIONS.keys())

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False
        """
        if not show_plots:
            plt.ioff()

        if names is None:
            names = list(self.profile_dict.keys())

        for subject in self.subjects:
            fig, axes = self._get_brain_axes(
                f"Lateral Contrast Indices by Bundle")
            for j in range(0, len(bundles), 2):
                bundle = bundles[j]
                other_bundle = bundles[j + 1]
                for i, name in enumerate(names):
                    profile = self._get_profile(
                        name, bundle, subject, scalar)
                    other_profile = self._get_profile(
                        name, other_bundle, subject, scalar)

                    if (profile is not None) and (other_profile is not None):
                        lateral_contrast_index = \
                            calc_contrast_index(profile, other_profile)
                        ax = axes[POSITIONS[bundle][0], POSITIONS[bundle][1]]
                        ax.plot(lateral_contrast_index, label=name)
                        ax.set_title(f"{bundle} vs {other_bundle}")
                        ax.set_ylim([-1, 1])
                        ax.set_xticklabels([])
            fig.legend(names, loc='center')
            fig.savefig(
                self._get_fname(
                    f"contrast_plots/{scalar}/",
                    f"{'_'.join(names)}_lateral_contrast_index"))

        if not show_plots:
            plt.ion()

    def reliability_plots(self, names=None,
                          scalars=["dti_fa", "dti_md"],
                          ylims=None,
                          bundles=POSITIONS.keys(),
                          show_plots=False):
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
            Default: None.

        bundles : list of strings, optional
            Bundles to correlate. Default: POSITIONS.keys()

        show_plots : bool, optional
            Whether to show plots if in an interactive environment.
            Default: False

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
        all_sub_coef = np.zeros((len(scalars), len(bundles)))
        all_sub_means = np.zeros(
            (len(scalars), len(bundles), 2, len(self.subjects)))
        all_profile_coef = \
            np.zeros((len(scalars), len(bundles), len(self.subjects)))
        all_node_coef = np.zeros((len(scalars), len(bundles), 100))
        for m, scalar in enumerate(scalars):
            for k, bundle in enumerate(bundles):
                bundle_profiles = np.zeros((2, len(self.subjects), 100))
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

                node_coefs = np.zeros(100)
                for i in range(100):
                    node_coefs[i] = self.masked_corr(bundle_profiles[:, :, i])
                all_node_coef[m, k] = node_coefs

        # plot histograms of subject pearson r's
        maxi = all_profile_coef.max()
        mini = all_profile_coef.min()
        bins = np.linspace(mini, maxi, 10)
        fig, axes = self._get_brain_axes(
            (f"Distribution of Pearson's r between profiles,"
                f" {names[0]}_vs_{names[1]}"))
        for k, bundle in enumerate(bundles):
            ax = axes[POSITIONS[bundle][0], POSITIONS[bundle][1]]
            for m, scalar in enumerate(scalars):
                bundle_coefs = all_profile_coef[m, k]
                ax.hist(bundle_coefs, bins, alpha=0.5, label=scalar)
            ax.set_title(bundle)

        fig.legend(scalars, loc='center')
        fig.savefig(
            self._get_fname(
                f"rel_plots/{'_'.join(scalars)}/verbose",
                f"{names[0]}_vs_{names[1]}_profile_r_distributions"))

        # plot node reliability profile
        all_node_coef[np.isnan(all_node_coef)] = 0
        if ylims is None:
            maxi = all_node_coef.max()
            mini = all_node_coef.min()
        else:
            maxi = ylims[1]
            mini = ylims[0]
        fig, axes = self._get_brain_axes(
            (f"node reliability profiles,"
                f" {names[0]}_vs_{names[1]}"))
        for k, bundle in enumerate(bundles):
            ax = axes[POSITIONS[bundle][0], POSITIONS[bundle][1]]
            for m, scalar in enumerate(scalars):
                ax.plot(all_node_coef[m, k], label=scalar)
            ax.set_title(bundle)
            ax.set_ylim([mini, maxi])
            ax.set_xticklabels([])

        fig.legend(scalars, loc='center')
        fig.savefig(
            self._get_fname(
                f"rel_plots/{'_'.join(scalars)}/verbose",
                f"{names[0]}_vs_{names[1]}_node_profiles"))

        # plot mean profile scatter plots
        for m, scalar in enumerate(scalars):
            this_sub_means = all_sub_means[m]
            maxi = np.nanmax(this_sub_means)
            mini = np.nanmin(this_sub_means)
            fig, axes = self._get_brain_axes(
                (f"Distribution of mean profiles,"
                    f" {names[0]}_vs_{names[1]}_{scalar}"))
            for k, bundle in enumerate(bundles):
                ax = axes[POSITIONS[bundle][0], POSITIONS[bundle][1]]
                ax.scatter(
                    this_sub_means[k, 0], this_sub_means[k, 1])
                ax.set_title(bundle)
                ax.set_xlabel(names[0])
                ax.set_ylabel(names[1])
                ax.set_ylim([mini, maxi])
                ax.set_xlim([mini, maxi])

            fig.savefig(
                self._get_fname(
                    f"rel_plots/{'_'.join(scalars)}/verbose",
                    f"{names[0]}_vs_{names[1]}_{scalar}_mean_profiles"))

        # plot bar plots of pearson's r
        width = 0.6
        spacing = 2
        x = np.arange(len(bundles)) * spacing
        x_shift = np.linspace(-0.5 * width, 0.5 * width, num=len(scalars))

        fig, axes = plt.subplots(2, 1)
        bundle_prof_means = np.nanmean(all_profile_coef, axis=2)
        bundle_prof_stds = sem(all_profile_coef, axis=2, nan_policy='omit')
        if ylims is None:
            maxi = np.maximum(bundle_prof_means.max(), all_sub_coef.max())
            mini = np.minimum(bundle_prof_means.min(), all_sub_coef.min())
        else:
            maxi = ylims[1]
            mini = ylims[0]
        for m, scalar in enumerate(scalars):
            axes[0].bar(
                x + x_shift[m],
                bundle_prof_means[m],
                width,
                label=scalar,
                yerr=bundle_prof_stds[m])
            axes[1].bar(
                x + x_shift[m],
                all_sub_coef[m],
                width,
                label=scalar
            )

        axes[0].set_ylabel('Mean of\nPearson\'s r\nof profiles')
        axes[0].set_ylim([mini, maxi])
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(bundles)
        axes[0].set_title("profile_reliability")
        axes[1].set_ylabel('Pearson\'s r\nof mean\nof profiles')
        axes[1].set_ylim([mini, maxi])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(bundles)
        axes[1].set_title(f"intersubejct_reliability")

        plt.setp(axes[0].get_xticklabels(),
                 rotation=45,
                 horizontalalignment='right')
        plt.setp(axes[1].get_xticklabels(),
                 rotation=45,
                 horizontalalignment='right')

        fig.suptitle(f"{names[0]}_vs_{names[1]}")
        fig.legend(
            scalars,
            loc='lower right',
            bbox_to_anchor=(0.5, 0.15, 0.5, 0.5),
            fontsize='small')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(self._get_fname(
            f"rel_plots/{'_'.join(scalars)}",
            f"{names[0]}_vs_{names[1]}"))

        extent = axes[1].get_window_extent().transformed(
            fig.dpi_scale_trans.inverted())
        axes[1].set_title(
            f"{names[0]}_vs_{names[1]}_intersubejct_reliability")
        fig.savefig(
            self._get_fname(
                f"rel_plots/{'_'.join(scalars)}",
                f"{names[0]}_vs_{names[1]}_intersubject"),
            bbox_inches=extent.translated(0, -0.2).expanded(1.4, 2.0))
        axes[1].set_title(f"intersubejct_reliability")

        if not show_plots:
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
