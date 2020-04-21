import numpy as np
import pandas as pd
import logging
import os

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin

from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.io.streamline import save_tractogram, load_tractogram

import AFQ.segmentation as seg


class Bundles:
    def __init__(self, reference='same', space=Space.RASMM,
                 origin=Origin.NIFTI, bundles_dict=None,
                 using_idx=False):
        """
        Collection of bundles.

        Parameters
        ----------
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram,
            optional.
            see DIPY.
            Default: 'same'

        origin : Enum (dipy.io.stateful_tractogram.Origin), optional
            see DIPY.
            Default: Origin.NIFTI

        space : string, optional.
            see DIPY.
            Default: Space.RASMM

        bundles : dict, optional.
            Keys are names of the bundles.
            If using_idx is False, values are StatefulTractograms
            or Streamline Objects.
            Else, values are dictionaries with StatefulTractograms
            or Streamline objects and indices.
            Default: None.

        using_idx : boolean
            Whether or not bundles_dict contains indices information.
            Default: False.
        """
        self.bundles = {}
        self.reference = reference
        self.origin = origin
        self.space = space

        if bundles_dict is not None:
            for bundle_name in bundles_dict:
                if using_idx:
                    self.add_bundle(bundle_name,
                                    bundles_dict[bundle_name]['sl'],
                                    bundles_dict[bundle_name]['idx'])
                else:
                    self.add_bundle(bundle_name,
                                    bundles_dict[bundle_name])
                logging.disable(level=logging.WARNING)
            logging.disable(logging.NOTSET)

    def add_bundle(self, bundle_name, streamlines, idx=None):
        """
        Add a bundle to bundles.

        Parameters
        ----------
        bundle_name : string
            Name of bundle.

        streamlines : nibabel.Streamlines or StatefulTractogram
            The streamlines constituting a bundle.

        idx : array of ints, optional
            Indices for streamlines in original tractography.
            Default: None.
        """
        if isinstance(streamlines, StatefulTractogram):
            if self.space == Space.VOX:
                streamlines.to_vox()
            elif self.space == Space.VOXMM:
                streamlines.to_voxmm()
            elif self.space == Space.RASMM:
                streamlines.to_rasmm()

            if idx is None:
                self.bundles[bundle_name] = streamlines
            else:
                self.bundles[bundle_name] = \
                    StatefulTractogram(streamlines.streamlines,
                                       self.reference,
                                       self.space,
                                       origin=self.origin,
                                       data_per_streamline={'idx': idx})
        else:
            self.bundles[bundle_name] = \
                StatefulTractogram(streamlines,
                                   self.reference,
                                   self.space,
                                   origin=self.origin,
                                   data_per_streamline={'idx': idx})

    def clean_bundles(self, **kwargs):
        """
        Clean each segmented bundle based on the Mahalnobis distance of
        each streamline

        Parameters
        ----------
        clean_rounds : int, optional.
            Number of rounds of cleaning based on the Mahalanobis distance
            from the mean of extracted bundles. Default: 5

        clean_threshold : float, optional.
            Threshold of cleaning based on the Mahalanobis distance (the units
            are standard deviations). Default: 3.

        min_sl : int, optional.
            Number of streamlines in a bundle under which we will
            not bother with cleaning outliers. Default: 20.

        stat : callable, optional.
            The statistic of each node relative to which the Mahalanobis is
            calculated. Default: `np.mean` (but can also use median, etc.)
        """

        for bundle_name, bundle in self.bundles.items():
            if bundle.data_per_streamline is not None:
                new_sls, idx_in_bundle = seg.clean_bundle(
                    bundle,
                    return_idx=True,
                    **kwargs)
                new_idx = bundle.data_per_streamline['idx'][idx_in_bundle]
            else:
                new_sls = seg.clean_bundle(bundle,
                                           return_idx=False,
                                           **kwargs)
                new_idx = None
            self.bundles[bundle_name] = \
                StatefulTractogram(new_sls.streamlines,
                                   self.reference,
                                   self.space,
                                   origin=self.origin,
                                   data_per_streamline={'idx': new_idx})
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)

    def _apply_affine_sft(self, sft, affine, reference, origin):
        sls = dtu.transform_tracking_output(sft.streamlines, affine)
        return StatefulTractogram(sls,
                                  reference,
                                  self.space,
                                  origin=origin,
                                  data_per_streamline=sft.data_per_streamline)

    def apply_affine(self, affine, reference, origin=Origin.NIFTI):
        """
        Apply a linear transformation, given by affine, to all
        streamlines.

        Parameters
        ----------
        affine : array (4, 4)
            Apply affine matrix to all streamlines

        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the new spatial attribute.

        origin : Enum (dipy.io.stateful_tractogram.Origin), optional
            New origin of streamlines.
            Default: Origin.NIFTI
        """
        for bundle_idx, bundle in self.bundles.items():
            self.bundles[bundle_idx] = self._apply_affine_sft(bundle,
                                                              affine,
                                                              reference,
                                                              origin)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)

    def to_space(self, space):
        """
        Transform streamlines to space.

        Parameters
        ----------
        space : Space
            Space to transform the streamlines to.
        """
        for bundle_idx, _ in self.bundles.items():
            self.bundles[bundle_idx].to_space(space)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)

    def save_bundles(self, file_path='./', file_suffix='.trk',
                     space=None, bbox_valid_check=False):
        """
        Save tractograms in bundles.

        Parameters
        ----------
        file_path : string, optional.
            Path to save trk files to.
            Default: './'
        file_suffix : string, optional.
            File name will be the bundle name + file_suffix.
            Default: '.trk'
        space : string
            Space to save the streamlines in. If not none, the streamlines
            will be transformed to this space, saved, then transformed back.
            Default: None.
        bbox_valid_check : boolean, optional.
            Whether to verify that the bounding box is valid in voxel space.
            Default: False
        """
        if space is not None:
            space_temp = self.space
            self.to_space(space)

        for bundle_name, bundle in self.bundles.items():
            save_tractogram(bundle,
                            os.path.join(file_path,
                                         f"{bundle_name}{file_suffix}"),
                            bbox_valid_check=bbox_valid_check)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)

        if space is not None:
            self.to_space(space_temp)

    def load_bundles(self, bundle_names, file_path='./', file_suffix='.trk',
                     affine=np.eye(4), bbox_valid_check=False):
        """
        load tractograms from file.

        Parameters
        ----------
        bundle_names : list of strings
            Names of bundles to load.
        file_path : string, optional.
            Path to load trk files from.
            Default: './'
        file_suffix : string, optional.
            File name will be the bundle name + file_suffix.
            Default: '.trk'
        affine : array_like (4, 4), optional.
            The mapping from the file's reference to this object's reference.
            Default: np.eye(4)
        bbox_valid_check : boolean, optional.
            Whether to verify that the bounding box is valid in voxel space.
            Default: False
        """

        for bundle_name in bundle_names:
            full_path = os.path.join(file_path, f"{bundle_name}{file_suffix}")
            if self.reference == 'same':
                sft = load_tractogram(
                    full_path,
                    self.reference,
                    bbox_valid_check=bbox_valid_check)
                self.reference = sft
                self.origin = sft.origin
                self.space = sft.space
            else:
                sft = load_tractogram(
                    full_path,
                    self.reference,
                    to_space=self.space,
                    bbox_valid_check=bbox_valid_check)
            sft = self._apply_affine_sft(
                sft, affine, self.reference, self.origin)
            self.add_bundle(bundle_name, sft)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)

    def tract_profiles(self, data, subject_label, affine=np.eye(4),
                       method='afq', metric='FA', n_points=100,
                       weight=True):
        """
        Calculate a summarized profile of data for each bundle along
        its length.

        Follows the approach outlined in [Yeatman2012]_.

        Parameters
        ----------
        data : 3D volume
            The statistic to sample with the streamlines.

        subject_label : string
            String which identifies these bundles in the pandas dataframe.

        affine : array_like (4, 4), optional.
            The mapping from voxel coordinates to 'data' coordinates.
            Default: np.eye(4)

        method : string
            Method used to segment streamlines.
            Default: 'afq'

        metric : string
            Metric of statistic in data.
            Default: 'FA'

        n_points : int
            Number of points to resample to.
            Default: 100

        weight : boolean
            Whether to calculate gaussian weights before profiling.
            Default: True
        """
        self.to_space(Space.VOX)
        profiles = []
        for bundle_name, bundle in self.bundles.items():
            if weight:
                weights = gaussian_weights(bundle.streamlines,
                                           n_points=n_points)
            else:
                weights = None
            profile = afq_profile(data, bundle.streamlines,
                                  affine, weights=weights, n_points=n_points)
            for ii in range(len(profile)):
                # Subject, Bundle, node, method, metric (FA, MD), value
                profiles.append([subject_label, bundle_name, ii, method,
                                 metric, profile[ii]])
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)
        profiles = pd.DataFrame(data=profiles,
                                columns=["Subject", "Bundle", "Node",
                                         "Method", "Metric", "Value"])
        return profiles
