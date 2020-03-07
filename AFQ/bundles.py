import numpy as np
import logging

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.io.streamline import save_tractogram, load_tractogram

import AFQ.segmentation as seg

# TODO: make tests

class Bundles:
    def __init__(self, reference, space=Space.RASMM,
                 bundles_dict=None, using_idx=False):
        """
        Collection of bundles.

        Parameters
        ----------
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the spatial attribute.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation

        space : string, optional.
            Current space in which the streamlines are (vox, voxmm or rasmm)
            Typically after tracking the space is VOX, after nibabel loading
            the space is RASMM
            Default: Space.RASMM

        bundles : dict, optional.
            Keys are names of the bundles.
            If using_idx is true, values are Streamline objects.
            Else, values are dictionaries with Streamline objects and indices.
            The streamlines in each object have all been oriented to have the
            same orientation (using `dts.orient_by_streamline`).
            Default: None.

        using_idx : boolean
            Whether or not bundles_dict contains indices information.
            Default: False.
        """
        self.bundles = {}
        self.reference = reference
        self.space = space
        self.logger = logging.getLogger('AFQ.Bundles')

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
        self.bundles[bundle_name] = {}

        if isinstance(streamlines, StatefulTractogram):
            if self.space == Space.VOX:
                streamlines.to_vox()
            elif self.space == Space.VOXMM:
                streamlines.to_voxmm()
            elif self.space == Space.RASMM:
                streamlines.to_rasmm()

            self.bundles[bundle_name]['sl'] = streamlines
        else:
            self.bundles[bundle_name]['sl'] = \
                StatefulTractogram(streamlines,
                                   self.reference,
                                   self.space)

        if idx is not None:
            self.bundles[bundle_name]['idx'] = idx
            self.bundles[bundle_name]['using_idx'] = True
        else:
            self.bundles[bundle_name]['using_idx'] = False


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

        for _, bundle in self.bundles:
            if bundle['using_idx']:
                bundle['sl'], idx_in_bundle = seg.clean_bundle(
                    bundle['sl'].streamlines,
                    return_idx=True
                    **kwargs)
                bundle['idx'] = bundle['idx'][idx_in_bundle]
            else:
                bundle['sl'] = seg.clean_bundle(bundle['sl'].streamlines,
                                                return_idx=False
                                                **kwargs)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)


    def apply_affine(self, affine):
        """
        Appliy a linear transformation, given by affine, to all
        streamlines.

        Parameters
        ----------
        affine : array (4, 4)
            Apply affine matrix to all streamlines
        """
        for _, bundle in self.bundles:
            bundle['sl'].apply_affine(affine)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)


    def to_space(self, space):
        """
        Transform streamlines to space.

        Parameters
        ----------
        space : string
            Space to transform the streamlines to.
        """
        space = space.lower()
        for _, bundle in self.bundles:
            bundle['sl'].to_space(space)
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

        for bundle_name, bundle in self.bundles:
            save_tractogram(bundle['sl'],
                            f'{file_path}{bundle_name}{file_suffix}',
                            bbox_valid_check=bbox_valid_check)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)
        
        if space is not None:
            self.to_space(space_temp)


    def load_bundles(self, bundle_names, file_path='./', file_suffix='.trk',
                     affine=np.eye(4), bbox_valid_check=False):
        """
        Save tractograms in bundles.

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
            sft = load_tractogram(f'{file_path}{bundle_name}{file_suffix}',
                                  self.reference,
                                  to_space=self.space,
                                  bbox_valid_check=bbox_valid_check)
            sft.apply_affine(affine)
            self.add_bundle(bundle_name, sft)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)


    def tract_profiles(self, data, affine=np.eye(4)):
        """
        Calculate a summarized profile of data for each bundle along
        its length.

        Follows the approach outlined in [Yeatman2012]_.

        Parameters
        ----------
        data : 3D volume
            The statistic to sample with the streamlines.

        affine : array_like (4, 4), optional.
            The mapping from voxel coordinates to 'data' coordinates.
            Default: np.eye(4)
        """
        self.to_space(Space.VOX)
        for _, bundle in self.bundles:
            weights = gaussian_weights(bundle['sl'].streamlines)
            bundle['profile'] = afq_profile(data, bundle['sl'].streamlines,
                                            affine, weights=weights)
            logging.disable(level=logging.WARNING)
        logging.disable(logging.NOTSET)