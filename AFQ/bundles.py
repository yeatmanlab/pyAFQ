import numpy as np

import dipy.tracking.streamline as dts
import dipy.tracking.utils as dtu
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from dipy.stats.analysis import afq_profile, gaussian_weights
from dipy.io.streamline import save_tractogram, load_tractogram

import AFQ.segmentation as seg

class FiberGroups:
    def __init__(self, fiber_groups_dict=None, using_idx=False):
        """
        Collection of fiber groups.

        Parameters
        ----------
        fiber_groups : dict, optional.
            Keys are names of the bundles.
            If using_idx is true, values are Streamline objects.
            Else, values are dictionaries with Streamline objects and indices.
            The streamlines in each object have all been oriented to have the
            same orientation (using `dts.orient_by_streamline`).
            Default: None.

        using_idx : boolean
            Whether or not fiber_groups_dict contains indices information.
            Default: False.
        """
        self.fiber_groups = {}
        if fiber_groups_dict is not None:
            for fg_name in fiber_groups_dict:
                if using_idx:
                    self.add_fiber_group(fg_name,
                                        fiber_groups_dict[fg_name]['sl'],
                                        fiber_groups_dict[fg_name]['idx'])
                else:
                    self.add_fiber_group(fg_name,
                                        fiber_groups_dict[fg_name])

    def add_fiber_group(self, fg_name, streamlines, idx=None):
        """
        Add a fiber group to fiber_groups.

        Parameters
        ----------
        fg_name : string
            Name of fiber group.

        streamlines : nibabel.Streamlines class instance.
            The streamlines constituting a fiber group.

        idx : array of ints, optional
            Indices for streamlines in original tractography.
            Default: None.
        """
        self.fiber_groups[fg_name] = {}
        self.fiber_groups[fg_name]['sl'] = streamlines

        if idx is not None:
            self.fiber_groups[fg_name]['idx'] = idx
            self.fiber_groups[fg_name]['using_idx'] = True
        else:
            self.fiber_groups[fg_name]['using_idx'] = False

    def clean_fiber_groups(self, **kwargs):
        """
        Clean each segmented fiber group based on the Mahalnobis distance of
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

        for _, fiber_group in self.fiber_groups:
            if fiber_group['using_idx']:
                fiber_group['sl'], idx_in_bundle = seg.clean_bundle(
                    fiber_group['sl'],
                    return_idx=True
                    **kwargs)
                fiber_group['idx'] = fiber_group['idx'][idx_in_bundle]
            else:
                fiber_group['sl'] = seg.clean_bundle(fiber_group['sl'],
                                                     return_idx=False
                                                     **kwargs)

    def transform_fiber_groups(self, affine):
        """
        Appliy a linear transformation, given by affine, to all
        streamlines.

        Parameters
        ----------
        affine : array (4, 4)
            The mapping between voxel indices and the point space for seeds.
            The voxel_to_rasmm matrix, typically from a NIFTI file.
        """
        for _, fiber_group in self.fiber_groups:
            fiber_group['sl'] = dts.Streamlines(
                dtu.transform_tracking_output(fiber_group['sl'],
                                            affine))

    def save_fiber_groups(self, reference, affine=np.eye(4),
                          space=Space.RASMM,
                          file_path='./', file_suffix='.trk',
                          bbox_valid_check=False):
        """
        Save tractograms in fiber groups.

        Parameters
        ----------
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the spatial attribute.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        affine : array (4, 4), optional.
            The mapping between voxel indices and the point space for seeds.
            The voxel_to_rasmm matrix, typically from a NIFTI file.
            Default: np.eye(4)
        space : string, optional.
            Current space in which the streamlines are (vox, voxmm or rasmm)
            Typically after tracking the space is VOX, after nibabel loading
            the space is RASMM
            Default: Space.RASMM
        file_path : string, optional.
            Path to save trk files to.
            Default: './'
        file_suffix : string, optional.
            File name will be the fiber group name + file_suffix.
            Default: '.trk'
        bbox_valid_check : boolean, optional.
            Whether to verify that the bounding box is valid in voxel space.
            Default: False
        """
        for fg_name, fiber_group in self.fiber_groups:
            sft = StatefulTractogram(
                dtu.transform_tracking_output(fiber_group['sl'], affine),
                reference, space)

            save_tractogram(sft, f'{file_path}{fg_name}{file_suffix}',
                            bbox_valid_check=bbox_valid_check)

    # TODO: load_fiber_groups
    # TODO: tract_profiles
    def load_fiber_groups(self, fg_names, reference,
                          space=Space.RASMM, affine=np.eye(4),
                          file_path='./', file_suffix='.trk',
                          bbox_valid_check=False):
        """
        Save tractograms in fiber groups.

        Parameters
        ----------
        fg_names : list of strings
            Names of fiber groups to load.
        reference : Nifti or Trk filename, Nifti1Image or TrkFile,
            Nifti1Header, trk.header (dict) or another Stateful Tractogram
            Reference that provides the spatial attribute.
            Typically a nifti-related object from the native diffusion used for
            streamlines generation
        affine : array (4, 4), optional.
            The mapping between voxel indices and the point space for seeds.
            The voxel_to_rasmm matrix, typically from a NIFTI file.
            Default: np.eye(4)
        space : string, optional.
            Current space in which the streamlines are (vox, voxmm or rasmm)
            Typically after tracking the space is VOX, after nibabel loading
            the space is RASMM
            Default: Space.RASMM
        file_path : string, optional.
            Path to load trk files from.
            Default: './'
        file_suffix : string, optional.
            File name will be the fiber group name + file_suffix.
            Default: '.trk'
        bbox_valid_check : boolean, optional.
            Whether to verify that the bounding box is valid in voxel space.
            Default: False
        """
        for fg_name in fg_names:
            sft = load_tractogram(f'{file_path}{fg_name}{file_suffix}',
                                  reference,
                                  to_space=space,
                                  bbox_valid_check=bbox_valid_check)

            streamlines = dts.Streamlines(
                dtu.transform_tracking_output(sft.streamlines,
                                              affine))
            self.add_fiber_group(fg_name, streamlines)

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
            The mapping from voxel coordinates to streamline points.
            The voxel_to_rasmm matrix, typically from a NIFTI file.
            Default: np.eye(4)
        """
        for _, fiber_group in self.fiber_groups:
            weights = gaussian_weights(fiber_group['sl'])
            fiber_group['profile'] = afq_profile(data, fiber_group['sl'],
                                                 affine, weights=weights)