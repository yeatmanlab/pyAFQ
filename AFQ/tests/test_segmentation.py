
import os.path as op
import numpy as np

import nibabel as nib
import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.data.fetcher as fetcher
import dipy.core.gradients as dpg

import AFQ.data as afd
import AFQ.utils.streamlines as sl
import AFQ.segmentation as seg


def test_segment():
    dpd.fetch_stanford_hardi()
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    file_dict = afd.read_stanford_hardi_tractograpy()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']

    fiber_groups = seg.segment(hardi_fdata,
                               hardi_fbval,
                               hardi_fbvec,
                               streamlines,
                               mapping=mapping)
