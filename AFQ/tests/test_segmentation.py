
import os.path as op

import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.data.fetcher as fetcher

import AFQ.data as afd

def test_segment():
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    streamlines = afd.read_stanford_tracks()
    fiber_groups = seg.segment(hardi_fdata, hardi_fbval, hardi_fbvec,
                               streamlines)
