import os.path as op

import numpy.testing as npt

import dipy.data as dpd
import dipy.data.fetcher as fetcher

import AFQ.data as afd
import AFQ.segmentation as seg


def test_segment():
    dpd.fetch_stanford_hardi()
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    file_dict = afd.read_stanford_hardi_tractograpy()
    mapping = file_dict['mapping.nii.gz']
    streamlines = file_dict['tractography_subsampled.trk']
    templates = afd.read_templates()
    bundles = {'ARC_L':{'ROIs': [templates['SLF_roi1_L'],
                                 templates['SLFt_roi2_L']],
                        'rules': [True, True]}}

    fiber_groups = seg.segment(hardi_fdata,
                               hardi_fbval,
                               hardi_fbvec,
                               streamlines,
                               bundles,
                               mapping=mapping,
                               as_generator=False)

    npt.assert_equal(len(fiber_groups), 1)
