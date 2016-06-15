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
    bundles = {'CST_L':{'ROIs': [templates['CST_roi1_L'],
                                 templates['CST_roi2_L']],
                        'rules': [True, True]},
               'CST_R':{'ROIs': [templates['CST_roi1_R'],
                                 templates['CST_roi1_R']],
                        'rules': [True, True]}}

    fiber_groups = seg.segment(hardi_fdata,
                               hardi_fbval,
                               hardi_fbvec,
                               streamlines,
                               bundles,
                               mapping=mapping,
                               as_generator=True)

    # We asked for 2 fiber groups:
    npt.assert_equal(len(fiber_groups), 2)
    # There happen to be 8 fibers in the right CST:
    CST_R_sl = list(fiber_groups['CST_R'])
    CST_L_sl = list(fiber_groups['CST_L'])
    npt.assert_equal(len(CST_R_sl), 8)
