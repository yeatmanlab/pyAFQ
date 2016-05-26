
import os.path as op

import dipy.data as dpd
import dipy.data.fetcher as fetcher


def test_segment():
    dpd.fetch_stanford_hardi()
    hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
    hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
    hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
    hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
    gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
    mapping = seg._register_to_template(hardi_fdata, gtab)
    streamlines = XXX
    fiber_groups = seg.segment(hardi_fdata, hardi_fbval, hardi_fbvec,
                               streamlines,
                               bundles=['ARC', 'UNC'],
                               mapping=mapping)
