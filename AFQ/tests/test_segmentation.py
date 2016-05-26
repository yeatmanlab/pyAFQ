
import os.path as op

import nibabel.tmpdirs as nbtmp

import dipy.data as dpd
import dipy.data.fetcher as fetcher


def test_segment():
    with nbtmp.InTemporaryDirectory() as tmpdir:
        MNI_T2 = dpd.read_mni_template()
        hardi_img, gtab = dpd.read_stanford_hardi()
        MNI_T2_data = MNI_T2.get_data()
        MNI_T2_affine = MNI_T2.get_affine()
        hardi_data = hardi_img.get_data()
        hardi_affine = hardi_img.get_affine()
        b0 = hardi_data[..., gtab.b0s_mask]
        mean_b0 = np.mean(b0, -1)
        subset_b0 = mean_b0[40:50, 40:50, 40:50]
        subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
        subset_b0_img = nib.Nifti1Image(subset_b0, hardi_affine)
        subset_t2_img = nib.Nifti1Image(subset_t2, MNI_T2_affine)
        dpd.fetch_stanford_hardi()
        hardi_dir = op.join(fetcher.dipy_home, "stanford_hardi")
        hardi_fdata = op.join(hardi_dir, "HARDI150.nii.gz")
        hardi_fbval = op.join(hardi_dir, "HARDI150.bval")
        hardi_fbvec = op.join(hardi_dir, "HARDI150.bvec")
        gtab = dpg.gradient_table(hardi_fbval, hardi_fbvec)
        mapping = seg._register_to_template(hardi_fdata, gtab)

        templates = dict()
        streamlines =

        fiber_groups = seg.segment(hardi_fdata, hardi_fbval, hardi_fbvec,
                                   streamlines,
                                   bundles=['FOO', 'BAR'],
                                   mapping=mapping)
