import numpy as np
import numpy.testing as npt

import nibabel as nib

import dipy.data as dpd
import dipy.core.gradients as dpg

from AFQ.registration import (syn_registration, register_series,
                              c_of_mass, translation, rigid, affine)


def test_syn_registration():
    MNI_T2 = dpd.read_mni_template()
    ni, gtab = dpd.read_stanford_hardi()
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()
    hardi_data = ni.get_data()
    hardi_affine = ni.get_affine()
    b0 = hardi_data[..., gtab.b0s_mask]
    mean_b0 = np.mean(b0, -1)

    # We select some arbitrary chunk of data so this goes quicker:
    subset_b0 = mean_b0[40:50, 40:50, 40:50]
    subset_t2 = MNI_T2_data[40:60, 40:60, 40:60]
    warped_moving, mapping = syn_registration(subset_b0,
                                              subset_t2,
                                              moving_affine=hardi_affine,
                                              static_affine=MNI_T2_affine,
                                              step_length=0.1,
                                              metric='CC',
                                              dim=3,
                                              level_iters=[10, 10, 5],
                                              sigma_diff=2.0,
                                              prealign=None)

    npt.assert_equal(warped_moving.shape, subset_t2.shape)


def test_register_series():
    fdata, fbval, fbvec = dpd.get_data('small_64D')
    img = nib.load(fdata)
    gtab = dpg.gradient_table(fbval, fbvec)
    transformed_list, affine_list = register_series(
                                        img,
                                        ref=np.where(gtab.b0s_mask),
                                        pipeline=[c_of_mass,
                                                  translation,
                                                  rigid,
                                                  affine])
