import os.path as op
import numpy as np

import nibabel as nib
import dipy.data as dpd

from AFQ.registration import syn_registration
from AFQ.utils.testing import make_dti_data


def test_syn_registration():
    MNI_T2 = dpd.read_mni_template()
    ni, gtab = dpd.read_stanford_hardi()
    MNI_T2_data = MNI_T2.get_data()
    MNI_T2_affine = MNI_T2.get_affine()
    hardi_data = ni.get_data()
    hardi_affine = ni.get_affine()
    b0 = hardi_data[..., gtab.b0s_mask]
    mean_b0 = np.mean(b0, -1)
    syn_registration(mean_b0, MNI_T2_data,
                     moving_affine=hardi_affine,
                     static_affine=MNI_T2_affine,
                     step_length=0.1,
                     metric='CC',
                     dim=3,
                     level_iters=[10, 10, 5],
                     sigma_diff=2.0,
                     prealign=None)
