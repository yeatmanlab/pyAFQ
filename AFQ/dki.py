import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import dki
from dipy.core import gradients as dpg


def fit_dki(data_files, bval_files, bvec_files, mask=None, min_d=0, max_d=3,
            out_dir=None):
    """
    Fit the DKI model, save files with parameters and derived maps

    Parameters
    ----------
    data_files : str or list
        if str, that's the full path
        if list, each entry is a full path
    bval_files : str or list
        ditto
    bvec_files : str or list
        ditto
    """
    # XXX Check that inputs are uniform

    if isinstance(data_files, str):
        data_files = [data_files]
        bval_files = [bval_files]
        bvec_files = [bvec_files]

    data = []
    bvals = []
    bvecs = []
    for dfile, bval_file, bvec_file in zip(data_files, bval_files, bvec_files):
        img = nib.load(dfile)
        data.append(img.get_data())
        bvals.append(np.loadtxt(bval_file))
        bvecs.append(np.loadtxt(bvec_file))

    data = np.concatenate(data, -1)
    gtab = dpg.gradient_table(np.concatenate(bvals),
                              np.concatenate(bvecs, -1))

    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data, mask=mask)

    FA = dkifit.fa
    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd
    MK = dkifit.mk(min_d, max_d)
    AK = dkifit.ak(min_d, max_d)
    RK = dkifit.rk(min_d, max_d)

    maps = [FA, MD, AD, RD, MK, AK, RK]
    names = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK']

    if out_dir is None:
        out_dir = op.join(op.split(dfile)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)


    aff = img.get_affine()
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dki_%s.nii.gz'%n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths
