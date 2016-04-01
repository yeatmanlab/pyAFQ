import os
import os.path as op

import numpy as np
import nibabel as nib

from dipy.reconst import dki
from dipy.core import gradients as dpg


def fit_dki(data_files, bval_files, bvec_files, mask=None, min_kurtosis=0,
            max_kurtosis=3, out_dir=None):
    """
    Fit the DKI model, save files with derived maps

    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.
    mask : ndarray, optional
        Binary mask, set to True or 1 in voxels to be processed.
        Default: Process all voxels.
    min_kurtosis : float, optional
        The minimal plausible value of kurtosis. Default: 0.
    max_kurtosis : float, optional
        The maximal plausible value of kurtosis. Default: 3.
    out_dir : str, optional
        A full path to a directory to store the maps that get computed.
        Default: maps get stored in the same directory as the last DWI file in
        `data_files`.

    Returns
    -------
    file_paths : a dict with the derived maps that were computed and full-paths
    to the files containing these maps.

    Note
    ----
    Maps that are calculated: FA, MD, AD, RD, MK, AK, RK

    """
    types = [type(f) for f in [data_files, bval_files, bvec_files]]
    if len(set(types)) > 1:
        e_s = "Please provide consistent inputs to `fit_dki`. All file"
        e_s += " inputs should be either lists of full paths, or a string"
        e_s += " with one full path."
        raise ValueError(e_s)

    if isinstance(data_files, str):
        data_files = [data_files]
        bval_files = [bval_files]
        bvec_files = [bvec_files]

    # Load the mask if it is a string
    if isinstance(mask, str):
        mask = nib.load(mask).get_data()

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
    MK = dkifit.mk(min_kurtosis, max_kurtosis)
    AK = dkifit.ak(min_kurtosis, max_kurtosis)
    RK = dkifit.rk(min_kurtosis, max_kurtosis)

    maps = [FA, MD, AD, RD, MK, AK, RK]
    names = ['FA', 'MD', 'AD', 'RD', 'MK', 'AK', 'RK']

    if out_dir is None:
        out_dir = op.join(op.split(dfile)[0], 'dki')

    if not op.exists(out_dir):
        os.makedirs(out_dir)

    aff = img.get_affine()
    file_paths = {}
    for m, n in zip(maps, names):
        file_paths[n] = op.join(out_dir, 'dki_%s.nii.gz' % n)
        nib.save(nib.Nifti1Image(m, aff), file_paths[n])

    return file_paths
