import numpy as np
import nibabel as nib
import dipy.core.gradients as dpg


def prepare_data(data_files, bval_files, bvec_files, mask=None,
                 b0_threshold=0):
    """
    Parameters
    ----------
    data_files : str or list
        Files containing DWI data. If this is a str, that's the full path to a
        single file. If it's a list, each entry is a full path.
    bval_files : str or list
        Equivalent to `data_files`.
    bvec_files : str or list
        Equivalent to `data_files`.
    mask : str or Nifti1Image object

    Returns
    -------
    img : Nifti1Image
    data : ndarray
    gtab : GradientTable
    mask : ndarray
    """
    types = [type(f) for f in [data_files, bval_files, bvec_files]]
    if len(set(types)) > 1:
        e_s = "Please provide consistent inputs to `prepare_data`. All file"
        e_s += " inputs should be either lists of full paths, or a string"
        e_s += " with one full path."
        raise ValueError(e_s)

    if isinstance(data_files, str):
        data_files = [data_files]
        bval_files = [bval_files]
        bvec_files = [bvec_files]

    # Load the mask if it is a string
    if isinstance(mask, str):
        mask = nib.load(mask).get_data().astype(bool)

    data = [nib.load(dfile).get_data() for dfile in data_files]
    img = nib.load(data_files[-1])
    bvals = [np.loadtxt(bval_file) for bval_file in bval_files]
    bvecs = [np.loadtxt(bvec_file) for bvec_file in bvec_files]

    data = np.concatenate(data, -1)
    gtab = dpg.gradient_table(np.concatenate(bvals),
                              np.concatenate(bvecs, -1),
                              b0_threshold=b0_threshold)

    return img, data, gtab, mask
