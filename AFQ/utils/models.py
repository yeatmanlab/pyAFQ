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
        mask = nib.load(mask).get_fdata().astype(bool)

    data = []
    bvals = []
    bvecs = []
    for dfile, bval_file, bvec_file in zip(data_files, bval_files, bvec_files):
        img = nib.load(dfile)
        data.append(img.get_fdata())
        bvals.append(np.loadtxt(bval_file))
        bvecs.append(np.loadtxt(bvec_file))

    data = np.concatenate(data, -1)
    gtab = dpg.gradient_table(np.concatenate(bvals),
                              np.concatenate(bvecs, -1),
                              b0_threshold=b0_threshold)

    return img, data, gtab, mask
