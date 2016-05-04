import numpy as np
import nibabel as nib
import dipy.reconst.dti as dti
import dipy.reconst.csdeconv as csd
import dipy.tracking.local as dtl
import dipy.tracking.utils as dtu
import dipy.direction as dpdir
import dipy.data as dpd

def csd_deterministic(params_file, max_angle=30., sphere=None,
                      seed_mask=None, seed_density=[2, 2, 2],
                      stop_mask=None, stop_threshold=0.2, step_size=0.5):
    """
    Deterministic tracking using CSD

    Parameters
    ----------
    params_file : str, nibabel img
        Full path to a nifti file containing CSD spherical harmonic
        coefficients, or nibabel img with model params.

    model : str
    """
    if isinstance(params_file, str):
        params_img = nib.load(params_file)
    else:
        params_img = params_file

    sh_coeff = params_img.get_data()
    affine = params_img.get_affine()

    if seed_mask is None:
        seed_mask = np.ones(params_img.shape[:3])
    seeds = dtu.seeds_from_mask(seed_mask, density=seed_density,
                                affine=affine)

    if sphere is None:
        sphere = dpd.default_sphere

    dg = dpdir.DeterministicMaximumDirectionGetter.from_shcoeff(
                                                sh_coeff,
                                                max_angle=30.,
                                                sphere=sphere)
    if stop_mask is None:
        stop_mask = np.ones(params_img.shape[:3])

    threshold_classifier = dtl.ThresholdTissueClassifier(stop_mask,
                                                         stop_threshold)

    streamlines = dtl.LocalTracking(dg, threshold_classifier,
                                    seeds, affine,
                                    step_size=step_size,
                                    return_all=True)
    return streamlines
