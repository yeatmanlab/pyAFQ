from scipy.linalg import blas
import numpy as np

from dipy.data import default_sphere
from dipy.reconst.shm import anisotropic_power

__all__ = ["extract_ODF", "anisotropic_index", "anisotropic_power"]


def extract_ODF(ODF, sphere=default_sphere, sh_order=8):
    """
    Calculates spherical harmonics coefficients and
    isotropic and anisotropic diffusion components
    from an ODF. Could include GFA in future updates.
    """
    ODF_norm = ODF / ODF.max()
    ASO = ODF_norm.max(axis=-1)
    ISO = ODF_norm.min(axis=-1)

    _, invB = shm.sh_to_sf_matrix(
        sphere, sh_order=sh_order, basis_type=None, return_inv=True)
    shm = blas.dgemm(
        alpha=1.,
        a=ODF.reshape(-1, invB.shape[0]), b=invB).reshape(
        (*ODF.shape[:-1], invB.shape[1]))

    return shm, ASO, ISO


def anisotropic_index(shm):
    """
    Calculates anisotropic index based on spherical harmonics coefficients.
    """
    sh_0 = shm[..., 0] ** 2
    sh_sum_squared = np.sum(shm ** 2, axis=-1)
    AI = np.zeros_like(sh_0)
    AI = np.sqrt(1 - sh_0 / sh_sum_squared)
    return AI
