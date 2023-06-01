from scipy.linalg import blas
import numpy as np

from dipy.data import default_sphere
from dipy.reconst.shm import anisotropic_power, sh_to_sf_matrix

__all__ = ["extract_odf", "anisotropic_index", "anisotropic_power"]


def extract_odf(odf, sphere=default_sphere, sh_order=8):
    """
    Calculates spherical harmonics coefficients and
    isotropic and anisotropic diffusion components
    from an ODF. Could include GFA in future updates.
    """
    odf_norm = odf / odf.max()
    ASO = odf_norm.max(axis=-1)
    ISO = odf_norm.min(axis=-1)

    _, invB = sh_to_sf_matrix(
        sphere, sh_order=sh_order, basis_type=None, return_inv=True)
    shm = blas.dgemm(
        alpha=1.,
        a=odf.reshape(-1, invB.shape[0]), b=invB).reshape(
        (*odf.shape[:-1], invB.shape[1]))

    return shm, ASO, ISO


def anisotropic_index(shm):
    """
    Calculates anisotropic index based on spherical harmonics coefficients.

    Code from Dmipy
    ---------------
    The MIT License (MIT)

    Copyright (c) 2017 Rutger Fick & Demian Wassermann

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    https://github.com/AthenaEPI/dmipy

    References
    ----------
    .. [1] Jespersen, Sune N., et al. "Modeling dendrite density from
        magnetic resonance diffusion measurements." Neuroimage 34.4 (2007):
        1473-1486.
    """
    sh_0 = shm[..., 0] ** 2
    sh_sum_squared = np.sum(shm ** 2, axis=-1)
    AI = np.zeros_like(sh_0)
    AI = np.sqrt(1 - sh_0 / sh_sum_squared)
    return AI
