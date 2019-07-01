from AFQ.dti import in_place_norm
from dipy.core.geometry import vector_norm
import numpy.testing as npt

def test_inplace_norm():
    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec)
    norm2 = in_place_norm(vec)
    npt.assert_equal(norm1, norm2)
    
    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec, keepdims=True)
    norm2 = in_place_norm(vec, keepdims=True)
    npt.assert_equal(norm1, norm2)

    vec = [[8, 15, 0], [0, 36, 77]]
    norm1 = vector_norm(vec, axis=0)
    norm2 = in_place_norm(vec, axis=0)
    npt.assert_equal(norm1, norm2)