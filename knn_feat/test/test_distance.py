import numpy as np
import sys
import os
import pytest
from knn_feat import _distance
sys.path.append(os.getcwd())


@pytest.mark.success
def test_distance():
    actual = 5
    a = np.array([0, 0])
    b = np.array([3, 4])

    expected = _distance(a, b)
    assert expected == actual
