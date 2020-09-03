import numpy as np
import sys
import os
import pytest
from knn_feat import knnExtract
sys.path.append(os.getcwd())


@pytest.mark.success
def test_knnFeat():
    x = np.reshape(np.array([0, 1, 3, 4, 5
                            , 6, 1, 1, 0, 3])
                   , (5, 2))
    y = np.array([0, 0, 0, 1, 1])

    expected = knnExtract(x, y, k=1, folds=5)

    assert expected.shape == x.shape

