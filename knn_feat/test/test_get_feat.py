import numpy as np
import sys
import os
import pytest
from knnFeat import _get_feat
sys.path.append(os.getcwd())


# case 1: class_index == 0 and k_index == 0
@pytest.mark.success
def test_get_feat_c0k0():
    data = np.array([0, 0])

    x_train = np.reshape(np.array([0, 1, 3, 4, 5
                                 , 6, 1, 1, 0, 3])
                         , (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])

    class_index = 0
    k_index = 0

    expected = _get_feat(data, x_train, y_train
                         , class_index, k_index)

    actual = 1

    assert expected == actual


# case 2: class_index == 0 and k_index == 1
@pytest.mark.success
def test_get_feat_c0k1():
    data = np.array([0, 0])
    x_train = np.reshape(np.array([0, 1, 3, 4, 5
                                 , 6, 1, 1, 0, 3])
                         , (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])

    class_index = 0
    k_index = 1

    expected = _get_feat(data, x_train, y_train
                         , class_index, k_index)

    actual = 1 + 5

    assert expected == actual


# case 3: class_index == 1 and k_index == 1
@pytest.mark.success
def test_get_feat_c0k1():
    data = np.array([0, 0])
    x_train = np.reshape(np.array([0, 1, 3, 4, 5
                                      , 6, 1, 1, 0, 3])
                         , (5, 2))
    y_train = np.array([0, 0, 0, 1, 1])

    class_index = 1
    k_index = 1

    expected = _get_feat(data, x_train, y_train
                         , class_index, k_index)

    actual = 2 + 3

    assert expected == actual


