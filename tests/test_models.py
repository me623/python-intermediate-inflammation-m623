"""Tests for statistics functions within the Model layer."""
import pytest
import numpy as np
import numpy.testing as npt


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected)


@ pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
    ]
)
def test_daily_max_integers(test, expected):
    "test the max function works for an array of positive integers"
    from inflammation.models import daily_max

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test), expected)


@ pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
    ]
)
def test_daily_min_integers(test, expected):
    "test the max function works for an array of positive integers"
    from inflammation.models import daily_min

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        error_expected = daily_min([['abd', 'ads'], ['asd', 'auhs']])


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [
         [0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ]
)
def test_patient_normalise(test, expected):
    '''Test normalisations works'''
    from inflammation.models import patient_normalise
    npt.assert_almost_equal(patient_normalise(
        np.array(test)), np.array(expected), decimal=2)
