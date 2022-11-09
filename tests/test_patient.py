#Tests for the Patient model.
import pytest
import numpy as np
import numpy.testing as npt

"""
@pytest.mark.parameterize(
    "test, expected, expected_raise",
    [
        ("Alice", "Alice", None),
        (0, 0, TypeError),
        (0.24, 0.24, TypeError)
    ]
)
def test_create_patient(test, expected, expect_raises):
    from inflammation.models import Patient, Person

    if expect_raises is not None:
        with pytest.raises(expect_raises):
            name = test
            test_patient = Patient(Person(name=name))
            assert test_patient.name == expected

    name = test
    test_patient = Patient(Person(name=name))
    assert test_patient.name == expected
"""