"""
Module: modules
Description:
    This module contains the implementation of the linear regression model.
Author: Oleg Tsupa
"""

import pytest

from src.models import model


@pytest.mark.parametrize(
        "value, weight, bias, expected",
        [
            (2, 3, 4, 10),          # Example from docstring.
            (0, 5, 10, 10),         # Zero value should return bias
            (1, 0, 5, 5),           # Zero weight should retur bias
            (3, 2, 1 , 7),          # General case
            (-1, 4, 2, -2),         # Negaative value 
            (2.5, 1.5, 0.5, 4.25),  # Float values
        ]
)
def test_model(value, weight, bias, expected):
    """Test linear regression model."""
    assert model(value, weight, bias) == expected