"""
Module: modules
Description:
    This module contains the implementation of the linear regression model.
Author: Oleg Tsupa
"""

from typing import Union


def model(
    value: Union[int, float], weight: Union[int, float], bias: Union[int, float]
) -> Union[int, float]:
    """Linear regression model

    Parametres
    ----------
    value : Union[int, float]
        Value to be predicted
    weight : Union[int, float]
        Weight of the model
    bias Union[int, float]
        Bias of the model

    Returns
    -------
        Union[int, float]
            Predicted value.

    Examples
    --------
    >>> model (2, 3, 4)
    10

    Notes
    -----
    The linear regression model is defined as:
    y = wx + b

    where w is the weight, x is the value, and b is the bias
    """
    return weight * value + bias