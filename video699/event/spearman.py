# -*- coding: utf-8 -*-

r"""This module implements a screen event detector that matches document page image data with
projection screen image data using the Spearman's rank correlation coefficient :math:`\rho` with a
sliding window of video frames. Related classes and methods are also implemented.

"""

import numpy as np


def benjamini_hochberg(p_values):
    """Adjusts p-values from independent hypothesis tests to q-values.

    The q-values are determined using the false discovery rate (FDR) controlling procedure of
    Benjamini and Hochberg [BenjaminiHochberg1995]_.

    .. [BenjaminiHochberg1995] Benjamini, Yoav; Hochberg, Yosef (1995). "Controlling the false
       discovery rate: a practical and powerful approach to multiple testing". Journal of the
       Royal Statistical Society, Series B. 57 (1): 289–300. MR 1325392.

    Notes
    -----
    This method was adapted from `code posted to Stack Overflow by Eric Talevich`_.

    .. _code posted to Stack Overflow by Eric Talevich: https://stackoverflow.com/a/33532498/657401

    Parameters
    ----------
    p_values : iterable of scalar
        p-values from independent hypothesis tests.

    Returns
    -------
    q_values : iterable of scalar
        The p-values adjusted using the FDR controlling procedure.
    """
    p_value_array = np.asfarray(p_values)
    num_pvalues = len(p_value_array)
    descending_order = p_value_array.argsort()[::-1]
    original_order = descending_order.argsort()
    steps = num_pvalues / np.arange(num_pvalues, 0, -1)
    descending_q_values = np.minimum.accumulate(steps * p_value_array[descending_order]).clip(0, 1)
    q_values = descending_q_values[original_order]
    return q_values
