
import numpy as np


def gaussian_penalty(a, b, r_x, s_x, s_y):
    """
    Gaussian style penalty - more localized effect
    """
    return a*s_y * np.exp(-np.sum((r_x / (b * s_x)) ** 2, axis=1) / np.sqrt(2))


def invquad_penalty(a, b, r_x, s_x, s_y):
    """
    Inverse quadratic penalty - less localized effect
    """
    return a*s_y / (1 + np.sum((r_x / (b * s_x)) ** 2, axis=1))

