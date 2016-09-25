"""Common functions for ODE solvers."""
from __future__ import division, print_function, absolute_import

import numpy as np


EPS = np.finfo(float).eps


def norm(x):
    """Compute RMS norm."""
    return np.linalg.norm(x) / x.size ** 0.5


def select_initial_step(fun, a, b, ya, fa, order, rtol, atol):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    a : float
        Initial value of the independent variable.
    b : float
        Final value value of the independent variable.
    ya : ndarray, shape (n,)
        Initial value of the dependent variable.
    fa : ndarray, shape (n,)
        Initial value of the derivative, i. e. ``fun(x0, y0)``.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    if ya.size == 0:
        return np.inf

    scale = atol + np.abs(ya) * rtol
    d0 = norm(ya / scale)
    d1 = norm(fa / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    s = np.sign(b - a)
    y1 = ya + h0 * s * fa
    f1 = fun(a + h0 * s, y1)
    d2 = norm((f1 - fa) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / order)

    return min(100 * h0, h1)


class PointSpline:
    # scipy interpolators don't interpolate single points
    def __init__(self, t, y):
        self.t_start = t
        self.t_end = t
        self.t = t
        self.y = y

    def __call__(self, t):
        if np.isscalar(t):
            return self.y
        else:
            return np.tile(self.y, (len(t), 1)).T


class EmptySpline:
    def __init__(self, t_start, t_end):
        self.t_start = t_start
        self.t_end = t_end

    def __call__(self, t):
        if np.isscalar(t):
            return np.empty((0,))
        else:
            return np.empty((len(t), 0))


def validate_rtol(rtol):
    if rtol <= 0:
        raise ValueError("`rtol` must be positive.")

    return rtol


def validate_atol(atol, n):
    atol = np.asarray(atol)

    if atol.ndim > 0 and atol.shape != (n,):
        raise ValueError("`atol` has wrong shape.")

    if np.any(atol < 0):
        raise ValueError("`atol` must be positive.")

    return atol
