from __future__ import division, print_function, absolute_import

from enum import Enum, IntEnum

import numpy as np
from scipy.interpolate import interp1d


class OdeSolver:
    class OdeState:
        def __init__(self, t, y):
            # TODO: decide whether to use (t,y), (t,x), (x,y) as names
            self.t = t
            self.y = y

    def __init__(self, state, fun, t_final):
        self.state = state

        t0 = self.t
        y0 = self.y

        self.n = y0.size

        self.fun = fun
        self.t_final = t_final
        if t_final - t0 >= 0:
            self.direction = SolverDirection.forward
        else:
            self.direction = SolverDirection.reverse

        if t0 != t_final:
            self.status = SolverStatus.started
        else:
            self.status = SolverStatus.finished

    @staticmethod
    def check_arguments(t0, t_final, y0, fun):
        y0 = np.asarray(y0, dtype=float)  # TODO: give user control over dtype?

        if y0.ndim != 1:
            raise ValueError("`y0` must be 1-dimensional.")

        def fun_wrapped(t, y):
            # TODO: decide if passing args and kwargs should be supported f(self, t, y, *args, **kwargs)
            return np.asarray(fun(t, y))

        return t0, t_final, y0, fun_wrapped

    @property
    def t(self):
        return self.state.t

    @property
    def y(self):
        return self.state.y

    def step(self):
        raise NotImplementedError()

    def spline(self, states):
        x = np.asarray([state.x for state in states])
        y = np.asarray([state.y for state in states])

        return interp1d(x, y, assume_sorted=True, kind='cubic')  # TODO: determine the best default interpolator


class SolverDirection(IntEnum):
    reverse = -1
    forward = 1


class SolverStatus(Enum):
    started = object()
    running = object()
    # TODO: add message to failure
    failed = object()
    finished = object()


class IntegrationException(Exception):
    def __init__(self, message, t, partial_solution):
        super().__init__("Integration failure at t = {}: {}".format(t, message))
        self.partial_solution = partial_solution
