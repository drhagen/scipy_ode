"""Generic interface for initial value problem solvers."""
from __future__ import division, print_function, absolute_import

import numpy as np

from .common import get_active_events, handle_events, prepare_events
from .solver import SolverStatus, SolverDirection, IntegrationException
from .rk import RungaKutta45


def ivp_solution(t0, tF, y0, fun, *, method=RungaKutta45, events=None, **options):
    """Solve an initial value problem for a system of ODEs.

    This function numerically integrates a system of ODEs given an initial
    value::

        dy / dx = f(x, y)
        y(a) = ya

    Here x is a 1-dimensional independent variable, y(x) is a n-dimensional
    vector-valued function and ya is a n-dimensional vector with initial
    values.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``.
        Here ``x`` is a scalar, and ``y`` is ndarray with shape (n,). It
        must return an array_like with shape (n,).
    x_span : 2-tuple of floats
        Interval of integration (a, b). The solver starts with x=a and
        integrates until it reaches x=b.
    ya : array_like, shape (n,)
        Initial values for y.
    rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the error estimates
        less than ``atol` + rtol * abs(y)``. Here `rtol` controls a relative
        accuracy (number of correct digits). But if a component of `y` is
        approximately below `atol` then the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    method : string, optional
        Integration method to use:

            * 'RK45' (default): Explicit Runge-Kutta method of order 5 with an
              automatic step size control [1]_. A 4-th order accurate quartic
              polynomial is used for the continuous extension [2]_.
            * 'RK23': Explicit Runge-Kutta method of order 3 with an automatic
              step size control [3]_. A 3-th order accurate cubic Hermit
              polynomial is used for the continuous extension.
            * 'Radau': Implicit Runge-Kutta method of Radau IIA family of
              order 5 [4]_. A 5-th order accurate cubic polynomial is available
              naturally as the method can be viewed as a collocation method.

        You should use 'RK45' or 'RK23' methods for non-stiff problems and
        'Radau' for stiff problems [5]_. If not sure, first try to run 'RK45'
        and if it does unusual many iterations or diverges then your problem
        is likely to be stiff and you should use 'Radau'.
    max_step : float or None, optional
        Maximum allowed step size. If None, a step size is selected to be 0.1
        of the length of `x_span`.
    jac : array_like, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect to
        `y`, required only by 'Radau' method. The Jacobian matrix has shape
        (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.
        There are 3 ways to define the Jacobian:

            * If array_like, then the Jacobian is assumed to be constant.
            * If callable, then the Jacobian is assumed to depend on both
              x and y, and will be called as ``jac(x, y)`` as necessary.
            * If None (default), then the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provided the Jacobian rather then
        relying on finite difference approximation.
    events : callable, list of callables or None, optional
        Events to track. Events are defined by functions which take
        a zero value at a point of an event. Each function must have a
        signature ``event(x, y)`` and return float, the solver will find an
        accurate value of ``x`` at which ``event(x, y(x)) = 0`` using a root
        finding algorithm. Additionally each ``event`` function might have
        attributes:

            * terminate: bool, whether to terminate integration if this
              event occurs. Implicitly False if not assigned.
            * direction: float, direction of crossing a zero. If `direction`
              is positive then `event` must go from negative to positive, and
              vice-versa if `direction` is negative. If 0, then either way will
              count. Implicitly 0 if not assigned.

        You can assign attributes like ``event.terminate = True`` to any
        function in Python. If None (default), events won't be tracked.

    Returns
    -------
    Bunch object with the following fields defined:
    sol : PPoly
        Found solution for y as `scipy.interpolate.PPoly` instance, a C1
        continuous spline.
    x : ndarray, shape (n_points,)
        Values of the independent variable at which the solver made steps.
    y : ndarray, shape (n, n_points)
        Solution values at `x`.
    yp : ndarray, shape (n, n_points)
        Solution derivatives at `x`, i.e. ``fun(x, y)``.
    x_events : ndarray, tuple of ndarray or None
        Arrays containing values of x at each corresponding events was
        detected. If `events` contained only 1 event, then `x_events` will
        be ndarray itself. None if `events` was None.
    status : int
        Reason for algorithm termination:

            * -1: Required step size became too small.
            * 0: The solver successfully reached the interval end.
            * 1: A termination event occurred.

    message : string
        Verbal description of the termination reason.
    success : bool
        True if the solver reached the interval end or a termination event
        (``status >= 0``).

    References
    ----------
    .. [1] J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
           formulae", Journal of Computational and Applied Mathematics, Vol. 6,
           No. 1, pp. 19-26, 1980.
    .. [2] L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics
           of Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
    .. [3] P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
    .. [4] E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II:
           Stiff and Differential-Algebraic Problems", Sec. IV.8.
    .. [5] `Stiff equation <https://en.wikipedia.org/wiki/Stiff_equation>`_ on
           Wikipedia.
    """

    solver = method(t0, y0, fun, t_final=tF, **options)
    events, is_terminal, direction = prepare_events(events)

    n = solver.n
    x = solver.t
    y = solver.y

    if events is not None:
        g = [event(x, y) for event in events]
        x_events = [[] for _ in range(len(events))]
    else:
        x_events = None

    states = [solver.state]

    while solver.status == SolverStatus.running or solver.status == SolverStatus.started:
        solver.step()

        if solver.status == SolverStatus.failed:
            sol = OdeSolution(t0, solver.t, n, solver.spline(states))
            raise IntegrationException("Step size has fallen below floating point precision", solver.t, sol)

        x_new = solver.t
        y_new = solver.y
        states.append(solver.state)

        if events is not None:
            g_new = [event(x_new, y_new) for event in events]
            active_events = get_active_events(g, g_new, direction)
            g = g_new
            if active_events.size > 0:
                sol = solver.spline(states[-2:])
                root_indices, roots, terminate = handle_events(
                    sol, events, active_events, is_terminal, x, x_new)

                for e, xe in zip(root_indices, roots):
                    x_events[e].append(xe)

                if terminate:
                    tF = roots[-1]
                    break

        x = x_new
        y = y_new

    return OdeSolution(t0, tF, n, solver.spline(states), x_events)


class OdeSolution:
    def __init__(self, t0, tF, n, spline, t_events):
        self.n = n
        if t0 <= tF:
            self.s = SolverDirection.forward
        else:
            self.s = SolverDirection.reverse
        self.t0 = t0
        self.tF = tF
        self.spline = spline
        self.t_events = t_events

    def __call__(self, t, iy=Ellipsis):
        def check_time(ti):
            if self.s == SolverDirection.forward and (ti < self.t0 or ti > self.tF) or \
                    self.s == SolverDirection.reverse and (ti > self.t0 or ti < self.tF):
                raise ValueError("Requested time {} is outside the solution interval [{}, {}]"
                                 .format(ti, self.t0, self.tF))

        if np.isscalar(t):
            check_time(t)
        else:
            for item in t:
                check_time(item)

        result = self.spline(t).T  # len(t) rows and
        return result[..., iy]
