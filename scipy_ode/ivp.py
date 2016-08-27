"""Generic interface for initial value problem solvers."""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.optimize import brentq

from .common import EPS
from .solver import SolverStatus, SolverDirection, IntegrationException
from .rk import RungeKutta45


def solve_ivp(t0, tF, y0, fun, *, method=RungeKutta45, events=None, **options):
    """Solve an initial value problem for a system of ODEs.

    This function numerically integrates a system of ODEs given an initial
    value until a terminal value:

        dy / dt = fun(t, y)
        y(t0) = y0

    Here t is a scalar independent variable with initial value t0, y(t) is a
    vector-valued function with initial value y0.

    Parameters
    ----------
    t0 : float
        The initial value of ``t``
    tF : float
        The value of ``t`` at which to stop integration
    y0 : array_like, shape (n,)
        Initial values for ``y``
        # TODO: allow for scalar y?
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``.
        Here ``x`` is a scalar, and ``y`` is ndarray with shape (n,). It
        must return an array_like with shape (n,).
    method : subclass of OdeSolver, optional
        The class of solver to use to integrate the system. Use one of the built-in solvers
        named below or see the base class `OdeSolver` on how to implement one yourself:
            * ``RungeKutta45`` (default): Explicit Runge-Kutta method of order 5 with an
              automatic step size control [1]_. A 4-th order accurate quartic
              polynomial is used for the continuous extension [2]_.
            * ``RungeKutta23``: Explicit Runge-Kutta method of order 3 with an automatic
              step size control [3]_. A 3-th order accurate cubic Hermit
              polynomial is used for the continuous extension.
            * ``Radau``: Implicit Runge-Kutta method of Radau IIA family of
              order 5 [4]_. A 5-th order accurate cubic polynomial is available
              naturally as the method can be viewed as a collocation method.

        You should use ``RungeKutta45`` or ``RungeKutta23`` methods for non-stiff problems and
        ``Radau`` for stiff problems [5]_. If not sure, first try to run ``RungeKutta45``
        and if it does unusually many iterations or diverges then your problem
        is likely to be stiff and you should use ``Radau``.
    events : callable, list of callables or None, optional
        Events to track. Events are defined by functions which take
        a zero value at a point of an event. Each function must have a
        signature ``event(x, y)`` and return float, the solver will find an
        accurate value of ``x`` at which ``event(x, y(x)) = 0`` using a root
        finding algorithm. Additionally each ``event`` function might have
        attributes:

            * terminate: bool, whether to terminate integration if this
              event occurs. Implicitly False if not assigned.
              # TODO: should this be "terminal"
            * direction: float, direction of crossing a zero. If `direction`
              is positive then `event` must go from negative to positive, and
              vice-versa if `direction` is negative. If 0, then either way will
              count. Implicitly 0 if not assigned.
        You can assign attributes like ``event.terminate = True`` to any
        function in Python. If None (default), events won't be tracked.
    options
        Additional keyword options passed to the solver. All options understood by any built-in
        solver are listed below. Not all options are understood by all solvers.
        See the individual solver docs for more information.
    * rtol, atol : float and array_like, optional
        Relative and absolute tolerances. The solver keeps the error estimates
        less than ``atol` + rtol * abs(y)``. Here `rtol` controls a relative
        accuracy (number of correct digits). But if a component of `y` is
        approximately below `atol` then the error only needs to fall within
        the same `atol` threshold, and the number of correct digits is not
        guaranteed. If components of y have different scales, it might be
        beneficial to set different `atol` values for different components by
        passing array_like with shape (n,) for `atol`. Default values are
        1e-3 for `rtol` and 1e-6 for `atol`.
    * max_step : float, optional
        Maximum allowed step size. Default is infinity.
    * jac : array_like, callable or None, optional
        Jacobian matrix of the right-hand side of the system with respect to
        ```y`, required only by ``Radau`` method. The Jacobian matrix has shape
        (n, n) and its element (i, j) is equal to ``d f_i / d y_j``.
        There are 3 ways to define the Jacobian:

            * If array_like, then the Jacobian is assumed to be constant.
            * If callable, then the Jacobian is assumed to depend on both
              x and y, and will be called as ``jac(t, y)`` as necessary.
            * If None (default), then the Jacobian will be approximated by
              finite differences.

        It is generally recommended to provided the Jacobian rather then
        relying on finite difference approximation.

    Returns
    -------
    ``OdeSolution`` object with the following fields defined:
    __call__ : callable, (t) -> y
        Returns the values of the solution at the times given by ``t``. If ``t`` is a
        scalar, then ``y`` is a vector of shape (n,). If ``t`` is a vector of shape
        (nt,), then ``y`` is a matrix of shape (nt,n). All values in ``t`` must be between
        ``t0`` and ``tF``.
        # TODO: should y be scalar when fun returns a scalar?
    n : int
        Size of system
    t0, tF : float
        Initial and final time. The final time is the time at which the solver actually
        stopped, whether the final time provided or the time at which a terminal
        event was encountered.
    t_events : array or list of arrays
        If events was callable, this is an array providing the times at which
        the event was detected.
        If events was a list of callables, this is a list of the same length
        where each element is the times at the which the corresponding event
        was detected.
        If events was None, this is None.
        # TODO: do we want to make this just two arrays, one for the indexes
        # and one for the times

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

    scalar_events = callable(events)
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

    if scalar_events:
        # Convert to list rather than list of lists when events is scalar
        x_events = x_events[0]

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
        self._spline = spline
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

        result = self._spline(t).T  # len(t) rows and
        return result[..., iy]


def prepare_events(events):
    if callable(events):
        events = (events,)

    if events is not None:
        is_terminal = np.empty(len(events), dtype=bool)
        direction = np.empty(len(events))
        for i, event in enumerate(events):
            try:
                is_terminal[i] = event.terminate
            except AttributeError:
                is_terminal[i] = False

            try:
                direction[i] = event.direction
            except AttributeError:
                direction[i] = 0
    else:
        is_terminal = None
        direction = None

    return events, is_terminal, direction


def get_active_events(g, g_new, direction):
    """Find which event occurred during an integration step.

    Parameters
    ----------
    g, g_new : array_like, shape (n_events,)
        Values of event functions at a current and next points.
    direction : ndarray, shape (n_events,)
        Event "direction" according to definition in `solve_ivp`.

    Returns
    -------
    active_events : ndarray
        Indices of events which occurred during the step.
    """
    g, g_new = np.asarray(g), np.asarray(g_new)
    up = (g <= 0) & (g_new >= 0)
    down = (g >= 0) & (g_new <= 0)
    either = up | down
    mask = (up & (direction > 0) |
            down & (direction < 0) |
            either & (direction == 0))

    return np.nonzero(mask)[0]


def handle_events(sol, events, active_events, is_terminal, x, x_new):
    """Helper function to handle events.

    Parameters
    ----------
    sol : callable
        Function ``sol(x)`` which evaluates an ODE solution.
    events : list of callables, length n_events
        Event functions.
    active_events : ndarray
        Indices of events which occurred
    is_terminal : ndarray, shape (n_events,)
        Which events are terminate.
    x, x_new : float
        Previous and new values of the independed variable, it will be used as
        a bracketing interval.

    Returns
    -------
    root_indices : ndarray
        Indices of events which take zero before a possible termination.
    roots : ndarray
        Values of x at which events take zero values.
    terminate : bool
        Whether a termination event occurred.
    """
    roots = []
    for event_index in active_events:
        roots.append(solve_event_equation(events[event_index], sol, x, x_new))

    roots = np.asarray(roots)

    if np.any(is_terminal[active_events]):
        if x_new > x:
            order = np.argsort(roots)
        else:
            order = np.argsort(-roots)
        active_events = active_events[order]
        roots = roots[order]
        t = np.nonzero(is_terminal[active_events])[0][0]
        active_events = active_events[:t + 1]
        roots = roots[:t + 1]
        terminate = True
    else:
        terminate = False

    return active_events, roots, terminate


def solve_event_equation(event, sol, x, x_new):
    """Solve an equation corresponding to an ODE event.

    The equation is ``event(x, y(x)) = 0``, here ``y(x)`` is known from an
    ODE solver using some sort of interpolation. It is solved by
    `scipy.optimize.brentq` with xtol=atol=4*EPS.

    Parameters
    ----------
    event : callable
        Function ``event(x, y)``.
    sol : callable
        Computed solution ``y(x)``. It should be defined only between `x` and
        `x_new`.
    x, x_new : float
        Previous and new values of the independed variable, it will be used as
        a bracketing interval.

    Returns
    -------
    root : float
        Found solution.
    """
    return brentq(lambda t: event(t, sol(t)), x, x_new, xtol=4 * EPS)


