from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_, assert_allclose, run_module_suite,
                           assert_equal, assert_raises)
from scipy_ode import SolverStatus, solve_ivp, RungeKutta23, RungeKutta45, Radau


all_methods = [RungeKutta23, RungeKutta45, Radau]


def fun_rational(t, y):
    return np.array([y[1] / t,
                     y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))])


def jac_rational(t, y):
    return np.array([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def sol_rational(t):
    return np.vstack((t / (t + 10), 10 * t / (t + 10) ** 2)).T


def event_rational_1(t, y):
    return y[0] - y[1] ** 0.7


def event_rational_2(t, y):
    return y[1] ** 0.6 - y[0]


def event_rational_3(t, y):
    return t - 7.4


def test_integration():
    rtol = 1e-3
    atol = 1e-6
    for method in all_methods:
        for t_span in ([5, 9], [5, 1]):
            res = solve_ivp(fun_rational, [1 / 3, 2 / 9], t_span[0], t_span[1], rtol=rtol, atol=atol, method=method)
            assert_equal(res.t0, t_span[0])
            assert_equal(res.tF, t_span[-1])
            assert_(res.t_events is None)

            tc = np.linspace(*t_span)
            yc_true = sol_rational(tc)
            yc = res(tc)
            assert_allclose(yc, yc_true, rtol=1e-2)


def test_events():
    event_rational_3.terminate = True

    for method in all_methods:
        res = solve_ivp(fun_rational, [1 / 3, 2 / 9], 5, 8, method=method, events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = solve_ivp(fun_rational, [1 / 3, 2 / 9], 5, 8, method=method, events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = solve_ivp(fun_rational, [1 / 3, 2 / 9], 5, 8, method=method, events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = solve_ivp(fun_rational, [1 / 3, 2 / 9], 5, 8, method=method, events=(event_rational_1, event_rational_2,
                                                                                   event_rational_3))
        assert_allclose(res.tF, 7.4)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_equal(len(res.t_events[2]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)

        # Also test that termination by event doesn't break interpolants.
        tc = np.linspace(res.t0, res.tF)
        yc_true = sol_rational(tc)
        yc = res(tc)
        assert_allclose(yc, yc_true, rtol=1e-2)

    # Test in backward direction.
    event_rational_1.direction = 0
    event_rational_2.direction = 0
    for method in all_methods:
        res = solve_ivp(fun_rational, [4 / 9, 20 / 81], 8, 5, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = solve_ivp(fun_rational, [4 / 9, 20 / 81], 8, 5, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = solve_ivp(fun_rational, [4 / 9, 20 / 81], 8, 5, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = solve_ivp(fun_rational, [4 / 9, 20 / 81], 8, 5, method=method, events=(event_rational_1, event_rational_2,
                                                                                     event_rational_3))
        assert_allclose(res.tF, 7.4)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_equal(len(res.t_events[2]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)

        # Also test that termination by event doesn't break interpolants.
        tc = np.linspace(res.t0, res.tF)
        yc_true = sol_rational(tc)
        yc = res(tc)
        assert_allclose(yc, yc_true, rtol=1e-2)


def test_step():
    for method in all_methods:
        t0 = 5
        y0 = [1/3, 2/9]
        solver = method(fun_rational, y0, t0)
        assert_equal(solver.t, t0)
        assert_equal(solver.y, y0)
        assert_equal(solver.status, SolverStatus.started)

        solver.step()
        assert_(solver.t > t0)
        assert_(np.all(solver.y != y0))
        assert_equal(solver.status, SolverStatus.running)


def test_solution():
    for method in all_methods:
        sol = solve_ivp(lambda t, y: -y, [2, 3], 0, 4, method=method)
        assert_equal(sol(0), [2, 3])
        assert_equal(sol([0, 2, 4]).shape, (3, 2))
        assert_raises(ValueError, sol, -1)
        assert_raises(ValueError, sol, [1, 4.00000001])


def test_no_integration():
    for method in all_methods:
        sol = solve_ivp(lambda t, y: -y, [2, 3], 4, 4, method=method)
        assert_equal(sol(4), [2, 3])
        assert_equal(sol([4, 4, 4]), [[2, 3], [2, 3], [2, 3]])


def test_events_and_infinity():
    def event(t, y):
        return y[1] - 0.1
    event.terminate = True

    for method in all_methods:
        sol = solve_ivp(lambda t, y: -y, [2, 3], 4, np.inf, method=method, events=event)
        assert_allclose(sol(sol.t_events[0])[1], 0.1)


def test_equilibrium():
    def fun(t, y):
        kf = 0.2
        kr = 0.5

        a, b, c = y

        rf = kf * a * b
        rr = kr * c

        return [
            -rf + rr,
            -rf + rr,
            rf - rr,
        ]

    ic = [4, 3, 0]

    dynamic_reference = None
    known_reference = [4-3/2, 3-3/2, 3/2]  # Calculated by hand

    for method in all_methods:
        sol = solve_ivp(fun, ic, 0, 100, method=method)
        dynamic_result = sol(1.5)
        if dynamic_reference is None:
            dynamic_reference = dynamic_result
        else:
            assert_allclose(dynamic_result, dynamic_reference, rtol=1e-2)

        static_result = sol(100)
        assert_allclose(static_result, known_reference, rtol=1e-2)


def test_parameters_validation():
    assert_raises(ValueError, solve_ivp, 1, 2, [[0, 0]], fun_rational)
    assert_raises(ValueError, solve_ivp, 1, 2, [0, 0], lambda t, y: np.zeros(3))
    assert_raises(ValueError, solve_ivp, 1, 2, [0, 0], fun_rational,
                  method=Radau, jac=lambda t, y: np.identity(3))
    assert_raises(ValueError, solve_ivp, 1, 2, [0, 0], fun_rational,
                  method=Radau, jac=np.identity(3))

if __name__ == '__main__':
    run_module_suite()
