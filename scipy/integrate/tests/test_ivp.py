from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import (assert_, assert_allclose, run_module_suite,
                           assert_equal, assert_raises)
from scipy_integrate.integrate import SolverStatus, ivp_solution
from scipy_integrate.integrate import RungaKutta23, RungaKutta45, Radau


all_methods = [RungaKutta23, RungaKutta45, Radau]


def fun_rational(x, y):
    return np.array([y[1] / x,
                     y[1] * (y[0] + 2 * y[1] - 1) / (x * (y[0] - 1))])


def jac_rational(x, y):
    return np.array([
        [0, 1 / x],
        [-2 * y[1] ** 2 / (x * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (x * (y[0] - 1))]
    ])


def sol_rational(x):
    return np.vstack((x / (x + 10), 10 * x / (x + 10)**2)).T


def event_rational_1(x, y):
    return y[0] - y[1] ** 0.7


def event_rational_2(x, y):
    return y[1] ** 0.6 - y[0]


def event_rational_3(x, y):
    return x - 7.4


def test_integration():
    rtol = 1e-3
    atol = 1e-6
    for method in all_methods:
        for x_span in ([5, 9], [5, 1]):
            res = ivp_solution(x_span[0], x_span[1], [1/3, 2/9], fun_rational, rtol=rtol,
                            atol=atol, method=method)
            assert_equal(res.t0, x_span[0])
            assert_equal(res.tF, x_span[-1])
            assert_(res.t_events is None)

            xc = np.linspace(*x_span)
            yc_true = sol_rational(xc)
            yc = res(xc)
            assert_allclose(yc, yc_true, rtol=1e-2)


def test_events():
    event_rational_3.terminate = True

    for method in all_methods:
        res = ivp_solution(5, 8, [1/3, 2/9], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = ivp_solution(5, 8, [1 / 3, 2 / 9], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = ivp_solution(5, 8, [1 / 3, 2 / 9], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 8)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = ivp_solution(5, 8, [1 / 3, 2 / 9], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2,
                                event_rational_3))
        assert_allclose(res.tF, 7.4)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_equal(len(res.t_events[2]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)

        # Also test that termination by event doesn't break interpolants.
        xc = np.linspace(res.t0, res.tF)
        yc_true = sol_rational(xc)
        yc = res(xc)
        assert_allclose(yc, yc_true, rtol=1e-2)

    # Test in backward direction.
    event_rational_1.direction = 0
    event_rational_2.direction = 0
    for method in all_methods:
        res = ivp_solution(8, 5, [4/9, 20/81], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 1)
        assert_(5.3 < res.t_events[0][0] < 5.7)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = -1
        event_rational_2.direction = -1
        res = ivp_solution(8, 5, [4/9, 20/81], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 1)
        assert_equal(len(res.t_events[1]), 0)
        assert_(5.3 < res.t_events[0][0] < 5.7)

        event_rational_1.direction = 1
        event_rational_2.direction = 1
        res = ivp_solution(8, 5, [4/9, 20/81], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2))
        assert_equal(res.tF, 5)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)

        event_rational_1.direction = 0
        event_rational_2.direction = 0

        res = ivp_solution(8, 5, [4/9, 20/81], fun_rational, method=method,
                        events=(event_rational_1, event_rational_2,
                                event_rational_3))
        assert_allclose(res.tF, 7.4)
        assert_equal(len(res.t_events[0]), 0)
        assert_equal(len(res.t_events[1]), 1)
        assert_equal(len(res.t_events[2]), 1)
        assert_(7.3 < res.t_events[1][0] < 7.7)
        assert_(7.3 < res.t_events[2][0] < 7.5)

        # Also test that termination by event doesn't break interpolants.
        xc = np.linspace(res.t0, res.tF)
        yc_true = sol_rational(xc)
        yc = res(xc)
        assert_allclose(yc, yc_true, rtol=1e-2)


def test_step():
    for method in all_methods:
        t0 = 5
        y0 = [1/3, 2/9]
        solver = method(t0, y0, fun_rational)
        assert_equal(solver.t, t0)
        assert_equal(solver.y, y0)
        assert_equal(solver.status, SolverStatus.started)

        solver.step()
        assert_(solver.t > t0)
        assert_(np.all(solver.y != y0))
        assert_equal(solver.status, SolverStatus.running)


def test_solution():
    for method in all_methods:
        sol = ivp_solution(0, 4, [2, 3], lambda t, y: -y, method=method)
        assert_equal(sol(0), [2, 3])
        assert_equal(sol(0, 1), 3)
        assert_equal(sol([0, 2, 4]).shape, (3, 2))
        assert_equal(sol([0, 2, 4], 0).shape, (3,))
        assert_equal(sol([0, 2, 4], [0, 1]).shape, (3, 2))
        assert_equal(sol(2, [0, 1]).shape, (2,))
        assert_raises(ValueError, sol, -1)
        assert_raises(ValueError, sol, [1, 4.00000001])


def test_no_integration():
    for method in all_methods:
        sol = ivp_solution(4, 4, [2, 3], lambda t, y: -y, method=method)
        assert_equal(sol(4), [2, 3])
        assert_equal(sol([4, 4, 4]), [[2, 3], [2, 3], [2, 3]])


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
        sol = ivp_solution(0, 100, ic, fun, method=method)
        dynamic_result = sol(1.5)
        if dynamic_reference is None:
            dynamic_reference = dynamic_result
        else:
            assert_allclose(dynamic_result, dynamic_reference, rtol=1e-2)

        static_result = sol(100)
        assert_allclose(static_result, known_reference, rtol=1e-2)


def test_parameters_validation():
    assert_raises(ValueError, ivp_solution, 1, 2, [[0, 0]], fun_rational)
    assert_raises(ValueError, ivp_solution, 1, 2, [0, 0], lambda x, y: np.zeros(3))
    assert_raises(ValueError, ivp_solution, 1, 2, [0, 0], fun_rational,
                  method=Radau, jac=lambda x, y: np.identity(3))
    assert_raises(ValueError, ivp_solution, 1, 2, [0, 0], fun_rational,
                  method=Radau, jac=np.identity(3))

if __name__ == '__main__':
    run_module_suite()
