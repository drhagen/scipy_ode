# ODE solvers for scipy
This is a collaboration space for the third generation ODE solvers of Scipy. This standalone package depends on scipy and several of its dependencies. It is probably best used as a project in an appropriate IDE as it is an experimental/developmental work-in-progress. All APIs are subject to rapid evolution.

### Organization
`scipy_ode/`: The root of the package. All source files for the solvers go here. This will ultimately be merged with `scipy/integrate` or similar substructure.

`scipy_ode/tests/`: Contains all tests. All tests are run with this working directory. Tests must import from the absolute path `scipy_ode` until they are ultimately merged back into the main line.

`scipy_ode/__init__.py`: Defines the public interface of `scipy_ode`. Derived from `scipy/integrate/__init__/py`.
