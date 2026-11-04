# ToyMD

ToyMD is an educational Python package for simulating a particle moving in a
one-dimensional potential. It provides analytical potentials, Euler-Cromer,
velocity-Verlet, and Langevin integration, energy analysis, and notebook
visualizations.

## Installation

Python 3.7 or later is required. From the project directory, install the package
in editable mode:

```console
python -m pip install -e .
```

For the teaching notebooks, SciPy examples, and HTML animations:

```console
python -m pip install -e ".[notebooks]"
python -m jupyterlab notebooks
```

Use a notebook kernel from the environment where ToyMD is installed. With an
editable installation, imports work from both the repository root and the
`notebooks/` directory.

## Quick start

```python
from toymd import Harmonic, Trajectory1D

potential = Harmonic(k=0.1, x0=0.0)
trajectory = Trajectory1D(
    potential=potential,
    m=1.0,
    dt=0.02,
    steps=400,
    x0=2.0,
    v0=0.0,
    integrator="velocity_verlet",
).run()

trajectory.plot_on_potential(xmin=-10, xmax=10)
trajectory.plot_energy()
```

In a notebook, display an animation in a separate cell:

```python
trajectory.animate_on_potential(xmin=-10, xmax=10, interval=80)
```

The arrays `t`, `x`, `v`, `E`, `E_v`, and `E_p` contain time, position, velocity,
total energy, kinetic energy, and potential energy, respectively. Each has
`steps + 1` samples, including the initial conditions.

## Potentials

All potentials implement `V(x)` and `dVdx(x)` for scalar or array inputs. The
force is **`-dVdx(x)`**. You can instantiate classes directly or use
`create_potential("Harmonic", k=0.1, x0=0.0)`.

| Class | Parameters | Potential energy |
| --- | --- | --- |
| `Harmonic` | `k`, `x0` | `k * (x - x0)**2 / 2` |
| `DoubleWell` | `a`, `b` | `a * (x**2 - b**2)**2` |
| `LennardJones` | `epsilon`, `sigma` | `4 * epsilon * ((sigma/x)**12 - (sigma/x)**6)` |
| `Morse` | `D_e`, `a`, `x0` | `D_e * (1 - exp(-a * (x - x0)))**2` |
| `AsymmetricDoubleWell` | `a`, `b`, `c` | `a * (x**2 - b**2)**2 + c*x` |
| `AsymmetricCubicDoubleWell` | `a`, `b`, `d` | `a * (x**2 - b**2)**2 + d*x**3` |

`list_potentials()` prints registered names and constructor signatures. Subclass
`Potential1D` and implement its two methods to register a custom model.

## Units and integration

Euler-Cromer and velocity-Verlet work with any consistent unit system, including
reduced units. Langevin dynamics uses the SI Boltzmann constant
`1.380649e-23 J/K`: use kilograms, meters, seconds, joules, and kelvin, and pass
both `gamma` (inverse seconds) and `T` (kelvin). Choose a time step that resolves
the motion and damping; the explicit Langevin update can become unstable for
large time steps.

For reproducible Langevin runs, pass `rng=numpy.random.default_rng(seed)`.
Without `rng`, the original `numpy.random.seed(...)` behavior is preserved.
See [the API guide](docs/api.md) for a complete Langevin example.

## Project layout

```text
ToyMD/
├── src/
│   └── toymd/
│       ├── __init__.py       # Public package interface
│       ├── potentials.py     # Potential energies, gradients, and registry
│       ├── trajectory.py     # Integration and energy bookkeeping
│       └── plotting.py       # Plots and notebook animations
├── notebooks/               # Teaching labs and exploratory notebooks
│   └── assets/               # Notebook image and existing animation
├── examples/                 # Runnable scripts
├── tests/                    # Numerical and compatibility regression tests
├── docs/                     # API and refactoring notes
├── pyproject.toml            # Packaging, dependencies, and tool configuration
└── README.md
```

