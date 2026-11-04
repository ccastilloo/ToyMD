# API guide

Public classes and functions include NumPy-style docstrings with parameters,
return values, units, and relevant exceptions. Use `help(Trajectory1D)` or
`help(Harmonic.V)` in Python to view them.

## Potential models

```python
import numpy as np
from toymd import DoubleWell, create_potential, list_potentials

potential = DoubleWell(a=0.01, b=6.5)
positions = np.linspace(-10, 10, 400)
energies = potential.V(positions)
forces = -potential.dVdx(positions)

list_potentials()
same_model = create_potential("DoubleWell", a=0.01, b=6.5)
```

The factory preserves the original interactive behavior: invalid names or
constructor arguments print an explanation and return `None`. Direct class
construction uses normal Python exceptions. Parameter values must be appropriate
for the intended physical model; constructors do not enforce a physical domain.
The Lennard-Jones potential is singular at zero separation.

To add a model, subclass `Potential1D` and implement both `V(x)` and `dVdx(x)`.
Both methods should support scalar positions and NumPy arrays. Class creation
automatically adds the model to `Potential1D.registry`.

## Trajectories

`Trajectory1D(potential, m=1.0, dt=0.01, steps=1000, x0=0.0, v0=0.0,
integrator="euler_cromer", gamma=None, T=None, *, rng=None)` allocates arrays.
Call `run()` to populate them. It returns the same trajectory for method chaining.
Repeated calls restart from the initial conditions; they do not append samples.

| Attribute | Meaning |
| --- | --- |
| `t` | Time, from zero through `steps * dt` |
| `x` | Position |
| `v` | Velocity |
| `E_v` | Kinetic energy, `m * v**2 / 2` |
| `E_p` | Potential energy, `potential.V(x)` |
| `E` | Total mechanical energy, `E_v + E_p` |

All arrays have length `steps + 1`. A zero-step run records only the initial
conditions and their energies. Invalid mass, time step, step count, integrator,
or Langevin bath parameters fail before integration.

### Langevin example in SI units

```python
import numpy as np
from toymd import Harmonic, Trajectory1D

trajectory = Trajectory1D(
    potential=Harmonic(k=1.0, x0=0.0),  # J/m^2
    m=1.66e-26,                         # kg
    dt=1e-15,                           # s
    steps=5000,
    x0=1e-11,                           # m
    v0=0.0,                             # m/s
    integrator="langevin",
    gamma=1e12,                         # 1/s
    T=300.0,                            # K
    rng=np.random.default_rng(42),
).run()
```

The update combines the potential force, friction `-gamma * m * v`, and a
Gaussian random force with standard deviation
`sqrt(2 * gamma * m * k_B * T / dt)`. It updates velocity first, then position.
Mechanical energy fluctuates as the particle exchanges energy with the bath.
Neither `kT` nor `k` is a temperature argument; use `T` in kelvin.

## Visualization

The trajectory's convenience methods delegate to `toymd.plotting`. Matplotlib
is loaded when a plotting method is called; IPython is only needed for HTML
animations.

```python
figure, axes = trajectory.plot_energy(show=False)
axes.set_title("My simulation")
figure.savefig("energy.png", dpi=150, bbox_inches="tight")
```

- `plot_on_potential(xmin=-10, xmax=10, npts=400, cmap="viridis", show=True)`
  returns the figure and axes with a time-colored trajectory.
- `plot_energy(show=True)` returns the figure and axes with all energy curves.
- `animate_on_potential(xmin=-10, xmax=10, npts=1000, interval=50)` returns
  `IPython.display.HTML`. The interval is in milliseconds, independent of the
  simulated time step. Set limits appropriate to the model's length units.

The standalone functions take a trajectory as their first argument and accept
the same plotting options. Large trajectories can produce large HTML animations.
