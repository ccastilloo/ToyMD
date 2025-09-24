# ToyMD

ToyMD is a simple Python project for simulating and analyzing 1D molecular dynamics (MD) trajectories in custom potentials. It is designed for educational and prototyping purposes, with a focus on clarity and extensibility.

## Features

- **1D Potential Energy Models:**  
  Includes several built-in potentials such as Harmonic, Double Well, Lennard-Jones, Morse, and Asymmetric Double Well.  
- **Trajectory Simulation:**  
  Integrate particle motion using Euler–Cromer and Velocity-Verlet algorithms.
- **Visualization:**  
  Plot trajectories, energies, and animate particle motion on the potential landscape.
- **Extensible:**  
  Easily add new potentials or integrators.
- **Jupyter Notebook Friendly:**  
  Designed for interactive exploration and visualization.

## Example Usage

```python
from potentials import create_potential
from walker import Trajectory1D

pot = create_potential("Harmonic", k=0.1, x0=0.0)
traj = Trajectory1D(potential=pot, m=1.0, dt=0.02, steps=400, x0=2.0, v0=0.0, integrator="velocity_verlet")
traj.run()
traj.plot_on_potential(xmin=-10, xmax=10)
traj.plot_energy()
traj.animate_on_potential(xmin=-10, xmax=10, interval=80)

## Installation
pip install -e .

## Requirements
Python 3.7+
numpy
matplotlib
