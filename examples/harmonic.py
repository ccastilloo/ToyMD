"""Simulate a harmonic oscillator and display its trajectory and energies.

Run ``python examples/harmonic.py`` after installing ToyMD.
"""

from toymd import Harmonic, Trajectory1D


def main():
    """Run and plot the harmonic oscillator example.

    Returns
    -------
    None
        The trajectory and energy figures are displayed with Matplotlib.
    """
    trajectory = Trajectory1D(
        potential=Harmonic(k=0.1, x0=0.0),
        m=1.0,
        dt=0.02,
        steps=400,
        x0=2.0,
        integrator="velocity_verlet",
    ).run()
    trajectory.plot_on_potential()
    trajectory.plot_energy()


if __name__ == "__main__":
    main()
