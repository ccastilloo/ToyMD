"""Time integration for a particle moving in a one-dimensional potential."""

import operator

import numpy as np

from .potentials import Potential1D

BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


class Trajectory1D:
    """Simulate one-dimensional motion with deterministic or Langevin dynamics.

    Parameters
    ----------
    potential : Potential1D
        Model implementing ``V(x)`` and ``dVdx(x)``.
    m : float, optional
        Positive particle mass. Default is 1.0.
    dt : float, optional
        Positive integration time step. Default is 0.01.
    steps : int, optional
        Number of integration steps, excluding the initial sample. Must be
        nonnegative. Default is 1000.
    x0 : float, optional
        Initial position. Default is 0.0.
    v0 : float, optional
        Initial velocity. Default is 0.0.
    integrator : {'euler_cromer', 'velocity_verlet', 'langevin'}, optional
        Integration scheme. Default is ``'euler_cromer'``.
    gamma : float or None, optional
        Nonnegative damping coefficient, in inverse seconds. Required for
        Langevin dynamics; otherwise unused.
    T : float or None, optional
        Nonnegative bath temperature, in kelvin. Required for Langevin
        dynamics; otherwise unused.
    rng : numpy.random.Generator or None, optional
        Random number generator for Langevin noise. If omitted, use NumPy's
        global random state, preserving support for ``numpy.random.seed``.

    Attributes
    ----------
    t : numpy.ndarray of shape (steps + 1,)
        Sample times, including zero.
    x : numpy.ndarray of shape (steps + 1,)
        Positions populated by :meth:`run`.
    v : numpy.ndarray of shape (steps + 1,)
        Velocities populated by :meth:`run`.
    E : numpy.ndarray of shape (steps + 1,)
        Total mechanical energy, including the initial sample.
    E_v : numpy.ndarray of shape (steps + 1,)
        Kinetic energy, including the initial sample.
    E_p : numpy.ndarray of shape (steps + 1,)
        Potential energy, including the initial sample.

    Raises
    ------
    TypeError
        If ``steps`` is not an integer or the potential lacks the interface.
    ValueError
        If simulation parameters or the integrator name are invalid.

    Notes
    -----
    Deterministic integration supports any consistent unit system. Langevin
    integration uses the SI Boltzmann constant, so use kilograms, meters,
    seconds, joules, and kelvin. Its explicit stochastic update requires a
    sufficiently small time step; it does not conserve mechanical energy.

    Examples
    --------
    >>> from toymd import Harmonic
    >>> trajectory = Trajectory1D(
    ...     Harmonic(k=1.0, x0=0.0), x0=1.0, integrator="velocity_verlet"
    ... ).run()
    >>> trajectory.x.shape
    (1001,)
    """

    def __init__(
        self,
        potential: Potential1D,
        m=1.0,
        dt=0.01,
        steps=1000,
        x0=0.0,
        v0=0.0,
        integrator="euler_cromer",
        gamma=None,
        T=None,
        *,
        rng=None,
    ):
        if not callable(getattr(potential, "V", None)) or not callable(
            getattr(potential, "dVdx", None)
        ):
            raise TypeError("potential must implement V(x) and dVdx(x).")
        if not np.isfinite(m) or m <= 0:
            raise ValueError("m must be finite and positive.")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        if isinstance(steps, (bool, np.bool_)):
            raise TypeError("steps must be an integer.")
        steps = operator.index(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative.")
        if not np.isfinite(x0) or not np.isfinite(v0):
            raise ValueError("x0 and v0 must be finite.")
        if integrator not in ("euler_cromer", "velocity_verlet", "langevin"):
            raise ValueError(
                "Unknown integrator: choose 'euler_cromer', "
                "'velocity_verlet', or 'langevin'."
            )
        if integrator == "langevin":
            for name, value in (("gamma", gamma), ("T", T)):
                if value is None or not np.isfinite(value) or value < 0:
                    raise ValueError(
                        "{} must be finite and nonnegative for Langevin dynamics.".format(
                            name
                        )
                    )

        self.potential = potential
        self.m = m
        self.dt = dt
        self.steps = steps
        self.x0 = x0
        self.v0 = v0
        self.integrator = integrator
        self.gamma = gamma
        self.T = T
        self.rng = rng

        self.t = np.arange(steps + 1) * dt
        self.x = np.zeros(steps + 1)
        self.v = np.zeros(steps + 1)
        self.E = np.zeros(steps + 1)
        self.E_v = np.zeros(steps + 1)
        self.E_p = np.zeros(steps + 1)

    def run(self):
        """Integrate the trajectory from its initial conditions.

        Returns
        -------
        Trajectory1D
            This instance with positions, velocities, and energies populated.

        Notes
        -----
        Each call restarts at ``x0`` and ``v0`` and overwrites the samples.
        The random generator state advances across Langevin runs.
        """
        self.x[0], self.v[0] = self.x0, self.v0
        self._record_energy(0)
        step = {
            "euler_cromer": self._step_euler_cromer,
            "velocity_verlet": self._step_velocity_verlet,
            "langevin": self._step_langevin,
        }[self.integrator]
        for n in range(self.steps):
            step(n)
            self._record_energy(n + 1)
        return self

    def _record_energy(self, n):
        """Store kinetic, potential, and total energy at sample index ``n``."""
        self.E_v[n] = 0.5 * self.m * self.v[n] ** 2
        self.E_p[n] = self.potential.V(self.x[n])
        self.E[n] = self.E_v[n] + self.E_p[n]

    def _step_euler_cromer(self, n):
        """Advance velocity, then position, from sample ``n`` to ``n + 1``."""
        force = -self.potential.dVdx(self.x[n])
        self.v[n + 1] = self.v[n] + force / self.m * self.dt
        self.x[n + 1] = self.x[n] + self.v[n + 1] * self.dt

    def _step_velocity_verlet(self, n):
        """Advance one step using forces at the old and new positions."""
        force = -self.potential.dVdx(self.x[n])
        self.x[n + 1] = (
            self.x[n] + self.v[n] * self.dt + 0.5 * force / self.m * self.dt**2
        )
        next_force = -self.potential.dVdx(self.x[n + 1])
        self.v[n + 1] = self.v[n] + 0.5 * (force + next_force) / self.m * self.dt

    def _step_langevin(self, n):
        """Advance one explicit Langevin step with fluctuation-dissipation noise."""
        force = -self.potential.dVdx(self.x[n])
        normal = np.random.normal if self.rng is None else self.rng.normal
        noise = normal(
            0.0,
            np.sqrt(2 * self.gamma * self.m * BOLTZMANN_CONSTANT * self.T / self.dt),
        )
        self.v[n + 1] = (
            self.v[n]
            + (force / self.m - self.gamma * self.v[n] + noise / self.m) * self.dt
        )
        self.x[n + 1] = self.x[n] + self.v[n + 1] * self.dt

    def plot_on_potential(
        self, xmin=-10, xmax=10, npts=400, cmap="viridis", *, show=True
    ):
        """Plot the trajectory over the potential, colored by time.

        Parameters
        ----------
        xmin, xmax : float, optional
            Position limits. Defaults are -10 and 10.
        npts : int, optional
            Number of points used to draw the potential. Default is 400.
        cmap : str, optional
            Matplotlib colormap name. Default is ``'viridis'``.
        show : bool, optional
            Display the figure immediately. Default is True.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure containing the plot and time colorbar.
        axes : matplotlib.axes.Axes
            Axes containing the potential and trajectory.
        """
        from .plotting import plot_on_potential

        return plot_on_potential(self, xmin, xmax, npts, cmap, show=show)

    def plot_energy(self, *, show=True):
        """Plot total, kinetic, and potential energies against time.

        Parameters
        ----------
        show : bool, optional
            Display the figure immediately. Default is True.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure containing the energy curves.
        axes : matplotlib.axes.Axes
            Axes containing the energy curves and legend.
        """
        from .plotting import plot_energy

        return plot_energy(self, show=show)

    def animate_on_potential(self, xmin=-10, xmax=10, npts=1000, interval=50):
        """Create an HTML animation of motion over the potential.

        Parameters
        ----------
        xmin, xmax : float, optional
            Position limits. Defaults are -10 and 10.
        npts : int, optional
            Number of points used to draw the potential. Default is 1000.
        interval : float, optional
            Delay between frames, in milliseconds. Default is 50.

        Returns
        -------
        IPython.display.HTML
            JavaScript animation for display in a Jupyter notebook.

        Raises
        ------
        ImportError
            If IPython is not installed. Install ``ToyMD[notebooks]`` to
            enable notebook animations.
        """
        from .plotting import animate_on_potential

        return animate_on_potential(self, xmin, xmax, npts, interval)
