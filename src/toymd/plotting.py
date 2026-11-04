"""Matplotlib plots and notebook animations for simulated trajectories."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_on_potential(
    trajectory, xmin=-10, xmax=10, npts=400, cmap="viridis", *, show=True
):
    """Plot a trajectory on its potential energy landscape.

    Parameters
    ----------
    trajectory : toymd.Trajectory1D
        Simulated trajectory, populated by ``run()``.
    xmin, xmax : float, optional
        Position limits. Defaults are -10 and 10.
    npts : int, optional
        Number of potential samples. Default is 400.
    cmap : str, optional
        Colormap used for elapsed time. Default is ``'viridis'``.
    show : bool, optional
        Display the figure immediately. Default is True.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the potential and time colorbar.
    axes : matplotlib.axes.Axes
        Axes containing the potential and trajectory.
    """
    positions = np.linspace(xmin, xmax, npts)
    fig, ax = plt.subplots()
    ax.plot(positions, trajectory.potential.V(positions), color="red", label="V(x)")

    points = np.column_stack((trajectory.x, trajectory.potential.V(trajectory.x)))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=Normalize(trajectory.t[0], max(trajectory.t[-1], trajectory.dt)),
        linewidths=3,
    )
    collection.set_array(trajectory.t[:-1])
    ax.add_collection(collection)
    ax.autoscale_view()
    fig.colorbar(collection, ax=ax, label="Time")
    ax.set(
        xlim=(xmin, xmax), xlabel="x", ylabel="Energy", title="Trajectory on potential"
    )
    if show:
        plt.show()
    return fig, ax


def plot_energy(trajectory, *, show=True):
    """Plot total, kinetic, and potential energies over time.

    Parameters
    ----------
    trajectory : toymd.Trajectory1D
        Simulated trajectory, populated by ``run()``.
    show : bool, optional
        Display the figure immediately. Default is True.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the energy curves.
    axes : matplotlib.axes.Axes
        Axes containing the curves and legend.
    """
    fig, ax = plt.subplots()
    ax.plot(trajectory.t, trajectory.E, label="Total energy")
    ax.plot(trajectory.t, trajectory.E_v, "--", label="Kinetic energy")
    ax.plot(trajectory.t, trajectory.E_p, "--", label="Potential energy")
    ax.set(xlabel="Time", ylabel="Energy", title="Energy vs. time")
    ax.legend()
    if show:
        plt.show()
    return fig, ax


def animate_on_potential(trajectory, xmin=-10, xmax=10, npts=1000, interval=50):
    """Render a trajectory over its potential as a notebook HTML animation.

    Parameters
    ----------
    trajectory : toymd.Trajectory1D
        Simulated trajectory, populated by ``run()``.
    xmin, xmax : float, optional
        Position limits. Defaults are -10 and 10.
    npts : int, optional
        Number of potential samples. Default is 1000.
    interval : float, optional
        Delay between frames, in milliseconds. Default is 50.

    Returns
    -------
    IPython.display.HTML
        Embedded JavaScript animation.

    Raises
    ------
    ImportError
        If the optional IPython dependency is unavailable.
    """
    try:
        from IPython.display import HTML
    except ImportError as error:
        raise ImportError(
            'Notebook animations require IPython. Install with pip install "ToyMD[notebooks]".'
        ) from error

    positions = np.linspace(xmin, xmax, npts)
    trajectory_y = trajectory.potential.V(trajectory.x)
    fig, ax = plt.subplots()
    ax.plot(positions, trajectory.potential.V(positions), label="V(x)")
    (path,) = ax.plot([], [], "r-", linewidth=2, label="Trajectory")
    (point,) = ax.plot([], [], "bo", markersize=6)
    ax.set(xlim=(xmin, xmax), xlabel="x", ylabel="Energy")
    ax.legend()

    def update(frame):
        """Update the marker and path for one animation frame."""
        point.set_data([trajectory.x[frame]], [trajectory_y[frame]])
        path.set_data(trajectory.x[: frame + 1], trajectory_y[: frame + 1])
        return point, path

    animation = FuncAnimation(
        fig, update, frames=len(trajectory.x), interval=interval, blit=False
    )
    try:
        return HTML(animation.to_jshtml())
    finally:
        plt.close(fig)
