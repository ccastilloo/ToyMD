"""Regression checks for zero-step plots and HTML animation rendering."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from toymd import Harmonic, Trajectory1D


@pytest.mark.parametrize("steps", [0, 3])
def test_potential_plot_handles_time_and_zero_steps(steps):
    trajectory = Trajectory1D(Harmonic(1, 0), x0=1, dt=0.2, steps=steps).run()
    figure, axes = trajectory.plot_on_potential(show=False)
    try:
        figure.canvas.draw()
        collection = axes.collections[0]
        assert len(collection.get_segments()) == steps
        np.testing.assert_array_equal(collection.get_array(), trajectory.t[:-1])
        assert collection.norm.vmax == pytest.approx(max(steps * 0.2, 0.2))
    finally:
        plt.close(figure)


def test_potential_plot_includes_negative_energies():
    class ShiftedHarmonic(Harmonic):
        def V(self, x):
            return super().V(x) - 100

    trajectory = Trajectory1D(ShiftedHarmonic(1, 0), steps=0).run()
    figure, axes = trajectory.plot_on_potential(xmin=-1, xmax=1, show=False)
    try:
        lower, upper = axes.get_ylim()
        assert lower < -100 < upper
    finally:
        plt.close(figure)


def test_energy_plot_labels_all_components():
    trajectory = Trajectory1D(Harmonic(1, 0), x0=1, steps=3).run()
    figure, axes = trajectory.plot_energy(show=False)
    try:
        assert [line.get_label() for line in axes.lines] == [
            "Total energy",
            "Kinetic energy",
            "Potential energy",
        ]
        np.testing.assert_array_equal(axes.lines[0].get_ydata(), trajectory.E)
    finally:
        plt.close(figure)


def test_animation_renders_marker_and_all_frames():
    pytest.importorskip("IPython")
    trajectory = Trajectory1D(Harmonic(1, 0), x0=1, steps=2).run()
    existing_figures = plt.get_fignums()
    html = trajectory.animate_on_potential(npts=10, interval=20)
    assert "<script" in html.data
    assert "data:image/png;base64," in html.data
    assert plt.get_fignums() == existing_figures
