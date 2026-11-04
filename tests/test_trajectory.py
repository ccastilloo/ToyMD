"""Integration checks against analytical solutions and stochastic properties."""

import numpy as np
import pytest

from toymd import Harmonic, Trajectory1D
from toymd.trajectory import BOLTZMANN_CONSTANT


@pytest.mark.parametrize(
    "integrator, tolerance",
    [("euler_cromer", 0.006), ("velocity_verlet", 0.00005)],
)
def test_harmonic_motion_follows_analytical_solution(integrator, tolerance):
    trajectory = Trajectory1D(
        Harmonic(k=2, x0=0.3),
        m=2,
        dt=0.01,
        steps=1000,
        x0=1.3,
        v0=0.4,
        integrator=integrator,
    ).run()
    expected = 0.3 + np.cos(trajectory.t) + 0.4 * np.sin(trajectory.t)
    np.testing.assert_allclose(trajectory.x, expected, atol=tolerance, rtol=0)
    if integrator == "velocity_verlet":
        np.testing.assert_allclose(trajectory.E, trajectory.E[0], rtol=3e-5)


@pytest.mark.parametrize("integrator", ["euler_cromer", "velocity_verlet", "langevin"])
def test_all_energy_components_include_initial_conditions(integrator):
    trajectory = Trajectory1D(
        Harmonic(k=2, x0=0),
        m=3,
        x0=2,
        v0=0.5,
        steps=5,
        integrator=integrator,
        gamma=0,
        T=0,
    )
    assert trajectory.run() is trajectory
    assert trajectory.E_p[0] == pytest.approx(4)
    assert trajectory.E_v[0] == pytest.approx(0.375)
    np.testing.assert_allclose(trajectory.E, trajectory.E_v + trajectory.E_p)
    np.testing.assert_allclose(trajectory.E_p, trajectory.potential.V(trajectory.x))
    np.testing.assert_allclose(trajectory.E_v, 0.5 * trajectory.m * trajectory.v**2)


def test_zero_steps_records_only_initial_state():
    trajectory = Trajectory1D(Harmonic(2, 0), steps=0, x0=2, v0=3).run()
    np.testing.assert_array_equal(trajectory.t, [0])
    np.testing.assert_array_equal(trajectory.x, [2])
    np.testing.assert_array_equal(trajectory.v, [3])
    np.testing.assert_array_equal(trajectory.E, [8.5])


def test_repeated_deterministic_runs_restart_at_initial_state():
    trajectory = Trajectory1D(Harmonic(1, 0), x0=1, steps=30).run()
    first = trajectory.x.copy()
    trajectory.run()
    np.testing.assert_array_equal(trajectory.x, first)


def test_langevin_without_bath_matches_euler_cromer():
    kwargs = dict(potential=Harmonic(1, 0), x0=1, v0=0.2, steps=100)
    deterministic = Trajectory1D(**kwargs).run()
    langevin = Trajectory1D(**kwargs, integrator="langevin", gamma=0, T=0).run()
    np.testing.assert_allclose(langevin.x, deterministic.x, atol=1e-14)
    np.testing.assert_allclose(langevin.v, deterministic.v, atol=1e-14)


def test_langevin_seeded_generators_are_reproducible():
    kwargs = dict(
        potential=Harmonic(1, 0),
        m=1.66e-26,
        dt=1e-15,
        steps=100,
        integrator="langevin",
        gamma=1e12,
        T=300,
    )
    first = Trajectory1D(**kwargs, rng=np.random.default_rng(42)).run()
    second = Trajectory1D(**kwargs, rng=np.random.default_rng(42)).run()
    different = Trajectory1D(**kwargs, rng=np.random.default_rng(43)).run()
    np.testing.assert_array_equal(first.v, second.v)
    assert not np.array_equal(first.v, different.v)


def test_free_langevin_velocity_variance_matches_discrete_thermal_limit():
    # A free particle has an AR(1) velocity update. Its stationary variance is
    # (k_B*T/m) / (1 - gamma*dt/2), approaching equipartition as dt -> 0.
    mass, temperature, gamma, dt = 1e-26, 300, 1e12, 1e-14
    trajectory = Trajectory1D(
        Harmonic(0, 0),
        m=mass,
        dt=dt,
        steps=100000,
        integrator="langevin",
        gamma=gamma,
        T=temperature,
        rng=np.random.default_rng(123),
    ).run()
    expected = BOLTZMANN_CONSTANT * temperature / mass / (1 - gamma * dt / 2)
    assert np.var(trajectory.v[5000:]) == pytest.approx(expected, rel=0.15)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"m": 0}, ValueError),
        ({"m": -1}, ValueError),
        ({"m": np.nan}, ValueError),
        ({"dt": 0}, ValueError),
        ({"dt": np.inf}, ValueError),
        ({"steps": -1}, ValueError),
        ({"steps": 1.5}, TypeError),
        ({"steps": True}, TypeError),
        ({"x0": np.inf}, ValueError),
        ({"v0": np.nan}, ValueError),
        ({"integrator": "missing"}, ValueError),
        ({"integrator": "langevin"}, ValueError),
        ({"integrator": "langevin", "gamma": -1, "T": 300}, ValueError),
        ({"integrator": "langevin", "gamma": 1, "T": -1}, ValueError),
        ({"integrator": "langevin", "gamma": 1, "T": np.nan}, ValueError),
    ],
)
def test_invalid_simulation_parameters_fail_early(kwargs, error):
    with pytest.raises(error):
        Trajectory1D(Harmonic(1, 0), **kwargs)


def test_missing_potential_interface_fails_early():
    with pytest.raises(TypeError, match="V\\(x\\) and dVdx"):
        Trajectory1D(object())
