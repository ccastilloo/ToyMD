"""Numerical gradient checks and compatibility coverage for potential models."""

import numpy as np
import pytest

from toymd import (
    AsimmetricDoublewell,
    AsymmetricCubicDoubleWell,
    AsymmetricCubicDoublewell,
    AsymmetricDoubleWell,
    DoubleWell,
    Doublewell,
    Harmonic,
    LennardJones,
    Morse,
    Potential1D,
    create_potential,
    list_potentials,
)


@pytest.mark.parametrize(
    "potential, positions",
    [
        (Harmonic(k=1.7, x0=0.2), [-1.1, 0.4, 1.6]),
        (DoubleWell(a=0.4, b=1.2), [-1.6, 0.3, 1.8]),
        (LennardJones(epsilon=0.8, sigma=1.1), [1.05, 1.4, 2.0]),
        (Morse(D_e=2.0, a=0.7, x0=0.3), [-0.2, 0.8, 1.5]),
        (AsymmetricDoubleWell(a=0.4, b=1.2, c=0.3), [-1.6, 0.3, 1.8]),
        (AsymmetricCubicDoubleWell(a=0.4, b=1.2, d=0.2), [-1.6, 0.3, 1.8]),
    ],
)
def test_analytical_gradient_matches_finite_difference(potential, positions):
    x = np.array(positions)
    step = 1e-6
    numerical = (potential.V(x + step) - potential.V(x - step)) / (2 * step)
    np.testing.assert_allclose(potential.dVdx(x), numerical, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(potential.V(positions), potential.V(x))
    np.testing.assert_allclose(potential.dVdx(positions), potential.dVdx(x))
    for index, position in enumerate(positions):
        assert potential.V(position) == pytest.approx(potential.V(x)[index])
        assert potential.dVdx(position) == pytest.approx(numerical[index], rel=1e-6)


def test_lennard_jones_force_repels_short_separations_and_attracts_long_ones():
    potential = LennardJones(epsilon=1.0, sigma=1.0)
    assert -potential.dVdx(0.9) > 0
    assert -potential.dVdx(1.5) < 0
    assert potential.V(2 ** (1 / 6)) == pytest.approx(-1)
    assert potential.dVdx(2 ** (1 / 6)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize(
    "alias, cls, kwargs",
    [
        ("Doublewell", DoubleWell, {"a": 0.1, "b": 2.0}),
        ("AsimmetricDoublewell", AsymmetricDoubleWell, {"a": 0.1, "b": 2, "c": 0.3}),
        (
            "AsymmetricCubicDoublewell",
            AsymmetricCubicDoubleWell,
            {"a": 0.1, "b": 2, "d": 0.3},
        ),
    ],
)
def test_factory_supports_original_and_canonical_names(alias, cls, kwargs):
    assert type(create_potential(alias, **kwargs)) is cls
    assert type(create_potential(cls.__name__, **kwargs)) is cls


def test_original_class_names_resolve_to_canonical_classes():
    assert Doublewell is DoubleWell
    assert AsimmetricDoublewell is AsymmetricDoubleWell
    assert AsymmetricCubicDoublewell is AsymmetricCubicDoubleWell


@pytest.mark.parametrize(
    "name, kwargs",
    [("missing", {}), ("Harmonic", {"k": 1}), ("Harmonic", {"k": 1, "x0": 0, "z": 2})],
)
def test_factory_reports_invalid_input_without_raising(name, kwargs, capsys):
    assert create_potential(name, **kwargs) is None
    assert "error" in capsys.readouterr().out.lower()


def test_custom_models_register_and_keep_constructor_defaults():
    class CustomHarmonic(Harmonic):
        def __init__(self, k=2.0, x0=0.0):
            super().__init__(k, x0)

    try:
        potential = create_potential("CustomHarmonic")
        assert isinstance(potential, CustomHarmonic)
        assert potential.V(1.0) == pytest.approx(1.0)
    finally:
        Potential1D.registry.pop("CustomHarmonic")


def test_list_potentials_shows_constructor_arguments(capsys):
    list_potentials()
    output = capsys.readouterr().out
    assert "Harmonic(k, x0)" in output
    assert "self" not in output
