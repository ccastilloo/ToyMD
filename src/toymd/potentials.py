"""Analytical one-dimensional potential energies and their derivatives.

Use a consistent set of units for positions, energies, and model parameters.
All built-in models accept scalar positions or NumPy-compatible arrays.
"""

import inspect

import numpy as np

__all__ = [
    "Potential1D",
    "Harmonic",
    "DoubleWell",
    "Doublewell",
    "LennardJones",
    "Morse",
    "AsymmetricDoubleWell",
    "AsimmetricDoublewell",
    "AsymmetricCubicDoubleWell",
    "AsymmetricCubicDoublewell",
    "list_potentials",
    "create_potential",
]


class Potential1D:
    """Define the interface and registry for one-dimensional potentials.

    Attributes
    ----------
    registry : dict of str to type
        Available potential classes, keyed by class name or compatibility alias.

    Notes
    -----
    Subclasses are registered automatically. Implement both ``V`` and ``dVdx``;
    the force used by the integrators is ``-dVdx(x)``.
    """

    registry = {}

    def __init_subclass__(cls, **kwargs):
        """Register a subclass under its class name.

        Parameters
        ----------
        **kwargs : dict
            Class creation options forwarded to the next superclass.
        """
        super().__init_subclass__(**kwargs)
        Potential1D.registry[cls.__name__] = cls

    def V(self, x):
        """Evaluate the potential energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions in the model's length units.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at each position, with the same shape as ``x``.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the potential energy.
        """
        raise NotImplementedError("Implement potential energy V(x).")

    def dVdx(self, x):
        """Evaluate the spatial derivative of the potential energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions in the model's length units.

        Returns
        -------
        float or numpy.ndarray
            Derivative at each position, in energy per unit length.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement the derivative.

        Notes
        -----
        This is the energy gradient. The force has the opposite sign.
        """
        raise NotImplementedError("Implement the gradient dV/dx.")


class Harmonic(Potential1D):
    """Harmonic potential, ``V(x) = k * (x - x0)**2 / 2``.

    Parameters
    ----------
    k : float
        Spring constant, in energy per squared unit length.
    x0 : float
        Equilibrium position, in units of length.
    """

    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0

    def V(self, x):
        """Evaluate the harmonic energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        return 0.5 * self.k * (np.asarray(x) - self.x0) ** 2

    def dVdx(self, x):
        """Evaluate the harmonic energy gradient.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient at ``x``.
        """
        return self.k * (np.asarray(x) - self.x0)


class DoubleWell(Potential1D):
    """Symmetric double-well potential, ``V(x) = a * (x**2 - b**2)**2``.

    Parameters
    ----------
    a : float
        Quartic coefficient, in energy per fourth power of length.
    b : float
        Half the distance between the minima, in units of length.

    Notes
    -----
    For positive ``a`` and ``b``, the minima lie at ``-b`` and ``b`` and
    the barrier height is ``a * b**4``.
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def V(self, x):
        """Evaluate the symmetric double-well energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        return self.a * (np.asarray(x) ** 2 - self.b**2) ** 2

    def dVdx(self, x):
        """Evaluate the symmetric double-well energy gradient.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient at ``x``.
        """
        x = np.asarray(x)
        return 4 * self.a * x * (x**2 - self.b**2)


class LennardJones(Potential1D):
    """Lennard-Jones potential, ``4 * epsilon * ((sigma/x)**12 - (sigma/x)**6)``.

    Parameters
    ----------
    epsilon : float
        Potential well depth, in units of energy.
    sigma : float
        Separation at which the potential crosses zero, in units of length.

    Notes
    -----
    Physical separations are positive. The potential is singular at zero;
    evaluating it there yields nonfinite values.
    """

    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def V(self, x):
        """Evaluate the Lennard-Jones energy.

        Parameters
        ----------
        x : float or array_like
            Particle separation or separations; use positive values.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio6 = (self.sigma / x) ** 6
            return 4 * self.epsilon * (ratio6**2 - ratio6)

    def dVdx(self, x):
        """Evaluate the Lennard-Jones energy gradient.

        Parameters
        ----------
        x : float or array_like
            Particle separation or separations; use positive values.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient, ``24 * epsilon * ((sigma/x)**6
            - 2 * (sigma/x)**12) / x``. The force is its negative.
        """
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio6 = (self.sigma / x) ** 6
            return 24 * self.epsilon * (ratio6 - 2 * ratio6**2) / x


class Morse(Potential1D):
    """Morse potential, ``V(x) = D_e * (1 - exp(-a * (x - x0)))**2``.

    Parameters
    ----------
    D_e : float
        Dissociation energy measured from the minimum.
    a : float
        Inverse length controlling the width of the potential well.
    x0 : float
        Equilibrium position, in units of length.

    Notes
    -----
    This convention sets the minimum energy to zero and the dissociation
    limit to ``D_e``.
    """

    def __init__(self, D_e, a, x0):
        self.D_e = D_e
        self.a = a
        self.x0 = x0

    def V(self, x):
        """Evaluate the Morse energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        return self.D_e * (1 - np.exp(-self.a * (np.asarray(x) - self.x0))) ** 2

    def dVdx(self, x):
        """Evaluate the Morse energy gradient.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient at ``x``.
        """
        exp_term = np.exp(-self.a * (np.asarray(x) - self.x0))
        return 2 * self.D_e * self.a * (1 - exp_term) * exp_term


class AsymmetricDoubleWell(DoubleWell):
    """Linearly tilted double well, ``V(x) = a * (x**2 - b**2)**2 + c*x``.

    Parameters
    ----------
    a : float
        Quartic coefficient, in energy per fourth power of length.
    b : float
        Reference half-separation of the untilted wells, in units of length.
    c : float
        Linear tilt, in energy per unit length.

    Notes
    -----
    The tilt shifts the minima away from ``+/-b``. A sufficiently strong
    tilt removes one of the wells.
    """

    def __init__(self, a, b, c):
        super().__init__(a, b)
        self.c = c

    def V(self, x):
        """Evaluate the linearly tilted double-well energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        return super().V(x) + self.c * np.asarray(x)

    def dVdx(self, x):
        """Evaluate the linearly tilted double-well energy gradient.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient at ``x``.
        """
        return super().dVdx(x) + self.c


class AsymmetricCubicDoubleWell(DoubleWell):
    """Cubically tilted double well, ``V(x) = a * (x**2 - b**2)**2 + d*x**3``.

    Parameters
    ----------
    a : float
        Quartic coefficient, in energy per fourth power of length.
    b : float
        Reference half-separation of the untilted wells, in units of length.
    d : float
        Cubic tilt coefficient, in energy per cubed unit length.
    """

    def __init__(self, a, b, d):
        super().__init__(a, b)
        self.d = d

    def V(self, x):
        """Evaluate the cubically tilted double-well energy.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Potential energy at ``x``.
        """
        return super().V(x) + self.d * np.asarray(x) ** 3

    def dVdx(self, x):
        """Evaluate the cubically tilted double-well energy gradient.

        Parameters
        ----------
        x : float or array_like
            Position or positions.

        Returns
        -------
        float or numpy.ndarray
            Energy gradient at ``x``.
        """
        return super().dVdx(x) + 3 * self.d * np.asarray(x) ** 2


# Preserve class imports and factory names used by the original notebooks.
Doublewell = DoubleWell
AsimmetricDoublewell = AsymmetricDoubleWell
AsymmetricCubicDoublewell = AsymmetricCubicDoubleWell
Potential1D.registry.update(
    Doublewell=DoubleWell,
    AsimmetricDoublewell=AsymmetricDoubleWell,
    AsymmetricCubicDoublewell=AsymmetricCubicDoubleWell,
)


def list_potentials():
    """Print registered potential names and constructor parameters.

    Returns
    -------
    None
        Names and signatures are written to standard output.
    """
    print("Available potentials:")
    for name, cls in Potential1D.registry.items():
        print(" - {}{}".format(name, inspect.signature(cls)))


def create_potential(name, **kwargs):
    """Create a registered potential from its name and model parameters.

    Parameters
    ----------
    name : str
        Registered class name or compatibility alias. Names are case-sensitive.
    **kwargs : dict
        Keyword arguments passed to the potential constructor.

    Returns
    -------
    Potential1D or None
        Initialized potential, or ``None`` if the name or constructor
        arguments are invalid. Invalid input prints an explanatory message,
        preserving the original notebook interface.

    Examples
    --------
    >>> pot = create_potential("Harmonic", k=1.0, x0=0.0)
    >>> float(pot.V(2.0))
    2.0
    """
    if name not in Potential1D.registry:
        print("Error: {!r} not found.".format(name))
        list_potentials()
        return None

    cls = Potential1D.registry[name]
    signature = inspect.signature(cls)
    try:
        signature.bind(**kwargs)
        return cls(**kwargs)
    except TypeError as error:
        print("Initialization error for {}: {}".format(name, error))
        print("Required signature: {}".format(signature))
        return None
