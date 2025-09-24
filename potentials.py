import numpy as np
import matplotlib.pyplot as plt
import inspect

# ------------------
# Base Potential class with registry
# ------------------
class Potential1D:
    """Abstract base class for 1D potentials."""
    registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register all subclasses."""
        super().__init_subclass__(**kwargs)
        Potential1D.registry[cls.__name__] = cls

    def V(self, x):
        raise NotImplementedError("Implement potential energy V(x)")

    def dVdx(self, x):
        raise NotImplementedError("Implement gradient dV/dx")

# ------------------
# List of potential implementations
# ------------------
class Harmonic(Potential1D):
    """V(x) = 0.5 * k * (x - x0)^2"""
    def __init__(self, k, x0):
        self.k = k
        self.x0 = x0
    def V(self, x):
        return 0.5 * self.k * (x - self.x0)**2
    def dVdx(self, x):
        return self.k * (x - self.x0)

class Doublewell(Potential1D):
    """V(x) = a (x^2 - b^2)^2"""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def V(self, x):
        return self.a * (x**2 - self.b**2)**2
    def dVdx(self, x):
        return 4 * self.a * x * (x**2 - self.b**2)
    
class LennardJones(Potential1D):
    """V(x) = 4 * epsilon * ((sigma/x)^12 - (sigma/x)^6)"""
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma
    def V(self, x):
        with np.errstate(divide='ignore'):
            return 4 * self.epsilon * ((self.sigma / x)**12 - (self.sigma / x)**6)
    def dVdx(self, x):
        with np.errstate(divide='ignore'):
            return 24 * self.epsilon * (2 * (self.sigma**12 / x**13) - (self.sigma**6 / x**7))

class Morse(Potential1D):
    """V(x) = D_e * (1 - exp(-a*(x - x0)))^2"""
    def __init__(self, D_e, a, x0):
        self.D_e = D_e
        self.a = a
        self.x0 = x0
    def V(self, x):
        return self.D_e * (1 - np.exp(-self.a * (x - self.x0)))**2
    def dVdx(self, x):
        exp_term = np.exp(-self.a * (x - self.x0))
        return 2 * self.D_e * self.a * (1 - exp_term) * exp_term

class AsimmetricDoublewell(Potential1D):
    """V(x) = a (x^2 - b^2)^2 + c*x"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    def V(self, x):
        return self.a * (x**2 - self.b**2)**2 + self.c * x
    def dVdx(self, x):
        return 4 * self.a * x * (x**2 - self.b**2) + self.c

# ------------------
# Helper functions
# ------------------
def list_potentials():
    """Print available potentials and their init arguments."""
    print("Available potentials:")
    for name, cls in Potential1D.registry.items():
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        print(f" - {name}{sig}")

def create_potential(name, **kwargs):
    """Factory function to create potentials with error handling."""
    if name not in Potential1D.registry:
        print(f"Error: '{name}' not found.")
        list_potentials()
        return None
    
    cls = Potential1D.registry[name]
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.values())[1:]  # skip 'self'

    # Check for missing required parameters
    missing = [p.name for p in params if p.name not in kwargs]
    if missing:
        print(f"Error: Missing required parameter(s) for {name}: {', '.join(missing)}")
        print(f"Required signature: {sig}")
        return None

    try:
        return cls(**kwargs)
    except TypeError as e:
        print(f"Initialization error for {name}: {e}")
        print(f"Required signature: {sig}")
        return None





