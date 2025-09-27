# ------------------
# Trajectory / Integrator
# ------------------
from potentials import Potential1D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
class Trajectory1D:
    """Simple 1D MD trajectory with Euler–Cromer, Velocity-Verlet and Langevin integration."""
    def __init__(self, potential: Potential1D, m=1.0, dt=0.01, steps=1000, x0=0.0, v0=0.0, integrator="euler_cromer",gamma=None,kT=None):
        self.potential = potential
        self.m = m
        self.dt = dt
        self.steps = steps
        self.x0 = x0
        self.v0 = v0
        self.integrator = integrator
        self.gamma = gamma
        self.kT = kT

        # initialize variables
        self.t = np.linspace(0, dt*steps, steps+1)
        self.x = np.zeros(steps+1)
        self.v = np.zeros(steps+1)
        self.E = np.zeros(steps+1)
        self.E_v = np.zeros(steps+1)  # kinetic energy
        self.E_p = np.zeros(steps+1)  # potential energy

    def run(self):
        self.x[0], self.v[0] = self.x0, self.v0
        self.E[0] = 0.5*self.m*self.v0**2 + self.potential.V(self.x0) # initial energy K + U

        for n in range(self.steps):
            if self.integrator == "euler_cromer":
                self._step_euler_cromer(n)
            elif self.integrator == "velocity_verlet":
                self._step_velocity_verlet(n)
            elif self.integrator == "langevin":
                self._langevin_step(n, self.gamma, self.kT)
            else:
                raise ValueError("Unknown integrator: choose 'euler_cromer' or 'velocity_verlet'")
        return self
    # ---------- Integration step methods ----------
    def _step_euler_cromer(self, n):
        f = -self.potential.dVdx(self.x[n])
        self.v[n+1] = self.v[n] + (f/self.m)*self.dt
        self.x[n+1] = self.x[n] + self.v[n+1]*self.dt
        self.E[n+1] = 0.5*self.m*self.v[n+1]**2 + self.potential.V(self.x[n+1])
        self.E_v[n+1] = 0.5*self.m*self.v[n+1]**2
        self.E_p[n+1] = self.potential.V(self.x[n+1])

    def _step_velocity_verlet(self, n):
        f_n = -self.potential.dVdx(self.x[n])
        # position update
        self.x[n+1] = self.x[n] + self.v[n]*self.dt + 0.5*(f_n/self.m)*self.dt**2
        # force at new position
        f_np1 = -self.potential.dVdx(self.x[n+1])
        # velocity update
        self.v[n+1] = self.v[n] + 0.5*(f_n + f_np1)/self.m * self.dt
        self.E[n+1] = 0.5*self.m*self.v[n+1]**2 + self.potential.V(self.x[n+1])
        self.E_v[n+1] = 0.5*self.m*self.v[n+1]**2
        self.E_p[n+1] = self.potential.V(self.x[n+1])

    def _langevin_step(self, n, gamma, kT):
        f = -self.potential.dVdx(self.x[n])
        # Random force from Gaussian distribution
        R = np.random.normal(0, np.sqrt(2 * gamma * kT / self.dt))
        self.v[n+1] = self.v[n] + (f/self.m - gamma*self.v[n] + R/self.m)*self.dt
        self.x[n+1] = self.x[n] + self.v[n+1]*self.dt
        self.E[n+1] = 0.5*self.m*self.v[n+1]**2 + self.potential.V(self.x[n+1])
        self.E_v[n+1] = 0.5*self.m*self.v[n+1]**2
        self.E_p[n+1] = self.potential.V(self.x[n+1])

    # ---------- Trajectory plot method ----------
    def plot_on_potential(self, xmin=-10, xmax=10, npts=400, cmap="viridis"):
        X = np.linspace(xmin, xmax, npts)
        Y = self.potential.V(X)

        fig, ax = plt.subplots()
        ax.plot(X, Y, color='red', lw=1.5, label="Potential V(x)")

        # Prepare trajectory segments
        points = np.array([self.x, self.potential.V(self.x)]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize time for colormap
        t_norm = (self.t - self.t.min()) / (self.t.max() - self.t.min())
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(t_norm)
        lc.set_linewidth(3)
        ax.add_collection(lc)

        # Colorbar for time
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label("Time")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(min(Y)*0.9, max(Y)*1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("Energy")
        ax.set_title("Trajectory on Potential (colored by time)")
        plt.show()

    # ---------- Energy plot method ----------
    def plot_energy(self):
        plt.figure()
        plt.plot(self.t, self.E)
        plt.plot(self.t, self.E_v, label="Kinetic Energy", linestyle='--')
        plt.plot(self.t, self.E_p, label="Potential Energy", linestyle='--')
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Total Energy")
        plt.title("Energy vs Time")
        plt.show()
    # ---------- Animation method ----------
    def animate_on_potential(self, xmin=-10, xmax=10, npts=400, interval=50):
        X = np.linspace(xmin, xmax, npts)
        Y = self.potential.V(X)
        trajectory_y = self.potential.V(self.x)

        fig, ax = plt.subplots()
        ax.plot(X, Y, label="Potential V(x)")
        path, = ax.plot([], [], 'r-', lw=2, label="Trajectory")
        point, = ax.plot([], [], 'bo', ms=6)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(min(Y)*0.9, max(Y)*1.1)
        ax.set_xlabel("x")
        ax.set_ylabel("Energy")
        ax.legend()

        def update(frame):
            point.set_data(self.x[frame], trajectory_y[frame])
            path.set_data(self.x[:frame+1], trajectory_y[:frame+1])
            return point, path

        ani = FuncAnimation(fig, update, frames=len(self.x), interval=interval, blit=False)
        plt.close(fig)  # avoid duplicate static plot in notebooks
        return HTML(ani.to_jshtml())
# ------------------
# Example usage
# ------------------
#if __name__ == "__main__":
#    pot = DoubleWell(a=0.01, b=6.5)
#    traj = Trajectory1D(potential=pot, m=1.0, dt=0.02, steps=2000, x0=1.0, v0=0.0, integrator="velocity_verlet")
#    traj.run()
#    traj.plot_on_potential(xmin=-10, xmax=10)
#    traj.plot_energy()