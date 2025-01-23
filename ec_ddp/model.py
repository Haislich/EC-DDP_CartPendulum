import math
import os
from dataclasses import dataclass

import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from typing_extensions import override


class BaseSystem:
    def __init__(self, name: str, n: int, m: int, g: float = 9.81):
        r"""
        A base class for dynamic systems.

        This class serves as a foundational structure for modeling dynamic systems,
        such as the cart-pendulum. It defines common properties like the number of
        state variables, control inputs, and the gravitational constant.

        Attributes:
        ---

            - name (str): The name of the model or system.
            - n (int): The number of state variables defining the system's state space.
            - m (int): The number of control inputs or actuators in the system.
            - g (float): The gravitational constant, defaulting to 9.81 m/s^2.
        """
        self.name = name
        """Name of the model"""
        self.n = n
        """Number of state variables"""
        self.m = m
        """Number of control inputs"""
        self.g = g
        """Gravitational contant"""

    def setup_animation(self, N_sim, x, u, save_frames):
        grid = GridSpec(2, 2)
        if not save_frames:
            self.ax_large = plt.subplot(grid[:, 0])
            self.ax_small1 = plt.subplot(grid[0, 1])
            self.ax_small2 = plt.subplot(grid[1, 1])
        else:
            self.ax_large = plt.subplot(grid[:, :])

        self.x_max = max(x.min(), x.max(), key=abs)
        self.u_max = max(u.min(), u.max(), key=abs)
        self.N_sim = N_sim

    def update_small_axes(self, x, u, i):
        self.ax_small1.cla()
        self.ax_small1.axis((0, self.N_sim, -self.x_max * 1.1, self.x_max * 1.1))
        self.ax_small1.plot(x[:, :i].T)

        self.ax_small2.cla()
        self.ax_small2.axis((0, self.N_sim, -self.u_max * 1.1, self.u_max * 1.1))
        self.ax_small2.plot(u[:, :i].T)

    def draw_frame(self, ax, i, x, u, alpha=1.0, x_pred=None): ...

    def animate(
        self,
        N_sim,
        x,
        u,
        show_trail=False,
        save_video=False,
        video_filename="animation.mp4",
        save_frames=False,
        frame_number=0,
        frame_folder="frames",
        x_pred=None,
    ):

        self.setup_animation(N_sim, x, u, save_frames)
        if x_pred is not None:
            x_pred.append(x_pred[-1])  # replicate last prediction to avoid crash

        frame_indices = np.linspace(0, self.N_sim, frame_number, dtype=int) if save_frames and frame_number > 0 else []

        if save_frames and frame_number > 0:
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)

        def update_frame(i):
            self.ax_large.cla()

            trail_length = show_trail * 10
            spacing = 10
            trail_indices = [i - j * spacing for j in range(trail_length) if i - j * spacing >= 0]

            for idx, j in enumerate(trail_indices):
                alpha = 1.0 - (idx / (len(trail_indices) + 1))  # make older frames more faded
                alpha /= 4.0  # make the trail more faded
                self.draw_frame(self.ax_large, j, x, u, alpha=alpha, x_pred=x_pred)

            self.draw_frame(self.ax_large, i, x, u, alpha=1.0, x_pred=x_pred)
            if not save_frames:
                self.update_small_axes(x, u, i)

            if save_frames and i in frame_indices:
                plt.savefig(os.path.join(frame_folder, f"frame_{i}.png"))

        ani = FuncAnimation(plt.gcf(), update_frame, frames=self.N_sim + 1, repeat=True, interval=10)  # type: ignore

        if save_video:
            writer = FFMpegWriter(fps=30, metadata=dict(artist="Me"), bitrate=1800)
            ani.save(video_filename, writer=writer)
        else:
            plt.show()


@dataclass
class CartPendulumParameters:

    l: float
    """Lenght of the pendulum"""

    m1: float
    """Mass of the cart"""

    m2: float
    """Mass of the pendulum"""

    b1: float
    """Damping coefficient for the cart"""

    b2: float
    """Damping coefficient for the pendulum"""


class CartPendulum(BaseSystem):
    def __init__(self):
        r"""
        A dynamic system representing a cart-pendulum model.

        This class models the dynamics of a cart-pendulum system, where a pendulum
        is attached to a cart that moves along a linear track.

        Attributes:
        ---
            - p (Parameters): The physical parameters of the system, including:
                - l (float): Length of the pendulum.
                - m1 (float): Mass of the cart.
                - m2 (float): Mass of the pendulum.
                - b1 (float): Damping coefficient for the cart.
                - b2 (float): Damping coefficient for the pendulum.
            - f1 (callable): Lambda function representing the horizontal acceleration
                of the cart $\ddot{x}$ based on the system state and control input.
            - f2 (callable): Lambda function representing the angular acceleration
                of the pendulum $\ddot{\theta}$ based on the system state and control input.
            - f (callable): Lambda function representing the full state-space dynamics
                of the system, combining $[x, \theta, \dot{x}, \dot{\theta}]$ and their derivatives.
        """
        super().__init__("cart_pendulum", 4, 1)
        self.p = CartPendulumParameters(l=1, m1=2, m2=1, b1=0, b2=0)
        self.f1 = (
            lambda x, u: (
                self.p.l * self.p.m2 * cs.sin(x[1]) * x[3] ** 2 + u + self.p.m2 * self.g * cs.cos(x[1]) * cs.sin(x[1])
            )
            / (self.p.m1 + self.p.m2 * (1 - cs.cos(x[1]) ** 2))
            - self.p.b1 * x[2]
        )
        """Equation of motion of the cart, gives the horizontal acceleration of the cart."""
        self.f2 = (
            lambda x, u: -(
                self.p.l * self.p.m2 * cs.cos(x[1]) * cs.sin(x[1]) * x[3] ** 2
                + u * cs.cos(x[1])
                + (self.p.m1 + self.p.m2) * self.g * cs.sin(x[1])
            )
            / (self.p.l * self.p.m1 + self.p.l * self.p.m2 * (1 - cs.cos(x[1]) ** 2))
            - self.p.b2 * x[3]
        )
        """Euqation of motion of the pendulum, gives the horizontal acceleration of the cart."""

        self.f = lambda x, u: cs.vertcat(x[2:4], self.f1(x, u), self.f2(x, u))
        """State space dynamics of the entire system."""

    @override
    def draw_frame(self, ax, i, x, u, alpha=1.0, x_pred=None):
        ax.axis((-1.5, 1.5, -1.5, 1.5))
        ax.set_aspect("equal")

        if x_pred is not None:
            x_p = x_pred[i]
            tip_x_pred = x_p[0, :] + np.sin(x_p[1, :])
            tip_y_pred = -np.cos(x_p[1, :])
            ax.plot(tip_x_pred, tip_y_pred, color="orange", alpha=alpha)

        ax.plot(
            x[0, i] + np.array((self.p.l, self.p.l, -self.p.l, -self.p.l, +self.p.l)) / 4,
            np.array((self.p.l, -self.p.l, -self.p.l, self.p.l, self.p.l)) / 4,
            color="orange",
            alpha=alpha,
        )
        ax.add_patch(
            plt.Circle((x[0, i] + math.sin(x[1, i]), -math.cos(x[1, i])), self.p.l / 8, color="blue", alpha=alpha)  # type: ignore
        )
        ax.plot(
            np.array((x[0, i], x[0, i] + math.sin(x[1, i]))),
            np.array((0, -math.cos(x[1, i]))),
            color="black",
            alpha=alpha,
        )

    def constraints(self, x, u):
        # h1 = x[2]
        h2 = x[1] - cs.pi
        return cs.vertcat(h2)
