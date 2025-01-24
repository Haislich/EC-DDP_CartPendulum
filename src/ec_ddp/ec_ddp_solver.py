import time

import casadi as cs
import numpy as np

from ec_ddp import model

DELTA: float = 0.01
"""Timestep for the simulation"""

N_TIMESTEPS: int = 100
"""Number of steps in the horizon"""

MAX_DDP_ITERS: int = 10
"""The maximum number of iterations for the DDP algorithm."""

MAX_LINE_SEARCH_ITERS: int = 10
"""The maximum number of line search iterations."""


class Solver:
    def __init__(self, model: model.BaseSystem = model.CartPendulum()) -> None:
        self.model = model
        self.n: int = model.n
        """Number of state variables"""
        self.m: int = model.m
        """Number of control inputs"""
        self.Q_RUNNING = np.eye(self.n) * 0
        """Weight matrix for the running cost."""
        self.Q_TERMINAL = np.eye(self.n) * 10000
        """Weight matrix for the terminal state cost."""

        self.R = np.eye(self.m) * 0.01
        """Weight matrix for the control inputs."""

        # TODO: By looking at the constraint function it could make sense to somehow couple this two things.
        # What I mean by that is that the constraint depends on the terminal state and vicevers
        # This will have the added benefit that we can be more general over the model which could become a normal variable.
        # Until that is sorted the cart pendulum will be our system of interest.
        if model.name != "cart_pendulum":
            raise ValueError("At the time only the cart pendulum model is supported.")
        self.X_TERMINAL = np.array((0, cs.pi, 0, 0))
        """Desired terminal state of the system"""

        self.OPT = cs.Opti()
        self.X = self.OPT.variable(self.n)
        """Symbolic decision variable that represents the state of the system."""
        self.U = self.OPT.variable(self.m)
        """Symbolic decision variable that represents the control input."""

        self.h_dim = model.constraints(self.X, self.U).shape[0]
        self.LAMBDAS = self.OPT.variable(self.h_dim)
        """Lagrangian multipliers associated with the constraint."""
        self.MU = self.OPT.parameter()
        """The penalty parameter in the augmented lagrangian method."""

        self.H = cs.Function("H", [self.X, self.U], [self.model.constraints(self.X, self.U)], {"post_expand": True})
        """Symbolic function named `h` for the constraints."""

        self.HU = cs.Function(
            "HU", [self.X, self.U], [cs.jacobian(self.H(self.X, self.U), self.U)], {"post_expand": True}
        )
        self.HX = cs.Function(
            "HX", [self.X, self.U], [cs.jacobian(self.H(self.X, self.U), self.X)], {"post_expand": True}
        )
        self.HUU = cs.Function(
            "HU", [self.X, self.U], [cs.jacobian(self.HU(self.X, self.U), self.U)], {"post_expand": True}
        )
        self.HUX = cs.Function(
            "HUX", [self.X, self.U], [cs.jacobian(self.HU(self.X, self.U), self.X)], {"post_expand": True}
        )
        self.HXX = cs.Function(
            "HXX", [self.X, self.U], [cs.jacobian(self.HX(self.X, self.U), self.X)], {"post_expand": True}
        )
        # TODO: This definition might be useless.
        L = cs.Function("L", [self.X, self.U], [self.running_cost(self.X, self.U)], {"post_expand": True})
        """Symbolic function for the running cost."""

        self.L_LAGRANGIAN = cs.Function(
            "L_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [self.augmented_lagrangian_cost(self.X, self.U)],
            {"post_expand": True},
        )
        """Symbolic function for the lagragian extended running cost."""
        self.LU_LAGRANGIAN = cs.Function(
            "LU_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [cs.jacobian(self.L_LAGRANGIAN(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        """First order partial derivative of the lagrangian with respect to `u`"""
        self.LX_LAGRANGIAN = cs.Function(
            "LX_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [cs.jacobian(self.L_LAGRANGIAN(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """First order partial derivative of the lagrangian with respect to `x`"""
        self.LUU_LAGRANGIAN = cs.Function(
            "LU_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [cs.jacobian(self.LU_LAGRANGIAN(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        """Second order partial derivative of the lagrangian with respect to `u`"""
        self.LUX_LAGRANGIAN = cs.Function(
            "LX_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [cs.jacobian(self.LU_LAGRANGIAN(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """Mixed partial derivative of the lagrangian with respect to `u` and `x`"""
        self.LXX_LAGRANGIAN = cs.Function(
            "LX_LAGRANGIAN",
            [self.X, self.U, self.LAMBDAS, self.MU],
            [cs.jacobian(self.LX_LAGRANGIAN(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """Second order partial derivative of the lagrangian with respect to `x`"""

        self.L_TERMINAL = cs.Function("L_TERMINAL", [self.X], [self.terminal_cost(self.X)], {"post_expand": True})
        """Symbolic function for the terminal cost."""
        self.LX_TERMINAL = cs.Function(
            "LX_TERMINAL", [self.X], [cs.jacobian(self.L_TERMINAL(self.X), self.X)], {"post_expand": True}
        )
        """First order partial derivative of the terminal cost with respect to `x`"""
        self.LXX_TERMINAL = cs.Function(
            "LX_TERMINAL", [self.X], [cs.jacobian(self.LX_TERMINAL(self.X), self.X)], {"post_expand": True}
        )
        """Second order partial derivative of the terminal cost with respect to `x`"""

        self.F_DISCRETE = cs.Function(
            "F_DISCRETE", [self.X, self.U], [self.X + DELTA * model.f(self.X, self.U)], {"post_expand": True}
        )
        """Symbolic function representing the discretized system dynamics. The discretization is done by using the forward Euler method."""
        self.FU_DISCRETE = cs.Function(
            "FU_DISCRETE",
            [self.X, self.U],
            [cs.jacobian(self.F_DISCRETE(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        """Second order partial derivative of the discretized dynamics with respect to `u`"""
        self.FX_DISCRETE = cs.Function(
            "FX_DISCRETE",
            [self.X, self.U],
            [cs.jacobian(self.F_DISCRETE(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """First order partial derivative of the discretized dynamics with respect to `x`"""
        self.FUU_DISCRETE = cs.Function(
            "FUU_DISCRETE",
            [self.X, self.U],
            [cs.jacobian(self.FU_DISCRETE(self.X, self.U), self.U)],
            {"post_expand": True},
        )
        """Second order partial derivative of the discretized dynamics with respect to `u`"""
        self.FUX_DISCRETE = cs.Function(
            "FUX_DISCRETE",
            [self.X, self.U],
            [cs.jacobian(self.FU_DISCRETE(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """Second order partial derivative of the discretized dynamics with respect to `u`"""
        self.FXX_DISCRETE = cs.Function(
            "FXX_DISCRETE",
            [self.X, self.U],
            [cs.jacobian(self.FX_DISCRETE(self.X, self.U), self.X)],
            {"post_expand": True},
        )
        """Second order partial derivative of the discretized dynamics with respect to `x`"""

    def running_cost(self, x, u):
        """
        Defines the running cost for the system.
        Penalizes trajectories of the current state from the desided state.
        """
        return (self.X_TERMINAL - x).T @ self.Q_RUNNING @ (self.X_TERMINAL - x) + u.T @ self.R @ u

    def augmented_lagrangian_cost(self, x, u):
        """
        Extends the cost function to include the equaliti constraint, using an augmented lagrangian approach.
        """
        return self.running_cost(x, u) + self.LAMBDAS.T @ self.H(x, u) + (self.MU / 2) * cs.sumsqr(self.H(x, u))

    def terminal_cost(self, x):
        """
        Defines the terminal cost, it penalizes deviationsof the final state from the desired final state.
        """
        return (self.X_TERMINAL - x).T @ self.Q_TERMINAL @ (self.X_TERMINAL - x)

    def equality_constrained_ddp(
        self,
        eta: float = 20.0,
        omega: float = 20.0,
        beta: float = 0.5,
        alpha: float = 0.1,
        k_mu=10,
        etha_threshold=10,
        omega_threshold: int = 10,
        max_iterations: int = 20,
        max_line_search_iters=10,
    ):
        x = np.zeros((self.n, N_TIMESTEPS))
        """State trajectory"""
        u = np.ones((self.n, N_TIMESTEPS))
        """Control sequence"""
        mu_zero = 1.0
        lambda_zero = np.zeros(self.h_dim)

        # Initial Forward pass: We propagate the state trajectory by using the discretized dynamics to get an initial guess.
        cost = 0
        for timestep in range(N_TIMESTEPS):
            x[:, timestep + 1] = np.array(self.F_DISCRETE(x[:, timestep], u[:, timestep])).flatten()
            # Accumulate the running cost at each step
            cost += self.L_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
        # Adn finally we add the terminal cost at the end of the trajectory
        cost += self.L_TERMINAL(x[:, N_TIMESTEPS])  # type: ignore

        V = np.zeros(N_TIMESTEPS + 1)
        """Value function"""
        VX = np.zeros((self.n, N_TIMESTEPS + 1))
        VXX = np.zeros((self.n, self.n, N_TIMESTEPS + 1))

        k = [np.zeros((self.m, 1))] * (self.n + 1)
        K = [np.zeros((self.m, self.n))] * (self.n + 1)

        total_time = 0

        for iteration in range(max_iterations):
            if eta > etha_threshold or omega > omega_threshold:
                break
            # Backward pass:

            backward_pass_start_time = time.time()
            # For Equation 4, the value at the final sequence element is the terminal value
            V[self.n] = self.L_TERMINAL(x[:, self.n])
            VX[:, self.n] = np.array(self.L_TERMINAL(x[:, self.n])).flatten()
            VXX[:, :, self.n] = self.L_TERMINAL(x[:, self.n])

            for timestep in reversed(range(N_TIMESTEPS)):
                fx_eval = self.FX_DISCRETE(x[:, timestep], u[:, timestep])
                fu_eval = self.FU_DISCRETE(x[:, timestep], u[:, timestep])
                fuu_eval = self.FUU_DISCRETE(x[:, timestep], u[:, timestep])
                fux_eval = self.FUX_DISCRETE(x[:, timestep], u[:, timestep])
                fxx_eval = self.FXX_DISCRETE(x[:, timestep], u[:, timestep])

                h_eval = self.H(x[:, timestep], u[:, timestep])
                hx_eval = self.HX(x[:, timestep], u[:, timestep])
                hu_eval = self.HU(x[:, timestep], u[:, timestep])
                huu_eval = self.HUU(x[:, timestep], u[:, timestep])
                hux_eval = self.HUX(x[:, timestep], u[:, timestep])
                hxx_eval = self.HXX(x[:, timestep], u[:, timestep])

                # Equation 9f
                q = (
                    self.L_LAGRANGIAN(x[:, timestep], u[:, timestep])
                    + V[timestep + 1]
                    # Constraint term
                    + lambda_zero.T @ h_eval  # type: ignore# type: ignore
                    + (self.MU / 2) * cs.sumsqr(h_eval)  # type: ignore
                )

                # Equation 9e
                QU = (
                    self.LU_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                    + VX[:, timestep + 1].T @ fu_eval  # type: ignore
                    # Augmented lagrangian term
                    + (lambda_zero + mu_zero * h_eval).T @ hu_eval  # type: ignore
                )

                # Equation 9d
                QX = (
                    self.LX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                    + VX[:, timestep + 1].T @ fx_eval  # type: ignore
                    # Augmented lagrangian term
                    + (lambda_zero + mu_zero * h_eval).T @ hx_eval  # type: ignore
                )

                # Equation 9c
                QUU = (
                    self.LUU_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                    + fu_eval.T @ VXX[:, timestep + 1] @ fu_eval  # type: ignore
                    + VX[:, timestep + 1].T @ fuu_eval  # type: ignore
                    # Augmented lagrangian term
                    + (lambda_zero + mu_zero * h_eval).T @ huu_eval  # type: ignore
                    + mu_zero * hu_eval.T @ hu_eval  # type: ignore
                )
                # Equation 9b
                QUX = (
                    self.LX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                    + fx_eval.T @ VX[:, timestep + 1]  # type: ignore
                    + VX[:, timestep + 1].T @ fux_eval  # type: ignore
                    # Augmented lagrangian term
                    + (lambda_zero + mu_zero * h_eval).T @ hux_eval  # type: ignore
                    + mu_zero * hu_eval.T @ hx_eval  # type: ignore
                )
                # Equation 9a
                QXX = (
                    self.LXX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                    + fx_eval.T @ VXX[:, timestep + 1] @ fx_eval  # type: ignore
                    + VX[:, timestep + 1].T @ fxx_eval  # type: ignore
                    # Augmented lagrangian term
                    + (lambda_zero + mu_zero * h_eval).T @ hxx_eval  # type: ignore
                    + mu_zero * hx_eval.T @ hx_eval  # type: ignore
                )

                # Necessary later for computing the optimal `u*`
                QUU_INV = np.linalg.inv(QUU)
                k[timestep] = -QUU_INV @ QU
                K[timestep] = -QUU_INV @ QUX

                # Value function approximation
                # TODO: Understand betyter Valerio's code
                # Equation 11a
                V[timestep] = q - 0.5 * np.array(cs.evalf(k[timestep].T @ QUU @ k[timestep])).flatten()[0]
                # Equation 11b
                VX[:, timestep] = np.array(QX - k[timestep].T @ QUU @ K[timestep]).flatten()
                # Equation 11c
                VXX[:, :, timestep] = QXX - K[timestep].T @ QUU @ K[timestep]
            backward_pass_time = time.time() - backward_pass_start_time
            backward_pass_time = round(backward_pass_time * 1000)
            # Forward pass
            forward_pass_start_time = time.time()
            u_new = np.ones((self.m, N_TIMESTEPS))
            x_new = np.zeros((self.n, N_TIMESTEPS))
            x_new[:, 0] = x[:, 0]

            for line_search_iter in range(max_line_search_iters):
                new_cost = 0
                # We propagate the state trajectory by using the discretized dynamics.
                for timestep in range(N_TIMESTEPS):
                    x_new[:, timestep + 1] = np.array(self.F_DISCRETE(x[:, timestep], u[:, timestep])).flatten()
                    # Accumulate the running cost at each step
                    new_cost += self.L_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                # Adn finally we add the terminal cost at the end of the trajectory
                new_cost += self.L_TERMINAL(x[:, N_TIMESTEPS])  # type: ignore

                if new_cost < cost:
                    cost = new_cost
                    x = x_new
                    u = u_new
                    break
                alpha /= 2.0
            # TODO: Ask valerio where does this come from.
            if (
                np.linalg.norm(self.LX_LAGRANGIAN(x_new[:, N_TIMESTEPS - 1], u_new[:, N_TIMESTEPS - 1]))
                < omega  # type:ignore
            ):
                if np.linalg.norm(self.H(x_new[:, N_TIMESTEPS - 1], u_new[:, N_TIMESTEPS - 1])) < eta:  # type:ignore
                    lambda_zero += mu_zero * self.H(x_new[:, self.N - 1], u_new[:, N_TIMESTEPS - 1])  # type:ignore
                    eta /= mu_zero**beta
                    omega = omega // mu_zero
                else:
                    mu_zero *= k_mu

            forward_pass_time = time.time() - forward_pass_start_time
            forward_pass_time = round(backward_pass_time * 1000)

            total_time += backward_pass_time + forward_pass_time
            print(
                f"{iteration=},{backward_pass_time=},{forward_pass_time=}" + "||h(x, u)||:",
                np.linalg.norm(self.H(x[:, :-1], u)),  # type: ignore
            )
            print("Total time: ", total_time * 1000, " ms")
            # check result
            x_check = np.zeros((self.n, N_TIMESTEPS + 1))
            x_check[:, 0] = np.zeros(self.n)
            for i in range(N_TIMESTEPS):
                x_check[:, i + 1] = np.array(self.F_DISCRETE(x_check[:, i], u[:, i])).flatten()

            # display
            self.model.animate(N_TIMESTEPS, x_check, u)


Solver().equality_constrained_ddp()
