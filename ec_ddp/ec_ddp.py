import time

import casadi as cs
import numpy as np

import ec_ddp.model as model

MODEL = model.CartPendulum()
DELTA: float = 0.01
"""Timestep for the simulation"""
N: int = MODEL.n
"""Number of state variables"""
M: int = MODEL.m
"""Number of control inputs"""
N_TIMESTEPS: int = 100
"""Number of steps in the horizon"""
MAX_DDP_ITERS: int = 10
"""The maximum number of iterations for the DDP algorithm."""
MAX_LINE_SEARCH_ITERS: int = 10
"""The maximum number of line search iterations."""

Q_RUNNING = np.eye(N) * 0
"""Weight matrix for the running cost."""
Q_TERMINAL = np.eye(N) * 10000
"""Weight matrix for the terminal state cost."""
R = np.eye(M) * 0.01
"""Weight matrix for the control inputs."""

# TODO: By looking at the constraint function it could make sense to somehow couple this two things.
# What I mean by that is that the constraint depends on the terminal state and vicevers
# This will have the added benefit that we can be more general over the model which could become a normal variable.
X_TERMINAL = np.array((0, cs.pi, 0, 0))
"""Desired terminal state of the system"""

OPT = cs.Opti()
X = OPT.variable(N)
"""Symbolic decision variable that represents the state of the system."""
U = OPT.variable(M)
"""Symbolic decision variable that represents the control input."""
H_DIM = MODEL.constraints(X, U).shape[0]
H = cs.Function("H", [X, U], [MODEL.constraints(X, U)], {"post_expand": True})
"""Symbolic function named `h` for the constraints."""
LAMBDAS = OPT.variable(H_DIM)
"""Lagrangian multipliers associated with the constraint."""
MU = OPT.parameter()
"""The penalty parameter in the augmented lagrangian method."""


def running_cost(x, u):
    """
    Defines the running cost for the system.
    Penalizes trajectories of the current state from the desided state.
    """
    return (X_TERMINAL - x).T @ Q_RUNNING @ (X_TERMINAL - x) + u.T @ R @ u


def augmented_lagrangian_cost(x, u):
    """
    Extends the cost function to include the equaliti constraint, using an augmented lagrangian approach.
    """
    return running_cost(x, u) + LAMBDAS.T @ H(x, u) + (MU / 2) * cs.sumsqr(H(x, u))


def terminal_cost(x):
    """
    Defines the terminal cost, it penalizes deviationsof the final state from the desired final state.
    """
    return (X_TERMINAL - x).T @ Q_TERMINAL @ (X_TERMINAL - x)


# TODO: This definition might be useless.
L = cs.Function("L", [X, U], [running_cost(X, U)], {"post_expand": True})
"""Symbolic function for the running cost."""

L_LAGRANGIAN = cs.Function("L_LAGRANGIAN", [X, U], [augmented_lagrangian_cost(X, U)], {"post_expand": True})
"""Symbolic function for the lagragian extended running cost."""
LU_LAGRANGIAN = cs.Function("LU_LAGRANGIAN", [X, U], [cs.jacobian(L_LAGRANGIAN(X, U), U)], {"post_expand": True})
"""First order partial derivative of the lagrangian with respect to `u`"""
LX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(L_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""First order partial derivative of the lagrangian with respect to `x`"""
LUU_LAGRANGIAN = cs.Function("LU_LAGRANGIAN", [X, U], [cs.jacobian(LU_LAGRANGIAN(X, U), U)], {"post_expand": True})
"""Second order partial derivative of the lagrangian with respect to `u`"""
LUX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(LU_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""Mixed partial derivative of the lagrangian with respect to `u` and `x`"""
LXX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(LX_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""Second order partial derivative of the lagrangian with respect to `x`"""

L_TERMINAL = cs.Function("L_TERMINAL", [X], [terminal_cost(X)], {"post_expand": True})
"""Symbolic function for the terminal cost."""
LX_TERMINAL = cs.Function("LX_TERMINAL", [X], [cs.jacobian(L_TERMINAL(X), X)], {"post_expand": True})
"""First order partial derivative of the terminal cost with respect to `x`"""
LXX_TERMINAL = cs.Function("LX_TERMINAL", [X], [cs.jacobian(LX_TERMINAL(X), X)], {"post_expand": True})
"""Second order partial derivative of the terminal cost with respect to `x`"""

F_DISCRETE = cs.Function("F_DISCRETE", [X, U], [X + DELTA * MODEL.f(X, U)], {"post_expand": True})
"""Symbolic function representing the discretized system dynamics. The discretization is done by using the forward Euler method."""
FU_DISCRETE = cs.Function("FU_DISCRETE", [X, U], [cs.jacobian(F_DISCRETE(X, U), U)], {"post_expand": True})
"""Second order partial derivative of the discretized dynamics with respect to `u`"""
FX_DISCRETE = cs.Function("FX_DISCRETE", [X, U], [cs.jacobian(F_DISCRETE(X, U), X)], {"post_expand": True})
"""First order partial derivative of the discretized dynamics with respect to `x`"""
FUU_DISCRETE = cs.Function("FUU_DISCRETE", [X, U], [cs.jacobian(FU_DISCRETE(X, U), U)], {"post_expand": True})
"""Second order partial derivative of the discretized dynamics with respect to `u`"""
FUX_DISCRETE = cs.Function("FUX_DISCRETE", [X, U], [cs.jacobian(FX_DISCRETE(X, U), X)], {"post_expand": True})
"""Second order partial derivative of the discretized dynamics with respect to `u`"""
FXX_DISCRETE = cs.Function("FXX_DISCRETE", [X, U], [cs.jacobian(FX_DISCRETE(X, U), X)], {"post_expand": True})
"""Second order partial derivative of the discretized dynamics with respect to `x`"""

HU = cs.Function("HU", [X, U], [cs.jacobian(H(X, U), U)], {"post_expand": True})
HX = cs.Function("HX", [X, U], [cs.jacobian(H(X, U), X)], {"post_expand": True})
HUU = cs.Function("HU", [X, U], [cs.jacobian(HU(X, U), U)], {"post_expand": True})
HUX = cs.Function("HUX", [X, U], [cs.jacobian(HU(X, U), X)], {"post_expand": True})
HXX = cs.Function("HXX", [X, U], [cs.jacobian(HX(X, U), X)], {"post_expand": True})


def equality_constrained_ddp(
    total_time: int = 0,
    eta: int = 20,
    omega: int = 20,
    beta: float = 0.5,
    etha_threshold=10,
    omega_threshold: int = 10,
    max_iterations: int = 20,
):
    x = np.zeros((N, N_TIMESTEPS))
    """State trajectory"""
    u = np.ones((M, N_TIMESTEPS))
    """Control sequence"""
    mu_zero = 1.0
    lambda_zero = np.zeros(H_DIM)

    # Forward pass: We propagate the state trajectory by using the discretized dynamics.
    cost = 0
    for timestep in range(N_TIMESTEPS):
        x[:, timestep + 1] = np.array(F_DISCRETE(x[:, timestep], u[:, timestep])).flatten()
        # Accumulate the running cost at each step
        cost += L_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
    # Adn finally we add the terminal cost at the end of the trajectory
    cost += L_TERMINAL(x[:, N_TIMESTEPS])  # type: ignore

    # Backpropagation:

    V = np.zeros(N_TIMESTEPS + 1)
    """Value function"""
    VX = np.zeros((N, N_TIMESTEPS + 1))
    VXX = np.zeros((N, N, N_TIMESTEPS + 1))

    k = [np.zeros((M, 1))] * (N + 1)
    K = [np.zeros((M, N))] * (N + 1)

    for iteration in range(max_iterations):
        if eta > etha_threshold or omega > omega_threshold:
            break
        backward_pass_start_time = time.time()
        # For Equation 4, the value at the final sequence element is the terminal value
        V[N] = L_TERMINAL(x[:, N])
        VX[:, N] = np.array(L_TERMINAL(x[:, N])).flatten()
        VXX[:, :, N] = L_TERMINAL(x[:, N])

        for timestep in reversed(range(N_TIMESTEPS)):
            fx_eval = FX_DISCRETE(x[:, timestep], u[:, timestep])
            fu_eval = FU_DISCRETE(x[:, timestep], u[:, timestep])
            fuu_eval = FUU_DISCRETE(x[:, timestep], u[:, timestep])
            fux_eval = FUX_DISCRETE(x[:, timestep], u[:, timestep])
            fxx_eval = FXX_DISCRETE(x[:, timestep], u[:, timestep])

            h_eval = H(x[:, timestep], u[:, timestep])
            hx_eval = HX(x[:, timestep], u[:, timestep])
            hu_eval = HU(x[:, timestep], u[:, timestep])
            huu_eval = HUU(x[:, timestep], u[:, timestep])
            hux_eval = HUX(x[:, timestep], u[:, timestep])
            hxx_eval = HXX(x[:, timestep], u[:, timestep])

            # Equation 9f
            q = (
                L_LAGRANGIAN(x[:, timestep], u[:, timestep])
                + V[timestep + 1]
                # Constraint term
                + lambda_zero.T @ h_eval  # type: ignore# type: ignore
                + (MU / 2) * cs.sumsqr(h_eval)  # type: ignore
            )

            # Equation 9e
            QU = (
                LU_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                + VX[:, timestep + 1].T @ fu_eval  # type: ignore
                # Augmented lagrangian term
                + (lambda_zero + mu_zero * h_eval).T @ hu_eval  # type: ignore
            )

            # Equation 9d
            QX = (
                LX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                + VX[:, timestep + 1].T @ fx_eval  # type: ignore
                # Augmented lagrangian term
                + (lambda_zero + mu_zero * h_eval).T @ hx_eval  # type: ignore
            )

            # Equation 9c
            QUU = (
                LUU_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                + fu_eval.T @ VXX[:, timestep + 1] @ fu_eval  # type: ignore
                + VX[:, timestep + 1].T @ fuu_eval  # type: ignore
                # Augmented lagrangian term
                + (lambda_zero + mu_zero * h_eval).T @ huu_eval  # type: ignore
                + mu_zero * hu_eval.T @ hu_eval  # type: ignore
            )
            # Equation 9b
            QUX = (
                LX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
                + fx_eval.T @ VX[:, timestep + 1]  # type: ignore
                + VX[:, timestep + 1].T @ fux_eval  # type: ignore
                # Augmented lagrangian term
                + (lambda_zero + mu_zero * h_eval).T @ hux_eval  # type: ignore
                + mu_zero * hu_eval.T @ hx_eval  # type: ignore
            )
            # Equation 9a
            QXX = (
                LXX_LAGRANGIAN(x[:, timestep], u[:, timestep])  # type: ignore
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
            V[timestep] = q - 0.5 * np.array(cs.evalf(k[timestep].T @ QUU @ k[timestep])).flatten()[0]
            VX[:, timestep] = np.array(QX - k[timestep].T @ QUU @ K[timestep]).flatten()
            VXX[:, :, timestep] = QXX - K[timestep].T @ QUU @ K[timestep]
