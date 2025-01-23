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
N_TIME_STEPS: int = 100
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

opt = cs.Opti()
X = opt.variable(N)
"""Symbolic decision variable that represents the state of the system."""
U = opt.variable(M)
"""Symbolic decision variable that represents the control input."""
H = cs.Function("h", [X, U], [MODEL.constraints(X, U)], {"post_expand": True})
"""Symbolic function named `h` for the constraints."""
h_dim = MODEL.constraints(X, U).shape[0]
LAMBDAS = opt.variable(h_dim)
"""Lagrangian multipliers associated with the constraint."""
MU = opt.parameter()
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


L = cs.Function("L", [X, U], [running_cost(X, U)], {"post_expand": True})
"""Symbolic function for the running cost."""
L_LAGRANGIAN = cs.Function("L_LAGRANGIAN", [X, U], [augmented_lagrangian_cost(X, U)], {"post_expand": True})
"""Symbolic function for the lagragian extended running cost."""
L_TERMINAL = cs.Function("L_TERMINAL", [X], [terminal_cost(X)], {"post_expand": True})
"""Symbolic function for the terminal cost."""

LX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(L_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""First order partial derivative of the lagrangian with respect to `x`"""
LU_LAGRANGIAN = cs.Function("LU_LAGRANGIAN", [X, U], [cs.jacobian(L_LAGRANGIAN(X, U), U)], {"post_expand": True})
"""First order partial derivative of the lagrangian with respect to `u`"""

LXX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(LX_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""Second order partial derivative of the lagrangian with respect to `x`"""
LUU_LAGRANGIAN = cs.Function("LU_LAGRANGIAN", [X, U], [cs.jacobian(LU_LAGRANGIAN(X, U), U)], {"post_expand": True})
"""Second order partial derivative of the lagrangian with respect to `u`"""
LUX_LAGRANGIAN = cs.Function("LX_LAGRANGIAN", [X, U], [cs.jacobian(LU_LAGRANGIAN(X, U), X)], {"post_expand": True})
"""Mixed partial derivative of the lagrangian with respect to `u` and `x`"""

LX_TERMINAL = cs.Function("LX_TERMINAL", [X, U], [cs.jacobian(L_TERMINAL(X, U), X)], {"post_expand": True})
"""First order partial derivative of the terminal cost with respect to `x`"""
LXX_TERMINAL = cs.Function("LX_TERMINAL", [X, U], [cs.jacobian(LX_TERMINAL(X, U), X)], {"post_expand": True})
"""Second order partial derivative of the terminal cost with respect to `x`"""
