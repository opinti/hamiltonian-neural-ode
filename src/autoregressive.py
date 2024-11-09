import torch
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Union


def autoregressive_rollout_ivp(
    model: torch.nn.Module,
    initial_conditions: Union[torch.Tensor, np.ndarray],
    t_span: Tuple[float, float],
    n_steps: int,
    method: str = "solve_ivp",
) -> torch.Tensor:
    """
    Applies the model autoregressively to simulate the system's trajectory using scipy's solve_ivp or Euler's method.

    Args:
        model (torch.nn.Module): The model to use in autoregressive rollout.
        initial_conditions (Union[torch.Tensor, np.ndarray]): Initial conditions on canonical coordinates,
            shape (1, 2 * n_dim).
        t_span (Tuple[float, float]): Time span for integration, e.g., (0, T).
        n_steps (int): Number of integration steps.
        method (str): Integration method, either 'solve_ivp' or 'euler'.

    Returns:
        torch.Tensor: Trajectories over time, shape (n_steps, batch_size, 2 * n_dim).
    """
    if initial_conditions.ndim == 1:
        pass
    elif initial_conditions.ndim == 2:
        assert initial_conditions.shape[0] == 1, "Batching not supported"
    else:
        raise ValueError("Invalid shape for initial conditions")

    if isinstance(initial_conditions, np.ndarray):
        initial_conditions = torch.tensor(initial_conditions, dtype=torch.float32)

    if method == "solve_ivp":
        states = _solve_ivp_rollout(model, initial_conditions, t_span, n_steps)
        return torch.tensor(states, dtype=torch.float32)
    elif method == "euler":
        return _euler_rollout(model, initial_conditions, t_span, n_steps)
    else:
        raise ValueError("Method must be either 'solve_ivp' or 'euler'")


def _solve_ivp_rollout(
    model: torch.nn.Module,
    initial_conditions: torch.Tensor,
    t_span: Tuple[float, float],
    n_steps: int,
) -> np.ndarray:
    """
    Simulates the system's trajectory using scipy's solve_ivp integration method.

    Args:
        model (torch.nn.Module): The model to compute the state derivatives.
        initial_conditions (torch.Tensor): Initial conditions of the system, shape (1, 2 * n_dim).
        t_span (Tuple[float, float]): Time span for integration.
        n_steps (int): Number of integration steps.

    Returns:
        np.ndarray: Array of states over time, shape (n_steps, 2 * n_dim).
    """
    state0 = initial_conditions.detach().numpy().reshape(-1)

    def state_dot(t: float, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.tensor(
            state, dtype=torch.float32, requires_grad=True
        ).reshape(1, -1)
        return model(state_tensor).detach().numpy()

    t_eval = np.linspace(t_span[0], t_span[1], n_steps)
    states = solve_ivp(
        fun=state_dot, t_span=t_span, y0=state0, t_eval=t_eval, method="RK45"
    )

    return states.y.T


def _euler_rollout(
    model: torch.nn.Module,
    initial_conditions: torch.Tensor,
    t_span: Tuple[float, float],
    n_steps: int,
) -> torch.Tensor:
    """
    Simulates the system's trajectory using the forward Euler integration method.

    Args:
        model (torch.nn.Module): The model to compute the state derivatives.
        initial_conditions (torch.Tensor): Initial conditions of the system, shape (1, 2 * n_dim).
        t_span (Tuple[float, float]): Time span for integration.
        n_steps (int): Number of integration steps.

    Returns:
        torch.Tensor: Array of states over time, shape (n_steps, 2 * n_dim).
    """
    n_dim = initial_conditions.shape[1]
    states = torch.zeros(n_steps, n_dim)
    dt = (t_span[1] - t_span[0]) / n_steps

    state = initial_conditions
    for i in range(n_steps):
        states[i, :] = state
        state_dot = model(state)
        state = state + dt * state_dot

    return states
