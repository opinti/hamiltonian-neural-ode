import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Tuple, Union, Dict


class Pendulum:
    """
    A class to represent a pendulum system.

    The generalized coordinates are defined as:
    - q: Angle of the pendulum.
    - q_dot: Angular velocity of the pendulum.
    """

    GRAVITY = 10.0
    MASS = 1.0
    LENGTH = 1.0

    def __init__(
        self,
        gravity: float = None,
        mass: float = None,
        length: float = None,
    ) -> None:
        """
        Initializes the pendulum with the specified parameters.

        Args:
            gravity (float, optional): Gravity acceleration.
            mass (float, optional): Mass of the pendulum bob.
            length (float, optional): Length of the pendulum rod.
        """
        self.gravity = gravity if gravity else self.GRAVITY
        self.mass = mass if mass else self.MASS
        self.length = length if length else self.LENGTH

    def generalized_to_canonical(
        self, generalized_coords: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Converts generalized coordinates and velocities to canonical coordinates.

        Args:
            generalized_coords (Tuple[float, float]): Generalized coordinates (q, q_dot).

        Returns:
            Tuple[float, float]: Canonical coordinates (q, p).
        """
        q, q_dot = generalized_coords
        p = self.mass * self.length**2 * q_dot
        return q, p

    def canonical_to_generalized(
        self, canonical_coords: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Converts canonical coordinates to generalized coordinates and velocities.

        Args:
            canonical_coords (Tuple[float, float]): Canonical coordinates (q, p).

        Returns:
            Tuple[float, float]: Generalized coordinates (q, q_dot).
        """
        q, p = canonical_coords
        q_dot = p / (self.mass * self.length**2)
        return q, q_dot

    def generalized_to_cartesian(
        self, generalized_coords: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts generalized coordinates and velocities to Cartesian coordinates and velocities.

        Args:
            generalized_coords (Tuple[float, float]): Generalized coordinates (q, q_dot).

        Returns:
            Tuple[float, float, float, float]: Cartesian coordinates (x, y, x_dot, y_dot).
        """
        q, q_dot = generalized_coords
        x = self.length * np.sin(q)
        y = -self.length * np.cos(q)
        x_dot = self.length * q_dot * np.cos(q)
        y_dot = self.length * q_dot * np.sin(q)
        return x, y, x_dot, y_dot

    def cartesian_to_generalized(
        self, cartesian_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Converts Cartesian coordinates and velocities to generalized coordinates and velocities.

        Args:
            cartesian_coords (Tuple[float, float, float, float]): Cartesian coordinates (x, y, x_dot, y_dot).

        Returns:
            Tuple[float, float]: Generalized coordinates (q, q_dot).
        """
        x, y, x_dot, y_dot = cartesian_coords
        q = np.arctan2(x, -y)
        q_dot = (x_dot**2 + y_dot**2) ** 0.5 / self.length
        return q, q_dot

    def canonical_to_cartesian(
        self, canonical_coords: Tuple[float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts canonical coordinates to Cartesian coordinates.

        Args:
            canonical_coords (Tuple[float, float]): Canonical coordinates (q, p).

        Returns:
            Tuple[float, float, float, float]: Cartesian coordinates (x, y, x_dot, y_dot).
        """
        generalized_coords = self.canonical_to_generalized(canonical_coords)
        return self.generalized_to_cartesian(generalized_coords)

    def cartesian_to_canonical(
        self, cartesian_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float]:
        """
        Converts Cartesian coordinates to canonical coordinates.

        Args:
            cartesian_coords (Tuple[float, float, float, float]): Cartesian coordinates (x, y, x_dot, y_dot).

        Returns:
            Tuple[float, float]: Canonical coordinates (q, p).
        """
        generalized_coords = self.cartesian_to_generalized(cartesian_coords)
        return self.generalized_to_canonical(generalized_coords)

    def hamiltonian(self, canonical_coords: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian of the system as a function of the canonical coordinates.

        Args:
            canonical_coords (torch.Tensor): Canonical coordinates (q, p) as a torch tensor.

        Returns:
            torch.Tensor: The Hamiltonian of the system.
        """
        q, p = canonical_coords
        hamiltonian = 0.5 * p**2 / (
            self.mass * self.length**2
        ) - self.mass * self.gravity * self.length * torch.cos(q)
        return hamiltonian

    def canonical_coords_dot(
        self, canonical_coords: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Computes the derivative of the Hamiltonian with respect to the canonical coordinates.

        Args:
            canonical_coords (Tuple[float, float]): Canonical coordinates (q, p) as torch tensors.

        Returns:
            Tuple[float, float]: Derivatives [dq/dt, dp/dt].
        """
        q, p = [torch.tensor(coord, requires_grad=True) for coord in canonical_coords]
        coords = [q, p]
        hamiltonian = self.hamiltonian(coords)
        dH_dq, dH_dp = torch.autograd.grad(hamiltonian, coords, create_graph=True)
        dq_dt = +dH_dp.item()
        dp_dt = -dH_dq.item()
        return dq_dt, dp_dt

    def generate_dataset(
        self,
        num_simulations: int,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        q_lim: Tuple[float, float] = (-3, 3),
        q_dot_lim: Tuple[float, float] = (-0.5, 0.5),
        return_trajectories: bool = False,
        noise_level: float = 0.0,
    ) -> Union[TensorDataset, Tuple[TensorDataset, torch.Tensor, torch.Tensor]]:
        """
        Initializes the dataset by generating simulation data.

        Args:
            num_simulations (int): Number of different simulations.
            t_span (Tuple[float, float]): Time span for the simulations.
            t_eval (np.ndarray): Time points to evaluate the solution.
            q_lim (Tuple[float, float], optional): Range of initial angles for the pendulum.
                Default is (-3, 3).
            q_dot_lim (Tuple[float, float], optional): Range of initial angular velocities for the pendulum.
            return_trajectories (bool, optional): If True, returns the canonical coords history.
                This is stored as a dictionary, where the key is the trajectory number and the value
                is the simulation data. Default is False.
            noise_level (float, optional): Intensity of noise applied to inputs and targets.
                A zero-mean Gaussian noise with standard deviation equal to noise_level is added to the
                canonical coordinates and their derivatives. Default is 0.0.

        Returns:
            Union[TensorDataset, Tuple[TensorDataset, torch.Tensor, torch.Tensor]]: Dataset containing simulation data.
        """
        inputs = []
        targets = []
        for _ in range(num_simulations):
            q = np.random.uniform(*q_lim)
            q_dot = np.random.uniform(*q_dot_lim)

            simulation_data = self.simulate(q, q_dot, t_span, t_eval)

            canonical_coords = torch.tensor(
                simulation_data["canonical_coords"], dtype=torch.float32
            ).reshape(-1, 2)
            canonical_coords_dot = torch.tensor(
                simulation_data["canonical_coords_dot"], dtype=torch.float32
            ).reshape(-1, 2)

            noise_coords = noise_level * torch.randn_like(canonical_coords)
            canonical_coords += noise_coords

            noise_coords_dot = noise_level * torch.randn_like(canonical_coords_dot)
            canonical_coords_dot += noise_coords_dot

            inputs.append(canonical_coords)
            targets.append(canonical_coords_dot)

        inputs_tensor = torch.cat(inputs)
        targets_tensor = torch.cat(targets)
        if return_trajectories:
            return (
                TensorDataset(inputs_tensor, targets_tensor),
                torch.stack(inputs),
                torch.stack(targets),
            )
        return TensorDataset(inputs_tensor, targets_tensor)

    def simulate(
        self,
        q: float,
        q_dot: float,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        method: str = "DOP853",
    ) -> Dict[str, np.ndarray]:
        """
        Simulates the pendulum using solve_ivp from scipy.integrate.

        Args:
            q (float): Initial angle of the pendulum.
            q_dot (float): Initial angular velocity of the bob.
            t_span (Tuple[float, float]): Time span (t0, tf).
            t_eval (np.ndarray): Time points to evaluate the solution.
            method (str): Integration method to use. Default is "DOP853".

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the simulation history.
        """
        initial_generalized_coords = [q, q_dot]
        initial_state = self.generalized_to_canonical(initial_generalized_coords)

        def state_dot(t, state):
            return self.canonical_coords_dot(state)

        solution = solve_ivp(
            fun=state_dot,
            t_span=t_span,
            y0=initial_state,
            t_eval=t_eval,
            method=method,
        )

        data = {
            "t": solution.t,
            "canonical_coords": solution.y.T,
            "canonical_coords_dot": np.array(
                [self.canonical_coords_dot(state) for state in solution.y.T]
            ),
            "generalized_coords": np.array(
                [self.canonical_to_generalized(state) for state in solution.y.T]
            ),
            "energy": self.compute_energy_history(solution.y.T),
            "initial_energy": self.energy_from_generalized(initial_generalized_coords),
        }

        data["cartesian_coords"] = np.array(
            [
                self.generalized_to_cartesian(state)
                for state in data["generalized_coords"]
            ]
        )
        return data

    def compute_energy_history(self, coords_history: np.ndarray) -> np.ndarray:
        """
        Computes the energy of the system from the history of canonical coordinates.

        Args:
            coords_history (np.ndarray): History of canonical coordinates.

        Returns:
            np.ndarray: Energy history of the system.
        """
        with torch.no_grad():
            energy_history = [
                self.hamiltonian([torch.tensor(coord) for coord in coords]).item()
                for coords in coords_history
            ]
        return np.array(energy_history)

    def energy_from_generalized(self, generalized_coords: Tuple[float, float]) -> float:
        """
        Computes the initial mechanical energy from the generalized coordinates.

        Args:
            generalized_coords (Tuple[float, float]): Generalized coordinates (q, q_dot).

        Returns:
            float: The initial mechanical energy.
        """
        q, q_dot = generalized_coords
        kinetic_energy = 0.5 * self.mass * self.length**2 * q_dot**2
        potential_energy = -self.mass * self.gravity * self.length * np.cos(q)
        return kinetic_energy + potential_energy

    def create_gif(
        self,
        t: np.ndarray,
        cart_coords: np.ndarray,
        filename: str = "pendulum.gif",
        interval: Union[int, None] = None,
        trail_length: int = 10,
    ) -> None:
        """
        Creates a GIF from the simulation history.

        Args:
            t (np.ndarray): Time history.
            cart_coords (np.ndarray): History of Cartesian coordinates, shape (n_steps, 2).
            filename (str): Name of the GIF file.
            interval (int, optional): Time interval between frames in milliseconds.
            trail_length (int): Number of previous points to show as a trail.

        Returns:
            None
        """
        if interval is None:
            interval = 1000 * (t[1] - t[0])

        x, y = cart_coords[:, 0], cart_coords[:, 1]

        fig, ax = plt.subplots()
        ax.set_title("t = 0 s", fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.5 * self.length, 1.5 * self.length)
        ax.set_ylim(-1.5 * self.length, 1.5 * self.length)

        (line,) = ax.plot([0, x[0]], [0, y[0]], "o-", c="k", lw=1.2)
        (point,) = ax.plot(x[0], y[0], "ko", markersize=4)
        (trail,) = ax.plot([], [], "k-", alpha=0.5, lw=1)

        def update(frame):
            line.set_data([0, x[frame]], [0, y[frame]])
            point.set_data([x[frame]], [y[frame]])
            start = max(0, frame - trail_length)
            trail.set_data(x[start : frame + 1], y[start : frame + 1])  # noqa: E203
            ax.set_title(f"t = {t[frame]:.1f} s", fontsize=14)
            return line, point, trail

        ani = FuncAnimation(fig, update, frames=len(x), blit=True, interval=interval)
        ani.save(filename, writer="pillow")
        plt.close(fig)
