import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple


class DoublePendulum:
    """
    A class to represent a double pendulum system.

    The generalized coordinates are defined as:
    - q1: Angle of the first pendulum.
    - q2: Angle of the second pendulum.
    - q1_dot: Angular velocity of the first pendulum.
    - q2_dot: Angular velocity of the second pendulum.

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
        Initializes the double pendulum with the specified parameters.

        Args:
            gravity (float, optional): Gravity acceleration.
            mass (float, optional): Mass of both the pendulum bobs, assumed to be equal.
            length (float, optional): Lengths of the pendulum rods, assumed to be equal.
        """
        self.gravity = gravity if gravity else self.GRAVITY
        self.mass = mass if mass else self.MASS
        self.length = length if length else self.LENGTH

    def generalized_to_canonical(
        self, generalized_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts generalized coordinates and velocities to canonical coordinates.

        Args:
            generalized_coords (Tuple[float, float, float, float]): Generalized coordinates (q1, q2, q1_dot, q2_dot).

        Returns:
            Tuple[float, float, float, float]: Canonical coordinates (q1, q2, p1, p2).
        """
        q1, q2, q1_dot, q2_dot = generalized_coords

        p1 = self.mass * self.length**2 * (2 * q1_dot + q2_dot * np.cos(q1 - q2))
        p2 = self.mass * self.length**2 * (q2_dot + q1_dot * np.cos(q1 - q2))
        return q1, q2, p1, p2

    def canonical_to_generalized(
        self, canonical_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts canonical coordinates to generalized coordinates and velocities.

        Args:
            canonical_coords (Tuple[float, float, float, float]): Canonical coordinates (q1, q2, p1, p2).

        Returns:
            Tuple[float, float, float, float]: Generalized coordinates (q1, q2, q1_dot, q2_dot).
        """
        q1, q2, p1, p2 = canonical_coords

        q1_dot = (
            (p1 - p2 * np.cos(q1 - q2))
            / (self.mass * self.length**2)
            / (1 + np.sin(q1 - q2) ** 2)
        )

        q2_dot = (
            (-p1 * np.cos(q1 - q2) + 2 * p2)
            / (self.mass * self.length**2)
            / (1 + np.sin(q1 - q2) ** 2)
        )
        return q1, q2, q1_dot, q2_dot

    def generalized_to_cartesian(
        self, generalized_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Converts generalized coordinates and velocities to Cartesian coordinates and velocities.

        Args:
            generalized_coords (Tuple[float, float, float, float]): Generalized coordinates (q1, q2, q1_dot, q2_dot).

        Returns:
            Tuple[float, float, float, float, float, float, float, float]:
            Cartesian coordinates (x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot).
        """
        q1, q2, q1_dot, q2_dot = generalized_coords

        x1 = self.length * np.sin(q1)
        y1 = -self.length * np.cos(q1)
        x2 = x1 + self.length * np.sin(q2)
        y2 = y1 - self.length * np.cos(q2)

        x1_dot = self.length * q1_dot * np.cos(q1)
        y1_dot = self.length * q1_dot * np.sin(q1)
        x2_dot = x1_dot + self.length * q2_dot * np.cos(q2)
        y2_dot = y1_dot + self.length * q2_dot * np.sin(q2)

        return x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot

    def cartesian_to_generalized(
        self,
        cartesian_coords: Tuple[float, float, float, float, float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Converts Cartesian coordinates and velocities to generalized coordinates and velocities.

        Args:
            cartesian_coords (Tuple[float, float, float, float, float, float, float, float]):
            Cartesian coordinates (x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot).

        Returns:
            Tuple[float, float, float, float]: Generalized coordinates (q1, q2, q1_dot, q2_dot).
        """
        x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot = cartesian_coords

        q1 = np.arctan2(x1, -y1)
        q2 = np.arctan2(x2 - x1, y1 - y2)

        q1_dot = (x1 * x1_dot + y1 * y1_dot) / (self.length**2)
        q2_dot = ((x2 - x1) * x2_dot + (y1 - y2) * y2_dot) / (self.length**2)

        return q1, q2, q1_dot, q2_dot

    def canonical_to_cartesian(
        self, canonical_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Converts canonical coordinates to Cartesian coordinates.

        Args:
            canonical_coords (Tuple[float, float, float, float]): Canonical coordinates (q1, q2, p1, p2).

        Returns:
            Tuple[float, float, float, float, float, float, float, float]:
            Cartesian coordinates (x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot).
        """
        generalized_coords = self.canonical_to_generalized(canonical_coords)
        return self.generalized_to_cartesian(generalized_coords)

    def cartesian_to_canonical(
        self,
        cartesian_coords: Tuple[float, float, float, float, float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Converts Cartesian coordinates to canonical coordinates.

        Args:
            cartesian_coords (Tuple[float, float, float, float, float, float, float, float]):
            Cartesian coordinates (x1, y1, x2, y2, x1_dot, y1_dot, x2_dot, y2_dot).

        Returns:
            Tuple[float, float, float, float]: Canonical coordinates (q1, q2, p1, p2).
        """
        generalized_coords = self.cartesian_to_generalized(cartesian_coords)
        return self.generalized_to_canonical(generalized_coords)

    def hamiltonian(self, canonical_coords: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian of the system as a function of the canonical coordinates.

        Args:
            canonical_coords (torch.Tensor): Canonical coordinates (q1, q2, p1, p2) as torch tensors.

        Returns:
            torch.Tensor: The Hamiltonian of the system.
        """
        q1, q2, p1, p2 = canonical_coords

        hamiltonian = (p1**2 + 2 * p2**2 - 2 * p1 * p2 * torch.cos(q1 - q2)) / (
            2 * self.mass * self.length**2 * (1 + torch.sin(q1 - q2) ** 2)
        ) + (
            -self.mass
            * self.gravity
            * self.length
            * (2 * torch.cos(q1) + torch.cos(q2))
        )

        return hamiltonian

    def canonical_coords_dot(
        self, canonical_coords: Tuple[float, float, float, float]
    ) -> List[float]:
        """
        Computes the derivative of the Hamiltonian with respect to the canonical coordinates.

        Args:
            canonical_coords (Tuple[float, float, float, float]): Canonical coordinates (q1, q2, p1, p2)
            as torch tensors.

        Returns:
            List[float]: List of derivatives [dq1/dt, dq2/dt, dp1/dt, dp2/dt].
        """
        q1, q2, p1, p2 = [
            torch.tensor(coord, requires_grad=True) for coord in canonical_coords
        ]
        coords = [q1, q2, p1, p2]

        hamiltonian = self.hamiltonian(coords)

        dH_dq1, dH_dq2, dH_dp1, dH_dp2 = torch.autograd.grad(
            hamiltonian, coords, create_graph=True
        )

        dq1_dt = dH_dp1.item()
        dq2_dt = dH_dp2.item()
        dp1_dt = -dH_dq1.item()
        dp2_dt = -dH_dq2.item()

        return dq1_dt, dq2_dt, dp1_dt, dp2_dt

    def generate_dataset(
        self,
        num_simulations: int,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        q_lim: Tuple[float, float] = (-3, 3),
        q_dot_lim: Tuple[float, float] = (-0.5, 0.5),
        return_trajectories: bool = False,
        noise_level: float = 0.0,
    ) -> Tuple[TensorDataset, torch.Tensor, torch.Tensor]:
        """
        Initializes the dataset by generating simulation data.

        Args:
            num_simulations (int): Number of different simulations.
            t_span (Tuple[float, float]): Time span for the simulations.
            t_eval (np.ndarray): Time points to evaluate the solution.
            q_lim (Tuple[float, float]): Range of initial angles for the pendulums.
            q_dot_lim (Tuple[float, float]): Range of initial angular velocities for the pendulums.
            return_trajectories (bool): If True, returns the canonical coords history
                of all runs as a dictionary, where the key is the trajectory number and the value
                is the simulation data. Default is False.
            noise_level (float, optional): Control intensity of noise applied to the inputs and targets.
                A zero-mean Gaussian noise with standard deviation equal to noise_level is added to the
                canonical coordinates and their derivatives. Default is 0.0.

        Returns:
            Tuple[TensorDataset, torch.Tensor, torch.Tensor]: Dataset containing the simulation data, inputs,
            and targets.
        """
        inputs = []
        targets = []
        for i in range(num_simulations):

            q1, q2 = np.random.uniform(*q_lim, size=2)
            q1_dot, q2_dot = np.random.uniform(*q_dot_lim, size=2)
            simulation_data = self.simulate(q1, q2, q1_dot, q2_dot, t_span, t_eval)

            canonical_coords = torch.tensor(
                simulation_data["canonical_coords"], dtype=torch.float32
            ).reshape(-1, 4)
            canonical_coords_dot = torch.tensor(
                simulation_data["canonical_coords_dot"], dtype=torch.float32
            ).reshape(-1, 4)

            noise_coords = noise_level * torch.randn_like(canonical_coords)
            canonical_coords = canonical_coords + noise_coords

            noise_coords_dot = noise_level * torch.randn_like(canonical_coords_dot)
            canonical_coords_dot = canonical_coords_dot + noise_coords_dot

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
        q1: float,
        q2: float,
        q1_dot: float,
        q2_dot: float,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        method: str = "DOP853",
    ) -> dict:
        """
        Simulates the double pendulum using solve_ivp function from scipy.integrate.

        Args:
            q1 (float): Initial angle of the first pendulum.
            q2 (float): Initial angle of the second pendulum.
            q1_dot (float): Initial angular velocity of the first bob.
            q2_dot (float): Initial angular velocity of the second bob.
            t_span (Tuple[float, float]): Time span (t0, tf).
            t_eval (np.ndarray): Time points to evaluate the solution.
            method (str): Integration method to use. Default is "DOP853".

        Returns:
            dict: Dictionary containing the simulation history.
        """

        initial_generalized_coords = [q1, q2, q1_dot, q2_dot]
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

    def energy_from_generalized(
        self, generalized_coords: Tuple[float, float, float, float]
    ) -> float:
        """
        Computes the initial mechanical energy from the generalized coordinates.

        Args:
            generalized_coords (Tuple[float, float, float, float]): Generalized coordinates (q1, q2, q1_dot, q2_dot).

        Returns:
            float: The initial mechanical energy.
        """

        q1, q2, q1_dot, q2_dot = generalized_coords
        kinetic_energy = (
            self.mass
            * self.length**2
            * (q1_dot**2 + 0.5 * q2_dot**2 + q1_dot * q2_dot * np.cos(q1 - q2))
        )

        potential_energy = (
            -self.mass * self.gravity * self.length * (2 * np.cos(q1) + np.cos(q2))
        )

        return kinetic_energy + potential_energy

    def create_gif(
        self,
        t: np.ndarray,
        cart_coords: np.ndarray,
        filename: str = "double_pendulum.gif",
        interval: int = 50,
        trail_length: int = 10,
    ) -> None:
        """
        Creates a GIF from the simulation history.

        Args:
            t (np.ndarray): Time history.
            cart_coords (np.ndarray): History of cartesian coordinates, shape (n_steps, 4).
            filename (str): Name of the GIF file.
            interval (int): Time interval between frames in milliseconds.
            trail_length (int): Number of previous points to show as a trail.
        """

        x1, y1, x2, y2 = (
            cart_coords[:, 0],
            cart_coords[:, 1],
            cart_coords[:, 2],
            cart_coords[:, 3],
        )

        fig, ax = plt.subplots()
        ax.set_title("t = 0 s", fontsize=14)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-3 * self.length, 3 * self.length)
        ax.set_ylim(-3 * self.length, 3 * self.length)

        (line,) = ax.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], "o-", c="k", lw=1.2)
        (point1,) = ax.plot(x1[0], y1[0], "ko", markersize=4)
        (point2,) = ax.plot(x2[0], y2[0], "ko", markersize=4)
        (trail1,) = ax.plot([], [], "k-", alpha=0.5, lw=1)
        (trail2,) = ax.plot([], [], "k-", alpha=0.5, lw=1)

        def update(frame):
            line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
            point1.set_data([x1[frame]], [y1[frame]])
            point2.set_data([x2[frame]], [y2[frame]])
            start = max(0, frame - trail_length)
            trail1.set_data(x1[start : frame + 1], y1[start : frame + 1])  # noqa
            trail2.set_data(x2[start : frame + 1], y2[start : frame + 1])  # noqa
            ax.set_title(f"t = {t[frame]:.1f} s", fontsize=14)
            return line, point1, point2, trail1, trail2

        ani = FuncAnimation(fig, update, frames=len(x1), blit=True, interval=interval)
        ani.save(filename, writer="pillow")
        plt.close(fig)
