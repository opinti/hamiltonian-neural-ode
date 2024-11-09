import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import Tuple, Union, Dict


class DoubleMassSpring:
    """
    A class to represent a double mass-spring system.

    The generalized coordinates are defined as:
    - q1: Position of the first mass with respect to the equilibrium position of the first spring.
    - q2: Position of the second mass with respect to the equilibrium position of both springs.
    - q1_dot: Velocity of the first mass.
    - q2_dot: Velocity of the second mass.
    """

    M1 = 1
    K1 = 10
    M2 = 1
    K2 = 10
    L1 = 2
    L2 = 2

    def __init__(
        self,
        mass1: float = None,
        k1: float = None,
        l1: float = None,
        mass2: float = None,
        k2: float = None,
        l2: float = None,
    ) -> None:
        """
        Initializes the double mass-spring system.

        Args:
            mass1 (float, optional): Mass of the first mass.
            k1 (float, optional): Spring constant of the first spring.
            l1 (float, optional): Length of the first spring.
            mass2 (float, optional): Mass of the second mass.
            k2 (float, optional): Spring constant of the second spring.
            l2 (float, optional): Length of the second spring.
        """
        self.m1 = mass1 or self.M1
        self.k1 = k1 or self.K1
        self.l1 = l1 or self.L1
        self.m2 = mass2 or self.M2
        self.k2 = k2 or self.K2
        self.l2 = l2 or self.L2

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
        p1 = self.m1 * q1_dot
        p2 = self.m2 * q2_dot
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
        q1_dot = p1 / self.m1
        q2_dot = p2 / self.m2
        return q1, q2, q1_dot, q2_dot

    def generalized_to_cartesian(
        self, generalized_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts generalized coordinates and velocities to Cartesian coordinates and velocities.

        Args:
            generalized_coords (Tuple[float, float, float, float]): Generalized coordinates (q1, q2, q1_dot, q2_dot).

        Returns:
            Tuple[float, float, float, float]: Cartesian coordinates (x1, x2, x1_dot, x2_dot).
        """
        q1, q2, q1_dot, q2_dot = generalized_coords
        x1 = q1 + self.l1
        x2 = q2 + self.l1 + self.l2
        x1_dot = q1_dot
        x2_dot = q2_dot
        return x1, x2, x1_dot, x2_dot

    def cartesian_to_generalized(
        self, cartesian_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts Cartesian coordinates and velocities to generalized coordinates and velocities.

        Args:
            cartesian_coords (Tuple[float, float, float, float]): Cartesian coordinates (x1, x2, x1_dot, x2_dot).

        Returns:
            Tuple[float, float, float, float]: Generalized coordinates (q1, q2, q1_dot, q2_dot).
        """
        x1, x2, x1_dot, x2_dot = cartesian_coords
        q1 = x1 - self.l1
        q2 = x2 - self.l1 - self.l2
        q1_dot = x1_dot
        q2_dot = x2_dot
        return q1, q2, q1_dot, q2_dot

    def canonical_to_cartesian(
        self, canonical_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts canonical coordinates to Cartesian coordinates.

        Args:
            canonical_coords (Tuple[float, float, float, float]): Canonical coordinates (q1, q2, p1, p2).

        Returns:
            Tuple[float, float, float, float]: Cartesian coordinates (x1, x2, x1_dot, x2_dot).
        """
        generalized_coords = self.canonical_to_generalized(canonical_coords)
        return self.generalized_to_cartesian(generalized_coords)

    def cartesian_to_canonical(
        self, cartesian_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Converts Cartesian coordinates to canonical coordinates.

        Args:
            cartesian_coords (Tuple[float, float, float, float]): Cartesian coordinates (x1, x2, x1_dot, x2_dot).

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
        hamiltonian = (
            0.5 * p1**2 / self.m1
            + 0.5 * p2**2 / self.m2
            + 0.5 * self.k1 * q1**2
            + 0.5 * self.k2 * (q2 - q1) ** 2
        )
        return hamiltonian

    def canonical_coords_dot(
        self, canonical_coords: Tuple[float, float, float, float]
    ) -> Tuple[float, float, float, float]:
        """
        Computes the derivative of the Hamiltonian with respect to the canonical coordinates.

        Args:
            canonical_coords (Tuple[float, float, float, float]): Canonical coordinates (q1, q2, p1, p2).

        Returns:
            Tuple[float, float, float, float]: Derivatives [dq1/dt, dq2/dt, dp1/dt, dp2/dt].
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
        q_lim: Tuple[float, float] = (-1, 1),
        q_dot_lim: Tuple[float, float] = (-0.5, 0.5),
        initial_conditions: Tuple[float, float, float, float] = None,
        return_trajectories: bool = False,
        noise_level: float = 0.0,
    ) -> Union[TensorDataset, Tuple[TensorDataset, torch.Tensor, torch.Tensor]]:
        """
        Initializes the dataset by generating simulation data.

        Args:
            num_simulations (int): Number of simulations.
            t_span (Tuple[float, float]): Time span for the simulations.
            t_eval (np.ndarray): Time points to evaluate the solution.
            q_lim (Tuple[float, float], optional): Limits for the initial positions.
            q_dot_lim (Tuple[float, float], optional): Limits for the initial velocities.
            return_trajectories (bool, optional): If True, returns canonical coords history of all runs.
            noise_level (float, optional): Intensity of noise applied to inputs and targets.

        Returns:
            Union[TensorDataset, Tuple[TensorDataset, torch.Tensor, torch.Tensor]]: Dataset containing simulation data.
        """
        if initial_conditions is not None and num_simulations > 1:
            raise ValueError(
                "If initial_conditions is provided, num_simulations must be 1."
            )

        inputs, targets = [], []
        for _ in range(num_simulations):

            if initial_conditions is not None:
                q1, q2, q1_dot, q2_dot = initial_conditions
            else:
                q1, q2 = np.random.uniform(*q_lim, size=2)
                q1_dot, q2_dot = np.random.uniform(*q_dot_lim, size=2)

            simulation_data = self.simulate(q1, q2, q1_dot, q2_dot, t_span, t_eval)
            canonical_coords = torch.tensor(
                simulation_data["canonical_coords"], dtype=torch.float32
            ).reshape(-1, 4)
            canonical_coords_dot = torch.tensor(
                simulation_data["canonical_coords_dot"], dtype=torch.float32
            ).reshape(-1, 4)

            canonical_coords += noise_level * torch.randn_like(canonical_coords)
            canonical_coords_dot += noise_level * torch.randn_like(canonical_coords_dot)

            inputs.append(canonical_coords)
            targets.append(canonical_coords_dot)
        inputs_tensor, targets_tensor = torch.cat(inputs), torch.cat(targets)
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
    ) -> Dict[str, np.ndarray]:
        """
        Simulates the double mass-spring system and returns the simulation history.

        Args:
            q1 (float): Initial position of the first mass.
            q2 (float): Initial position of the second mass.
            q1_dot (float): Initial velocity of the first mass.
            q2_dot (float): Initial velocity of the second mass.
            t_span (Tuple[float, float]): Time span (t0, tf).
            t_eval (np.ndarray): Time points to evaluate the solution.
            method (str): Integration method to use. Default is "DOP853".

        Returns:
            Dict[str, np.ndarray]: Dictionary containing the simulation history.
        """
        initial_generalized_coords = (q1, q2, q1_dot, q2_dot)
        initial_state = self.generalized_to_canonical(initial_generalized_coords)

        def state_dot(t, state):
            return self.canonical_coords_dot(state)

        solution = solve_ivp(
            fun=state_dot, t_span=t_span, y0=initial_state, t_eval=t_eval, method=method
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
            generalized_coords (Tuple[float, float, float, float]): Initial generalized coordinates.

        Returns:
            float: The initial mechanical energy.
        """
        q1, q2, q1_dot, q2_dot = generalized_coords
        kinetic_energy = 0.5 * self.m1 * q1_dot**2 + 0.5 * self.m2 * q2_dot**2
        potential_energy = 0.5 * self.k1 * q1**2 + 0.5 * self.k2 * (q2 - q1) ** 2
        return kinetic_energy + potential_energy

    def create_gif(
        self,
        t: np.ndarray,
        cart_coords: np.ndarray,
        filename: str = "double_mass_spring.gif",
        interval: int = 50,
    ) -> None:
        """
        Creates a GIF from the simulation history of the double mass-spring system.

        Args:
            t (np.ndarray): Time history.
            cart_coords (np.ndarray): History of cartesian coordinates, shape (n_steps, 2).
            filename (str): Name of the GIF file.
            interval (int): Time interval between frames in milliseconds.

        Returns:
            None
        """

        def draw_spring(x_start, x_end, amplitude=0.05, coils=10):
            x = np.linspace(x_start, x_end, 100)
            y = amplitude * np.sin(
                2 * np.pi * coils * (x - x_start) / (x_end - x_start)
            )
            return x, y

        x1, x2 = cart_coords[:, 0], cart_coords[:, 1]
        xlims, ylims = (-0.5 * self.l1, 2 * (self.l1 + self.l2)), (-0.5, 1.5)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title("t = 0 s", fontsize=16)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.fill_between(xlims, ylims[0], 0, color="lightgray", alpha=0.5)

        (mass1_marker,) = ax.plot(
            [], [], "r", marker="s", markersize=20, label="Mass 1"
        )
        (mass2_marker,) = ax.plot(
            [], [], "b", marker="s", markersize=20, label="Mass 2"
        )
        spring1_x, spring1_y = draw_spring(0, x1[0])
        spring2_x, spring2_y = draw_spring(x1[0], x2[0])
        (spring1_line,) = ax.plot(spring1_x, spring1_y, "r-", lw=1.5)
        (spring2_line,) = ax.plot(spring2_x, spring2_y, "b-", lw=1.5)

        def update(frame):
            mass1_marker.set_data([x1[frame]], [0])
            mass2_marker.set_data([x2[frame]], [0])
            spring1_x, spring1_y = draw_spring(0, x1[frame])
            spring2_x, spring2_y = draw_spring(x1[frame], x2[frame])
            spring1_line.set_data(spring1_x, spring1_y)
            spring2_line.set_data(spring2_x, spring2_y)
            ax.set_title(f"t = {t[frame]:.1f} s", fontsize=16)
            return spring1_line, spring2_line, mass1_marker, mass2_marker

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Mass 1",
                markerfacecolor="b",
                markersize=6,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Mass 2",
                markerfacecolor="r",
                markersize=6,
            ),
        ]
        ax.legend(handles=legend_handles, loc="upper right")
        plt.tight_layout()

        ani = FuncAnimation(fig, update, frames=len(x1), blit=True, interval=interval)
        ani.save(filename, writer="pillow")
        plt.close(fig)
