import numpy as np
import torch
from scipy.integrate import solve_ivp


class MechanicalSystem:
    """
    A class to represent a mechanical system.


    """

    CONSTANT1 = 1.0
    CONSTANT2 = 1.0

    def __init__(
        self,
        constant1: float = None,
        constant2: float = None,
    ):
        """
        Initializes the double mass-spring system.

        Parameters:
        - constant1 (float): Meaning of the constant1 parameter.
        - constant2 (float): Meaning of the constant2 parameter.
        """

        self.constant1 = constant1 or self.CONSTANT1
        self.constant2 = constant2 or self.CONSTANT2

    def generalized_to_canonical(self, generalized_coords):
        """
        Converts generalized coordinates and valocities to canonical coordinates.

        Args:
            generalized_coords (tuple): Generalized coordinates (q, q_dot).

        Returns:
            tuple: Canonical coordinates (q, p).
        """

        return

    def canonical_to_generalized(self, canonical_coords):
        """
        Converts canonical coordinates to generalized coordinates and velocities.

        Args:
            canonical_coords (tuple): Canonical coordinates (q, p).

        Returns:
            tuple: Generalized coordinates  (q, q_dot).
        """

        return

    def generalized_to_cartesian(self, generalized_coords):
        """
        Converts generalized coordinates and velocities to Cartesian coordinates and velocities.

        Args:
            generalized_coords (tuple): Generalized coordinates (q, q_dot).

        Returns:
            tuple: Cartesian coordinates (x, x_dot).
        """

        return

    def cartesian_to_generalized(self, cartesian_coords):
        """
        Converts Cartesian coordinates and velocities to generalized coordinates and velocities.

        Args:
            cartesian_coords (tuple): Cartesian coordinates (x, x_dot).

        Returns:
            tuple: Generalized coordinates (q, q_dot).
        """

        return

    def hamiltonian(self, canonical_coords: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hamiltonian of the system as a function of the canonical coordinates.

        Args:
            canonical_coords (torch.Tensor): Canonical coordinates (q, p) as torch tensors.

        Returns:
            torch.Tensor: The Hamiltonian of the system.
        """

        return

    def canonical_coords_dot(self, canonical_coords):
        """
        Computes the derivative of the Hamiltonian with respect to the canonical coordinates.

        Args:
            canonical_coords (tuple): Canonical coordinates (q, p) as torch tensors.

        Returns:
            list: List of derivatives [dq/dt, dp/dt].
        """

        # hamiltonian = self.hamiltonian(canonical_coords)

        # hamiltonian_grad = torch.autograd.grad(
        #     hamiltonian, canonical_coords, create_graph=True
        # )[0]

        # ...

        return

    def generate_data(
        self,
        q: float,
        q_dot: float,
        t_span: tuple,
        dt: float,
        method: str = "DOP853",
    ) -> np.ndarray:
        """
        Simulates the double mass-spring system and returns the simulation history.

        Parameters:
            - q (float): Generalized coordinates.
            - q_dot (float): Generalized velocities.
            - t_span (tuple): Time span (t0, tf).
            - dt (float): Time step for simulation history. The actual time step used for simulation is adaptive.
            - method (str): Integration method to use. Default is "DOP853".

        Returns:
            dict: Dictionary containing the simulation history.
        """

        t_eval = np.arange(t_span[0], t_span[1], dt)

        initial_generalized_coords = (q, q_dot)
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

        data = {}

        data["t"] = solution.t
        data["canonical_coords"] = solution.y.T
        data["canonical_coords_dot"] = np.array(
            [self.canonical_coords_dot(state) for state in solution.y.T]
        )
        data["generalized_coords"] = np.array(
            [self.canonical_to_generalized(state) for state in solution.y.T]
        )
        data["cartesian_coords"] = np.array(
            [
                self.generalized_to_cartesian(state)
                for state in data["generalized_coords"]
            ]
        )
        data["energy"] = self.compute_energy_history(data["canonical_coords"])
        data["initial_energy"] = self.energy_from_generalized(
            initial_generalized_coords
        )

        return data

    def compute_energy_history(self, coords_history):
        """
        Computes the energy of the system from the history of canonical coordinates.

        Args:
            history (np.ndarray): History of canonical coordinates.

        Returns:
            np.ndarray: Energy history of the system.
        """

        with torch.no_grad():
            energy_history = [
                self.hamiltonian([torch.tensor(coord) for coord in coords]).item()
                for coords in coords_history
            ]
        return np.array(energy_history)

    def energy_from_generalized(self, generelaized_coords) -> float:
        """
        Computes the initial mechanical energy from the generalized coordinates.

        Parameters:
            generelaized_coords (tuple): Initial generalized coordinates, (q, q_dot).

        Returns:
            float: The initial mechanical energy.
        """

        # kinetic_energy = ...
        # potential_energy = ...

        return
