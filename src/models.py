import torch
import torch.nn as nn
import torch.nn.functional as F


class HamiltonianNeuralNetwork(nn.Module):
    """
    Wrapper class for a neural network that computes the Hamiltonian of a system.

    Args:
        input_dim (int): Dimension of the canonical coordinates (2 * n_dim).
        core_model (torch.nn.Module): Core model used to compute the Hamiltonian.

    ..note::
        The 'core_model' takes the canonical coordinates [q, p] as input, of shape (2 * n_dim,),
        and returns a scalar output, of shape (1,), representing the Hamiltonian of the system.
        Here, n_dim is the number of generalized coordinates of the system.

    ..note::
       This model computes the time derivatives of the canonical coordinates by taking the gradient
         of the Hamiltonian.

    """

    def __init__(self, input_dim: int, core_model: nn.Module) -> None:
        super(HamiltonianNeuralNetwork, self).__init__()
        assert isinstance(core_model, nn.Module), "Model must be a torch.nn.Module"

        self.core_model = core_model
        self.transform_mat = self._hamiltonian_grad_to_coords_dot(
            input_dim
        ).requires_grad_(True)

    def forward(self, canonical_coords: torch.Tensor) -> torch.Tensor:
        """
        Use the model to compute the Hamiltonian and return the time derivatives
        of the canonical coordinates.

        Args:
            canonical_coords (torch.Tensor): Canonical coordinates, i.e., concatenated generalized
                coordinates and momenta [q, p]. Shape (batch_size, 2 * n_dim).

        Returns:
            torch.Tensor: Time derivatives of the canonical coordinates [dq_dt, dp_dt].
        """
        hamiltonian = self.compute_hamiltonian(canonical_coords)
        if len(hamiltonian.shape) == 1:
            assert hamiltonian.shape[0] == 1, "Hamiltonian must be a scalar"
        else:  # batched
            assert hamiltonian.shape[1] == 1, "Hamiltonian must be a scalar"

        hamiltonian_grad = torch.autograd.grad(
            hamiltonian,
            canonical_coords,
            grad_outputs=torch.ones_like(hamiltonian),
            create_graph=True,
        )[0]

        return torch.matmul(hamiltonian_grad, self.transform_mat)

    def compute_hamiltonian(self, canonical_coords: torch.Tensor) -> torch.Tensor:
        """
        Returns the Hamiltonian of the system.

        Args:
            canonical_coords (torch.Tensor): Canonical coordinates, i.e., concatenated generalized
                coordinates and momenta [q, p]. Shape (batch_size, 2 * n_dim).

        Returns:
            torch.Tensor: Hamiltonian of the system.
        """
        return self.core_model(canonical_coords)

    def _hamiltonian_grad_to_coords_dot(self, input_dim: int) -> torch.Tensor:
        """
        Returns a transformation matrix to obtain time derivatives of the canonical coordinates
        when applied to the gradient of the Hamiltonian.

        Args:
            input_dim (int): Dimension of the canonical coordinates (must be even).

        Returns:
            torch.Tensor: Transformation matrix.

        ..note::
            Hamilton's equations are:

                dq_dt = dH/dp
                dp_dt = -dH/dq

            We want a matrix such that:

                [dq_dt, dp_dt] = [dH/dq, dH/dp] @ transform_mat

            For a 2D system, q and p are each 2D vectors, and the transformation matrix is:

                transform_mat = [
                    [0, 0, -1, 0],
                    [0, 0, 0, -1],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                ]
        """
        assert input_dim % 2 == 0, "Input dimension must be even"

        n_dim = input_dim // 2
        transform_mat = torch.zeros(input_dim, input_dim)
        transform_mat[:n_dim, n_dim:] = -torch.eye(n_dim)
        transform_mat[n_dim:, :n_dim] = +torch.eye(n_dim)

        return transform_mat


class HamiltonianDynamics(nn.Module):
    """
    Wrapper class for a HaHamiltonianNeuralNetwork to include the (unused) time variable in the forward pass.
    """

    def __init__(self, hnn_model):
        super(HamiltonianDynamics, self).__init__()
        self.hnn = hnn_model

    def forward(self, t, canonical_coords: torch.Tensor) -> torch.Tensor:
        return self.hnn(canonical_coords)


class SimpleMLP(nn.Module):
    """
    Simple feedforward neural network.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output data.
    """

    def __init__(self, input_dim: int = 1, output_dim: int = 1) -> None:
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP(nn.Module):
    """
    Feedforward neural network.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output data.
        hidden_dim (int): Dimension of hidden layers.
        n_hidden_layers (int): Number of hidden layers.
        activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super(MLP, self).__init__()
        self.activation = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        return self.output_layer(x)


class ResMLP(nn.Module):
    """
    Residual feedforward neural network.

    Args:
        input_dim (int): Dimension of input data.
        output_dim (int): Dimension of output data.
        hidden_dim (int): Dimension of hidden layers.
        n_hidden_layers (int): Number of hidden layers.
        activation (torch.nn.Module): Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        n_hidden_layers: int = 4,
        activation: nn.Module = nn.ReLU,
    ) -> None:
        super(ResMLP, self).__init__()
        self.activation = activation()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x)) + x
        return self.output_layer(x)
