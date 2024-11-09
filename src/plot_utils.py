import numpy as np
import matplotlib.pyplot as plt


def plot_canonical_time_series(
    t: np.ndarray,
    canonical_coords: np.ndarray,
) -> None:
    """
    Plots the time series of the canonical coordinates.

    Args:
        t (np.ndarray): Array of time values.
        canonical_coords (np.ndarray): Canonical coordinates history with shape (n_steps, n_coords).

    Returns:
        None
    """
    n_coords = canonical_coords.shape[1]

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle("Canonical coordinates evolution", fontsize=16)

    axs[0].set_title("Generalized coordinates time series", fontsize=12)
    for i in range(n_coords // 2):
        axs[0].plot(t, canonical_coords[:, i], label=f"q{i+1}")

    axs[0].grid(True)

    axs[1].set_title("Generalized momenta time series", fontsize=12)
    for i in range(n_coords // 2, n_coords):
        axs[1].plot(t, canonical_coords[:, i], label=f"p{i+1}")

    axs[1].set_xlabel("Time [s]")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_phase_space_quiver(
    t: np.ndarray,
    generalized_coords: np.ndarray,
    quiver: bool = False,
    trail: bool = True,
) -> None:
    """
    Quiver plot in the phase space showing trajectory and velocity vectors.

    Args:
        t (np.ndarray): Array of time values corresponding to each point.
        generalized_coords (np.ndarray): Generalized coordinates (q1, q2, q1_dot, q2_dot) with shape (n_steps, 4).
        quiver (bool): If True, plot arrows representing the velocity vectors. Default is False.
        trail (bool): If True, plot a trajectory line. Default is True.

    Raises:
        ValueError: If `generalized_coords` does not have shape (n_steps, 4).

    Returns:
        None
    """
    if generalized_coords.shape[1] != 4:
        raise ValueError(
            "Quiver plot only available for 2-dimensional coordinates at this time."
        )

    q1, q2, q1_dot, q2_dot = (
        generalized_coords[:, 0],
        generalized_coords[:, 1],
        generalized_coords[:, 2],
        generalized_coords[:, 3],
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Evolution in phase space")
    scatter = ax.scatter(q1, q2, c=t, cmap="viridis", marker="o", label="Phase Points")
    plt.colorbar(scatter, label="Time")
    if trail:
        ax.plot(q1, q2, c="gray", alpha=0.5, label="Trajectory")

    if quiver:
        magnitudes = np.sqrt(q1_dot**2 + q2_dot**2)
        q1_dot_normalized = q1_dot / magnitudes
        q2_dot_normalized = q2_dot / magnitudes

        ax.quiver(
            q1,
            q2,
            q1_dot_normalized,
            q2_dot_normalized,
            scale=20,
            width=0.0035,
        )

    ax.set_xlabel(r"$q_1$")
    ax.set_ylabel(r"$q_2$")
    ax.grid(True)
    plt.show()
