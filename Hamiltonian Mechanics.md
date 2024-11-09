# Crash Course on Classical Mechanics

## Basics

- **Cartesian coordinates** $\mathbf{x} = [x_1, x_2, \dots]$ are the coordinates of a point in 2D or 3D space.

- **Velocities** $ \dot{\mathbf{x}} = [\dot{x}_1, \dot{x}_2, \dots] $ are the rate of change of coordinates with respect to time.

- **Generalized coordinates** $ \mathbf{q} = [q_1, q_2, \dots] $ are a set of coordinates that completely defines the state of a system. These coordinates are not necessarily Cartesian, e.g. the angle between the rod and the vertical axis for a pendulum.

- **Generalized velocities** $ \dot{\mathbf{q}} = [\dot{q}_1, \dot{q}_2, \dots] $ are the rate of change of generalized coordinates with respect to time.

- The **configuration space** of a system is the space where the generalized coordinates $ \mathbf{q} $ live.

- The **trajectory** of a system $ \mathbf{q}(t) $ is the evolution of the generalized coordinates over time in the configuration space.

## Lagrangian Mechanics

- The **Lagrangian** $ \mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) $ is a function of the generalized coordinates and generalized velocities. It fully describes the dynamics of a system. The Lagrangian is defined as the difference between the kinetic and potential energy of the system.

- The **action** $ \mathcal{S} $ is a functional of the Lagrangian, defined as the integral of the Lagrangian over time:

    $$
    \mathcal{S}\{\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}})\} = \int_{t_0}^{t_1} \mathcal{L}(\mathbf{q}(\tau), \dot{\mathbf{q}}(\tau)) \mathrm{d}\tau
    $$

- One of the cornerstones of Lagrangian mechanics is the **principle of least action**, which states that among all possible trajectories in the configuration space, the realized trajectory of the system is the one that minimizes the action, i.e. the trajectory that ensures $\delta \mathcal{S}=0$.

- Setting $\delta \mathcal{S}=0$ yields a set of differential equations that the Lagrangian must satisfiy. These equations, which effectively govern the evolution of the system, are called **Euler-Lagrange equations**, and are:

    $$
    \frac{\mathrm{d}}{\mathrm{d}t}\left(\frac{\partial \mathcal{L}}{\partial \dot{\mathbf{q}}}\right) - \frac{\partial \mathcal{L}}{\partial \mathbf{q}} = 0
    $$

- **Generalized momenta** $ \mathbf{p} = [p_1, p_2, \dots] $ are defined as the partial derivatives of the Lagrangian with respect to the generalized velocities: 
    $$
    \mathbf{p} = \frac{\partial \mathcal{L}}{\partial \dot{\mathbf{q}}}
    $$

- Generalized coordinates and momenta $(\mathbf{q}, \mathbf{p})$ are also called **canonical coordinates**. 

- The change of coordintes from generalized to canonical  $(\mathbf{q}, \dot{\mathbf{q}}) \rightarrow (\mathbf{q}, \mathbf{p})$ is perfectly legitimate, as canonical coordinates also fully describe the state of a system. Interestingly, using canonical coordintes often proves to be advantegous. In that regards, we can write the Lagrangian in terms of canonical coordinates:

    $$ 
    \mathcal{L}(\mathbf{q}, \mathbf{p}) = \mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}(\mathbf{q}, \mathbf{p}))
    $$

- The space where the canonical coordinates $(\mathbf{q}(t), \mathbf{p}(t))$ live is called **phase space**.

## Hamiltonian Mechanics

- The **Hamiltonian** is defined as the _Legendre transform_ of the Lagrangian expressed in canonical coordinates with respect to the generalized momenta:
  
    $$
    \mathcal{H}(\mathbf{q}, \mathbf{p}) = \mathbf{p} \cdot \dot{\mathbf{q}}(\mathbf{q}, \mathbf{p}) - \mathcal{L}(\mathbf{q}, \mathbf{p})
    $$

- The Hamiltonian also fully describes the dynamics of a system.

- **Hamilton's equations** are a set of first-order differential equations that describe the evolution of the generalized coordinates and momenta over time. Hamilton's equations are:

    $$
    \frac{\mathrm{d} \mathbf{q}}{\mathrm{d} t} = \frac{\partial \mathcal{H}}{\partial \mathbf{p}}
    $$
    
    $$
    \frac{\mathrm{d} \mathbf{p}}{\mathrm{d}t} = -\frac{\partial \mathcal{H}}{\partial \mathbf{q}}
    $$

- The vector field defined over the phase space $\bm{g}_{\mathcal{H}}(\mathbf{q}, \, \mathbf{p}) = \left(\frac{\partial \mathcal{H}}{\partial \mathbf{p}}, \, -\frac{\partial \mathcal{H}}{\partial \mathbf{q}} \right)$ is also called a **symplectic gradient**. It represents the direction in  phase space along which a given function, in this case $\mathcal{H}$, remains constant. 

- If the Lagrangian is not an explicit function of time (which we have silently assumed here), then the Hamiltonian is conserved:

    $$
    \frac{\mathrm{d} \mathcal{H}}{\mathrm{d}t} = \frac{\partial \mathcal{H}}{\partial \mathbf{q}} \dot{\mathbf{q}} + \frac{\partial \mathcal{H}}{\partial \mathbf{p}} \dot{\mathbf{p}} \equiv0.
    $$

- Under certain conditions, the Hamiltonian corresponds to the total energy of the system.

- **Liouville's theorem** states that if a system follows Hamilton's equations, then any region in the phase space $ \mathcal{R} =\{(\mathbf{q}, \mathbf{p}) \, | \, \mathbf{q} \in \left[ \mathbf{q}_0, \mathbf{q}_0 +\delta \mathbf{q} \right], \mathbf{p} \in \left[\mathbf{p}_0, \mathbf{p}_0 +\delta \mathbf{p}\right]\}$ preserves its volume throughout its evolution, that is:

    $$
    \frac{\mathrm{d}}{\mathrm{d} t} \int_{\mathcal{R}(t)} \mathrm{d}\mathbf{q} \mathrm{d}\mathbf{p} = 0
    $$