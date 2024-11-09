# Hamiltonian Neural ODE

This project aims to explore Hamiltonian Neural Networks (HNNs) coupled with Neural Ordinary Differential Equations (Neural ODEs) for predicting the evolution of dynamical systems.
The goal is to learn the Hamiltonian of the system and then infer the dynamics from it using Hamilton's equartions, rather than directly learning the map between consequtive states.
The advantage of this approach is being able to ensure energy conservation and promotes stability in autoregressive inference mode.

## Project Structure
```
hamiltonian-neural-ode/
│
├── experiments/                      # Notebooks with examples and experiments
│   └── test_notebook.ipynb              # Notebook showcasing examples and visualizations
│
├── src/
│   ├── mechanical_systems/            # Class modules for various mechanical systems
│   │   ├── double_pendulum.py     
│   │   ├── double_mass_spring.py  
│   │   └── pendulum.py
|   |
│   ├── autoregressive.py                # Utility functions for autoregressive modeling
│   ├── models.py                        # Core models, including Hamiltonian NN
│   └── plot_utils.py                    # Plotting utilities for results visualization
│
├── Hamiltonian Mechanics.md           # Summary of Hamiltonian mechanics concepts
├── LICENSE.txt                        # License information
├── settings.json                      # Python environment and path configuration
└── requirements.txt                   # Required Python packages
```