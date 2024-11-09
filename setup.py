from setuptools import setup, find_packages

setup(
    name="hamiltonian-neural-ode",
    version="0.1",
    packages=find_packages("src"),
    install_requires=[
        "torch==2.6.0",
        "numpy==1.26.0",
        "matplotlib==3.8.0",
        "scipy==1.11.3",
        "scikit-learn==1.5.1",
        "jupyterlab==4.2.5",
        "torchdiffeq==0.2.4",
    ],
)
