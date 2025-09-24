from setuptools import setup, find_packages

setup(
    name="ToyMD",
    version="0.1.0",
    description="A simple molecular dynamics project with 1D potentials and trajectory analysis.",
    author="Carlos Castillo",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib"
    ],
    python_requires=">=3.7",
)