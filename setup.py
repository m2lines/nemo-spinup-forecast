import os

from setuptools import find_packages, setup


# Function to parse requirements.txt
def parse_requirements(filename):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as file:
        return file.read().splitlines()


# Get the list of requirements from requirements.txt
requirements_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "requirements.txt")
)
install_requires = parse_requirements(requirements_file)

setup(
    name="spinup_nemo",
    version="0.1",
    packages=find_packages(include=["lib"]),
    package_dir={"": "."},  # The root directory is the base
    install_requires=install_requires,
)
