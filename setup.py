from setuptools import find_packages, setup
from setup_configuration import setup_configuration

setup(packages=find_packages(exclude=["tests"]), **setup_configuration)
