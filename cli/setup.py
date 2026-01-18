"""
Setup script for Mora CLI.

Install with:
    pip install -e ./cli

Then use:
    mora score recording.wav
"""

from setuptools import setup, find_packages

setup(
    name="mora-cli",
    version="1.0.0",
    description="Mora Sleep Scoring CLI",
    author="Phil Clark",
    packages=find_packages(),
    py_modules=["mora_cli"],
    install_requires=[
        "click>=8.0",
        "requests>=2.28",
        "pandas>=2.0",
        "scipy>=1.11",
        "joblib>=1.3",
    ],
    entry_points={
        "console_scripts": [
            "mora=mora_cli:main",
        ],
    },
    python_requires=">=3.9",
)
