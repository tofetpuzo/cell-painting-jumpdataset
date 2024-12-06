# setup.py
from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="cell-painting-vae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=required,
    author="The MDCTeam",
    author_email="",
    description="A VAE model for tissue image analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tissue-vae",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="=3.11.10",
)
