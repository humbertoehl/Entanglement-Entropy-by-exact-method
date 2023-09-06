# Entanglement Entropy Calculator for Extended Bose-Hubbard Model

This repository contains a Python program for calculating entanglement entropy ($S_{ent}$) for an extended Bose-Hubbard model using exact diagonalization. The program performs a mapping on a one-body density matrix to obtain the reduced density matrix, which is then used to compute the entanglement entropy.

## Table of Contents
- [Explanation](#explanation)
- [Instructions](#instructions)
- [Optional](#optional)

## Explanation

The extended Bose-Hubbard model is a theoretical model used in condensed matter physics to describe the behavior of interacting bosons in a lattice. Entanglement entropy is a measure of the quantum entanglement between two subsystems of a larger quantum system. In this project, we calculate the entanglement entropy for a specific extended Bose-Hubbard model using the following formula:

$$S_{ent} = - \text{Tr} [\rho_A \log \rho_A]$$

Where:
- $S_{ent}$ is the entanglement entropy.
- $\rho_A$ is the reduced density matrix of subsystem A.

## Instructions

To use this program, follow these instructions:

1. Clone this repository to your local machine using the following command:
git clone https://github.com/humbertoehl/Exact_Diagonalization_BH.git

2. Navigate to the repository directory:
cd Exact_Diagonalization_BH

3. Run the main calculation script:
python3 Perform_calculation.py

4. The program will prompt you for the following inputs:
- `M`: The total number of lattice sites.
- `N`: The number of particles.
- `CAVITY`: Enter `True` or `False` to specify whether the system has a cavity (Boolean input).

5. The program will perform the calculation and generate the following output files:
- `result.txt`: Contains the calculated entanglement entropy.
- `entanglement_plot.png`: A plot visualizing the results.

6. The output files will be stored in the corresponding directories.

## Optional

You can also use or edit the `txt_reader.py` script provided in this repository to:
- Replot the results if needed.
- Customize the appearance of the plots according to your preferences.

