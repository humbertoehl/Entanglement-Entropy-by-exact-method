# Entanglement Entropy Calculator for Extended Bose-Hubbard Model

This repository contains a Python program for calculating entanglement entropy ($S_{ent}$) for an extended Bose-Hubbard model using exact diagonalization. The program performs a mapping on a one-body density matrix to obtain the reduced density matrix, which is then used to compute the entanglement entropy.

## Table of Contents
- [Explanation](#explanation)
- [Instructions](#instructions)

## Explanation

The extended Bose-Hubbard model is a theoretical model used in condensed matter physics to describe the behavior of interacting bosons in a lattice. Entanglement entropy is a measure of the quantum entanglement between two subsystems of a larger quantum system. In this project, we calculate the entanglement entropy for a specific extended Bose-Hubbard model using the following formula:

$$S_{ent} = - \text{Tr} [\rho_A \log \rho_A]$$

for two types of partitions
half-half: $A={0,1,2,...,M/2-1}$, $B={M/2,M/2+1,M/2+2,...,M}$
even.odd: $A={0,2,4,...}$, $B={1,3,5,...}$

Where:
- $S_{ent}$ is the entanglement entropy.
- $\rho_A$ is the reduced density matrix of subsystem A.

## Instructions
When run, you'll be prompted with the size of the system and it will output a plot of the order parameters of the system as functions of hopping parameter $t$
