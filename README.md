# Two-Level Inexact ADMM–DFO
This repository provides a reference implementation of a **two-level inexact ADMM framework for distributed derivative-free optimization (DFO)**.  
The method supports both **smooth and nonsmooth trust-region subproblems** and is designed for scalable optimization of additive, high-dimensional black-box objectives with linear constraints.


## Repository Structure
- `two_level_inexact_admm_dfo.py` — Core solver implementing the two-level Inexact ADMM–DFO algorithm  
- `arrowhead_main.py` — Benchmark 1: Arrowhead test function (convex & smooth)  
- `rosenbrock_main.py` — Benchmark 2: Rosenbrock test function (nonconvex & smooth)


## External DFO Solvers
This framework interfaces with external derivative-free optimization (DFO) solvers that must be installed separately.
These solvers are not included in this repository and remain under their respective licenses.
• DFO Algorithm
Repository: https://github.com/TheClimateCorporation/dfo-algorithm
• Advanced DFO-TRNS Algorithm (Advanced DFO trust-region algorithm for nonsmooth problems)
Repository: https://github.com/DerivativeFreeLibrary/DFTRNS

##  Python Version
This implementation was done using **Python 3.13t (nogil build)**, which allows disabling the Global Interpreter Lock (GIL) for efficient multithreading.  
