# Physics-Informed-Constrained-RL-OPF
# Constrained Policy Gradient for Optimal Power Flow (CPG-OPF)

This repository implements a **Physics-Informed Constrained Policy Gradient (CPG)** framework for solving the Optimal Power Flow (OPF) problem.

The project integrates:
- Deep Reinforcement Learning (DRL)
- Differentiable Power Flow constraints
- Constrained Policy Gradient optimization
- Comparison with conventional OPF solutions

---

## 📌 Overview

Traditional OPF solvers compute optimal dispatch under strict physical constraints.  
This project investigates whether a **physics-informed reinforcement learning agent** can learn to approximate OPF solutions while explicitly enforcing network constraints through constrained policy gradients.

Key features:

- Differentiable branch flow constraints
- Generator P/Q limits enforcement
- Voltage magnitude constraints
- Comparison with MATPOWER / PYPOWER OPF benchmarks
- Physics-consistent Actor network

---

## 🧠 Methodology

The agent is trained using:

\[
\max_\theta \; \mathbb{E}[Q(s,a)]
\]

Where:

- \( Q(s,a) \) is the critic-estimated return
- \( \mathcal{C}(s,a) \) is the constraint violation penalty
- \( \beta \) is the Lagrangian penalty coefficient

The Actor explicitly predicts:
- Generator active power \( P_g \)
- Voltage magnitudes \( V_m \)
- Voltage angles \( V_a \)

Branch flow limits and generator constraints are computed in a differentiable manner.
