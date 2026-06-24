---
title: >-
  [Paper Note] Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics
description: >-
  [ICML2025][Sensor scheduling] This work proposes an optimal sensor scheduling framework for continuous-discrete Kalman filtering (CD-KF). By modeling multi-sensor observations as independent Poisson processes, it derives a continuously differentiable upper bound for the posterior covariance matrix. A gradient-based optimization method is then utilized to jointly optimize observation rates and auxiliary dynamics inputs, and the deterministic observation times are selected via…
tags:
  - "ICML2025"
  - "Sensor scheduling"
  - "Kalman filtering"
  - "continuous-discrete systems"
  - "Poisson processes"
  - "optimal control"
  - "Gaussian process regression"
date: 2026-05-08
content_hash: 4f52a8fd5ffbfd97
---

# Optimal Sensor Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics

**Conference**: ICML2025  
**arXiv**: [2507.11240](https://arxiv.org/abs/2507.11240)  
**Code**: [GitHub](https://github.com/MOHAMMADZAHD93/When2measureKF)  
**Area**: Control & Filtering  
**Keywords**: Sensor scheduling, Kalman filtering, continuous-discrete systems, Poisson processes, optimal control, Gaussian process regression

## TL;DR

This work proposes an optimal sensor scheduling framework for continuous-discrete Kalman filtering (CD-KF). By modeling multi-sensor observations as independent Poisson processes, it derives a continuously differentiable upper bound for the posterior covariance matrix. A gradient-based optimization method is then utilized to jointly optimize observation rates and auxiliary dynamics inputs, and the deterministic observation times are selected via Wasserstein-2 optimal quantization.

## Background & Motivation

The states of many real-world systems (e.g., blood pressure, ocean temperature, radiation levels) evolve continuously, but can only be intermittently observed via discrete, irregular sensors. This represents a typical scenario for **continuous-discrete Kalman filtering (CD-KF)**.

The practical challenges lie in:

- **Multi-sensor heterogeneity**: Different sensors have varying accuracies, energy consumptions, and constraints.
- **Auxiliary state coupling**: The observation process is coupled with auxiliary state-space models (e.g., sensor temperature affects accuracy, energy depletes with observations).
- **Resource limitations**: It is impractical to simply sample all sensors at their maximum frequencies.

Prior works exhibit gaps in the following aspects:

1. Most works only consider discrete-time settings, leaving the continuous-discrete scenario unaddressed.
2. No existing method simultaneously considers the coupling between sensor scheduling and general auxiliary state dynamics.

This work fills this gap by proposing a unified optimal control framework.

## Method

### 1. System Model

The continuous-discrete state-space model (SSM) is described by the following equations:

**State evolution** (continuous-time SDE):

$$dx = A(\xi, t) x \, dt + \sigma(\xi, t) \, dW, \quad x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$$

**Discrete observations** (sensor $s$ at time $t_i$):

$$y^s(t_i) = C_s(\xi(t_i), t_i) x(t_i) + v^s(\xi(t_i), t_i)$$

where $\xi$ represents the auxiliary states (divided into the perturbed part $\xi_p$ and unperturbed part $\xi_u$), and $v^s \sim \mathcal{N}(0, R_s(\xi, t))$.

### 2. Poisson Process Modeling

The observation arrivals of each sensor $s$ are modeled as non-homogeneous Poisson processes $N_s(t)$ with rate $\lambda_s(t)$. The randomized covariance evolution is given by:

$$d\Sigma = \left(A\Sigma + \Sigma A^\top + \sigma\sigma^\top\right)dt - \sum_{s=1}^{S} K_s \cdot C_s \cdot \Sigma \, dN_s$$

### 3. Covariance Upper Bound (Proposition 6.1)

Core theoretical contribution: A **continuously differentiable upper bound** $\hat{\Sigma}(t)$ of the mean posterior covariance matrix is derived, which satisfies the ODE:

$$\frac{d\hat{\Sigma}}{dt} = A\hat{\Sigma} + \hat{\Sigma}A^\top + \sigma\sigma^\top - \sum_{s=1}^{S} \lambda_s(t) K_s C_s \hat{\Sigma}$$

Key property: $\bar{\Sigma}(t; \xi^*) \preceq \hat{\Sigma}(t)$ (under Loewner order), and $\hat{\Sigma}$ is continuously differentiable with respect to $\lambda_s$, making it directly amenable to gradient optimization.

### 4. Upper Bound on Auxiliary States (Proposition 6.2)

Under the concavity/convexity assumptions (Assumption 5.1), the mean auxiliary state $\bar{\xi}$ can be upper-/lower-bounded by a deterministic trajectory $\hat{\xi}$:

$$\frac{d\hat{\xi}_p}{dt} = f_p(\hat{\xi}, u, t) + \sum_{s=1}^{S} \lambda_s(t) g_s(\hat{\xi}, u, t)$$

For affine systems, $\hat{\xi}_p = \bar{\xi}_p$ (exact equality).

### 5. Optimal Control Problem (OCP)

Jointly optimize the observation rates $\lambda$ and auxiliary inputs $u$:

$$\min_{\lambda \geq 0, \, u \in \mathcal{U}} \int_0^T \mathcal{L}(\hat{\xi}, \hat{\Sigma}, u, \lambda) \, dt + \mathcal{L}_T(\cdot)$$

Example running cost: $\mathcal{L} = w_\Sigma \operatorname{tr}(\hat{\Sigma}) + w_\lambda \|\lambda\|^2 + w_u \|u\|^2 + w_\varepsilon \varepsilon^2$

Constraints include: energy lower bound $\hat{\eta} \geq c_\eta$, input upper and lower bounds, and covariance trace constraint $\operatorname{tr}(\hat{\Sigma}) \leq c_\Sigma + \varepsilon$, etc.

### 6. Selection of Deterministic Observation Times (Proposition 8.1)

After obtaining the optimized continuous rates $\lambda_s(t)$, the deterministic observation times are selected by minimizing the **Wasserstein-2 distance**:

$$\bar{t}_i^s = \mathbb{E}[\tau_s \mid \tau_s \in [a_{i-1}^s, a_i^s]] = \frac{\int_{a_{i-1}^s}^{a_i^s} t \lambda_s(t) dt}{\int_{a_{i-1}^s}^{a_i^s} \lambda_s(t) dt}$$

The timeline is partitioned into $n_s$ intervals of equal cumulative intensity $\Lambda_s(t)$, and the conditional centroid of each interval is selected. This method preserves first-moment matching and minimizes quantization error.

## Key Experimental Results

### Experimental Setup

- **Scenario**: A robot equipped with two heterogeneous sensors performs temporal Gaussian process regression (with a Matérn kernel).
- **Sensor 1**: High precision (small $R_{1_{\max}}$), high energy consumption (large $c_1$).
- **Sensor 2**: Low precision, low energy consumption.
- **Observation noise**: Dependent on physical distance between the robot and target $R_s(p_r) = R_{s_{\max}} \exp(r_s \|p_r - p_p\|^2)$.
- **Extended scenario**: Radioactive environment, where observation causes cumulative radiation degradation $\zeta_s$.

### Baseline Methods

| Method | Description |
|------|------|
| **Optimized** (Ours) | Jointly optimizes $\lambda$ and $u$, using deterministic quantization for interval selection |
| M-Optimized | Optimizes rates via the same approach, but selects the best outcome by sampling the Poisson process multiple times |
| Greedy | Greedily selects the sensor with the highest score at each step |
| Random | Applies random Poisson sampling with a uniform rate |

### Core Results (Radioactive Environment)

| Metric | Method | Mean | Std | Max |
|------|------|------|--------|--------|
| Covariance Trace | **Optimized** | **1.902** | **0.341** | **2.945** |
| | M-Optimized | 1.925 | 0.364 | 3.001 |
| | Greedy | 2.608 | 0.488 | 3.219 |
| | Random | 2.228 | 0.470 | 2.921 |
| Remaining Energy $\eta$ | **Optimized** | **21.54** | 10.17 | 50.0 |
| | Greedy | -1.67 | 14.06 | 50.0 |
| | Random | -17.44 | 31.35 | 50.0 |
| Radiation Degradation | **Optimized** | **0.091** | 0.065 | 0.186 |
| | Greedy | 0.193 | 0.081 | 0.232 |
| | Random | 0.792 | 0.558 | 1.576 |

**Key Observation**: Optimized achieves the lowest covariance trace (estimation error) while maintaining positive remaining energy (21.54 vs Greedy -1.67 and Random -17.44) and the lowest radiation degradation.

## Highlights & Insights

1. **Theoretical elegance**: Formulating stochastic discrete observations via Poisson processes converts the problem into a continuously differentiable objective, elegantly bypassing the combinatorial optimization challenge.
2. **Unified framework**: It is the first to unify sensor scheduling, auxiliary state dynamics, and trajectory optimization within a single optimal control framework.
3. **Deterministic quantization**: The Wasserstein-2 optimal quantization-based selection of observation times offers closed-form solutions, high computational efficiency, and preserved first moments.
4. **Broad applicability**: The framework can be extended to non-linear SSMs (using EKF/UKF approximations), water quality monitoring (Appendix E), and spacecraft monitoring (Appendix F).
5. **Practical significance**: While Greedy and Random baselines frequently violate constraints in constrained scenarios (limited energy, radiation degradation), the proposed method inherently satisfies them.

## Limitations & Future Work

1. **Linear assumptions**: The core theory addresses linear SSMs; non-linear cases only provide approximate guarantees (similar to EKF assumptions).
2. **Open-loop planning**: The method solves a finite-horizon open-loop OCP without considering online feedback or closed-loop replanning (e.g., in an MPC fashion).
3. **Computational cost**: Numerical solving of the OCP (via direct collocation or multiple shooting methods) can be computationally expensive for high-dimensional state spaces.
4. **Poisson approximation**: Sensor observation events in real-world scenarios might not rigorously follow a Poisson process.
5. **Single-target process**: Experiments only consider a single observed process; scaling to simultaneous multi-target monitoring has not yet been discussed.

## Related Work & Insights

- Compared to the continuous-time sensor management in (Ny et al., 2009), this work handles **discrete irregular observations**.
- Compared to the Neural ODE approach in (Qin et al., 2024), this work provides an **analytical differentiable upper bound** instead of black-box learning.
- The concept of Wasserstein optimal quantization can inspire other problems requiring the selection of discrete points from continuous distributions.
- The conceptualization of auxiliary state modeling (e.g., energy, degradation) can be extended to active perception in sensor networks and autonomous driving.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to unify CD-KF + auxiliary dynamics + Poisson scheduling
- Experimental Thoroughness: ⭐⭐⭐ — Convincing scenarios, but limited to low-dimensional simulations
- Writing Quality: ⭐⭐⭐⭐ — Mathematically rigorous and well-structured
- Value: ⭐⭐⭐⭐ — Highly relevant for active sensing under resource constraints

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes](../../ACL2025/others/neodiff_unified_text_diffusion.md)
- [\[ICML 2025\] Discrete Neural Algorithmic Reasoning](discrete_neural_algorithmic_reasoning.md)
- [\[ICML 2025\] Optimal Auction Design in the Joint Advertising](optimal_auction_design_in_the_joint_advertising.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](../../NeurIPS2025/others/continuous_thought_machines.md)

</div>

<!-- RELATED:END -->
