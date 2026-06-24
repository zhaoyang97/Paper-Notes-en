---
title: >-
  [Paper Note] Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems
description: >-
  [AAAI2026][Physics & Scientific Computing][sparse feedback control] This paper proposes a randomized sparse feedback control framework in which the controller accesses only a random subset of the state vector at each time step. Feedback gain matrices and Bernoulli sparsification parameters are jointly designed via LMIs to guarantee asymptotic mean-square stability (AMSS) while minimizing the required number of active sensors. Experiments demonstrate that as few as 0.3% of sta…
tags:
  - "AAAI2026"
  - "Physics & Scientific Computing"
  - "sparse feedback control"
  - "randomized sparsification"
  - "asymptotic mean-square stability"
  - "LMI"
  - "large-scale systems"
date: 2026-05-08
content_hash: 2da84b82dbc8f668
---

# Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems

**Conference**: AAAI2026
**arXiv**: [2511.13870](https://arxiv.org/abs/2511.13870)  
**Code**: None  
**Area**: Scientific Computing
**Keywords**: sparse feedback control, randomized sparsification, asymptotic mean-square stability, LMI, large-scale systems

## TL;DR

This paper proposes a randomized sparse feedback control framework in which the controller accesses only a random subset of the state vector at each time step. Feedback gain matrices and Bernoulli sparsification parameters are jointly designed via LMIs to guarantee asymptotic mean-square stability (AMSS) while minimizing the required number of active sensors. Experiments demonstrate that as few as 0.3% of state components suffice to achieve performance comparable to full-state feedback.

## Background & Motivation

Classical control theory assumes that the controller has access to the complete state vector (or output) at every time step. In high-dimensional systems or resource-constrained environments, however, measuring the full state is prohibitively expensive. Large-scale systems such as power networks face strict constraints on sensor deployment, data acquisition, and communication bandwidth.

Existing methods address this problem primarily from a deterministic sparsity perspective:

- Reducing actuator/sensor usage via row/column sparsification of the control gain matrix $K$
- Obtaining fixed sparse structures through $\ell_1$-norm minimization
- Stability analysis of open-loop control strategies under deterministic sparsity

A common limitation of these approaches is that the sparse structure is fixed and deterministic, lacking systematic analysis of randomized sparsification strategies. This paper is the first to study the stability of control systems in which the feedback is formed from a **randomly selected subset of states**.

## Core Problem

Given a discrete-time linear system $x(k+1) = Ax(k) + Bu(k)$, where the controller can access only a random subset of the state vector $x(k)$ at each time step $k$, the paper addresses three questions:

1. **Stability guarantee**: Under randomized sparsification, when can the closed-loop system achieve AMSS, i.e., $\lim_{k\to\infty} \mathbb{E}\|x(k)\|_2^2 = 0$?
2. **Minimizing sensing requirements**: How can the feedback gain $K$ and sparsification strategy be jointly designed to minimize the expected number of active sensors?
3. **Heterogeneous sparsification**: When the measurement cost varies across state components, how should distinct sampling probabilities be assigned to each component?

## Method

### Randomized Sparsification Strategy

At each time step $k$, the sparsification matrix is a diagonal matrix:

$$\mathcal{C}(k) = \text{Diag}\left(\frac{c_1(k)}{p_1}, \ldots, \frac{c_n(k)}{p_n}\right)$$

where $c_i(k) \sim \text{Ber}(p_i)$ are independent Bernoulli random variables. The scaling by $p_i$ ensures unbiasedness: $\mathbb{E}[\mathcal{C}(k)] = I_n$. The control input is $u(k) = K\mathcal{C}(k)x(k)$, yielding closed-loop dynamics:

$$x(k+1) = (A + BK\mathcal{C}(k))x(k)$$

### Scenario 1: Uniform Sparsification

All components share the same Bernoulli parameter $p$. The core result (Proposition 1) provides an upper bound:

$$\mathbb{E}(\|x(k+1)\|_2^2) \leq f(p) \cdot \mathbb{E}(\|x(k)\|_2^2)$$

where $f(p) = \|D^\top D + \frac{1-p}{p} \cdot \text{Diag}(L^\top L)\|$, $D = A + BK$, and $L = BK$. The system is AMSS when $f(p) < 1$.

**Key Theorem (Theorem 1)**: Let $K_\gamma$ denote the LMI solution and $s_{\max} = \max_i \sum_k l_{k,i}^2$. The system is AMSS when $p > p_{K_\gamma} = \frac{1}{1 + \alpha_\gamma}$, where $\alpha_\gamma = \frac{1 - \|D_\gamma\|^2}{s_{\max}}$.

### Scenario 2: Adaptive Sparsification

Each state component is assigned an independent probability $p_i$. Theorem 2 provides a component-level condition:

$$p_i > \frac{1}{1 + \frac{1 - \|D_\gamma\|^2}{s_i}}$$

where $s_i = \sum_k l_{k,i}^2$ reflects the influence of the $i$-th state component on the control input. Components with small $s_i$ can be dropped more frequently, while those with large $s_i$ require higher sampling rates.

### LMI Solving Framework

Existence conditions rest on two assumptions: (1) $B$ has full column rank; (2) the largest singular value $a_n$ of $(I_n - B(B^\top B)^{-1}B^\top)A$ satisfies $a_n < 1$. The intuition behind Assumption 2 is that the action of $A$ along directions orthogonal to the image of $B$ must be bounded, ensuring that feedback through $BK$ can sufficiently correct the system dynamics.

**Algorithm 1** (Uniform Sparsification):
1. Compute $a_n$ and discretize $[a_n, 1]$.
2. For each $\gamma$, solve the LMI to obtain $K_\gamma$ and compute $p_{K_\gamma}$.
3. Return the minimum $p^\star$ and the corresponding $K$.

**Algorithm 2** (Adaptive Sparsification): Extending Algorithm 1, compute component-level $p_{i,K_\gamma}$ for each $\gamma$ and minimize the weighted expected sparsity $ES = \sum_i w_i p_i$.

### Observer Extension

Via the separation principle, when $(A, C)$ is observable/detectable, states reconstructed by a Luenberger observer can replace the true states for randomized sparsification, with the theoretical guarantees remaining intact.

## Key Experimental Results

### Grid-Forming Converter ($n=3$)

| Metric | Value |
|--------|-------|
| Uniform sparsification $p^\star$ | 0.79 |
| Adaptive sparsification $\mathbf{p}^\star$ | [0.026, 0.026, 0.794] |
| Expected active sensors | Uniform: 2.37/3; Adaptive: 0.846/3 |

The adaptive strategy identifies that the first two state components have little influence on the control input ($p_i = 0.026$), and only the third component requires a high sampling rate.

### Large-Scale Power System ($n=1000$, state dimension 2000)

| Metric | Value |
|--------|-------|
| Uniform sparsification $p^\star$ | 0.0026 |
| Expected active sensors | $\approx 5.2$ / 2000 (**0.26%**) |
| Minimum measurements (deterministic) | 37 (NP-hard problem) |

Monte Carlo simulations confirm that all trajectories converge in mean square to zero for $p \geq p^\star$, while the system diverges for $p < p^\star$. Higher sparsification (smaller $p$) yields slower convergence rates, consistent with the theoretical analysis (Remark 1).

## Highlights & Insights

1. **First study of randomized sparse feedback**: All prior work on sparse control employs deterministic methods; this paper opens a new direction for randomized sparsification.
2. **Extreme sparsification**: In a 1000-node system, only 0.3% of state measurements suffice to guarantee AMSS, far outperforming the 37 measurements required by deterministic methods.
3. **Theoretical completeness**: A complete stability hierarchy is established, from expected stability to mean-square stability to almost sure stability.
4. **Adaptive mechanism**: Component-level probability assignment naturally allocates higher sampling rates to high-influence states and lower rates to low-influence states, accommodating heterogeneous sensing costs.
5. **Computational feasibility**: The LMI-based algorithm can be efficiently solved with standard convex optimization solvers (MOSEK).

## Limitations & Future Work

1. **Conservatism**: The upper bound analysis based on spectral norm inequalities may be overly conservative; $p^\star$ may not be a tight lower bound.
2. **Noise-free assumption**: The current analysis does not account for process or measurement noise, which are inevitable in practical systems.
3. **Linear systems only**: The theoretical framework applies only to linear systems; extension to nonlinear systems requires new tools.
4. **Strength of Assumption 2**: Requiring $a_n < 1$ is stronger than classical stabilizability conditions, limiting the scope of applicability.
5. **Discretization accuracy**: Algorithms 1 and 2 rely on grid search over $\gamma \in [a_n, 1]$; accuracy depends on the step size $\delta$.
6. **Lack of physical experiments**: Validation is limited to numerical simulations without experiments on real physical systems.

## Related Work & Insights

| Method | Sparsity Type | Feedback Type | Stability | Adaptive |
|--------|--------------|--------------|-----------|----------|
| $\ell_1$-norm methods | Deterministic (fixed zero structure) | Closed-loop | Asymptotic | ✗ |
| Row cardinality constraints | Deterministic (mixed-integer) | Closed-loop | Asymptotic | ✗ |
| $s$-sparse control | Deterministic | Open-loop | Asymptotic | ✗ |
| Sensor scheduling | Deterministic/periodic | Estimator | Estimation convergence | ✗ |
| **Ours** | **Randomized (Bernoulli)** | **Closed-loop** | **AMSS** | **✓** |

The key distinction is that this paper presents the first framework to jointly design time-varying randomized sensor selection and control gains to guarantee global asymptotic mean-square stability.

The sparsification strategy is directly inspired by gradient sparsification in distributed learning (e.g., top-$k$ sparsification by Wangni et al.), reflecting deep connections between optimization and control. The framework has direct applicability to ultra-high-dimensional systems such as smart grids and transportation networks. The paradigm of random measurements combined with stability guarantees bears structural resemblance to the RIP conditions in compressed sensing. The approach also extends naturally to networked control systems with communication bandwidth constraints.

## Rating

- Novelty: ⭐⭐⭐⭐ — First introduction of randomized sparsification into the stability analysis of feedback control; problem formulation is original.
- Experimental Thoroughness: ⭐⭐⭐ — Numerical simulations adequately validate the theory, but real-system experiments are absent.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are rigorous, the structure is clear, and notation is consistently defined.
- Value: ⭐⭐⭐⭐ — Opens a new research direction with important implications for large-scale IoT control systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Supervised Evolution Operator Learning for High-Dimensional Dynamical Systems](../../ICLR2026/physics/self-supervised_evolution_operator_learning_for_high-dimensional_dynamical_syste.md)
- [\[ICML 2026\] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval](../../ICML2026/physics/mathbbr2k_is_theoretically_large_enough_for_embedding-based_top-k_retrieval.md)
- [\[ICLR 2026\] Feedback-driven Recurrent Quantum Neural Network Universality](../../ICLR2026/physics/feedback-driven_recurrent_quantum_neural_network_universality.md)
- [\[ICLR 2026\] Enhancing Stability of Physics-Informed Neural Network Training Through Saddle-Point Reformulation](../../ICLR2026/physics/enhancing_stability_of_physics-informed_neural_network_training_through_saddle-p.md)
- [\[ICLR 2026\] RealPDEBench: A Benchmark for Complex Physical Systems with Real-World Data](../../ICLR2026/physics/realpdebench_a_benchmark_for_complex_physical_systems_with_real-world_data.md)

</div>

<!-- RELATED:END -->
