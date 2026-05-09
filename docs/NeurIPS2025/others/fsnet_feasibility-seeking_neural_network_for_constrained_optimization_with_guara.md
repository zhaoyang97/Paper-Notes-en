---
title: >-
  [Paper Note] FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees
description: >-
  [NeurIPS 2025][constrained optimization] This paper proposes FSNet, a framework that integrates **differentiable feasibility-seeking steps** into neural networks. By minimizing constraint violations via unconstrained optimization, FSNet guarantees constraint satisfaction while supporting end-to-end training. It significantly outperforms traditional solvers in speed across convex/non-convex and smooth/non-smooth problems while maintaining feasibility.
tags:
  - NeurIPS 2025
  - constrained optimization
  - feasibility seeking
  - neural network surrogate
  - end-to-end training
  - convergence guarantees
date: 2026-05-08
content_hash: 77cfb12bd5644420
---

# FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2506.00362](https://arxiv.org/abs/2506.00362)
**Code**: [GitHub](https://github.com/MOSSLab-MIT/FSNet)
**Area**: Other
**Keywords**: constrained optimization, feasibility seeking, neural network surrogate, end-to-end training, convergence guarantees

## TL;DR

This paper proposes FSNet, a framework that integrates **differentiable feasibility-seeking steps** into neural networks. By minimizing constraint violations via unconstrained optimization, FSNet guarantees constraint satisfaction while supporting end-to-end training. It significantly outperforms traditional solvers in speed across convex/non-convex and smooth/non-smooth problems while maintaining feasibility.

## Background & Motivation

Using ML methods as fast surrogates for constrained optimization has become a prominent trend, yet **constraint satisfaction** remains a critical bottleneck:

- **Traditional solvers**: Guarantee feasibility but are computationally expensive and unsuitable for real-time applications.
- **Penalty methods**: Provide no feasibility guarantees and often yield severely infeasible solutions.
- **Projection methods**: Theoretically feasible but projection onto general constraint sets is costly.
- **Iterative methods** (DC3, ENFORCE): Handle equality and inequality constraints separately without unified guarantees.
- **Key Challenge**: The fundamental trade-off between speed and feasibility.

## Method

### Overall Architecture

FSNet follows a **predict → feasibility-seek → projection-inspired loss** pipeline. Given input parameter $x$, the neural network produces prediction $y_\theta(x)$, which is then corrected by a feasibility-seeking step to yield a feasible solution $\hat{y}_\theta(x)$. The entire pipeline is trained end-to-end via unrolled gradients.

### Key Designs

**1. Feasibility-Seeking Step (Core Contribution)**

An unconstrained problem minimizing constraint violations is defined as:

$$\min_s \phi(s;x) := \|h(s;x)\|_2^2 + \|g^+(s;x)\|_2^2$$

Gradient descent is performed starting from the neural network's prediction. Key advantages: the unconstrained formulation (simpler than projection), unified handling of equality and inequality constraints, and practical use of L-BFGS.

**2. Projection-Inspired Loss**

$$F = f(\hat{y}_\theta; x) + \frac{\rho}{2}\|y_\theta - \hat{y}_\theta\|_2^2$$

The first term drives the objective value lower; the second encourages the neural network prediction to remain close to the feasible solution. Geometrically, this acts as a soft projection.

**3. End-to-End Unrolled Differentiation**

The feasibility-seeking iterations are unrolled and differentiated via automatic differentiation to propagate gradients back to the neural network. Since $y_\theta$ does not appear in the KKT conditions of $\phi$, implicit differentiation is not applicable.

**4. Truncated Gradient Tracking**

Gradients are computed only for the first $K'$ steps, with truncation bias decaying exponentially in $K'$.

### Loss & Training

The training loss consists of the expected objective value plus a penalty on the distance between the prediction and the feasible solution. To stabilize early training, an additional penalty is applied when violations exceed threshold $Q$. The SGD step size follows a schedule of $\eta_0/\sqrt{T}$.

### Convergence Guarantees

**Theorem 1**: Under the PL condition, feasibility converges exponentially in $\mathcal{O}(\log(1/\epsilon))$ steps.

**Theorem 2**: Training gradients are $\mathcal{O}(T^{-1/4} + \gamma^K)$, with truncation bias decaying exponentially.

**Theorem 3**: When the neural network is sufficiently expressive and $\rho \to \infty$, the method converges to the global optimum.

## Key Experimental Results

### Main Results 1: Convex Problems (100 Variables)

| Method | Equality Violation | Inequality Violation | Optimality Gap |
|--------|--------------------|----------------------|----------------|
| Solver | ≈0 | ≈0 | Baseline |
| Penalty | Large | Large | Negative (illusory) |
| DC3 | Low | High | Negative |
| **FSNet** | **≈0** | **≈0** | **< 0.2%** |

### Main Results 2: Large-Scale Convex Problems (500 Variables)

| Problem | Batch Speedup | Sequential Speedup |
|---------|---------------|--------------------|
| QP | 188× | 8× |
| QCQP | 147× | 143× |
| SOCP | 36× | 2754× |

### Main Results 3: Non-Convex Problems

| Problem | Constraint Violation | Optimality | Speedup |
|---------|----------------------|------------|---------|
| NC-QP | ≈0 | Matches solver | 85–3000× |
| NC-QCQP | ≈0 | **Superior** | 85–3000× |
| NC-SOCP | ≈0 | **Superior** | 85–3000× |

### Ablation Study

- A small number of truncation steps is sufficient to maintain performance.
- Performance is insensitive to $\rho$ over a wide range.
- ACOPF: 3× speedup on 30-bus, 11× on 118-bus, with optimality gap < 1%.
- Non-smooth non-convex setting: 152–3316× batch speedup.

### Key Findings

1. FSNet is the only ML method achieving near-zero constraint violations across all problem classes.
2. The "low optimality gaps" reported by other methods are artifacts of infeasibility.
3. FSNet surpasses IPOPT on non-convex problems.
4. The unconstrained feasibility-seeking formulation is the key to computational efficiency.

## Highlights & Insights

1. **Paradigm shift**: Predict-then-correct, with the unconstrained formulation being the essential insight.
2. **Three-level theoretical guarantees**: Convergence for feasibility, training, and optimality.
3. **Strong practicality**: L-BFGS combined with truncated gradients yields efficiency and stability.
4. **Unexpected advantage on non-convex problems**: Implicit regularization from the neural network can outperform local solvers.
5. **Unified framework**: A single $\phi$ handles all constraint types.

## Limitations & Future Work

1. Theoretical analysis relies on the PL condition and smoothness assumptions.
2. Scalability to extremely large-scale problems remains to be validated.
3. The feasibility-seeking step introduces additional inference overhead.
4. The assumptions in Theorem 3 are relatively strong.
5. More efficient feasibility-seeking algorithms could be explored.

## Related Work & Insights

- **DC3**: A pioneering constraint correction method but without guarantees.
- **OptNet**: Embeds QP solvers into neural networks but is limited by problem structure.
- **Inspiration**: The feasibility-seeking plus end-to-end training paradigm can be extended to physics-informed neural networks and safe reinforcement learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — Concise and effective, though built on a combination of existing elements.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of convex/non-convex and smooth/non-smooth settings.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with good alignment between theory and experiments.
- Value: ⭐⭐⭐⭐⭐ — Addresses a critical pain point in constraint satisfaction; open-source.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)
- [\[NeurIPS 2025\] Neural Network for Simulating Radio Emission from Extensive Air Showers](neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](depth-supervised_fusion_network_for_seamless-free_image_stitching.md)

<!-- RELATED:END -->
