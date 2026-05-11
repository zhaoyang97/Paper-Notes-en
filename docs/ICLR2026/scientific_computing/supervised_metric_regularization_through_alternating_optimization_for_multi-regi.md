---
title: >-
  [Paper Note] Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs
description: >-
  [Scientific Computing] This paper proposes a Topology-Aware PINN (TAPINN) that structures the latent space via supervised metric regularization (Triplet Loss) and stabilizes training through an alternating optimization s…
tags:
  - "Scientific Computing"
date: 2026-05-08
content_hash: 616a313793f4c41b
---

# Supervised Metric Regularization Through Alternating Optimization for Multi-Regime PINNs

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2602.09980](https://arxiv.org/abs/2602.09980)
- **Code**: Not released
- **Area**: Scientific Computing / Physics-Informed Neural Networks
- **Keywords**: PINN, metric learning, alternating optimization, bifurcation systems, Duffing oscillator, topology-aware

## TL;DR

This paper proposes a Topology-Aware PINN (TAPINN) that structures the latent space via supervised metric regularization (Triplet Loss) and stabilizes training through an alternating optimization schedule. On the multi-regime Duffing oscillator benchmark, TAPINN reduces physics residuals by approximately 49% (0.082 vs. 0.160) and gradient variance by 2.18× compared to baselines.

## Background & Motivation

Physics-Informed Neural Networks (PINNs) have shown promise for solving parameterized dynamical systems, but face fundamental challenges in systems with **sharp regime transitions** (e.g., bifurcations):

**Spectral Bias**: Standard MLPs struggle to approximate discontinuous or non-smooth solution dependencies on system parameters.

**Mode Collapse**: Networks tend to average across distinct physical behaviors rather than discriminating among them.

**Jacobian Singularity at Bifurcation Points**: Leads to ill-conditioned optimization.

Limitations of existing approaches:
- **HyperPINNs**: Hypernetwork-generated weights incur high parameter counts (39,169 vs. 8,003).
- **MoE**: Routing instability.
- Both introduce additional architectural complexity.

**Core Idea**: Rather than adopting more complex architectures, TAPINN structures the latent space via metric learning so that it mirrors the separation of physical regimes.

## Method

### Overall Architecture

TAPINN = LSTM Encoder $E$ + PINN Generator $G$

- **Encoder**: $z = E(\mathbf{x}_{\text{obs}})$, maps an observation window (first 100 time steps) to a latent vector $z$.
- **Generator**: $\hat{\mathbf{x}}(t) = G(t, z)$, a 4-layer MLP (32 hidden units, tanh activation).

Key distinction: TAPINN infers regime information solely from the observation window, without requiring the known parameter $\lambda$ (unlike parameterized baselines and HyperPINN).

### Composite Loss Function

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \mathcal{L}_{\text{physics}} + \beta \mathcal{L}_{\text{metric}}$$

- **Data Loss** $\mathcal{L}_{\text{data}}$: Reconstruction error over the observation window.
- **Physics Loss** $\mathcal{L}_{\text{physics}} = \frac{1}{N_c}\sum\|\mathcal{N}[\hat{\mathbf{x}}(t_i);\lambda]\|^2_2$: ODE residual evaluated at $N_c = 10^4$ collocation points.
- **Metric Loss** $\mathcal{L}_{\text{metric}} = \max(0, d(z_a, z_p) - d(z_a, z_n) + m)$: Triplet Loss with margin $m = 0.2$.

### Alternating Optimization (AO) Schedule

To mitigate gradient conflicts between the metric and physics objectives:

1. **Phase I (Metric Alignment, 5 epochs)**: Optimize the encoder only using the Triplet Loss to organize the latent space.
2. **Phase II (Physics Reconstruction, 20 epochs)**: Freeze the encoder and optimize the generator only.
3. **Alternating Joint Fine-tuning**: Joint updates every $k=5$ batches (~20% of steps) over $\mathcal{L}_{\text{total}}$.

Intuition: Stabilize the latent manifold first (separating embeddings of different regimes), then train the solver conditioned on a stable $z$.

### Triplet Construction

The driving amplitude $F_0$ serves as a proxy for regime similarity:
- **Anchor/Positive**: Share the same $F_0$.
- **Negative**: Differ in $F_0$.
- Constructed within each batch using Euclidean distance, without hard or semi-hard mining.

## Experiments

### Test Problem: Duffing Oscillator

$$\ddot{x} + \delta\dot{x} + \alpha x + \beta x^3 = F_0 \cos(\omega t)$$

Standard parameters $\delta=0.3, \alpha=-1, \beta=1, \omega=1$; $F_0 \in [0.3, 0.8]$ spans periodic to chaotic regimes.

### Main Results

| Method | Physics Res. ↓ | # Params | Data MSE ↓ |
|--------|---------------|----------|-----------|
| Parametric Baseline | 0.160 | 8,577 | 0.392 |
| Multi-Output (Sobolev) | 0.192 | 8,069 | 0.426 |
| HyperPINN | 0.158 | **39,169** | **0.281** |
| **TAPINN (Ours)** | **0.082** | 8,003 | 0.425 |

### Key Findings

1. **Lowest Physics Residual**: TAPINN achieves a physics residual of 0.082, 49% lower than the parametric baseline (0.160).
2. **Parameter Efficiency**: Only 8,003 parameters vs. HyperPINN's 39,169 (~5× fewer).
3. **HyperPINN Overfitting**: HyperPINN achieves the lowest Data MSE (0.281) but a high physics residual (0.158), indicating memorization of data at the expense of physical consistency.
4. **Training Stability**: Gradient norm mean is 2.14× lower and variance is 2.18× lower compared to the multi-output baseline.
5. **Latent Space Structure**: t-SNE visualization reveals well-separated clusters for distinct regimes; linear probe regression of $F_0$ achieves MSE of only $3.5 \times 10^{-4}$.
6. **Necessity of AO**: Joint training without AO yields physics residuals of ~0.158, comparable to standard baselines, demonstrating that metric regularization alone is insufficient.

## Highlights & Insights

- Elegant formulation: metric learning is used to structure the latent space for regime transitions rather than increasing architectural complexity.
- The AO schedule is well-motivated and effectively resolves gradient conflicts between metric and physics objectives.
- TAPINN achieves the best physics residual with only 1/5 the parameters of HyperPINN.
- Reveals a "memorization pathology" in HyperPINN: data fitting at the cost of physical law violation.

## Limitations & Future Work

- Validation is limited to the Duffing oscillator (1D ODE); no experiments on PDE systems or higher-dimensional problems.
- Lacks statistical validation across multiple random seeds.
- Sensitivity to observation window length is not analyzed.
- Hyperparameters $\alpha$ and $\beta$ are selected via grid search without an adaptive strategy.
- No comparison against domain decomposition methods (XPINNs) or operator learning frameworks (Fourier Neural Operator).
- Although physics residuals are low, Data MSE is higher than HyperPINN, leaving trajectory reconstruction accuracy to be further verified.

## Related Work & Insights

- **Parameterized PINNs**: Standard approaches that take $\lambda$ as input directly fail near bifurcations.
- **HyperPINNs**: Almeida et al. — weight generation for regime transitions, at the cost of high parameter counts.
- **MoE-PINN**: Bischof & Kraus — mixture-of-experts routing, subject to routing instability.
- **PINN Optimization Pathologies**: Krishnapriyan et al. — characterizes failure modes of PINNs.
- **Gradient Pathology Mitigation**: Wang et al. — gradient flow ill-conditioning in PINNs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of metric learning and PINNs is novel.
- **Technical Depth**: ⭐⭐⭐⭐ — Method design is principled with sufficient ablation evidence.
- **Experimental Thoroughness**: ⭐⭐⭐ — Only one test problem (Duffing oscillator); limited scale.
- **Value**: ⭐⭐⭐⭐ — Provides a lightweight solution for multi-regime PINNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HyperKKL: Enabling Non-Autonomous State Estimation through Dynamic Weight Conditioning](hyperkkl_enabling_non-autonomous_state_estimation_through_dynamic_weight_conditi.md)
- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](../../NeurIPS2025/scientific_computing/oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] A Regularized Newton Method for Nonconvex Optimization with Global and Local Complexity Guarantees](../../NeurIPS2025/scientific_computing/a_regularized_newton_method_for_nonconvex_optimization_with.md)
- [\[AAAI 2026\] PIMRL: Physics-Informed Multi-Scale Recurrent Learning for Burst-Sampled Spatiotemporal Dynamics](../../AAAI2026/scientific_computing/pimrl_physics-informed_multi-scale_recurrent_learning_for_burst-sampled_spatiote.md)
- [\[NeurIPS 2025\] Towards Universal Neural Operators through Multiphysics Pretraining](../../NeurIPS2025/scientific_computing/towards_universal_neural_operators_through_multiphysics_pretraining.md)

</div>

<!-- RELATED:END -->
