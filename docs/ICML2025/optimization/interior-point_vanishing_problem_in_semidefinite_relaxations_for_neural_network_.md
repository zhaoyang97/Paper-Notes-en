---
title: >-
  [Paper Note] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification
description: >-
  [ICML2025][Optimization][SDP relaxation] This work is the first to identify the "interior-point vanishing" problem when SDP relaxation is applied to deep neural network verification, where the SDP problem loses strict feasibility as the network depth increases, resulting in numerical instability and solver failures. The authors propose five mitigation methods, among which B-Remove (removing layer boundary constraints) is the most effective, resolving 88% of previously unsolva…
tags:
  - "ICML2025"
  - "Optimization"
  - "SDP relaxation"
  - "Neural network verification"
  - "Interior-point method"
  - "Strict feasibility"
  - "ReLU"
  - "Adversarial robustness"
date: 2026-05-08
content_hash: 3a3c3160af65572b
---

# Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification

**Conference**: ICML2025  
**arXiv**: [2506.10269](https://arxiv.org/abs/2506.10269)  
**Code**: Not released  
**Area**: Neural Network Verification / Optimization  
**Keywords**: SDP relaxation, Neural network verification, Interior-point method, Strict feasibility, ReLU, Adversarial robustness

## TL;DR

This work is the first to identify the "interior-point vanishing" problem when SDP relaxation is applied to deep neural network verification, where the SDP problem loses strict feasibility as the network depth increases, resulting in numerical instability and solver failures. The authors propose five mitigation methods, among which B-Remove (removing layer boundary constraints) is the most effective, resolving 88% of previously unsolvable cases.

## Background & Motivation

**DNN Verification Problem**: Given a classification model $\boldsymbol{f}: \mathbb{R}^d \to \mathbb{R}^m$, an input $\bar{\boldsymbol{x}}$, and a perturbation radius $\rho$, the goal is to determine whether the predicted label remains unchanged within $\|\boldsymbol{x}_0 - \bar{\boldsymbol{x}}\|_\infty \le \rho$. This is equivalent to solving the optimization problem:

$$\gamma^* = \min_{\{\boldsymbol{x}_i\}} \boldsymbol{c}^\top \boldsymbol{x}_L + c_0 \quad \text{s.t. ReLU constraints, perturbation constraints, boundary constraints}$$

The model is robust to this input when $\gamma^* > 0$. Since ReLU makes the problem non-convex (NP-hard), **convex relaxation** is adopted in practice for approximation.

**Advantages and Dilemmas of SDP Relaxation**: SDP relaxation introduces a matrix variable $\boldsymbol{P} = \boldsymbol{v}\boldsymbol{v}^\top$ via polynomial lifting and relaxes the rank-1 constraint, making it widely recognized as one of the tightest convex relaxation methods. However, SDP methods are rarely used in SOTA verifiers, which is typically attributed to high computational overhead. This paper reveals a **more fundamental reason**: as network depth increases, the strict feasibility (Slater condition) of the SDP problem is lost, resulting in:

- Failure of strong duality $\to$ primal-dual gap does not converge to zero.
- Numerical instability of interior-point methods $\to$ solvers return erroneous results or fail directly.
- More severe consequences for first-order methods due to the lack of second-order Hessian information.

## Method

### Theoretical Analysis of Interior-Point Vanishing

**Definition**: Interior-point vanishing refers to the phenomenon where the SDP verification problem loses feasible interior points (no solution exists for $\boldsymbol{P} \succ \boldsymbol{O}$) as network depth increases.

**Reason 1: Inactive Neurons**. If the upper bound of a certain neuron is $(u_i)_j = 0$, then from the boundary constraint (5e) and the positive semidefinite constraint, it can be derived that $(\boldsymbol{P}[\boldsymbol{x}_i \boldsymbol{x}_i^\top])_{jj} = 0$. By Proposition 3.3, this implies $\lambda_{\min}(\boldsymbol{P}) = 0$, meaning $\boldsymbol{P}$ cannot be positive definite. A single inactive neuron is sufficient to trigger interior-point vanishing, and preprocessing cannot completely eliminate all inactive neurons.

**Reason 2: Weight Norm Constraints** (Theorem 3.6). Define $\widetilde{\boldsymbol{W}}_i = (\boldsymbol{b}_i \; \boldsymbol{W}_i)$, the recurrence constant $T_0 = (\|\bar{\boldsymbol{x}}\|_2 + \rho\sqrt{n_0})^2$, and $T_{i+1} = (1+T_i)\|\widetilde{\boldsymbol{W}}_i\|_F^2$. Then:

$$\lambda_{\min}(\boldsymbol{P}) \le \min_{i \in [L]} \min_{j} (1+T_i) \cdot \|\widetilde{\boldsymbol{W}}_i(j,:)\|_2^2$$

When there is a row vector with a small norm in any layer, the minimum eigenvalue of all feasible solutions is squeezed close to zero.

**Strict Feasibility Verification** (Proposition 3.2): Introduce an auxiliary variable $\lambda$, replace the original variable with $\boldsymbol{X} + \lambda\boldsymbol{I}$, and maximize $\lambda$; if the optimal value is $\lambda^* > 0$, the original problem is strictly feasible.

### Five Mitigation Methods

| Method | Core Idea | Modified Constraints |
|------|---------|-----------|
| **ε-SDP** | Relax ReLU equality constraints into inequalities with $\pm\varepsilon$ tolerance | (5d) $\to$ $\|...\| \le \varepsilon$ |
| **LeakySDP** | Replace ReLU constraints with Leaky ReLU ($\alpha \cdot x, x<0$) | (5b)(5c)(5d) $\to$ Leaky ReLU constraints |
| **D-Scale** | Diagonal scaling $\boldsymbol{D} = \text{diag}(1, \boldsymbol{u}_1, ..., \boldsymbol{u}_L)$ | Equivalently transform all constraint matrices |
| **W-Scale** | Scale weight matrices by minimum row norm based on Theorem 3.6 | $\widetilde{\boldsymbol{W}}_i' = \widetilde{\boldsymbol{W}}_i / \check{w}_i$ |
| **B-Remove** | Directly remove upper and lower bound constraints of intermediate layers ($l_i, u_i$ for $i > 0$) | Remove the $i > 0$ parts in (5e) |

**Key Insight**: The reason B-Remove works is that upper and lower bound constraints of intermediate layers are not necessary for SDP relaxation—SDP relaxation of ReLU itself does not require these bounds (see Eqs. (2)-(3)). These constraints originate from the "triangle region" of LP relaxation and have been adopted indiscriminately by prior work, proving to be not only unhelpful but actually harmful.

## Key Experimental Results

**Experimental Setup**: MNIST dataset, fully-connected ReLU networks, depth $L \in \{2,4,...,16\}$, 20 neurons per layer, 5 random seeds, 10 images. Baselines: SDP-IP, LayerSDP.

### Solver Success Rate (%)

| Depth L | ε-SDP | LeakySDP | B-Remove | D-Scale | W-Scale | LayerSDP | SDP-IP |
|-------|-------|----------|----------|---------|---------|----------|--------|
| 2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 6 | 100 | 98 | 100 | 100 | 100 | 100 | 100 |
| 8 | 100 | 82 | 100 | 88 | 88 | 82 | 98 |
| 10 | 100 | 4 | 100 | 8 | 8 | 2 | 14 |
| 12 | 78 | 0 | **100** | 0 | 0 | 0 | 0 |
| 16 | 24 | 0 | **66** | 0 | 0 | 0 | 0 |

### Empirical Evidence of Strict Feasibility (High-precision SDPA-GMP, 512-bit)

| Depth L | MNIST Solved(%) | Avg. Obj. | FashionMNIST Solved(%) | Avg. Obj. |
|-------|-----------------|-----------|------------------------|-----------|
| 2 | 98% | 2.13E-05 | 100% | 5.79E-05 |
| 8 | 98% | 3.52E-09 | 94% | 4.98E-09 |
| 10 | 18% | **-4.09E-10** | 26% | **-2.57E-10** |
| 16 | **0%** | -1.20E-09 | **0%** | -9.35E-10 |

For $L \ge 10$, the optimal value approaches zero or even becomes negative (numerical error), verifying the complete loss of strict feasibility.

### Relaxation Quality

B-Remove loses almost no verification capability (showing minimal gap in objective function values compared to LayerSDP), because the bound constraints of intermediate layers do not affect the quality of SDP relaxation in the first place. The verification capability loss of ε-SDP is relatively larger.

## Highlights & Insights

1. **First Identification of the Interior-Point Vanishing Problem**: Revealing the fundamental reason why SDP verification methods fail on deep networks, which is not solely due to computational overhead.
2. **Dual Verification via Theory and Empirical Evidence**: Rigorously proving the ubiquity of the problem through a minimum eigenvalue bound (Theorem 3.6) and high-precision solver experiments.
3. **Counter-intuitive Finding of "Useful Constraints Being Harmful"**: The upper and lower bound constraints for intermediate layers stem from the convention of LP relaxation; in SDP, they do not improve the relaxation quality, yet they destroy strict feasibility.
4. **Minimalistic Effectiveness of B-Remove**: Eliminating only non-essential constraints resolves 88% of the failure cases with almost no loss of verification accuracy, demonstrating that simpler methods can be more profound.

## Limitations & Future Work

1. **Limited Experimental Scale**: Validated only on small networks with 20 neurons/layer, whereas real-world verification scenarios involve larger networks (100–1000+ neurons/layer).
2. **Restricted to Fully-Connected ReLU Networks**: Convolutional networks, residual connections, and other activation functions are not covered.
3. **Scalability Bottleneck of SDP Solvers**: Even when interior-point vanishing is resolved, the $O(n^6)$ computational complexity of SDP remains a fundamental barrier to practical application.
4. **Lack of End-to-End Comparison with SOTA Verifiers**: Such as $\alpha,\beta$-CROWN + BaB; it remains unclear whether the tighter initial relaxation of SDP benefits the overall pipeline.
5. **Theoretical Guarantees for B-Remove**: Despite strong empirical performance, a theoretical lower-bound analysis regarding the loss of relaxation quality is missing.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to systematically identify and analyze the interior-point vanishing problem; the counter-intuitive findings are highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐ — Solid theoretical analysis but with a small experimental scale; lacks validation on larger networks.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, rigorous theoretical derivations, and intuitive charts.
- **Value**: ⭐⭐⭐⭐ — Makes significant theoretical contributions to the field of SDP verification, though practical utility may be limited by the inherent scalability issues of SDP itself.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming](sdp-crown_efficient_bound_propagation_for_neural_network_verification_with_tight.md)
- [\[CVPR 2025\] Convex Relaxation for Robust Vanishing Point Estimation in Manhattan World](../../CVPR2025/optimization/convex_relaxation_for_robust_vanishing_point_estimation_in_manhattan_world.md)
- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[ICML 2025\] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)

</div>

<!-- RELATED:END -->
