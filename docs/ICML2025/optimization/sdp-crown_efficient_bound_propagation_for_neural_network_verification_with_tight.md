---
title: >-
  [Paper Note] SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming
description: >-
  [ICML2025][Optimization][Neural Network Verification] Proposes SDP-CROWN, which integrates the tightness of Semidefinite Programming (SDP) relaxation into the linear bound propagation framework. By adding only one parameter $\lambda$ per layer, it tightens the verification relaxation by up to a factor of $\sqrt{n}$ under $\ell_2$ perturbations while maintaining scalability comparable to $\alpha$-CROWN.
tags:
  - "ICML2025"
  - "Optimization"
  - "Neural Network Verification"
  - "Semidefinite Programming"
  - "Linear Bound Propagation"
  - "ℓ₂ Robustness"
  - "ReLU Relaxation"
  - "CROWN"
date: 2026-05-08
content_hash: 3de2f22603a53925
---

# SDP-CROWN: Efficient Bound Propagation for Neural Network Verification with Tightness of Semidefinite Programming

**Conference**: ICML2025  
**arXiv**: [2506.06665](https://arxiv.org/abs/2506.06665)  
**Code**: [Hong-Ming/SDP-CROWN](https://github.com/Hong-Ming/SDP-CROWN)  
**Area**: Optimization  
**Keywords**: Neural Network Verification, Semidefinite Programming, Linear Bound Propagation, ℓ₂ Robustness, ReLU Relaxation, CROWN

## TL;DR

Proposes SDP-CROWN, which integrates the tightness of Semidefinite Programming (SDP) relaxation into the linear bound propagation framework. By adding only one parameter $\lambda$ per layer, it tightens the verification relaxation by up to a factor of $\sqrt{n}$ under $\ell_2$ perturbations while maintaining scalability comparable to $\alpha$-CROWN.

## Background & Motivation

- **Linear bound propagation** (CROWN / $\alpha$-CROWN / $\beta$-CROWN) represents the most scalable neural network verification methods today, performing excellently in VNN-COMP competitions, particularly for $\ell_\infty$ perturbations.
- **Limitations of Prior Work**: When dealing with $\ell_2$ perturbations, bound propagation methods must first relax the $\ell_2$ ball into a bounding $\ell_\infty$ box, which scales the perturbation radius by a factor of $\sqrt{n}$ (where $n$ is the input dimension), leading to extremely loose bounds. For example, on ConvLarge ($\approx 2.47$M parameters, $65$k neurons), $\alpha$-CROWN only achieves an $\ell_2$ verified accuracy of 2.5%.
- **SDP methods** can precisely capture dependencies between neurons via $n \times n$ coupling matrices, but their $O(n^3)$ complexity restricts them to small-scale models ($<10$k neurons).
- **Goal**: To inject the tightness of SDP into the bound propagation framework without sacrificing scalability.

## Method

### Mechanism

Standard bound propagation constructs a linear relaxation $g(\alpha)^T x + h(\alpha)$ for the objective function $c^T f(x)$, where $h(\alpha) = -\rho \|\min\{g(\alpha), 0\}\|_1$ (based on the $\ell_\infty$ box). SDP-CROWN replaces the offset with a new expression $h(g,\lambda)$ based on the $\ell_2$ ball, while keeping the slope $g(\alpha)$ unchanged.

### Theorem 4.1

For any $\lambda \ge 0$ and $g \in \mathbb{R}^n$:

$$c^T \operatorname{ReLU}(x) \ge g^T x + h(g,\lambda), \quad \forall\, x \in \mathcal{B}_2(\hat{x}, \rho)$$

where:

$$h(g,\lambda) = -\frac{1}{2}\left(\lambda(\rho^2 - \|\hat{x}\|_2^2) + \frac{1}{\lambda}\|\phi(g,\lambda)\|_2^2\right)$$

$$\phi_i(g,\lambda) = \min\{c_i - g_i - \lambda\hat{x}_i,\; g_i + \lambda\hat{x}_i,\; 0\}$$

### Tightness Guarantee (Theorem 5.2)

When $\hat{x}=0$, the optimal offset under the best $\lambda$ is:

$$h^* = -\rho \|\min\{c-g, g, 0\}\|_2$$

Compared to the offset $h = -\rho \|\min\{g, 0\}\|_1$ of traditional bound propagation, the norm type changes from $\ell_1$ to $\ell_2$, directly achieving up to a **$\sqrt{n}$ times** tightness improvement. Furthermore, the SDP relaxation itself is tight (Theorem 5.3), meaning the SDP dual value matches the optimal value of the original non-convex problem.

### Implementation Details

1. First, use standard $\alpha$-CROWN to compute the slope $g(\alpha)$ and the initial offset $h(\alpha)$ on the bounding $\ell_\infty$ box.
2. Replace the offset with $h(g(\alpha), \lambda)$ using the formula in Theorem 4.1.
3. Perform jointly projected gradient ascent on both $\alpha$ and $\lambda$ to find the tightest bound.
4. **Only adds one scalar parameter $\lambda$ per layer**, significantly reducing the overhead compared to the $n^2$ parameters in standard SDP.

## Key Experimental Results

Comparison of multiple verifiers on MNIST and CIFAR-10, displaying the **$\ell_2$ verified accuracy** (%) over 200 images:

| Model | Parameter | SDP-CROWN (Ours) | α-CROWN | β-CROWN | GCP-CROWN | BICCOS | LipNaive | Upper Bound |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MNIST MLP | ρ=1.0 | **32.5%** (2.5s) | 1.5% (1.2s) | 36% (302s) | 38% (198s) | 41% (173s) | 29% | 54% |
| MNIST ConvSmall | ρ=0.3 | **81.5%** (12s) | 0% (17s) | 16% (257s) | 17% (265s) | 19.5% (248s) | 77.5% | 84.5% |
| MNIST ConvLarge | ρ=0.3 | **79.5%** (88s) | 0% (66s) | 0% (307s) | 0% (304s) | 0% (309s) | 77% | 84% |
| CIFAR ConvLarge | ρ=8/255 | **63.5%** (73s) | 2.5% (47s) | 5% (289s) | 6% (282s) | 6% (286s) | 47.5% | 72.5% |
| CIFAR CNN-A | ρ=24/255 | **49%** (12s) | 7.5% (3.8s) | 20% (201s) | 20% (224s) | 20% (210s) | 39% | 55.5% |
| CIFAR CNN-B | ρ=24/255 | **49.5%** (16s) | 0% (8.7s) | 3% (193s) | 3% (302s) | 3% (290s) | 33% | 59.5% |

**Key Findings**:

- On ConvLarge ($65$k neurons), standard $\alpha$-CROWN / $\beta$-CROWN / BICCOS fail completely with nearly 0% verified accuracy, whereas SDP-CROWN achieves 63.5%–79.5%.
- SDP-CROWN exceeds the LipNaive baseline by 2%–16% in verified accuracy and significantly outperforms branch-and-bound-based methods.
- Traditional SDP methods (e.g., BM-Full/LP-All) cannot run on medium-to-large-scale models (marked as "-"), while SDP-CROWN successfully scales up to 2.47M parameters.
- Moderate running times: 73–88s per sample on ConvLarge, which is substantially lower than BICCOS's 173–309s.

## Highlights & Insights

1. **Theoretical Elegance**: Derives a closed-form linear bound with only one extra parameter $\lambda$ from SDP duality, proving it is exact and tight when $\hat{x}=0$.
2. **Guaranteed $\sqrt{n}$-fold Improvement**: Replaces the $\ell_1$-norm bound with an $\ell_2$-norm bound, precisely reflecting the volume ratio between the $\ell_2$ ball and the $\ell_\infty$ box.
3. **Plug-and-play**: Seamlessly integrates into any linear bound propagation pipeline (like CROWN, $\alpha$-CROWN), requiring no architectural modifications.
4. **First to scale SDP relaxations to efficient $\ell_2$ verification**, filling a crucial gap in large-scale $\ell_2$ robustness certification.
5. **Impressive Contrast**: Under $\ell_2$ perturbations, $\beta$-CROWN and BICCOS obtain virtually 0% verified accuracy even when utilizing complex branch-and-bound and cutting-plane approaches, whereas SDP-CROWN reaches 60%+ verified accuracy with just a single extra parameter.
6. **Insight from the LipNaive Baseline**: The fact that a simple layer-by-layer product of Lipschitz constants outperforms complex bound propagation methods under $\ell_2$ perturbations highlights the critical importance of modeling neuron coupling in $\ell_2$ scenarios.

## Limitations & Future Work

1. **Theoretical analysis is limited to $\hat{x}=0$**: Tightness in more general cases where $\hat{x} \neq 0$ relies on experimental validation and lacks analytical guarantees.
2. **Restricted to ReLU activation functions**: Extensions to non-piecewise linear activations like GeLU and Swish remain unexplored.
3. **Layer-wise scalar $\lambda$**: Utilizing a finer-grained per-neuron $\lambda$ might further enhance performance, though at the cost of increased parameter count.
4. **No performance gain for $\ell_\infty$ perturbations**: Designed specifically for $\ell_2$ perturbations, reverting to standard bound propagation under $\ell_\infty$ bounds.
5. **Under-explored integration with Branch-and-Bound**: Joint optimization of splits (from $\beta$-CROWN) and $\lambda$ (from SDP-CROWN) is a compelling avenue for research.
6. **Lack of large-scale vision model evaluation**: The largest model in experiments has only 2.47M parameters, lacking tests on modern architectures like ResNet or ViTs.
7. **Co-design with adversarial training is not discussed**: Whether the tight bounds of SDP-CROWN can be used as a training objective for $\ell_2$ adversarial training remains to be investigated.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — An entirely fresh perspective on integrating SDP with bound propagation, featuring elegant closed-form derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Encompasses multiple architectures and datasets, with comprehensive comparisons against mainstream verifiers, though lacking verification on larger scales (e.g., ImageNet).
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear theoretical derivations, with Figures 1 & 2 providing intuitive visualizations of the core ideas.
- Value: ⭐⭐⭐⭐⭐ — High practical utility, solving an important bottleneck in $\ell_2$ verification.

## Related Work & Insights

- **SDP Verification**: Raghunathan et al. 2018, Brown et al. 2022, Chiu & Zhang 2023 — Tight but suffers from $O(n^3)$ non-scalability.
- **Bound Propagation**: CROWN (Zhang 2018), $\alpha$-CROWN (Xu 2021), $\beta$-CROWN (Wang 2021) — Scalable but heavily relaxed under $\ell_2$ perturbations.
- **Lipschitz Methods**: LipSDP (Fazlyab 2019), 1-Lipschitz networks (Singla & Feizi 2022) — Too conservative due to global constants.
- **Cutting Planes**: GCP-CROWN (Zhang 2022), BICCOS (Zhou 2024) — Incorporates MIP constraints but still fails under large $\ell_2$ perturbations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Interior-Point Vanishing Problem in Semidefinite Relaxations for Neural Network Verification](interior-point_vanishing_problem_in_semidefinite_relaxations_for_neural_network_.md)
- [\[ICML 2025\] The Butterfly Effect: Neural Network Training Trajectories Are Highly Sensitive to Initial Conditions](the_butterfly_effect_neural_network_training_trajectories_are_highly_sensitive_t.md)
- [\[ICML 2025\] Integer Programming for Generalized Causal Bootstrap Designs](integer_programming_for_generalized_causal_bootstrap_designs.md)
- [\[CVPR 2025\] Automatic Joint Structured Pruning and Quantization for Efficient Neural Network Training and Compression](../../CVPR2025/optimization/automatic_joint_structured_pruning_and_quantization_for_efficient_neural_network.md)
- [\[ICML 2025\] Widening the Network Mitigates the Impact of Data Heterogeneity on FedAvg](widening_the_network_mitigates_the_impact_of_data_heterogeneity_on_fedavg.md)

</div>

<!-- RELATED:END -->
