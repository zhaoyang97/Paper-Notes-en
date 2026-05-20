---
title: >-
  [Paper Note] Generalization Below the Edge of Stability: The Role of Data Geometry
description: >-
  [ICLR 2026][Optimization][Generalization Theory] This paper introduces the principle of *data shatterability* to provide a unified explanation of how data geometry governs the strength of implicit regularization induced…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Generalization Theory"
  - "Edge of Stability"
  - "Data Geometry"
  - "ReLU Networks"
  - "Implicit Regularization"
date: 2026-05-08
content_hash: d8debac154847572
---

# Generalization Below the Edge of Stability: The Role of Data Geometry

**Conference**: ICLR 2026
**arXiv**: [2510.18120](https://arxiv.org/abs/2510.18120)  
**Code**: None  
**Area**: Learning Theory / Optimization
**Keywords**: Generalization Theory, Edge of Stability, Data Geometry, ReLU Networks, Implicit Regularization

## TL;DR

This paper introduces the principle of *data shatterability* to provide a unified explanation of how data geometry governs the strength of implicit regularization induced by gradient descent near the Edge of Stability (EoS). For the Beta(α) radial distribution family, the authors derive a spectrum of generalization upper and lower bounds that depend on α. For mixture distributions supported on low-dimensional subspaces, they prove that the generalization rate adapts to the intrinsic dimension $m$ rather than the ambient dimension $d$.

## Background & Motivation

**Background**: Overparameterized neural networks generalize well even without explicit regularization (e.g., weight decay), a phenomenon that classical statistical learning theory fails to explain. The discovery of the Edge of Stability (EoS)—where large-step GD training drives the Hessian's largest eigenvalue to $\lambda_{\max}(\nabla^2\mathcal{L}) \approx 2/\eta$—offers a new lens for understanding implicit regularization.

**Limitations of Prior Work**:

1. Existing work has shown that the EoS condition is equivalent to a data-dependent weighted path norm constraint, but generalization bounds derived for the uniform spherical distribution suffer from the curse of dimensionality—contradicting the empirical success of deep learning.
2. No unified theoretical framework exists to determine which data geometries lead to generalization and which lead to memorization.
3. Existing generalization bounds are distribution-agnostic and cannot distinguish the effects of different data geometries.

**Key Challenge**: The data-dependent regularization induced by EoS varies dramatically across distributions—networks trained on data lying on a sphere can memorize without penalty, while data inside the ball is subject to strong regularization constraints. A unifying principle is needed to explain this discrepancy.

**Goal**: The paper introduces the concept of *data shatterability*—the difficulty of shattering a data distribution using ReLU half-spaces—as the core geometric quantity governing generalization behavior.

## Method

### Overall Architecture

The theoretical analysis is built on the BEoS (Below Edge of Stability) condition for two-layer ReLU networks:

$$f_{\boldsymbol{\theta}}(\boldsymbol{x}) = \sum_{k=1}^K v_k \phi(\boldsymbol{w}_k^\top \boldsymbol{x} - b_k) + \beta, \quad \phi(z) = \max\{z, 0\}$$

The BEoS condition $\lambda_{\max}(\nabla^2_{\boldsymbol{\theta}}\mathcal{L}) \leq 2/\eta$ is equivalent to an upper bound on the data-dependent weighted path norm. The core technical pipeline is:

**Half-space depth stratification → Good/bad region decomposition → Generalization upper bound → Instantiation with data geometry**

### Key Design 1: Half-Space Depth for Partition Quantification

The paper introduces the Tukey half-space depth to stratify the input space:

$$\text{depth}(\boldsymbol{x}, \mathcal{P}_X) = \inf_{\boldsymbol{u} \in \mathbb{S}^{d-1}} \mathbb{P}(\boldsymbol{u}^\top(\boldsymbol{X} - \boldsymbol{x}) \geq 0)$$

For the $T$-deep region $\Omega_T$, any ReLU activation boundary passing through this region must retain at least a $T$-fraction of data on each side, giving a positive lower bound on the weight function $g(\boldsymbol{u}, t)$. This effectively controls the (unweighted) path norm of neurons in this region by $O(1/g_{\min}(T))$.

This yields the following key generalization decomposition:

$$\sup_{\boldsymbol{\theta} \in \Theta_{\text{BEoS}}} \text{Gap}(f_{\boldsymbol{\theta}}, \mathcal{D}) \leq \underbrace{\tilde{O}(\mathbb{P}(\boldsymbol{X} \notin \Omega_T))}_{\text{shallow region}} + \underbrace{\tilde{O}(g_{\min}(T)^{-d/(2d+3)} n^{-(d+3)/(4d+6)})}_{\text{T-deep region}}$$

### Key Design 2: Generalization Spectrum for Isotropic Beta(α) Radial Distributions

Define $\boldsymbol{X} = h(R)\boldsymbol{U}$, where $h(r) = 1 - (1-r)^{1/\alpha}$, $R \sim \text{Uniform}[0,1]$, $\boldsymbol{U} \sim \text{Uniform}(\mathbb{S}^{d-1})$.

- **Large $\alpha$** → mass concentrated near the origin → small shallow-region probability → good generalization.
- **Small $\alpha$** → mass concentrated near the sphere → many disjoint spherical caps can be packed → easy memorization.
- **$\alpha \to 0$** (spherical limit) → a network of width $\leq n$ can perfectly interpolate under the BEoS condition with $\lambda_{\max} \leq 1 + (D^2+2)/n$.

Both the **upper bound** (Theorem 3.4) and the **lower bound** (Theorem 3.5) depend on $\alpha$, with rates $n^{-\alpha(d+3)/(2(d^2+4\alpha d+3\alpha))}$ and $n^{-2\alpha/(d-1+2\alpha)}$, respectively.

### Key Design 3: Adaptation to Low-Dimensional Structure

For a mixture distribution $\mathcal{P}_X = \sum_{j=1}^J \pi_j \mathcal{P}_{X,j}$, where each component is a uniform ball distribution on an $m$-dimensional affine subspace ($m < d$), the paper proves the generalization rate:

$$\text{Gap} \lessapprox_d \left(\frac{1}{\eta} - \frac{1}{2} + 4M\right)^{\frac{m}{m^2+4m+3}} M^2 J^{4/m} n^{-1/(2m+4)}$$

The core mechanism: when the network is restricted to subspace $V_j$, neuron activations are determined solely by $\text{proj}_{V_j} \boldsymbol{w}_k$, so high-dimensional hyperplanes degenerate into low-dimensional "knots," dramatically reducing shatterability.

## Key Experimental Results

### Main Results: Generalization Rate Verification on Isotropic Distributions

In $d=5$ dimensional space, two-layer ReLU networks of width 1000 are trained for 20,000 epochs with learning rate 0.4 on Beta(α) radial distributions for $\alpha \in \{0.1, 0.3, 1.5, 5.0\}$.

| Distribution parameter $\alpha$ | Log-log slope (observed) | Theoretical trend |
|:---:|:---:|:---:|
| 0.1 | ≈ −0.05 (nearly no generalization) | Mass near sphere → memorization |
| 0.3 | ≈ −0.12 | Weak generalization |
| 1.5 | ≈ −0.25 | Moderate generalization |
| 5.0 | ≈ −0.38 (steepest) | Mass near origin → strong generalization |

Larger $\alpha$ yields a steeper log-log slope, indicating faster generalization, consistent with theory.

### Ablation Study: Intrinsic Dimension Adaptation

20 one-dimensional lines embedded in $\mathbb{R}^d$ for $d \in \{10, 50, 100, 500\}$:

| Ambient dimension $d$ | Log-log slope | Variation |
|:---:|:---:|:---:|
| 10 | ≈ −0.22 | Baseline |
| 50 | ≈ −0.21 | +0.01 |
| 100 | ≈ −0.21 | +0.01 |
| 500 | ≈ −0.20 | +0.02 |

The slope remains nearly constant, confirming that generalization adapts to the intrinsic dimension ($m=1$) and is unaffected by the ambient dimension. As a control, the uniform ball distribution ($\alpha=1$) exhibits significantly degraded generalization as $d$ increases.

### Validation on MNIST

| Data type | Clean MSE after 20,000 steps | Behavior |
|:---:|:---:|:---:|
| $\mathcal{N}(0, I_{784})$ | ≈ 1.0 (noise level) | Rapid memorization |
| MNIST images | ≈ 0.2 | Resists overfitting for 10,000+ steps |

Gaussian data concentrates on a thin spherical shell (high shatterability) → rapid memorization; MNIST approximately lies on a low-dimensional structure → resists overfitting. Shallower MNIST points exhibit larger prediction errors, consistent with theoretical predictions.

## Highlights & Insights

### Strengths

1. **Theoretical depth**: The paper unifies EoS implicit regularization, data geometry, and generalization within a single "shatterability" framework, providing both upper and matching lower bounds.
2. **Breakthrough insight**: It explains why real data (low-dimensional manifolds) is harder to overfit than random Gaussian data—a theoretical response to Zhang et al. (2017) "Rethinking Generalization."
3. **Technical innovation**: Half-space depth stratification avoids the explosion of global metric entropy, breaking through the bottleneck of distribution-agnostic bounds.
4. **Practical implications**: The framework provides theoretical justification for Mixup data augmentation and activation-frequency-based pruning.

### Limitations & Future Work

1. The analysis is restricted to two-layer ReLU networks; extension to deeper networks faces theoretical challenges regarding the propagation of EoS regularization.
2. The depth-quantization concentration exponent $\mathsf{S}_{\text{DQ}}$ admits precise characterization only for isotropic distributions; quantifying shatterability for non-isotropic data remains heuristic.
3. Experiments are limited to simple synthetic data and MNIST; the predictive power of the theory on more complex datasets such as CIFAR/ImageNet has not been validated.

## Rating

⭐⭐⭐⭐⭐

This is a theoretically rigorous and broadly applicable work that, for the first time, establishes a quantitative connection between EoS implicit regularization and data geometry. The concept of "data shatterability" elegantly unifies previously disparate empirical observations—real data is harder to overfit than random data, low-dimensional data generalizes better, data on a sphere is easily memorized—and provides a solid theoretical foundation for understanding why deep learning can escape the curse of dimensionality in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](../../NeurIPS2025/optimization/a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](../../NeurIPS2025/optimization/unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[ICLR 2026\] FedDAG: Clustered Federated Learning via Global Data and Gradient Integration for Heterogeneous Environments](feddag_clustered_federated_learning_via_global_data_and_gradient_integration_for.md)
- [\[CVPR 2026\] The Power of Decaying Steps: Enhancing Attack Stability and Transferability for Sign-based Optimizers](../../CVPR2026/optimization/the_power_of_decaying_steps_enhancing_attack_stability_and_transferability_for_s.md)
- [\[NeurIPS 2025\] Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry](../../NeurIPS2025/optimization/projecting_assumptions_the_duality_between_sparse_autoencoders_and_concept_geome.md)

</div>

<!-- RELATED:END -->
