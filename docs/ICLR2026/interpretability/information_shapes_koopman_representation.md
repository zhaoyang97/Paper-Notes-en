---
title: >-
  [Paper Note] Information Shapes Koopman Representation
description: >-
  [ICLR 2026][koopman operator] This paper revisits the problem of finite-dimensional Koopman operator representation learning from the perspective of the Information Bottleneck (IB) framework. The Koopman operator lifts nonlinear dynamical systems into infinite-dimensional linear evolution, yet practical applications require approximation within finite-dimensional subspaces, giving rise to a fundamental tension between compactness and expressiveness. The authors prove that (1) latent mutual information controls an upper bound on prediction error, but excessive maximization leads to mode collapse; and (2) von Neumann entropy prevents collapse and preserves effective dimensionality. Building on these results, an information-theoretic Lagrangian formulation is proposed that jointly balances three objectives—temporal coherence, predictive sufficiency, and structural consistency—and yields a tractable loss function. The method outperforms existing Koopman approaches on three categories of tasks: physics simulation, visual control, and graph-structured dynamics.
tags:
  - ICLR 2026
  - koopman operator
  - information bottleneck
  - dynamical systems
  - representation learning
  - von neumann entropy
date: 2026-05-08
content_hash: 9aa2790657be6892
---

# Information Shapes Koopman Representation

**Conference**: ICLR 2026
**arXiv**: [2510.13025](https://arxiv.org/abs/2510.13025)
**Code**: [https://github.com/Wenxuan52/InformationKoopman](https://github.com/Wenxuan52/InformationKoopman)
**Area**: Interpretability
**Keywords**: koopman operator, information bottleneck, dynamical systems, representation learning, von neumann entropy

## TL;DR

This paper revisits the problem of finite-dimensional Koopman operator representation learning from the perspective of the Information Bottleneck (IB) framework. The Koopman operator lifts nonlinear dynamical systems into infinite-dimensional linear evolution, yet practical applications require approximation within finite-dimensional subspaces, giving rise to a fundamental tension between compactness and expressiveness. The authors prove that (1) latent mutual information controls an upper bound on prediction error, but excessive maximization leads to mode collapse; and (2) von Neumann entropy prevents collapse and preserves effective dimensionality. Building on these results, an information-theoretic Lagrangian formulation is proposed that jointly balances three objectives—temporal coherence, predictive sufficiency, and structural consistency—and yields a tractable loss function. The method outperforms existing Koopman approaches on three categories of tasks: physics simulation, visual control, and graph-structured dynamics.

## Background & Motivation

1. **The infinite-dimensional dilemma of the Koopman operator**: Koopman operator theory can in principle linearize nonlinear dynamics, but its infinite-dimensional nature makes it extremely difficult to identify a suitable finite-dimensional subspace within deep networks; existing methods frequently exhibit instability or mode collapse.
2. **Absence of general representation learning principles**: Prior work relies on domain-specific priors (symmetries, conservation laws, etc.) to constrain Koopman representations, but lacks general guiding principles for balancing compactness and expressiveness.
3. **Natural fit of the IB perspective**: The IB framework naturally captures the trade-off between compressing inputs and retaining predictive information, yet standard IB does not account for the linear evolution constraints inherent to dynamical systems.
4. **Stricter linearity constraints in latent space**: Unlike VAEs, Koopman learning requires the latent space not only to encode the current state but also to support linear forward propagation, imposing stronger structural constraints on the representation.
5. **Simply increasing dimensionality does not resolve the problem**: Prior studies show that naively enlarging the latent space dimension does not improve performance and may in fact disrupt temporal coherence.
6. **Error accumulates and amplifies in autoregressive prediction**: Small deviations in the Koopman representation propagate and amplify over time steps, motivating the need for theoretical tools to quantify and control this cumulative error.

## Method

### Information Flow Analysis

The authors first establish a probabilistic perspective on Koopman representations. Given an initial state $x_0$, the trajectory distribution induced by the Koopman representation is:

$$p^{KR}(x_{1:t}|x_0) = \int p(z_0|x_0) \prod_{n=1}^{t} p(z_n|z_{n-1}) p(x_n|z_n) dz_{0:t}$$

where the encoder $p(z_0|x_0)$ maps states to latent space, the linear Gaussian transition $p(z_n|z_{n-1}) = \mathcal{N}(z_n|\mathcal{K}z_{n-1}, \Sigma)$ implements Koopman evolution, and the decoder $p(x_n|z_n)$ reconstructs the state.

### Autoregressive Error Bound

The central theoretical contribution is a proof that the discrepancy between the true trajectory and the Koopman trajectory is controlled by the cumulative information loss at each step:

$$\|p(x_{1:t}|x_0) - q^{KR}(x_{1:t}|x_0)\|_{TV} \leq \sqrt{\frac{1}{2}\sum_{n=1}^{t}(I(x_{n-1};x_n) - I(z_{n-1};z_n)) + \mathcal{E}}$$

The information gap $I(x_{n-1};x_n) - I(z_{n-1};z_n)$ directly measures the dynamic coupling information lost by the Koopman approximation.

### Information Decomposition and Spectral Properties

The mutual information $I(z_t; x_t)$ is decomposed into three components:
- **Temporally coherent information** $I(z_{t-n}; z_t)$: corresponds to Koopman modes with eigenvalues $|\lambda| \approx 1$, representing information that can be preserved over long horizons.
- **Rapidly dissipating information** $I(z_t; x_{t-1}|z_{t-n})$: corresponds to modes with $|\lambda| < 1$, decaying exponentially over time.
- **Residual information** $I(z_t; x_t|x_{t-1})$: has no spectral counterpart and belongs to unpredictable components such as noise; it can be compressed.

### Information-Theoretic Lagrangian

Based on the above analysis, the unified optimization objective is:

$$\max_z \alpha \log I(z_{t-n};z_t) - \beta I(z_t;x_t|z_{t-n}) + \gamma S\left(\frac{\mathcal{C}}{\text{tr}(\mathcal{C})}\right) + \log p(x_t|z_t)$$

The $\alpha$ term preserves temporal coherence; the $\beta$ term compresses dissipating and residual components; the $\gamma$ term uses von Neumann entropy $S(\cdot)$ to prevent mode collapse and maintain effective dimensionality; and the final term is the reconstruction loss.

### Tractable Loss Function

The Lagrangian is converted into a practical loss: temporally coherent information is computed via closed-form mutual information or InfoNCE; structural consistency is implemented as $\mathbb{E}_{p_\theta(z_n|x_n)}[\log q_\psi(z_n|z_{n-1})]$, i.e., the likelihood under the linear Koopman transition; and von Neumann entropy is computed from the normalized covariance matrix over mini-batches. The overall framework is architecture-agnostic and compatible with both VAE and AE backbones.

## Key Experimental Results

### Table 1: Performance Comparison on Physics Simulation Tasks (NRMSE ↓ / SSIM ↑ / SDE ↓)

| Task | Metric | VAE | KAE | KKR | PFNN | **Ours** |
|------|------|-----|-----|-----|------|----------|
| Lorenz 63 | 5-NRMSE | 0.005 | 0.006 | 0.004 | 0.005 | **0.003** |
| Lorenz 63 | 50-NRMSE | 0.019 | 0.023 | 0.017 | 0.017 | **0.013** |
| Lorenz 63 | KLD | 1.047 | 0.464 | 0.342 | 0.293 | **0.285** |
| Kármán Vortex | 5-NRMSE | 0.127 | 0.149 | 0.114 | 0.075 | **0.068** |
| Kármán Vortex | 5-SSIM | 0.743 | 0.719 | 0.868 | 0.920 | **0.936** |
| Kármán Vortex | SDE | 0.538 | 0.620 | 0.799 | 0.278 | **0.256** |
| Dam Flow | 50-NRMSE | 0.034 | 0.046 | 0.031 | – | **0.026** |
| Dam Flow | SDE | 0.563 | 0.488 | 0.373 | – | **0.244** |
| ERA5 Weather | 5-NRMSE | – | 0.055 | 0.058 | 0.049 | **0.028** |
| ERA5 Weather | 5-SSIM | – | 0.666 | 0.664 | 0.697 | **0.867** |

### Table 2: Ablation Study — Effect of Each Regularization Term on Pendulum Manifold Learning

| Configuration | Temporal Coherence (α) | Structural Consistency (β) | Von Neumann Entropy (γ) | Manifold Quality |
|------|:-:|:-:|:-:|------|
| Full model | ✓ | ✓ | ✓ | Closest to ground-truth $\mathcal{S}^1 \times \mathbb{R}$ |
| α=0 | ✗ | ✓ | ✓ | Degenerates to scattered points; no geometric structure |
| β=0 | ✓ | ✗ | ✓ | Manifold collapses; dynamical structure lost |
| γ=0 | ✓ | ✓ | ✗ | Retains $\mathcal{S}^1$ but loses $\mathbb{R}$ dimension |
| α only increased | ↑↑ | ✓ | ✗ | Representation concentrates on $\mathcal{S}^1$ component |
| α + γ | ✓ | ✓ | ✓ | Full $\mathcal{S}^1 \times \mathbb{R}$ recovered |

## Highlights & Insights

- **Theoretical depth**: The paper is the first to establish an information-theoretic framework for Koopman representations, rigorously linking mutual information to autoregressive error bounds and spectral properties, and revealing the dual role of MI in promoting compactness (at the risk of mode collapse) and von Neumann entropy in maintaining expressiveness.
- **Insightful information decomposition**: Decomposing latent information into temporally coherent, rapidly dissipating, and residual components—each mapped to corresponding Koopman eigenvalues—provides a new analytical tool for understanding dynamical system representations.
- **Architecture-agnostic general framework**: The proposed Lagrangian is compatible with both VAE and AE architectures and demonstrates consistent effectiveness across physics simulation, visual control, and graph-structured dynamics.
- **Clear and compelling ablation study**: Manifold visualizations on the Pendulum task clearly illustrate the role of each regularization term, with theoretical predictions and experimental observations in close agreement.

## Limitations & Future Work

- **Computational overhead insufficiently discussed**: Computing von Neumann entropy requires eigendecomposition of a covariance matrix, which may become a bottleneck in high-dimensional latent spaces.
- **Hyperparameter selection relies on experience**: The choice of Lagrangian multipliers $\alpha, \beta, \gamma$ significantly affects performance, yet the paper does not provide systematic selection guidelines.
- **Relatively limited experimental scale**: Physics simulation tasks are of moderate dimensionality (at most 64×64×2); validation on larger-scale or more complex real-world systems is absent.
- **Limitations of the linear Koopman assumption**: The framework fundamentally assumes linear latent evolution; the boundary of applicability to strongly nonlinear or chaotic systems (e.g., turbulence) is not thoroughly analyzed.
- **No comparison with modern foundation models**: The method is not benchmarked against Transformer-based temporal prediction approaches such as FourCastNet.

## Related Work & Insights

- **Koopman operator learning**: KAE (Pan et al., 2023) is a representative Koopman autoencoder; KKR (Bevanda et al., 2023) is kernel-based; PFNN (Cheng et al., 2025) introduces a Poincaré flow structure for chaotic systems. This paper unifies and surpasses these methods from an information-theoretic standpoint.
- **Information Bottleneck methods**: Tishby et al. (2000) proposed the classical IB framework; β-VAE (Burgess et al., 2018) extended IB to variational autoencoders. This paper extends IB to sequential Koopman representations in dynamical systems.
- **Representation learning for dynamical systems**: E2C (Banijamali et al., 2019) and PCC (Levine et al., 2020) learn controllable representations from a VAE perspective. This paper achieves superior manifold structure through information-theoretic constraints.
- **Spectral analysis and effective dimensionality**: Von Neumann entropy is used in quantum information to measure entanglement. This paper innovatively applies it to Koopman representations to prevent mode collapse.

## Rating

| Dimension | Score (1–10) |
|------|:-----------:|
| Novelty | 8 |
| Theoretical Depth | 9 |
| Experimental Thoroughness | 7 |
| Writing Quality | 8 |
| Value | 7 |
| **Overall** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)

</div>

<!-- RELATED:END -->
