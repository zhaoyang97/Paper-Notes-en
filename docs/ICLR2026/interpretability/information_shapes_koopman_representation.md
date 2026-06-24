---
title: >-
  [Paper Note] Information Shapes Koopman Representation
description: >-
  [ICLR 2026][Interpretability][koopman operator] This paper re-examines the finite-dimensional representation learning problem of the Koopman operator from the perspective of the Information Bottleneck (IB). The Koopman operator lifts nonlinear dynamical systems into infinite-dimensional linear evolutions, but practical applications require approximation in finite-dimensional subspaces, leading to a fundamental contradiction between "simplicity and expressivity." The authors p…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "koopman operator"
  - "information bottleneck"
  - "dynamical systems"
  - "representation learning"
  - "von neumann entropy"
date: 2026-05-08
content_hash: 136617d9c1d5e94f
---

# Information Shapes Koopman Representation

**Conference**: ICLR 2026  
**arXiv**: [2510.13025](https://arxiv.org/abs/2510.13025)  
**Code**: [https://github.com/Wenxuan52/InformationKoopman](https://github.com/Wenxuan52/InformationKoopman)  
**Area**: Interpretability  
**Keywords**: koopman operator, information bottleneck, dynamical systems, representation learning, von neumann entropy

## TL;DR

This paper re-examines the finite-dimensional representation learning problem of the Koopman operator from the perspective of the Information Bottleneck (IB). The Koopman operator lifts nonlinear dynamical systems into infinite-dimensional linear evolutions, but practical applications require approximation in finite-dimensional subspaces, leading to a fundamental contradiction between "simplicity and expressivity." The authors prove that: (1) latent mutual information controls the upper bound of prediction error, but excessive maximization leads to mode collapse; (2) von Neumann entropy prevents collapse and maintains effective dimensionality. Based on this, an information-theoretic Lagrangian formulation is proposed to unify the balancing of three major objectives: temporal coherence, predictive sufficiency, and structural consistency, deriving computable loss functions. The method outperforms existing Koopman approaches in physical simulation, visual control, and graph-structured dynamics tasks.

## Background & Motivation

1. **Infinite-dimensional dilemma of Koopman operator**: Theoretically, the Koopman operator can linearize nonlinear dynamics, but its infinite-dimensional nature makes finding suitable finite-dimensional subspaces in deep networks extremely difficult. Existing methods frequently suffer from instability or mode collapse.
2. **Lack of general representation learning principles**: Prior work relies on domain priors (symmetries, conservation laws, etc.) to constrain Koopman representations, but lacks general guiding principles to balance simplicity and expressivity.
3. **Natural fit of the Information Bottleneck perspective**: The IB framework is inherently suited for describing the tradeoff between "compressing input while retaining predictive information," but standard IB does not consider the linear evolution constraints of dynamical systems.
4. **Stricter linear constraints in latent space**: Unlike VAEs, Koopman learning requires the latent space to not only encode the current state but also support linear forward propagation, imposing stronger structural constraints on the representation.
5. **Simple dimensionality increase does not solve the problem**: Previous research indicates that blindly increasing the latent space dimension does not improve performance and may instead damage temporal coherence.
6. **Error accumulation in autoregressive prediction**: Small deviations in Koopman representations propagate and amplify over time steps, necessitating theoretical tools to quantify and control this cumulative error.

## Method

### Overall Architecture

The authors place Koopman representation learning on the scale of the Information Bottleneck: an encoder compresses the state into a finite-dimensional latent space, a linear Koopman operator performs forward evolution in the latent space, and a decoder reconstructs the state. The quality of the entire chain depends on "how much information of the original dynamics is preserved by the latent dynamics." Centered on this core quantity, the paper follows a progressive derivation chain—first using probabilistic trajectory distributions to characterize the Koopman representation and derive an information-theoretic upper bound for prediction error, then decomposing latent mutual information into "retained vs. compressed" components based on the Koopman spectral structure, and finally distilling this into an information-theoretic Lagrangian that balances temporal coherence, predictive sufficiency, and structural consistency, implemented as an architecture-agnostic trainable loss.

### Key Designs

**1. Probabilistic Trajectory Distribution and Autoregressive Error Bound: Translating "prediction accuracy" into "latent mutual information sufficiency"**

To discuss "lost information," a probabilistic object is required. The authors write the trajectory distribution induced by the Koopman representation as $p^{KR}(x_{1:t}|x_0) = \int p(z_0|x_0) \prod_{n=1}^{t} p(z_n|z_{n-1}) p(x_n|z_n) dz_{0:t}$, where the encoder $p(z_0|x_0)$ maps states to latent space, linear Gaussian transitions $p(z_n|z_{n-1}) = \mathcal{N}(z_n|\mathcal{K}z_{n-1}, \Sigma)$ complete the Koopman evolution, and the decoder $p(x_n|z_n)$ reconstructs the state. This form allows "encoding-linear evolution-decoding" to be represented as a chain of comparable random variables.

On this basis, the authors use Total Variation distance to pin the prediction drift—which amplifies over time steps—to an upper bound:

$$\|p(x_{1:t}|x_0) - q^{KR}(x_{1:t}|x_0)\|_{TV} \leq \sqrt{\frac{1}{2}\sum_{n=1}^{t}\big(I(x_{n-1};x_n) - I(z_{n-1};z_n)\big) + \mathcal{E}}$$

In this equation, the step-wise information gap $I(x_{n-1};x_n) - I(z_{n-1};z_n)$ measures the loss of dynamic coupling information in the latent linear transition compared to the real state transition. This step translates abstract "prediction accuracy" into "sufficient latent mutual information" and explains why maximizing mutual information directly tightens the prediction error—it serves as the theoretical pivot for the objective function.

**2. Information Decomposition and Spectral Correspondence: Distinguishing what to retain and what to compress**

Blindly increasing mutual information forces representations to collapse into a few modes. Thus, the authors decompose latent mutual information $I(z_t; x_t)$ into three parts based on the Koopman eigenvalue $\lambda$ structure: Temporal coherent information $I(z_{t-n}; z_t)$ corresponds to $|\lambda|\approx 1$ modes, which are long-term persistent and worth retaining; fast dissipating information $I(z_t; x_{t-1}|z_{t-n})$ corresponds to $|\lambda|<1$ modes, decaying exponentially; residual information $I(z_t; x_t|x_{t-1})$ has no spectral correspondence and belongs to unpredictable components like noise, which can be safely compressed. This decomposition refines the general "compression vs. expressivity" into differentiated treatment of different spectral components.

**3. Information-Theoretic Lagrangian and Computable Loss: Integrating three types of information into a trainable objective**

Based on the decomposition, the paper proposes a unified optimization objective:

$$\max_z\ \alpha \log I(z_{t-n};z_t) - \beta\, I(z_t;x_t|z_{t-n}) + \gamma\, S\!\left(\frac{\mathcal{C}}{\text{tr}(\mathcal{C})}\right) + \log p(x_t|z_t)$$

Where the $\alpha$ term rewards temporal coherence to preserve long-term predictable modes; the $\beta$ term compresses dissipative and residual components for simplicity; the $\gamma$ term uses von Neumann entropy $S(\cdot)$ over the normalized covariance matrix $\mathcal{C}/\text{tr}(\mathcal{C})$ to resist mode collapse and maintain effective dimensionality; the last term is the reconstruction loss. The introduction of von Neumann entropy is critical: when the representation collapses into a few directions, the entropy drops sharply, pushing gradients back toward "spreading dimensions," forming a dual balance with MI maximization. All terms are implemented in computable forms independent of network architecture, ensuring the framework's universality.

## Key Experimental Results

### Main Results: Physical Simulation Task Performance Comparison (NRMSE ↓ / SSIM ↑ / SDE ↓)

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

### Ablation Study: Impact of Regularization Terms on Pendulum Manifold Learning

| Configuration | Temporal Coherence (α) | Structural Consistency (β) | von Neumann Entropy (γ) | Manifold Quality |
|------|:-:|:-:|:-:|------|
| Full Model | ✓ | ✓ | ✓ | Closest to ground truth $\mathcal{S}^1 \times \mathbb{R}$ |
| α=0 | ✗ | ✓ | ✓ | Degenerated into points, no geometric structure |
| β=0 | ✓ | ✗ | ✓ | Manifold collapse, loss of dynamical structure |
| γ=0 | ✓ | ✓ | ✗ | Retains $\mathcal{S}^1$ but loses $\mathbb{R}$ dimension |
| Increasing α | ↑↑ | ✓ | ✗ | Representation concentrated on $\mathcal{S}^1$ component |
| α + γ | ✓ | ✓ | ✓ | Recovers full $\mathcal{S}^1 \times \mathbb{R}$ |

## Highlights & Insights

- **Significant theoretical depth**: Establishes the first information-theoretic framework for Koopman representations, strictly linking mutual information with autoregressive error bounds and spectral properties, revealing the dual relationship where MI promotes simplicity while von Neumann entropy maintains expressivity.
- **Insightful information decomposition**: Decomposes latent information into temporal coherent/fast dissipating/residual components and maps them to Koopman eigenvalues, providing a new analytical tool for understanding dynamical system representations.
- **Architecture-agnostic framework**: The proposed Lagrangian is compatible with both VAE and AE structures and proves effective across physical simulation, visual control, and graph-structured dynamics.
- **Intuitive ablation experiments**: Clear visualization of the Pendulum manifold demonstrates the specific role of each regularization term, showing perfect alignment between theoretical prediction and experimental observation.

## Limitations & Future Work

- **Computational overhead not fully discussed**: Calculating von Neumann entropy requires eigen-decomposition of the covariance matrix, which may become a bottleneck in high-dimensional latent spaces.
- **Hyperparameter tuning relies on experience**: Choice of Lagrangian multipliers $\alpha, \beta, \gamma$ has a significant impact on performance, but the paper lacks a systematic guide for selection.
- **Relatively limited experimental scale**: Physical simulation tasks have moderate dimensionality (max 64×64×2); the approach has not been validated on larger or more complex real-world systems.
- **Limitations of the linear Koopman assumption**: The framework fundamentally assumes linear latent evolution, leaving the boundary of applicability for strongly nonlinear or chaotic systems (like turbulence) unanalyzed.
- **Lack of comparison with modern foundation models**: Comparison with Transformer-based time-series prediction methods (e.g., FourCastNet) is missing.

## Related Work & Insights

- **Koopman Operator Learning**: KAE (Pan et al., 2023) is a classic Koopman autoencoder; KKR (Bevanda et al., 2023) is based on kernel methods; PFNN (Cheng et al., 2025) designs Poincaré flow structures for chaotic systems. This work unifies and surpasses these methods from an information-theoretic perspective.
- **Information Bottleneck Methods**: Tishby et al. (2000) proposed the classic IB framework; β-VAE (Burgess et al., 2018) introduced IB to Variational Autoencoders. This work extends IB to temporal Koopman representations of dynamical systems.
- **Dynamical System Representation Learning**: E2C (Banijamali et al., 2019) and PCC (Levine et al., 2020) learn controllable representations using VAEs. This work achieves better manifold structures through information-theoretic constraints.
- **Spectral Analysis & Effective Dimension**: von Neumann entropy is used in quantum information to measure entanglement. This work innovatively introduces it to Koopman representations to prevent mode collapse.

## Rating

| Dimension | Score (1-10) |
|------|:-----------:|
| Novelty | 8 |
| Theoretical Depth | 9 |
| Experimental Thoroughness | 7 |
| Writing Quality | 8 |
| Value | 7 |
| **Total Score** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Concepts' Information Bottleneck Models](concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] The Deleuzian Representation Hypothesis](the_deleuzian_representation_hypothesis.md)
- [\[ICLR 2026\] Gauge-invariant Representation Holonomy](gauge-invariant_representation_holonomy.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)
- [\[ICLR 2026\] On the Predictive Power of Representation Dispersion in Language Models](on_the_predictive_power_of_representation_dispersion_in_language_models.md)

</div>

<!-- RELATED:END -->
