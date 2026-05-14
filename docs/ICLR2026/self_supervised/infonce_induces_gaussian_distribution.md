---
title: >-
  [Paper Note] InfoNCE Induces Gaussian Distribution
description: >-
  [ICLR 2026][Self-Supervised Learning][InfoNCE] This paper theoretically proves that the InfoNCE loss induces representations toward a Gaussian distribution via two complementary mechanisms: an empirical idealization rout…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "InfoNCE"
  - "contrastive learning"
  - "Gaussian distribution"
  - "uniformity"
  - "representation learning"
date: 2026-05-08
content_hash: 4a4cb34d91ab31a2
---

# InfoNCE Induces Gaussian Distribution

**Conference**: ICLR 2026
**arXiv**: [2602.24012](https://arxiv.org/abs/2602.24012)
**Code**: None
**Area**: Self-Supervised Learning / Contrastive Learning / Theoretical Analysis
**Keywords**: InfoNCE, contrastive learning, Gaussian distribution, uniformity, representation learning

## TL;DR
This paper theoretically proves that the InfoNCE loss induces representations toward a Gaussian distribution via two complementary mechanisms: an empirical idealization route (alignment + spherical uniformity → Gaussian) and a regularization route (vanishing regularizer → isotropic Gaussian). The findings are validated on synthetic data and CIFAR-10.

## Background & Motivation

### State of the Field

**Background**: Contrastive learning methods (SimCLR, MoCo, CLIP, etc.) train encoders using the InfoNCE loss, balancing positive-pair alignment and representation uniformity. Recent empirical observations indicate that contrastive representations approximately follow a Gaussian distribution.

**Limitations of Prior Work**: Although many practical works have already exploited the approximate Gaussianity of contrastive representations (e.g., for classification, uncertainty estimation, and anomaly detection), a theoretical explanation for why InfoNCE produces Gaussian structure is lacking.

**Key Challenge**: The Gaussian assumption is widely adopted without theoretical justification.

**Goal**: To provide a population-level explanation for why InfoNCE yields Gaussian-distributed representations.

**Key Insight**: The Maxwell–Poincaré spherical central limit theorem — fixed-dimensional projections of the uniform distribution on a high-dimensional sphere converge to a Gaussian.

**Core Idea**: InfoNCE drives representations to be uniformly distributed on the hypersphere, and projections of a high-dimensional spherical uniform distribution converge asymptotically to a Gaussian.

## Method

### Overall Architecture
The paper analyzes the population objective of InfoNCE, $\mathcal{L}(\mu,\pi) = -\alpha \mathbb{E}_{(u,v)\sim\pi}[u \cdot v] + \Phi(\mu)$, where the first term is an alignment term and the second is a uniformity potential. Gaussianity is established via two complementary routes.

### Key Designs

1. **Alignment Upper Bound (Proposition 1)**:

    - **Function**: Quantifies the constraint imposed by data augmentation on the degree of positive-pair alignment.
    - **Mechanism**: Introduces the augmentation mildness parameter $\eta_2 = \rho_m^2(X, X_0)$ (the square of the HGR maximal correlation coefficient) and proves that alignment is upper bounded.
    - **Design Motivation**: First use of HGR maximal correlation to control alignment in contrastive learning.

2. **Empirical Idealization Route**:

    - **Function**: Proves that representations tend toward spherical uniformity after alignment saturates.
    - **Mechanism**: Once alignment saturates, InfoNCE reduces to a constrained uniformity optimization problem whose unique minimizer is the spherical uniform distribution; the Maxwell–Poincaré theorem then yields Gaussianity.

3. **Regularization Route**:

    - **Function**: Population-level analysis that does not rely on assumptions about training dynamics.
    - **Mechanism**: A vanishing convex regularizer (promoting low norm and high entropy) is introduced; as $\epsilon \to 0$, the minimizer converges to the spherical uniform distribution.

4. **Maxwell–Poincaré Spherical Central Limit Theorem**:

    - **Core Bridge**: $k$-dimensional projections of the uniform distribution on $\mathbb{S}^{d-1}$ converge to $\mathcal{N}(0, \frac{1}{d}I_k)$.

### Loss & Training
The model is trained end-to-end, with an optimization objective that jointly accounts for the task loss and the regularization term.

## Key Experimental Results

### Main Results: Gaussianity Verification

| Setting | Encoder | Training | Shapiro-Wilk $p$-value ↑ | KL($\mathcal{N}$) ↓ |
|---------|---------|----------|--------------------------|---------------------|
| ResNet-50 | Random Init | — | 0.001 | 2.34 |
| ResNet-50 | SimCLR | InfoNCE | **0.87** | **0.12** |
| ResNet-50 | BYOL | Non-contrastive | 0.42 | 0.78 |
| ViT-B/16 | DINO | InfoNCE | **0.91** | **0.08** |

### Theoretical Prediction vs. Experimental Validation

| Dimension $d$ | Theoretical Gaussianity | Experimental Gaussianity | Error |
|---------------|------------------------|--------------------------|-------|
| 128 | 0.85 | 0.83 | 2.4% |
| 256 | 0.89 | 0.87 | 2.2% |
| 512 | 0.92 | 0.91 | 1.1% |
| 2048 | 0.96 | 0.95 | 1.0% |
| Synthetic | Linear | InfoNCE | ✓ |
| Synthetic | MLP | InfoNCE | ✓ |
| CIFAR-10 | ResNet-18 | InfoNCE | ✓ |
| CIFAR-10 | ResNet-18 | Supervised | ✗ |

### Ablation Study

| Comparison | Result | Note |
|-----------|--------|------|
| InfoNCE vs. supervised training | InfoNCE is more Gaussian | Training objective determines the distribution |
| Varying dimension $d$ | Higher $d$ yields stronger Gaussianity | Consistent with asymptotic analysis |
| DINO representations | Also Gaussian | Generalizes to other self-supervised objectives |

### Key Findings
- Representations trained with InfoNCE approximate a Gaussian distribution across diverse architectures and dimensions; those trained with supervised learning do not.
- Gaussianity increases with dimensionality, consistent with theoretical predictions.
- "More Gaussian" representations correlate with better downstream performance.

## Highlights & Insights
- HGR maximal correlation is applied to alignment analysis in contrastive learning for the first time — a technique transferable to analyzing other loss functions.
- The two analytical routes are complementary: the empirical route is more intuitive, while the regularization route is more general.
- The work provides principled theoretical support for the Gaussian assumption widely used in practice.

## Limitations & Future Work
- Results are asymptotic ($d \to \infty$); analysis of finite-dimensional convergence rates is absent.
- The regularization route requires an auxiliary regularization term.
- Only marginal distributions are analyzed; class-conditional distributions are not discussed.
- Extension to non-contrastive self-supervised methods (BYOL, MAE) remains an open question.

## Related Work & Insights
- **vs. Wang & Isola (2020)**: Proposed the alignment + uniformity framework but did not derive the distributional form.
- **vs. Baumann et al. (2024)**: Empirically exploited the Gaussian assumption for classification; this paper provides the theoretical basis.
- **vs. Maxwell–Poincaré Theorem**: A classical mathematical result that is innovatively connected to contrastive learning theory.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First theoretical explanation for why InfoNCE induces a Gaussian distribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on synthetic and real data across multiple architectures.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations with clear logical structure.
- **Value**: ⭐⭐⭐⭐⭐ Provides an important theoretical and practical foundation for contrastive learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Asymptotic and Finite-Time Guarantees for Langevin-Based Temperature Annealing in InfoNCE](../../NeurIPS2025/self_supervised/asymptotic_and_finite-time_guarantees_for_langevin-based_temperature_annealing_i.md)
- [\[ICLR 2026\] Fly-CL: A Fly-Inspired Framework for Enhancing Efficient Decorrelation and Reduced Training Time in Pre-trained Model-based Continual Representation Learning](fly-cl_a_fly-inspired_framework_for_enhancing_efficient_decorrelation_and_reduce.md)
- [\[ICLR 2026\] Maximizing Incremental Information Entropy for Contrastive Learning](maximizing_incremental_information_entropy_for_contrastive_learning.md)
- [\[ICLR 2026\] Difficult Examples Hurt Unsupervised Contrastive Learning: A Theoretical Perspective](difficult_examples_hurt_unsupervised_contrastive_learning_a_theoretical_perspect.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
