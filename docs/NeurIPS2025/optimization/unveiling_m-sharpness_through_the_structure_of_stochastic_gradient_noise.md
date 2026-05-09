---
title: >-
  [Paper Note] Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise
description: >-
  [NeurIPS 2025][Optimization][SAM] This paper reveals the theoretical mechanism underlying m-sharpness in SAM through an extended stochastic differential equation (SDE) framework — smaller micro-batch size $m$ induces stronger implicit regularization via the covariance of stochastic gradient noise (SGN) — and proposes a parallelizable Reweighted SAM (RW-SAM) method based on this insight.
tags:
  - NeurIPS 2025
  - Optimization
  - SAM
  - m-sharpness
  - stochastic gradient noise
  - SDE approximation
  - generalization
date: 2026-05-08
content_hash: 024eefd777cc1983
---

# Unveiling m-Sharpness Through the Structure of Stochastic Gradient Noise

**Conference**: NeurIPS 2025
**arXiv**: [2509.18001](https://arxiv.org/abs/2509.18001)
**Code**: Not available
**Area**: Optimization
**Keywords**: SAM, m-sharpness, stochastic gradient noise, SDE approximation, generalization

## TL;DR

This paper reveals the theoretical mechanism underlying m-sharpness in SAM through an extended stochastic differential equation (SDE) framework — smaller micro-batch size $m$ induces stronger implicit regularization via the covariance of stochastic gradient noise (SGN) — and proposes a parallelizable Reweighted SAM (RW-SAM) method based on this insight.

## Background & Motivation

Sharpness-Aware Minimization (SAM) is one of the most successful techniques for improving generalization in recent years. SAM seeks flat minima by minimizing the perturbed loss $\min_x f(x + \rho\epsilon^*(x))$. However, one puzzling phenomenon has long lacked a theoretical explanation:

**The m-sharpness phenomenon**: When a mini-batch is split into smaller micro-batches (of size $m$) and perturbation directions are computed independently before merging the update — generalization improves monotonically as $m$ decreases. Specifically:
- n-SAM (full-batch perturbation) yields almost no generalization improvement
- Mini-batch SAM (standard version) achieves good generalization
- m-SAM ($m <$ batch size) generalizes even better

This phenomenon is particularly important in distributed training: in multi-GPU settings, each GPU computing its perturbation direction from local data is itself an instance of m-SAM ($m =$ per-GPU batch size), while smaller $m$ requires serial computation, precluding parallelization and incurring substantial computational overhead.

Andriushchenko & Flammarion (2022) proposed several hypotheses to explain m-sharpness, but all were refuted by their own experiments. **The root cause of m-sharpness remains an open problem.**

The authors' starting point is to extend existing SDE frameworks to jointly track both the learning rate $\eta$ and perturbation radius $\rho$ to arbitrary order, thereby precisely characterizing differences in drift terms across SAM variants.

## Method

### Overall Architecture

By approximating discrete iterations as continuous SDEs, the paper compares the drift terms of three variants — n-SAM, mini-batch SAM, and m-SAM — and shows that the strength of implicit regularization is directly linked to the covariance of SGN, which increases as $m$ decreases.

### Key Designs

1. **Two-parameter weak approximation framework (Definition 3.2)**: Defines order-$(\alpha, \beta)$ weak approximations, allowing $\eta$ and $\rho$ to approach zero at independent rates. Unlike prior work (Compagnoni et al., 2023), this avoids fixing the ratio $\eta/\rho$. The Dynkin expansion (rather than the full Itô–Taylor expansion) is used to control remainder terms under the two-parameter setting.

2. **SDEs for three USAM variants (Theorems 3.3–3.5)**: USAM is the unnormalized version of SAM (without gradient norm normalization), which admits closed-form drift terms. The drift terms of the three variants are compared as follows:

    - **n-USAM (Theorem 3.4)**: $dX_t = -\nabla\left(f + \frac{\rho}{2}\|\nabla f\|^2\right)dt + \sqrt{\eta\Sigma^{n}}dW_t$
      Implicit regularization includes only the full gradient norm term $\frac{\rho}{2}\|\nabla f\|^2$, with **no SGN covariance term**.

    - **Mini-batch USAM (Theorem 3.3)**: $dX_t = -\nabla\left(f + \frac{\rho}{2}\|\nabla f\|^2 + \frac{\rho}{2|\gamma|}\text{tr}(V)\right)dt + \sqrt{\eta\Sigma^{U}}dW_t$
      An additional SGN covariance trace term $\text{tr}(V)$ appears with coefficient $\rho/(2|\gamma|)$.

    - **m-USAM (Theorem 3.5)**: $dX_t = -\nabla\left(f + \frac{\rho}{2}\|\nabla f\|^2 + \frac{\rho}{2m}\text{tr}(V)\right)dt + \sqrt{\frac{m\eta}{|\gamma|}\Sigma^{m}}dW_t$
      The SGN regularization coefficient becomes $\rho/(2m)$, which is stronger when $m < |\gamma|$. The diffusion coefficient is also reduced by a factor of $m/|\gamma|$, decreasing stochastic perturbation.

3. **Analogous results for normalized SAM (Theorems 3.6–3.8)**: After normalization, the expected gradient norm has no elementary closed form, but Proposition 3.9 proves that $\mathbb{E}\|\nabla f_\gamma\|$ increases monotonically as $|\gamma|$ decreases (strictly so under log-concave distributions). Consequently, the regularization term $\frac{\rho}{m}\mathbb{E}\|\sum_{i \in \mathcal{I}} \nabla f_i\|$ in m-SAM grows as $m$ decreases.

4. **SGN covariance regularization and generalization (Section 3.5)**: Two perspectives explain why regularizing $\text{tr}(V)$ is beneficial:

    - Information-theoretic view: The generalization error bound decomposes into a sum of mutual information terms, where the "trajectory term" is controlled by the SGN covariance.
    - Convergence phase: Near a minimum, $V(x) \approx \text{FIM}(x) \approx \nabla^2 f(x)$ (the Hessian), so regularizing $\text{tr}(V)$ effectively regularizes the trace of the Hessian — a well-established sharpness measure.

5. **Reweighted SAM (RW-SAM)**: Motivated by the theoretical analysis, this parallelizable method is designed to simulate the effect of m-SAM. Core idea: samples with larger gradient norms carry stronger SGN and should receive higher weights.

   The optimization objective for the perturbation direction is:
   $$\max_{P \in \Delta} \max_{\|\epsilon\| \leq 1} \langle \sum_i p_i \nabla f_i, \epsilon \rangle + \mathbb{H}(P)/\lambda$$

   Solving the relaxation yields Gibbs distribution weights: $p_i^* = \frac{\exp(\lambda\|\nabla f_i\|)}{\sum_j \exp(\lambda\|\nabla f_j\|)}$

   Per-sample gradient norms are estimated via finite differences and Monte Carlo (one additional forward pass): $\|\nabla f_i\| \approx |f_i(x + \delta z) - f_i(x)|/\delta$, where $z$ is a Rademacher random vector (optimal variance).

## Key Experimental Results

### Training from Scratch (Table 1)

| Model | Dataset | SGD | SAM | RW-SAM |
|---|---|---|---|---|
| ResNet-18 | CIFAR-10 | 95.62 | 95.99 | **96.24** |
| ResNet-50 | CIFAR-10 | 95.64 | 96.06 | **96.34** |
| WRN-28-10 | CIFAR-10 | 96.47 | 96.91 | **97.11** |
| ResNet-18 | CIFAR-100 | 78.91 | 78.90 | **79.31** |
| ResNet-50 | CIFAR-100 | 79.55 | 80.31 | **80.83** |
| WRN-28-10 | CIFAR-100 | 81.55 | 83.25 | **83.52** |

### Large-Scale Training (Table 2a) & Fine-Tuning (Table 2b)

| Setting | SGD/AdamW | SAM | RW-SAM |
|---|---|---|---|
| ResNet-50 / ImageNet | 76.67 | 77.16 | **77.37** |
| ViT-B/16 / CIFAR-10 | 98.24 | 98.40 | **98.58** |
| ViT-B/16 / CIFAR-100 | 88.71 | 89.63 | **89.89** |

### Robustness to Label Noise (Table 4)

| Noise Ratio | SGD | SAM | RW-SAM |
|---|---|---|---|
| 20% | 87.54 | 90.01 | **90.34** |
| 40% | 83.66 | 86.40 | **86.87** |
| 60% | 76.64 | 78.79 | **81.52** |
| 80% | 46.53 | 37.69 | **53.17** |

At 80% noise, RW-SAM outperforms SAM by 16%, indicating that the reweighting mechanism is especially advantageous under extreme noise.

### Verification of SGN Covariance (Table 6)

| Optimizer | Trace of Gradient Covariance |
|---|---|
| SGD | 572.39±24.15 |
| SAM | 198.40±6.20 |
| RW-SAM | **177.79±5.10** |

RW-SAM converges to minima with the smallest SGN covariance, empirically validating the theoretical analysis.

### Key Findings

- n-SAM lacks the SGN regularization effect, explaining its failure to improve generalization.
- m-SAM offers a dual advantage: (1) the SGN regularization coefficient is strengthened from $\rho/(2|\gamma|)$ to $\rho/(2m)$; (2) the diffusion term is reduced by a factor of $m/|\gamma|$.
- In "bad minima escape" experiments (Figs. 1–2), smaller $m$ leads to faster escape and faster decay of SGN variance.
- RW-SAM matches the performance of m-SAM at $m=64$ while avoiding nearly twice the training time (adding only approximately 1/6 overhead).

## Highlights & Insights

- The paper provides a clear theoretical explanation for m-sharpness, a long-standing open phenomenon: it is fundamentally an implicit regularization of sharpness through SGN.
- The two-parameter SDE framework is more flexible than existing single-parameter frameworks, allowing $\eta$ and $\rho$ to approach zero independently.
- RW-SAM is practically appealing — requiring only one additional forward pass, fully parallelizable, with direct value in multi-GPU settings.
- Using Rademacher rather than Gaussian perturbations to minimize the variance of gradient norm estimation is a useful technical detail.

## Limitations & Future Work

- The accuracy of the SDE approximation relies on $\eta$ and $\rho$ being sufficiently small; results may be biased under the large learning rates used in practice.
- The hyperparameter $\lambda$ in RW-SAM, while not overly sensitive (Table 5), still requires tuning.
- The theoretical analysis primarily applies to USAM (which admits closed forms); for normalized SAM, only qualitative conclusions are available.
- The impact of noise introduced by finite-difference gradient norm estimation in deep networks is not yet fully understood.

## Related Work & Insights

- The paper directly addresses the open problem of m-sharpness posed by Foret et al. (2021) and Andriushchenko & Flammarion (2022).
- The SDE framework of Compagnoni et al. (2023) serves as the technical foundation; this work extends it from single-parameter to two-parameter settings.
- A connection is established with the information-theoretic generalization bounds of Neu et al. (2021): the SGN covariance controls the "trajectory term" in generalization.
- The analysis provides direct guidance for practical distributed SAM training: the per-GPU batch size corresponds to the choice of $m$.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First theoretical explanation of m-sharpness from the SGN perspective, with a practical RW-SAM method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers training from scratch, fine-tuning, label noise, GLUE, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are well integrated, though the SDE derivations may be challenging for non-specialist readers.
- Value: ⭐⭐⭐⭐⭐ Resolves a core theoretical problem in the SAM literature; RW-SAM has direct engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)
- [\[NeurIPS 2025\] Unveiling the Power of Multiple Gossip Steps: A Stability-Based Generalization Analysis in Decentralized Training](unveiling_the_power_of_multiple_gossip_steps_a_stability-based_generalization_an.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](a_unified_approach_to_submodular_maximization_under_noise.md)

</div>

<!-- RELATED:END -->
