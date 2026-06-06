---
title: >-
  [Paper Note] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence
description: >-
  [ICML2026][Optimization][LoRA fine-tuning] This work reveals that over-parameterization in LoRA causes different low-rank factors $(A…
tags:
  - "ICML2026"
  - "Optimization"
  - "LoRA fine-tuning"
  - "parameter invariance"
  - "condition number optimization"
  - "balanced manifold projection"
  - "convergence acceleration"
date: 2026-05-08
content_hash: 58227bb8fb27762c
---

# Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence

**Conference**: ICML2026  
**arXiv**: [2605.31484](https://arxiv.org/abs/2605.31484)  
**Code**: https://github.com/vcastin/balora  
**Area**: optimization  
**Keywords**: LoRA fine-tuning, parameter invariance, condition number optimization, balanced manifold projection, convergence acceleration  

## TL;DR

This work reveals that over-parameterization in LoRA causes different low-rank factors $(A,B)$ to have different condition numbers. It proves that **balanced minima** ($A^\top A = BB^\top$) possess the optimal condition number, and accordingly proposes BaLoRA—projecting adapters onto the balanced manifold after each optimization step to accelerate convergence and enhance fine-tuning performance with near-zero overhead.

## Background & Motivation

**Background**: LoRA is currently the mainstream method for Parameter-Efficient Fine-Tuning (PEFT) of large language models. It approximates weight updates through the product $AB$ of low-rank matrices $A \in \mathbb{R}^{a \times r}$ and $B \in \mathbb{R}^{r \times b}$, reducing trainable parameters from $ab$ to $r(a+b)$.

**Limitations of Prior Work**: LoRA suffers from inherent over-parameterization—for any invertible matrix $R$, $(AR, R^{-1}B)$ and $(A,B)$ produce the exact same adapter matrix $AB$. This means the minimum of the loss function is not an isolated point but an $r^2$-dimensional continuous manifold. Existing works (LoRA+, OLoRA, etc.) mainly improve initialization or learning rates but fail to fundamentally address the optimization inefficiency caused by over-parameterization.

**Key Challenge**: Different factorizations $(A,B)$ of the same adapter matrix $AB$ have **vastly different condition numbers**, leading to significant discrepancies in asymptotic convergence rates when gradient descent converges to different minima. Minima with poor condition numbers correspond to steeper loss surfaces, causing severe optimizer oscillations in their vicinity.

**Key Insight**: Starting from the spectral analysis of the Hessian matrix, the authors discover that the condition number of the minimum $(A,B)$ is entirely determined by the singular values of $A$ and $B$. When $A^\top A = BB^\top$ (the "balance condition") holds, the singular values of the two factors are perfectly aligned, achieving the theoretical optimal condition number.

**Core Idea**: Project $(A,B)$ onto the balanced manifold after each optimization step to exchange a lightweight computational cost of $\mathcal{O}((a+b)r^2)$ for the optimal condition number, thereby accelerating asymptotic convergence.

## Method

### Overall Architecture

The workflow of BaLoRA is extremely concise: after every optimizer update (e.g., AdamW) in standard LoRA, an additional **balanced projection** $P(A,B)$ is executed to map the low-rank factors onto the "hyper-balanced manifold" $\mathcal{H}$. The projection preserves the product $AB$ (and thus the loss value), but changes the factorization to optimize the condition number. Inputs and outputs are identical to standard LoRA, allowing seamless integration into existing training pipelines.

### Key Designs

1. **Balanced Projection Operator $P(A,B)$**:

    - **Function**: Maps any factor pair $(A,B)$ to the hyper-balanced manifold $\mathcal{H} = \{(US^{1/2}, S^{1/2}V) \mid U^\top U = VV^\top = I_r, S \in \mathbb{D}_+^r\}$ while strictly preserving $AB$.
    - **Mechanism**: First perform polar decomposition on $A$ and $B$ respectively as $A = R_A S_A$ and $B = S_B R_B$. Then perform SVD on $S = S_A S_B$ to get $S = U\Sigma V^\top$. Finally output $A^{\text{proj}} = R_A U \Sigma^{1/2}$ and $B^{\text{proj}} = \Sigma^{1/2} V^\top R_B$. The computational complexity is only $\mathcal{O}((a+b)r^2)$ because SVD is performed only on the small $r \times r$ matrix.
    - **Design Motivation**: Direct SVD on $AB$ costs $\mathcal{O}(abr)$. The strategy of polar decomposition combined with small-matrix SVD reduces the cost to small-scale operations related to $r$, making the projection overhead negligible compared to optimizer steps.

2. **Theoretical Guarantees for Optimal Condition Number**:

    - **Function**: Provides a rigorous theoretical foundation for balanced projection, proving that balanced minima have the minimum condition number among all equivalent minima.
    - **Mechanism**: For the matrix factorization case ($\text{rk}(Z)=r$), the Hessian eigenvalues are $\sigma_i(A)^2 + \sigma_j(B)^2$, and the condition number is $\kappa = (\sigma_1(A)^2 + \sigma_1(B)^2) / \min(\sigma_r(A)^2, \sigma_r(B)^2)$. When $A^\top A = BB^\top$, then $\sigma_i(A) = \sigma_i(B) = \sigma_i(Z)^{1/2}$, and the condition number reaches its minimum $\kappa_{\min} = 2\sigma_1(Z)/\sigma_r(Z)$. For the general case ($\text{rk}(Z) \geq r$), the key quantity becomes the $r$-spectral gap $\sigma_r(Z) - \sigma_{r+1}(Z)$.
    - **Design Motivation**: Explains why different factorizations of the same $AB$ lead to different training speeds and provides a principled method for selecting the optimal decomposition.

3. **Intrinsic Geometric Interpretation under the Bures Metric**:

    - **Function**: Reformulates BaLoRA-GD as Riemannian Gradient Descent on the rank-$r$ matrix manifold $\mathcal{N}_r$ with respect to the Bures metric.
    - **Mechanism**: Define the inverse Bures metric $H_X[W] = (XX^\top)^{1/2}W + W(X^\top X)^{1/2}$. The BaLoRA-GD iteration can be written as $X_{k+1} = R(X_k, -\tau_k \Delta_k)$, where $\Delta_k = H_{X_k}[\nabla g(X_k)]$ is the Riemannian gradient and $R$ is a retraction on the manifold.
    - **Design Motivation**: Provides an elegant geometric perspective—BaLoRA is essentially natural gradient descent on the low-rank matrix manifold, with factored $(A,B)$ as an efficient implementation.

## Key Experimental Results

### Main Results: Multi-dataset Fine-tuning Comparison (Qwen-2.5-3B, r=8)

| Method | Alpaca | CodeFeedback | OpenHermes | OpenOrca | WizardLM |
|------|--------|-------------|------------|----------|----------|
| LoRA | 1.352 | 0.638 | 0.707 | 0.774 | 0.663 |
| DoRA | 1.352 | 0.639 | 0.707 | 0.776 | 0.662 |
| LoRA-RITE | 1.353 | 0.639 | 0.707 | 0.776 | 0.663 |
| LORO | 1.504 | 0.669 | 0.750 | 0.859 | 0.689 |
| OLoRA | 1.360 | 0.641 | 0.712 | 0.782 | 0.666 |
| RefLoRA | 1.350 | 0.638 | 0.706 | 0.773 | 0.661 |
| **BaLoRA** | **1.350** | **0.638** | 0.707 | **0.773** | 0.662 |

BaLoRA and RefLoRA (another balancing method) consistently rank in the top two, validating the effectiveness of balanced constraints for convergence acceleration.

### Ablation Study: Rank Ablation (Qwen-2.5-3B, DM Mathematics 1B tokens)

| Method | r=8 | r=16 | r=32 | r=64 | r=128 |
|------|-----|------|------|------|-------|
| LoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| DoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| LoRA-RITE | 1.047 | 1.045 | 1.046 | 1.052 | 1.069 |
| OLoRA | 1.039 | 1.037 | 1.036 | 1.036 | 1.036 |
| RefLoRA | 1.027 | 1.023 | 1.024 | 1.027 | 1.032 |
| **BaLoRA** | **1.026** | **1.020** | **1.017** | **1.015** | **1.014** |

BaLoRA's advantage is particularly significant at **high ranks** (r=64/128): while RefLoRA shows performance degradation at high ranks, BaLoRA continues to improve, leading LoRA by ~1.5% and RefLoRA by ~1.8% at r=128.

## Highlights & Insights

- **Theory-Practice Closed Loop**: Complete logical chain from Hessian spectral analysis deriving optimal balance conditions $\rightarrow$ designing lightweight projection operators $\rightarrow$ experimental validation of convergence acceleration.
- **Unique Advantages in High-Rank Scenarios**: As $r$ increases, the dimensions of over-parameterized invariance ($r^2$) grow faster, making BaLoRA's condition number improvement more significant.
- **Hyperparameter Robustness**: BaLoRA is noticeably less sensitive to learning rates and initialization scaling than LoRA/OLoRA/LoRA-GA, making parameter tuning easier in practice.
- **Bures Metric Connection**: Links LoRA optimization with Bures-Wasserstein geometry in optimal transport, opening new perspectives for future theoretical analysis.

## Limitations & Future Work

- Theoretical analysis is primarily focused on **single-layer adapters** and **regression loss**; condition number analysis for simultaneous multi-layer fine-tuning and cross-entropy loss remains unfinished.
- The current projection keeps $AB$ constant but modifies the momentum/variance states of the Adam optimizer, potentially introducing brief training oscillations initially (BaLoRA starts slightly slower in synthetic experiments).
- Comparison with non-LoRA PEFT paradigms like GaLore is missing.
- Although the projection step is lightweight, polar decomposition and SVD still incur some overhead when $r$ is large; further optimization for $r > 128$ scenarios is worthwhile.

## Related Work & Insights

- **RefLoRA** (Zhang et al., 2025) also enforces balance but uses a different balancing map and requires a 100-step warmup; BaLoRA's projection is simpler and requires no warmup.
- **LORO** (Mo et al., 2025) takes a Riemannian optimization approach but requires a specialized solver; BaLoRA is compatible with any optimizer via post-projection.
- **LoRA+** (Hayou et al., 2024) improves A/B training dynamics via different learning rates; it can be orthogonally combined with BaLoRA's balanced projection.

## Rating

- Novelty: 9/10 — First to establish a theoretical link between balanced factors and optimal convergence rates from a condition number perspective.
- Experimental Thoroughness: 8/10 — Covers multiple models/datasets/ranks, but lacks accuracy evaluation on downstream tasks.
- Writing Quality: 9/10 — Clear theoretical derivation and well-explained geometric intuition.
- Value: 8/10 — Highly practical with elegant theory, though the improvement margin is limited in small-rank scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
