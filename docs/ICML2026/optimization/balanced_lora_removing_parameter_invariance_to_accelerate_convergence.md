---
title: >-
  [Paper Note] Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence
description: >-
  [ICML2026][Optimization][LoRA Fine-tuning] This paper reveals that the overparameterization of LoRA leads to varying condition numbers for different low-rank factors $(A, B)$. It proves that the **balanced minimum point** ($A^\top A = BB^\top$) possesses the optimal condition number. Based on this, it proposes BaLoRA—projecting adapters onto the balanced manifold after each optimization step to accelerate convergence and enhance fine-tuning performance with almost zero overhe…
tags:
  - "ICML2026"
  - "Optimization"
  - "LoRA Fine-tuning"
  - "Parameter Invariance"
  - "Condition Number Optimization"
  - "Balanced Manifold Projection"
  - "Convergence Acceleration"
date: 2026-05-08
content_hash: 2cdac61c8e12a69f
---

# Balanced LoRA: Removing Parameter Invariance to Accelerate Convergence

**Conference**: ICML2026  
**arXiv**: [2605.31484](https://arxiv.org/abs/2605.31484)  
**Code**: https://github.com/vcastin/balora  
**Area**: Optimization  
**Keywords**: LoRA Fine-tuning, Parameter Invariance, Condition Number Optimization, Balanced Manifold Projection, Convergence Acceleration  

## TL;DR

This paper reveals that the overparameterization of LoRA leads to varying condition numbers for different low-rank factors $(A, B)$. It proves that the **balanced minimum point** ($A^\top A = BB^\top$) possesses the optimal condition number. Based on this, it proposes BaLoRA—projecting adapters onto the balanced manifold after each optimization step to accelerate convergence and enhance fine-tuning performance with almost zero overhead.

## Background & Motivation

**Background**: LoRA is currently the standard method for Parameter-Efficient Fine-Tuning (PEFT) of Large Language Models. It approximates weight updates through the product $AB$ of low-rank matrices $A \in \mathbb{R}^{a \times r}$ and $B \in \mathbb{R}^{r \times b}$, reducing trainable parameters from $ab$ to $r(a+b)$.

**Limitations of Prior Work**: LoRA suffers from inherent overparameterization—for any invertible matrix $R$, $(AR, R^{-1}B)$ produces the exact same adapter matrix $AB$ as $(A, B)$. Consequently, the loss function's minimum is not an isolated point but an $r^2$-dimensional continuous manifold. Existing works (LoRA+, OLoRA, etc.) primarily improve via initialization or learning rates, failing to fundamentally resolve the optimization inefficiency caused by overparameterization.

**Key Challenge**: Different factorizations $(A, B)$ of the same adapter matrix $AB$ exhibit **starkly different condition numbers**, leading to significant differences in asymptotic convergence rates when gradient descent converges to different minima. Minima with poor condition numbers correspond to steeper loss surfaces, causing severe oscillations for the optimizer.

**Key Insight**: Starting from spectral analysis of the Hessian matrix, the authors find that the condition number at a minimum $(A, B)$ is entirely determined by the singular values of $A$ and $B$. When the "balance condition" $A^\top A = BB^\top$ holds, the singular values of both factors are perfectly aligned, reaching the theoretical optimal condition number.

**Core Idea**: Project $(A, B)$ onto the balanced manifold after each optimization step. This achieves the optimal condition number at a lightweight computational cost of $\mathcal{O}((a+b)r^2)$, thereby accelerating asymptotic convergence.

## Method

### Overall Architecture

The workflow of BaLoRA is extremely concise: after each optimizer update (e.g., AdamW) in standard LoRA, an additional **balanced projection** $P(A, B)$ is executed to map the low-rank factors onto the "hyper-balanced manifold" $\mathcal{H}$. The projection keeps the product $AB$ unchanged (thus preserving the loss value) but alters the factorization to optimize the condition number. Input and output are fully consistent with standard LoRA, allowing seamless integration into existing training pipelines.

### Key Designs

**1. Balanced Projection Operator $P(A, B)$: Preserving $AB$ while optimizing the condition number**

The pain point of LoRA is that a single adapter matrix $AB$ has infinite factorizations, and gradient descent randomly lands on one that might have a poor condition number. BaLoRA performs a projection after each optimization step to map $(A, B)$ onto the "hyper-balanced manifold" $\mathcal{H}=\{(US^{1/2}, S^{1/2}V)\mid U^\top U=VV^\top=I_r,\, S\in\mathbb{D}_+^r\}$, while strictly maintaining the product $AB$. The algorithm utilizes factorization techniques: first perform polar decomposition $A=R_A S_A$ and $B=S_B R_B$, then perform SVD on $S=S_A S_B$ to get $S=U\Sigma V^\top$, and finally output $A^{\text{proj}}=R_A U\Sigma^{1/2}$ and $B^{\text{proj}}=\Sigma^{1/2}V^\top R_B$. This is designed to minimize cost—performing SVD directly on $AB$ would cost $\mathcal{O}(abr)$, whereas this SVD runs only on an $r\times r$ matrix. The total projection cost is only $\mathcal{O}((a+b)r^2)$, which is negligible compared to optimizer updates.

**2. Theoretical Guarantee of Optimal Condition Number: Proving the balanced point $A^\top A=BB^\top$ has the minimum condition number**

The projection target is derived from Hessian spectral analysis. For the matrix factorization case ($\text{rk}(Z)=r$), the Hessian eigenvalues at the minimum are $\sigma_i(A)^2+\sigma_j(B)^2$, corresponding to the condition number:

$$\kappa=\frac{\sigma_1(A)^2+\sigma_1(B)^2}{\min\!\big(\sigma_r(A)^2,\,\sigma_r(B)^2\big)}.$$

When the balance condition $A^\top A=BB^\top$ is met, the singular values of the two factors align: $\sigma_i(A)=\sigma_i(B)=\sigma_i(Z)^{1/2}$, and the condition number is compressed to the theoretical lower bound $\kappa_{\min}=2\sigma_1(Z)/\sigma_r(Z)$. For the more general $\text{rk}(Z)\ge r$ case, the key quantity becomes the $r$-spectral gap $\sigma_r(Z)-\sigma_{r+1}(Z)$. This analysis explains why different factorizations of the same $AB$ result in different training speeds.

**3. Intrinsic Geometric Interpretation via Bures Metric: BaLoRA as Natural Gradient Descent on low-rank manifolds**

The authors provide a more elegant perspective by reformulating BaLoRA-GD as Riemannian Gradient Descent on the rank-$r$ matrix manifold $\mathcal{N}_r$ with respect to the Bures metric. Defining the inverse Bures metric $H_X[W]=(XX^\top)^{1/2}W+W(X^\top X)^{1/2}$, iterations can be written as $X_{k+1}=R(X_k,-\tau_k\Delta_k)$, where $\Delta_k=H_{X_k}[\nabla g(X_k)]$ is the Riemannian gradient and $R$ is a retraction. This shows BaLoRA is essentially natural gradient descent on the low-rank manifold; factorization $(A, B)$ is simply an efficient implementation. This connects LoRA optimization with Bures–Wasserstein geometry in optimal transport.

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

BaLoRA and RefLoRA (another balancing method) consistently rank in the top two, validating the effectiveness of balance constraints for convergence acceleration.

### Ablation Study: Rank Sensitivity (Qwen-2.5-3B, DM Mathematics 1B tokens)

| Method | r=8 | r=16 | r=32 | r=64 | r=128 |
|------|-----|------|------|------|-------|
| LoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| DoRA | 1.035 | 1.032 | 1.031 | 1.030 | 1.030 |
| LoRA-RITE | 1.047 | 1.045 | 1.046 | 1.052 | 1.069 |
| OLoRA | 1.039 | 1.037 | 1.036 | 1.036 | 1.036 |
| RefLoRA | 1.027 | 1.023 | 1.024 | 1.027 | 1.032 |
| **BaLoRA** | **1.026** | **1.020** | **1.017** | **1.015** | **1.014** |

BaLoRA's advantage is particularly significant at **high ranks** ($r=64/128$). While RefLoRA's performance degrades at high ranks, BaLoRA continues to improve, leading LoRA by approximately 1.5% and RefLoRA by 1.8% at $r=128$.

## Highlights & Insights

- **Theory-Practice Loop**: Deriving the optimal balance condition from Hessian spectral analysis to designing a lightweight projection operator and validating convergence acceleration creates a complete logical chain.
- **Advantage in High-rank Scenarios**: As $r$ increases, the invariance dimension ($r^2$) of overparameterization grows faster, making BaLoRA's condition number improvement more significant.
- **Hyperparameter Robustness**: BaLoRA is significantly less sensitive to learning rates and initialization scaling than LoRA/OLoRA/LoRA-GA, making tuning easier in practice.
- **Bures Metric Connection**: Linking LoRA optimization to Bures-Wasserstein geometry in optimal transport opens new avenues for future theoretical analysis.

## Limitations & Future Work

- Theoretical analysis focuses on **single-layer adapters** and **regression loss**; condition number analysis for multi-layer fine-tuning and cross-entropy loss remains incomplete.
- The current projection preserves $AB$ but alters the momentum/variance states of the Adam optimizer, potentially introducing short-term training oscillations (as seen in synthetic experiments where BaLoRA starts slightly slower).
- Comparisons with non-LoRA PEFT paradigms like GaLore are missing.
- Although the projection step is lightweight, polar decomposition and SVD still involve overhead at very large $r$, warranting further optimization for $r > 128$.

## Related Work & Insights

- **RefLoRA** (Zhang et al., 2025) also enforces balance but uses a different mapping and requires 100 warmup steps; BaLoRA's projection is simpler and requires no warmup.
- **LORO** (Mo et al., 2025) approaches from a Riemannian optimization perspective but requires a specialized solver; BaLoRA is compatible with any optimizer via post-projection.
- **LoRA+** (Hayou et al., 2024) improves A/B dynamics through different learning rates; this can be combined orthogonally with BaLoRA's balanced projection.

## Rating

- Novelty: 9/10 — First to establish a theoretical link between balanced factors and optimal convergence rates via condition numbers.
- Experimental Thoroughness: 8/10 — Covers multiple models, datasets, and ranks, but lacks downstream task accuracy evaluations.
- Writing Quality: 9/10 — Clear theoretical derivations and well-articulated geometric intuition.
- Value: 8/10 — Strong practicality and elegant theory, though improvement margins are limited in small-rank scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICML 2026\] TPV: Parameter Perturbations Through the Lens of Test Prediction Variance](tpv_parameter_perturbations_through_the_lens_of_test_prediction_variance.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[ICLR 2026\] Sign-SGD via Parameter-Free Optimization](../../ICLR2026/optimization/sign-sgd_via_parameter-free_optimization.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)

</div>

<!-- RELATED:END -->
