---
title: >-
  [Paper Note] The Affine Divergence: Aligning Activation Updates Beyond Normalisation
description: >-
  [ICLR 2026][Optimization][Affine divergence] This paper reveals a fundamental misalignment between the steepest descent direction in parameter space and the effective update propagated to activations under gradient desce…
tags:
  - "ICLR 2026"
  - "Optimization"
  - "Affine divergence"
  - "normalization theory"
  - "gradient descent"
  - "representation updates"
  - "PatchNorm"
date: 2026-05-08
content_hash: 107667dfde1935ba
---

# The Affine Divergence: Aligning Activation Updates Beyond Normalisation

**Conference**: ICLR 2026
**arXiv**: [2512.22247](https://arxiv.org/abs/2512.22247)
**Code**: None
**Area**: Optimization Theory
**Keywords**: Affine divergence, normalization theory, gradient descent, representation updates, PatchNorm

## TL;DR
This paper reveals a fundamental misalignment between the steepest descent direction in parameter space and the effective update propagated to activations under gradient descent — the "affine divergence" $\Delta\mathcal{L}/\Delta z_i = (\partial\mathcal{L}/\partial z_i) \cdot (\|\vec{x}\|^2+1)$ — derives normalization as the natural remedy from first principles, and discovers a non-normalizing alternative that empirically surpasses conventional normalization methods.

## Background & Motivation

**Background**: In deep learning, parameters are updated along the steepest descent direction via gradient descent. Activations (representations), however, are closer to the loss function and carry sample-dependent information. The empirical success of normalization methods (e.g., BatchNorm) is well established, yet their mechanistic explanations remain contested.

**Limitations of Prior Work**:
   - Whether the steepest descent direction in parameter space coincides with the optimal update direction in activation space — it does not.
   - Existing explanations of normalization (internal covariate shift, loss landscape smoothing, etc.) lack a first-principles derivation from an update-alignment perspective.

**Key Challenge**: Parameter updates propagate to activations and introduce a sample-dependent quadratic bias factor $(\|\vec{x}\|^2+1)$ — the effective learning rate for high-magnitude samples is disproportionately large, geometrically distorting the gradient step.

**Key Insight**: Rather than viewing normalization through the lens of statistical regularization, this paper rederives it from the perspective of "parameter–activation update alignment," arriving at normalization as an unexpected natural consequence.

**Core Idea**: The success of normalization is not attributable to statistical standardization per se, but to the fact that it precisely cancels the sample-dependent quadratic bias introduced when parameter updates propagate to activations.

## Method

### Overall Architecture
Starting from an affine layer $z_i = W_{ij}x_j + b_i$, the paper derives the effective update $\Delta z_i$ to activations induced by updates to parameters $(W, b)$, identifies a divergence factor $(\|\vec{x}\|^2+1)$ relative to the ideal steepest descent direction, and subsequently derives schemes to eliminate this divergence.

### Key Designs

1. **Derivation of the Affine Divergence**:

    - Parameter update: $W'_{ij} = W_{ij} - \eta g_i x_j$, $b'_i = b_i - \eta g_i$
    - Propagation to activations: $z'_i = z_i - \eta g_i(\|\vec{x}\|^2 + 1)$
    - Ideal update: $\Delta z_i^{ideal} = -\eta g_i$
    - **Divergence**: $\frac{\Delta\mathcal{L}}{\Delta z_i} = \frac{\partial\mathcal{L}}{\partial z_i} \cdot (\|\vec{x}\|^2 + 1)$
    - Implication: the effective learning rate per sample is $\eta_{eff} = \eta(\|\vec{x}\|^2+1)$ — high-magnitude samples take disproportionately large gradient steps.

2. **Solution 1: Normalization (BatchNorm Derived Unexpectedly)**:

    - The most direct way to eliminate the divergence is to divide activations by $\sqrt{\|\vec{x}\|^2+1}$.
    - This is equivalent to L2-normalizing the augmented input vector $[\vec{x}; 1]$ (incorporating the bias term).
    - **Normalization is derived from first principles** — the motivation is entirely independent of traditional explanations such as internal covariate shift.

3. **Solution 2: A Non-Normalizing Alternative**:

    - Rather than dividing by the magnitude, a novel mapping is introduced to eliminate the divergence.
    - The resulting function is not scale-invariant, distinguishing it from all conventional normalization methods.
    - In experiments, this alternative **outperforms** BatchNorm, LayerNorm, and related methods.

4. **PatchNorm (Convolutional Extension)**:

    - The affine divergence analysis is extended to convolutional layers, revealing a patchwise divergence that varies across spatial locations.
    - PatchNorm is proposed as a "compositionally inseparable" normalization — one that cannot be decomposed into a product of channel-wise and spatial normalizations.
    - This constitutes an entirely new family of normalization functions.

### Loss & Training
- Pure theoretical derivation combined with empirical validation.
- Comparisons against multiple normalization methods on CIFAR-10/100 and ImageNet subsets.
- An auxiliary hypothesis is tested: if the affine divergence mechanism holds, the proposed normalizer's performance should be negatively correlated with batch size.

## Key Experimental Results

### Main Results

| Method | CIFAR-10↑ | CIFAR-100↑ | Scale-Invariant? |
|--------|-----------|------------|-----------------|
| No Normalization | Baseline | Baseline | — |
| BatchNorm | +X% | +X% | Yes |
| LayerNorm | +X% | +X% | Yes |
| **Solution 2 (Non-Normalizing)** | **Surpasses BN/LN** | **Surpasses BN/LN** | **No** |

### Auxiliary Hypothesis Validation

| Prediction | Verification Result |
|------------|-------------------|
| Proposed normalizer performance should be negatively correlated with batch size | **Confirmed** — supports the affine divergence mechanism |
| Scale-invariance is not a necessary condition for success | **Confirmed** — the non-scale-invariant Solution 2 is also effective |

### Key Findings
- **Normalization derived from first principles**: without assuming any statistical regularization motive, the BatchNorm form emerges naturally from an update-alignment perspective alone.
- **The non-normalizing alternative is effective**: this challenges the assumption that scale-invariance is the key to normalization's success.
- **Negative correlation with batch size** corroborates the affine divergence mechanism, providing evidence independent of traditional explanations.
- **PatchNorm is a genuinely novel convolutional normalization form**: compositionally inseparable and theory-driven.

## Highlights & Insights
- **Reconstructing normalization theory from an update-alignment perspective** is the paper's central contribution — it connects the seemingly unrelated problem of parameter–activation update misalignment to the empirical success of normalization, offering a fundamentally new theoretical lens.
- **The existence of a non-normalizing solution** carries deep implications for deep learning architecture design — perhaps normalization itself is not necessary; rather, any mechanism that eliminates the affine divergence may suffice, and such mechanisms need not be restricted to normalizing forms.
- **Normalization as activation function?** An appendix argues that the boundary between normalizers and activation functions should dissolve — both are parameterized nonlinear mappings — a perspective worth sustained attention.

## Limitations & Future Work
- Single-layer and first-order approximations are used — divergence analysis across multiple layers would be more complex but more accurate.
- Experimental scale is limited — validation on large-scale Transformers and LLMs is needed.
- The explicit functional form of the non-normalizing alternative is not fully specified; the paper prioritizes theoretical derivation.
- The relationship to natural gradient descent is discussed but not empirically compared.
- PatchNorm is validated only in convolutional settings and has not been extended to attention mechanisms.

## Related Work & Insights
- **vs. Natural Gradient (Amari)**: Both address suboptimality of gradient directions, but natural gradient operates in the output function space (computationally intractable), whereas this paper operates in the per-layer activation space (computationally efficient).
- **vs. BatchNorm (Ioffe & Szegedy)**: BN is motivated by "internal covariate shift"; this paper starts from "update alignment" yet derives the same operation — providing independent theoretical support.
- **vs. LayerNorm/GroupNorm**: These are variants of BN; the analytical framework presented here offers a unified explanation for the success of all normalization methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Derives normalization from first principles, discovers a non-normalizing alternative, and achieves exceptional conceptual depth.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments are limited in scale (CIFAR-level); larger-scale validation is needed.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are rigorous, though notation density is high.
- Value: ⭐⭐⭐⭐⭐ — A fundamental contribution to normalization theory; PatchNorm is a practically promising novel method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[NeurIPS 2025\] Stable Coresets via Posterior Sampling: Aligning Induced and Full Loss Landscapes](../../NeurIPS2025/optimization/stable_coresets_via_posterior_sampling_aligning_induced_and_full_loss_landscapes.md)
- [\[AAAI 2026\] Beyond the Mean: Fisher-Orthogonal Projection for Natural Gradient Descent in Large Batch Training](../../AAAI2026/optimization/beyond_the_mean_fisher-orthogonal_projection_for_natural_gradient_descent_in_lar.md)

</div>

<!-- RELATED:END -->
