---
title: >-
  [Paper Note] Self-Supervised Learning from Structural Invariance
description: >-
  [ICLR 2026][Causal Inference][self-supervised learning] This paper proposes AdaSSL, which introduces latent variables to model conditional uncertainty between positive pairs…
tags:
  - "ICLR 2026"
  - "Causal Inference"
  - "self-supervised learning"
  - "latent variable model"
  - "structural invariance"
  - "heteroscedasticity"
  - "causal representation"
date: 2026-05-08
content_hash: 5ebd45e2d855ee92
---

# Self-Supervised Learning from Structural Invariance

**Conference**: ICLR 2026
**arXiv**: [2602.02381](https://arxiv.org/abs/2602.02381)
**Code**: [https://github.com/SkrighYZ/AdaSSL](https://github.com/SkrighYZ/AdaSSL)
**Area**: Self-Supervised Learning / Causal Representation Learning
**Keywords**: self-supervised learning, latent variable model, structural invariance, heteroscedasticity, causal representation

## TL;DR
This paper proposes AdaSSL, which introduces latent variables to model conditional uncertainty between positive pairs, derives a variational lower bound on mutual information, and enables SSL to handle complex (multimodal, heteroscedastic) conditional distributions in naturally paired data. AdaSSL outperforms baselines on causal representation learning, fine-grained image understanding, and video world models.

## Background & Motivation

**Background**: Joint-embedding SSL methods (e.g., SimCLR, BYOL) learn representations by encouraging similarity between positive pair embeddings, typically relying on hand-crafted data augmentations to construct semantically related pairs.

**Limitations of Prior Work**: Hand-crafted augmentations (cropping, color jitter) cannot precisely simulate real-world variation factors, may discard fine-grained information, require modality-specific heuristics, and differ from natural distribution shifts. Naturally paired data (e.g., adjacent video frames, image-text pairs) better reflects real-world variation, but introduces complex conditional distributions $p(\mathbf{z}^+|\mathbf{z})$—heteroscedastic and multimodal—which existing SSL methods cannot model.

**Key Challenge**: InfoNCE's dot-product similarity implicitly assumes a vMF distribution (isotropic noise), and AnInfoNCE extends this to anisotropic but still constant noise. However, Proposition 2.1 theoretically establishes that even when noise in the latent space is isotropic, mapping to the normalized embedding space inevitably induces heteroscedasticity—a necessary consequence of geometric mismatch.

**Goal**: Enable SSL to flexibly model arbitrarily complex conditional distributions $p(\mathbf{z}^+|\mathbf{z})$ while keeping the similarity function simple.

**Key Insight**: Inspired by JEPA, a latent variable $\mathbf{r}$ is introduced to capture predictive uncertainty, decomposing the complex conditional distribution into two steps: first sampling $\mathbf{r}$ (e.g., camera motion, actions), then predicting $\mathbf{z}^+$ with a simple model.

**Core Idea**: Via the chain rule of mutual information $I(f(\mathbf{x}); f(\mathbf{x}^+)) = I(f(\mathbf{x}), \mathbf{r}; f(\mathbf{x}^+)) - I(\mathbf{r}; f(\mathbf{x}^+)|f(\mathbf{x}))$, the first term is optimized with an extended InfoNCE (simple similarity + latent variable), and the second term is regularized with KL divergence to prevent $\mathbf{r}$ from encoding shortcuts.

## Method

### Overall Architecture

An encoder $f$ extracts embeddings, and a latent variable $\mathbf{r}$ captures uncertainty between positive pairs. An editing function $t(f(\mathbf{x}), \mathbf{r})$ modifies embeddings to bring them closer to $f(\mathbf{x}^+)$. The objective combines an SSL loss (InfoNCE or BYOL) with a regularization term that limits the information content of $\mathbf{r}$.

### Key Designs

1. **AdaSSL-V (Variational Version)**:

    - **Function**: Models the latent variable using a variational distribution $q_\phi(\mathbf{r}|\mathbf{x}, \mathbf{x}^+)$
    - **Mechanism**: $\mathcal{L} = \mathcal{L}_{SSL}(\mathbb{E}_{q_\phi} \psi_1(\mathbf{x}, \mathbf{r}), \psi_2(\mathbf{x}^+)) + \beta D_{KL}(q_\phi(\mathbf{r}|\mathbf{x}, \mathbf{x}^+) \| p_\theta(\mathbf{r}|\mathbf{x}))$, where KL regularization prevents $\mathbf{r}$ from directly encoding $f(\mathbf{x}^+)$
    - **Design Motivation**: Derives a tractable lower bound on $I(f(\mathbf{x}); f(\mathbf{x}^+))$ with theoretical rigor

2. **AdaSSL-S (Sparse Version)**:

    - **Function**: Deterministically predicts $\mathbf{r}$ and regularizes its sparsity
    - **Mechanism**: $\mathbf{r} = m(f(\mathbf{x}), f(\mathbf{x}^+))$, with differentiable L0 penalty via Gumbel-Sigmoid. The editing function adopts a modular low-rank design: $t(f(\mathbf{x}), \mathbf{r}) = f(\mathbf{x}) + \sum_i r_i (\mathbf{B}_i \mathbf{A}_i f(\mathbf{x}) + b_i)$
    - **Design Motivation**: Natural variations typically correspond to sparse changes in latent factors; the sparsity inductive bias better aligns with causal representation learning

3. **Necessity of Heteroscedasticity (Proposition 2.1)**:

    - **Function**: Theoretically proves that the conditional distribution of pairs in embedding space is necessarily heteroscedastic
    - **Mechanism**: When the latent space $\mathbb{R}^{d_z}$ is mapped to a curved manifold (e.g., unit sphere $\mathbb{S}^{d_f}$), local neighborhood distortions are position-dependent, producing location-dependent variance even when the original noise is isotropic
    - **Design Motivation**: Provides a fundamental proof of the inadequacy of standard SSL similarity functions

### Loss & Training

- AdaSSL-V: InfoNCE + KL regularization ($\beta$ controls strength)
- AdaSSL-S: InfoNCE + L0 sparsity regularization (Gumbel-Sigmoid)
- Compatible with non-contrastive methods such as BYOL

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | AdaSSL | InfoNCE | AnInfoNCE | H-InfoNCE |
|----------------|--------|--------|---------|-----------|-----------|
| Numerical Heteroscedastic (OOD) | R² | 0.92+ | <0.27 | <0.40 | 0.76 |
| 3DIdent (CRL) | DCI | 0.85+ | 0.72 | 0.74 | 0.78 |
| CelebA Fine-Grained | 40-attr Acc | Best | Lower | Lower | Moderate |
| Moving-MNIST Acceleration | R² | 0.55 (BYOL baseline 0.15) | — | — | — |

### Ablation Study

| Configuration | Numerical OOD R² | Notes |
|---------------|------------------|-------|
| AdaSSL-V | 0.92+ | Full variational version |
| AdaSSL-S | 0.90+ | Sparse version, slightly lower but sparser |
| H-InfoNCE | 0.76 | Heteroscedastic but no latent variable |
| InfoNCE | <0.27 | Baseline completely fails |
| AnInfoNCE | <0.40 | Anisotropy insufficient |

### Key Findings
- Under complex conditional distributions (multimodal + heteroscedastic), InfoNCE and AnInfoNCE completely fail (OOD R² < 0.4), while AdaSSL maintains 0.9+
- Naturally paired data (vs. standard augmentations) significantly improves downstream performance when modeled correctly
- The sparse $\mathbf{r}$ learned by AdaSSL-S aligns with ground-truth variation factors
- In video world models, AdaSSL captures stochastic acceleration, which BYOL discards

## Highlights & Insights
- The **heteroscedasticity theorem** reveals a fundamental limitation of standard SSL—not an empirical observation but a mathematical inevitability
- **Generality of latent variable modeling**: the same framework is compatible with both contrastive and distillation-based SSL, and applies across numerical, image, and video domains
- The **sparse modular editing** design ($\mathbf{r}$ controlling low-rank editing modules) is conceptually analogous to LoRA-style ideas

## Limitations & Future Work
- AdaSSL-S requires additional treatment when applied to distillation methods (e.g., BYOL)
- The latent variable dimension $d_r$ must be preset; automatic determination would be preferable
- Large-scale validation is lacking (no ImageNet-scale experiments)
- When the number of modes in multimodal conditional distributions is unknown, the choice of variational prior warrants further investigation

## Related Work & Insights
- **vs. AnInfoNCE**: The anisotropic weight $\Lambda$ is global and does not vary with data. AdaSSL achieves data-adaptive modeling through latent variables
- **vs. JEPA/V-JEPA**: JEPA assumes $\mathbf{r}$ is known (e.g., actions); AdaSSL infers $\mathbf{r}$ from data pairs
- **vs. LieSSL**: Lie group transformations assume invertible and structured changes; AdaSSL is more flexible

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Heteroscedasticity theorem + MI lower bound + dual-variant design; theoretically and methodologically deep
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation (numerical/CRL/image/video), but lacking large-scale comparisons
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical motivation is clear; the logic from theory to method to experiments flows smoothly
- Value: ⭐⭐⭐⭐ Addresses a fundamental theoretical problem in SSL with strong methodological generality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [\[ICLR 2026\] Learning Robust Intervention Representations with Delta Embeddings](learning_robust_intervention_representations_with_delta_embeddings.md)
- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](../../CVPR2026/causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[AAAI 2026\] Learning Subgroups with Maximum Treatment Effects without Causal Heuristics](../../AAAI2026/causal_inference/learning_subgroups_with_maximum_treatment_effects_without_causal_heuristics.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)

</div>

<!-- RELATED:END -->
