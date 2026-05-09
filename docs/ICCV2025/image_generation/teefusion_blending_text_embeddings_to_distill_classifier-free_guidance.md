---
title: >-
  [Paper Note] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance
description: >-
  [ICCV 2025][Image Generation][CFG Distillation] This paper proposes TeEFusion, which encodes the guidance magnitude of CFG directly as a linear combination of conditional and unconditional text embeddings to replace dual forward passes, achieving efficient CFG distillation with zero additional parameters. The method is compatible with complex teacher sampling strategies (e.g., Z-Sampling, W2SD), enabling a 6× inference speedup over the teacher model.
tags:
  - ICCV 2025
  - Image Generation
  - CFG Distillation
  - Text Embedding Fusion
  - Sampling Acceleration
  - Guidance Distillation
  - DiT
date: 2026-05-08
content_hash: c815a69dd00f5122
---

# TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance

**Conference**: ICCV 2025
**arXiv**: [2507.18192](https://arxiv.org/abs/2507.18192)
**Code**: [https://github.com/AIDC-AI/TeEFusion](https://github.com/AIDC-AI/TeEFusion)
**Area**: Diffusion Models / Image Generation / Distillation
**Keywords**: CFG Distillation, Text Embedding Fusion, Sampling Acceleration, Guidance Distillation, DiT

## TL;DR

This paper proposes TeEFusion, which encodes the guidance magnitude of CFG directly as a linear combination of conditional and unconditional text embeddings to replace dual forward passes, achieving efficient CFG distillation with zero additional parameters. The method is compatible with complex teacher sampling strategies (e.g., Z-Sampling, W2SD), enabling a 6× inference speedup over the teacher model.

## Background & Motivation

**The cost of CFG**: Classifier-Free Guidance is essential for generation quality in current T2I models (SD3, FLUX), but requires two forward passes (conditional + unconditional), doubling inference cost. More complex sampling strategies (e.g., Z-Sampling + CFG requiring 6× overhead) further exacerbate this issue.

**Limitations of Prior Work**:
- Methods such as DistillCFG require additional MLPs to encode the guidance scale, increasing architectural complexity.
- Existing methods are restricted to scenarios where the teacher and student use identical sampling algorithms, and cannot distill complex sampling strategies.
- Sufficient validation on large-scale SOTA models (e.g., SD3) is lacking.

**Key Findings**: Addition and subtraction of embeddings from different prompts in the text embedding space can effectively blend or eliminate specific semantic components, providing a theoretical basis for moving the linear combination of CFG into the embedding space.

## Method

### Core Idea

The standard CFG formulation is:

$$\tilde{\epsilon}_\theta(x_t, c) = (1+w)\epsilon_\theta(x_t, c) - w\epsilon_\theta(x_t)$$

which requires two forward passes. TeEFusion moves this linear combination from the model output space into the text embedding space, requiring only a single forward pass:

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \hat{c})$$

where $\hat{c} = (1+w)c - w\varnothing = c + w(c - \varnothing)$.

### Embedding Fusion Formula

In DiT models, the joint embedding of timestep and text is $z_{t,c} = \mathcal{G}(\psi(t)) + \mathcal{F}(c)$. TeEFusion encodes the guidance scale $w$ as:

$$\hat{z}_{t,c,\varnothing,w} = \mathcal{G}(\psi(t)) + \mathcal{F}(c) + \underbrace{\mathcal{G}(\psi(w))\mathcal{F}(c - \varnothing)}_{\text{extra term}}$$

Key design considerations:
- A sinusoidal embedding $\psi(w)$ projects the scalar $w$ into a vector space, avoiding the $\mathcal{O}(w^2)$ variance issue.
- In the extra term, $\mathcal{G}(\psi(w))$ acts as an element-wise scaling factor on $\mathcal{F}(c-\varnothing)$, precisely corresponding to the semantics of $w \cdot (c-\varnothing)$.
- **No new parameters are introduced**: the existing $\mathcal{G}$ and $\mathcal{F}$ networks are fully reused.

### Distillation Objective

$$L_{\text{distill}} = \|\epsilon_{\theta_S}(x_t, \hat{z}_{t,c,\varnothing,w}) - \tilde{\epsilon}_{\theta_T}(x_t, w, c)\|_2^2$$

- The student model takes the fused embedding $\hat{z}$ as input and performs a single forward pass.
- The teacher model uses standard CFG (or advanced strategies such as W2SD) as the training target.
- The guidance scale $w$ is uniformly sampled from $[w_{\min}, w_{\max}]$ at each iteration.
- Any complex teacher sampling strategy (Z-Sampling, W2SD) is supported.

## Key Experimental Results

### HPS Aesthetic Score Comparison

| Model | Method | Cost | Anime | Concept-Art | Paintings | Photo |
|-------|--------|------|-------|-------------|-----------|-------|
| SD3 | CFG | 2× | 30.78 | 30.06 | 30.28 | 27.93 |
| SD3 | W2SD+CFG | 6× | 31.96 | 30.65 | 30.67 | 29.76 |
| SD3 | DistillCFG | 1× | 31.14 | 29.52 | 30.03 | 29.04 |
| SD3 | **TeEFusion** | **1×** | **32.37** | **30.88** | **30.74** | **29.84** |

### Ablation Study & Training Efficiency

| Component | Effect |
|-----------|--------|
| Remove $\mathcal{G}(\psi(w))$ scaling | Significant HPS drop; guidance scale becomes uncontrollable |
| Direct linear combination (without sin/cos encoding) | Numerical instability at large $w$ values |
| Training convergence | Converges in ~10K steps; training is efficient |

### Key Findings

- TeEFusion surpasses the teacher model's CFG results across all aesthetic dimensions while requiring only 1/6 of the inference cost.
- It also leads on DPG-Bench compositional object and prompt-following evaluations.
- The method is architecturally minimal—the student model is identical to the original, facilitating deployment.
- A student distilled from a W2SD+CFG teacher even exceeds teacher quality, validating the feasibility of test-time scaling distillation.

## Highlights & Insights

1. **Minimal design**: Zero additional parameters; a single extra term eliminates the need for dual forward passes.
2. **Sampling strategy distillation**: The first work to distill complex reflective sampling strategies into simple Euler sampling.
3. **Solid empirical validation**: Thoroughly validated on industrial-scale large models (SD3, in-house T2I).

## Limitations & Future Work

- The linear fusion assumption in the text embedding space may not hold at extreme guidance scales.
- Validation is currently limited to the DiT architecture; applicability to U-Net architectures remains unexplored.
- Distillation still requires the teacher model to generate training targets, and offline training costs are non-negligible.

## Related Work & Insights

- **Guidance Distillation**: DistillCFG, DICE, Progressive Distillation
- **Test-Time Scaling**: Z-Sampling, W2SD, RePaint
- **Advanced Samplers**: DPM-Solver, DDIM

## Rating

- Novelty: ⭐⭐⭐⭐ — The embedding-space fusion idea is concise and elegant.
- Technical Depth: ⭐⭐⭐⭐ — Provides in-depth analysis of the CFG mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated on industrial-scale models.
- Practical Value: ⭐⭐⭐⭐⭐ — 6× speedup, zero parameter increase, easy to deploy.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[ICCV 2025\] GAP: Gaussianize Any Point Clouds with Text Guidance](gap_gaussianize_any_point_clouds_with_text_guidance.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)

<!-- RELATED:END -->
