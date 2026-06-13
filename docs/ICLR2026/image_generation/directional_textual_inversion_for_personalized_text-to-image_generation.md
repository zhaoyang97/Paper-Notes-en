---
title: >-
  [Paper Note] Directional Textual Inversion for Personalized Text-to-Image Generation
description: >-
  [ICLR 2026][Image Generation][Textual Inversion] This paper identifies a norm inflation problem in token embeddings learned by Textual Inversion (TI)…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Textual Inversion"
  - "Directional Optimization"
  - "Hypersphere"
  - "von Mises-Fisher"
  - "Personalized Text-to-Image"
date: 2026-05-08
content_hash: b8cfd6bffcdfe4ec
---

# Directional Textual Inversion for Personalized Text-to-Image Generation

**Conference**: ICLR 2026
**arXiv**: [2512.13672](https://arxiv.org/abs/2512.13672)  
**Code**: [https://github.com/kunheek/dti](https://github.com/kunheek/dti)  
**Area**: Diffusion Models / Personalized Generation
**Keywords**: Textual Inversion, Directional Optimization, Hypersphere, von Mises-Fisher, Personalized Text-to-Image

## TL;DR

This paper identifies a norm inflation problem in token embeddings learned by Textual Inversion (TI), which degrades text alignment under complex prompts. The proposed Directional Textual Inversion (DTI) fixes the embedding norm at an in-distribution scale and optimizes only the direction on the unit hypersphere via Riemannian SGD, regularized by a von Mises-Fisher prior, substantially improving prompt faithfulness.

## Background & Motivation

**Background**: Personalized text-to-image generation has two main paradigms—parameter fine-tuning (e.g., DreamBooth) and embedding optimization (e.g., Textual Inversion). TI, by optimizing only token embeddings, offers compact storage and easy integration, making it a foundational component for many subsequent methods.

**Limitations of Prior Work**: TI performs poorly under complex prompts—e.g., "A painting of \<dog\> wearing a santa hat" may generate the dog while ignoring the hat and background details. The root cause is that embedding norms inflate to extreme values during TI optimization (>20, compared to ~0.4 for normal vocabulary tokens).

**Key Challenge**: Semantic information is primarily encoded in the **direction** of embeddings (cosine similarity is semantically consistent, whereas Euclidean distance is not), yet TI imposes no norm constraint, leading to: (a) large norms suppressing positional encoding information ($\mathcal{O}(1/m)$) in pre-norm Transformers; and (b) residual updates stagnating, preventing subsequent layers from effectively modifying hidden state directions.

**Goal**: To resolve the text alignment failures caused by norm inflation while preserving TI's lightweight advantages.

**Key Insight**: The authors analyze the geometric structure of the CLIP token embedding space and establish through both empirical and theoretical evidence that "direction encodes semantics and norm inflation is harmful." This constitutes an interpretability-driven analytical perspective.

**Core Idea**: Fix the embedding norm at an in-distribution scale, optimize only the direction on the unit hypersphere, and regularize with a vMF prior.

## Method

### Overall Architecture

DTI decouples the token embedding $\bm{e} \in \mathbb{R}^d$ into a norm $m^\star$ and a direction $\bm{v} \in \mathbb{S}^{d-1}$, i.e., $\bm{e} = m^\star \bm{v}$. The norm is fixed to the mean norm of pretrained vocabulary embeddings, and only the direction $\bm{v}$ is optimized. Optimization is performed on the unit hypersphere using Riemannian SGD, with a von Mises-Fisher (vMF) directional prior as regularization.

### Key Designs

1. **Hypersphere Directional Optimization (Riemannian SGD)**:

    - Function: Optimizes the embedding direction on $\mathbb{S}^{d-1}$, preventing norm inflation.
    - Mechanism: The Euclidean gradient is first projected onto the tangent space $\bm{g} = \bm{g}_{\text{euc}} - (\bm{v}_k^\top \bm{g}_{\text{euc}})\bm{v}_k$, then retracted back to the sphere via $\bm{v}_{k+1} = \frac{\bm{v}_k - \eta \bm{g}}{\|\bm{v}_k - \eta \bm{g}\|}$. Gradient normalization $\bm{g}' = \bm{g}/\|\bm{g}\|$ is also applied.
    - Design Motivation: Euclidean AdamW causes parameters to drift off the manifold and is unsuitable for spherical constraints. RSGD respects the manifold geometry; ablation studies confirm its superiority over AdamW with projection.

2. **von Mises-Fisher (vMF) Directional Prior**:

    - Function: Frames directional optimization as MAP estimation, introducing a vMF distribution as a prior to prevent semantic drift.
    - Mechanism: $p(\bm{v}|\bm{\mu}, \kappa) \propto \exp(\kappa \bm{\mu}^\top \bm{v})$, where $\bm{\mu}$ is the normalized embedding of the corresponding category word (e.g., 'dog'). The negative log-prior gradient is the constant $-\kappa\bm{\mu}$, which is directly added to the data gradient.
    - Design Motivation: Analogous to decoupled weight decay but adapted to the sphere. $\kappa$ is fixed at 1e-4, incurring negligible computational overhead.

3. **Norm Scale Selection**:

    - Function: Fixes $m^\star$ to the mean norm of pretrained vocabulary embeddings.
    - Design Motivation: Ablation studies show that using the minimum norm collapses subject similarity, while OOD large norms degrade text alignment; the mean norm is optimal.

### Loss & Training

The data loss is the standard diffusion denoising MSE: $\mathcal{L}_{\text{data}}(m^\star \bm{v}) = \mathbb{E}[\|\bm{\epsilon} - \bm{\epsilon}_\theta(\bm{z}_t, t, c(m^\star \bm{v}))\|^2]$. The prior loss is $\mathcal{L}_{\text{prior}} = -\kappa \bm{\mu}^\top \bm{v}$, and the total loss is their sum. Training takes approximately 7 minutes per concept (SDXL, single A6000 GPU).

## Key Experimental Results

### Main Results

| Model | Method | Image Sim (DINOv2) | Text Sim (SigLIP) |
|------|------|------|------|
| SDXL | TI | 0.561 | 0.292 |
| SDXL | TI-rescaled | 0.243 | 0.466 |
| SDXL | CrossInit | 0.545 | 0.464 |
| SDXL | **DTI (ours)** | **0.450** | **0.522** |
| SANA 1.5-1.6B | TI | 0.480 | 0.621 |
| SANA 1.5-1.6B | **DTI (ours)** | **0.479** | **0.744** |
| SANA 1.5-4.8B | TI | 0.446 | 0.646 |
| SANA 1.5-4.8B | **DTI (ours)** | **0.452** | **0.757** |

DTI substantially improves text alignment across all models (0.292→0.522 on SDXL) while maintaining reasonable subject similarity. The advantage becomes more pronounced as model size increases.

### Ablation Study

| Optimizer | $m^\star$ | $\kappa \times 10^{-3}$ | Image | Text |
|--------|---------|---------|-------|------|
| AdamW | mean | 0.1 | 0.335 | 0.463 |
| RSGD | min | 0.1 | 0.030 | 0.074 |
| RSGD | 5.0 (OOD) | 0.1 | 0.383 | 0.373 |
| RSGD | mean | 0.0 | 0.507 | 0.436 |
| RSGD | mean | 0.5 | 0.278 | 0.688 |
| **RSGD** | **mean** | **0.1** | **0.450** | **0.522** |

### Key Findings

- RSGD significantly outperforms AdamW with projection, underscoring the importance of respecting manifold geometry.
- Setting the norm to the minimum or an OOD value yields substantially worse results; the mean norm is optimal.
- The vMF prior is indispensable ($\kappa=0$ noticeably degrades text alignment), though excessively large $\kappa$ also harms image similarity.
- In a user study (100 participants via AMT), DTI ranks first in both subject fidelity (43.45%) and text alignment (66.77%).

## Highlights & Insights

- **Rigorous Theoretical Analysis**: Starting from the mathematical structure of pre-norm Transformers, the paper formally establishes the causal chain of norm inflation → positional information attenuation + residual update stagnation (Proposition 1, Corollary 1), providing the first systematic theoretical explanation for TI's failure modes.
- **Spherical Interpolation (SLERP) Capability**: DTI's hyperspherical parameterization naturally supports smooth semantic interpolation between learned concepts (e.g., dog↔teapot, cat↔dog), which standard TI cannot achieve. This capability opens up creative applications in concept blending.
- **Minimal and Efficient**: Compared to TI, the entire method modifies only the optimization procedure—fixed norm + RSGD + constant prior gradient—with no additional networks, no extra storage, and no increase in training time.

## Limitations & Future Work

- DTI primarily improves text faithfulness and does not directly optimize subject similarity; high subject fidelity requires combining it with methods such as LoRA.
- The theoretical analysis focuses on pre-norm architectures (CLIP, Gemma); applicability to post-norm or other normalization schemes remains unknown.
- The vMF concentration parameter $\kappa$ requires manual specification; although the paper reports 1e-4 as generally applicable, adjustment may be needed for concepts of varying complexity.
- Per-concept training is still required (~7 min for SDXL), precluding zero-shot personalization.

## Related Work & Insights

- **vs. TI**: TI's unconstrained norm causes embeddings to become OOD; DTI fundamentally addresses this by fixing the norm and optimizing only the direction.
- **vs. CrossInit**: CrossInit achieves reasonable text alignment on SDXL but fails on SANA (LLM-based encoder); DTI generalizes better across architectures.
- **vs. P+/NeTI**: These methods improve TI by enriching the embedding space but introduce substantial computational overhead; DTI preserves TI's lightweight advantage.
- The directional optimization + vMF prior paradigm is transferable to VLM prompt tuning or LLM soft prompt optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ — Geometric perspective on TI failures with an elegant solution; insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple models (SDXL/SANA), complete ablations, user study, interpolation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theory–experiment–method logical chain is exceptionally clear; figures and tables are polished.
- Value: ⭐⭐⭐⭐ — High practical value; plug-and-play; broad impact on the TI ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](../../AAAI2026/image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)
- [\[CVPR 2026\] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models](../../CVPR2026/image_generation/tina_text-free_inversion_attack_for_unlearned_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](consistent_text-to-image_generation_via_scene_de-contextualization.md)

</div>

<!-- RELATED:END -->
