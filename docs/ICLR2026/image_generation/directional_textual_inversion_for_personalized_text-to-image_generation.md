---
title: >-
  [Paper Note] Directional Textual Inversion for Personalized Text-to-Image Generation
description: >-
  [ICLR 2026][Image Generation][Textual Inversion] This paper discovers that token embeddings learned by Textual Inversion (TI) suffer from "norm inflation," leading to decreased text alignment in complex prompts. It proposes Directional Textual Inversion (DTI), which fixes the embedding norm to an in-distribution scale and optimizes only the direction on the unit hype
tags:
  - ICLR 2026
  - Image Generation
  - Textual Inversion
  - von Mises-Fisher
date: 2026-05-08
content_hash: c8cc84924bebcf17
---
# Directional Textual Inversion for Personalized Text-to-Image Generation

**Conference**: ICLR 2026  
**arXiv**: [2512.13672](https://arxiv.org/abs/2512.13672)  
**Code**: [https://github.com/kunheek/dti](https://github.com/kunheek/dti)  
**Area**: Diffusion Models / Personalized Generation  
**Keywords**: Textual Inversion, Directional Optimization, Hypersphere, von Mises-Fisher, Personalized Text-to-Image  

## TL;DR

This paper discovers that token embeddings learned by Textual Inversion (TI) suffer from "norm inflation," leading to decreased text alignment in complex prompts. It proposes Directional Textual Inversion (DTI), which fixes the embedding norm to an in-distribution scale and optimizes only the direction on the unit hypersphere using Riemannian SGD. Combined with a von Mises-Fisher prior, this method significantly improves prompt faithfulness.

## Background & Motivation

**Background**: Personalized text-to-image generation follows two main paradigms: parameter fine-tuning (e.g., DreamBooth) and embedding optimization (e.g., Textual Inversion). TI is a fundamental component for many subsequent methods due to its small storage footprint and ease of integration.

**Limitations of Prior Work**: TI performs poorly with complex prompts—for example, in "A painting of \<dog\> wearing a santa hat," the model might generate the dog but ignore the hat and background details. The root cause is that embedding norms inflate to extreme values (>20, compared to ~0.4 for normal vocabulary) during TI optimization.

**Key Challenge**: Semantic information is primarily encoded in the **direction** of embeddings (cosine similarity maintains semantic consistency, whereas Euclidean distance does not). However, TI does not constrain the norm, leading to: (a) large norms suppressing positional encoding information in pre-norm Transformers ($\mathcal{O}(1/m)$); (b) stagnation of residual updates, preventing subsequent layers from effectively modifying the hidden state direction.

**Goal**: To resolve the text alignment failures caused by norm inflation while maintaining the lightweight advantages of TI.

**Key Insight**: Starting from the geometric structure of the CLIP token embedding space, the authors prove through both experiments and theory that "direction encodes semantics, while norm inflation is harmful." This is an interpretability-driven analytical perspective.

**Core Idea**: Fix the embedding norm to an in-distribution scale and optimize only the direction on the unit hypersphere, regularized by a vMF prior.

## Method

### Overall Architecture

DTI addresses the chronic norm inflation issue of TI: since semantics are encoded in directions and norm inflation degrades text alignment, the norm is excluded from optimization. Specifically, the token embedding $\bm{e} \in \mathbb{R}^d$ is decomposed into "norm × direction" as $\bm{e} = m^\star \bm{v}$, where the direction $\bm{v} \in \mathbb{S}^{d-1}$ is a point on the unit hypersphere. Throughout training, the norm $m^\star$ is fixed to the mean norm of the pre-trained vocabulary embeddings, allowing only the direction $\bm{v}$ to move on the sphere. The optimizer is replaced with Riemannian SGD, which respects spherical geometry, and a von Mises-Fisher directional prior is applied to pull $\bm{v}$ toward the category word. Compared to standard TI, it only modifies "how to optimize" without adding networks or increasing storage.

### Key Designs

**1. Hyperspherical Directional Optimization (Riemannian SGD): Constraining optimization to the sphere to prevent norm inflation at the source**

The fundamental problem with TI is the use of Euclidean AdamW to freely optimize the entire embedding, causing parameters to drift away from the reasonable manifold and norms to inflate beyond 20. DTI restricts movement to the unit sphere $\mathbb{S}^{d-1}$: in each step, the Euclidean gradient is first projected onto the tangent space of the current point by removing the radial component $\bm{g} = \bm{g}_{\text{euc}} - (\bm{v}_k^\top \bm{g}_{\text{euc}})\bm{v}_k$. The tangent gradient is then normalized $\bm{g}' = \bm{g}/\|\bm{g}\|$, and a retraction maps the update back onto the sphere:

$$\bm{v}_{k+1} = \frac{\bm{v}_k - \eta \bm{g}}{\|\bm{v}_k - \eta \bm{g}\|}$$

This ensures the norm remains $m^\star$ and the direction stays on the manifold. Ablations comparing "AdamW + projection" and RSGD show that the former yields significantly worse text alignment, indicating that respecting manifold geometry is just as crucial as fixing the norm.

**2. von Mises-Fisher (vMF) Directional Prior: Anchoring directions to prevent semantic drift**

Simply constraining the direction to the sphere is insufficient, as it may still drift far from the category word. DTI treats directional optimization as MAP estimation and introduces a vMF distribution as a prior $p(\bm{v}\mid\bm{\mu}, \kappa) \propto \exp(\kappa \bm{\mu}^\top \bm{v})$, where the mean direction $\bm{\mu}$ is the normalized embedding of the corresponding category word (e.g., 'dog'), and concentration $\kappa$ controls the anchor strength. Implementation is simple: the gradient of the negative log prior with respect to $\bm{v}$ is a constant $-\kappa\bm{\mu}$, which is added to the data gradient during training. This approach is similar to decoupled weight decay but adapted for spheres. $\kappa$ is fixed at 1e-4 with almost zero overhead.

**3. Norm Scale Selection: Using an "in-distribution" mean norm**

Since the norm is fixed, its value is critical. DTI sets $m^\star$ to the mean norm of all embeddings in the pre-trained vocabulary (approximately 0.4), ensuring inverted concepts reside on the same scale as normal words. Ablations verify this: using the minimum vocabulary norm causes subject similarity to collapse (Image Sim drops to 0.030), while an OOD large norm (e.g., 5.0) reduces text alignment. Only the mean norm balances both metrics.

### Loss & Training

The data loss is the standard diffusion denoising MSE $\mathcal{L}_{\text{data}}(m^\star \bm{v}) = \mathbb{E}[\|\bm{\epsilon} - \bm{\epsilon}_\theta(\bm{z}_t, t, c(m^\star \bm{v}))\|^2]$. The prior loss is $\mathcal{L}_{\text{prior}} = -\kappa \bm{\mu}^\top \bm{v}$. Training takes approximately 7 minutes per concept (SDXL on a single A6000), which is comparable to original TI.

## Key Experimental Results

### Main Results

| Model | Method | Image Sim (DINOv2) | Text Sim (SigLIP) |
|------|------|------|------|
| SDXL | TI | 0.561 | 0.292 |
| SDXL | TI-rescaled | 0.243 | 0.466 |
| SDXL | CrossInit | 0.545 | 0.464 |
| SDXL | **Ours** | **0.450** | **0.522** |
| SANA 1.5-1.6B | TI | 0.480 | 0.621 |
| SANA 1.5-1.6B | **Ours** | **0.479** | **0.744** |
| SANA 1.5-4.8B | TI | 0.446 | 0.646 |
| SANA 1.5-4.8B | **Ours** | **0.452** | **0.757** |

DTI significantly improves text alignment across all models (0.292 → 0.522 on SDXL) while maintaining reasonable subject similarity. The advantage becomes more pronounced as the model size increases.

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

- RSGD significantly outperforms AdamW with projection, proving the importance of manifold geometry.
- Setting the norm to minimum or OOD values performs poorly; the mean is optimal.
- The vMF prior is essential (text alignment drops without it, $\kappa=0$), though excessively large $\kappa$ harms image similarity.
- In a user study (100 AMT participants), DTI ranked first in both subject faithfulness (43.45%) and text alignment (66.77%).

## Highlights & Insights

- **Solid Theoretical Analysis**: Based on the mathematical structure of pre-norm Transformers, the paper proves a causal chain: norm inflation → positional information decay + residual update stagnation (Proposition 1, Corollary 1). This is the first systematic theoretical explanation for TI failure modes.
- **Spherical Interpolation (SLERP) Capability**: The hyperspherical parameterization of DTI naturally supports smooth semantic interpolation between learned concepts (e.g., dog ↔ teapot, cat ↔ dog), which standard TI cannot achieve. This enables creative concept-blending applications.
- **Minimalist and Efficient**: Compared to TI, the method only modifies the optimization process—fixed norm + RSGD + constant prior gradient—requiring no extra networks or storage, with no increase in training time.

## Limitations & Future Work

- DTI primarily improves text faithfulness and does not directly optimize subject similarity; high subject fidelity may still require methods like LoRA.
- Theoretical analysis focuses on pre-norm architectures (CLIP, Gemma); applicability to post-norm or other normalization schemes is unknown.
- The $\kappa$ parameter for the vMF prior requires manual setting; although 1e-4 is generally effective, different concept complexities might require adjustment.
- It still requires individual training for each concept (~7min for SDXL), making it unsuitable for zero-shot personalization.

## Related Work & Insights

- **vs TI**: TI's lack of norm constraints leads to OOD embeddings; DTI fundamentally solves this via fixed norm and directional optimization.
- **vs CrossInit**: CrossInit achieves good text alignment on SDXL but fails on SANA (LLM-based encoder), whereas DTI generalizes better across architectures.
- **vs P+/NeTI**: These methods improve TI by exploring richer embedding spaces but introduce significant computational overhead. DTI maintains the lightweight advantage of TI.
- The directional optimization and vMF prior approach can be transferred to VLM prompt tuning or LLM soft prompt optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ Deep insights into TI failure from a geometric perspective with a concise solution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models (SDXL/SANA), complete ablations, user study, and interpolation experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theory-experiment-method logical chain with high-quality visualizations.
- Value: ⭐⭐⭐⭐ High practical value, plug-and-play, with broad implications for the TI ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift](preserve_and_personalize_personalized_text-to-image_diffusion_models_without_dis.md)
- [\[AAAI 2026\] DOS: Directional Object Separation in Text Embeddings for Multi-Object Image Generation](../../AAAI2026/image_generation/dos_directional_object_separation_in_text_embeddings_for_mul.md)
- [\[ICLR 2026\] MOSAIC: Multi-Subject Personalized Generation via Correspondence-Aware Alignment and Disentanglement](mosaic_multi-subject_personalized_generation_via_correspondence-aware_alignment_.md)
- [\[ECCV 2024\] Textual-Visual Logic Challenge: Understanding and Reasoning in Text-to-Image Generation](../../ECCV2024/image_generation/textual-visual_logic_challenge_understanding_and_reasoning_in_text-to-image_gene.md)
- [\[CVPR 2026\] Premier: Personalized Preference Modulation with Learnable User Embedding in Text-to-Image Generation](../../CVPR2026/image_generation/premier_personalized_preference_modulation_with_learnable_user_embedding_in_text.md)

</div>

<!-- RELATED:END -->
