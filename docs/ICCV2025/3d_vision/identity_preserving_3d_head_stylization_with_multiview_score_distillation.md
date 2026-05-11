---
title: >-
  [Paper Note] Identity Preserving 3D Head Stylization with Multiview Score Distillation
description: >-
  [3D Vision] This paper proposes a 3D head stylization framework based on Likelihood Distillation (LD), achieving high-quality stylization with identity preservation under 360-degree consistent rendering through multiview…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 764ab1bb67ca7565
---

# Identity Preserving 3D Head Stylization with Multiview Score Distillation

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2411.13536](https://arxiv.org/abs/2411.13536)
- **Code**: Not released
- **Area**: 3D Vision
- **Keywords**: 3D head stylization, identity preservation, score distillation, GAN, PanoHead

## TL;DR

This paper proposes a 3D head stylization framework based on Likelihood Distillation (LD), achieving high-quality stylization with identity preservation under 360-degree consistent rendering through multiview grid scoring, mirror gradients, and rank-weighted score tensors.

## Background & Motivation

3D head stylization aims to transform real human faces into artistic style representations, with broad applications in gaming and virtual reality. Existing methods are primarily built on EG3D, which can only synthesize near-frontal views and relies on Score Distillation Sampling (SDS) for text-guided training.

**Core Problem**:

**Limited viewpoints**: EG3D can only synthesize near-frontal views, making 360-degree consistent stylization difficult.

**Identity loss**: SDS-guided GAN fine-tuning leads to mode collapse, causing different inputs to produce similar outputs (e.g., different faces converging to the same face under the Joker style).

**Insufficient diversity**: Existing methods such as StyleGANFusion and DiffusionGAN3D introduce regularization, yet identity preservation remains poor.

## Method

### Overall Architecture

The framework performs domain-adaptive fine-tuning on PanoHead (a 360-degree consistent 3D-aware GAN), with four core innovations:

### 1. Likelihood Distillation (LD) as a Replacement for SDS

Unlike SDS, which optimizes the reverse KL divergence, LD directly optimizes the negative log-likelihood:

$$\nabla_\theta \mathcal{L}_{\text{LD}} = -\mathbb{E}_{\pi, x_t}\{\nabla_{x_t} \log p(x_t^\pi | y) \frac{\partial x_t^\pi}{\partial x_0^\pi} \frac{\partial x_0^\pi}{\partial \theta}\}$$

**Key distinction between LD and SDS**: SDS subtracts the true noise $\epsilon$ from the estimated noise $\hat{\epsilon}$, making it inherently mode-seeking; it requires a high CFG weight to avoid divergence, resulting in blurry outputs. LD does not use $\epsilon$ and is diversity-seeking, requiring no high CFG weight. It is better suited to leverage GAN priors and produces sharp, diverse results.

### 2. Rank-Weighted Score Tensor

SVD decomposition is applied to the score tensor along the VAE latent channels (4-dimensional):

$$\mathbf{U}\mathbf{\Sigma}\mathbf{V}^T = \text{SVD}(\nabla_\theta \log p(x_0^\pi | y))$$

The singular values are re-weighted using linearly decaying coefficients: $\mathbf{W} = \text{diag}(1, 0.75, 0.5, 0.25)$

$$\nabla_\theta \log \tilde{p}(x_0^\pi | y) = \mathbf{U}\mathbf{W}\mathbf{\Sigma}\mathbf{V}^T$$

**Insight**: The rank-1 component captures the majority of stylization information, while lower-rank components introduce undesirable color shifts (e.g., hue artifacts in hair and ear regions).

### 3. Mirror Gradient Extension

This component exploits the yaw-symmetry prior of human heads. Given that $\pi$ and $\pi'$ are yaw-symmetric camera matrices, $x_t^\pi = \mathbf{M}(x_t^{\pi'})$ (where $\mathbf{M}$ denotes vertical flipping):

$$\nabla_\theta \mathcal{L}_{\text{LD}} = -\mathbb{E}_{\pi, x_t}\{\nabla_{x_t} \log p(x_t^\pi | y) \sqrt{\bar{\alpha_t}}(\frac{\partial x_0^\pi}{\partial \theta} + \mathbf{M}\frac{\partial x_0^{\pi'}}{\partial \theta})\}$$

A single score estimate is applied to the mirrored view, with gradients flipped before backpropagation, preventing divergence caused by inconsistent score directions across the two views.

### 4. Multiview Grid Distillation

Four 256×256 renderings from different viewpoints are arranged into a 2×2 grid (total 512×512), conditioned on depth via ControlNet:
- The denoising UNet can associate cross-view rendering consistency.
- Grid gradients are propagated only up to the super-resolution (SR) layer (i.e., the 64×64 renderer output), avoiding blurring and oversaturation caused by resolution mismatch.

### Loss & Training

The overall framework updates PanoHead parameters $\theta$ via CFG-guided LD gradients. No additional loss functions are required; all guidance signals are derived from score function estimation.

## Experiments

### Main Results

| Method | Pixar FID↓ | Pixar CLIP↑ | Pixar ID↑ | Joker FID↓ | Joker CLIP↑ | Joker ID↑ |
|------|------|------|------|------|------|------|
| StyleCLIP | High | Low | Low | High | Low | Low |
| StyleGAN-NADA | High | Low | Low | High | Low | Low |
| StyleGANFusion | Mid | Mid | Low | Mid | Mid | Low |
| DiffusionGAN3D | Mid | Mid | Low | Mid | Mid | Low |
| **Ours** | **Lowest** | **Highest** | **Highest** | **Lowest** | **Highest** | **Highest** |

The proposed method achieves the best FID, CLIP, and identity preservation scores across all five styles: Pixar, Joker, Werewolf, Sketch, and Statue.

### Ablation Study

| Configuration | Effect |
|------|------|
| LD (full-rank) | Color artifacts appear (especially in hair and ear regions) |
| LD (weighted-rank) | Color shifts eliminated; provides a strong baseline |
| + Mirror gradients | Enhances stylization of 3D-aware features such as glasses |
| + Grid denoising | Further improves multi-view style consistency |
| Grid without skipping SR | Blurring, oversaturation, and color shifts observed |
| Grid with SR skipped | Stylization quality significantly improved |

### Key Findings
1. LD outperforms SDS for GAN domain adaptation, producing sharper and more diverse results.
2. Rank weighting effectively eliminates color interference across VAE latent channels.
3. The SR network plays a critical role in stylized GANs; grid distillation should avoid propagating gradients through the SR layer.

## Highlights & Insights

1. **First to reveal the essential difference between LD and SDS in the GAN setting**: The mode-seeking vs. diversity-seeking perspective explains the root cause of diversity loss induced by SDS.
2. **Rank weighting introduces a novel mechanism for distillation control**: SVD analysis of the score tensor's frequency structure enables fine-grained control over style signals.
3. **Mirror gradients elegantly exploit facial symmetry priors**: Cross-view consistency is achieved without training a multiview diffusion model.

## Limitations & Future Work

- Applicable only to human head stylization; generalization to arbitrary 3D objects is non-trivial.
- Performance depends on the pre-training quality of PanoHead.
- Rank-weighting coefficients are empirically determined and may require adjustment for different latent spaces.

## Related Work & Insights

- **3D generators**: EG3D, PanoHead
- **Domain adaptation**: StyleGAN-NADA, StyleGANFusion, DiffusionGAN3D
- **Distillation methods**: SDS (DreamFusion), PlacidDreamer

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of LD, rank weighting, and mirror gradients is highly creative)
- Technical depth: ⭐⭐⭐⭐⭐ (Rigorous mathematical derivations with deep insights)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive quantitative and qualitative evaluation across multiple styles)
- Value: ⭐⭐⭐ (Limited to human heads, restricting the scope of application)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MemoryTalker: Personalized Speech-Driven 3D Facial Animation via Audio-Guided Stylization](memorytalker_personalized_speech-driven_3d_facial_animation_via_audio-guided_sty.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[ICCV 2025\] SplatTalk: 3D VQA with Gaussian Splatting](splattalk_3d_vqa_with_gaussian_splatting.md)
- [\[ICCV 2025\] GaussianProperty: Integrating Physical Properties to 3D Gaussians with LMMs](gaussianproperty_integrating_physical_properties_to_3d_gaussians_with_lmms.md)

</div>

<!-- RELATED:END -->
