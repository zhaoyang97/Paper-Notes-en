---
title: >-
  [Paper Note] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)
description: >-
  [ICLR 2026][Image Generation][Object-Centric Learning] This paper proposes CODA, a framework that addresses slot entanglement and weak alignment in diffusion-based object-centric learning by introducing register slots to…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Object-Centric Learning"
  - "Slot Attention"
  - "Register Slots"
  - "contrastive learning"
  - "compositional generation"
date: 2026-05-08
content_hash: 6e7626f32e7cb05a
---

# Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)

**Conference**: ICLR 2026
**arXiv**: [2601.01224](https://arxiv.org/abs/2601.01224)  
**Code**: [GitHub](https://github.com/sony/coda)  
**Area**: Object-Centric Learning / Diffusion Models
**Keywords**: Object-Centric Learning, Slot Attention, Register Slots, contrastive learning, compositional generation

## TL;DR

This paper proposes CODA, a framework that addresses slot entanglement and weak alignment in diffusion-based object-centric learning by introducing register slots to absorb residual attention, fine-tuning cross-attention projections, and applying a contrastive alignment loss. CODA achieves substantial improvements in object discovery and compositional generation quality on both synthetic and real-world datasets.

## Background & Motivation

Object-Centric Learning (OCL) aims to decompose complex scenes into structured, composable object representations, supporting downstream tasks such as visual reasoning, causal inference, world modeling, and compositional generation. Slot Attention (SA) is a fully unsupervised approach but performs poorly on real-world scenes. Recent methods combining SA with pretrained diffusion models (e.g., Stable-LSD, SlotAdapt) have made progress, yet two core problems persist:

**Slot Entanglement**: A single slot encodes features from multiple objects, causing image distortion or semantic inconsistency during single-slot generation. The root cause is that softmax normalization forces attention weights to sum to 1 across all slots; when certain U-Net queries do not strongly match any semantic slot, attention disperses across multiple slots.

**Weak Alignment**: Slots fail to consistently correspond to distinct image regions, leading to over-segmentation (one object split into multiple slots) or under-segmentation (multiple objects merged into one slot).

These two problems severely undermine the accuracy of object-centric representations and the practicality of compositional scene generation.

## Method

### Overall Architecture

CODA extracts image features via DINOv2, produces slot representations through Slot Attention, and uses pretrained Stable Diffusion v1.5 as a slot decoder to reconstruct the input image. Three improvement modules are introduced on top of this backbone.

### Key Designs

1. **Register Slots**: Pure padding tokens are passed through SD's frozen text encoder (CLIP ViT-L/14) to obtain a fixed-length embedding sequence $\bar{\mathbf{r}}$, serving as input-agnostic register slots. In cross-attention, these register slots act as "attention sinks," absorbing residual attention that should not be allocated to semantic slots. Due to the softmax normalization constraint, when U-Net queries do not match any semantic slot, attention naturally flows to register slots rather than contaminating semantic slots, thereby alleviating slot entanglement. SD v1.5 uses 77 padding tokens, yielding 77 register slots. Experiments show that fixed register slots outperform trainable variants.

2. **Cross-Attention Finetuning**: Because SD is pretrained on image-text pairs, directly using it as a slot decoder introduces a text-conditioning bias, where the model tends toward language-driven semantics rather than slot-level representations. CODA fine-tunes only the key/value/output projection matrices $\boldsymbol{\theta}$ in the cross-attention layers, without introducing additional layers or adapters, keeping the design conceptually clean and computationally efficient. The denoising objective is:

$$\mathcal{L}_{\mathrm{dm}}(\phi, \boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{z}, \mathbf{s}), \epsilon, \gamma} \left[\|\epsilon - \epsilon_{\boldsymbol{\theta}}(\mathbf{z}_\gamma, \gamma, \mathbf{s}, \bar{\mathbf{r}})\|_2^2\right]$$

3. **Contrastive Alignment Objective**: The denoising loss alone cannot explicitly ensure that slots capture concepts actually present in the image. CODA introduces a contrastive loss using negative samples $\tilde{\mathbf{s}}$ (constructed as hard negatives by randomly replacing half the slots across images), encouraging the model to assign high likelihood to matched slots and low likelihood to mismatched slots:

$$\mathcal{L}_{\mathrm{cl}}(\phi) = -\mathbb{E}_{(\mathbf{z}, \tilde{\mathbf{s}}), \epsilon, \gamma} \left[\|\epsilon - \epsilon_{\bar{\boldsymbol{\theta}}}(\mathbf{z}_\gamma, \gamma, \tilde{\mathbf{s}}, \bar{\mathbf{r}})\|_2^2\right]$$

where $\bar{\boldsymbol{\theta}}$ denotes stop-gradient parameters. Crucially, the contrastive loss updates only the SA module while freezing the diffusion decoder, preventing the decoder from taking shortcuts.

### Loss & Training

The total loss is a weighted sum of the denoising loss and the contrastive loss:

$$\mathcal{L}(\phi, \boldsymbol{\theta}) = \mathcal{L}_{\mathrm{dm}}(\phi, \boldsymbol{\theta}) + \lambda_{\mathrm{cl}} \mathcal{L}_{\mathrm{cl}}(\phi)$$

Theoretically, this objective is equivalent to a tractable proxy for maximizing mutual information between slots and images (Theorem 1), where the denoising gap $\Delta$ serves as a practical approximation of mutual information. Hard negative construction via shared initialization ensures semantic validity.

## Key Experimental Results

### Main Results: Object Discovery

| Dataset | Metric | CODA | Prev. SOTA (SlotAdapt) | Gain |
|--------|------|------|----------|------|
| MOVi-C | FG-ARI | 59.19 | 51.98 (LSD) | +7.21 |
| MOVi-E | FG-ARI | 59.04 | 56.45 | +2.59 |
| VOC | FG-ARI | 32.23 | 29.6 | +2.63 |
| VOC | mBOi | 55.38 | 51.5 | +3.88 |
| VOC | mIoUc | 56.30 | 49.3 (SlotDiff) | +7.00 |
| COCO | FG-ARI | 47.54 | 41.4 | +6.14 |

### Ablation Study

| Configuration | FG-ARI | mBOi | mBOc | mIoUi | mIoUc |
|------|--------|------|------|-------|-------|
| Baseline (Frozen SD) | 12.27 | 47.21 | 54.20 | 48.72 | 55.71 |
| + Register Slots | 19.21 | 55.76 | 64.02 | 49.93 | 57.14 |
| + CA Finetuning | 15.44 | 47.03 | 52.63 | 49.75 | 55.63 |
| + Contrastive | 11.96 | 47.16 | 54.17 | 49.40 | 56.56 |
| Reg + CA | 19.62 | 56.27 | 65.05 | 50.40 | 58.02 |
| Reg + CA + CO (w/o stop-gradient) | 10.54 | 30.64 | 35.86 | 37.74 | 43.61 |
| **Reg + CA + CO (CODA)** | **32.23** | **55.38** | **61.32** | **50.77** | **56.30** |

### Key Findings

- Register Slots constitute the single most impactful component (mBO improves by ~10 points), effectively alleviating slot entanglement.
- Stop-gradient on the diffusion decoder is essential for the contrastive loss; omitting it causes training instability and severe performance degradation.
- In compositional generation, CODA reduces FID from 40.57 (SlotAdapt) to 31.03.
- Category classification accuracy in attribute prediction improves from 43.92% to 78.06% on MOVi-E.

## Highlights & Insights

- The register slots design is inspired by the "attention sink" phenomenon in LLMs, offering a conceptually simple solution with near-zero computational overhead.
- The denoising loss and contrastive loss are unified under a mutual information maximization perspective.
- Fine-tuning only the KVO projections in cross-attention suffices to eliminate text-conditioning bias without any additional architectural modifications.
- The framework supports fine-grained compositional editing (object removal, object swapping), demonstrating strong practical utility.

## Limitations & Future Work

- 3D bounding box prediction remains weak, as DINOv2 features lack fine-grained geometric detail.
- Validation is currently limited to SD v1.5; performance on larger models (SDXL, SD3) is unknown.
- Segmentation quality in heavily occluded scenes still has room for improvement.
- The number of register slots (77) is determined by the SD text encoder and would require redesign when switching to a different backbone.

## Related Work & Insights

- Methods such as DINOSAUR and SPOT improve OCL from the self-supervised feature side, whereas CODA improves from the diffusion decoder side.
- The register token concept from ViTs can be transferred to other settings that involve attention competition.
- The contrastive alignment objective design can be generalized to other slot-based generative tasks, such as video object-centric learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of register slots and contrastive alignment is novel, though each individual technique is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers synthetic and real datasets, object discovery, attribute prediction, compositional generation, and ablation studies — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, experimental presentation is thorough, and figures are highly informative.
- Value: ⭐⭐⭐⭐ Makes a substantive contribution to the OCL community; the method is simple, efficient, and readily reproducible on existing frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICCV 2025\] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning](../../ICCV2025/image_generation/genflowrl_shaping_rewards_with_generative_object-centric_flow_in_visual_reinforc.md)
- [\[ICLR 2026\] Diffusion Alignment as Variational Expectation-Maximization](diffusion_alignment_as_variational_expectation-maximization.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
