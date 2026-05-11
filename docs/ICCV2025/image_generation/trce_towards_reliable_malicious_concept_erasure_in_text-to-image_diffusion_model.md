---
title: >-
  [Paper Note] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models
description: >-
  [Image Generation] This paper proposes TRCE, a two-stage concept erasure strategy—textual semantic erasure followed by denoising trajectory steering—that reliably removes malicious concepts while minimizing degradation o…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 1c8e70ded503499e
---

# TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models

> **Conference**: ICCV 2025
> **arXiv**: [2503.07389](https://arxiv.org/abs/2503.07389)
> **Code**: [GitHub](https://github.com/ddgoodgood/TRCE)
> **Area**: Diffusion Model Safety · Concept Erasure · Image Generation
> **Keywords**: concept erasure, text-to-image safety, adversarial robustness, cross-attention editing, denoising trajectory

## TL;DR

This paper proposes TRCE, a two-stage concept erasure strategy—textual semantic erasure followed by denoising trajectory steering—that reliably removes malicious concepts while minimizing degradation of the model's general generation capability.

## Background & Motivation

Text-to-image diffusion models such as Stable Diffusion face significant safety risks in generating NSFW content alongside high-quality images. Concept Erasure (CE) addresses this by modifying model parameters to suppress specific concepts; however, existing methods exhibit a **fundamental tension between erasure reliability and knowledge preservation**:

**Implicit malicious semantics are difficult to erase**: Existing methods (e.g., ESD, UCE) primarily suppress specific keywords, yet malicious concepts are frequently expressed indirectly through metaphors, associations, or adversarial prompts (e.g., describing nudity-like scenes without using the word "nudity").

**Trade-off between knowledge preservation and erasure reliability**: To handle adversarial prompts, existing methods often over-modify the model, substantially degrading generation quality for unrelated content (higher FID, lower CLIP-Score).

**Vulnerability to adversarial attacks**: Red-teaming tools such as MMA, P4D, and Ring-A-Bell can readily bypass most erasure methods.

**Core insight of TRCE**: The elimination of malicious semantics and the generation of safe visual content should be handled at distinct levels—Stage 1 removes implicit malicious semantics at the textual level, while Stage 2 steers the sampling trajectory toward safe directions during denoising.

## Method

### Overall Architecture (Fig. 3)

TRCE consists of two stages:

**Stage 1: Textual Semantic Erasure** → modifies cross-attention matrices
**Stage 2: Denoising Trajectory Steering** → contrastive fine-tuning of the U-Net

### Key Design 1: [EoT] as the Mapping Target

TRCE identifies a critical mapping target—the **[EoT] (End of Text) embedding**. Unlike existing methods that directly remap keyword embeddings (leading to rapid knowledge forgetting), [EoT] plays a distinctive role:

- It carries the semantic information of the entire prompt
- It attends to salient regions in the generated image
- Modifying [EoT] alters image content while **preserving the overall context of the prompt**

An LLM (GPT-4o) is used to expand malicious concepts into 20 synonyms × 15 templates = 300 prompts, with corresponding safe prompt sets constructed accordingly. The cross-attention matrices $W_K, W_V$ are optimized via a closed-form solution:

$$W' = \left(\sum_{i=1}^n W \cdot e_i^s \cdot (e_i^m)^\top + \eta \sum_{j=1}^q W \cdot e_j^k \cdot (e_j^k)^\top\right) \cdot \left(\sum_{i=1}^n e_i^m \cdot (e_i^m)^\top + \eta \sum_{j=1}^q e_j^k \cdot (e_j^k)^\top\right)^{-1}$$

### Key Design 2: Denoising Trajectory Steering

This stage exploits the deterministic nature of diffusion model sampling—early perturbations along the ODE trajectory are sufficient to steer the final generated content toward safe directions.

**Trajectory preparation**: Early-stage sampling trajectories $\{z_t^m\}$ are cached using the original U-Net $\epsilon_\theta$ with malicious prompts.

**Guidance augmentation**: Semantically augmented safe/unsafe directions are constructed using classifier-free guidance scaling:

$$f_{safe} = \epsilon_\theta(z_t^m, \varnothing, t) + \beta(\epsilon_\theta(z_t^m, c^s, t) - \epsilon_\theta(z_t^m, \varnothing, t))$$

**Contrastive loss**: A triplet margin loss pulls denoising predictions toward the safe direction and away from the unsafe direction:

$$L_{erase} = \mathbb{E}[\max(\|\hat{\epsilon}_\theta - f_{safe}\|^2 - \|\hat{\epsilon}_\theta - f_{unsafe}\|^2 + margin, 0)]$$

A regularization term preserves unconditional predictions: $L_{preserve} = \|\hat{\epsilon}_\theta(z_t^u, \varnothing, t) - \epsilon_\theta(z_t^u, \varnothing, t)\|^2$

Only visual layers (Q matrices of self-attention and cross-attention) are fine-tuned over 3 epochs, requiring approximately 300 seconds.

## Key Experimental Results

### Main Results: Sexual Concept Erasure (Tab. 1)

| Method | I2P ↓ | MMA ↓ | P4D ↓ | Ring ↓ | UnDiff ↓ | FID_real ↓ | CLIP-S ↑ |
|------|-------|-------|-------|--------|----------|-----------|---------|
| SD1.4 | 34.69% | 79.00% | 83.44% | 59.49% | 57.75% | 27.18 | 30.97 |
| ESD | 31.15% | 58.50% | 82.67% | 50.63% | 77.46% | 26.88 | 31.21 |
| UCE | 8.16% | 30.80% | 43.71% | 13.92% | 19.72% | 27.20 | 30.92 |
| RECE | 6.34% | 23.10% | 32.00% | 6.33% | 15.49% | 28.26 | 30.79 |
| MACE | 7.09% | 10.60% | 7.95% | 10.13% | 11.27% | 26.98 | 28.84 |
| AdvUnlearn | 1.71% | 0.30% | 1.99% | 6.33% | 3.52% | 29.65 | 28.93 |
| **TRCE(T+V)** | **1.29%** | **1.40%** | **1.99%** | **1.27%** | **0.70%** | **26.89** | **30.71** |

TRCE(T+V) achieves approximately 1% ASR across all five attack types while maintaining optimal FID_real and CLIP-Score—**representing the first method to genuinely reconcile erasure reliability with knowledge preservation**.

### Multi-Concept Erasure (Tab. 2, I2P 7-category malicious concepts)

| Method | Overall ↓ | FID_real ↓ | CLIP-S ↑ |
|------|--------|-----------|---------|
| MACE | 5.6% | 26.20 | 28.13 |
| TRCE(T) | 3.6% | 27.25 | 30.43 |
| **TRCE(T+V)** | **2.0%** | **27.23** | **30.48** |

Key finding: In the multi-concept erasure setting, MACE's CLIP-S drops from 30.97 to 28.13 (severe knowledge degradation), whereas TRCE's CLIP-S decreases only marginally from 30.97 to 30.48.

### Ablation Study: Contribution of Each Stage

| Stage | I2P ↓ | MMA ↓ | P4D ↓ |
|------|-------|-------|-------|
| TRCE(T): Stage 1 only | 5.05% | 7.80% | 7.95% |
| TRCE(V): Stage 2 only | 13.86% | 35.00% | 48.00% |
| **TRCE(T+V): Both stages** | **1.29%** | **1.40%** | **1.99%** |

Key findings:
- Textual erasure alone is already highly effective (demonstrating the advantage of the [EoT] mapping target)
- Trajectory steering alone performs poorly, as malicious semantics remaining in the prompt are reintroduced during later denoising steps
- The two stages exhibit a multiplicative synergistic effect when combined

## Highlights & Insights

1. **[EoT] as the mapping target** is the central contribution—more effective and less damaging to general knowledge than direct keyword remapping, as [EoT] encodes holistic prompt semantics rather than isolated concepts.
2. The **two-stage cooperative design** is conceptually elegant: the textual stage "defuses" the malicious semantics, while the denoising stage acts as a "safety lock."
3. Reducing adversarial ASR to approximately 1% represents a breakthrough result in this field.
4. Fine-tuning requires only 300 seconds on a single RTX 4090, making the method highly practical.

## Limitations & Future Work

- Evaluation is conducted on SD1.4; generalization to newer architectures such as SDXL and SD3 remains to be verified.
- The closed-form modification of cross-attention matrices may accumulate errors across multiple iterative erasure passes.
- Edge cases involving highly sophisticated adversarial prompt engineering may still exist.

## Related Work & Insights

- Concept erasure: ESD, UCE, RECE, MACE, SPM, AdvUnlearn
- Red-teaming attacks: P4D, MMA, Ring-A-Bell, UnlearnDiff
- Inference-time guidance: SLD, Safree

## Rating

- **Novelty**: ★★★★☆ — The [EoT] mapping target and two-stage cooperative design are novel and practically motivated.
- **Technical Depth**: ★★★★★ — Demonstrates a deep understanding of the internal mechanisms of diffusion models.
- **Experimental Thoroughness**: ★★★★★ — Evaluation spans 5 attack types, multi-concept settings, and ablation studies, providing comprehensive coverage.
- **Writing Quality**: ★★★★☆ — Problem motivation is clearly articulated, and the two-stage logic is presented with a natural progression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)

</div>

<!-- RELATED:END -->
