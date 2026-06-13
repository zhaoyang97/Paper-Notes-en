---
title: >-
  [Paper Note] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Diffusion Model Interpretability] This paper employs linear probing to demonstrate that **implicit decisions** in diffusion models—such as defaulting to male when gender is unspecified—are p…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion Model Interpretability"
  - "Self-Attention"
  - "Debiasing"
  - "Linear Probing"
  - "Implicit Decision"
date: 2026-05-08
content_hash: c57c7247bd430f2c
---

# Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2604.06052](https://arxiv.org/abs/2604.06052)  
**Code**: [https://github.com/kzaleskaa/icm](https://github.com/kzaleskaa/icm)  
**Area**: Image Generation / Diffusion Model Interpretability
**Keywords**: Diffusion Model Interpretability, Self-Attention, Debiasing, Linear Probing, Implicit Decision

## TL;DR
This paper employs linear probing to demonstrate that **implicit decisions** in diffusion models—such as defaulting to male when gender is unspecified—are primarily governed by self-attention layers rather than cross-attention layers. Building on this finding, the paper proposes ICM, a method that intervenes on a small number of critical self-attention layers to achieve state-of-the-art debiasing while minimizing image quality degradation.

## Background & Motivation

**Background**: Text-to-image diffusion models are highly capable but internally opaque. When a prompt is underspecified (e.g., "a photo of a person" without specifying gender), the model must make implicit decisions to fill in missing details.

**Dual Nature of Implicit Decisions**:
- Benign cases: deciding the color and shape of a generated "flower"
- Harmful cases: defaulting to male when generating "a doctor" (gender bias), or producing only specific individuals when prompted with "U.S. President" (representational bias)

**Limitations of Prior Work**:
- Mainstream methods (e.g., prompt injection, activation patching) measure layer influence by injecting different text prompts at specific layers
- However, these methods fundamentally rely on explicit textual conditioning—which **obscures the model's internal decision-making when explicit conditioning is absent**
- Prior work has attributed semantic integration to cross-attention layers, so most interventions target cross-attention

**Core Hypothesis**: The computational mechanisms underlying implicit decisions are **localized** and **distinct** from those governing explicit textual conditioning.

**Key Insight**: Linear probes are trained directly on intermediate activations to localize decision-relevant layers without prompt engineering, thereby avoiding the confounding effects of prompt injection.

## Method

### Overall Architecture
ICM (Implicit Choice-Modification) proceeds in three stages:
1. Generate images with generic prompts and collect intermediate layer activations
2. Apply an external classifier to pseudo-label the generated images
3. Train linear probes to quantify attribute separability per layer → identify the most critical layers → perform targeted intervention

### Key Designs

1. **Linear Probe-Based Layer Selection**:

    - Define a concept $\mathcal{C}$ (e.g., gender) and mutually exclusive options $\mathcal{A} = \{a_1, ..., a_K\}$ (e.g., male, female)
    - Generate $N$ images using a generic prompt $p_{gen}$ (e.g., "a photo of a person")
    - Extract activations $H_{l,t}^{(i)}$ (after average pooling) from each layer $l$ at each timestep $t$
    - Pseudo-label generated images using external classifiers such as FairFace
    - Train logistic regression for each $(l, t)$ pair: $f_{l,t}: \mathbb{R}^{d_l} \to \mathcal{A}$
    - Layers with high classification accuracy exhibit strong attribute separability, indicating key sites of implicit decision-making
    - **Core Finding**: Linear probes on self-attention layers achieve substantially higher accuracy than those on cross-attention layers, with peak performance near the middle blocks

2. **Activation Steering**:

    - Use the logistic regression weight vector as the steering direction: $s_{\ell,t} = \hat{w}_{\ell,t} / \|\hat{w}_{\ell,t}\|_2$
    - Modify activations at selected layers during generation: $H_{\ell,t}' = H_{\ell,t} + \alpha \cdot s_{\ell,t}$
    - $\alpha$ controls the intervention strength
    - **Why it works**: The weight vector defines the direction of maximum class separability; shifting activations along this direction directly influences implicit decisions

3. **Selective Fine-Tuning**:

    - Apply LoRA fine-tuning exclusively to selected self-attention layers
    - Generate images with specific attributes (e.g., "a young doctor") while using generic prompts (e.g., "a doctor") as training conditions
    - Standard diffusion loss: $\mathcal{L} = \mathbb{E}_{\epsilon,t}[\|\epsilon - \hat{\epsilon}_\theta(x_t, p, t)\|_2^2]$

### Key Insight: Self-Attention vs. Cross-Attention

| Layer Type | Role | Relevance to Implicit Decisions |
|--------|------|---------------|
| Cross-attention | Integrates semantic information from text conditioning | Low — when the prompt contains no attribute tokens, cross-attention has no signal to draw from |
| Self-attention | Propagates and consolidates stochastic cues | **High** — unifies independent cues from the initial noise (e.g., hairstyle + lipstick) into a coherent generation |

The authors hypothesize that self-attention layers make implicit decisions by propagating and consolidating random cues from the initial noise.

## Key Experimental Results

### Main Results (SD v1.5 Debiasing)

| Method | Gender FD↓ | Age FD↓ | Race FD↓ | FID↓ | CLIP-T↑ |
|------|------------|---------|----------|------|---------|
| Original | 0.564 | 0.752 | 0.558 | 120.06 | 0.6155 |
| DIFFLENS | **0.046** | **0.049** | 0.401 | 112.83 | 0.6090 |
| Finetuning | 0.050 | 0.746 | 0.198 | 161.47 | 0.6095 |
| **ICM (Steering)** | 0.087 | 0.133 | **0.266** | 122.08 | **0.6140** |
| **ICM (Finetuning)** | 0.535 | 0.681 | 0.449 | 143.98 | 0.6189 |

### Ablation Study: Self-Attention vs. Cross-Attention Steering

| Intervention Target | Gender FD↓ | Age FD↓ | Race FD↓ | FID↓ |
|----------|------------|---------|----------|------|
| Cross-attn steering | 0.365 | 0.612 | 0.428 | 118.39 |
| **Self-attn steering** | **0.085** | **0.273** | **0.298** | 118.31 |
| Cross-attn finetuning | 0.535 | 0.681 | 0.449 | 143.98 |
| **Self-attn finetuning** | **0.463** | **0.770** | **0.524** | 139.04 |

### Key Findings
- **Self-attention steering substantially outperforms cross-attention steering across all attributes** (Gender FD: 0.085 vs. 0.365)
- ICM Steering achieves the best quality–fairness trade-off: FD is reduced with minimal impact on FID/CLIP-T
- Intervening on only 2–4 critical self-attention layers outperforms intervening on all layers (the latter introduces artifacts and quality degradation)
- Activations from generic and explicit prompts are **linearly separable** (test accuracy 56–89%), confirming that the two generation mechanisms are indeed distinct
- The method generalizes to SDXL (70 layers) and SANA (Transformer architecture, 20 layers)

## Highlights & Insights
- **The finding that implicit decisions are localized in self-attention layers** is a significant contribution, challenging the prevailing assumption that all semantic information resides in cross-attention
- **Precise rather than comprehensive intervention**: modifying only 2–4 layers achieves state-of-the-art performance, more efficiently than modifying all layers or entire intermediate blocks
- The linear probing approach avoids the confounding effects of prompt injection, providing a cleaner view of internal model mechanisms
- The experimental design is rigorous: probes trained to distinguish implicit from explicit generation validate the two distinct mechanisms

## Limitations & Future Work
- The steering strength $\alpha$ must be tuned individually for each scenario, lacking automation
- The effectiveness of simultaneous multi-attribute debiasing (e.g., gender + age + race) is not thoroughly explored
- Linear probing assumes attributes are linearly separable in activation space, which may not hold for more complex concepts
- Generalization from debiasing to broader implicit concept control requires further validation

## Related Work & Insights
- DIFFLENS achieves better performance on certain metrics but relies on sparse autoencoders, entailing greater methodological complexity
- The methodology is analogous to "knowledge localization" research in LLMs (e.g., ROME, activation patching)
- The paper challenges prompt injection approaches—explicitly injecting attributes does not faithfully reflect the model's internal decision-making mechanisms

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The finding that self-attention layers serve as the locus of implicit decisions is an important contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three attributes × three architectures × multiple intervention strategies, with thorough analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Elegant narrative with a complete logical chain from hypothesis to validation to application
- Value: ⭐⭐⭐⭐ Provides a foundational contribution to understanding diffusion model internals, with practical debiasing applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](../../ICCV2025/image_generation/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[ICML 2026\] Localizing Memorized Regions in Diffusion Models via Coordinate-Wise Curvature Differences](../../ICML2026/image_generation/localizing_memorized_regions_in_diffusion_models_via_coordinate-wise_curvature_d.md)
- [\[CVPR 2026\] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)

</div>

<!-- RELATED:END -->
