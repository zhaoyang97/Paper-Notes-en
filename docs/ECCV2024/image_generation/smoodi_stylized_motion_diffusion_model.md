---
title: >-
  [Paper Note] SMooDi: Stylized Motion Diffusion Model
description: >-
  [ECCV 2024][Image Generation] Introduces SMooDi—the first diffusion model that adapts a pre-trained text-to-motion model for stylized motion generation. Through a style adaptor and dual style guidance (classifier-free guidance + classifier-based guidance), it enables diverse stylized motion generation driven by content text and style motion sequences.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 4e78c1acdc5bb5f3
---

# SMooDi: Stylized Motion Diffusion Model

**Conference**: ECCV 2024  
**arXiv**: [2407.12783](https://arxiv.org/abs/2407.12783)  
**Area**: Image Generation

## TL;DR

Introduces SMooDi—the first diffusion model that adapts a pre-trained text-to-motion model for stylized motion generation. Through a style adaptor and dual style guidance (classifier-free guidance + classifier-based guidance), it enables diverse stylized motion generation driven by content text and style motion sequences.

## Background & Motivation

- Human motion consists of two dimensions: content (e.g., walking, waving) and style (e.g., elderly, happy, angry).
- **Text-driven motion generation** (e.g., MDM, MLD) has made significant progress but primarily focuses on content, lacking style control.
- **Motion style transfer** methods can transfer style from one sequence to another but require an existing content motion sequence as input, which limits flexibility.
- **Simple concatenation** of the two (generation followed by transfer) suffers from three issues:
  1. Low efficiency: Each sequence must be processed sequentially.
  2. Error accumulation: Style transfer models degrade in performance on imperfectly generated motions.
  3. Data limitations: Style transfer methods rely on limited motion content in specific style datasets.

## Method

### Overall Architecture

SMooDi is built upon the pre-trained MLD (Motion Latent Diffusion) model and comprises two core modules:
1. **Style Adaptor**: Injects style conditions through residual features.
2. **Style Guidance**: Dual guidance consisting of classifier-free and classifier-based guidance.

In the denoising step $t$, the model takes the content text $\mathbf{c}$, style motion $\mathbf{s}$, and noisy latent variable $\mathbf{z}_t$ as inputs to predict the noise $\epsilon_t$.

### Key Designs

**1. Style Adaptor**

- Train a trainable replica of the Transformer Encoder in MLD.
- An independent style encoder extracts embeddings from the style motion sequence.
- The connection between the Style Adaptor and MLD is established through **zero-initialized linear layers** (inspired by the ControlNet design).
- During training, the adaptor progressively learns style constraints and applies corrective features to the corresponding layers of MLD.

**2. Classifier-Free Style Guidance**

Decomposes the conditional guidance into two independent components for content and style:

$$\epsilon_\theta(\mathbf{z}_t, t, \mathbf{c}, \mathbf{s}) = \epsilon_\theta(\mathbf{z}_t, t, \emptyset, \emptyset) + w_c(\epsilon_c - \epsilon_\emptyset) + w_s(\epsilon_{cs} - \epsilon_c)$$

Where $w_c$ and $w_s$ control the strength of content and style guidance, respectively, allowing a flexible balance between content preservation and style expression.

**3. Classifier-Based Style Guidance**

Specifies an analytical function $G(\mathbf{z}_t, t, \mathbf{s})$ that computes the L1 distance between the generated motion and the reference style in the style embedding space:

$$G(\mathbf{z}_t, t, \mathbf{s}) = |f(\hat{\mathbf{x}}_0) - f(\mathbf{s})|$$

Its gradient is utilized to guide the generated motion toward the target style. The style feature extractor is obtained by training a style classifier on the 100STYLE dataset.

### Loss & Training

The total training loss consists of three terms:

$$\mathcal{L}_{all} = \mathcal{L}_{std} + \lambda_{pr}\mathcal{L}_{pr} + \lambda_{cyc}\mathcal{L}_{cyc}$$

- $\mathcal{L}_{std}$: Standard denoising loss computed on the 100STYLE dataset.
- $\mathcal{L}_{pr}$: **Content prior preservation loss**—computed by sampling from HumanML3D to prevent "content forgetting."
- $\mathcal{L}_{cyc}$: **Cycle prior preservation loss**—reconstructs the original sequences after swapping the content and style of the two datasets, encouraging style-content disentanglement.

## Key Experimental Results

### Main Results

Comparison of the stylized text-to-motion generation task (HumanML3D content + 100STYLE style):

| Method | FID↓ | Foot Skating↓ | MM Dist↓ | R-precision↑ | Diversity→ | SRA(%)↑ |
|------|------|--------------|---------|-------------|-----------|--------|
| MLD+Motion Puzzle | 6.127 | 0.185 | 6.467 | 0.290 | 6.476 | 63.769 |
| MLD+Aberman et al. | 3.309 | 0.347 | 5.983 | 0.406 | 8.816 | 54.367 |
| ChatGPT+MLD | 0.614 | 0.131 | 4.313 | 0.605 | 8.836 | 4.819 |
| **SMooDi** | **1.609** | **0.124** | **4.477** | **0.571** | **9.235** | **72.418** |

The SRA of ChatGPT+MLD is only 4.82%, demonstrating that MLD cannot achieve stylized generation even when description text of style is provided.

### Ablation Study

Contribution of each module to the performance:

| Configuration | FID↓ | Foot Skating↓ | MM Dist↓ | R-precision↑ | Diversity→ | SRA(%)↑ |
|------|------|--------------|---------|-------------|-----------|--------|
| Full Model | 1.609 | 0.124 | 4.477 | 0.571 | 9.235 | 72.418 |
| w/o $L_{cyc}$ | 2.046 | 0.136 | 4.465 | 0.569 | 8.869 | 64.866 |
| w/o $L_{pr}+L_{cyc}$ | 5.996 | 0.166 | 6.098 | 0.335 | 7.456 | 81.841 |
| w/o Classifier Guidance | 1.050 | 0.111 | 4.085 | 0.630 | 9.445 | 20.245 |
| w/o Adaptor | 2.984 | 0.123 | 4.526 | 0.550 | 8.372 | 69.952 |

**Key Observations**:
- Removing $L_{pr}+L_{cyc}$ yields an SRA of 81.84% but causes the FID to surge to 5.996 $\rightarrow$ severe "content forgetting," where motions degrade entirely to locomotion from the style dataset.
- Removing classifier guidance drops the SRA sharply from 72.4% to 20.2% $\rightarrow$ classifier guidance is crucial for reflecting style.
- The adaptor and classifier guidance are complementary: the adaptor establishes the basic style direction, while classifier guidance performs fine-grained adjustment.

### Key Findings

1. **Complementary Dual Guidance**: Classifier-free guidance captures coarse, style-related features, while classifier-based guidance provides precise style control; both are indispensable.
2. **Crucial Content Preservation Loss**: Attempting to generate without the prior preservation loss leads to severe content forgetting. A single model cannot support diverse content and style simultaneously without it.
3. **Motion Style Transfer as a Downstream Task**: Content motion sequences can be inverted into noisy latent representations via DDIM-Inversion, enabling style transfer without requiring additional optimization.
4. **In user studies**, SMooDi receives higher user preference across three dimensions: realism, style reflection, and content preservation.

## Highlights & Insights

- Re-adapts pre-trained text-to-motion diffusion models for stylized generation for the first time, offering a clear and effective design approach.
- The cycle prior preservation loss is designed ingeniously, encouraging disentanglement by swapping the style and content of two datasets.
- Decomposing style guidance into classifier-free and classifier-based components allows flexible adjustment of the balance between content and style.
- A single model supports 100 styles $\times$ diverse content, eliminating the need for per-style fine-tuning required by previous methods.

## Limitations & Future Work

- Classifier-based style guidance relies on a style classifier trained on the 100STYLE dataset; performance may degrade when the content text deviates significantly from locomotion.
- The 100STYLE dataset contains only locomotion-related movements, which limits the variety of learnable style types.
- The SRA metric in quantitative evaluation relies on a pre-trained classifier, which may not fully reflect the visually perceived style quality.
- Requires a pre-trained MLD model and additional style datasets, making the training pipeline relatively complex.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to adapt a pre-trained motion diffusion model for stylized generation; the dual guidance mechanism is novel.
- **Practicality**: ⭐⭐⭐⭐ — Multi-style with a single model, supporting motion style transfer as a downstream task.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Complete quantitative, qualitative, user study, and ablation experiments, with in-depth ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive diagrams, and fully articulated motivation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

</div>

<!-- RELATED:END -->
