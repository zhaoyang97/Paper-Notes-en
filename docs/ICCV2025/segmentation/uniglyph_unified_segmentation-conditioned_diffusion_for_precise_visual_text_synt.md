---
title: >-
  [Paper Note] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis
description: >-
  [ICCV 2025][Segmentation][Visual text generation] This paper proposes UniGlyph, a visual text generation framework that adopts segmentation masks as a unified conditioning signal. By replacing conventional rendered glyph conditions with Adaptive Glyph Conditioning (AGC) and Glyph Region Loss (GRL), UniGlyph achieves state-of-the-art bilingual (Chinese and English) text image generation under a single ControlNet architecture, with particularly large margins in small-font and complex-layout scenarios.
tags:
  - ICCV 2025
  - Segmentation
  - Visual text generation
  - segmentation mask conditioning
  - diffusion models
  - bilingual glyph
  - small-font generation
  - ControlNet
date: 2026-05-08
content_hash: 170e234658c3d32d
---

# UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis

**Conference**: ICCV 2025
**arXiv**: [2507.00992](https://arxiv.org/abs/2507.00992)
**Code**: Not released
**Area**: Image Segmentation
**Keywords**: Visual text generation, segmentation mask conditioning, diffusion models, bilingual glyph, small-font generation, ControlNet

## TL;DR

This paper proposes UniGlyph, a visual text generation framework that adopts segmentation masks as a unified conditioning signal. By replacing conventional rendered glyph conditions with Adaptive Glyph Conditioning (AGC) and Glyph Region Loss (GRL), UniGlyph achieves state-of-the-art bilingual (Chinese and English) text image generation under a single ControlNet architecture, with particularly large margins in small-font and complex-layout scenarios.

## Background & Motivation

Accurately rendering visual text (glyphs) in text-to-image generation remains a core open problem. Existing methods suffer from blurry character edges, semantic inconsistency, and insufficient font/color control. Mainstream ControlNet-based approaches (e.g., AnyText, GlyphDraw2) use pre-rendered glyph images as conditions, but this introduces a fundamental limitation:

**Information degradation**: Pre-rendered glyphs retain only shape and position, discarding original font and color information. This incomplete conditioning signal forces the model to learn an implicit mapping from synthetic glyphs (typically in a default font) to real typographic variations. To compensate for this information loss, existing methods resort to auxiliary modules:
- AnyText: text embedding replacement module (positional encoding)
- GlyphDraw2: style-guided branch (font/color control)

This leads to **multi-branch architectural bloat**: increased computational complexity, reduced model reusability, and optimization conflicts—most notably when generating small fonts or stylized glyphs.

The core insight of this paper is that **segmentation masks naturally preserve all glyph attributes**—shape, position, font style, and color—without requiring any auxiliary control modules. Accordingly, UniGlyph replaces rendered glyph images with pixel-level visual text segmentation masks as the unified conditioning input.

## Method

### Overall Architecture

UniGlyph comprises three core components:
1. **Bilingual text segmentation model**: Fine-tuned from Hi-SAM (SAM-TS) to extract pixel-level text segmentation masks from images.
2. **Flow Matching diffusion model + DiT ControlNet**: Built on FLUX.1-dev, using segmentation masks as the ControlNet condition.
3. **LayoutTransformer (optional)**: Automatically generates text layout and style information at inference time.

### Key Design 1: Adaptive Glyph Condition (AGC)

Directly using segmentation masks is problematic: black text merges with black backgrounds. Additionally, segmentation models are inaccurate for small fonts. An adaptive strategy is therefore designed:

PP-OCRv4 is used to obtain the bounding box of each glyph region $R_i$. The average per-character area is computed as $A_{\text{avg},i} = A_i / N_i$, with threshold $T = 4900$ pixels:

$$G_i = \begin{cases} \text{Canny}(M_{\text{seg}}) + M_{\text{seg}} \odot I, & \text{if } A_{\text{avg},i} > T \\ (M_{\text{pos}} \odot I_i)^{\text{blur}}, & \text{if } A_{\text{avg},i} \leq T \end{cases}$$

- **Large glyph regions**: Original-color glyphs extracted via segmentation masks + Canny edge enhancement (resolving the black-text merging issue).
- **Small glyph regions**: Fall back to position-mask-cropped image regions + Gaussian-blurred boundaries (avoiding interference from inaccurate segmentation).

The final condition is: $G = \bigcup_i G_i$

### Key Design 2: Flow Matching Diffusion Model

Under the Flow Matching framework, the model learns a continuous-time velocity field $\mathbf{v}^*(z_t, t)$. The image $I$ and glyph condition $G$ are encoded into latent representations $z_0, z_g$ via a VAE. The ControlNet produces glyph features $z_s = C(z_g, c_{te}, t)$. The flow matching loss is:

$$L_{\text{fm}} = \mathbb{E}_{z_0, z_s, c_{te}, t}\left[\|\mathbf{v}_\theta(z_t, z_s, c_{te}, t) - \mathbf{v}^*(z_t, t)\|_2^2\right]$$

### Key Design 3: Glyph Region Loss (GRL)

The segmentation mask is used to impose an additional MSE loss on glyph regions in pixel space, effectively assigning higher loss weight to those regions.

The mask selection also follows the adaptive strategy:

$$M_{\text{gr}} = \begin{cases} M_{\text{seg}}, & \text{if } A_{\text{avg},i} > T \\ M_{\text{pos}}, & \text{if } A_{\text{avg},i} \leq T \end{cases}$$

The glyph region loss is:

$$L_{\text{gr}} = \mathbb{E}_{\mathbf{x}_0, \hat{\mathbf{x}}_0}\left[\|M_{\text{gr}} \odot (\hat{\mathbf{x}}_0 - \mathbf{x}_0)\|_2^2\right]$$

The total loss is $L = L_{\text{fm}} + \lambda \cdot L_{\text{gr}}$, where $\lambda = 1$, and $L_{\text{gr}}$ is disabled for the first 100K training steps.

### LLM-based Layout Prediction

An open-source LLM is fine-tuned to convert user prompts into structured layouts: `<rewritten prompt, texts, bboxes, fonts, colors>`. Predefined font and color sets are mapped to special tokens, and only 1,000 poster samples are used for fine-tuning.

## Experiments

### Main Results (AnyText-benchmark)

| Method | Chinese Sen.Acc | Chinese NED | English Sen.Acc | English NED |
|--------|----------------|-------------|----------------|-------------|
| AnyText-V1.1 | 0.6923 | 0.8423 | 0.6564 | 0.8685 |
| GlyphDraw2 | 0.7350 | 0.8451 | 0.7369 | 0.8921 |
| AnyText2 | 0.7130 | 0.8516 | 0.8096 | 0.9184 |
| CharGen | 0.7499 | 0.8609 | 0.8096 | 0.9205 |
| **UniGlyph** | **0.8267** | **0.8976** | **0.9018** | **0.9582** |

UniGlyph surpasses AnyText2 by +11.4% in Chinese accuracy and CharGen by +9.2% in English accuracy.

### Small-Font Generation (MiniText-benchmark)

| Method | Sen.Acc | NED | ClipScore |
|--------|---------|-----|-----------|
| SD3 | 0.0000 | 0.0005 | 0.7990 |
| AnyText-V1.1 | 0.0138 | 0.4680 | 0.8098 |
| GlyphDraw2 | 0.0100 | 0.4508 | 0.8146 |
| Glyph-ByT5 | 0.3881 | 0.8268 | 0.8594 |
| **UniGlyph** | **0.7925** | **0.9537** | 0.8124 |

UniGlyph achieves more than **twice** the accuracy of the second-best method (Glyph-ByT5) on small-font generation.

### Ablation Study

Effect of $\lambda$:

| $\lambda$ | Sen.Acc | NED | ClipScore |
|-----------|---------|-----|-----------|
| 0 | 0.8179 | 0.8952 | 0.7868 |
| 0.1 | 0.8166 | 0.8945 | 0.7871 |
| 1 | **0.8188** | **0.8958** | **0.7896** |
| 4 | 0.8158 | 0.8949 | 0.7870 |

Necessity of AGC:

| Method | Sen.Acc | NED | ClipScore |
|--------|---------|-----|-----------|
| w/o AGC | 0.7724 | 0.9348 | 0.8064 |
| w/o Gaussian Blur | 0.7851 | 0.9508 | 0.7963 |
| UniGlyph | **0.7849** | **0.9507** | **0.8097** |

Key findings:
1. The Glyph Region Loss ($\lambda > 0$) improves both accuracy and image quality (ClipScore) over the no-loss baseline ($\lambda = 0$).
2. The adaptive hybrid strategy significantly improves small-font generation accuracy (0.7724→0.7851), and Gaussian blurring further restores ClipScore.
3. Training requires only 7.36M samples—far fewer than AnyText (30M) and TextDiffuser (tens of millions)—demonstrating high sample efficiency.

## Highlights & Insights

1. **Paradigm shift**: Replacing rendered glyphs with segmentation masks fundamentally resolves the conditioning signal degradation problem from an information-preservation perspective.
2. **Architectural simplification**: A single ControlNet replaces multi-branch architectures, eliminating auxiliary modules such as positional encoding and style-guided branches.
3. **Adaptive design**: The method elegantly handles the limitations of segmentation models on small fonts by switching between precise segmentation and coarse positional masks based on glyph region size.
4. **Dataset contribution**: GlyphMM-3M (3M+ high-resolution bilingual images) and MiniText-benchmark fill gaps in the community.

## Limitations & Future Work

1. The segmentation model remains inaccurate for extremely small fonts, necessitating fallback to positional masks.
2. Running the text segmentation model at inference time adds complexity to the inference pipeline.
3. Due to resource constraints, only a subset of the dataset was used for training, leaving the method's full potential unrealized.
4. The Glyph Region Loss requires reconstructing latent representations back to pixel space during training, reducing training speed.

## Related Work & Insights

- **Text rendering methods**: AnyText and GlyphDraw2 are based on ControlNet with rendered glyph conditions; Glyph-ByT5 employs a character-level encoder.
- **Controllable generation**: ControlNet, T2I-Adapter, and IP-Adapter provide diverse control signals.
- **Text segmentation**: Hi-SAM is a hierarchical text segmentation model built on SAM.
- **LLM-based layout generation**: LayoutGPT and TextDiffuser-2 leverage LLMs to generate layout bounding boxes.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The paradigm shift of using segmentation masks as a unified condition is pioneering.
- **Technical Quality**: ⭐⭐⭐⭐ — The adaptive strategy is elegantly designed, though ablation experiments are conducted at reduced resolution.
- **Practicality**: ⭐⭐⭐⭐⭐ — Bilingual support, strong small-font capability, and a clean architecture.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is clear and experimental setup is thorough.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] VSC: Visual Search Compositional Text-to-Image Diffusion Model](vsc_visual_search_compositional_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](../../NeurIPS2025/segmentation/seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)

<!-- RELATED:END -->
