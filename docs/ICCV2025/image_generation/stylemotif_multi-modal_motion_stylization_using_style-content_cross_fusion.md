---
title: >-
  [Paper Note] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion
description: >-
  [ICCV 2025][Image Generation][motion stylization] This paper proposes StyleMotif, a single-branch motion latent diffusion framework that unifies content generation and multi-modal style injection (text/image/video/audio/motion) via a style-content cross normalization mechanism. Compared to SMooDi's dual-branch design, StyleMotif reduces trainable parameters by 43.9% and improves inference speed by 22.5%, while achieving a 5.23% gain in Style Recognition Accuracy (SRA).
tags:
  - ICCV 2025
  - Image Generation
  - motion stylization
  - multi-modal
  - diffusion model
  - style-content fusion
  - motion generation
date: 2026-05-08
content_hash: 691a881493a0d7fe
---

# StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion

**Conference**: ICCV 2025
**arXiv**: 2503.21775
**Code**: [https://stylemotif.github.io](https://stylemotif.github.io)
**Area**: Image Generation
**Keywords**: motion stylization, multi-modal, diffusion model, style-content fusion, motion generation

## TL;DR

This paper proposes StyleMotif, a single-branch motion latent diffusion framework that unifies content generation and multi-modal style injection (text/image/video/audio/motion) via a style-content cross normalization mechanism. Compared to SMooDi's dual-branch design, StyleMotif reduces trainable parameters by 43.9% and improves inference speed by 22.5%, while achieving a 5.23% gain in Style Recognition Accuracy (SRA).

## Background & Motivation

**Why are existing methods insufficient?** The quality of human motion is determined by two dimensions: **content** (action categories such as walking and jumping) and **style** (emotional or personality expressions such as jubilant or aggressive). Existing methods suffer from the following limitations:

**Text-to-Motion methods** (MDM, MLD): proficient at generating diverse content but neglect the stylistic details of *how* actions are performed. Simply appending independent style transfer modules increases complexity and introduces cumulative errors.

**Motion style transfer methods** (Aberman et al., Motion Puzzle): effectively decouple content and style for small-scale tasks, but become cumbersome when stylizing a large variety of content motions. Transfer quality also degrades when the input motion is synthetic or noisy.

**SMooDi (latest representative)**: augments a pre-trained MLD with a dual-branch ControlNet-style style adapter and classifier-based style guidance, but (a) the dual-branch design increases model complexity and training overhead; (b) only motion sequences are supported as style input.

**Key Challenge**: The dual-branch design of existing methods (main generation network + style control network) requires maintaining additional parameters $\theta_s$ and zero-initialized linear layers $\theta_{z_i}$, limiting parallelism. Moreover, style input is restricted to the single modality of motion sequences.

## Method

### Overall Architecture

StyleMotif is built upon the pre-trained Motion Latent Diffusion (MLD) model and adopts a **single-branch** design. It comprises three core modules:

1. **Style Encoder Pre-training**: a VAE encoder jointly trained on HumanML3D (content knowledge) and 100STYLE (style knowledge).
2. **Style-Content Cross Fusion**: style features are injected into the diffusion process via statistical transformation.
3. **Multi-Modal Alignment**: cross-modal style conditioning is enabled through ImageBind.

### Key Design 1: Style Encoder Pre-training

The style encoder is derived from MLD's VAE encoder and is pre-trained in two stages:

1. First pre-trained on HumanML3D (14,616 motion sequences + 44,970 text descriptions) to learn feature representations of content motions.
2. Then fine-tuned on 100STYLE (45,303 style motions) in a variational autoencoding manner to align the content and style data distributions.

After training, the decoder is discarded and only the encoder is retained as the style encoder. This dual-dataset pre-training strategy enables the encoder to understand both content structure and style variations.

### Key Design 2: Style-Content Cross Normalization

The core innovation replaces additional network parameters with **statistical transformation**. Given the content feature $\mathcal{F}_c^i$ of the $i$-th MLD block and the style feature $\mathcal{F}_s$:

**Step 1**: Compute the mean and variance of the content feature:

$$\mu_c = \frac{1}{D}\sum_{j=1}^{D}\mathcal{F}_c^{i,j}, \quad \sigma_c^2 = \frac{1}{D}\sum_{j=1}^{D}(\mathcal{F}_c^{i,j} - \mu_c)^2$$

**Step 2**: Normalize the style feature using content statistics:

$$\widetilde{\mathcal{F}}_{s,c} = \frac{\mathcal{F}_s - \mu_c}{\sqrt{\sigma_c^2 + \eta}}$$

**Step 3**: Add the normalized style feature back to the content feature:

$$\mathcal{F}^i_{out} = \mathcal{F}_c^i + \gamma \cdot \widetilde{\mathcal{F}}_{s,c}$$

where $\gamma = 0.6$ is the optimal scaling factor. A key constraint is that fusion is applied only once after the $m$-th block to avoid excessive distortion of content.

**Comparison with SMooDi**: SMooDi injects style at each block through zero-initialized linear layers $\mathcal{Z}(\cdot)$ (requiring a copy of $\theta_s$), whereas StyleMotif achieves equivalent functionality through parameter-free statistical transformation.

### Key Design 3: Multi-Modal Alignment

Cross-modal style conditioning is realized through ImageBind's unified multi-modal feature space:

1. ImageBind's text encoder is frozen, and a lightweight projection layer is added to align feature dimensions.
2. A contrastive learning loss aligns the feature space on motion-text pairs (from 100STYLE):

$$\mathcal{L}_{align} = -\frac{1}{2}\sum_{(i,j)} \log\frac{\exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^j / \tau_0)}{\sum_k \exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^k / \tau_0)} + \log\frac{\exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^j / \tau_0)}{\sum_k \exp(\mathcal{F}_t^k \cdot \mathcal{F}_s^j / \tau_0)}$$

3. At inference time, any modality input (image/video/audio) is passed through ImageBind to extract features, and the most similar motion style feature is retrieved for stylization.

### Loss & Training

- Only the style encoder is trained; all other MLD parameters are frozen.
- AdamW optimizer with learning rate $10^{-5}$.
- A hybrid guidance strategy combining classifier-free and classifier-based guidance is employed.

## Key Experimental Results

### Main Results: Motion-Guided Stylization (Table 1)

| Method | SRA ↑ | FID ↓ | MM Dist ↓ | R-Precision ↑ | Diversity | Foot Skate ↓ |
|--------|-------|-------|-----------|---------------|-----------|--------------|
| MLD + Aberman | 54.37 | 3.309 | 5.983 | 0.406 | 8.816 | 0.347 |
| MLD + Motion Puzzle | 63.77 | 6.127 | 6.467 | 0.290 | 6.476 | 0.185 |
| SMooDi | 72.42 | 1.609 | 4.477 | 0.571 | 9.235 | 0.124 |
| **StyleMotif** | **77.65** | **1.551** | **4.354** | **0.586** | 7.567 | **0.097** |

**Key Findings**: StyleMotif outperforms SMooDi by 5.23% in SRA (72.42→77.65), while achieving lower FID (1.551 vs. 1.609) and Foot Skate Ratio (0.097 vs. 0.124), indicating more realistic generated motions.

### Ablation Study: Style Encoder Pre-training Strategy (Table 3, upper)

| Pre-training Data | SRA ↑ | FID ↓ | MM Dist ↓ | R-Precision ↑ | Foot Skate ↓ |
|-------------------|-------|-------|-----------|---------------|--------------|
| 100STYLE only | 76.73 | 1.788 | 4.349 | 0.571 | 0.101 |
| HumanML3D only | 76.58 | 1.635 | 4.458 | 0.572 | 0.109 |
| **Both combined** | **77.65** | **1.551** | **4.354** | **0.586** | **0.097** |

**Key Findings**: Dual-dataset pre-training outperforms single-dataset training across all metrics, validating the necessity of jointly learning content and style knowledge.

### Motion Style Transfer (Table 2)

| Method | SRA ↑ | FID ↓ | Foot Skate ↓ |
|--------|-------|-------|--------------|
| MLD + Aberman | 61.01 | 3.892 | 0.338 |
| SMooDi | 65.15 | 1.582 | 0.095 |
| **StyleMotif** | **68.81** | **1.375** | **0.094** |

### Text-Guided Stylization

| Method | SRA ↑ | FID ↓ |
|--------|-------|-------|
| MLD + ChatGPT | 4.82 | 0.614 |
| **StyleMotif** | **56.71** | **0.603** |

Under text guidance, SRA increases dramatically from 4.82% to 56.71%, validating the effectiveness of multi-modal alignment.

### Efficiency Comparison (Table 4)

| Method | Total Params | Trainable Params | Inference Time |
|--------|-------------|------------------|----------------|
| SMooDi | 468M | 13.9M | 4.0s |
| **StyleMotif** | **462M** | **7.8M (−43.9%)** | **3.1s (−22.5%)** |

## Highlights & Insights

1. **Parameter-free style injection**: Replacing zero-initialized linear layers with statistical normalization eliminates redundant trainable parameters — an elegant design choice.
2. **Single-branch outperforms dual-branch**: This work challenges the paradigm that style control requires an additional network branch, demonstrating that carefully designed feature fusion can achieve superior results with a simpler architecture.
3. **Emergent multi-modal capability**: By aligning the motion encoder with ImageBind, the model gains image/video/audio-guided stylization without requiring modality-specific training.
4. **Style interpolation**: The framework supports weighted blending of multiple style text inputs, producing smooth style transitions.

## Limitations & Future Work

1. **Limited style data**: The scale of the 100STYLE dataset constrains the model's style generalization, making it difficult to cover all possible motion styles.
2. **Single-label optimality paradox**: Ablations show that short single-style labels (e.g., "Old") outperform detailed descriptions, suggesting the current alignment method may have insufficient understanding of complex style descriptions.
3. **Motion quality bounded by the base model**: Reliance on MLD as the backbone means the content generation quality ceiling is constrained by the base model.
4. **Manual tuning of scaling factor γ**: The value $\gamma = 0.6$ is a fixed constant determined via ablation and may not be optimal across different tasks.

## Related Work & Insights

- **SMooDi**: Representative of ControlNet-style dual-branch design, providing the baseline for stylized motion generation.
- **ImageBind**: A foundation model for unified multi-modal embedding; this work demonstrates its potential in the motion domain.
- **AdaIN (image style transfer)**: The inspiration for cross normalization, transferring the idea of statistical style transfer from the image domain to the motion domain.
- **Insight**: Statistical normalization as a parameter-free feature fusion mechanism is generalizable to conditional injection in other generative tasks, such as style control in 3D and video generation.

## Rating ⭐⭐⭐⭐

The single-branch design philosophy is clear, and the cross normalization strategy is both elegant and effective. Multi-modal extension is a highlight, demonstrating emergent capability. Ablation studies provide comprehensive coverage (pre-training strategy, scaling factor, text expression style). The dramatic improvement in text-guided stylization (4.82→56.71 SRA) is particularly impressive.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](aicomposer_any_style_and_content_image_composition_via_feature_integration.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)

<!-- RELATED:END -->
