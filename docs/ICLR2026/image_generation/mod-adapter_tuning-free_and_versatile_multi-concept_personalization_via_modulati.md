---
title: >-
  [Paper Note] Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter
description: >-
  [ICLR 2026][Image Generation][Multi-concept Personalization] This paper proposes Mod-Adapter, a tuning-free multi-concept personalization method that predicts concept-specific modulation directions in the modulation space of DiT, enabling decoupled customization of both object and abstract concepts (pose, lighting, material, etc.), substantially outperforming existing methods on multi-concept personalization benchmarks.
tags:
  - ICLR 2026
  - Image Generation
  - Multi-concept Personalization
  - Tuning-Free
  - DiT Modulation Space
  - Mixture-of-Experts
  - VLM Pre-training
date: 2026-05-08
content_hash: e79b40b5b01c9dfe
---

# Mod-Adapter: Tuning-Free and Versatile Multi-concept Personalization via Modulation Adapter

**Conference**: ICLR 2026
**arXiv**: [2505.18612](https://arxiv.org/abs/2505.18612)
**Code**: [Project Page](https://weizhi-zhong.github.io/Mod-Adapter)
**Area**: Diffusion Models / Personalized Generation
**Keywords**: Multi-concept Personalization, Tuning-Free, DiT Modulation Space, Mixture-of-Experts, VLM Pre-training

## TL;DR

This paper proposes Mod-Adapter, a tuning-free multi-concept personalization method that predicts concept-specific modulation directions in the modulation space of DiT, enabling decoupled customization of both object and abstract concepts (pose, lighting, material, etc.), substantially outperforming existing methods on multi-concept personalization benchmarks.

## Background & Motivation

**State of the Field**: Personalized text-to-image generation aims to synthesize target concepts from user-provided reference images. Existing methods predominantly focus on object concepts (persons, animals, everyday items), and multi-concept personalization methods similarly handle combinations of multiple objects.

**Limitations of Prior Work**: (a) Existing tuning-free methods (e.g., IP-Adapter, MS-Diffusion) fail to disentangle object and abstract concepts — when given an image of a person in a specific pose, they replicate the entire person rather than extracting only the pose; (b) TokenVerse supports abstract concepts but requires test-time fine-tuning for each new image, which is time-consuming and prone to overfitting.

**Root Cause**: Abstract concepts (pose, lighting, material) are not independent visual entities — they are tightly coupled with objects and difficult to extract in isolation. Moreover, there exists a substantial gap between extracted visual features and the modulation space of DiT.

**Paper Goals**: (i) Generalize to new concepts without test-time fine-tuning; (ii) Support customization of both object and abstract concepts simultaneously; (iii) Achieve decoupled control across multiple concepts.

**Starting Point**: The locality and semantic additivity of the AdaLN modulation space in DiT — assigning different modulation vectors to different tokens enables localized concept control.

**Core Idea**: Train a Mod-Adapter module to predict concept-specific modulation directions, with VLM-guided pre-training to bridge the large gap between the image and modulation spaces.

## Method

### Overall Architecture

Built upon FLUX (DiT architecture), Mod-Adapter takes a concept image and its corresponding concept word (e.g., "surface") as input and predicts concept-specific modulation directions $\{\Delta_i \mid i=1,\ldots,N\}$ (for $N=57$ DiT blocks). These directions are added to the modulation vectors of the corresponding concept text tokens, producing localized effects on concept-relevant image regions via the joint attention layers. During multi-concept inference, modulation directions for different concepts are applied independently to their respective text tokens.

### Key Designs

1. **Vision-Language Cross-Attention**:

    - **Function**: Extracts visual features of the target concept from the concept image.
    - **Mechanism**: The concept word (e.g., "surface") is encoded via a CLIP text encoder and an MLP projection layer to obtain a neutral feature, which is projected into $N$ queries (with sinusoidal positional encodings to distinguish different DiT blocks); the concept image is encoded via a CLIP image encoder to obtain keys and values; cross-attention $\text{Attention}(Q_i, K, V)$ is used to extract concept visual features.
    - **Design Motivation**: Leverages CLIP's vision-language alignment to precisely extract concept-relevant features from the image using the concept word as an anchor, rather than naively extracting global features.

2. **Mixture-of-Experts (MoE) Projection**:

    - **Function**: Maps extracted concept visual features accurately into the DiT modulation space.
    - **Mechanism**: Since different concept types (object vs. material vs. pose) exhibit substantially different mapping patterns, a single MLP is insufficient. Twelve expert MLPs are introduced, each handling concepts with similar mapping patterns. The routing mechanism adopts a parameter-free k-means clustering scheme — experts are assigned by clustering the neutral features of all concept words in the training set via k-means.
    - **Design Motivation**: Learnable linear gating networks suffer from imbalanced expert utilization; k-means routing avoids this issue in a simple and effective manner.

3. **VLM-Guided Pre-training**:

    - **Function**: Provides a good initialization for Mod-Adapter and bridges the large gap between the concept image space and the DiT modulation space.
    - **Mechanism**: A VLM generates detailed descriptions $p^+$ of concept images (e.g., "transparent cyan-green glass surface"), which are encoded as supervision signals in the modulation space. The pre-training loss is $\mathcal{L}_{\text{pretrain}} = \frac{1}{N}\sum_{i=1}^N \|F_i^+ - \mathcal{M}(\text{CLIP}(p^+))\|_2^2$.
    - **Design Motivation**: Pre-training does not require DiT forward passes, making it lightweight and efficient; the strong image understanding capability of VLMs provides high-quality semantic bridges.

### Loss & Training

The pre-training stage uses only $\mathcal{L}_{\text{pretrain}}$ (MSE loss) without connecting to DiT; the main training stage uses the standard diffusion denoising loss of FLUX. Training data includes MVImgNet (objects), AFHQ (animal faces), and FLUX self-distillation synthetic data (abstract concepts), totaling 106K images.

## Key Experimental Results

### Main Results

| Method | Multi-concept CP | Multi-concept PF | Multi-concept CP·PF | Single-concept CP·PF |
|--------|-----------------|-----------------|--------------------|--------------------|
| Emu2 | 0.53 | 0.48 | 0.25 | 0.42 |
| MIP-Adapter | 0.68 | 0.55 | 0.37 | 0.27 |
| MS-Diffusion | 0.62 | 0.51 | 0.32 | 0.23 |
| TokenVerse (tuning) | 0.56 | 0.56 | 0.31 | 0.38 |
| **Mod-Adapter** | **0.70** | **0.89** | **0.62** | **0.54** |

The multi-concept composite score CP·PF reaches 0.62, surpassing the second-best method MIP-Adapter (0.37) by **67.6%**.

### Ablation Study

| Configuration | Multi-concept CP·PF | Single-concept CP·PF |
|---------------|--------------------|--------------------|
| w/o k-means routing | 0.49 | 0.44 |
| w/o MoE | 0.35 | 0.42 |
| w/o VL-attn | 0.39 | 0.49 |
| w/o pre-training | 0.17 | 0.24 |
| **Full model** | **0.62** | **0.54** |

### Key Findings

- VLM pre-training is the **most critical** component — removing it causes CP·PF to drop sharply from 0.62 to 0.17, indicating that the gap from the image to the modulation space is substantial.
- MoE outperforms a single MLP (0.62 vs. 0.35), and k-means routing outperforms learnable routing (0.62 vs. 0.49).
- In a user study (32 participants, 4,000 votes), Mod-Adapter achieves a large margin of superiority over baselines on both CP and PF (multi-concept CP: 4.29/5, PF: 4.40/5).
- Existing tuning-free methods generally fail on abstract concepts, tending to copy-paste the original object rather than extracting abstract attributes.

## Highlights & Insights

- **First tuning-free method for abstract concept personalization**: By exploiting the locality and semantic additivity of the DiT modulation space, Mod-Adapter achieves unified and decoupled customization of both object and abstract concepts — a capability previously unavailable in tuning-free frameworks.
- **VLM-guided pre-training**: Using VLM image understanding as a semantic bridge to narrow the image-to-modulation gap is an elegant warm-up strategy. Since it does not require backpropagation through DiT, the pre-training overhead is minimal.
- **K-means MoE routing**: Replacing learnable gating with parameter-free clustering fundamentally resolves the expert imbalance problem; the approach is simple yet effective.

## Limitations & Future Work

- The model has 1.67B parameters; while it is the only trainable component, this is considerably heavier than textual inversion-based methods.
- Training data for abstract concepts is synthesized via FLUX self-distillation, which may limit data quality and diversity.
- Inference efficiency is not discussed — multi-concept inference requires a separate Mod-Adapter forward pass for each concept.
- The method is built on the FLUX architecture; transferring it to non-DiT architectures (e.g., U-Net) would require redesign.

## Related Work & Insights

- **vs. TokenVerse**: Both exploit the DiT modulation space, but TokenVerse requires per-image fine-tuning of an MLP, whereas Mod-Adapter is a tuning-free generalization.
- **vs. IP-Adapter/MIP-Adapter**: These methods inject image features via cross-attention but lack localized control, rendering them unable to handle abstract concepts.
- **vs. MS-Diffusion**: Employs a layout-guided scheme for multi-subject generation but is likewise limited to object concepts.
- The direction manipulation strategy in the modulation space may inspire other controllable generation tasks, such as affective control and style transfer.

## Rating

- Novelty: ⭐⭐⭐⭐ First tuning-free framework to unify object and abstract concept customization; leveraging the modulation space is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative + qualitative + user study + comprehensive ablation; inference efficiency analysis is absent.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, method description is thorough, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ High practical value; tuning-free multi-concept personalization has broad application scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Preventing Shortcuts in Adapter Training via Providing the Shortcuts](../../NeurIPS2025/image_generation/preventing_shortcuts_in_adapter_training_via_providing_the_shortcuts.md)
- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[ICLR 2026\] TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](tavae_a_vae_with_adaptable_priors_explains_contextual_modulation_in_the_visual_c.md)
- [\[ICLR 2026\] SongEcho: Towards Cover Song Generation via Instance-Adaptive Element-wise Linear Modulation](songecho_towards_cover_song_generation_via_instance-adaptive_element-wise_linear.md)

<!-- RELATED:END -->
