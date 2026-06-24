---
title: >-
  [Paper Note] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting
description: >-
  [Image Generation] EasyCraft proposes an end-to-end automatic avatar crafting framework. It maps facial images of arbitrary styles to a unified feature distribution using a general ViT encoder pre-trained with MAE, which are then converted into avatar customization parameters of game engines. Meanwhile, it integrates text-to-image technology to support text input, allowing easy adaptation to different game engines.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: c3e68639eb69d60a
---

# EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting

## TL;DR

EasyCraft proposes an end-to-end automatic avatar crafting framework. It maps facial images of arbitrary styles to a unified feature distribution using a general ViT encoder pre-trained with MAE, which are then converted into avatar customization parameters of game engines. Meanwhile, it integrates text-to-image technology to support text input, allowing easy adaptation to different game engines.

## Background & Motivation

Character customization ("avatar crafting") is a core feature of RPG games, allowing players to adjust facial structures, hairstyles, and makeup to create personalized virtual avatars. However, manual avatar crafting is time-consuming and difficult, and automated methods face the following challenges:

1. **Poor cross-engine generalization**: Existing methods (e.g., F2P, T2P) rely on semantic constraints (segmentation, perceptual loss, CLIP) from intermediate image domains. The effectiveness of these constraints is highly affected by the engine's style (realistic, anime, cartoon), requiring re-adaptation whenever the engine changes.
2. **Single input modality**: Most methods only support either image or text inputs, but not both.
3. **Limitations of prior work**: Because parameter controllers are non-differentiable, existing methods need to train neural renderers to simulate the parameter-to-image process, and then obtain parameters through optimization (inversion). However, the renderers can only constrain outputs belonging to a specific image domain.
4. **Distribution gap**: A significant distribution gap exists between the input images (real photos, anime drawings, cartoons) and the specific styles of game engines.

Mechanism: If the ViT encoder produces a unified feature distribution for images of different styles, then a translator trained solely on engine data can handle inputs of any style.

## Method

### Overall Architecture

EasyCraft comprises two main components: (1) a Translator that converts facial images into avatar parameters, whose ViT encoder is pre-trained via self-supervised MAE to obtain cross-style unified features, and (2) an engine-style Stable Diffusion model that transforms text descriptions into engine-style facial images, which are then passed to the translator for text-driven avatar crafting.

### Key Designs

#### 1. MAE Pre-trained General ViT Encoder

- **Function**: Encodes facial images of any style (real, anime, cartoon, game style) into a unified feature distribution.
- **Mechanism**: A large-scale dataset containing 5.1 million multi-style facial images (AffectNet, CelebA, SeePrettyFace, AnimeFace, Danbooru, etc.) is constructed to train the ViT encoder using an MAE self-supervised learning strategy. During training, 75% of the patches are randomly masked; the encoder processes the visible patches to generate tokens, and the decoder reconstructs the original image.
- **Design Motivation**: The key hypothesis is that MAE pre-training enables the ViT to learn a unified representation of facial structures and makeup characteristics across different styles. Since the reconstruction objective of MAE does not depend on the image style but rather requires understanding structural facial information (e.g., eye positions, nose width), which is shared across styles. After pre-training, the encoder parameters are frozen to prevent feature distribution shift during subsequent training on engine data.

#### 2. Engine-Specific Translator

- **Function**: Maps unified facial features to the specific parameter space of a game engine (facial structures, makeup textures, and makeup attributes).
- **Mechanism**: The translator consists of a frozen ViT encoder $\mathcal{T}_e$ and a trainable parameter generation module $\mathcal{T}_p$. The module $\mathcal{T}_p$ contains three parallel MLP networks that respectively output facial structure parameters $\delta_s$ (continuous values, supervised by $L_1$ loss), makeup texture parameters $\delta_t$ (discrete values, supervised by cross-entropy loss), and makeup attribute parameters $\delta_a$ (continuous values with a conditional mask, supervised by $L_1$ loss). The training data is generated completely by randomly sampling parameter-rendered image pairs from the engine.
- **Design Motivation**: By freezing the pre-trained encoder and training only light MLPs, a "pre-train once, adapt to multiple engines" paradigm is achieved. Since the feature distribution output by the encoder is consistent across different styles, the MLP trained solely on engine data can also handle non-engine style inputs. For a new engine, only the MLP parameter generator needs to be retrained.

#### 3. Engine-Style Stable Diffusion

- **Function**: Generates facial images matching the visual style of the game engine, working in tandem with the translator to perform text-driven avatar crafting.
- **Mechanism**: An engine-specific dataset is constructed by gathering 7,000 randomly rendered images from the engine and using GPT-4o to generate descriptive captions. Then, the UNet and text encoder of SD v1.5 are fine-tuned. The fine-tuned SD model can generate facial images that conform to the engine's style, while retaining the rich semantic diversity of SD.
- **Design Motivation**: Directly using native SD leads to a massive style discrepancy from the engine (e.g., mismatched makeup features) and cannot guarantee consistent generation of frontal faces. The engine-style SD ensures that generated images align with the translator's training domain while leveraging SD's diversity to support varied output results under the same text description.

### Loss & Training

Translator training loss:

$$\mathcal{L} = \alpha\|\delta_s - \hat{\delta}_s\|_1 + \gamma\|\delta_a \cdot \mathcal{M} - \hat{\delta}_a \cdot \mathcal{M}\|_1 - \lambda\delta_t \cdot \log(\hat{\delta}_t})$$

where $\alpha=5, \gamma=1, \lambda=0.1$, and $\mathcal{M}$ is a conditional mask indicating the validity of makeup attribute parameters. Data augmentations such as random cropping, rotation, color jitter, and Gaussian blur are also applied during training.

## Key Experimental Results

### Main Results

**Quantitative comparison of image-to-parameter avatar crafting (Tab. 1)**:

| Method | Justice Mobile ID Sim↑ | Justice FID↓ | Naraka ID Sim↑ | Naraka FID↓ | Speed↓ |
|------|----------------------|-------------|---------------|-------------|------|
| F2P | 0.376 | 40.69 | 0.334 | 42.20 | 1.140s |
| F2P v2 | 0.275 | 34.27 | 0.217 | 33.04 | 0.007s |
| **Ours** | 0.351 | **17.65** | 0.316 | **18.32** | 0.026s |

**Quantitative comparison of text-driven avatar crafting (Tab. 2)**:

| Method | Justice LPIPS↑ | Justice FID↓ | CLIP Score↑ | Naraka FID↓ | Speed↓ |
|------|---------------|-------------|-------------|-------------|------|
| T2P | 0.027 | 32.9 | 0.211 | 33.51 | 1.725s |
| **Ours** | **0.093** | **18.76** | **0.241** | **19.43** | **0.643s** |

### Ablation Study

**Ablation study on ViT pre-training (Tab. 3 & 4)**:

| Variant | Image ID Sim↑ | Image FID↓ | Text FID↓ |
|------|-----------------|----------|----------|
| w/o pretrain | 0.243 | 97.31 | 18.73 |
| **Full model** | **0.351** | **17.65** | 18.76 |

### Key Findings

1. **Significant lead in FID**: The FID of image-driven crafting drops sharply from F2P's 40.69/42.20 to 17.65/18.32, indicating that the generated avatars are much closer to the quality of those created by real players.
2. **Crucial role of ViT pre-training**: Removing pre-training causes the image-driven crafting FID to skyrocket from 17.65 to 97.31, but has minimal impact on text-driven crafting (18.76 vs 18.73), because the engine-style SD ensures the input style aligns with the training domain.
3. **Real-time performance**: An inference speed of 0.026s easily supports real-time applications.
4. An 87% user preference rate in the user study (vs. 11% for F2P and 2% for F2P v2).

## Highlights & Insights

1. **The core hypothesis of a unified feature distribution is simple yet effective**: Instead of relying on complex domain adaptation or multi-domain supervision, a cross-style unified representation is directly obtained via self-supervised pre-training, after which the encoder is frozen, and only light translators are trained.
2. **Extremely simplified training flow**: No paired real-to-virtual data is required. The training data is generated purely via random sampling in the engine, which offers extreme scalability.
3. **Modular design facilitates industrial deployment**: The MAE pre-training is performed once and can be reused. Adapting to a new engine only requires training an MLP and fine-tuning SD, which incurs low costs.
4. **Ablation studies reveal the design logic**: ViT pre-training is critical for image inputs but irrelevant for text inputs, whereas the engine-style SD is critical for the accuracy of text-driven avatar crafting.

## Limitations & Future Work

1. **Suboptimal identity similarity**: F2P achieves a higher ID Sim (0.376 vs 0.351) by directly supervising with identity similarity. However, F2P severely fails on non-realistic images.
2. **Limited engine-style SD training data**: Fine-tuning with only 7,000 images might lead to an incomplete understanding of detailed makeup.
3. **MAE pre-training cost**: Training on 5.1 million images for two weeks on 8 A100 GPUs incurs a non-negligible one-time cost.
4. **Facial-only support**: It does not cover other character customization dimensions such as hairstyle and clothing.
5. Future work can explore the feasibility of replacing MAE with larger pre-trained models (e.g., DINOv2).

## Related Work & Insights

- **F2P / F2P v2**: Optimization-based image-to-parameter methods that rely on intermediate domain constraints.
- **T2P**: A text-driven avatar crafting method based on CLIP constraints.
- **MAE**: A self-supervised vision pre-training method. EasyCraft leverages its capability to learn general visual features.
- **Stable Diffusion**: A text-to-image diffusion model, adapted to specific styles via fine-tuning.
- Insight: The unified feature space produced by self-supervised pre-training could be a general approach to solving cross-domain transfer problems, which is worth exploring in other gaming or virtual avatar tasks.

## Rating: ⭐⭐⭐⭐

The problem addressed is highly practical (a rigid demand in the gaming industry), the method is simple yet effective and scales well to new engines, and experiments on two commercial games have validated its practicality. One star is deducted because the identity similarity is not optimal, and it is limited only to facial customization without involving the complete block of character creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)

</div>

<!-- RELATED:END -->
