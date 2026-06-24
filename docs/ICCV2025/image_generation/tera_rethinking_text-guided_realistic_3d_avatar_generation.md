---
title: >-
  [Paper Note] TeRA: Rethinking Text-guided Realistic 3D Avatar Generation
description: >-
  [ICCV 2025][Image Generation][3D Avatar] TeRA is proposed as the first text-guided 3D realistic avatar generation framework based on a latent diffusion model. By distilling a large-scale human reconstruction model to construct a structured latent space, TeRA generates realistic 3D human avatars in 12 seconds—two orders of magnitude faster than SDS-based methods.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "3D Avatar"
  - "Latent Diffusion"
  - "Text-to-3D"
  - "SMPL-X"
  - "UV Gaussian"
date: 2026-05-08
content_hash: a5c597a01d00efb2
---

# TeRA: Rethinking Text-guided Realistic 3D Avatar Generation

**Conference**: ICCV 2025
**arXiv**: [2509.02466](https://arxiv.org/abs/2509.02466)  
**Code**: [Available](https://yanwen-w.github.io/TeRA-Page/)  
**Area**: Image Generation / 3D Avatar Generation
**Keywords**: 3D Avatar, Latent Diffusion, Text-to-3D, SMPL-X, UV Gaussian

## TL;DR

TeRA is proposed as the first text-guided 3D realistic avatar generation framework based on a latent diffusion model. By distilling a large-scale human reconstruction model to construct a structured latent space, TeRA generates realistic 3D human avatars in 12 seconds—two orders of magnitude faster than SDS-based methods.

## Background & Motivation

3D avatar creation is a critical requirement for the metaverse, film, gaming, and AR/VR applications. Existing approaches face dilemmas along two paradigms:

**SDS-based methods (TADA, HumanGaussian, HumanNorm, etc.)**:
- Advantages: leverage rich human priors from 2D diffusion models; no 3D training data required.
- Limitations: ① iterative optimization is extremely slow (hours per scene); ② lack of explicit 3D structure in 2D models leads to multi-view inconsistency; ③ oversaturation, cartoonization, and distorted proportions.

**General large-scale 3D generation models**:
- Training data contains an excess of cartoon assets and scarce realistic human models, resulting in severe style bias.
- Unable to generate photorealistic 3D humans.

Core idea: **train a native 3D diffusion model directly on 3D human data**. The key challenge is constructing an efficient 3D human latent space suitable for diffusion model learning.

## Method

### Overall Architecture

TeRA adopts a two-stage training pipeline:

**Stage 1: Distillation Decoder**
- Distills a compact structured latent space from the large-scale human reconstruction model IDOL.
- IDOL encodes input images into UV-aligned features, but at an excessively high resolution (1536×1536).

**Stage 2: Structured Latent Diffusion Model**
- Trains a text-conditioned diffusion model within the distilled latent space (256×256).
- Generates UV feature maps from noise and decodes them into 3D Gaussian human avatars.

### Key Designs

**1. 3D Human Representation: UV-Structured Gaussians**

The 3D human is represented using SMPL-X-aligned UV-structured Gaussians:
- Each Gaussian position is initialized to an SMPL-X mesh vertex.
- A neural network predicts offset values (position, rotation, and scale offsets) as well as color and opacity.
- All attributes are stored in multi-channel attribute maps in the SMPL-X UV space.

Supports: ① direct animation driving, ② texture editing, ③ shape editing.

**2. Distillation-based Latent Encoding**

Directly training a VAE on complex 3D human models is unstable and computationally expensive. TeRA's alternative:

- Uses the IDOL encoder to extract UV features (which already exhibit good structural properties and generalization).
- Downsamples UV features from 1536×1536 to 256×256.
- Trains a compact convolutional distillation decoder: upsamples to 1024×1024 and decodes geometry and color attributes via two branches.
- Avoids the instability and posterior collapse associated with training a VAE.

Training loss:

$$L_{dist} = \sum_{i=1}^{N} (\|I_{pred} - I_{gt}\|^2 + \lambda_{vgg}L_{vgg}) + \lambda_{offset}\|G_{offset}\|^2$$

**3. Text Annotation Pipeline**

Large vision-language models are used to generate precise text annotations for the HuGe100K dataset:
- Qwen2.5-VL processes front/back/left/right multi-view images to obtain descriptions of individual body parts.
- Qwen2.5 extracts key information to produce refined descriptions of ≤40 words.
- Five additional phrase-level descriptions of varying lengths (8–16 words) are generated.

**4. Structured Latent Diffusion Model**

- **Text encoding**: CLIP text encoder, 77 tokens × 768 dimensions.
- **Noise schedule**: DDPM, 1000 training steps, 100 inference steps.
- **Prediction target**: $x_0$-prediction.
- **Classifier-free guidance**: text condition dropped with 20% probability.

**5. Structure-Aware Editing (Virtual Try-On)**

Leverages the inpainting capability of the diffusion model:
- Body regions to be preserved (background) are kept in the latent space.
- Garment regions (foreground) are denoised from noise under the guidance of a target text prompt.
- At each step, the clean background is re-noised to the corresponding timestep and merged with the foreground.
- Produces seamlessly transitioned outfit transfer results.

### Loss & Training

- **Stage 1**: 4× RTX A6000, batch=2, L2 + VGG + offset regularization.
- **Stage 2**: 4× RTX 3090, batch=8, DDPM 1000 steps, MSE loss.
- **Total training time**: approximately 90 hours.
- **Inference time**: 12 seconds (single RTX 3090).

## Key Experimental Results

### Main Results (Table)

| Method | CLIP Score↑ | VQA Score↑ | Text Consistency↑ | Visual Quality↑ | Realism↑ | Time↓ |
|--------|------------|-----------|------------------|----------------|---------|------|
| TADA | 29.86 | 0.64 | 3.27 | 2.25 | 2.11 | 2.3h |
| X-Oscar | 32.46 | 0.80 | 3.56 | 2.54 | 2.26 | 2.0h |
| HumanGaussian | 29.31 | 0.82 | 3.74 | 2.49 | 2.28 | 1.0h |
| HumanNorm | 29.94 | 0.72 | 3.79 | 3.01 | 3.04 | 4.0h |
| **TeRA** | 30.17 | **0.82** | **4.54** | **4.33** | **4.35** | **12s** |

TeRA comprehensively outperforms all baselines in user ratings: text consistency 4.54 vs. 3.79, visual quality 4.33 vs. 3.01, realism 4.35 vs. 3.04. Speed is improved by two orders of magnitude.

### Ablation Study (Table)

| Ablation | Result |
|----------|--------|
| Latent space 128×128 vs. 256×256 | 256 resolution yields richer detail and fewer artifacts |
| Direct feature replacement vs. inpainting editing | Inpainting produces more natural transitions with fewer artifacts |

### Key Findings

1. **Fundamental limitations of SDS**: SDS baselines are consistently oversaturated and proportionally distorted; even HumanNorm's improved geometry still exhibits artifacts on faces and hands.
2. **Feed-forward vs. iterative**: Single-pass feed-forward generation is not only 100× faster but also avoids the multi-view inconsistency inherent to SDS.
3. **Necessity of distillation**: Directly connecting the diffusion model to the VAE encoder causes posterior collapse; the distillation module is essential.
4. **Effect of latent space resolution**: 256×256 significantly reduces artifacts compared to 128×128.

## Highlights & Insights

1. **Paradigm shift**: From "guiding 3D optimization with 2D diffusion models" to "training a diffusion model directly in 3D space."
2. **Distillation over VAE**: Cleverly repurposes the encoding space of an existing large-scale reconstruction model (IDOL), avoiding the instability of training a VAE from scratch.
3. **Multiple benefits of structured representation**: UV Gaussians natively support downstream applications including animation driving, editing, and virtual try-on.
4. **VLM annotation pipeline**: The collaborative annotation scheme of Qwen2.5-VL + Qwen2.5 provides a reusable solution for large-scale 3D dataset text annotation.
5. **12-second generation**: Achieves production-level speed on a single RTX 3090.

## Limitations & Future Work

1. **Static models**: Training data consists of static 3D humans; dynamic details such as cloth wrinkles induced by motion cannot be modeled.
2. **Loose garments**: Dependence on SMPL-X representation limits the modeling quality of loose clothing such as skirts.
3. **Dataset dependency**: Requires large-scale 3D human datasets such as HuGe100K.
4. **Hand and facial detail**: Although superior to SDS-based methods, fidelity in these regions still has room for improvement.
5. **Single-person limitation**: Currently only supports single-person generation.

## Related Work & Insights

- **IDOL**: A large-scale human reconstruction model; TeRA distills its encoding space as the latent space foundation.
- **HuGe100K**: A dataset of 100K realistic 3D humans providing the required training data.
- **Stable Diffusion / LDM**: The latent diffusion model framework; TeRA extends it from 2D images to 3D humans.
- **TADA**: An SDS-based method using SMPL-X + UV displacement maps, representing the previous SOTA paradigm.
- **HumanNorm**: An SDS-based method introducing normal diffusion to partially alleviate geometry issues.
- Insight: **Domain-specific large-scale reconstruction models can serve as "encoder priors" for generative models**; the distillation approach is more practical than training a VAE from scratch.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4.5 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Practical Value | 4.5 |
| Overall | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FaceCraft4D: Animated 3D Facial Avatar Generation from a Single Image](facecraft4d_animated_3d_facial_avatar_generation_from_a_single_image.md)
- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ECCV 2024\] RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models](../../ECCV2024/image_generation/rodinhd_high-fidelity_3d_avatar_generation_with_diffusion_models.md)
- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](../../CVPR2025/image_generation/nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)

</div>

<!-- RELATED:END -->
