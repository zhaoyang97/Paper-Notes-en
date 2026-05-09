---
title: >-
  [Paper Note] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception
description: >-
  [CVPR 2026][Image Generation][Image Aesthetics] Ours proposes the DIAE framework, which transforms vague aesthetic instructions into joint signals of HSV/contour maps and text via Multimodal Aesthetic Perception (MAP). It leverages an "imperfectly paired" dataset, IIAEData, to achieve weakly supervised image aesthetic enhancement.
tags:
  - CVPR 2026
  - Image Generation
  - Image Aesthetics
  - Diffusion Models
  - Multimodal Perception
  - Weakly Supervised
  - ControlNet
date: 2026-05-08
content_hash: e9193501b37ba9a1
---

# Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**Conference**: CVPR 2026  
**arXiv**: [2603.11556](https://arxiv.org/abs/2603.11556)  
**Code**: None  
**Area**: Image Aesthetic Enhancement / Diffusion Models  
**Keywords**: Image Aesthetics, Diffusion Models, Multimodal Perception, Weakly Supervised, ControlNet

## TL;DR

Ours proposes the DIAE framework, which transforms vague aesthetic instructions into joint signals of HSV/contour maps and text via Multimodal Aesthetic Perception (MAP). It leverages an "imperfectly paired" dataset, IIAEData, to achieve weakly supervised image aesthetic enhancement.

## Background & Motivation

Image Aesthetic Enhancement (IAE) requires models to possess both aesthetic assessment and attribute enhancement capabilities. Existing diffusion models face two major challenges in this task:

**Limitations of Aesthetic Perception**: Aesthetics represent high-level human visual capabilities influenced by uncontrollable factors like culture, experience, and emotion. Simple text encoders struggle to understand abstract aesthetic descriptions such as "insufficient saturation" or "rule-of-thirds composition."

**Limitations of Prior Work (Data Scarcity)**: Full supervision requires "perfectly paired" images—identical in content but different in aesthetic quality—which necessitates professional photographers/artists for manual editing, incurring extremely high costs.

The authors note that existing image editing methods (InstructPix2Pix, MGIE, etc.) show limited effectiveness in aesthetic enhancement due to a lacks of specialized aesthetic perception, while reinforcement learning-based methods (DOODL, DiffusionDPO) guide generation directions without sufficient precision.

## Method

### Overall Architecture

Based on the Stable Diffusion v1.5 + ControlNet architecture, the framework consists of three core modules: (1) IIAEData dataset construction, (2) Multimodal Aesthetic Perception (MAP), and (3) a dual-branch supervision framework.

### Key Designs

1.  **IIAEData Dataset**: Construction of an "imperfectly paired" dataset where images share the same semantics but differ in aesthetic quality:
    - Collected images from AVA, TAD66K, KonIQ, and FLICKR.
    - High MOS images serve as reference images; low MOS images serve as input images.
    - LLaVA-13B generates image descriptions, and pairs are formed based on semantic matching.
    - UNIAA-LLaVA generates standardized aesthetic evaluations across dimensions like color, lighting, and composition.
    - Resulted in **47.5K samples** (45K training + 1.5K testing).

2.  **Multimodal Aesthetic Perception (MAP)**: Translates vague aesthetic descriptions into actionable multimodal signals:
    - **Color Attributes**: HSV maps as visual representations (directly reflecting saturation, value, hue) + color aesthetic text descriptions.
    - **Structural Attributes**: HED contour maps as visual representations (reflecting focus, composition, shooting style) + structural aesthetic text descriptions.
    - Visual embeddings extracted via CNN: $F^I_{col} = \Phi_i(I_{col}), F^I_{str} = \Phi_i(I_{str})$
    - Text embeddings encoded via CLIP: $F^T_{col} = \Phi_t(T_{col}), F^T_{str} = \Phi_t(T_{str})$
    - Injected into the diffusion model via the ControlNet "adding structure" mechanism.

3.  **Dual-Branch Supervision Framework**: Addresses the weakly supervised training of "imperfectly paired" data:
    - Training is divided into two stages via a timestep parameter $t_s$ (default 900/1000).
    - $t \leq t_s$: Semantic supervision branch, supervised by the input image (maintaining content consistency).
    - $t > t_s$: Aesthetic supervision branch, supervised by the reference image (introducing high aesthetic attributes).

    $$L = L_{ref} + \lambda L_{inp}$$

    Where $L_{ref}$ is supervised by the reference image across all timesteps, and $L_{inp}$ is supervised by the input image only in early timesteps.

### Loss & Training

$$L_{ref} = \|\epsilon_{ref} - \epsilon_\theta(x_{ref}(t), c, x_{ref}, t, cond)\|_2^2$$
$$L_{inp} = \|\epsilon_{inp} - \epsilon_\theta(x_{inp}(t\%t_s), c, x_{inp}, t\%t_s, cond)\|_2^2$$

- Base Model: Stable Diffusion v1.5
- Optimizer: AdamW, Learning Rate 1e-5
- Training: 100K iterations on 4×A800 GPUs
- Inference: 50 denoising steps (~4s for 256×256, ~9s for 512×512)

## Key Experimental Results

### Main Results

| Method | LAIONs(256) | LAIONs(512) | MLLMs(256) | MLLMs(512) | CLIP-I(256) | CLIP-I(512) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Original | 4.962 | 5.123 | 3.243 | 3.300 | 1.000 | 1.000 |
| ControlNet | 4.979 | 5.522 | 3.271 | 3.415 | 0.628 | 0.617 |
| InstructPix2Pix | 4.991 | 5.396 | 3.264 | 3.325 | 0.764 | 0.690 |
| MGIE | 4.947 | 5.519 | 3.045 | 3.411 | 0.557 | 0.770 |
| DOODL | 5.102 | 5.140 | 3.255 | 3.297 | 0.775 | 0.703 |
| **Ours** | **5.324** | **6.012** | **3.339** | **3.662** | **0.772** | **0.784** |

At 512 resolution, LAIONs improved by 17.4%, and MLLMs improved by 11.0%.

### Ablation Study

| Configuration | LAIONs | MLLMs | CLIP-I | Description |
| :--- | :--- | :--- | :--- | :--- |
| Ours (w/o v) | 5.250 | 3.343 | 0.623 | Remove visual guidance |
| Ours (w/o t) | 5.428 | 3.410 | 0.792 | Remove text evaluation |
| **Ours** | **5.668** | **3.501** | **0.778** | Complete MAP |

### Key Findings

- Dual visual modalities (HSV + contour maps) contribute significantly to content consistency (CLIP-I 0.792 vs. 0.623).
- The text modality is more critical for aesthetic improvement (LAIONs 5.668 vs. 5.428).
- A larger $t_s$ parameter results in generated images closer to the input (retaining more original aesthetic attributes).
- Gains are particularly significant for images with low aesthetic quality (MOS < 4.0).

## Highlights & Insights

- **Practical Weakly Supervised Solution**: Replaces unavailable "perfect pairs" with "imperfect pairs," using a dual-branch framework to gracefully handle the decoupling of content and aesthetics.
- **Targeted MAP Design**: Mapping HSV to color aesthetics and contour maps to structural aesthetics aligns better with the perceptual dimensions of aesthetic assessment than using raw RGB.
- **Integration with MLLMs**: Ours can integrate with aesthetic-understanding MLLMs to automatically generate MAP inputs, forming an end-to-end pipeline.

## Limitations & Future Work

- **Portrait Scenarios Excluded**: Aesthetic quality in portraits involves complex factors like facial features and body posture, which the authors explicitly excluded.
- **Older SD v1.5 Backbone**: Generation quality lags behind newer models like SDXL or SD3.
- **Semantic Matching Quality**: The quality of paired data is limited by the accuracy of the captions generated by LLaVA.
- The $t_s$ parameter requires manual tuning; different scenarios may necessitate different settings.

## Related Work & Insights

- Unlike DOODL or RAHF, which use MOS as a classifier to guide generation, ours directly controls aesthetic attribute editing via multimodal conditions, offering finer precision.
- The dual-branch supervision concept is inspired by the content-style decoupling in StyleDiffusion but is concretized as semantic/aesthetic separation at early/late timesteps.
- Insight: This framework could be extended to video aesthetic enhancement or combined with personalized user preferences.

## Rating

- Novelty: ⭐⭐⭐ MAP module is creative, but the overall technical stack is standard (ControlNet + SD v1.5).
- Experimental Thoroughness: ⭐⭐⭐ Evaluation dimensions are complete, but comparisons with SOTA are slightly dated, and a user study is missing.
- Writing Quality: ⭐⭐⭐ Clear structure, but some details (e.g., selection of $\lambda$) are insufficiently explained.
- Value: ⭐⭐⭐ Image aesthetic enhancement has strong practical application value, and the "imperfectly paired" data scheme is a useful reference.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Enhancing Spatial Understanding in Image Generation via Reward Modeling](enhancing_spatial_understanding_in_image_generation_via_reward_modeling.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] Reviving ConvNeXt for Efficient Convolutional Diffusion Models](reviving_convnext_for_efficient_convolutional_diffusion_models.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

<!-- RELATED:END -->
