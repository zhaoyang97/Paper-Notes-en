---
title: >-
  [Paper Note] Generative Zoo
description: >-
  [ICCV 2025][LLM Evaluation][Synthetic data generation] A scalable pipeline is proposed for synthesizing animal 3D pose and shape training data using conditional image generation models (FLUX + ControlNet)…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Synthetic data generation"
  - "animal pose estimation"
  - "SMAL"
  - "conditional image generation"
  - "ControlNet"
date: 2026-05-08
content_hash: 9af5bd76c8564a27
---

# Generative Zoo

**Conference**: ICCV 2025
**arXiv**: [2412.08101](https://arxiv.org/abs/2412.08101)  
**Code**: [https://genzoo.is.tue.mpg.de](https://genzoo.is.tue.mpg.de)  
**Area**: LLM Evaluation
**Keywords**: Synthetic data generation, animal pose estimation, SMAL, conditional image generation, ControlNet

## TL;DR

A scalable pipeline is proposed for synthesizing animal 3D pose and shape training data using conditional image generation models (FLUX + ControlNet), producing the million-scale GenZoo dataset. Training exclusively on synthetic data achieves state-of-the-art performance on real-world benchmarks.

## Background & Motivation

3D animal pose and shape estimation faces a severe training data bottleneck:
- **Real annotations are difficult to obtain**: Animals cannot cooperate with multi-view MoCap systems or marker-based setups as humans do, and in-the-wild collection is impractical.
- **2D annotations → 3D pseudo-labels are unreliable**: Manually annotated 2D keypoints and silhouettes are used to optimize SMAL parameters, but monocular 3D fitting is an ill-posed problem, and silhouette alignment does not guarantee physically plausible pose or shape.
- **Traditional synthetic data pipelines are costly**: Game-engine-based rendering requires extensive manual 3D assets; adding new species or environments demands redesign, and achieving both visual realism and diversity is challenging.

The authors propose **replacing traditional rendering engines with conditional image generation models**: adding a new species requires only modifying the text prompt, while precise control over 3D parameters is maintained.

## Method

### Overall Architecture

The pipeline proceeds as follows: sample species name → sample shape parameters ($\beta$) → sample pose parameters ($\theta$) → render control signals via Pyrender → describe orientation with a VLM and synthesize prompts with an LLM → generate the final image with FLUX + ControlNet. Each generated image is paired with precise SMAL pose/shape ground truth.

### Key Designs

1. **Species Sampling**:

    - Samples from the superorder Laurasiatheria in the Mammal Diversity Database (excluding order Eulipotyphla, whose morphology cannot be represented by SMAL's fixed skeletal topology).
    - Special handling for 247 dog breeds: canine breeds exhibit large inter-breed morphological variation, and are balanced with other species at a 50:50 sampling ratio.
    - **Core advantage**: Adding a new species requires only a text prompt, with no 3D assets needed.

2. **Shape & Pose Sampling**:

    - **Shape**: Rather than sampling $\beta$ parameters directly (which may yield implausible shapes), samples are drawn in CLIP embedding space and decoded via the AWOL model. For each species, CLIP embeddings of 128 appearance descriptions are computed, a multivariate Gaussian is fitted, and samples are drawn to balance realism and diversity.
    - **Pose**: In the absence of animal MoCap data, BITE (an optimization-based canine pose estimation method) is applied to a large collection of online dog images to extract a pseudo-pose set. Dog poses are found to transfer reasonably to other quadrupeds.

3. **Prompt Synthesis and Conditional Generation**:

    - SMAL models are rendered with Pyrender to produce raw images, which are fed into the Molmo-7B VLM to obtain orientation descriptions.
    - Species name, camera settings, and scene descriptions are combined, and Qwen2.5-7B LLM synthesizes a coherent prompt.
    - **FLUX + ControlNet** with dual control signals (Canny edges + depth maps) is used to generate 1024×1024 images.
    - Depth-only yields higher realism but poor pose alignment; Canny-only achieves good alignment but lower realism; the dual-signal combination balances both.

### Loss & Training

The regression model (ViTPose backbone) employs three losses:
- 2D joint projection L1 loss (weight 0.01)
- 9D rotation matrix MSE loss (after symmetric orthogonalization; weight 100 for `body_pose` and `global_orient`)
- Vertex transformation L1 loss after applying $\beta$ (weight 50)
- Batch size 128, single GPU, with early stopping based on validation set 2D joint projection loss.

## Key Experimental Results

### Main Results

Comparison of methods on the Animal3D real-world benchmark:

| Method | PCK@0.5↑ | S-MPJPE↓ | PA-MPJPE↓ |
|--------|----------|----------|-----------|
| HMR* | 63.1 | 496.2 | 124.8 |
| PARE* | 85.6 | 374.9 | 127.2 |
| WLDO* | 65.1 | 484.0 | 123.9 |
| Ours (ResNet) | 95.11 | 201.1 | 132.67 |
| **Ours (ViTPose)** | **97.0** | **160.1** | **116.6** |

S-MPJPE decreases from 374.9 to 160.1 (a 57% reduction), using only synthetic training data.

### Ablation Study

Impact of individual components on performance (trained with 100K samples):

| Configuration | PCK@0.5↑ | S-MPJPE↓ | PA-MPJPE↓ | S-V2V↓ | PA-V2V↓ |
|---------------|----------|----------|-----------|--------|---------|
| Full | 97.1 | 166.9 | 118.4 | 59.3 | 50.2 |
| -Depth | 96.7 | 184.1 | 135.1 | 95.4 | 65.9 |
| -Canny | 96.2 | 172.3 | 119.4 | 57.7 | 39.1 |
| -Caption | 96.9 | 167.1 | 120.1 | 71.0 | 48.6 |
| -LLM | 97.2 | 168.2 | 120.7 | 69.4 | 49.7 |

Image generation model ablation (FLUX vs. Hunyuan-DiT vs. SD3): FLUX achieves the best performance on most 3D metrics.

### Key Findings

- **Synthetic data can surpass training on real pseudo-labels**: GenZoo, trained purely on synthetic data, achieves state-of-the-art on Animal3D.
- Animal3D ground truth itself contains implausible 3D annotations (a perceptual study finds that human raters prefer model predictions over GT in 27% of cases).
- Data scaling follows a log-linear growth trend with diminishing returns, suggesting a performance ceiling on the Animal3D benchmark.
- The balance between Depth and Canny dual control signals is critical: depth maps ensure realism while Canny edges ensure pose alignment.

## Highlights & Insights

- **Paradigm innovation**: Replacing the traditional rendering pipeline with text-driven conditional image generation reduces adding a new species from "designing 3D assets" to "writing a prompt."
- The **CLIP-space shape sampling** design is elegant: sampling in embedding space rather than parameter space balances shape realism and diversity.
- Perceptual experiments reveal annotation quality issues in the Animal3D benchmark — model predictions are more plausible than GT in side-view cases.
- Construction of the million-scale dataset demonstrates the scalability of the proposed approach.

## Limitations & Future Work

- Strong occlusion may cause erroneous detection (e.g., regressing on a foreground human instead of the animal).
- Pose sampling is derived from dog images, resulting in insufficient coverage of species-specific poses (e.g., grooming postures in cats).
- SMAL's fixed skeletal topology limits the range of representable species (e.g., elephant trunks cannot be modeled).
- FLUX has limited understanding of rare species and may generate visually similar but taxonomically incorrect images.
- High-quality real-world 3D annotation benchmarks are lacking.

## Related Work & Insights

- The SMAL/SMAL+/AWOL ecosystem constitutes the core infrastructure for animal body modeling, analogous to the SMPL ecosystem for human bodies.
- BEDLAM (CVPR 2023) conducted similar work on synthetic human data; the present paper transfers this idea to the animal domain.
- The dual-signal ControlNet strategy has direct reference value for other synthetic data generation tasks.
- The proposed GenZoo-Felidae test set (excluding 47 feline species seen during training) provides a more rigorous generalization evaluation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of replacing rendering pipelines with generative models is novel and practically motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional ablations are comprehensive and perceptual experiments are convincing, though real-world validation across more species is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Pipeline description is clear and motivation is well-developed.
- **Value**: ⭐⭐⭐⭐ Open-sourcing the million-scale dataset and pipeline directly advances the animal behavior analysis community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](a_real-world_display_inverse_rendering_dataset.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[ICCV 2025\] Degradation-Modeled Multipath Diffusion for Tunable Metalens Photography](degradation-modeled_multipath_diffusion_for_tunable_metalens_photography.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)

</div>

<!-- RELATED:END -->
