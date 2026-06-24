---
title: >-
  [Paper Note] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions
description: >-
  [ECCV 2024][3D Vision][Monocular Depth Estimation] Leverages a text-to-image diffusion model (ControlNet/T2I-Adapter) to transform clean-weather images into adverse-condition images while preserving the same 3D structure. The existing monocular depth estimation networks are fine-tuned via self-distillation to uniformly address out-of-distribution challenges like adverse weather and non-Lambertian surfaces.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Diffusion Models"
  - "Data Augmentation"
  - "Adverse Weather"
  - "Non-Lambertian Surfaces"
date: 2026-05-08
content_hash: 71dd7e4c18a290fb
---

# Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions

**Conference**: ECCV 2024  
**arXiv**: [2407.16698](https://arxiv.org/abs/2407.16698)  
**Code**: [https://diffusion4robustdepth.github.io/](https://diffusion4robustdepth.github.io/)  
**Area**: 3D Vision  
**Keywords**: Monocular Depth Estimation, Diffusion Models, Data Augmentation, Adverse Weather, Non-Lambertian Surfaces

## TL;DR

Leverages a text-to-image diffusion model (ControlNet/T2I-Adapter) to transform clean-weather images into adverse-condition images while preserving the same 3D structure. The existing monocular depth estimation networks are fine-tuned via self-distillation to uniformly address out-of-distribution challenges like adverse weather and non-Lambertian surfaces.

## Background & Motivation

**Background**: Monocular depth estimation has made great progress with deep learning. SOTA models such as DPT and Depth Anything perform exceptionally well in typical conditions after being trained on large-scale mixed datasets. Training schemes include LiDAR supervision, self-supervision (stereo pairs / video sequences), etc.

**Limitations of Prior Work**: Even the most robust generalist models face severe performance degradation on out-of-distribution data (long-tail scenarios)—such as adverse weather (rain, night, fog) and non-Lambertian surfaces (transparent/reflective objects). This is because: (1) high-quality annotated data is extremely scarce (LiDAR is unreliable in rain/snow and fails on transparent objects); (2) active sensors themselves fail under these challenging conditions.

**Key Challenge**: Training robust models requires vast amounts of depth-annotated data under adverse conditions, yet such labels are virtually impossible to acquire. Existing methods typically design separate, dedicated solutions for different challenges (e.g., GANs for night-time, specialized methods for transparent objects), lacking a unified framework.

**Goal**: To propose a unified framework that simultaneously addresses adverse weather and non-Lambertian surfaces without requiring real-world adverse-condition data, starting solely from clean-weather images.

**Key Insight**: A depth-conditioned text-to-image diffusion model can transform clean-weather scenes into arbitrary complex conditions via text prompts while maintaining the 3D structure (depth map) unchanged. This implies that depth labels from clean images can be directly transferred to the generated adverse-condition images.

**Core Idea**: Use diffusion models for "condition-preserving style transfer"—generating adverse-condition images that maintain the same depth structure as clean-weather images, using the original model's depth predictions on clean images as pseudo-labels to fine-tune the model via self-distillation for adaptation to adverse scenes.

## Method

### Overall Architecture

The pipeline consists of two stages:
1. **Data Generation Stage**:
    - Select a clean image $e_i$ (normal weather / Lambertian surface).
    - Predict the depth map $d_i$ using a pre-trained depth network (Teacher).
    - Input $(e_i, d_i, p_c)$ (image, depth, and text prompt) into a depth-conditioned diffusion model (T2I-Adapter) to generate an adverse-condition image $h_i^c$.
    - The depth map $d_i$ serves simultaneously as the spatial control condition for the diffusion model and the pseudo-depth label for the generated image.
2. **Self-Distillation Fine-tuning Stage**:
    - Use the same pre-trained network as the Student model.
    - Fine-tune the model using the pairs $(h_i^c, d_i)$ and original $(e_i, d_i)$.
    - Optimize using a scale-and-shift-invariant loss.

### Key Designs

1. **Depth-aware Conditional Diffusion Data Generation**:

    - **Function**: Generates adverse-condition images while preserving the same 3D structure as the clean-weather images.
    - **Mechanism**: Harnesses ControlNet/T2I-Adapter with the depth map $d_i$ input as a spatial constraint to preserve 3D structural consistency, while the text prompt $p_c$ dictates the type of adverse conditions in the generated scene. The two conditional inputs operate independently: depth preserves the structure, and text controls the style.
    - **Design Motivation**: Unlike GANs that require training separate models for each style/condition, diffusion models can generate an **infinite** variety of complex scenes (rain/snow/fog/night/transparent/reflective...) via text prompts, significantly expanding the coverage. Moreover, no real-world samples under the target conditions are required.

2. **Self-Distillation Fine-tuning Protocol**:

    - **Function**: Fine-tunes the depth estimation network using a teacher-student paradigm.
    - **Mechanism**: The pre-trained depth network acts simultaneously as the teacher (generating the pseudo-label $d_i$ on clean images) and the student (fine-tuned on the adverse-condition images). It optimizes the scale-and-shift-invariant loss:
    $$L_{ssi}(\hat{d}, \hat{d}^*) = \frac{1}{2M} \sum_{i=1}^{M} \rho(\hat{d}_i - \hat{d}_i^*)$$
      where $\hat{d}$ is the normalized prediction, and $\hat{d}^*$ is the normalized pseudo-label provided by the teacher.
    - **Design Motivation**: Depth predictions on clean-weather images are accurate enough to serve as reliable pseudo-labels. The scale-and-shift-invariant loss handles the scale discrepancies across different networks and datasets. This strategy is completely model-agnostic and can be applied to any pre-trained depth network.

3. **Generative Data Construction for Non-Lambertian Surfaces**:

    - **Function**: Generates training data entirely from text descriptions without requiring real transparent/reflective object datasets.
    - **Mechanism**: First uses Stable Diffusion to generate ~20K images containing Lambertian objects (e.g., ceramic bottles, wooden containers) using text prompts, and then uses T2I-Adapter to transform these objects into transparent/reflective variations while keeping the depth structure unchanged.
    - **Design Motivation**: Collecting paired data of transparent/reflective objects in real scenarios is highly challenging (requiring manual setups of purely Lambertian counterparts). Existing methods (e.g., Depth4ToM) rely heavily on real transparent object datasets and ground-truth segmentation masks. The proposed generative approach completely bypasses this limitation.

### Loss & Training

- **Loss Function**: Scale-and-shift-invariant loss (SSI loss), which compares the normalized predicted depth with the normalized pseudo-label depth.
- Driving Scenarios: Follows the training protocol of [27] using the md4all framework.
- Non-Lambertian Scenarios: Follows the training protocol of [18] using the Depth4ToM framework.
- Fine-tuning other networks: 30K iterations with a learning rate decaying from $10^{-6}$ to $10^{-7}$ (decay at 25K); Depth Anything is fine-tuned for 5K iterations.
- Hardware/Optimizer: Single RTX 3090 GPU, batch size 8, AdamW optimizer.
- Data Augmentation: Color jitter, RGB shift, horizontal flip, etc.

## Key Experimental Results

### Main Results (Monocular Depth Estimation on nuScenes)

| Method | day-clear AbsRel | night AbsRel | day-rain AbsRel |
|------|-----------------|--------------|-----------------|
| Depth Anything (Original) | 0.137 | 0.291 | 0.167 |
| **Depth Anything + Ours** | **0.134** | **0.219** | **0.157** |
| DPT (Original) | 0.189 | 0.354 | 0.237 |
| **DPT + Ours** | **0.184** | **0.224** | **0.199** |
| MiDaS (Original) | 0.171 | 0.261 | 0.218 |
| **MiDaS + Ours** | **0.168** | **0.254** | **0.195** |

Performance improvement is most significant in night-time scenes: Depth Anything's night AbsRel drops from 0.291 to 0.219 ($24.7\%\downarrow$), and DPT's drops from 0.354 to 0.224 ($36.7\%\downarrow$).

### Non-Lambertian Surfaces (Booster + ClearGrasp Datasets)

| Method | Real ToM Data | GT Seg. | Booster ToM MAE(mm) | ClearGrasp ToM MAE(mm) |
|------|-----------|--------|---------------------|----------------------|
| DPT (baseline) | ✗ | ✗ | 113.14 | 41.04 |
| **DPT + Ours** | **✗** | **✗** | **79.64** | **31.32** |
| DPT + Costanzino | ✓ | ✓ | 70.68 | 31.55 |
| Depth Anything (baseline) | ✗ | ✗ | 137.96 | 82.22 |
| **Depth Anything + Ours** | **✗** | **✗** | **54.31** | **33.88** |

Without relying on any real non-Lambertian data or GT segmentation, the method achieves performance close to or even exceeding specialized approaches that require such extra information.

### Ablation Study & Cross-Dataset Generalization

| Method | DrivingStereo rain AbsRel | nuScenes night AbsRel | RobotCar night AbsRel |
|------|--------------------------|----------------------|---------------------|
| DPT (baseline) | 0.188 | 0.354 | 0.154 |
| **DPT + Ours** | **0.124** | **0.263** | **0.130** |
| Depth Anything (baseline) | 0.112 | 0.291 | 0.125 |
| **Depth Anything + Ours** | **0.110** | **0.250** | **0.117** |

Using only clean day-time training data from Mapillary/Cityscapes/KITTI/Apolloscapes, the method achieves significant improvements on completely unseen adverse-condition datasets.

### Key Findings

- Consistent performance improvements across all evaluation metrics and weather conditions, including clean day-time scenes.
- The method is completely model-agnostic, demonstrating effectiveness across four distinct SOTA depth estimation networks.
- Eliminates the need to know specific characteristics of the target domains (requires no real rain or night-time images).
- Using depth maps generated by different networks (DPT vs. Depth Anything) as conditioning inputs for the diffusion model yields robust, stable generation results with minimal variance.
- In non-Lambertian scenarios, Depth Anything's MAE for the ToM category drops drastically from 137.96mm to 54.31mm ($61\%\downarrow$), representing a massive improvement.

## Highlights & Insights

- **A Unified Framework for Multi-type Challenges**: Unlike previous works that require tailoring separate models for night-time, rain, and transparent objects, this work covers all scenarios via a single text-to-image diffusion model with diverse text prompts. This represents a highly generalized and scalable approach.
- **Zero Real-World Challenging Data**: Bypasses the need for real-world target-domain data entirely. Training datasets are generated purely from clean-weather images guided by text descriptions. This provides significant paradigm-shifting insights for domains where ground-truth labeling is extremely difficult.
- **Model-Agnostic Design**: The backbone depth estimation network can dynamically leverage any architecture or training paradigm. The method functions as an adaptable, robust fine-tuning wrapper.
- **Elegance of Self-Distillation**: The same network acts as both the teacher and the student—serving as an expert predictor in clean domains and learning in adverse domains. It effortlessly generalizes and extends capacity using its own foundational knowledge.

## Limitations & Future Work

- The quality and fidelity of diffusion-generated images are bounded by the generative capabilities of the foundation model; some extreme conditions may lack absolute realism.
- The depth consistency assumption only holds fully at the semantic level; the diffusion model may introduce slight distortions in fine-grained 3D structure (e.g., altering surface texture details).
- The fidelity of pseudo-labels remains heavily constrained by the accuracy of the pre-trained model in clean-weather scenarios.
- Text prompt engineering relies on minor manual efforts; varying textual descriptions can influence the generation quality.
- For non-Lambertian scenes, training representations rely solely on synthetic generation (no real objects are presented during training), which might preserve a subtle domain gap.
- Temporal video consistency has not been explored, which could lead to inter-frame depth inconsistencies in consecutive sequence estimation.

## Related Work & Insights

- **vs. md4all (Gasperini et al.)**: Unlike GAN-based approaches that train dedicated architecture copies per environmental style while requiring access to targeted style images, this work employs a single, flexible diffusion model guided strictly by text prompts.
- **vs. Depth4ToM (Costanzino et al.)**: While prior state-of-the-art designs targeting transparent/reflective surfaces rely on real-world datasets with pixel-wise segmentation masks, this work synthesizes and adapts representations via text prompts, achieving comparable performance with zero extra annotations.
- **vs. R4Dyn**: R4Dyn relies heavily on radar-assisted depth tracking which demands extra hardware sensors. This method is purely monocular-based, circumventing expensive sensor setups.

## Rating

- Novelty: ⭐⭐⭐⭐ First to leverage depth-conditioned diffusion models to resolve the out-of-distribution robustness of monocular depth estimation. The motivation is elegant and natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively verified across 4 depth networks, 5+ datasets, and 3 major challenging domain subsets (night, rain, non-Lambertian). The cross-dataset generalization experiments are well-structured.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of methodology alongside comprehensive, fair experimental comparisons.
- Value: ⭐⭐⭐⭐⭐ Delivers a highly generalized, scalable paradigm to reinforce OOD robustness in existing backbones, holding strong utility for autonomous driving and robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[CVPR 2026\] Iris: Integrating Language into Diffusion-based Monocular Depth Estimation](../../CVPR2026/3d_vision/iris_integrating_language_into_diffusion-based_monocular_depth_estimation.md)
- [\[ECCV 2024\] High-Precision Self-Supervised Monocular Depth Estimation with Rich-Resource Prior](high-precision_self-supervised_monocular_depth_estimation_with_rich-resource_pri.md)

</div>

<!-- RELATED:END -->
