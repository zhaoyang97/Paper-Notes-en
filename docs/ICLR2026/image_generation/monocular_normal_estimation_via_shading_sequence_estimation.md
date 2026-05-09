---
title: >-
  [Paper Note] Monocular Normal Estimation via Shading Sequence Estimation
description: >-
  [ICLR 2026 (Oral)][Image Generation][Normal estimation] This paper proposes RoSE, which reformulates monocular normal estimation as a shading sequence estimation problem. An image-to-video (I2V) generative model is used to predict shading sequences under multiple illuminations, and a simple ordinary least squares solver then converts the shading sequence into a normal map. RoSE achieves state-of-the-art performance on real-world benchmark datasets.
tags:
  - ICLR 2026 (Oral)
  - Image Generation
  - Normal estimation
  - shading sequence
  - video generation model
  - least squares solver
  - monocular 3D reconstruction
date: 2026-05-08
content_hash: 753b828655ec3413
---

# Monocular Normal Estimation via Shading Sequence Estimation

**Conference**: ICLR 2026 (Oral)  
**arXiv**: [2602.09929](https://arxiv.org/abs/2602.09929)  
**Code**: [GitHub](https://github.com/LMozart/ICLR2026-RoSE)  
**Area**: Image Generation / 3D Vision  
**Keywords**: Normal estimation, shading sequence, video generation model, least squares solver, monocular 3D reconstruction

## TL;DR

This paper proposes RoSE, which reformulates monocular normal estimation as a shading sequence estimation problem. An image-to-video (I2V) generative model is used to predict shading sequences under multiple illuminations, and a simple ordinary least squares solver then converts the shading sequence into a normal map. RoSE achieves state-of-the-art performance on real-world benchmark datasets.

## Background & Motivation

Monocular normal estimation aims to recover a surface normal map from a single RGB image captured under arbitrary illumination, serving as a critical intermediate representation for 3D reconstruction and rendering. Existing methods suffer from a core issue termed **3D Misalignment**:

**Visually plausible but geometrically distorted results**: Current deep models directly predict normal maps that appear visually reasonable, yet the reconstructed 3D surfaces frequently fail to align with true geometric details.

**Root cause — subtle color variation**: Geometric differences in normal maps are reflected only through relatively weak color variations, making it difficult for models to accurately distinguish and reconstruct distinct geometric structures from such subtle cues.

**Limitations of Prior Work — direct prediction paradigm**: The prevailing "RGB input → Normal Map output" paradigm forces the model to simultaneously disentangle illumination and infer geometry in a single forward pass, posing an excessively high learning burden.

The core insight of this paper is that **shading sequences (sequences of brightness variations under multiple illuminations) are far more sensitive to geometry**. Different surface normal orientations produce markedly distinct shading patterns under varying light directions, which are substantially more discriminative than the color differences in a single normal map. Estimating shading sequences first and then recovering normals from them thus effectively alleviates the 3D misalignment problem.

## Method

### Overall Architecture

The RoSE pipeline consists of three stages:

1. **Preprocessing**: Convert the input RGB image to a grayscale image.
2. **Shading sequence generation**: Apply a video diffusion model to generate a multi-illumination shading sequence from the grayscale input.
3. **Normal solving**: Analytically recover the normal map from the shading sequence via ordinary least squares.

### Key Designs

1. **Shading Sequence Reformulation**:

    - **Core Idea**: Reformulate normal estimation from a direct "single image → normal map" mapping into a two-stage process: "single image → shading sequence → normal map."
    - **Shading sequence definition**: A collection of shading images of an object rendered under a set of known illumination directions $\{l_1, l_2, ..., l_K\}$.
    - **Physical basis**: Under the Lambertian reflectance model, pixel intensity is given by $I_k = \rho \cdot (n \cdot l_k)$, where $\rho$ is albedo, $n$ is the surface normal, and $l_k$ is the light direction.
    - **Geometric sensitivity**: Points with different normal orientations exhibit distinctly different intensity variation patterns across the multi-illumination sequence, which are far more discriminative than RGB differences in a single normal map.
    - **Design Motivation**: Exploit the "amplification effect" of multi-illumination to render geometric information more accessible to the model.

2. **Video Diffusion Model for Shading Generation**:

    - An image-to-video (I2V) generative model is employed to predict the shading sequence: the grayscale input serves as the "first frame," and subsequent frames correspond to shading images under different illuminations.
    - **Feature guidance**:
        - CLIP encoder: Extracts semantic features of the image to provide global object understanding.
        - VAE encoder: Extracts fine-grained texture and structural features.
    - The two feature streams complement each other to guide the video diffusion model toward generating consistent and accurate shading sequences.
    - **Training strategy**: Only the video diffusion model is trained; the CLIP and VAE encoders are frozen.
    - **Design Motivation**: I2V generative models are inherently well-suited for generating temporally consistent sequences, which aligns perfectly with the continuity requirements of shading sequences.

3. **OLS Normal Solver**:

    - Given $K$ shading frames and the corresponding illumination directions, normal estimation reduces to a simple linear system: $I = L \cdot n$, where $I \in \mathbb{R}^K$ is the shading value vector, $L \in \mathbb{R}^{K \times 3}$ is the illumination direction matrix, and $n \in \mathbb{R}^3$ is the surface normal.
    - The system is solved analytically via ordinary least squares (OLS): $n = (L^T L)^{-1} L^T I$.
    - **Core advantage**: The solving process is entirely analytical, requires no additional learning, has negligible computational cost, and is mathematically guaranteed to be optimal.
    - **Design Motivation**: Clearly separate the difficult learned component (shading estimation) from the straightforward analytical component (linear solving).

4. **MultiShade Synthetic Dataset**:

    - A large-scale synthetic training dataset constructed specifically for this task.
    - Contains diverse 3D shapes, materials, and illumination conditions.
    - Each sample includes: an object image, a multi-illumination shading sequence, and a ground-truth normal map.
    - **Design Motivation**:
        - Real-world multi-illumination normal data are extremely difficult to acquire.
        - Synthetic data provide perfect ground-truth supervision.
        - Diverse training conditions enhance generalization and robustness.

### Loss & Training

- The video diffusion model is trained with the standard denoising loss.
- Training data are drawn from the MultiShade synthetic dataset.
- The CLIP and VAE encoders remain frozen; only the video diffusion model is optimized.
- Normal solving at inference is a purely analytical process and requires no training.

## Key Experimental Results

### Main Results

| Dataset | Metric | RoSE | Prev. SOTA | Note |
|---------|--------|------|------------|------|
| DiLiGenT | Mean Angular Error (MAE)↓ | SOTA | — | Real-world object normal estimation benchmark |
| DiLiGenT-102 | MAE↓ | SOTA | — | Larger-scale real-world benchmark |
| Apple/Google Dataset | MAE↓ | SOTA | — | Industrial-grade object scan data |
| Complex Object Scenes | MAE↓ | SOTA | — | Objects with complex geometry and materials |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Direct normal prediction vs. shading sequence | Shading sequence substantially better | Validates the effectiveness of the core paradigm innovation |
| w/o CLIP guidance | Performance drop | Semantic features are important for generation quality |
| w/o VAE guidance | Performance drop | Fine-grained texture features are indispensable |
| Number of shading frames $K$ | Optimal $K$ exists | Too few frames yield insufficient information; too many increase generation difficulty |
| Different video generation backbones | Performance varies | Foundation model capability affects final results |
| Real vs. synthetic training data | Synthetic superior | Diversity of the MultiShade dataset is the key factor |

### Key Findings

1. **Paradigm shift is effective**: The shading sequence paradigm significantly outperforms the conventional direct normal prediction paradigm, validating the "indirect but more learnable" route.
2. **3D misalignment alleviated**: Surface geometry reconstructed by RoSE aligns with ground-truth geometry substantially better than baseline methods.
3. **Novel application of video generation models**: Applying I2V models to structured physical sequence generation is a novel and effective direction.
4. **Reliability of analytical solving**: The analytical nature of the OLS solver avoids error accumulation that could arise from additional learned components.
5. **Generalization from synthetic training**: Models trained on the MultiShade synthetic dataset generalize well to real-world data.
6. **ICLR Oral recognition**: The acceptance as an Oral paper reflects broad recognition of the paradigm-level innovation by the research community.

## Highlights & Insights

1. **Paradigm-level innovation**: Rather than refining the existing "direct prediction" framework, the paper proposes an entirely new "shading sequence + analytical solving" paradigm, which constitutes its most significant contribution.
2. **Seamless integration of physical intuition and deep learning**: The Lambertian reflectance model serves as a physical prior to design the learning objective, enabling deep models to learn "more learnable targets."
3. **Elegant problem decomposition**: A difficult end-to-end learning problem is decomposed into a "learning + analytical solving" two-step process, each leveraging the most appropriate tool.
4. **Cross-domain application of generative models**: Video generative models are creatively applied to 3D geometry estimation, opening a new direction for generative models in 3D vision.
5. **Clean pipeline**: Despite involving complex components such as video diffusion models, the overall pipeline follows a clear and concise logical chain.

## Limitations & Future Work

1. **Lambertian assumption**: The shading model is built on the Lambertian reflectance assumption, limiting its ability to handle specular, transparent, or translucent materials.
2. **Inference speed**: Multi-step sampling in video diffusion models may result in slow inference.
3. **Assumed illumination directions**: Illumination direction sequences must be predefined, and the choice of directions may affect estimation quality.
4. **Object-level scope**: The method primarily targets object-level normal estimation; extending to scene-level estimation requires additional work.
5. **Synthetic-to-real domain gap**: Despite good generalization, a domain gap between the MultiShade dataset and real-world data remains.
6. **Occlusion and self-shadowing**: Handling of complex occlusion relationships and self-shadows may be insufficient.
7. **Resolution constraints**: The generation resolution of video diffusion models may limit the level of detail in the recovered normal maps.

## Related Work & Insights

- **Photometric Stereo**: The classical multi-illumination normal estimation approach; RoSE can be viewed as a generalization thereof into the deep learning era.
- **Marigold / GeoWizard**: Diffusion model-based monocular depth/normal estimation methods that adopt the direct prediction paradigm.
- **Video Diffusion Models (SVD, AnimateDiff)**: RoSE leverages these models' capacity to generate temporally consistent sequences.
- **Shape from Shading**: The classical single-illumination normal estimation approach; RoSE extends its capability by generating multi-illumination shading.
- **Insights**:
    - "Transforming a target that is difficult to learn directly into a more learnable intermediate representation" is a general strategy transferable to tasks such as depth estimation and material estimation.
    - Video generative models hold substantial potential for 3D perception tasks, including dynamic 3D scene reconstruction and 4D generation.
    - Combining physical priors with generative models is a direction worthy of deeper exploration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[NeurIPS 2025\] MMG: Mutual Information Estimation via the MMSE Gap in Diffusion](../../NeurIPS2025/image_generation/mmg_mutual_information_estimation_via_the_mmse_gap_in_diffusion.md)

</div>

<!-- RELATED:END -->
