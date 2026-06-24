---
title: >-
  [Paper Note] Denoising Functional Maps: Diffusion Models for Shape Correspondence
description: >-
  [CVPR 2025][3D Vision][Functional maps] This paper proposes DenoisFM, the first method to apply denoising diffusion models to directly predict functional maps between shapes. It reduces learning complexity through template matching and introduces an unsupervised approach to resolve the sign ambiguity of Laplace eigenvectors, achieving competitive performance on human and animal shape matching.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Functional maps"
  - "denoising diffusion models"
  - "shape correspondence"
  - "Laplace eigenvector sign ambiguity"
  - "template matching"
date: 2026-05-08
content_hash: 0317d435b7470f69
---

# Denoising Functional Maps: Diffusion Models for Shape Correspondence

**Conference**: CVPR 2025  
**arXiv**: [2503.01845](https://arxiv.org/abs/2503.01845)  
**Code**: [https://github.com/alekseizhuravlev/denoising-functional-maps/](https://github.com/alekseizhuravlev/denoising-functional-maps/)  
**Area**: 3D Shape Matching / Shape Correspondence  
**Keywords**: Functional maps, denoising diffusion models, shape correspondence, Laplace eigenvector sign ambiguity, template matching

## TL;DR
This paper proposes DenoisFM, the first method to apply denoising diffusion models to directly predict functional maps between shapes. It reduces learning complexity through template matching and introduces an unsupervised approach to resolve the sign ambiguity of Laplace eigenvectors, achieving competitive performance on human and animal shape matching.

## Background & Motivation

**Background**: Shape correspondence (finding point-to-point mapping between two 3D shapes) is often solved via the functional map framework—compressing high-dimensional pointwise mappings into small matrix representations. Existing methods typically learn descriptor functions first and then solve for functional maps, but they are limited by small datasets (fewer than a hundred shapes) and lack cross-category generalization capabilities.

**Limitations of Prior Work**: (1) Descriptor-based methods require category-specific training data; (2) Large-scale deformation methods require a vast amount of data but directly learn the deformation; (3) The application of diffusion models in shape matching is limited to extracting features from pretrained models, rather than directly predicting functional maps.

**Key Challenge**: Functional maps are small-sized matrices ($n \times n$), making them suitable for diffusion models to process. However, the sign ambiguity of Laplace eigenvectors results in $2^n$ different possible functional maps for the same correspondence, which significantly increases the learning difficulty.

**Goal**: Direct prediction of functional maps using diffusion models, training on large-scale synthetic data, and addressing the sign ambiguity issue.

**Key Insight**: (1) Use template matching (comparing all shapes to a single template) instead of pairwise matching to reduce complexity from $O(N^2)$ to $O(N)$; (2) Propose an unsupervised sign correction method to reduce ambiguity.

**Core Idea**: Employ a DDPM to learn the denoising process "from noise to functional map" conditioned on shape geometric features, train it using 230k human shapes from SURREAL, and unify the $2^n$ possible functional maps into a single one via unsupervised sign correction.

## Method

### Overall Architecture
During training: Extract functional maps between 230k shapes from the large-scale SURREAL human dataset and the template. Compute the Laplace eigenvectors for each shape and correct their signs using the unsupervised method. Train a standard U-Net diffusion model conditioned on geometric features to predict the functional map. During inference: Compute conditional features for a new shape, sample the functional map multiple times, and select the one with the lowest Dirichlet energy as the final result.

### Key Designs

1. **Unsupervised Eigenvector Sign Correction**:

    - **Function**: Unifies the $2^n$ possible bases of each shape into a single deterministic choice.
    - **Mechanism**: Extracts unsupervised surface features (e.g., HKS) for each shape, computes the weighted inner product $\langle \phi_i, \varsigma_i \rangle_A$ between each eigenvector $\phi_i$ and feature $\varsigma_i$, and flips the sign if it is negative. This exploits the statistical consistency between eigenvectors and surface features of same-class shapes.
    - **Design Motivation**: Without resolving sign ambiguity, the diffusion model would need to learn all $2^n$ mappings, leading to exponential complexity explosion.

2. **Template Matching Strategy**:

    - **Function**: Reduces complexity during training and inference.
    - **Mechanism**: All functional maps are "shape $\rightarrow$ template" rather than "shape A $\rightarrow$ shape B". During inference, if the correspondence A $\rightarrow$ B is needed, the functional maps A $\rightarrow$ template and B $\rightarrow$ template are obtained first, and then composed to obtain A $\rightarrow$ B.
    - **Design Motivation**: Pairwise matching requires $O(N^2)$ functional maps, whereas template matching only requires $O(N)$.

3. **Multi-sample Selection Strategy**:

    - **Function**: Leverages the stochastic nature of diffusion models to improve result quality.
    - **Mechanism**: Generates multiple candidate functional maps and selects the optimal one using Dirichlet energy (which measures the smoothness of the correspondence).
    - **Design Motivation**: The probabilistic nature of diffusion models becomes an advantage here—a multi-sample and selection strategy is more robust than a single deterministic prediction.

### Loss & Training
Standard DDPM denoising loss. Training data consists of 230k human shapes generated by the SMPL parametric model.

## Key Experimental Results

### Main Results

| Dataset | Type | Ours | Descriptor-based Methods | Large-scale Deformation Methods |
|--------|------|------|-----------|-------------|
| FAUST (Human) | In-distribution | Competitive | Better | Competitive |
| SHREC (Human) | Varying connectivity | Competitive | - | - |
| DT4D (Non-isometric Humanoid) | Cross-category | Strong generalization | Requires retraining | - |
| SMAL (Animal) | Cross-category | Feasible | Category-specific | - |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| No Sign Correction | Significant performance drop | Ambiguity increases learning difficulty |
| Pairwise vs. Template Matching | Template matching is better | Reduced complexity + consistency |
| Single-sample vs. Multi-sample Selection | Multi-sample is better | Leverages probabilistic properties |

### Key Findings
- Diffusion models can effectively learn the functional map space without requiring hand-crafted descriptors.
- Sign correction is crucial—performance drops significantly without it.
- Zero-shot generalization is good on non-isometric humanoids and animals, proving the model has learned a general geometric prior.

## Highlights & Insights
- **New Paradigm**: Directly predicts low-dimensional representations of shape correspondence using a generative model, bypassing the traditional descriptor $\rightarrow$ optimization pipeline.
- **Unsupervised Resolution of Sign Ambiguity**: Uses surface feature statistics to determine eigenvector signs, which is simple and effective.
- **Diffusion Model Multi-sample Strategy**: Turns the randomness of diffusion from a "disadvantage" into an "advantage".

## Limitations & Future Work
- The template matching assumption limits applicability to shapes that cannot establish correspondence with the template.
- Training data is limited to human shapes, showing limited generalization to shapes with completely different topologies.
- The resolution of the functional map is restricted by the number of eigenvectors $n$.

## Related Work & Insights
- **vs. Descriptor Methods (DiffusionNet, etc.)**: Does not require a hand-crafted descriptor pipeline; directly end-to-end.
- **vs. 3D-CODED**: Both use large-scale synthetic data, but this work utilizes a diffusion model instead of an encoder-decoder.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of diffusion models to functional map prediction; novel sign correction method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset validation + cross-category generalization.
- Writing Quality: ⭐⭐⭐⭐ Well-defined problem and solid methodological motivation.
- Value: ⭐⭐⭐⭐ Opens up a new technical route for shape matching.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Volumetric Functional Maps](../../CVPR2026/3d_vision/volumetric_functional_maps.md)
- [\[CVPR 2025\] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)
- [\[CVPR 2025\] Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)
- [\[CVPR 2025\] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction](dualpm_dual_point_maps_shape_pose.md)
- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)

</div>

<!-- RELATED:END -->
