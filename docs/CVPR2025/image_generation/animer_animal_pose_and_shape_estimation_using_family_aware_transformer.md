---
title: >-
  [Paper Note] AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer
description: >-
  [CVPR 2025][Image Generation][Animal Pose Estimation] This paper proposes AniMer, which introduces a high-capacity ViT backbone to quadrupedal SMAL parameter estimation for the first time. By distinguishing shape distributions of different species through animal family-level supervised contrastive learning, and using ControlNet-based synthetic dataset CtrlAni3D (10k images), it comprehensively outperforms existing methods on Animal3D, CtrlAni3D…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Animal Pose Estimation"
  - "SMAL Model"
  - "Transformer"
  - "Contrastive Learning"
  - "Synthetic Dataset"
date: 2026-05-08
content_hash: ca2f08a966767e35
---

# AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer

**Conference**: CVPR 2025  
**arXiv**: [2412.00837](https://arxiv.org/abs/2412.00837)  
**Code**: [https://luoxue-star.github.io/AniMer_project_page/](https://luoxue-star.github.io/AniMer_project_page/) (Project Page)  
**Area**: Image Generation  
**Keywords**: Animal Pose Estimation, SMAL Model, Transformer, Contrastive Learning, Synthetic Dataset

## TL;DR
This paper proposes AniMer, which introduces a high-capacity ViT backbone to quadrupedal SMAL parameter estimation for the first time. By distinguishing shape distributions of different species through animal family-level supervised contrastive learning, and using ControlNet-based synthetic dataset CtrlAni3D (10k images), it comprehensively outperforms existing methods on Animal3D, CtrlAni3D, and the cross-domain Animal Kingdom dataset.

## Background & Motivation

**Background**: Animal pose and shape estimation is crucial for animal welfare, ecology, and biomechanics research. The SMAL model is the standard parametric representation for quadrupeds, analogous to SMPL for human bodies. However, existing methods focus primarily on a single species (horses or dogs), use CNN backbones, and suffer from limited training data.

**Limitations of Prior Work**: (1) The capacity of CNN backbones is insufficient to unify shape differences across multiple species in a single network (e.g., cats vs. cows vs. hippos); (2) 3D-annotated multi-species data is extremely scarce; although Animal3D provides a large-scale benchmark, its diversity remains insufficient; (3) The simple and effective paradigm of "ViT + large-scale data" (such as HMR2.0) in human body reconstruction has never been validated in the animal domain.

**Key Challenge**: Animals exhibit much greater inter-species shape differences than humans (inter-species variation > intra-species variation), while the available 3D-annotated data is far scarcer than in the human domain.

**Goal**: Validate the effectiveness of the "high-capacity backbone + large-scale data" paradigm for animal pose and shape estimation, and alleviate the data scarcity issue.

**Key Insight**: (1) Introduce a ViT backbone to replace CNNs; (2) Design family-level contrastive learning to capture inter-species shape variations; (3) Use ControlNet to synthesize large-scale training data.

**Core Idea**: A ViT encoder and a Transformer decoder are utilized to directly regress SMAL parameters, employing family-level supervised contrastive learning via class tokens to enhance the shape discriminative ability among different animal families.

## Method

### Overall Architecture
Given a single RGB image and a learnable class token as input, the ViT encoder extracts image features ($192 \times 1280$), and the Transformer decoder outputs a feature vector ($1 \times 1024$). Independent MLP heads then predict the shape parameters $\hat{\beta} \in \mathbb{R}^{41}$, pose parameters $\hat{\theta} \in \mathbb{R}^{35 \times 3}$, and camera parameters. Simultaneously, the class token is processed by a prediction head to output animal family features for contrastive learning.

### Key Designs

1. **Family Supervised Contrastive Learning**:

    - **Function**: Enhances the network's discriminative capability for shapes across different animal families (e.g., Felidae, Canidae, Equidae, Bovidae).
    - **Mechanism**: The learnable class token of the ViT interacts with image features to extract animal family details. A family-supervised contrastive loss $\mathcal{L}_{\text{con}}$ is applied within a mini-batch: pulling class token features of the same family closer, and pushing those of different families apart. A temperature parameter $\tau$ controls the resolution of contrastive learning.
    - **Design Motivation**: While shape parameters in human SMPL originate from a single multivariate normal distribution, animals exhibit at least two levels of variation: inter-species and intra-species. Contrastive learning explicitly encodes this hierarchical structure in the feature space.

2. **CtrlAni3D Synthetic Dataset**:

    - **Function**: Alleviates the scarcity of 3D-annotated animal data.
    - **Mechanism**: ControlNet is used to render realistic animal images conditioned on masks and depth maps from SMAL animations. Text prompts describe animal behaviors, and backgrounds are sourced from COCO or AI synthesis. Substandard images are filtered out using SAM2.0 and manual verification. The final dataset contains 9,711 pixel-aligned SMAL annotated images across 10 species.
    - **Design Motivation**: Traditional CG rendering is limited by texture quality and lighting control, whereas generative AI can produce high-quality, diverse images with minimal human effort.

3. **Direct Parameter Decoding + Two-Stage Training**:

    - **Function**: A training strategy tailored to the data characteristics of the animal domain.
    - **Mechanism**: Unlike the residual parameter decoding of HMR2.0 (which relies on the mean of large-scale motion databases), AniMer utilizes direct parameter decoding (due to the lack of a SMAL pose prior). A two-stage training scheme is adopted: the first stage uses only 3D data (500 epochs) to ensure the network predicts reasonable shapes and poses, and the second stage introduces all 2D+3D data (700 epochs).
    - **Design Motivation**: There is a severe imbalance between 3D and 2D data volumes; direct mixed training would degrade 3D regression quality.

### Loss & Training
The total loss is formulated as $\mathcal{L}_{\text{total}} = \lambda_{\text{3D}}\mathcal{L}_{\text{3D}} + \lambda_{\text{2D}}\mathcal{L}_{\text{2D}} + \lambda_{\text{prior}}\mathcal{L}_{\text{prior}} + \lambda_{\text{adv}}\mathcal{L}_{\text{adv}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}}$. 3D loss includes key vertex, joint, and parameter regressions; 2D loss represents keypoint re-projection errors; prior loss constrains SMAL parameters to reasonable ranges.

## Key Experimental Results

### Main Results

| Method | Animal3D PA-MPJPE↓ | Animal3D PCK↑ | CtrlAni3D PA-MPJPE↓ |
|------|-------------------|---------------|---------------------|
| Ours (ViT-H) | **Best** | **Best** | **Best** |
| HMR2.0 (Animal Ver.) | Second Best | Second Best | Second Best |
| WLDO (CNN) | Poor | Poor | Poor |
| HMR (CNN) | Poor | Poor | Poor |

### Ablation Study

| Configuration | Key Effects |
|------|---------|
| Without family-level contrastive learning | Decreased accuracy in shape estimation |
| Without CtrlAni3D | Significant drop in OOD generalization capability |
| Replacing ViT with CNN backbone | Substantial performance drop, proving the importance of high-capacity backbones |
| Replacing direct decoding with residual decoding | Performance drop due to the lack of SMAL pose priors |

### Key Findings
- The improvement brought by the ViT backbone to animal reconstruction is far greater than in the human domain (due to larger inter-species variations demanding stronger representation capability).
- Family-level contrastive learning enhances pose and shape estimation accuracy across all benchmarks.
- CtrlAni3D significantly improves generalization performance on the unseen Animal Kingdom dataset.
- Two-stage training is crucial for handling the imbalance between 3D and 2D data.

## Highlights & Insights
- Successfully transfers the successful paradigm of human body reconstruction (HMR2.0's ViT + large data) to the animal domain, validating its generalizability.
- The approach of generating training data via generative models (ControlNet) from parametric model (SMAL) animations can be extended to other domains lacking 3D annotations.
- Family-level contrastive learning captures the hierarchical structure of animal shapes. This idea of "hierarchical shape modeling" can inspire other multi-category 3D reconstruction tasks.

## Limitations & Future Work
- The SMAL model itself is constructed from scans of only 41 toy figurines, limiting its geometric accuracy on real animals.
- Strictly covers quadrupeds, with other categories like birds and reptiles left unaddressed.
- There remains a domain gap between synthetic and real-world data.

## Related Work & Insights
- **vs HMR2.0**: SOTA in human body reconstruction; this work validates the effectiveness of a similar paradigm for animals.
- **vs Animal3D**: Provided the first large-scale benchmark but relied on a CNN backbone. AniMer achieves substantial performance gains using ViT and contrastive learning.
- **vs SPAC-Net**: Similarly uses ControlNet to generate synthetic data, but relies on textured CAD assets. AniMer generates directly from untextured SMAL models.

## Rating
- Novelty: ⭐⭐⭐⭐ Family-level contrastive learning and the ControlNet synthetic data pipeline are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation on multiple benchmarks with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured paper.
- Value: ⭐⭐⭐⭐ Advances the state-of-the-art in 3D animal reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts](shapewords_guiding_text-to-image_synthesis_with_3d_shape-aware_prompts.md)
- [\[CVPR 2025\] Free-viewpoint Human Animation with Pose-correlated Reference Selection](free-viewpoint_human_animation_with_pose-correlated_reference_selection.md)
- [\[ECCV 2024\] Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation](../../ECCV2024/image_generation/harnessing_text-to-image_diffusion_models_for_category-agnostic_pose_estimation.md)
- [\[CVPR 2025\] SGMatch: Semantic-Guided Non-Rigid Shape Matching with Flow Regularization](sgmatch_semantic-guided_non-rigid_shape_matching_with_flow_regularization.md)

</div>

<!-- RELATED:END -->
