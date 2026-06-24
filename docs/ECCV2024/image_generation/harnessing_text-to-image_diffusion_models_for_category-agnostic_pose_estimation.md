---
title: >-
  [Paper Note] Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation
description: >-
  [ECCV 2024][Image Generation][Category-Agnostic Pose Estimation] Proposes the Prompt Pose Matching (PPM) framework, which leverages the rich knowledge in pre-trained text-to-image diffusion models to address Category-Agnostic Pose Estimation (CAPE). By learning pseudo prompts corresponding to keypoints, it achieves few-shot keypoint detection without training on base categories.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Category-Agnostic Pose Estimation"
  - "Text-to-Image Diffusion Models"
  - "Pseudo Prompt Learning"
  - "Few-shot Keypoint Detection"
  - "Prompt Pose Matching"
date: 2026-05-08
content_hash: adb7557d73793775
---

# Harnessing Text-to-Image Diffusion Models for Category-Agnostic Pose Estimation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Diffusion Models / Pose Estimation  
**Keywords**: Category-Agnostic Pose Estimation, Text-to-Image Diffusion Models, Pseudo Prompt Learning, Few-shot Keypoint Detection, Prompt Pose Matching

## TL;DR

Proposes the Prompt Pose Matching (PPM) framework, which leverages the rich knowledge in pre-trained text-to-image diffusion models to address Category-Agnostic Pose Estimation (CAPE). By learning pseudo prompts corresponding to keypoints, it achieves few-shot keypoint detection without training on base categories.

## Background & Motivation

**Background**: The goal of Category-Agnostic Pose Estimation (CAPE) is to detect corresponding keypoints in any image of an unseen category given only a few support images with keypoint annotations. This is a challenging generalization problem. Existing methods (e.g., CapeFormer, POMNet) typically require training on a large number of predefined base categories, leveraging their rich annotations to learn cross-category keypoint matching capability.

**Limitations of Prior Work**: (1) Dependence on base categories restricts methods to the quality and coverage of annotated data—if the base categories differ significantly from the test categories, generalization performance drops substantially; (2) preparing training data for base categories itself requires massive annotation effort; (3) these methods are inherently learning a "keypoint matcher", but the matching patterns learned from limited supervised categories may overfit to specific visual patterns.

**Key Challenge**: The key challenge of CAPE lies in generalization to unseen categories—limited data (few-shot) makes direct learning on test categories infeasible, while training on base categories introduces domain shift. A knowledge source is needed that does not rely on large-scale annotated base categories but possesses strong generalization capabilities.

**Goal**: (1) How to leverage existing rich vision-semantic knowledge to replace training on base categories; (2) how to adapt the knowledge of pre-trained diffusion models to the keypoint localization task; (3) how to capture the semantic information of keypoints given only a few support examples.

**Key Insight**: The authors observe that text-to-image diffusion models (such as Stable Diffusion) learn extremely rich vision-semantic correspondences during pre-training—they understand concepts like "where the eyes are" or "where the wheels are". This knowledge is precisely what CAPE needs: understanding the semantic locations of object parts. The key question is how to explicitly utilize this implicit knowledge for keypoint localization.

**Core Idea**: Use learnable pseudo prompts to encode keypoint semantics in the text space of the diffusion model, achieving keypoint localization via text-image attention mechanisms without training on base categories.

## Method

### Overall Architecture

Given a set of few-shot examples (support images + keypoint annotations), the PPM framework operates in three steps: (1) learn a set of pseudo prompt vectors for each keypoint so that they correspond to the correct spatial locations in the text-image attention of the diffusion model; (2) apply the learned pseudo prompts to query images to extract keypoint locations via attention maps; (3) introduce a Category-shared Prompt Training (CPT) scheme to enhance generalization.

### Key Designs

1. **Pseudo Prompt Learning**:

    - **Function**: Learn trainable vectors in the text embedding space of the diffusion model that correspond one-to-one to keypoints.
    - **Mechanism**: For each keypoint $k$, a learnable pseudo prompt vector $p_k \in \mathbb{R}^d$ is initialized. $p_k$ is injected as a condition into the cross-attention layers of the diffusion model's UNet to obtain the corresponding attention map $A_k$. During the diffusion process, using the ground-truth (GT) locations of keypoints in support images as supervision, $p_k$ is optimized to maximize the activation of $A_k$ at the keypoint locations. The loss function is the MSE between the attention maps and the Gaussian heatmaps of the GT keypoint locations: $L = \sum_k \|A_k - G_k\|^2$, where $G_k$ is a Gaussian kernel centered at the keypoint coordinates.
    - **Design Motivation**: The cross-attention of the diffusion model has already learned the mapping of "text description $\rightarrow$ spatial location" (e.g., "cat's eye" activates the location of the cat's eye). Pseudo prompts operate directly in this semantic space, eliminating the need to understand specific textual meanings—they only need to find embedding vectors that activate the correct spatial locations.

2. **Prompt Pose Matching Inference**:

    - **Function**: Locate keypoints on novel images using the learned pseudo prompts.
    - **Mechanism**: During inference, the learned pseudo prompt vectors $\{p_1, ..., p_K\}$ are injected into the diffusion model. A single forward diffusion (noising + denoising) step is performed on the query image to extract the cross-attention map $A_k$ corresponding to each $p_k$. After parsing the attention maps with softmax normalization, the location with the maximum activation is taken as the predicted coordinate of the keypoint: $\hat{x}_k = \arg\max_{(i,j)} A_k(i,j)$. To improve accuracy, multi-scale attention fusion is employed—extracting attention maps from different layers of the UNet and performing weighted averaging.
    - **Design Motivation**: This avoids any extra keypoint detection heads or post-processing steps, fully utilizing the inherent spatial localization capability of the diffusion model. Multi-scale fusion leverages the characteristic of different UNet layers capturing different granularities of information.

3. **Category-shared Prompt Training (CPT)**:

    - **Function**: Further enhance the cross-category generalization capability of pseudo prompts.
    - **Mechanism**: A set of shared "base prompts" is trained simultaneously across multiple different categories, where the keypoint prompt for each category consists of "base prompt + category-specific offset": $p_k^c = p_{base} + \Delta p_k^c$. The base prompts capture generic keypoint semantics across categories (e.g., "protrusion points", "joint points"), while the offset term captures category-specific semantics. During training, support examples of different categories are used alternately, encouraging the base prompts to learn more generalized representations.
    - **Design Motivation**: Pure per-category prompt learning might overfit to the specific appearance of the support images. Through category-shared base prompts, the model is forced to learn the "general semantics" of keypoints rather than "category-specific appearances", resembling task-shared initialization in meta-learning.

### Loss & Training

The primary loss is the MSE heatmap matching loss on the attention maps. During training, all parameters of the diffusion model (UNet + text encoder) are frozen, and only the pseudo prompt vectors are optimized. In the CPT training phase, a multi-category alternating training strategy is used, where support examples from a randomly sampled category are optimized in each iteration. No gradient calculation is required during inference, needing only a single forward pass.

## Key Experimental Results

### Main Results

| Dataset | Setting | Ours (PPM) | CapeFormer | POMNet | Description |
|--------|------|---------|------------|--------|------|
| MP-100 | 1-shot | 72.8 | 68.5 | 67.2 | PCK@0.2 |
| MP-100 | 5-shot | 80.3 | 76.1 | 74.8 | PCK@0.2 |
| AP-10K | 1-shot | 65.4 | 61.2 | 59.8 | PCK@0.2 |
| CUB-200 | 1-shot | 78.1 | 74.3 | 72.5 | PCK@0.2 |

### Ablation Study

| Configuration | PCK@0.2 | Description |
|------|---------|------|
| PPM + CPT (Full) | 72.8 | Full model |
| PPM w/o CPT | 69.5 | Drops by 3.3 after removing category-shared training |
| Single-layer attention | 68.2 | Using only a single UNet attention layer |
| Multi-layer attention fusion | 72.8 | Fusing multiple layers performs the best |
| Random prompt init | 71.0 | Random initialization vs. text initialization |
| Text-guided init | 72.8 | Initialization with keypoint descriptive text is superior |

### Key Findings
- CPT contributes a 3.3% PCK improvement, indicating that cross-category shared knowledge is crucial for generalization.
- Multi-scale attention fusion is 4.6 points higher than single-layer attention, where lower layers provide positional precision and higher layers provide semantic accuracy.
- Initializing pseudo prompts with descriptive text (e.g., "left eye of the animal") is 1.8 points better than random initialization, demonstrating that the text-comprehension capability of the diffusion model provides a meaningful starting point.
- On test categories that differ significantly from the base categories, PPM shows a more pronounced relative advantage—since it does not rely on training on base categories.

## Highlights & Insights

- **Using diffusion models as general vision-semantic knowledge bases**: Instead of generating images, the internal attention mechanism is leveraged for spatial localization. This paradigm of "diffusion model as feature extractor" is highly inspiring and transferable to other dense prediction tasks such as segmentation and matching.
- **Pseudo prompt learning bypasses the difficulties of prompt engineering**: There is no need to manually craft text accurately describing the keypoints. Direct optimization in the embedding space is more flexible and free from the limitations of natural language expressions.
- **Zero base-category training** substantially reduces data requirements, working with just a few support images, which is valuable for pose estimation of low-resource categories.

## Limitations & Future Work

- Inference speed is limited by the forward propagation of the diffusion model: Even though only a single denoising step is needed, the computational cost of the UNet remains high, making real-time application difficult.
- When the keypoint definitions of a category vary significantly from the semantics in the diffusion model's pre-training data (e.g., special markers on industrial parts), the pseudo prompts may struggle to learn effective embeddings.
- Limited capability in handling occluded keypoints: The cross-attention maps may yield low activations at the locations of occluded keypoints.
- One can explore incorporating diffusion models with pose constraints (such as ControlNet) to further enhance the spatial awareness of keypoints.
- The method can be extended to 3D keypoint estimation—leveraging the 3D spatial understanding capabilities learned by the diffusion model.

## Related Work & Insights

- **vs CapeFormer**: CapeFormer learns cross-category keypoint matching via Transformers, requiring a large amount of annotated data from base categories for training. PPM does not require any base-category training, offering a distinct advantage in generalization.
- **vs POMNet**: POMNet uses prototype networks for few-shot matching, but the representation capability of prototypes is limited. PPM leverages the rich semantic space of the diffusion model, providing stronger representation capability.
- **vs DiffusionDet/Marigold**: Similarly utilizing diffusion models for non-generative tasks (DiffusionDet for detection, Marigold for depth estimation), reflecting the trend of diffusion models acting as general vision foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to apply the attention mechanism of diffusion models to CAPE. The idea of pseudo prompt learning is novel, entirely departing from the traditional base-category training paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple benchmarks with relatively complete ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, and the description of the method is easy to understand.
- Value: ⭐⭐⭐⭐⭐ Opens up a new direction for using diffusion models in keypoint detection, offering broad inspiration for few-shot vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diff-Tracker: Text-to-Image Diffusion Models are Unsupervised Trackers](diff-tracker_text-to-image_diffusion_models_are_unsupervised_trackers.md)
- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](../../CVPR2025/image_generation/can_generative_video_models_help_pose_estimation.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] TextDiffuser-2: Unleashing the Power of Language Models for Text Rendering](textdiffuser-2_unleashing_the_power_of_language_models_for_text_rendering.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](../../ICCV2025/image_generation/diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)

</div>

<!-- RELATED:END -->
