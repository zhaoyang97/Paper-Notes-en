---
title: >-
  [Paper Note] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos
description: >-
  [CVPR 2026][Nutrition Estimation] This paper proposes V-Nutri, the first framework to leverage process information from egocentric cooking videos for dish-level nutrition estimation. A VideoMamba-based keyframe selection…
tags:
  - "CVPR 2026"
  - "Nutrition Estimation"
  - "Egocentric Video"
  - "Keyframe Selection"
  - "Multimodal Fusion"
  - "Food Analysis"
date: 2026-05-08
content_hash: 98c00b491dff6f27
---

# V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos

**Conference**: CVPR 2026
**arXiv**: [2604.11913](https://arxiv.org/abs/2604.11913)  
**Code**: [https://github.com/K624-YCK/V-Nutri](https://github.com/K624-YCK/V-Nutri)  
**Area**: Food Computing / Video Understanding
**Keywords**: Nutrition Estimation, Egocentric Video, Keyframe Selection, Multimodal Fusion, Food Analysis

## TL;DR

This paper proposes V-Nutri, the first framework to leverage process information from egocentric cooking videos for dish-level nutrition estimation. A VideoMamba-based keyframe selection module identifies ingredient addition moments, which are fused with the final dish image to predict calories and macronutrients.

## Background & Motivation

**Background**: Visual nutrition estimation methods primarily rely on single images of the final dish to predict caloric content and nutritional composition, as exemplified by works such as Nutrition5K and Im2Calories.

**Limitations of Prior Work**: Single final-dish images are inherently information-limited: nutritionally significant ingredients such as oils, sauces, and dairy products are absorbed, melted, or visually integrated into the finished dish during cooking, making accurate estimation from appearance alone difficult.

**Key Challenge**: Nutritionally critical information progressively "disappears" during the cooking process, yet existing methods exploit only the final state, which retains the least information.

**Goal**: To investigate whether process information embedded in cooking videos can provide complementary evidence for dish-level nutrition estimation.

**Key Insight**: Egocentric cooking videos preserve complete temporal nutritional evidence—ingredient identity, addition events, and intermediate states—and the increasing availability of wearable cameras makes this direction increasingly practical.

**Core Idea**: A keyframe selection module locates nutritionally information-dense moments (e.g., ingredient additions) within long videos, which are then fused with the final dish image to improve nutrition estimation accuracy.

## Method

### Overall Architecture

V-Nutri is a staged pipeline consisting of: (1) a cooking keyframe selector (VideoMamba) that identifies ingredient addition events from egocentric video; (2) a final dish frame selector that locates the finished dish frame; (3) a Nutrition5K-pretrained visual backbone that extracts features from both process keyframes and the dish frame; (4) an attention-weighted fusion module that aggregates process evidence; and (5) an MLP regressor that predicts four nutritional targets: calories, protein, fat, and carbohydrates.

### Key Designs

1. **Cooking Keyframe Selector**:

    - **Function**: Locates nutritionally information-dense moments within long, redundant egocentric cooking videos.
    - **Mechanism**: Employs the selective state space model of VideoMamba with a sliding window to segment the video into short clips, detecting candidate events such as ingredient additions. The linear complexity of VideoMamba is well-suited to long, information-sparse egocentric videos.
    - **Design Motivation**: Dense processing of full videos is computationally inefficient and introduces noise; it is therefore necessary to first identify a sparse, informative subset of frames.

2. **Nutrition5K Pretrained Backbone + Lightweight Fusion**:

    - **Function**: Fuses visual features from process keyframes and the final dish frame for nutrition prediction.
    - **Mechanism**: A frozen Nutrition5K-pretrained backbone (ResNet-101/ViT-B/ViT-L) encodes each process keyframe as embeddings $z_1, \ldots, z_K$ and the dish frame as $z_d$. Learned attention weights $\alpha_1, \ldots, \alpha_K$ aggregate the process embeddings into a pooled representation $z_p$ via weighted pooling, which is then fused with the dish embedding.
    - **Design Motivation**: Leveraging a backbone pretrained on food data enables extraction of nutrition-relevant features without training from scratch; lightweight fusion mitigates overfitting.

3. **HD-EPIC Benchmark Annotation Extension**:

    - **Function**: Establishes the first video-based nutrition estimation benchmark.
    - **Mechanism**: The HD-EPIC dataset is augmented with temporal annotations of cooking process keyframes and final dish frames, together with dish-level nutritional ground truth.
    - **Design Motivation**: Existing datasets lack annotations linking cooking videos to nutritional labels.

### Loss & Training

A standard regression loss (e.g., MAE/MSE) is applied to predict the four-dimensional nutritional vector $\mathbf{y}_c = [y^{kcal}, y^{protein}, y^{fat}, y^{carb}]$. The backbone is frozen; only the fusion module and regressor are trained.

## Key Experimental Results

### Main Results

| Backbone | Input | Calorie MAE↓ | Protein MAE↓ | Fat MAE↓ | Carb MAE↓ |
|----------|-------|-------------|-------------|---------|----------|
| ViT-L | Dish only | 185.3 | 12.1 | 9.8 | 18.5 |
| ViT-L | Dish + Process | **172.8** | **11.2** | **9.1** | **17.3** |
| ViT-B | Dish only | 198.7 | 13.5 | 10.6 | 19.8 |
| ViT-B | Dish + Process | 191.2 | 12.8 | 10.1 | 19.0 |
| ResNet-101 | Dish + Process | 205.1 | 14.2 | 11.3 | 20.5 |

### Ablation Study

| Configuration | Calorie MAE↓ | Notes |
|---------------|-------------|-------|
| Full model (ViT-L) | 172.8 | Dish + process frames + event detection |
| w/o event detection (random frames) | 182.1 | Random frame sampling replaces event detection |
| Dish frame only | 185.3 | Final dish image only |
| Uniform process frame sampling | 179.5 | Uniform sampling replaces event detection |

### Key Findings

- The benefit of process keyframes is strongly dependent on backbone representational capacity: ViT-L yields the largest gain, while ResNet-101 shows limited improvement.
- Event detection quality is critical: randomly sampled frames yield substantially smaller gains than detected ingredient addition frames.
- Under controlled conditions, process information does provide complementary nutritional evidence.

## Highlights & Insights

- "Process-aware" nutrition estimation is a well-motivated and practical research direction: as wearable cameras become more prevalent, leveraging cooking videos for dietary monitoring is increasingly feasible.
- The lightweight fusion strategy (frozen backbone + attention-weighted pooling) mitigates overfitting and is well-suited to data-limited settings.

## Limitations & Future Work

- The HD-EPIC dataset is limited in scale, leaving generalization insufficiently validated.
- The benefit of process frames is marginal with weaker backbones, indicating strong backbone dependency.
- The impact of cooking methods (pan-frying, deep-frying, steaming, etc.) on nutritional changes is not considered.
- Integration with ingredient recognition and portion size estimation could further improve accuracy.

## Related Work & Insights

- **vs. Nutrition5K**: Nutrition5K relies solely on final dish images; V-Nutri extends this to video by exploiting process information to recover nutritional cues invisible in the final dish.
- **vs. Long Video Understanding**: Rather than pursuing full-sequence understanding, this work efficiently extracts sparse process evidence from long videos.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to incorporate cooking process information from video into nutrition estimation.
- Experimental Thoroughness: ⭐⭐⭐ Dataset scale is limited, but ablation analysis is reasonably comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clearly articulated.
- Value: ⭐⭐⭐ The research direction is meaningful, though the magnitude of improvement is modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](omnifood8k_nutrition_estimation.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](../../ICML2026/others/from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[ICCV 2025\] Toward Material-Agnostic System Identification from Videos](../../ICCV2025/others/toward_material-agnostic_system_identification_from_videos.md)

</div>

<!-- RELATED:END -->
