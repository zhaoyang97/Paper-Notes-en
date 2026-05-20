---
title: >-
  [Paper Note] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion
description: >-
  [CVPR 2026][Food Nutrition Estimation] This work introduces OmniFood8K, a multimodal Chinese food nutrition dataset comprising 8,036 samples…
tags:
  - "CVPR 2026"
  - "Food Nutrition Estimation"
  - "Multimodal Dataset"
  - "Depth Estimation"
  - "Frequency-Domain Fusion"
  - "Chinese Cuisine"
date: 2026-05-08
content_hash: b5e34e64f21f2169
---

# OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion

**Conference**: CVPR 2026
**arXiv**: [2604.12356](https://arxiv.org/abs/2604.12356)  
**Code**: [https://yudongjian.github.io/OmniFood8K-food/](https://yudongjian.github.io/OmniFood8K-food/)  
**Area**: Food Computing / Multimodal Fusion
**Keywords**: Food Nutrition Estimation, Multimodal Dataset, Depth Estimation, Frequency-Domain Fusion, Chinese Cuisine

## TL;DR

This work introduces OmniFood8K, a multimodal Chinese food nutrition dataset comprising 8,036 samples, along with a synthetic dataset NutritionSynth-115K containing 115K samples. An end-to-end framework is proposed that predicts nutritional information from a single RGB image via a Scale-Shift depth adapter, frequency-aligned fusion, and a mask-based prediction head.

## Background & Motivation

**Background**: Food nutrition estimation is critical for public health, and deep learning methods have shown promise in automatically recognizing food and estimating its weight, volume, and nutritional content.

**Limitations of Prior Work**: (1) *Data limitation*: Existing datasets are heavily biased toward Western cuisines with insufficient coverage of Chinese food. (2) *Algorithm limitation*: State-of-the-art methods rely on depth cameras to obtain depth information, whereas food photographs in everyday scenarios are typically captured with RGB cameras only.

**Key Challenge**: Depth information is essential for accurate food volume and nutrition estimation, yet real-world deployment scenarios typically provide only RGB images.

**Goal**: (1) Construct a comprehensive multimodal food dataset covering Chinese cuisine. (2) Propose an end-to-end nutrition prediction framework that requires only a single RGB image.

**Key Insight**: A pretrained depth estimation model is employed to predict depth from RGB images; an adapter corrects the predicted depth, and frequency-domain fusion replaces the need for actual depth sensors.

**Core Idea**: Predict depth map → adapter correction → frequency-aligned fusion of RGB and depth features → mask-aware prediction.

## Method

### Overall Architecture

Given a single RGB image, a pretrained depth estimation model first predicts the depth map. The Scale-Shift Residual Adapter (SSRA) then corrects global scale bias and local structural errors in the predicted depth. The Frequency-Aligned Fusion Module (FAFM) subsequently fuses RGB and corrected depth features hierarchically in the frequency domain. Finally, the Mask-based Prediction Head (MPH) predicts nutritional values via dynamic channel selection and region-aware attention.

### Key Designs

1. **Scale-Shift Residual Adapter (SSRA)**:

    - **Function**: Corrects global scale bias and local structural errors in the pretrained depth estimates.
    - **Mechanism**: Learns global scale factors and shift parameters for affine transformation to achieve global calibration, while a residual network predicts local corrections to preserve fine-grained structure.
    - **Design Motivation**: Pretrained depth models exhibit scale inconsistencies and local distortions when applied to food images.

2. **Frequency-Aligned Fusion Module (FAFM)**:

    - **Function**: Fuses RGB and depth features hierarchically in the frequency domain.
    - **Mechanism**: Features are transformed into the frequency domain, where RGB and depth components at different frequencies are aligned—low frequencies capture global shape while high frequencies capture texture details—enabling hierarchical cross-modal fusion.
    - **Design Motivation**: Direct spatial-domain fusion of RGB and depth features may cause information conflicts due to modality gaps; frequency-domain alignment provides a more natural fusion strategy.

3. **Mask-based Prediction Head (MPH)**:

    - **Function**: Focuses on key ingredient regions to improve prediction accuracy.
    - **Mechanism**: Dynamically selects the most informative feature channels and combines them with region-aware attention to emphasize key ingredient areas.
    - **Design Motivation**: Nutritional information density varies across regions in food images; backgrounds and containers introduce noise to the prediction.

### Loss & Training

Standard regression losses are used to predict calories and macronutrients. The NutritionSynth-115K synthetic dataset is used for pretraining to enhance generalization.

## Key Experimental Results

### Main Results

| Method | Calorie MAE↓ | Protein MAE↓ | Fat MAE↓ | Carbs MAE↓ |
|---|---|---|---|---|
| Im2Calories | 224.5 | 15.8 | 13.2 | 22.1 |
| Nutrition5K | 198.3 | 13.5 | 11.4 | 19.7 |
| RoDE | 185.7 | 12.8 | 10.6 | 18.3 |
| FBFPN (RGB+D) | 172.4 | 11.2 | 9.8 | 16.5 |
| **Ours (RGB only)** | **165.8** | **10.5** | **9.2** | **15.8** |

### Ablation Study

| Configuration | Calorie MAE↓ | Note |
|---|---|---|
| Full model | 165.8 | SSRA + FAFM + MPH |
| w/o SSRA | 178.2 | No depth correction |
| w/o FAFM (direct concat) | 175.6 | Spatial-domain concatenation |
| w/o MPH | 171.3 | Standard MLP head |
| w/o depth branch | 182.5 | RGB only |

### Key Findings

- SSRA contributes the most: removing depth correction increases MAE by approximately 12 points, confirming that raw predictions from the pretrained depth model exhibit significant bias.
- Frequency-domain fusion outperforms spatial-domain concatenation, validating the design motivation of FAFM.
- The RGB-only pipeline outperforms FBFPN, which uses real depth sensors.

## Highlights & Insights

- The strategy of replacing depth sensors with pretrained depth estimation models has practical value, enabling nutrition estimation to be deployed in everyday scenarios.
- The OmniFood8K dataset covers the complete cooking pipeline (ingredients → recipes → cooking videos → multi-view finished dishes), making it one of the most comprehensive datasets in the field.
- The construction methodology of NutritionSynth-115K offers a useful reference for data-scarce scenarios.

## Limitations & Future Work

- Coverage is limited to Chinese cuisine; cross-cultural generalizability has not been validated.
- The dataset scale (8,036 samples) remains relatively small by deep learning standards.
- The applicability of pretrained depth models to food images warrants further analysis.
- Integration with ingredient recognition and portion size estimation could further improve performance.

## Related Work & Insights

- **vs. Nutrition5K**: Nutrition5K focuses primarily on Western food and requires multi-view inputs, whereas this work covers Chinese food with only a single view.
- **vs. FBFPN**: FBFPN requires real RGB-D input, yet the proposed method, which predicts depth from a single RGB image, achieves superior performance.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the dataset and the frequency-domain fusion framework are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset comparisons and detailed ablation studies are provided.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured and clearly presented.
- Value: ⭐⭐⭐⭐ The dataset contribution is particularly significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] V-Nutri: Dish-Level Nutrition Estimation from Egocentric Cooking Videos](v_nutri_nutrition_estimation_cooking_videos.md)
- [\[AAAI 2026\] Private Frequency Estimation via Residue Number Systems](../../AAAI2026/others/private_frequency_estimation_via_residue_number_systems.md)
- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](../../AAAI2026/others/hierarchical_semantic_alignment_for_image_clustering.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)

</div>

<!-- RELATED:END -->
