---
title: >-
  [Paper Note] Zero-shot Object Counting with Good Exemplars (VA-Count)
description: >-
  [ECCV 2024][Multimodal VLM][Zero-shot object counting] This work proposes the VA-Count framework, which leverages Grounding DINO via an Exemplar Enhancement Module (EEM) to discover high-quality positive and negative exemplars, combined with a Noise Suppression Module (NSM) utilizing contrastive learning to distinguish positive and negative density maps, achieving state-of-the-art zero-shot object counting performance on FSC-147 and CARPK.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Zero-shot object counting"
  - "Vision-Language Pre-training"
  - "Contrastive learning"
  - "Exemplar enhancement"
  - "Noise suppression"
date: 2026-05-08
content_hash: e6e0720f4e7599a7
---

# Zero-shot Object Counting with Good Exemplars (VA-Count)

**Conference**: ECCV 2024  
**arXiv**: [2407.04948](https://arxiv.org/abs/2407.04948)  
**Code**: [GitHub](https://github.com/HopooLinZ/VA-Count)  
**Area**: Multimodal VLM  
**Keywords**: Zero-shot object counting, Vision-Language Pre-training, Contrastive learning, Exemplar enhancement, Noise suppression

## TL;DR

This work proposes the VA-Count framework, which leverages Grounding DINO via an Exemplar Enhancement Module (EEM) to discover high-quality positive and negative exemplars, combined with a Noise Suppression Module (NSM) utilizing contrastive learning to distinguish positive and negative density maps, achieving state-of-the-art zero-shot object counting performance on FSC-147 and CARPK.

## Background & Motivation

Object counting is crucial in scenarios such as security monitoring. Traditional methods are limited to specific categories (e.g., crowds, vehicles) and fail to generalize to unseen classes. Existing class-agnostic counting methods can be categorized as:

**Few-shot methods**: Require a few annotated bounding boxes, but still need annotations for new categories, limiting practical deployment.

**Reference-free methods**: No annotations are required, but they cannot specify the target category for counting, making them susceptible to background noise.

**Zero-shot methods**: Require only the category name to count, thus possessing the highest practical value.

The core issue of existing zero-shot counting methods is **the inability to effectively identify high-quality exemplars**:
- **Vision-Language Alignment methods** (e.g., CLIP-Count): Directly align text and images using CLIP, making it difficult to precisely represent target categories with atypical shapes.
- **Exemplar Discovery methods** (e.g., ZSC): Match image patches using prototypes generated from text, but the selection of patches is arbitrary, failing to accurately bound complete objects, and is limited to predefined categories in the training set.

**Key Challenge**: How to automatically discover high-quality exemplars that accurately represent the target category and establish robust visual associations without manual annotation. The **Core Idea** of VA-Count is to leverage the open-vocabulary detection capability of VLP models (specifically Grounding DINO) to discover exemplars, followed by binary classification filtering and contrastive learning to enhance exemplar quality and suppress noise.

## Method

### Overall Architecture

VA-Count consists of two core modules:
- **Exemplar Enhancement Module (EEM)**: Responsible for discovering and filtering high-quality positive and negative exemplars from images.
- **Noise Suppression Module (NSM)**: Utilizes contrastive learning to distinguish positive and negative density maps, suppressing the impact of erroneous exemplars.

Overall Pipeline: Input image $\rightarrow$ Grounding DINO candidate box detection $\rightarrow$ Single-object filter $\rightarrow$ Positive & negative exemplar separation $\rightarrow$ Counter generates positive & negative density maps $\rightarrow$ Joint optimization with contrastive loss + density loss.

### Key Designs

1. **Grounding DINO-guided candidate bounding box selection**:

    - **Function**: Leverage the open-vocabulary detection capability of Grounding DINO to generate candidate bounding boxes for any category.
    - **Mechanism**:
        - Positive exemplar detection: Input a specific category text (e.g., "dog"), Grounding DINO outputs candidate boxes and their confidence scores.
        - Negative exemplar detection: Input a generic text "object" to detect bounding boxes of all objects in the image.
        - Deduplication: Filter out negative exemplar boxes overlapping with positive ones using an IoU threshold ($\tau_{iou}=0.5$).
    - **Design Motivation**: Grounding DINO is pre-trained on large-scale data and possesses strong open-vocabulary detection capability, ensuring the adaptability of the framework to arbitrary categories.

2. **Single-object Exemplar Filter**:

    - **Function**: Ensure that each exemplar box contains only one target object.
    - **Mechanism**: Construct a binary classifier $\delta(\cdot)$ that uses a frozen CLIP ViT-B/16 to extract features, followed by a feed-forward network (FFN) to perform binary classification to determine if the box contains a single object or multiple objects.
    - Training Data Construction:
        - Single-object positive exemplars: Annotated exemplars from the training set.
        - Multi-object negative exemplars: Randomly cropped image patches and full images.
        - The data is split into 7:3 for training/validation, with non-overlapping classes to maintain class agnosticism.
    - **Design Motivation**: High-confidence boxes from Grounding DINO might contain multiple objects (the confidence of multi-object boxes can sometimes be higher than that of single-object boxes), which disrupts subsequent visual association learning. Therefore, strict filtering is necessary.

3. **Counter Network (Feature Interaction and Density Map Generation)**:

    - **Function**: Generate density maps based on positive and negative exemplars separately.
    - **Mechanism**: Based on the CounTR architecture, use image features as the Query, and linear projections of exemplar features as Key/Value for cross-attention fusion.
    - Fusion formula: $F_{fuse} = \Gamma_{fuse}(F_{query}, W^k \cdot F_{key}, W^v \cdot F_{value})$
    - The decoder upsamples the fused features to original image size and outputs the density map.
    - The top-3 boxes with the highest confidence scores are selected as final exemplars for positive and negative samples, respectively.

### Loss & Training

**Contrastive Loss $L_C$**:
- Maximize the similarity between the positive density map and the ground-truth (GT) density map.
- Minimize the similarity between the negative density map and the GT density map.
- Adopt InfoNCE formulation: $L_C = -\log \frac{\exp(\text{sim}(D^p, D^g))}{\exp(\text{sim}(D^p, D^g)) + \exp(\text{sim}(D^n, D^g))}$

**Density Loss $L_D$**:
- Pixel-wise mean squared error between the positive exemplar density map and the GT density map.
- $L_D = \frac{1}{HW} \sum \|D^p - D^g\|^2$

**Total Loss**: $L_{total} = L_C + L_D$

**Training Strategy**:
- Adopt the two-stage training scheme of CounTR: MAE pre-training + fine-tuning.
- Learning rate: $10^{-5}$, batch size: 8.
- Grounding DINO confidence threshold $\tau_l = 0.02$.
- The single-object classifier is trained for 100 epochs with a learning rate of $1e-4$.

## Key Experimental Results

### Main Results: FSC-147 Zero-Shot Counting

| Method | Type | Val MAE↓ | Val RMSE↓ | Test MAE↓ | Test RMSE↓ |
|------|------|---------|----------|----------|-----------|
| ZSC | Zero-shot | 26.93 | 88.63 | 22.09 | 115.17 |
| CLIP-Count | Zero-shot | 18.79 | 61.18 | 17.78 | 106.62 |
| PseCo | Zero-shot | 23.90 | 100.33 | 16.58 | 129.77 |
| **VA-Count** | **Zero-shot** | **17.87** | 73.22 | **17.88** | 129.31 |
| CounTR | Few-shot(3) | 13.13 | 49.83 | 11.95 | 91.23 |
| CACViT | Few-shot(3) | 10.63 | 37.95 | 9.13 | 48.96 |

### Cross-Domain Experiment: CARPK

| Method | Type | F→C MAE↓ | F→C RMSE↓ | Description |
|------|------|----------|-----------|------|
| FamNet | Few-shot(3) | 28.84 | 44.47 | Cross-domain few-shot |
| RCC | Zero-shot | 21.38 | 26.61 | - |
| CLIP-Count | Zero-shot | 11.96 | 16.61 | - |
| Grounding DINO | Zero-shot | 29.72 | 31.60 | Direct counting using G-DINO |
| G-DINO + Filter | Zero-shot | 18.54 | 21.71 | With single-object filter |
| **VA-Count** | **Zero-shot** | **10.63** | **13.20** | SOTA |

### Ablation Study

| G(·) | Φ(·) | $L_D$ | $L_C$ | Val MAE | Test MAE | Description |
|------|------|-----|-----|---------|----------|------|
| ● | ○ | ○ | ○ | 52.82 | 54.48 | Grounding DINO detection only |
| ● | ● | ○ | ○ | 52.12 | 54.27 | + Single-object filter (slight improvement) |
| ● | ● | ● | ○ | 19.63 | 18.93 | + Density loss (significant decrease) |
| ● | ● | ● | ● | **17.87** | **17.88** | + Contrastive loss (further improvement) |

### Key Findings
- VA-Count achieves the lowest MAE in the FSC-147 zero-shot setting, proving that its exemplar discovery strategy is superior to ZSC.
- Prominent performance in cross-domain CARPK experiments, with zero-shot performance being close to few-shot methods (MAE 10.63 vs 10.44).
- The single-object filter shows a significant effect on CARPK, reducing Grounding DINO's MAE from 29.72 to 18.54 (a reduction of approximately 10).
- The introduction of contrastive loss further reduces MAE by about 2 points on top of the density loss.
- The density loss is the most critical component, whose introduction drops the MAE from 52 to ~19.

## Highlights & Insights
- **Clever Leveraging of VLP Models**: Introducing the open-vocabulary detection capability of Grounding DINO into zero-shot counting breaks the limitation of relying on fixed category prototypes.
- **Dual-Channel Positive/Negative Exemplar Design**: Instead of only using positive exemplars to establish association, it also uses negative exemplars (objects of non-target classes) for contrastive learning to suppress noise, analogous to "teaching the model what it is not" in classification.
- **Simple and Effective Single-Object Filter**: A simple binary classifier elegantly solves the key problem where Grounding DINO boxes may contain multiple objects.
- **Strong Cross-Domain Generalization**: The transfer performance from FSC-147 to CARPK is superior to most baselines.

## Limitations & Future Work
- The RMSE metric is inferior to CLIP-Count, indicating the existence of a few extreme error samples and suggesting room for improvement in robustness.
- The computational overhead of Grounding DINO itself is large, which may pose a bottleneck for inference efficiency.
- The construction of training data for the single-object classifier is relatively simple (random cropping), which may introduce class bias.
- Grounding DINO's detection quality may degrade in dense scenarios, thereby affecting exemplar quality.
- Selecting a fixed top-3 exemplars lacks an adaptive mechanism.
- It is worth exploring the use of SAM to replace or assist Grounding DINO for more precise instance segmentation.

## Related Work & Insights
- **CounTR**: The base architecture source for the Counter network, utilizing the two-stage MAE pre-training + fine-tuning strategy.
- **CLIP-Count**: Encodes text and images with CLIP to establish semantic associations; VA-Count introduces explicit exemplar discovery on top of this.
- **Grounding DINO**: The core external tool providing open-vocabulary detection capability.
- **BMNet**: A bilinear matching network used for fine-grained similarity evaluation, which inspired the design of feature interactions.
- *Insight*: VLP models acting as an "external knowledge source" offer a new paradigm for zero-shot tasks; the key lies in how to handle the noise of their outputs.

## Rating
- Novelty: ⭐⭐⭐⭐ First to introduce Grounding DINO to zero-shot counting; the dual-channel positive/negative design along with contrastive learning is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation featuring two datasets, cross-domain experiments, component-by-component ablation, and qualitative analysis.
- Writing Quality: ⭐⭐⭐ Method descriptions are slightly verbose with numerous symbolic definitions, leading to moderate readability.
- Value: ⭐⭐⭐⭐ High practical value for zero-shot counting; the ideas can be migrated to other zero-shot visual tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Locality-Aware Zero-Shot Human-Object Interaction Detection](../../CVPR2025/multimodal_vlm/locality-aware_zero-shot_human-object_interaction_detection.md)
- [\[ECCV 2024\] Meta-Prompting for Automating Zero-Shot Visual Recognition with LLMs](meta-prompting_for_automating_zero-shot_visual_recognition_with_llms.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[ECCV 2024\] Elysium: Exploring Object-level Perception in Videos via MLLM](elysium_exploring_object-level_perception_in_videos_via_mllm.md)
- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](../../ACL2025/multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)

</div>

<!-- RELATED:END -->
