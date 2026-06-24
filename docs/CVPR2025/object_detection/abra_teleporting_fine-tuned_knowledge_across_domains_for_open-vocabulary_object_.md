---
title: >-
  [Paper Note] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection
description: >-
  [CVPR 2025][Object Detection][Open-Vocabulary Detection] Proposed ABRA (Aligned Basis Relocation for Adaptation), which "teleports" class-specific detection knowledge from a source domain to an unlabeled target domain by performing SVD decomposition and orthogonal rotation alignment in the weight space, achieving zero-shot cross-domain object detection.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Open-Vocabulary Detection"
  - "Domain Adaptation"
  - "Weight-Space Transmission"
  - "SVD Decomposition"
  - "Parameter-Efficient Fine-Tuning"
date: 2026-05-08
content_hash: 064b35d879ab4845
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2603.12409](https://arxiv.org/abs/2603.12409)  
**Code**: To be released (declared open source upon acceptance)  
**Area**: Object Detection / Open-Vocabulary Detection / Domain Adaptation  
**Keywords**: Open-Vocabulary Detection, Domain Adaptation, Weight-Space Transmission, SVD Decomposition, Parameter-Efficient Fine-Tuning

## TL;DR

Proposed ABRA (Aligned Basis Relocation for Adaptation), which "teleports" class-specific detection knowledge from a source domain to an unlabeled target domain by performing SVD decomposition and orthogonal rotation alignment in the weight space, achieving zero-shot cross-domain object detection.

## Background & Motivation

**Background**: Open-vocabulary object detection (OVD) models such as Grounding DINO showcase strong zero-shot capabilities, but their performance significantly degrades under domain shifts (e.g., nighttime, foggy weather).

**Limitations of Prior Work**:
   - Traditional domain adaptation (DAOD) methods rely on Mean Teacher and pseudo-labeling, which become unreliable under severe domain shifts.
   - Existing methods implicitly require the target domain to contain images of all categories, which constitutes weak supervision even without annotations.
   - Many practical scenarios where rare categories are completely absent in specific domains make traditional methods inapplicable.

**Key Challenge**: Certain classes in the target domain completely lack image data (no images, no labels), rendering any adaptation method requiring target-domain class images unusable.

**Goal**: In cases where certain categories in the target domain are completely unavailable (no images, no labels), transfer category-specific detection knowledge from the source domain to the target domain.

**Key Insight**: Decompose the adaptation process into two orthogonal dimensions: "domain knowledge" and "category knowledge", achieving cross-domain transfer via geometric transformations in the weight space.

**Core Idea**: In the singular value space of SVD decomposition, rotate the source-domain category residual into the target-domain basis space via orthogonal Procrustes alignment, achieving knowledge teleportation through closed-form solutions.

## Method

### Overall Architecture

The pipeline of ABRA consists of three steps:
1. **Objectification (Domain Expert Construction)**: Train category-agnostic domain experts for the source and target domains respectively.
2. **Class Expert (Category Expert Construction)**: Train lightweight class residuals using SVFT based on the source domain expert.
3. **Teleportation**: Teleport class residuals from the source domain to the target domain via SVD rotation alignment.

### Key Designs

1. **Objectification (Domain Expert)**

    - **Function**: Build class-agnostic domain experts to capture domain-specific visual statistics (e.g., illumination, texture).
    - **Mechanism**: Uniformly replace the annotations of the top-3 most frequent categories in the training set with "object" and discard other categories.
    - **Design Motivation**: Force the model to learn domain features rather than semantic features, ensuring domain experts are category-agnostic, which facilitates subsequent knowledge decomposition and transmission.

2. **SVFT Category Expert**

    - **Function**: Learn lightweight category-specific residuals $\Delta\Sigma$ based on domain experts.
    - **Mechanism**: Perform SVD decomposition $\theta = U\Sigma V^T$ on domain expert weights, freeze $U$, $\Sigma$, and $V$, and only train the diagonal residual $\Delta\Sigma$.
    - **Design Motivation**: The compact parameterization of SVFT (diagonal vectors only) enables efficient transfer of category knowledge with minimal parameter overhead.

3. **ABRA Teleportation Mechanism**

    - **Function**: Teleport source category residuals $\tau_S^{(c)}$ to the target domain.
    - **Mechanism**:
        - Perform SVD on the source and target domain experts to obtain their respective singular bases $(U_S, V_S)$ and $(U_T, V_T)$.
        - Solve the orthogonal Procrustes problem to find the optimal rotation matrices $L^* = U_T^T \cdot U_S$ and $R^* = V_T^T \cdot V_S$.
        - Target domain category weights after teleportation: $\theta_T^{(c)} = U_T \cdot (\Sigma_T + L \cdot \Delta\Sigma_S^{(c)} \cdot R^T) \cdot V_T^T$.
    - **Design Motivation**: A closed-form solution that requires no training in the target domain, mathematically equivalent to rotating the same residual between different data-specific basis spaces.

### Loss & Training

- **Domain Expert Training**: Standard detection loss, target-tuning only encoder attention layers, 10 epochs, lr=1e-4, batch=2.
- **Category Expert Training**: Fine-tuning with SVFT, 12 epochs, lr=1e-2, batch=4, only updating diagonal residuals.
- **Teleportation Process**: Requires no training, pure closed-form computation (SVD + matrix multiplication).

## Key Experimental Results

### Main Results

**Cityscapes → Foggy Cityscapes (mAP / AP50)**

| Method | Bus | Motor | Rider | Train | Truck | Average |
|------|-----|-------|-------|-------|-------|---------|
| Fine-tuning (Upper Bound) | 58.75/73.50 | 31.22/55.01 | 43.87/68.06 | 32.95/58.75 | 40.03/57.08 | 41.36/62.48 |
| Zero shot | 48.63/61.10 | 23.20/41.91 | 18.96/31.70 | 16.31/44.02 | 31.20/41.88 | 27.66/44.12 |
| Source | 54.77/66.93 | 29.62/50.96 | 40.25/61.99 | 29.40/55.30 | 37.23/51.54 | 38.25/57.34 |
| Task Analogy | 41.14/48.21 | 10.24/20.32 | 9.35/16.75 | 10.10/24.00 | 19.77/24.70 | 18.12/26.79 |
| ParamΔ | 50.23/60.41 | 20.59/40.59 | 18.53/31.44 | 21.70/49.42 | 30.42/40.26 | 28.29/44.42 |
| **ABRA (ours)** | **57.24/70.53** | **29.98/55.47** | **42.27/66.08** | **35.09/59.94** | **38.10/53.27** | **40.54/61.06** |

**SDGOD Cross-Domain Experiments (mAP / AP50)**

| Method | Day Foggy | Dusk Rainy | Night Clear | Night Rainy | Average |
|------|-----------|------------|-------------|-------------|---------|
| Fine-tuning | 36.37/57.83 | 26.77/50.65 | 36.86/69.40 | 16.81/29.85 | 29.20/51.93 |
| Zero shot | 26.36/41.10 | 19.55/34.93 | 27.50/49.63 | 9.19/13.60 | 20.65/34.82 |
| ABRA (ours) | **32.35/53.81** | **27.99/51.39** | 35.94/66.11 | **16.13/30.97** | **28.10/50.57** |

### Ablation Study

**Ablation of Domain Expert Design Choices**

| Strategy | Effect |
|------|------|
| Zero Shot w/ Obj. | Combination of zero-shot + objectification, performance is limited |
| Supervised (Keeping Semantic Labels) | Kept original class labels for training, poor transferability |
| **Objectification** | Optimal, category-agnostic representation is most suitable for cross-domain transfer |

**Initialization Effects**

| Method | Init. | mAP | AP50 |
|------|-------|-----|------|
| FFT | $\theta_0$ | 41.36 | 62.48 |
| FFT | ABRA | **42.80** | **62.77** |
| FDA | $\theta_0$ | 38.25 | 57.80 |
| FDA | ABRA | **40.74** | **61.35** |

### Key Findings

- ABRA achieves an average mAP of 40.54 in the Cityscapes→Foggy experiment, approaching the full fine-tuning upper bound of 41.36, and far exceeding all competitors.
- Task Analogy and ParamΔ fail under severe domain shifts (the average mAP of ParamΔ on SDGOD is only 8.87).
- Class-level expertization (one model per class) yields higher AP50 across all classes than merged experts (Merge).
- ABRA provides better initialization for downstream tasks; both FFT and FDA initialized with ABRA improve by 1-3 percentage points.
- In few-shot experiments, ABRA consistently outperforms $\theta_0$ initialization across 1/5/10/20/30 shots.

## Highlights & Insights

1. **Elegant Mathematical Framework**: Models cross-domain knowledge transfer as a geometric transformation problem in the weight space, where the orthogonal Procrustes alignment has a closed-form solution without requiring iterative optimization.
2. **Knowledge Decomposition Philosophy**: Orthogonally decomposes "domain knowledge" and "category knowledge", allowing category knowledge to be transferred independently. This conceptualization is highly ingenious.
3. **Objectification Design**: Eliminates category semantics using a unified "object" label, forcing the model to learn pure domain features—simple yet effective.
4. **Compactness of SVFT**: Category residuals are only diagonal matrices (vectors), entailing extremely low transfer costs.
5. **Training-Free Teleportation**: The target domain gains category detection capabilities without requiring any training steps.

## Limitations & Future Work

1. **Dependency on SVD Decomposition Assumption**: Assumes that the weight spaces of the source and target domains can be aligned via orthogonal transformation, which may not hold true under extreme domain shifts.
2. **Objectification Using Only Top-3 Categories**: The category selection strategy is relatively simple and may omit important domain information.
3. **Independent Single-Category Teleportation**: Each category requires independent SVFT training and teleportation, leaving inter-category correlation unmodeled.
4. **Limited Baselines**: Currently only compared with Task Analogy and ParamΔ; more DAOD methods could be included.
5. **Experimental Scale**: Validated only on Cityscapes and SDGOD; not yet verified on larger-scale scenarios (e.g., COCO to other domains).

## Related Work & Insights

- **Relationship with LoRA/SVFT**: ABRA leverages the compact residuals from SVFT to parameterize category knowledge, which can be generalized to other PEFT methods.
- **Relationship with Model Rebasin**: The core idea is similar to model rebasin, but focuses on cross-domain transmission rather than model fusion.
- **Difference from Task Arithmetic**: Task Arithmetic performs simple addition/subtraction, while ABRA performs addition after rotational alignment, considering the differences in basis spaces.
- **Inspiration**: The orthogonal decomposition of domain/category can be extended to other vision-language adaptation scenarios (e.g., domain adaptation for VLMs).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Elegant theory modeling cross-domain adaptation as geometric teleportation in weight space)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough ablations, but dataset scale and baseline coverage can be strengthened)
- Writing Quality: ⭐⭐⭐⭐⭐ (Well-structured with complete mathematical derivations)
- Value: ⭐⭐⭐⭐ (Addresses a practical and previously overlooked problem setting)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)
- [\[AAAI 2026\] VK-Det: Visual Knowledge Guided Prototype Learning for Open-Vocabulary Aerial Object Detection](../../AAAI2026/object_detection/vk-det_visual_knowledge_guided_prototype_learning_for_open-vocabulary_aerial_obj.md)
- [\[CVPR 2026\] Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](../../CVPR2026/object_detection/parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](../../NeurIPS2025/object_detection/dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[CVPR 2026\] WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](../../CVPR2026/object_detection/wedetect_fast_open-vocabulary_object_detection_as_retrieval.md)

</div>

<!-- RELATED:END -->
