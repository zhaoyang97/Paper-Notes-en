---
title: >-
  [Paper Note] SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images
description: >-
  [ECCV 2024][Segmentation][Hierarchical Segmentation] SPIN constructs SubPartImageNet, the first hierarchical semantic segmentation dataset with subpart-level granularity for natural images, containing 203 subpart categories and 106k annotations. It proposes two hierarchical consistency evaluation metrics (SpCS / SeCS) and performs a comprehensive benchmark on over 20 modern models, revealing severe limitations of current models at the subpart level.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Hierarchical Segmentation"
  - "Subpart Segmentation"
  - "Dataset Benchmark"
  - "Evaluation Metrics"
  - "Fine-grained Visual Understanding"
date: 2026-05-08
content_hash: f4eb284d3ceeb0f8
---

# SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images

**Conference**: ECCV 2024  
**arXiv**: [2407.09686](https://arxiv.org/abs/2407.09686)  
**Code**: [https://joshmyersdean.github.io/spin/index.html](https://joshmyersdean.github.io/spin/index.html) (Yes, dataset is public)  
**Area**: Segmentation  
**Keywords**: Hierarchical Segmentation, Subpart Segmentation, Dataset Benchmark, Evaluation Metrics, Fine-grained Visual Understanding

## TL;DR

SPIN constructs SubPartImageNet, the first hierarchical semantic segmentation dataset with subpart-level granularity for natural images, containing 203 subpart categories and 106k annotations. It proposes two hierarchical consistency evaluation metrics (SpCS / SeCS) and performs a comprehensive benchmark on over 20 modern models, revealing severe limitations of current models at the subpart level.

## Background & Motivation

Hierarchical image analysis involves two types of relationships: "is-a" relationships (e.g., category inheritance like "Subaru is a car") and "is-part-of" relationships (e.g., compositional decomposition like "door is part of a car"). While the former has been widely studied, research on the latter is mostly limited to two-level object-to-part decomposition, leaving deeper subparts (parts of parts, e.g., an eye is a subpart of a head) almost entirely unexplored.

**Limitations of Prior Work**:
- Lack of subpart annotation data in natural images: Although synthetic 3D datasets provide hierarchical annotations, models trained on synthetic data generalize poorly to real-world images.
- ADE20K provides subpart annotations for only 10% of objects, and these annotations are non-exhaustive, making quantitative evaluation unfeasible.
- A few models supporting subpart prediction (such as HIPIE, VDT, and Semantic-SAM) can only provide qualitative demonstrations.
- Existing evaluation metrics assess the IoU/AP of each granularity level independently, ignoring the spatial and semantic consistency across different hierarchical levels.

**Core Idea**: To construct the first natural image segmentation dataset with exhaustive three-layer semantic annotations of object→part→subpart, design cross-hierarchical evaluation metrics, and establish a complete benchmarking framework.

## Method

### Overall Architecture

SPIN does not propose a new segmentation method; instead, its contributions consist of three parts:
1. **Dataset**: Extending subpart annotations on top of PartImageNet.
2. **Evaluation Metrics**: SpCS and SeCS to measure spatial and semantic consistency across hierarchies.
3. **Benchmark**: Evaluating three tasks (open-vocabulary localization, interactive segmentation, and semantic recognition) across over 20 models.

### Key Designs

1. **SPIN Dataset Construction**:

    - Data Source: Based on PartImageNet (24,080 images, 158 ImageNet categories), selecting 10,387 images containing at least one segmented part.
    - Subpart Category Selection: GPT-4 was utilized to generate a candidate subpart list for each object-part pair. Three authors manually removed invisible or ambiguous categories, retaining 206 candidates, of which 203 were used in the final annotations.
    - Category Composition: 168 general subparts (e.g., "mouth" spanning multiple object categories) + 38 target-specific subparts (e.g., "shell" only applicable to the body of a turtle).
    - Annotation Pipeline: 18 AMT annotators, validated through long-term collaboration, were employed. Quality was guaranteed via a five-fold mechanism: onboarding tests, detailed guidelines, real-time "Q&A office hours," phased releases, and continuous quality checks.
    - Final Scale: 11 object superclasses, 40 part categories, 203 subpart categories, and 106,324 semantic annotations.
    - Splits: Train 8,828 / Val 519 / Test 1,040

2. **Spatial Consistency Score (SpCS)**:

    - **Function**: Evaluates whether a child region prediction is correctly spatially contained within its parent region.
    - **Formula**: $SpCS = \frac{1}{|\mathcal{R}|} \sum_{(\text{child}, \text{parent})} \frac{|\text{child} \cap \text{parent}|}{|\text{child}|}$
    - Range [0,1], where 1 represents perfect containment (e.g., the predicted eye is completely within the predicted head).
    - **Design Motivation**: Traditional IoU evaluates each hierarchy level independently, failing to penalize cases where a model predicts a subpart outside of its parent region.

3. **Semantic Consistency Score (SeCS)**:

    - **Function**: Evaluates whether pixel-level predictions follow reasonable cross-hierarchical semantic entailment.
    - **Mechanism**: For each foreground pixel $x$ that has predictions at all three levels, it checks whether the semantic entailment chain of subpart → part → object holds true in the ground truth relationships.
    - For example, "eye → head → quadruped" is valid, whereas "windshield → head → bottle" is invalid.
    - SeCS is calculated as the average proportion of semantically correct predictions across all foreground pixels.

### Loss & Training

This paper does not propose a new training method. The only fine-tuning experiment conducts full-parameter fine-tuning of GLaMM, the best-performing zero-shot model, on the SPIN training set, following GLaMM's original training strategy. The resulting model is named GLaMM-FT.

## Key Experimental Results

### Main Results

Zero-shot open-vocabulary localization performance (mIoU / SpCS):

| Method | Params | mIoU_Subpart | mIoU_Part | mIoU_Object | SpCS_S2P | SpCS_P2O |
|------|--------|-------------|-----------|-------------|----------|----------|
| HIPIE (R-50) | 200M | 0.80 | 8.05 | 51.36 | 100.0 | 95.64 |
| HIPIE (ViT-H) | 800M | 0.90 | 7.21 | 66.77 | 100.0 | 96.08 |
| PixelLLM-13B | 13B | 9.53 | 32.04 | 82.90 | 82.13 | 92.60 |
| LISA-13B | 13B | 11.52 | 32.29 | 87.78 | 82.87 | 96.02 |
| GLaMM | 7B | 11.03 | 39.41 | 86.29 | 84.21 | 95.72 |
| **GLaMM-FT** | 7B | **24.25** | **59.37** | 86.42 | 75.93 | 90.23 |
| CoGVLM | 17B | 13.94 | 38.13 | 46.47 | 72.37 | 86.19 |
| SAM (GT box) | 630M | 49.61 | 69.23 | **90.06** | 83.85 | 92.30 |

### Ablation Study

Impact of fine-tuning on SPIN (General category prompts):

| Configuration | mIoU_S | Gain | mIoU_P | Gain | mIoU_O | Gain |
|------|--------|------|--------|------|--------|------|
| GLaMM Zero-Shot | 11.00 | - | 40.00 | - | 86.31 | - |
| GLaMM-FT | 24.56 | +123% | 60.76 | +52% | 91.08 | +5.5% |

Distribution statistics of subpart sizes:

| Scale Category | Area Threshold | Annotation Share | Count |
|---------|---------|---------|------|
| Small | ≤ 32² | 54.10% | 57,525 |
| Medium | 32²–96² | 38.08% | 40,488 |
| Large | ≥ 96² | 7.82% | 8,311 |

### Key Findings

1. **All models fail severely at the subpart level**: The best zero-shot mIoU is only ~14, and even with fine-tuning, it only reaches ~25.
2. **Specialized model HIPIE performs the worst**: Its subpart IoU is < 1, as its training part vocabulary is limited to Pascal Parts.
3. **SAM + positional priors significantly outperform language models**: SAM using GT BBoxes achieves a subpart mIoU of 49.6, which is 4.5 times higher than that of GLaMM.
4. **Subparts are inherently small objects**: 54% of subpart areas are < 32², naturally embedding the challenges of small object detection.
5. **Part → Object spatial consistency is high (>90%), while Subpart → Part drops significantly with large fluctuations (46%~85%)**.
6. **Granularity of category names affects localization**: The effect is significant at the object level (e.g., "quadruped" vs. "dog"), but very small at the subpart level.

## Highlights & Insights

- **Defines a new fine-grained segmentation problem**: The object → part → subpart three-layer decomposition sets a new benchmark for hierarchical visual understanding.
- **Elegant evaluation metric design**: SpCS and SeCS incorporate hierarchical relationships into segmentation evaluation for the first time, complementing traditional IoU.
- **GPT-4 assisted dataset category design**: Demonstrates a practical approach to leveraging LLMs to accelerate dataset ontology construction.
- **Reveals the performance gap between positional and language priors**: Suggests that future directions should combine the localization capability of SAM with the semantic understanding of VLMs.

## Limitations & Future Work

- Limited dataset scale (~10k images), where each image typically contains only one salient object.
- Subjectivity in subpart category definitions (derived from GPT-4 and manual filtering).
- Annotation budget constraints limit each superclass to a maximum of 1,200 images.
- The work is limited to benchmarking and does not propose a new method specifically designed for subpart segmentation.
- The distinction between rigid and non-rigid objects does not hold at the subpart level, but the underlying reasons are not deeply analyzed.

## Related Work & Insights

- **PartImageNet**: The direct predecessor of SPIN, providing two-layer object-part annotations.
- **ADE20K**: Contains a small amount of subpart annotations but is non-exhaustive, making it unsuitable for quantitative evaluation.
- **HIPIE**: The only hierarchical model attempting subpart prediction, but performs poorly due to limitations in its training part vocabulary.
- **SAM**: Demonstrates upper-bound potential in fine-grained localization.
- Insights: (1) Training data scale can be expanded through synthetic data and 3D-to-2D projections; (2) Hierarchical segmentation requires hierarchy-aware loss functions rather than independent optimization of each layer.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The first natural image dataset at the subpart level, with pioneering problem definition)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation of over 20 models across multiple tasks, but lacks a dedicated new methodology)
- Writing Quality: ⭐⭐⭐⭐⭐ (Well-organized and highly detailed data analysis)
- Value: ⭐⭐⭐⭐ (Provides essential infrastructure for fine-grained hierarchical segmentation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](../../AAAI2026/segmentation/bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[ECCV 2024\] Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images](active_coarsetofine_segmentation_of_moveable_parts_from_real.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](../../CVPR2025/segmentation/binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[ICLR 2026\] Hierarchical Prototype Learning for Semantic Segmentation](../../ICLR2026/segmentation/hierarchical_prototype_learning_for_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
