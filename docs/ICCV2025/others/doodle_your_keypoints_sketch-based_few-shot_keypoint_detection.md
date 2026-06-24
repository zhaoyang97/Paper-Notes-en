---
title: >-
  [Paper Note] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection
description: >-
  [ICCV 2025][keypoint detection] This paper proposes the first sketch-based cross-modal few-shot keypoint detection framework. By leveraging a prototype network, grid-based locator, prototype domain adaptation, and a de-stylization network, the framework detects novel keypoints on unseen categories in real photographs using only a handful of annotated sketches.
tags:
  - "ICCV 2025"
  - "keypoint detection"
  - "few-shot learning"
  - "sketch"
  - "cross-modal"
  - "domain adaptation"
date: 2026-05-08
content_hash: ba29ea56e21569a6
---

# Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection

**Conference**: ICCV 2025
**arXiv**: [2507.07994](https://arxiv.org/abs/2507.07994)  
**Code**: [https://subhajitmaity.me/DYKp](https://subhajitmaity.me/DYKp)  
**Area**: Keypoint Detection / Few-Shot Learning
**Keywords**: keypoint detection, few-shot learning, sketch, cross-modal, domain adaptation

## TL;DR

This paper proposes the first sketch-based cross-modal few-shot keypoint detection framework. By leveraging a prototype network, grid-based locator, prototype domain adaptation, and a de-stylization network, the framework detects novel keypoints on unseen categories in real photographs using only a handful of annotated sketches.

## Background & Motivation

Keypoint detection is a fundamental problem in computer vision, with broad applications in pose estimation and landmark detection. Existing methods suffer from the following limitations:
- **Reliance on large-scale annotations**: Heatmap regression and direct regression methods require large annotated datasets.
- **Restricted few-shot generalization**: Existing few-shot keypoint methods are confined to specific image domains and cannot generalize to novel keypoints or unseen categories.
- **Inaccessible source data**: In real-world scenarios, source-domain images may be unavailable due to privacy, ethical, or scarcity constraints.

**Why sketches?** As one of the most natural forms of human expression, sketches offer unique advantages:
- Easy to obtain: a few strokes suffice to outline an object and annotate keypoints.
- No source-domain real images required: enables source-free few-shot detection.
- Practical relevance: for rare species, privacy-restricted, or heavily occluded scenarios, sketches may be the only feasible reference.

Core challenges include: (1) the large domain gap between sketches and photographs; (2) cross-modal embedding alignment at the keypoint level; and (3) style variation induced by differences in user drawing styles.

## Method

### Overall Architecture

The task is formulated as an N-way K-shot learning problem: given K annotated sketches (support set), detect N keypoints in M real photographs (query set). The framework consists of:
1. **Image encoder F**: extracts feature maps from support edge maps and query photographs.
2. **Keypoint extractor P**: extracts keypoint embeddings from feature maps via Gaussian pooling.
3. **De-stylization network Z**: maps support embeddings of varying styles to a style-agnostic representation.
4. **Prototype construction**: averages de-stylized support embeddings to form keypoint prototypes.
5. **Feature modulator M**: produces attention features via element-wise multiplication of prototypes and query features.
6. **Descriptor network D + Grid-Based Locator (GBL)**: multi-scale grid classification combined with offset regression for keypoint localization.

### Key Designs

**Gaussian pooling for keypoint extraction**:
$$\mathcal{P}(f_k, \mathbf{u}_{k,n}) = \sum_{\mathbf{x}} \exp\left(\frac{-\|\mathbf{x} - \mathbf{u}_{k,n}\|_2^2}{2\xi^2}\right) \cdot f_k[\mathbf{x}]$$

This extracts discriminative local context without requiring hard boundaries.

**Grid-Based Locator (GBL)**:
- Decomposes keypoint localization into two sub-problems:
    - **Grid classification**: predicts the $L_i \times L_i$ grid cell containing the keypoint (cross-entropy loss).
    - **Grid offset regression**: predicts the precise offset within the selected cell (L1 loss).
- Employs multi-scale grids $L = \{8, 12, 16\}$; the final prediction is the mean across scales.
- Simpler than FSKD's uncertainty modeling and better suited to the sparse nature of sketches.

**Prototype domain adaptation**:
- Inspired by Tanwisuth et al., a transport loss is used to align support prototypes with query keypoint embeddings.
- Replaces discriminative class probabilities with normalized distance-based similarity, better suited for keypoint localization.
- Converted to a supervised setting by exploiting known keypoint correspondences.

**De-stylization network Z**:
- Addresses style discrepancies arising from different edge detectors (PiDiNet, HED, Canny).
- Employs multi-scale channel attention to incorporate global context into local keypoint embeddings.
- A style loss minimizes embedding distances across different style variants.

### Loss & Training

The total loss comprises keypoint localization, domain adaptation, and de-stylization terms, each with an auxiliary counterpart:

$$\mathcal{L}_{total} = \lambda_{KP}(\mathcal{L}_{KP} + \mathcal{L}_{KP\text{-aux}}) + \lambda_{DA}(\mathcal{L}_{DA} + \mathcal{L}_{DA\text{-aux}}) + \lambda_{style}(\mathcal{L}_{style} + \mathcal{L}_{style\text{-aux}})$$

Hyperparameters: $\lambda_{KP} = 0.5$, $\lambda_{DA} = 0.001$, $\lambda_{style} = 0.001$, $\xi = 14$.

Auxiliary keypoints are generated by interpolation between pairs of visible keypoints at $t = \{0.25, 0.5, 0.75\}$, with up to 18 auxiliary keypoints per sample, substantially enhancing training.

The encoder is an ImageNet-pretrained ResNet50. Training runs for 80,000 episodes using the Adam optimizer with lr=0.0001.

## Key Experimental Results

### Main Results

Animal Pose dataset (1-shot), PCK@0.1:

| Category | Keypoints | B-Vanilla | FSKD | **Proposed** |
|----------|-----------|-----------|------|------------|
| Seen | Base | 44.16 | 48.75 | **55.10** |
| Seen | Novel | 18.06 | 37.99 | **45.14** |
| Unseen | Base | 40.47 | 38.14 | **43.17** |
| Unseen | Novel | 17.39 | 33.92 | **39.00** |

The proposed method surpasses FSKD by approximately 5 PCK points on the most challenging setting (unseen categories + novel keypoints).

Animal Kingdom dataset results (5 super-categories, 1-shot):

| Setting | B-Vanilla | FSKD | **Proposed** |
|---------|-----------|------|------------|
| Unseen Novel | 5.22 | 10.06 | **14.42** |

### Ablation Study

Contribution of each module (Unseen Novel, 1-shot):

| Method | w/o Aux | w/ Aux |
|--------|---------|--------|
| B-Vanilla | 17.39 | 29.98 |
| B-DA (+domain adaptation) | 18.31 | 31.76 |
| B-Style (+de-stylization) | 18.97 | 32.51 |
| **B-Full** | 19.03 | **39.00** |

- Auxiliary keypoints yield the largest gain (+12–20 PCK), far exceeding the individual contribution of any single module.
- B-Full benefits most from auxiliary keypoints (19.03 → 39.00), indicating strong synergy among all modules.

Generalization to real hand-drawn sketches (Sketchy database, 30 real sketches):
- Unseen Base: 42.40% (↓0.77)
- Unseen Novel: 38.49% (↓0.51)
- Negligible performance drop, validating robust transfer from synthetic edge maps to real sketches.

### Key Findings

1. **B-Vanilla baseline is extremely weak**: without domain adaptation and auxiliary keypoints, performance on novel keypoints is very poor (only 17–18 PCK).
2. **Auxiliary keypoints are critical**: they provide additional training signal for all modules, yielding gains far exceeding any individual component.
3. **Joint multi-modal training is superior**: using both sketches and photographs as support achieves 46.54 PCK, outperforming photo-only FSKD (44.75).
4. The transfer from synthetic edge map training to real hand-drawn sketch testing is surprisingly stable.

## Highlights & Insights

1. **First source-free cross-modal few-shot keypoint detection framework**: practically significant for rare species, privacy-restricted scenarios, and similar use cases.
2. **Elegant de-stylization design**: simulates style variation across edge detectors to adapt to real-world user drawing differences.
3. **Auxiliary keypoint strategy** achieves remarkable semi-supervised augmentation effects, offering a general data augmentation paradigm for few-shot tasks.
4. Demonstrates the viability of sketches as "the only feasible source data," opening a new research direction.

## Limitations & Future Work

1. Training relies on synthetic edge maps (PiDiNet/HED/Canny) rather than real sketches; actual user drawing variation may be substantially larger.
2. Evaluation is limited to animal datasets; generalization to artifacts, mechanical parts, or other domains remains unverified.
3. Accuracy under the 1-shot setting still has considerable room for improvement (best 39.00 PCK vs. 70+ for fully supervised methods).
4. The simplified GBL design (without uncertainty modeling) may be less flexible than FSKD in certain scenarios.
5. The shared encoder may limit the disentanglement of cross-modal features.

## Related Work & Insights

- **FSKD (Lu et al.)**: the pioneering work in few-shot keypoint detection and the primary baseline; employs uncertainty-modeling GBL.
- **Prototypical Networks**: the prototypical network paradigm is naturally extended from classification to keypoint localization.
- **Tanwisuth et al.**: prototype domain adaptation method that inspires the cross-modal keypoint alignment in this work.
- Insight: the application of sketches in computer vision expands from retrieval to structured geometric understanding (keypoints), with potential extensions to segmentation, 3D reconstruction, and beyond.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3.5 |
| Practical Value | 4 |
| Overall | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Feedforward Few-shot Species Range Estimation](../../ICML2025/others/feedforward_few-shot_species_range_estimation.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](../../ACL2025/others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ICCV 2025\] Stroke2Sketch: Harnessing Stroke Attributes for Training-Free Sketch Generation](stroke2sketch_harnessing_stroke_attributes_for_training-free_sketch_generation.md)

</div>

<!-- RELATED:END -->
