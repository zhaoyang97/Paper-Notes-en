---
title: >-
  [Paper Note] Free-Grained Hierarchical Visual Recognition
description: >-
  [CVPR 2026][LLM Evaluation][Hierarchical Classification] This paper proposes *free-grained hierarchical recognition*, a setting in which training labels may appear at any level of a taxonomy. Two complementary methods are introduced to compensate for missing supervision — text-guided pseudo-attributes (Text-Attr) and taxonomy-guided semi-supervised learning (Taxon-SSL) — while at inference time the model adaptively selects its prediction depth.
tags:
  - CVPR 2026
  - LLM Evaluation
  - Hierarchical Classification
  - Mixed-Granularity Annotation
  - Semi-Supervised Learning
  - Text Guidance
  - Taxonomy
date: 2026-05-08
content_hash: a9c74cb2565f409f
---

# Free-Grained Hierarchical Visual Recognition

**Conference**: CVPR 2026
**arXiv**: [2510.14737](https://arxiv.org/abs/2510.14737)
**Code**: [FreeGrainLearning](https://github.com/seulkipark/FreeGrainLearning)
**Area**: LLM Evaluation
**Keywords**: Hierarchical Classification, Mixed-Granularity Annotation, Semi-Supervised Learning, Text Guidance, Taxonomy

## TL;DR

This paper proposes *free-grained hierarchical recognition*, a setting in which training labels may appear at any level of a taxonomy. Two complementary methods are introduced to compensate for missing supervision — text-guided pseudo-attributes (Text-Attr) and taxonomy-guided semi-supervised learning (Taxon-SSL) — while at inference time the model adaptively selects its prediction depth.

## Background & Motivation

- **State of the Field**: Conventional hierarchical classification assumes that every training image is fully annotated at all levels of the taxonomy (e.g., Bird → Bird of prey → Bald eagle). In practice, however, annotations are frequently incomplete.
- **Limitations of Prior Work**: Labels may be absent for intrinsic reasons (insufficient visual evidence to support fine-grained assignment, e.g., a distant bird identifiable only as "bird") or extrinsic reasons (annotation cost, annotator expertise, or labeling protocols).
- **Root Cause**: Existing SOTA hierarchical classification methods (e.g., H-CAST) assume complete label paths; their performance degrades catastrophically when labels are available only at coarser levels.
- **Paper Goals**: The paper formalizes the **free-grained learning** setting, in which training labels may appear at any taxonomy level and annotation depth may vary across samples. The model must learn consistent hierarchical predictions from this incomplete, mixed-granularity supervision.
- **Starting Point**: Experiments show that H-CAST, when transferred from full-label to free-grained settings, suffers a Full-Path Accuracy (FPA) drop of 19–40 percentage points (e.g., iNat21-mini: 64.9% → 25.6%), confirming the difficulty of this setting.

## Method

### Overall Architecture

The work comprises three main components:
1. **Benchmark dataset construction**: adapting existing hierarchical datasets to the free-grained setting.
2. **Two training methods**: Text-Attr and Taxon-SSL.
3. **Free-grained inference**: adaptive selection of prediction depth at test time.

### Key Designs

1. **ImageNet-3L Dataset Construction**: The original ImageNet WordNet hierarchy is irregular (depths of 5–19 levels, 30% of classes with multiple paths) and unsuitable for hierarchical evaluation. This work reorganizes it into a clean three-level taxonomy (20 basic / 127 subordinate / 505 fine-grained classes) following principles from cognitive psychology. Design criteria include: removing single-child paths, maximizing intra-group diversity, refining ambiguous categories, and validation via LLM-assisted human review.

2. **Foundation-based Pruning**: Zero-shot predictions from CLIP/BioCLIP are used to simulate realistic mixed-granularity annotation. Starting from coarse levels and proceeding to fine levels, each prediction is checked for correctness: the subordinate label is retained if the prediction at that level is correct, and the fine-grained label is further retained if correct. Incorrect level labels are removed. In ImageNet-F, 32.6% of samples retain all three levels, 28.0% retain two levels, and 39.4% retain only the basic level.

3. **Text-Attr (Text-Guided Pseudo-Attributes)**: The core observation is that many visual attributes (e.g., "short legs," "pointed ears") are consistent across taxonomy levels even when class labels differ. A frozen VLM (Llama-3.2-11B) generates textual descriptions for each image; these are encoded by a CLIP text encoder and aligned with image features via contrastive learning. This text-based supervision is independent of class labels and provides additional semantic cues when fine-grained annotations are absent.

4. **Taxon-SSL (Taxonomy-Guided Semi-Supervised Learning)**: Missing-level labels are treated as unlabeled data. The key innovation is a **taxonomy-aligned affinity graph**: two samples are treated as a positive pair only when their pseudo-labels agree at **all** levels (Equation 3). This effectively filters noisy pseudo-labels and enforces hierarchical consistency. A contrastive loss then pulls positive pairs together and pushes negative pairs apart.

### Loss & Training

- **Free-grained hierarchical loss**: $\mathcal{L}_{hier} = \sum_l \mathbb{1}_{y_l \text{ exists}} \cdot \mathcal{L}(f_l(x), y_l)$, applying supervision only at levels where labels exist.
- **Text contrastive loss**: InfoNCE loss for image–text embedding alignment.
- **Taxonomy-aligned contrastive loss**: contrastive learning based on pseudo-labels consistent across all hierarchy levels.
- Backbone: ViT-Small (H-ViT) or H-CAST; trained for 100 epochs (200 epochs on ImageNet-F).

## Key Experimental Results

### Main Results

| Dataset | Method | FPA ↑ | Fine ↑ | Sub ↑ | Basic ↑ | TICE ↓ |
|---------|--------|-------|--------|-------|---------|--------|
| ImageNet-F | H-CAST (full→free) | 57.59 | 59.02 | 82.69 | 93.53 | 21.81 |
| ImageNet-F | Text-Attr (H-CAST) | **63.20** | 64.91 | 84.47 | 93.56 | 18.58 |
| ImageNet-F | Taxon-SSL | 48.40 | 52.34 | 65.74 | 82.96 | 19.87 |
| iNat21-mini-F | H-CAST | 25.63 | 28.61 | 67.20 | 83.62 | 47.17 |
| iNat21-mini-F | Taxon-SSL + Text-Attr | **31.93** | 37.08 | 69.76 | 82.20 | 37.04 |
| iNat21-mini-F | Taxon-SSL | 31.74 | 37.11 | 69.53 | 82.02 | 37.31 |

### Ablation Study

| Setting | Key Metric | Notes |
|---------|-----------|-------|
| H-CAST full → free (CUB) | FPA: 84.9% → 45.1% | Missing annotations cause a 39.8 pp drop |
| H-CAST full → free (iNat) | FPA: 64.9% → 25.6% | Missing annotations cause a 39.3 pp drop |
| Text-Attr under sparse labels | Outperforms Taxon-SSL | Text bridges supervision gap when labels are scarce |
| Taxon-SSL under sufficient labels | Outperforms Text-Attr | SSL is more effective when data is abundant |

### Key Findings

1. **Severe degradation of existing methods under the free-grained setting**: H-CAST's FPA drops by 19–40 pp, establishing the necessity of dedicated research for this setting.

2. **Complementary strengths of Text-Attr and Taxon-SSL**: Text-Attr performs better on large-scale diverse datasets (ImageNet-F), where textual descriptions provide rich semantic cues; Taxon-SSL is superior on fine-grained biological datasets (iNat21-mini-F), where inter-class visual similarity makes visual consistency more critical.

3. **Consistency-based stopping outperforms confidence-based stopping**: Halting prediction when hierarchical consistency breaks yields more reliable and deeper correct predictions than halting based on a softmax confidence threshold, and requires no threshold tuning.

4. **Text guidance improves semantic focus**: Saliency maps show that Text-Attr directs model attention toward semantically relevant regions (e.g., musical instruments rather than people), whereas Taxon-SSL can be misled by visually salient but semantically irrelevant regions.

## Highlights & Insights

- The paper formalizes an important new problem setting — free-grained hierarchical recognition — that is more realistic than the conventional assumption of complete hierarchical annotation.
- The construction of the ImageNet-3L benchmark is itself a significant contribution, providing a large-scale, clean evaluation platform for hierarchical classification.
- The complementarity of the two methods reflects a deep insight: external semantic knowledge (text) compensates for supervision when labels are scarce, while structured SSL exploits hierarchical consistency when labels are moderately available. This offers practical guidance for strategy selection.
- Consistency-based inference is an elegant, parameter-free stopping strategy.

## Limitations & Future Work

- Class-level and level-wise imbalance are not explicitly addressed.
- Label pruning relies on CLIP, which may introduce bias; improved pruning strategies (e.g., multi-model ensemble) remain to be explored.
- The gains of both proposed methods are modest (5–25%), indicating substantial room for improvement in free-grained learning.
- The methods are not extended to deeper taxonomies (beyond three levels).
- Inference considers only when to "stop predicting," without accounting for cross-level information propagation or error correction.

## Related Work & Insights

- H-CAST (CVPR'23) is the hierarchical classification SOTA, encouraging consistent visual grouping across levels.
- HRN (CVPR'22) handles multi-level supervision by maximizing marginal probabilities within a tree-constrained space.
- CHMatch leverages coarse labels to improve pseudo-labels, but is limited to a two-level setting.
- This paper unifies long-tail recognition, semi-supervised learning, weakly supervised learning, and hierarchical consistency within a single framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Defines an important new problem setting; contributions in both dataset construction and methodology)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, diverse settings, thorough analysis and visualization)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear problem formulation, excellent figures, well-organized structure)
- Value: ⭐⭐⭐⭐⭐ (Opens a new research direction; provides benchmarks and baselines with strong practical relevance)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ACE-Merging: Data-Free Model Merging with Adaptive Covariance Estimation](ace-merging_data-free_model_merging_with_adaptive_covariance_estimation.md)
- [\[CVPR 2026\] Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning](flow3r_factored_flow_prediction_for_scalable_visual_geometry_learning.md)
- [\[ICLR 2026\] Enabling Fine-Grained Operating Points for Black-Box LLMs](../../ICLR2026/llm_evaluation/enabling_fine-grained_operating_points_for_black-box_llms.md)
- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[CVPR 2026\] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval](sattc_structure-aware_label-free_test-time_calibration_for_cross-subject_eeg-to-.md)

<!-- RELATED:END -->
