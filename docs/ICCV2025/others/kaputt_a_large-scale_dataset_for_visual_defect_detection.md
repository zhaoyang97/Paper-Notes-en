---
title: >-
  [Paper Note] Kaputt: A Large-Scale Dataset for Visual Defect Detection
description: >-
  [ICCV 2025][defect detection] Kaputt introduces a large-scale retail logistics defect detection dataset comprising 230,000+ images and 48…
tags:
  - "ICCV 2025"
  - "defect detection"
  - "anomaly detection"
  - "large-scale dataset"
  - "retail logistics"
  - "benchmark"
date: 2026-05-08
content_hash: 2ad635cb25964714
---

# Kaputt: A Large-Scale Dataset for Visual Defect Detection

**Conference**: ICCV 2025
**arXiv**: [2510.05903](https://arxiv.org/abs/2510.05903)
**Code**: [Dataset](https://www.kaputt-dataset.com)
**Area**: Other
**Keywords**: defect detection, anomaly detection, large-scale dataset, retail logistics, benchmark

## TL;DR

Kaputt introduces a large-scale retail logistics defect detection dataset comprising 230,000+ images and 48,000+ unique items — 40× the scale of MVTec-AD — and is the first to incorporate significant pose and appearance variation. State-of-the-art anomaly detection methods achieve no more than 56.96% AUROC on this benchmark, exposing critical shortcomings of existing approaches in real-world retail scenarios.

## Background & Motivation

Automated visual defect detection is a critical component of quality assurance. Existing anomaly detection benchmarks (MVTec-AD, VisA) primarily target **manufacturing scenarios**, characterized by highly controlled object poses and limited categories (15 and 12, respectively). State-of-the-art methods have reached 99.9% AUROC on these datasets, approaching saturation.

**Retail logistics scenarios**, however, present fundamentally different challenges:

**Extreme item diversity**: physical properties vary widely, from food products to electronics

**Highly variable defect types**: ranging from subtle wrinkles to severe damage, many of which are difficult even for human inspectors

**Severe sample scarcity**: most items are observed only a few times, with limited normal and defective samples alike

**Significant pose variation**: items are placed arbitrarily in logistics containers, making pose uncontrollable

Existing datasets fail to capture these challenges. MVTec-AD contains only 5,354 images (1,258 defective), and VisA only 10,821. Leading anomaly detection methods suffer dramatic performance drops when transferred to logistics settings.

**Core Problem**: How can generalizable defect detection methods be developed when per-item samples are scarce, both normal and defective examples are limited, and intra-class variation is substantial?

## Method

### Overall Architecture

The primary contribution of this paper is the dataset and its accompanying comprehensive evaluation benchmark, rather than a novel method. The dataset design reflects careful engineering considerations.

### Key Designs

1. **Dataset Structure**:

    - **Query set**: 100,267 annotated images containing 29,316 defect instances
    - **Reference set**: 1–3 unannotated "normal" reference images per item (138,154 images total)
    - **Item count**: 48,376 unique items, with train/val/test splits strictly partitioned by item ID to prevent leakage
    - **Resolution**: 12MP RGB camera, cropped to 2048×2048 pixels
    - Train/val/test split: 85% / 5% / 10%

2. **Multi-level Annotation Scheme**:

    - **Defect severity**: no defect / minor / severe, determined by majority vote among three independent annotators
    - **Defect type** (7 categories, multi-label): penetration (holes/tears), deformation (dents/crushes), opened (open box/bag), deconstruction, spillage, surface (dirt/scratches), missing unit
    - **Item material**: cardboard, plastic bag, hard plastic, bubble wrap, paper, books, etc.
    - Deformation is the most common defect type but tends to be minor; spillage and deconstruction are typically severe

3. **Data Collection Methodology**:

    - **Hardware**: 12MP RGB camera with f/12mm lens, top-down capture, uniform LED panel illumination to reduce plastic reflections
    - **Defect sample collection**: two-stage strategy — (1) manually flagged defective items; (2) iterative mining using trained classifiers to surface candidates for manual annotation
    - **Quality control**: filtering low-quality images, capping at 15 images per item, balancing defect rate to 28.6%, excluding items with no normal samples

### Loss & Training

Rather than proposing a new method, this paper systematically evaluates four categories of baselines:
- **Training-free, reference-free** (zero-shot): CLIP, Claude 3.5, Pixtral-12B
- **Training-free, reference-based** (few-shot anomaly detection): PatchCore, WinCLIP
- **Training-based, reference-free** (supervised): ResNet50, ViT-S/DINOv2, AutoGluon
- **Training-based, reference-based** (hybrid): PatchCore with fine-tuned backbone, AutoGluon + reference

## Key Experimental Results

### Main Results

| Method | Type | APany (%) ↑ | APmajor (%) ↑ | AUROC ↑ |
|------|------|-----------|-------------|---------|
| Random | - | 31.84 | 14.00 | 50.00 |
| CLIP | Zero-shot | 36.20 | 17.15 | 56.05 |
| Claude-icl | Zero-shot + context | 36.57 | 24.76 | 56.96 |
| PatchCore50 | Few-shot AD | 35.86 | 17.80 | 54.69 |
| WinCLIP-few | Few-shot AD | 34.05 | 19.29 | 52.41 |
| ResNet50 | Supervised | 81.06 | 74.93 | 88.36 |
| **ViT-S** | **Supervised** | **90.67** | **91.45** | **94.27** |
| PatchCore50-ft | Hybrid | 40.18 | 20.98 | 60.14 |

### Ablation Study

**Performance degradation when reducing defective training samples**:

| Configuration | APany (%) | APmajor (%) | AUROC |
|------|----------|------------|-------|
| ViT-S full training set | 90.67 | 91.45 | 94.27 |
| ViT-S 1% defect rate (Query only) | 57.7 | 40.5 | 74.4 |
| ViT-S 1% defect rate (Query + ref) | 40.4 | 14.9 | 63.2 |

**Key comparison: anomaly detection methods across datasets**:

| Dataset | AUROC |
|--------|-------|
| MVTec-AD (SOTA) | 99.9% |
| VisA (SOTA) | 99.5% |
| **Kaputt (best unsupervised)** | **56.96%** |

### Key Findings

1. **Anomaly detection methods fail comprehensively**: All unsupervised/few-shot methods achieve no more than 56.96% AUROC, barely above random chance.
2. **VLMs are insufficient**: Claude/Pixtral can describe objects but fail to detect subtle defects, consistent with findings by Jiang et al.
3. **Reference images are counterproductive**: Naively incorporating reference images (e.g., feature averaging) degrades supervised method performance (96%→87% APany on the training set).
4. **Ceiling of supervised methods**: ViT-S achieves 90.67% APany, yet still makes errors on deformable items and "adversarial" packaging designs (e.g., packaging printed with hole-like patterns).
5. **Pose variation is the core challenge**: Anomaly detection methods misidentify normal pose and appearance variation as anomalies.

## Highlights & Insights

- **Genuinely exposes the bottleneck of anomaly detection**: the issue is not method inadequacy but a fundamental shift in problem nature — from controlled manufacturing to open retail environments.
- **Rigorous dataset design**: item-ID-based splits prevent data leakage; three-annotator majority voting ensures label quality; defect rates are aligned with existing benchmarks.
- **Four-scenario evaluation framework**: the 2×2 matrix of training vs. no training × reference vs. no reference provides a comprehensive perspective.
- **Scale advantage**: 48K unique items and 29K defect instances constitute the largest benchmark of its kind.

## Limitations & Future Work

- Only a single top-down viewpoint is captured; multi-view information is not exploited.
- Reference image quality is not guaranteed (<1% of reference images contain defects themselves), potentially introducing noise.
- Annotation errors remain (e.g., non-observable defects due to occlusion; confusion between design patterns and actual defects).
- No pixel-level segmentation annotations are provided, precluding evaluation of defect localization accuracy.
- All experiments use RGB images; depth, infrared, and other modalities are not explored.

## Related Work & Insights

- MVTec-AD and VisA have saturated; Kaputt represents the next frontier for anomaly detection research.
- ARMBench targets a similar scenario but contains only one-quarter as many defective samples as Kaputt and covers only 2 defect types.
- Adapting anomaly detection methods to **large intra-class variation** remains a key open problem.
- Effective utilization of reference images is an underexplored research direction — naive feature averaging is clearly insufficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Dataset-driven contribution with precise problem formulation, but no methodological innovation
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four scenarios × multiple methods + training set reduction experiments + detailed error analysis
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-documented dataset descriptions
- **Value**: ⭐⭐⭐⭐⭐ Fills the benchmark gap in retail logistics defect detection and will drive community progress

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2026/others/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[ICCV 2025\] A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition](a_hyperdimensional_one_place_signature_to_represent_them_all_stackable_descripto.md)
- [\[CVPR 2026\] UniSpector: Towards Universal Open-set Defect Recognition via Spectral-Contrastive Visual Prompting](../../CVPR2026/others/unispector_towards_universal_open-set_defect_recognition_via_spectral-contrastiv.md)

</div>

<!-- RELATED:END -->
