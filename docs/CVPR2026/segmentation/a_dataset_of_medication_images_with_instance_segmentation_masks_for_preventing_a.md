---
title: >-
  [Paper Note] MEDISEG: A Dataset of Medication Images with Instance Segmentation Masks for Preventing Adverse Drug Events
description: >-
  [CVPR 2026][Segmentation][medication recognition] This paper introduces MEDISEG — a dataset of 8,262 real-world multi-pill scene images covering 32 pill types (including overlapping, occluded, and varying-illumination scenarios within dosette boxes), with instance segmentation annotations. YOLOv8/v9 achieve mAP@50 of 99.5% on the 3-Pills subset and 80.1% on the 32-Pills subset. Few-shot experiments demonstrate that MEDISEG as a base training set significantly outperforms the CURE dataset.
tags:
  - CVPR 2026
  - Segmentation
  - medication recognition
  - instance segmentation
  - drug safety
  - dosette box
  - few-shot detection
  - YOLOv8
  - YOLOv9
date: 2026-05-08
content_hash: eb83064d8767f4dd
---

# MEDISEG: A Dataset of Medication Images with Instance Segmentation Masks for Preventing Adverse Drug Events

**Conference**: CVPR 2026
**arXiv**: [2603.10825](https://arxiv.org/abs/2603.10825)
**Code**: None (Dataset publicly available, CC BY 4.0)
**Area**: Instance Segmentation / Medical Imaging / Dataset
**Keywords**: medication recognition, instance segmentation, drug safety, dosette box, few-shot detection, YOLOv8, YOLOv9

## TL;DR
This paper introduces MEDISEG — a dataset of 8,262 real-world multi-pill scene images covering 32 pill types (including overlapping, occluded, and varying-illumination scenarios within dosette boxes), with instance segmentation annotations. YOLOv8/v9 achieve mAP@50 of 99.5% on the 3-Pills subset and 80.1% on the 32-Pills subset. Few-shot experiments demonstrate that MEDISEG as a base training set significantly outperforms the CURE dataset.

## Background & Motivation
Medication errors and Adverse Drug Events (ADEs) pose a major threat to patient safety worldwide. Statistics indicate approximately 237 million medication errors per year in the UK, and ADEs cause around 100,000 deaths annually in the United States. Administration errors account for a substantial proportion of these incidents, particularly when multiple medications are co-stored in dosette boxes or pill organisers, where pills of similar shape and color are easily confused.

Computer vision-assisted medication recognition is a promising solution; however, existing datasets suffer from critical limitations:
- **National Library of Medicine (NLM) Pill Image Dataset**: Single-pill images captured in controlled environments with no overlap or occlusion
- **CURE Dataset**: Contains 19,000+ images but only bounding-box detection annotations, lacking instance segmentation
- **RxImage/C3PI**: Similarly controlled environments that do not reflect real-world usage scenarios

There is a clear absence of instance-level segmentation annotated datasets covering realistic multi-pill scenes with stacking, occlusion, and illumination variation.

## Core Problem
How to construct a medication image dataset that faithfully reflects real-world dosette box usage scenarios, provides instance segmentation annotations, and supports automated medication identification under multi-pill overlap and occlusion conditions.

## Method

### Data Collection
- **Device**: iPhone 12 Pro Max (primary camera)
- **Shooting Scenario**: Multiple pill types mixed within a dosette box, simulating real-world medication management
- **Illumination Conditions**: Diverse conditions under both natural and artificial lighting
- **Pill Types**: 32 common prescription and OTC pill types, covering varying shapes (round, oval, capsule), colors, and sizes

### Dataset Structure
Two subsets targeting different levels of recognition difficulty:

1. **3-Pills Subset**:
    - 3 pill types with highly similar visual appearance
    - Focuses on fine-grained discrimination capability
    - Each image contains a small number of pills, testing model precision in distinguishing similar medications

2. **32-Pills Subset**:
    - All 32 pill types
    - Up to 13 pills per frame
    - Extensive overlap, occlusion, and partial visibility
    - 8,262 images in total

### Annotation Pipeline
- **Annotation Tool**: COCO Annotator
- **Annotation Format**: COCO instance segmentation format (polygon masks)
- **Annotation Content**: Precise contour boundaries for each pill, supporting instance-level segmentation
- **Quality Control**: Multiple rounds of verification to ensure annotation consistency

### Baseline Model Evaluation

#### Object Detection / Instance Segmentation
- **YOLOv8**: Trained and evaluated on both 3-Pills and 32-Pills subsets
- **YOLOv9**: Evaluated on both subsets
- **Training Strategy**: Fine-tuning from standard COCO pre-trained weights

#### Few-Shot Detection Experiments
- **Framework**: FsDet (Few-shot Detection)
- **Design Rationale**: Using MEDISEG as the base-class training set to evaluate recognition capability on unseen pill types
- **Comparison**: CURE dataset as base training vs. MEDISEG as base training
- **Hyperparameter Optimization**: Genetic algorithm (GA) search with 70 iterations

## Key Experimental Results

### Object Detection Performance (mAP@50)

| Model | 3-Pills | 32-Pills |
|-------|---------|----------|
| YOLOv8 | **99.5%** | **80.1%** |
| YOLOv9 | ~99% | ~78% |

### Few-Shot Experiments

| Base Training Set | Novel Class mAP@50 |
|-------------------|--------------------|
| CURE | Lower (baseline) |
| MEDISEG | **Significantly higher** |

When used as the base training set for few-shot transfer to unseen pill types, MEDISEG substantially outperforms CURE, indicating that the instance segmentation annotations of real-world multi-pill scenes provide the model with richer spatial, occlusion, and contextual information.

### Dataset Statistics

| Metric | 3-Pills | 32-Pills |
|--------|---------|----------|
| Number of pill types | 3 | 32 |
| Max pills per frame | ~3 | 13 |
| Annotation type | Instance segmentation | Instance segmentation |

## Highlights & Insights
- **Pioneering Dataset**: The first instance segmentation dataset targeting real-world multi-pill dosette box scenarios, bridging the gap between controlled environments and real-world deployment
- **High Practical Value**: Directly addresses safety pain points in medication administration (ADE prevention), with clear clinical application prospects
- **Few-Shot Transfer Validation**: Beyond being a dataset contribution, the paper validates MEDISEG's superiority as a base-class training set for few-shot learning
- **GA Hyperparameter Search**: Employs a genetic algorithm with 70 iterations to optimize few-shot detector hyperparameters, demonstrating methodological rigor
- **CC BY 4.0 License**: Open access, facilitating community reproduction and extension

## Limitations & Future Work
- **Limited Pill Variety**: 32 pill types cover only a small fraction of commonly used medications; real pharmacy settings may involve hundreds of types
- **Single Acquisition Device**: Images are captured exclusively with an iPhone 12 Pro Max, lacking diversity across different phones and cameras
- **32-Pills mAP of Only 80.1%**: An approximately 20% miss/misclassification rate may be clinically unacceptable in real deployment settings
- **No Cross-Dataset Generalization Evaluation**: Transfer performance on other medication datasets (beyond few-shot experiments) is not assessed
- **No Worn or Damaged Pills**: Real-world pills may exhibit abrasion, fragmentation, or fading
- **Annotation Consistency**: The paper does not report inter-annotator agreement metrics in detail

## Related Work & Insights
- **NLM Pill Image Dataset**: Single pill / controlled background / classification labels only → MEDISEG offers multi-pill / real-world scenes / instance segmentation
- **CURE Dataset**: 19,000+ images but bounding-box only → MEDISEG provides polygon masks enabling precise contour segmentation
- **RxImage/C3PI**: Reference pharmaceutical images, unsuitable for training detection models → MEDISEG is designed specifically for detection and segmentation
- **EPill-Seg (MICCAI)**: Focuses on endoscopic pill segmentation in a completely different scenario → MEDISEG targets everyday medication management settings

The dataset construction methodology is transferable to other medical safety scenarios (e.g., surgical instrument counting, infusion bottle recognition). The few-shot experiments demonstrate the transfer value of real-world scene data, offering insight into the tradeoff between investing in high-quality annotations versus large-scale weak annotations. The fine-grained discrimination of 32 pill classes shares challenges with general fine-grained visual recognition (birds, car models), from which complementary methods may be borrowed. Future work may integrate foundation models such as SAM for zero-shot or few-shot medication segmentation.

## Rating
- Novelty: ⭐⭐⭐ — The core contribution is a dataset rather than a methodological innovation, but it fills an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ — YOLOv8/v9 baselines + few-shot experiments + GA hyperparameter search; however, comparisons with more SOTA methods are lacking
- Writing Quality: ⭐⭐⭐⭐ — Clear structure for a dataset paper with well-motivated problem formulation
- Value: ⭐⭐⭐⭐ — Direct practical value for medication safety; CC BY 4.0 open license benefits the research community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MEDISEG: A Medication Image Instance Segmentation Dataset for Preventing Adverse Drug Events](a_dataset_of_medication_images_with_instance_segme.md)
- [\[CVPR 2026\] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment](lod-loc_v3_generalized_aerial_localization_in_dense_cities_using_instance_silhou.md)
- [\[CVPR 2026\] UnrealPose: Leveraging Game Engine Kinematics for Large-Scale Synthetic Human Pose Data](unrealpose_leveraging_game_engine_kinematics_for_large-scale_synthetic_human_pos.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[CVPR 2026\] Phrase-Instance Alignment for Generalized Referring Segmentation](phrase-instance_alignment_for_generalized_referring_segmentation.md)

</div>

<!-- RELATED:END -->
