---
title: >-
  [Paper Note] AnimalClue: Recognizing Animals by their Traces
description: >-
  [ICCV 2025][Segmentation][animal trace recognition] This paper introduces AnimalClue, the first large-scale dataset for animal trace recognition, containing 159,605 bounding boxes spanning 968 species across five categories of indirect clues (footprints, feces, eggs, bones, and feathers), and establishes four benchmarks covering classification, detection, instance segmentation, and attribute prediction.
tags:
  - ICCV 2025
  - Segmentation
  - animal trace recognition
  - wildlife conservation
  - indirect evidence
  - dataset
  - instance segmentation
date: 2026-05-08
content_hash: 36b35f5b981f56ac
---

# AnimalClue: Recognizing Animals by their Traces

**Conference**: ICCV 2025
**arXiv**: [2507.20240](https://arxiv.org/abs/2507.20240)
**Code**: [https://dahlian00.github.io/AnimalCluePage/](https://dahlian00.github.io/AnimalCluePage/)
**Area**: Segmentation / Object Detection / Image Classification
**Keywords**: animal trace recognition, wildlife conservation, indirect evidence, dataset, instance segmentation

## TL;DR

This paper introduces AnimalClue, the first large-scale dataset for animal trace recognition, containing 159,605 bounding boxes spanning 968 species across five categories of indirect clues (footprints, feces, eggs, bones, and feathers), and establishes four benchmarks covering classification, detection, instance segmentation, and attribute prediction.

## Background & Motivation

Wildlife monitoring is critical for biodiversity conservation. Computer vision has made significant advances in direct animal recognition (appearance-based detection), yet species identification from indirect evidence (e.g., footprints, feces) remains largely underexplored. Existing datasets suffer from severe limitations:
- OpenAnimalTracks contains only 18 species and 3,579 bounding boxes
- FeathersV1 supports classification tasks only
- Existing datasets cover few species and provide limited annotation types

Ecological surveys extensively rely on indirect evidence for species identification, yet this process is highly labor-intensive and urgently requires automated computer vision solutions. AnimalClue aims to fill this gap by providing a large-scale benchmark encompassing multiple trace types and tasks.

## Method

### Overall Architecture

AnimalClue is a dataset-and-benchmark contribution whose core value lies in data construction and experimental evaluation.

### Key Designs

1. **Data Collection**

    - Images are collected from the iNaturalist platform, selecting research-grade observations whose labels have been verified by multiple citizen scientists
    - Only Creative Commons licensed images are retained; blurry, unclear, and face-containing images are removed
    - Five categories of animal traces are covered: footprints (18,291 bboxes), feces (18,932 bboxes), bones (16,553 bboxes), eggs (29,434 bboxes), and feathers (76,395 bboxes)

2. **Annotation Strategy**

    - Footprints: bounding boxes only (since footprints are traces rather than physical entities, and boundaries are often ambiguous)
    - Feces, bones, eggs, feathers: pixel-level segmentation masks are provided
    - SAM is used to assist initial annotation for feces and eggs, with manual verification by the authors
    - Multiple images from the same iNaturalist observation are not split across train/test sets, preventing data leakage

3. **Fine-Grained Attribute Annotation**

    - A total of 22 ecological and behavioral attributes are annotated, including:
        - Taxonomic information (order, family)
        - Diet type (herbivore, carnivore, omnivore)
        - Activity pattern (diurnal, nocturnal, crepuscular)
        - Habitat preference (forest, grassland, desert, wetland, mountain, urban)
        - Climate distribution (tropical, subtropical, temperate, boreal, polar)
        - Social behavior (gregarious, migratory, predator)

4. **Frequency Partitioning**

    - Species are divided into three groups based on training-set frequency: frequent (top 20%), intermediate (middle 60%), and rare (bottom 20%)
    - Partitioning is performed independently for each of the five trace types

### Dataset Statistics

| Trace Type | BBoxes | Images | Species | Families | Orders |
|------------|--------|--------|---------|----------|--------|
| Footprints | 18,291 | 7,581 | 117 | 46 | 20 |
| Feces | 18,932 | 6,433 | 101 | 46 | 21 |
| Bones | 16,553 | 12,908 | 269 | 112 | 45 |
| Eggs | 29,434 | 9,394 | 283 | 67 | 20 |
| Feathers | 76,395 | 60,491 | 555 | 89 | 30 |

## Key Experimental Results

### Main Results — Classification

| Model | Footprints (Species) | Feces (Species) | Eggs (Species) | Bones (Species) | Feathers (Species) |
|-------|----------------------|-----------------|----------------|-----------------|-------------------|
| VGG-16 | 28.8 | 29.6 | 45.2 | 14.7 | 56.7 |
| ResNet-50 | 23.7 | 29.4 | 41.1 | 18.3 | 59.7 |
| ViT-B | 29.2 | 32.2 | 46.7 | 15.0 | 55.9 |
| Swin-B | **32.3** | **38.6** | **49.4** | **20.5** | **65.3** |

### Ablation Study / Detection Results

| Detection Model | Footprints (Species mAP) | Eggs (Species mAP) | Feathers (Species mAP) |
|-----------------|--------------------------|--------------------|-----------------------|
| YOLOv8 | 0.10 | 0.13 | 0.25 |
| YOLOv11 | 0.10 | 0.14 | 0.25 |
| RT-DETR | 0.10 | 0.04 | 0.17 |
| DINO | 0.08 | 0.20 | 0.15 |

| Segmentation Model | Feces (Species) | Eggs (Species) | Bones (Species) | Feathers (Species) |
|--------------------|-----------------|----------------|-----------------|-------------------|
| YOLOv8 | 0.11 | 0.11 | 0.07 | 0.24 |
| MaskDINO | **0.13** | **0.25** | **0.07** | 0.18 |
| YOLOv11 | 0.11 | 0.12 | 0.06 | **0.24** |

### Key Findings

- Swin-B consistently achieves the best performance across all classification tasks, indicating that Transformer architectures are better suited to capturing fine-grained trace features
- Feather recognition yields the highest accuracy (65.3%) despite covering the most species (555), owing to distinctive color and texture patterns
- Bone recognition is the most challenging (20.5%), as appearance varies substantially across body parts
- Rare-species recognition is extremely difficult: Swin-B achieves only 14.2% at the species level for rare footprints and 2.52% for rare feathers
- Detection and segmentation mAP values are consistently low (the highest order-level detection mAP is 0.57), indicating the task is far from solved
- CLIP fine-tuned on AnimalClue exhibits the best feature separation in t-SNE visualizations

## Highlights & Insights

- **Novel problem formulation**: Identifying animal species from indirect evidence is complementary to conventional appearance-based recognition and holds significant ecological application value
- **Scale and comprehensiveness**: Covering 968 species, 5 trace types, 4 tasks, and 22 attribute annotations, AnimalClue substantially surpasses all existing datasets
- **Revealing key challenges**: Difficulty in generalizing to rare species and extremely low species-level detection/segmentation mAP demonstrate that substantial research opportunities remain in this field

## Limitations & Future Work

- Species distribution is highly imbalanced, with a severe long-tail problem
- Footprints are annotated with bounding boxes only, lacking segmentation masks
- Only standard baseline models are evaluated; strategies such as pre-training or domain adaptation are not explored
- Joint recognition across trace types (e.g., simultaneously leveraging footprints and feces to identify the same species) is not investigated
- Data sourced primarily from iNaturalist may introduce geographic and species bias

## Related Work & Insights

- AnimalClue is complementary to conventional animal appearance recognition datasets (iNat, CUB-200)
- The 22 attribute annotations provide rich auxiliary signals for multi-task learning and zero-shot learning
- The approach could inspire extensions to other indirect evidence recognition scenarios, such as crime scene analysis and archaeology

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale indirect animal trace dataset with a novel problem formulation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four task benchmarks, comprehensive multi-model evaluation, and thorough frequency analysis
- Writing Quality: ⭐⭐⭐⭐ Dataset construction is clearly described with complete statistics
- Value: ⭐⭐⭐⭐ Opens a new direction for computer vision research in wildlife monitoring with lasting dataset impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_open-vocabulary_remote_sensing_instance_segmentat.md)
- [\[ICCV 2025\] ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones](roadwork_a_dataset_and_benchmark_for_learning_to_recognize_observe_analyze_and_d.md)
- [\[ICCV 2025\] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model](himtok_learning_hierarchical_mask_tokens_for_image_segmentation_with_large_multi.md)
- [\[ICCV 2025\] PartField: Learning 3D Feature Fields for Part Segmentation and Beyond](partfield_learning_3d_feature_fields_for_part_segmentation_and_beyond.md)
- [\[ICCV 2025\] Region-based Cluster Discrimination for Visual Representation Learning](region-based_cluster_discrimination_for_visual_representation_learning.md)

</div>

<!-- RELATED:END -->
