---
title: >-
  [Paper Note] E-SAM: Training-Free Segment Every Entity Model
description: >-
  [ICCV 2025][Segmentation][Entity Segmentation] E-SAM is a training-free framework that systematically addresses over-segmentation and under-segmentation in SAM's Automatic Mask Generation (AMG) via three cascaded modules—Multi-level Mask Generation (MMG), Entity-level Mask Refinement (EMR), and Under-Segmentation Refinement (USR)—surpassing existing entity segmentation methods by **+30.1 points** on benchmark metrics.
tags:
  - ICCV 2025
  - Segmentation
  - Entity Segmentation
  - SAM
  - Training-Free
  - Automatic Mask Generation
  - Over-segmentation
  - Under-segmentation
  - NMS
date: 2026-05-08
content_hash: c2b92219600a986c
---

# E-SAM: Training-Free Segment Every Entity Model

**Conference**: ICCV 2025
**arXiv**: [2503.12094](https://arxiv.org/abs/2503.12094)
**Code**: Not released (mentioned in paper with link)
**Area**: Entity Segmentation / Foundation Models
**Keywords**: Entity Segmentation, SAM, Training-Free, Automatic Mask Generation, Over-segmentation, Under-segmentation, NMS

## TL;DR

E-SAM is a training-free framework that systematically addresses over-segmentation and under-segmentation in SAM's Automatic Mask Generation (AMG) via three cascaded modules—Multi-level Mask Generation (MMG), Entity-level Mask Refinement (EMR), and Under-Segmentation Refinement (USR)—surpassing existing entity segmentation methods by **+30.1 points** on benchmark metrics.

## Background & Motivation

### State of the Field

**Background**: Entity Segmentation (ES) aims to segment all visually distinguishable entities in an image without relying on predefined category labels. Unlike conventional semantic/instance/panoptic segmentation, ES is category-agnostic and more closely aligned with human visual perception. Representative methods such as EntitySeg and CropFormer depend on large amounts of annotated data, incur high training costs, and exhibit limited generalization capability.

**SAM's Potential and Limitations**:

### Limitations of Prior Work

**Potential**: SAM, trained on over one billion masks, possesses strong zero-shot segmentation capability. Its AMG mode generates full-image segmentation via uniform point sampling, making it theoretically suitable for ES tasks.

### Root Cause

**Key Challenge**: **Limitations**: AMG generates 3-level masks (object/part/subpart) for each sampled point, then applies simple NMS for deduplication. This strategy leads to severe **over-segmentation** (retaining too many overlapping masks) and **under-segmentation** (incorrectly removing critical masks, missing entities or fine details).

**Key Challenge**: SAM's AMG produces a large number of multi-granularity masks but lacks an effective post-processing strategy to organize them into accurate entity-level segmentation maps. Simple NMS cannot handle the complex overlap relationships among multi-granularity masks.

**Goal**: How to efficiently obtain high-quality entity-level segmentation from SAM's AMG output without any additional training.

## Method

### Overall Architecture

E-SAM freezes all SAM components (Image Encoder $E_{img}$, Prompt Encoder $E_{prompt}$, Mask Decoder $D_{mask}$) and optimizes AMG outputs through three cascaded modules at inference time only:

$\text{Input Image} \xrightarrow{AMG} \text{Multi-granularity Masks} \xrightarrow{MMG} \text{Hierarchical Masks} \xrightarrow{EMR} \text{Entity-level Masks} \xrightarrow{USR} \text{Final ES Map}$

### Key Designs

1. **Multi-level Mask Generation (MMG)**:

    - **Function**: Stratifies AMG output masks by area and confidence, applying differentiated NMS strategies.
    - **Mechanism**:
        - Uniformly generates 32 point prompts per edge (denser than SAM's default)
        - Categorizes SAM-returned masks into three levels—object, part, subpart—based on mask area
        - Applies strict NMS (high IoU threshold) to object-level masks to retain high-confidence large masks
        - Retains more candidate masks for part/subpart-level masks in dense regions
    - **Design Motivation**: A one-size-fits-all NMS threshold cannot handle masks of different granularities simultaneously. Large objects require aggressive deduplication, while small details require more candidates to be preserved.

2. **Entity-level Mask Refinement (EMR)**:

    - **Function**: Refines object-level masks into accurate entity-level masks, resolving overlaps and redundancy.
    - **Steps**:
        - (a) **Increase sampling density**: Constructs a mask gallery (high-confidence candidate mask pool) using denser uniform points.
        - (b) **Separate overlapping masks**: Identifies overlapping regions among object-level masks and uses masks from the gallery to separate them into independent adjacent masks.
        - (c) **Merge similar masks**: Constructs a pairwise mask similarity matrix and leverages the mask gallery to assess entity-level consistency, merging masks with high similarity.
    - **Design Motivation**: Object-level AMG masks may overlap (when multiple points cover the same region) or be over-fragmented. EMR systematically organizes them into non-overlapping entity masks through a "separate-then-merge" strategy.

3. **Under-Segmentation Refinement (USR)**:

    - **Function**: Repairs under-segmented regions in EMR output.
    - **Mechanism**:
        - Uses superpixel centroids as additional prompts
        - Employs centroids of part/subpart-level masks as supplementary prompts
        - Feeds these prompts into SAM to generate additional high-confidence masks
        - Merges results with EMR output to ensure previously uncovered entities are segmented
    - **Design Motivation**: EMR may still miss certain entities—those not covered by initial uniform sampling points or incorrectly removed during NMS. USR compensates by generating supplementary masks from multi-source prompts.

### Fully Training-Free

- All three modules are rule-based post-processing strategies.
- No SAM weights are modified.
- The only hyperparameters are NMS thresholds, area stratification thresholds, and similarity merge thresholds at each stage.

## Key Experimental Results

### Main Results

On the EntitySeg benchmark:

- E-SAM outperforms the previous state-of-the-art entity segmentation methods by **+30.1 points** on benchmark metrics.
- Under the same backbone size, E-SAM consistently achieves more than **2× performance** over SAM AMG.

### Comparison with Different Methods

| Baseline | Type | E-SAM Advantage |
|---------|------|----------------|
| SAM AMG | Foundation Model AMG | Resolves over/under-segmentation; 2× performance gain |
| Semantic-SAM | Enhanced SAM | No additional training required; better entity-level accuracy |
| EntitySeg/CropFormer | Dedicated ES Models | No training data needed; stronger zero-shot generalization |

### Generalization Experiments

- On unseen datasets (open-world scenarios), E-SAM demonstrates strong generalization (Figure 7), validating the advantage of the training-free approach—as no overfitting to any specific distribution occurs.

### Ablation Study

- **MMG contribution**: Hierarchical NMS vs. uniform NMS → hierarchical strategy significantly reduces over-segmentation.
- **EMR contribution**: Both the separation and merging steps are indispensable; removing either leads to performance degradation.
- **USR contribution**: USR primarily addresses large uncovered regions and is critical for complete entity discovery.
- **Joint effect of three modules**: Each module alone provides limited improvement; maximum effectiveness is achieved only through cascaded combination of all three.

### Key Findings

- Although AMG output quality varies, it contains sufficiently rich multi-granularity information; the challenge lies in how to organize and filter it.
- A training-free approach can substantially outperform dedicated trained methods on ES tasks, demonstrating that SAM's foundational capabilities can be fully unleashed under appropriate post-processing.
- Over-segmentation and under-segmentation require distinct strategies—handled separately by EMR (deduplication/merging) and USR (gap-filling).

## Highlights & Insights

- **Strong evidence that "training-free beats trained"**: E-SAM, being entirely training-free, surpasses methods such as CropFormer that require large amounts of annotation and high training costs. This demonstrates that large-scale pretrained foundation models like SAM already contain sufficient segmentation capability; the key lies in effectively invoking and organizing their outputs.
- **Systematic diagnosis and targeted resolution of AMG deficiencies**: MMG addresses over-segmentation, EMR refines entity-level masks, and USR compensates for under-segmentation—each module targets a specific problem with clear design rationale and self-consistent logic.
- **"Separate-then-merge" mask refinement strategy**: EMR first separates overlapping masks and then merges them based on entity consistency, which is more robust than direct merging or direct separation alone. This strategy echoes the "over-segment then merge" philosophy of watershed algorithms in classical image segmentation.
- **Complementarity of multi-source prompts**: USR simultaneously leverages superpixel centroids, part centroids, and subpart centroids as prompts; these three sources are spatially complementary across scales, maximizing coverage of missed regions.
- **Strong generalization**: The primary advantage of a training-free approach is freedom from training data distribution constraints; E-SAM's strong performance on unseen datasets validates this point.

## Limitations & Future Work

- **Low inference efficiency**: All three cascaded modules require multiple invocations of SAM's encoder and decoder (especially mask gallery construction in EMR and additional prompt generation in USR), resulting in significantly higher inference time than single-pass AMG or trained end-to-end methods.
- **Hyperparameter sensitivity**: Multiple thresholds (NMS thresholds, area stratification, similarity merge thresholds, etc.) require manual tuning and may need different settings across datasets.
- **Dependence on SAM version**: E-SAM is entirely reliant on SAM's segmentation quality; if the underlying SAM fails in certain scenarios (e.g., extreme occlusion, transparent objects), E-SAM cannot compensate.
- **No video/temporal extension**: The method is designed for single-frame images only and does not consider entity tracking or segmentation improvements from temporal consistency in video.
- **Upper bound constrained by SAM**: E-SAM can only select and combine masks already generated by SAM and cannot produce mask shapes that SAM has never output.

## Highlights & Insights

## Limitations & Future Work

## Related Work & Insights

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](../../AAAI2026/segmentation/saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
