---
title: >-
  [Paper Note] Tracking and Segmenting Anything in Any Modality
description: >-
  [AAAI 2026][Segmentation][Unified tracking and segmentation] SATA proposes a unified tracking and segmentation framework that models cross-modal shared and modality-specific knowledge via a Decoupled Mixture-of-Experts (…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Unified tracking and segmentation"
  - "any modality"
  - "mixture of experts"
  - "multi-task learning"
  - "generalist model"
date: 2026-05-08
content_hash: c6182d99a1db172e
---

# Tracking and Segmenting Anything in Any Modality

**Conference**: AAAI 2026
**arXiv**: [2511.19475](https://arxiv.org/abs/2511.19475)
**Code**: Available
**Area**: Segmentation / Video Understanding
**Keywords**: Unified tracking and segmentation, any modality, mixture of experts, multi-task learning, generalist model

## TL;DR
SATA proposes a unified tracking and segmentation framework that models cross-modal shared and modality-specific knowledge via a Decoupled Mixture-of-Experts (DeMoE) mechanism, and introduces a Task-aware Multi-Object Tracking (TaMOT) pipeline to unify the output format across all tasks, demonstrating superior performance across 18 tracking and segmentation benchmarks.

## Background & Motivation

**Background**: Tracking and segmentation are fundamental tasks in video understanding. Existing methods typically rely on task-specific architectures or modality-specific parameters to handle different sub-tasks (e.g., VOT, VOS, MOT, VIS), which limits generalization and scalability.

**Limitations of Prior Work**: (1) A distribution gap exists across modalities (RGB, infrared, depth, etc.), making direct parameter sharing ineffective; (2) A representation gap exists across tasks (tracking vs. segmentation, single-object vs. multi-object), hindering cross-task knowledge sharing; (3) Prior attempts at unifying these tasks overlook both gaps.

**Key Challenge**: Building a truly generalist model requires simultaneously addressing cross-modal distribution discrepancy and cross-task representation discrepancy.

**Goal**: To construct a unified framework capable of handling a broad range of tracking and segmentation sub-tasks with arbitrary modality inputs.

**Key Insight**: (1) Apply decoupled MoE to separate cross-modal shared and modality-specific knowledge; (2) Eliminate inter-task output discrepancies via a unified instance-set output format.

**Core Idea**: DeMoE decouples unified representation learning into cross-modal shared knowledge and modality-specific information modeling; TaMOT unifies all task outputs into instance sets with calibrated ID assignments.

## Method

### Overall Architecture
Given a video sequence in any modality, features are extracted via a backbone network. DeMoE adaptively assigns shared and modality-specific experts for feature enhancement, after which a unified decoder generates tracking/segmentation results. The TaMOT pipeline unifies the output format of all sub-tasks into instance sets with ID information.

### Key Designs

1. **Decoupled Mixture-of-Experts (DeMoE)**:

    - **Function**: Handles cross-modal distribution discrepancy within a unified framework.
    - **Mechanism**: Decouples the standard MoE into two groups of experts: shared experts that learn modality-invariant knowledge (e.g., motion patterns, object shape), and modality-specific experts that capture unique characteristics of each modality. A router dynamically assigns expert weights based on the input modality, enabling the model to remain flexible while enhancing generalization.
    - **Design Motivation**: Sharing all parameters directly causes conflicts due to modality gaps; fully independent parameters fail to exploit cross-modal commonalities. DeMoE strikes a balance between the two.

2. **Task-aware Multi-Object Tracking (TaMOT) Pipeline**:

    - **Function**: Unifies the output format of all tracking/segmentation sub-tasks.
    - **Mechanism**: Defines all task outputs as a unified instance set, where each instance contains spatial location (bbox/mask) and temporal ID information. Task tokens distinguish the inference mode for different sub-tasks. Training employs a unified instance-matching and ID-association loss.
    - **Design Motivation**: Output format discrepancies across sub-tasks (e.g., single-object tasks output only masks, while multi-object tasks output bbox + ID) impede unified multi-task training. TaMOT resolves this through format unification.

3. **Multi-modal Multi-task Joint Training**:

    - **Function**: Simultaneously learns multiple modalities and tasks within a single model.
    - **Mechanism**: Adopts a mixed dataset training strategy where each batch contains data from different modalities and tasks. The DeMoE router automatically allocates experts based on the input, while TaMOT task tokens guide decoding. The loss function is a weighted sum of individual task losses.
    - **Design Motivation**: Joint training enables cross-task knowledge transfer, while DeMoE ensures that different modalities and tasks do not interfere with one another.

### Loss & Training
Joint training employs a segmentation loss (Dice + BCE), a detection loss (L1 + GIoU), and an ID association loss. Task tokens are used to differentiate sub-tasks.

## Key Experimental Results

### Main Results

| Task | # Benchmarks | SATA Ranking | Notes |
|------|-------------|--------------|-------|
| Single Object Tracking | Multiple | Top-tier | RGB + infrared + depth |
| Multi-Object Tracking | Multiple | Top-tier | Unified ID management |
| Video Object Segmentation | Multiple | Top-tier | Semi-supervised / unsupervised |
| Video Instance Segmentation | Multiple | Top-tier | Detection + segmentation + tracking |
| Total | 18 | Comprehensively leading | Generalist model advantage |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Full SATA | Best | DeMoE + TaMOT in synergy |
| Standard MoE replacing DeMoE | Degraded | Shared/specific knowledge not decoupled |
| Modality-independent training | Degraded | No cross-modal knowledge transfer |
| Without TaMOT | Per-task degradation | Inconsistent output formats hinder knowledge sharing |

### Key Findings
- Top-tier or highly competitive results are achieved across all 18 benchmarks, validating the feasibility of generalist models.
- DeMoE's decoupled design yields a clear advantage over standard MoE, demonstrating the importance of separating shared and modality-specific knowledge.
- TaMOT's unified output format effectively mitigates task-specific knowledge degradation during multi-task training.

## Highlights & Insights
- **A truly generalist tracking and segmentation model**: A single model covers 18 benchmarks, multiple modalities, and diverse tasks, demonstrating the viability of a foundation model for video understanding.
- **DeMoE's decoupling principle**: The explicit separation of shared and modality-specific knowledge is a design principle transferable to other multi-modal and multi-task learning scenarios.
- **TaMOT's format unification**: Reformulating heterogeneous task outputs as a unified instance set is an elegant and practical engineering contribution.

## Limitations & Future Work
- Training data across 18 benchmarks is substantial, resulting in high training costs.
- The number of experts and routing strategy in DeMoE still require manual design.
- Efficiency for extremely long videos or real-time applications is not discussed.
- Model scale may limit deployment on embedded devices.

## Related Work & Insights
- **vs. SAM 2**: SAM 2 excels at video segmentation but does not manage tracking IDs; SATA unifies tracking and segmentation.
- **vs. UniTrack**: UniTrack attempts to unify tracking but excludes segmentation; SATA covers a broader scope.
- **vs. OneTracker**: OneTracker supports multi-modal tracking but excludes segmentation; SATA is more comprehensive.

## Rating
- Novelty: ⭐⭐⭐⭐ DeMoE's decoupled design is novel; the generalist framework is well-rounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 18 benchmarks is highly rigorous.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear.
- Value: ⭐⭐⭐⭐⭐ Makes an important contribution toward unified video understanding models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[CVPR 2026\] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video](../../CVPR2026/segmentation/robotseg_a_model_and_dataset_for_segmenting_robots_in_image_and_video.md)
- [\[CVPR 2026\] SAP: Segment Any 4K Panorama](../../CVPR2026/segmentation/sap_segment_any_4k_panorama.md)

</div>

<!-- RELATED:END -->
