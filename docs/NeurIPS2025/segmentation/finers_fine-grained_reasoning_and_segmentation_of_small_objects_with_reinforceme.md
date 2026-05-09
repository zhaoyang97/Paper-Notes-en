---
title: >-
  [Paper Note] FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning
description: >-
  [NeurIPS 2025][Segmentation][small object segmentation] FineRS is a two-stage MLLM reinforcement learning framework comprising Global Semantic Exploration (GSE) and Localized Perceptual Refinement (LPR), coupled via a locate-informed retrospective reward. Evaluated on the newly constructed FineRS-4k UAV high-resolution dataset, it achieves reasoning and segmentation of ultra-small objects with a gIoU of 55.1% (surpassing Seg-Zero† by 8.5%) while simultaneously supporting VQA (MVQA 83.3%).
tags:
  - NeurIPS 2025
  - Segmentation
  - small object segmentation
  - MLLM
  - reinforcement-learning
  - GRPO
  - coarse-to-fine
  - high-resolution
  - UAV
date: 2026-05-08
content_hash: 63cef2c5b5231e6f
---

# FineRS: Fine-grained Reasoning and Segmentation of Small Objects with Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.21311](https://arxiv.org/abs/2510.21311)
**Code**: [https://iiau-zhanglu.github.io/FINERS/](https://iiau-zhanglu.github.io/FINERS/)
**Area**: Image Segmentation
**Keywords**: small object segmentation, MLLM, reinforcement-learning, GRPO, coarse-to-fine, high-resolution, UAV

## TL;DR

FineRS is a two-stage MLLM reinforcement learning framework comprising Global Semantic Exploration (GSE) and Localized Perceptual Refinement (LPR), coupled via a locate-informed retrospective reward. Evaluated on the newly constructed FineRS-4k UAV high-resolution dataset, it achieves reasoning and segmentation of ultra-small objects with a gIoU of 55.1% (surpassing Seg-Zero† by 8.5%) while simultaneously supporting VQA (MVQA 83.3%).

## Background & Motivation

**Background**: Methods such as LISA integrate MLLMs with SAM to enable reasoning segmentation; however, they are designed for standard-resolution images and large-scale objects, failing severely on ultra-small objects (area ratio <0.1%) in high-resolution images — LISA 7B achieves only 9.0% gIoU on FineRS-4k.

**Limitations of Prior Work**: Existing high-resolution understanding methods (SEAL, DC2, MLLMs-Know) capture fine-grained details via tiling or attention mechanisms, but most operate in a training-free manner, lack precise localization capability, and cannot produce pixel-level masks.

**Limitations of Vision RFT**: Seg-Zero introduces GRPO into segmentation but is constrained by resolution, does not support multi-task unification (joint VQA and segmentation), and employs insufficiently flexible reward designs.

**Key Challenge**: A unified framework is needed to handle ultra-small objects in 4K images, support instruction-guided segmentation, open-ended VQA, and multiple-choice VQA simultaneously, and achieve data-efficient training through reinforcement learning.

## Method

### Overall Architecture: Two-Stage Coarse-to-Fine

FineRS is built upon Qwen2.5-VL-7B and adopts a two-stage pipeline:

**Stage 1: Global Semantic Exploration (GSE)**

- Input: 1920×1080 high-resolution image + user instruction
- Output: text response $A^{pre}$ + coarse localization region $B_r^{pre}$ (fixed 256×256 size, optimizing only the center offset)
- Function: comprehends instruction semantics in a global view and predicts the approximate target location

**Stage 2: Localized Perceptual Refinement (LPR)**

- Input: 512×512 local crop guided by GSE output + instruction
- Output: precise bounding box $B^{pre}$ + two points $P_1^{pre}, P_2^{pre}$
- The predicted box and points are subsequently fed into frozen SAM2 to generate the segmentation mask

### Training: GRPO Reinforcement Learning

Training employs the GRPO algorithm in two sequential steps without relying on large-scale supervised data:

1. **Train LPR first**: learns precise localization on randomly cropped local images, using 6 reward signals — Box IoU, Box L1, Point L1, JSON format, Think format, and QA accuracy.
2. **Then train GSE**: leverages the trained LPR to select the optimal coarse region as pseudo-GT, using 7 reward signals — Region IoU, Region L1, region size, Box-in-region, format, and QA accuracy.

### Key Designs: Locate-informed Retrospective Reward

- **Problem**: Coarse regions predicted by GSE lack explicit GT annotations.
- **Solution**: For each sample, $n$ randomly offset candidate coarse regions are generated (each covering the GT box); the trained LPR then predicts on each region, and the region yielding the highest LPR prediction IoU is selected as the pseudo-GT $B_r^{gt}$ for GSE.
- **Effect**: LPR's localization capability inversely supervises GSE's exploration behavior, forming a closed-loop optimization.

### Unified Multi-task Response Reward

$R_{response}$ unifies three task types:

- **Instruction-guided segmentation**: a score of 1 is assigned if the response contains "is detected/found"
- **Multiple-choice VQA**: exact match against the option
- **Open-ended VQA**: fuzzy matching with similarity >0.8

## FineRS-4k Dataset

- **Source**: YouTube and self-collected drone videos at 3840×2060 resolution
- **Scale**: 4,563 high-resolution images, 8,411 small object instances, 12,132 text–mask annotation pairs
- **Splits**: Train 8,956 / Validation 749 / Test 2,427
- **Object scales**: Small (>0.055%), Extra Small (0.017%–0.055%), Extra-Extra Small (<0.017%)
- **Task types**: instruction-guided segmentation 39%, multiple-choice VQA 30.5%, open-ended VQA 30.5%
- **Attribute dimensions**: color, shape, position, and others
- **Annotation pipeline**: 14 volunteers with pairwise cross-checking + final quality review by 4 senior annotators

Compared with existing datasets: V* contains only 191 samples without masks; HR-Bench has only 200 samples without masks; refCOCOg and ReasonSeg target standard resolution. FineRS-4k is the first high-resolution small-object dataset that provides both QA and mask annotations.

## Key Experimental Results

### Main Results: FineRS-4k Test Set

| Method | gIoU | cIoU | MVQA | OVQA |
|--------|------|------|------|------|
| LISA 7B (training-free) | 9.0 | 2.4 | 0.0 | 5.5 |
| LISA++ 7B | 12.3 | 5.2 | 6.7 | 8.2 |
| MLLMs-Know 13B + LISA 13B | 17.9 | 12.8 | 52.6 | 48.8 |
| Seg-Zero 7B (training-free) | 32.1 | 6.6 | – | – |
| LISA† 7B (retrained) | 12.1 | 9.6 | 23.9 | 22.0 |
| Seg-Zero† 7B (retrained) | 46.6 | 38.6 | – | – |
| **FineRS 7B** | **55.1** | **46.5** | **83.3** | **56.7** |

Key observations:

- FineRS surpasses retrained Seg-Zero† by 8.5 points in gIoU and 7.9 points in cIoU.
- FineRS is the only method that achieves strong performance on both segmentation and VQA simultaneously.
- The advantage is most pronounced on XXS (Extra-Extra Small) objects: gIoU 47.2 vs. Seg-Zero† 31.7 (+15.5).

### Cross-dataset Generalization (without additional fine-tuning)

| Dataset | FineRS | Runner-up |
|---------|--------|-----------|
| V* Overall | **77.5** | SEAL 75.4 |
| HR-Bench 4K Avg | **63.8** | DC2 50.0 |
| HR-Bench 8K Avg | **58.1** | DC2 40.8 |

FineRS achieves substantial improvements on non-UAV scenes as well, demonstrating strong generalizability.

### Ablation Study

| Setting | gIoU | cIoU | MVQA | OVQA |
|---------|------|------|------|------|
| Full FineRS | 55.1 | 46.5 | 83.3 | 56.7 |
| w/o Retrospective Reward | 54.0 | 44.0 | 82.3 | 53.0 |
| w/o LPR random region augmentation | 53.9 | 46.7 | 83.7 | 61.9 |
| w/o QA Acc. Reward | 52.8 | 45.7 | – | – |
| w/o Box Size Reward | 51.0 | 43.4 | 56.5 | 40.8 |
| w/o Box-in-region Reward | 50.1 | 42.0 | 56.5 | 40.8 |

- The retrospective reward has the largest impact on cIoU (+2.5), validating the effectiveness of cross-stage closed-loop optimization.
- Box Size and Box-in-region rewards have a substantial effect on VQA performance (removing them drops MVQA from 83.3 to 56.5), indicating that region constraints are critical for multi-task unification.

## Highlights & Insights

**Strengths**:

1. The two-stage coarse-to-fine design effectively bypasses the resolution bottleneck of MLLMs, enabling a 7B model to handle 4K small objects.
2. The retrospective reward is elegantly designed, providing effective supervision for coarse regions without additional annotation.
3. The unified framework jointly outputs text responses and masks, integrating reasoning and segmentation.
4. Data-efficient — approximately 9K training samples suffice to outperform large-scale SFT methods.

**Limitations**:

1. The two-stage sequential pipeline increases inference latency; inference speed is not reported in the paper.
2. The GSE coarse region size is fixed at 256×256, limiting adaptability to extreme scale variations.
3. The dataset focuses exclusively on UAV aerial-view scenes; applicability to other high-resolution domains (e.g., medical imaging) remains unverified.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to combine two-stage coarse-to-fine with GRPO reinforcement learning; the cross-stage closed-loop design of the retrospective reward is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers a self-constructed dataset, multiple public benchmarks, and comprehensive ablations, but lacks inference efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Method descriptions are clear and figures are well presented.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical gap in MLLM-based high-resolution small-object segmentation; both the dataset and the method offer long-term research value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SAM-R1: Leveraging SAM for Reward Feedback in Multimodal Segmentation via Reinforcement Learning](sam-r1_leveraging_sam_for_reward_feedback_in_multimodal_segmentation_via_reinfor.md)
- [\[NeurIPS 2025\] GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset](gtpbd_a_fine-grained_global_terraced_parcel_and_boundary_dataset.md)
- [\[NeurIPS 2025\] PartNeXt: A Next-Generation Dataset for Fine-Grained and Hierarchical 3D Part Understanding](partnext_a_next-generation_dataset_for_fine-grained_and_hierarchical_3d_part_und.md)
- [\[NeurIPS 2025\] Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning](novel_class_discovery_for_point_cloud_segmentation_via_joint_learning_of_causal_.md)
- [\[CVPR 2026\] Combining Boundary Supervision and Segment-Level Regularization for Fine-Grained Action Segmentation](../../CVPR2026/segmentation/boundary_segment_action_segmentation.md)

</div>

<!-- RELATED:END -->
