---
title: >-
  [Paper Note] ProMerge: Prompt and Merge for Unsupervised Instance Segmentation
description: >-
  [ECCV 2024][Segmentation][Unsupervised Instance Segmentation] ProMerge is proposed to utilize self-supervised visual features (DINO) for initial patch grouping, followed by strategic merging and background-aware mask pruning for unsupervised instance segmentation. Its inference speed significantly outperforms normalized-cut methods, and training a detector with the generated pseudo-labels outpaces existing unsupervised SOTA models.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Unsupervised Instance Segmentation"
  - "Self-Supervised Features"
  - "DINO"
  - "Group-and-Merge"
  - "Pseudo-Labels"
date: 2026-05-08
content_hash: a421bc116ea5b987
---

# ProMerge: Prompt and Merge for Unsupervised Instance Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2409.18961](https://arxiv.org/abs/2409.18961)  
**Code**: None  
**Area**: Segmentation / Unsupervised Learning  
**Keywords**: Unsupervised Instance Segmentation, Self-Supervised Features, DINO, Group-and-Merge, Pseudo-Labels

## TL;DR
ProMerge is proposed to utilize self-supervised visual features (DINO) for initial patch grouping, followed by strategic merging and background-aware mask pruning for unsupervised instance segmentation. Its inference speed significantly outperforms normalized-cut methods, and training a detector with the generated pseudo-labels outpaces existing unsupervised SOTA models.

## Background & Motivation

**Background**: Unsupervised instance segmentation aims to segment independent object instances in images without relying on manually annotated data. Recent SOTA methods leverage rich visual features from self-supervised models like DINO, representing images as graphs and solving normalized-cuts to generate foreground masks.

**Limitations of Prior Work**: (1) Normalized-cut-based methods require solving generalized eigenvalue systems, which incurs high computational complexity and slow inference speed; (2) despite good segmentation quality, they fail to meet the demands of real-time or large-scale applications; (3) the memory consumption of graph construction and eigenvalue solving is also prohibitive.

**Key Challenge**: Self-supervised features provide rich local correspondence information for grouping, but mainstream methods (normalized-cut) utilize them in a heavyweight manner—requiring a much lighter grouping strategy.

**Goal**: Maintain or surpass the segmentation quality of normalized-cut methods while substantially reducing inference time.

**Key Insight**: Directly perform patch grouping and merging in the DINO feature space, replacing complex graph optimization with simple similarity thresholds and merging rules.

**Core Idea**: First, perform initial patch grouping (Prompt) using self-supervised features, then eliminate over-segmentation through strategic merging (Merge), followed by background-aware mask pruning to remove false positives. Finally, the predicted masks are used as pseudo-labels to train a standard detector.

## Method

### Overall Architecture
Input image → DINO extracts patch features → Initial grouping (patch aggregation based on feature similarity) → Strategic merging (eliminating over-segmentation) → Background-aware mask pruning → Output instance masks. Optional: Train Mask R-CNN using the generated masks as pseudo-labels.

### Key Designs

1. **Initial Patch Grouping (Prompt)**:

    - **Function**: Performs initial semantic grouping by leveraging the local correspondence of DINO features.
    - **Mechanism**: Computes the cosine similarity between adjacent patches, and groups highly similar patches together. Initial over-segmented results are obtained through connected component analysis.
    - **Design Motivation**: Since DINO features naturally encode semantic similarity, a simple similarity threshold can yield meaningful initial groupings without the need for complex graph optimization.

2. **Strategic Merging (Merge)**:

    - **Function**: Merges over-segmented fragments into complete instance masks.
    - **Mechanism**: Computes the feature similarity between adjacent groups and merges them when the similarity exceeds a threshold. The merging strategy considers spatial adjacency and semantic consistency, proceeding iteratively until no more fragments can be merged.
    - **Design Motivation**: Since initial groupings tend to be over-segmented (e.g., a single object split into multiple parts), the merging step restores complete instance boundaries.

3. **Background-Aware Mask Pruning**:

    - **Function**: Removes false positive masks belonging to the background.
    - **Mechanism**: Leverages the attention map of the [CLS] token in DINO features to estimate foreground probability, discarding masks with low foreground probability.
    - **Design Motivation**: The grouping-and-merging process may generate false foreground masks in background areas. Using DINO's global attention as a foreground prior effectively filters these out.

### Loss & Training
ProMerge itself does not require training. When training Mask R-CNN with the generated masks, the standard instance segmentation loss is applied.

## Key Experimental Results

### Main Results

| Method | COCO AP | Inference Speed | Type |
|------|---------|---------|------|
| Ours (ProMerge) | Competitive | **Much Faster** | Training-free |
| Normalized-cut based | High | Slow | Training-free |
| Ours → Mask R-CNN | **Surpasses SOTA** | Standard Detection Speed | Pseudo-label Training |

### Ablation Study

| Configuration | Segmentation Quality | Description |
|------|---------|---------|
| Initial grouping only | Over-segmentation | Requires merging |
| + Merging | Significant improvement | Restores complete instances |
| + Background pruning | Further improvement | Removes background false positives |
| Training detector | Optimal | Maximizes the value of pseudo-labels |

### Key Findings
- ProMerge's inference speed is significantly superior to normalized-cut methods, making unsupervised instance segmentation feasible on actual large-scale datasets.
- Training a detector with pseudo-labels generated by ProMerge outperforms the direct usage of normalized-cut methods, indicating that the diverse pseudo-labels generated by simpler methods are more suitable for detector training than those from complex methods.

## Highlights & Insights
- A classic case of **simplicity defeating complexity**: replacing complex normalized-cut problems with simple grouping and merging achieved orders of magnitude faster inference speed without compromising quality.
- The pseudo-label training pipeline (generating masks → training a standard detector) serves as a practical deployment solution.
- The utilization of DINO features is highly lightweight, requiring neither graph construction nor eigenvalue decomposition.

## Limitations & Future Work
- The merging strategy relies on manual thresholds, which may require tuning across different datasets.
- Adjacent objects with highly similar textures may be over-merged.
- Background pruning depends on the quality of DINO attention.
- Detailed quantitative results need to be supplemented from the full paper.

## Related Work & Insights
- **vs CutLER/TokenCut**: These are SOTA methods based on normalized-cuts, which yield good segmentation quality but slow inference; ProMerge achieves comparable quality but is much faster.
- **vs FreeSOLO**: Another line of unsupervised instance segmentation (based on SOLO); ProMerge achieves better performance by training a detector via pseudo-labels.

## Rating
- Novelty: ⭐⭐⭐⭐ The grouping-and-merging concept is elegant and effective, serving as an efficient alternative to normalized-cuts.
- Experimental Thoroughness: ⭐⭐⭐ Validated across multiple benchmarks, though detailed data still need to be supplemented.
- Writing Quality: ⭐⭐⭐ Evaluated based on abstract information.
- Value: ⭐⭐⭐⭐ Significantly lowers the inference barrier for unsupervised instance segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation](part2object_hierarchical_unsupervised_3d_instance_segmentation.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](un-evimo_unsupervised_event-based_independent_motion_segmentation.md)
- [\[CVPR 2026\] BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation](../../CVPR2026/segmentation/bipa_bilevel_prompt_adaptation_for_underwater_instance_segmentation.md)
- [\[ECCV 2024\] VISAGE: Video Instance Segmentation with Appearance-Guided Enhancement](visage_video_instance_segmentation_with_appearance-guided_enhancement.md)

</div>

<!-- RELATED:END -->
