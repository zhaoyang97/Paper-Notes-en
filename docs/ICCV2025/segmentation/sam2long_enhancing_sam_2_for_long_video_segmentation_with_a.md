---
title: >-
  [Paper Note] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree
description: >-
  [ICCV 2025][Segmentation][SAM 2] To address the error accumulation caused by SAM 2's greedy selection strategy in long videos, this paper proposes a training-free constrained tree search memory strategy that maintains multiple segmentation paths and selects the optimal result at the video level, achieving an average improvement of 3.7 J&F across 9 VOS and 3 VOT benchmarks, with up to 5.3 gains on long-video scenarios.
tags:
  - ICCV 2025
  - Segmentation
  - SAM 2
  - long video segmentation
  - memory tree
  - error accumulation
  - occlusion recovery
  - training-free
date: 2026-05-08
content_hash: 5f6ba6422696ad95
---

# SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree

**Conference**: ICCV 2025
**arXiv**: [2410.16268](https://arxiv.org/abs/2410.16268)
**Code**: [https://github.com/Mark12Ding/SAM2Long](https://github.com/Mark12Ding/SAM2Long)
**Area**: Segmentation / Video Object Segmentation
**Keywords**: SAM 2, long video segmentation, memory tree, error accumulation, occlusion recovery, training-free

## TL;DR
To address the error accumulation caused by SAM 2's greedy selection strategy in long videos, this paper proposes a training-free constrained tree search memory strategy that maintains multiple segmentation paths and selects the optimal result at the video level, achieving an average improvement of 3.7 J&F across 9 VOS and 3 VOT benchmarks, with up to 5.3 gains on long-video scenarios.

## Background & Motivation

### Limitations of Prior Work

**Background**: SAM 2 is currently the most powerful foundational model for video object segmentation, with its memory module as a core design — leveraging memories from previous frames to prompt segmentation in the current frame. However, SAM 2 employs a greedy strategy: only the single mask with the highest predicted IoU is stored in memory per frame. While this works adequately in simple scenarios, once an incorrect mask is selected during occlusion or object reappearance, the error propagates through memory to all subsequent frames ("error accumulation"), causing irrecoverable tracking loss. This problem worsens with video length — SAM 2's performance degrades significantly in the later segments of long videos.

### Root Cause

**Goal**: How to resolve error accumulation caused by greedy memory selection in SAM 2 — without retraining — so that it maintains stable segmentation tracking in complex scenarios such as long videos and frequent occlusions.

## Method

### Overall Architecture
SAM2Long is built entirely upon SAM 2 without modifying any model parameters, introducing new parameters, or requiring training. The core modification replaces SAM 2's single-path greedy memory with a multi-path constrained tree search memory. At each frame, $P$ parallel segmentation paths are maintained (each with an independent memory bank and cumulative score), and the path with the highest cumulative score is selected as the final output.

### Key Designs
1. **Constrained Tree Memory Search**: SAM 2's decoder generates 3 candidate masks per frame. SAM2Long maintains $P$ paths, each branching into 3 candidate forks at the current frame, producing $3P$ candidates in total. These are ranked by cumulative log-IoU scores ($S_{p,k}[t] = S_p[t-1] + \log(\text{IoU}_{t}^{p,k} + \epsilon)$), and only the top-$P$ paths are retained. This resembles beam search but searches for the optimal segmentation sequence across the entire video dimension. Experiments show $P=3$ is sufficient (only 14% FPS reduction, 8% GFlops increase, 4% memory increase).

2. **Uncertainty-Aware Diversity Preservation**: When all paths are uncertain (the maximum absolute occlusion score across all paths $< \delta_\text{conf}=2$), candidate masks with distinct predicted IoU values are enforced (determined by rounding to two decimal places), preventing all paths from converging to the same erroneous prediction. Experiments show that after rounding, the actual IoU among candidate masks drops from 84.5% to 51.4%, significantly increasing diversity.

3. **Object-Aware Memory Bank Construction**: Rather than simply storing the most recent $N$ frames as in SAM 2, frames satisfying $\text{IoU}_i > \delta_\text{IoU}$ and $o_i > 0$ (positive occlusion score indicating object presence) are selected by forward traversal. Memory attention weights are further modulated by occlusion scores (reliable frames receive higher weights), directing the model to focus on memory frames where the object is clearly visible.

### Loss & Training
Entirely training-free; no parameters are modified. Hyperparameters: $P=3$, $\delta_\text{conf}=2$, $\delta_\text{IoU}=0.3$, $[w_\text{low}, w_\text{high}]=[0.95, 1.05]$. The same hyperparameter set is used across all datasets, and ablation studies demonstrate insensitivity to these values.

## Key Experimental Results

| Dataset | Avg. Duration | SAM 2.1-L (J&F) | SAM2Long-L (J&F) | Gain |
|--------|---------|-----------------|---------------------|------|
| SA-V val | 13.8s | 78.6 | 81.1 | +2.5 |
| SA-V test | 13.8s | 79.6 | 81.2 | +1.6 |
| LVOS v1 | 95.4s | 80.2 | 83.4 | +3.2 |
| LVOS v2 | 68.4s | 84.1 | 85.9 | +1.8 |
| MOSE | 12.4s | 74.5 | 75.2 | +0.7 |
| DAVIS-17 | 1.8s | 90.1 | 90.2 | +0.1 |

- SAM2Long-L achieves +5.3 on SA-V test (SAM2 version); SAM2Long-S achieves +4.7 on SA-V val.
- Average improvement of **3.7 J&F** across 12 experimental groups.
- Performance gain is positively correlated with video length: +3.2 on long videos (LVOS, 95s), negligible gain on short videos (DAVIS, 1.8s).
- Competitive performance is also observed on VOT benchmarks (LaSOT/GOT-10k).

### Ablation Study
- $P=1$ corresponds to the SAM 2 baseline; $P=2$ immediately yields +4 J&F; $P=3$ is optimal; $P=4$ provides no additional gain.
- Computational overhead is minimal: 19 FPS at $P=3$ vs. 22 FPS at $P=1$, only a 14% slowdown.
- IoU rounding for uncertainty handling reduces candidate mask diversity from 84.5% to 51.4% (actual IoU), contributing +0.4 J&F.
- IoU-based filtering for memory frame selection outperforms additional temporal or spatial selection strategies.
- The optimal memory attention modulation range is $[0.95, 1.05]$; subtle modulation is sufficient.

## Highlights & Insights
- **Precise Problem Insight**: SAM 2's mask decoder already generates multiple candidate masks, but only one is used in the original method. SAM2Long's core insight is to "defer the decision" — retaining multiple hypotheses until the end.
- **MHT-Inspired Design**: Multiple Hypothesis Tracking (MHT) is introduced into video segmentation, elegantly resolving error accumulation via constrained beam search.
- **Fully Training-Free**: No model modification, no additional parameters, no extra data — purely unlocking the latent potential of SAM 2.
- **Minimal Overhead**: The image encoder runs only once; the multi-path approach adds computation only to the lightweight decoder, trading 14% FPS for a 4.5 J&F improvement.
- **Performance Gain Positively Correlates with Video Length**: This perfectly validates the design motivation.

## Limitations & Future Work
- The performance upper bound is constrained by SAM 2 itself — if all 3 candidate masks from SAM 2 are incorrect, SAM2Long cannot recover.
- The method is primarily designed for single-object scenarios; multi-object settings are functional but leave room for optimization.
- Failures still occur when the background changes drastically and distractors are present, due to the lack of semantic understanding.
- A fixed path count of $P=3$ may lack flexibility for videos of varying complexity; adaptive path count selection is a promising future direction.

## Related Work & Insights
- **vs. SAM 2**: SAM2Long is a plug-in enhancement for SAM 2; the key difference lies in the memory strategy (multi-path tree search vs. single-path greedy).
- **vs. XMem/Cutie**: These methods design memory at the feature level, whereas SAM2Long selects optimal memory paths at the mask level — the two approaches are complementary.
- **vs. MHT Methods**: The multi-hypothesis concept is borrowed from the tracking community but is innovatively applied to segmentation memory management.

## Related Work & Insights
- The constrained tree search concept shares conceptual similarity with SWIRES in LLaVA-CoT — both defer commitment on stage-wise decisions by retaining multiple hypotheses.
- This "defer the decision" strategy is transferable to other sequential prediction tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Novel application of MHT ideas in VOS; uncertainty-aware diversity preservation is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 9 VOS + 3 VOT benchmarks, multiple model scales (T/S/B+/L) and versions (SAM2/2.1), with highly detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; per-frame performance curves intuitively demonstrate error accumulation and recovery.
- **Value**: ⭐⭐⭐⭐⭐ Plug-and-play enhancement for SAM 2, extremely practical for the community; has become the standard augmentation for SAM 2 long-video applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] E-SAM: Training-Free Segment Every Entity Model](e-sam_training-free_segment_every_entity_model.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[NeurIPS 2025\] Robust Ego-Exo Correspondence with Long-Term Memory](../../NeurIPS2025/segmentation/robust_ego-exo_correspondence_with_long-term_memory.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](../../CVPR2026/segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)

</div>

<!-- RELATED:END -->
