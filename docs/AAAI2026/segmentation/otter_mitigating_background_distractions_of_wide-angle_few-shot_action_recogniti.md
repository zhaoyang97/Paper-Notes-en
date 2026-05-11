---
title: >-
  [Paper Note] Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV
description: >-
  [AAAI 2026][Segmentation][few-shot action recognition] To address background distraction in few-shot action recognition (FSAR) for wide-angle videos — where subjects occupy a small portion of the frame and temporal relat…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "few-shot action recognition"
  - "wide-angle video"
  - "RWKV"
  - "background distraction"
  - "temporal reconstruction"
date: 2026-05-08
content_hash: 86f043c1292a1f32
---

# Otter: Mitigating Background Distractions of Wide-Angle Few-Shot Action Recognition with Enhanced RWKV

**Conference**: AAAI 2026
**arXiv**: [2511.06741](https://arxiv.org/abs/2511.06741)
**Code**: [GitHub](https://github.com/wenbohuang1002/Otter)
**Area**: Image Segmentation
**Keywords**: few-shot action recognition, wide-angle video, RWKV, background distraction, temporal reconstruction

## TL;DR
To address background distraction in few-shot action recognition (FSAR) for wide-angle videos — where subjects occupy a small portion of the frame and temporal relationships degrade — this paper proposes Otter, an enhanced RWKV-based framework. It introduces a Compound Segmentation Module (CSM) for subject highlighting and a Temporal Reconstruction Module (TRM) for recovering temporal relationships, achieving state-of-the-art results on SSv2, Kinetics, UCF101, and HMDB51 benchmarks.

## Background & Motivation

### Limitations of Prior Work

**Background**: Few-shot action recognition (FSAR) classifies unseen action categories from only a handful of video samples. Wide-angle videos (FoV > 80°) can provide scene context (e.g., "climbing wall" or "construction site") that helps distinguish visually similar actions. However, approximately 35% of samples in mainstream FSAR benchmarks are wide-angle videos, and this setting remains underexplored.

Wide-angle videos introduce two core challenges for FSAR: (1) **Insufficient subject saliency** — subjects occupy a smaller proportion of wide-angle frames, causing global modeling methods such as RWKV to capture abundant secondary background information (e.g., "snow") rather than the key subject (e.g., "athlete"), resulting in a foreground–background inversion; (2) **Temporal relationship degradation** — many frames in wide-angle videos share similar backgrounds, obscuring the progression of subject actions, while RWKV lacks the capacity to reconstruct such degraded temporal relationships.

Existing methods perform well under standard viewpoints but rarely address both challenges simultaneously in wide-angle scenarios.

### Root Cause

**Goal**: How can background distraction in wide-angle few-shot action recognition be effectively mitigated to address both insufficient subject saliency and temporal relationship degradation?

## Method

### Overall Architecture
Otter is built upon the RWKV-5/6 architecture, comprising three core units (Spatial Mixing, Time Mixing, and Channel Mixing) and two primary modules. Given an $N$-way $K$-shot support set and query videos, each video is uniformly sampled at $F=8$ frames.

### Key Designs

**Compound Segmentation Module (CSM)**
Each frame is partitioned into $HW/p^2$ patches ($p=56$, i.e., $4 \times 4$ division). Adaptive weights for each patch are learned via RWKV's Spatial Mixing and Channel Mixing:

$$lw^{\vartriangle} = \text{Sigmoid}[\text{Conv}(\vartriangle^{\beta}) \oplus \vartriangle^{p}]$$

The weighted patches are restored to their original spatial positions, highlighting subject regions while suppressing background regions. A residual connection is applied to merge the result, achieving subject saliency enhancement prior to feature extraction.

**Temporal Reconstruction Module (TRM)**
Bidirectional scanning (forward + backward) is performed on the extracted frame features $S_f^{n,k}, Q_f^{\gamma} \in \mathbb{R}^{F \times D}$. Temporal weights are learned via Time Mixing and Channel Mixing, and the two directional outputs are averaged and combined with the original input via a residual connection:

$$\tilde{\vartriangle} = [\vartriangle + \text{Avg}(\grave{\vartriangle}, \acute{\vartriangle})] \in \mathbb{R}^{F \times D}$$

**Dual Prototype Fusion**
Two complementary prototypes are constructed: Prototype 1 is enhanced by TRM for temporal modeling, while Prototype 2 preserves the subject saliency effect of CSM. The final distance is a weighted average of both: $D = \omega_1 D_1 + \omega_2 D_2$ ($\omega_1 = \omega_2 = 0.5$).

Training objective: $\mathcal{L}_{\text{total}} = 0.8 \mathcal{L}_{\text{ce}} + 0.1 \mathcal{L}_{P}^1 + 0.1 \mathcal{L}_{P}^2$, where $\mathcal{L}_P$ further separates class prototypes using cosine similarity.

## Key Experimental Results

**Main Results (5-way, Acc%)**:

### Main Results

| Method | Backbone | SSv2 1/5-shot | Kinetics 1/5-shot | UCF101 1/5-shot | HMDB51 1/5-shot |
|--------|----------|---------------|-------------------|-----------------|-----------------|
| Manta | RN50 | 63.4/87.4 | 87.4/94.2 | 95.9/99.2 | 86.8/88.6 |
| **Otter** | **RN50** | **64.7/88.5** | **90.5/96.4** | **96.8/99.2** | **88.1/89.8** |
| Manta | ViT | 66.2/89.3 | 88.2/96.3 | 97.2/99.5 | 88.9/88.8 |
| **Otter** | **ViT** | **67.2/89.9** | **91.8/97.3** | **97.7/99.4** | **89.9/90.6** |

**Ablation Study** (RN50, SSv2 1-shot): Baseline 54.6% → +CSM 61.3% → +TRM 59.5% → CSM+TRM 64.7%

**Wide-Angle Dataset (VideoBadminton)**:

### Ablation Study

| Method | VB→VB 1/5-shot | KI→VB 1/5-shot |
|--------|----------------|----------------|
| Manta | 64.1/67.1 | 62.1/65.3 |
| **Otter** | **71.2/75.8** | **69.5/72.6** |

Otter outperforms Manta by **7.1%** on the wide-angle dataset (1-shot), validating the effectiveness of the design for wide-angle scenarios.

**Patch size ablation**: $p=56$ (4×4) is optimal; $p=224$ drops to 62.7%, and $p=28$ also decreases slightly to 64.1%.

## Highlights & Insights
- This work is the first to introduce RWKV into wide-angle FSAR, identifying and addressing the dual challenges of subject saliency and temporal degradation specific to wide-angle videos.
- CSM achieves data-driven subject highlighting through adaptive patch-level weighting, requiring no additional detection or segmentation annotations.
- The bidirectional scanning design in TRM significantly outperforms unidirectional scanning (forward-only: 63.2% vs. bidirectional: 64.7%).
- The dual prototype fusion strategy is simple yet effective, separately strengthening subject saliency and temporal modeling.
- Experiments on the VideoBadminton wide-angle dataset are compelling, and CAM visualizations intuitively demonstrate the subject-focusing effect.

## Limitations & Future Work
- The patch size $p$ is a fixed hyperparameter; adaptive segmentation granularity could be explored.
- CSM assumes subjects can be highlighted through patch-level weight learning, which may be limited for extremely small targets.
- Validation is restricted to the meta-learning paradigm; finetune-based FSAR methods have not been evaluated.
- The combination of wide-angle distortion correction with Otter remains unexplored.

## Related Work & Insights
Compared to state-of-the-art methods including MoLo (CVPR'23), SOAP (MM'24), and Manta (AAAI'25), Otter achieves superior results across all benchmarks, with particularly notable advantages in wide-angle scenarios (7+% lead on VideoBadminton). Compared to multimodal methods (e.g., AmeFu-Net using depth information), Otter achieves better performance using RGB input alone. The key distinction is that Otter is the first FSAR method designed explicitly from a wide-angle video perspective.

## Related Work & Insights
- The global modeling capability of RWKV in video understanding warrants further exploration; its linear complexity is well-suited for long-video scenarios.
- The adaptive patch weight learning approach in CSM can be generalized to other video understanding tasks requiring foreground–background discrimination.
- The bidirectional temporal scanning design is consistent with the bidirectional scan paradigm in SSM models such as Mamba, offering mutual inspiration.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Bridging Granularity Gaps: Hierarchical Semantic Learning for Cross-Domain Few-Shot Segmentation](bridging_granularity_gaps_hierarchical_semantic_learning_for_cross-domain_few-sh.md)
- [\[CVPR 2026\] SCOPE: Scene-Contextualized Incremental Few-Shot 3D Segmentation](../../CVPR2026/segmentation/scope_scene-contextualized_incremental_few-shot_3d_segmentation.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](../../ICCV2025/segmentation/object-level_correlation_for_few-shot_segmentation.md)
- [\[AAAI 2026\] From Attribution to Action: Jointly ALIGNing Predictions and Explanations](from_attribution_to_action_jointly_aligning_predictions_and_explanations.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](../../ICCV2025/segmentation/move_motion-guided_few-shot_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
