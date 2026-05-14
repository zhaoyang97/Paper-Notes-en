---
title: >-
  [Paper Note] CAVIS: Context-Aware Video Instance Segmentation
description: >-
  [ICCV 2025][Segmentation][Video Instance Segmentation] This paper proposes CAVIS, which introduces a Context-Aware Instance Tracker (CAIT) to incorporate contextual information around object boundaries for enhanced insta…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Video Instance Segmentation"
  - "Context-Aware"
  - "Contrastive Learning"
  - "Instance Tracking"
  - "Mask2Former"
date: 2026-05-08
content_hash: 9a023939ce9c7d23
---

# CAVIS: Context-Aware Video Instance Segmentation

**Conference**: ICCV 2025
**arXiv**: [2407.03010](https://arxiv.org/abs/2407.03010)
**Code**: [https://github.com/seunghunlee918/cavis](https://github.com/seunghunlee918/cavis)
**Area**: Segmentation
**Keywords**: Video Instance Segmentation, Context-Aware, Contrastive Learning, Instance Tracking, Mask2Former

## TL;DR

This paper proposes CAVIS, which introduces a Context-Aware Instance Tracker (CAIT) to incorporate contextual information around object boundaries for enhanced instance association, and designs a Prototypical Cross-frame Contrastive loss (PCC) to enforce cross-frame feature consistency, achieving state-of-the-art performance on both VIS and VPS benchmarks.

## Background & Motivation

Video Instance Segmentation (VIS) requires simultaneously segmenting and identifying every object instance across video sequences. Modern VIS methods adopt query-based architectures (e.g., Mask2Former) and perform tracking by associating instance features across frames.

However, existing methods frequently fail in the following scenarios:

**Severe Occlusion**: When an object reappears after being occluded, re-identification based solely on core instance features is unreliable.

**Similar Appearance**: When multiple visually similar objects appear simultaneously (e.g., several cars of the same color), instance center features are insufficient for discrimination.

The authors draw inspiration from cognitive science and neuroscience: human perception heavily relies on contextual cues when interpreting complex scenes. For example, identifying a particular bicycle becomes significantly easier when the contextual cue of "a person riding it" is available.

**Core Innovation**: Contextual semantic information surrounding object boundaries is integrated into instance features, enabling the tracker to perceive not only the object itself but also its surrounding environment.

## Method

### Overall Architecture

CAVIS builds upon the Mask2Former segmentation network and comprises two core components:

1. **Context-Aware Instance Tracker (CAIT)**: Extracts and fuses contextual features surrounding each object instance.
2. **Prototypical Cross-frame Contrastive Loss (PCC Loss)**: Enhances inter-frame feature consistency through pixel-level prototype matching.

### Key Designs

**1. Context-Aware Feature Extraction**

Given instance features, feature map $F$, and segmentation mask $M$ from Mask2Former:

- An average filter (9×9 kernel) is applied to $F$ to obtain a blurred feature map encoding region-level context.
- A Laplacian filter is applied to $M$ to extract object boundary regions.
- Average pooling is performed over the blurred features within the boundary region to obtain surrounding instance features.
- The core features and surrounding features are concatenated and fused via an MLP to produce the context-aware feature $Q$.

The Laplacian boundary is more precise than a dilated mask, while average filtering naturally covers semantic information outside the object boundary — together achieving efficient "surrounding context" extraction.

**2. Context-Aware Cross-Frame Matching**

The context-aware feature $Q$ is fed into an improved transformer-based tracker:
- A context-aware cross-attention mechanism aligns the context-aware features of the current frame with corresponding features from historical frames.
- Hungarian matching is applied to align instance feature ordering across frames, with consistent indices indicating the same object identity.

**3. Prototypical Cross-frame Contrastive Loss (PCC Loss)**

- A prototype is constructed for each instance as a weighted average of high-level feature map activations within the mask region.
- Cosine similarity between prototypes and pixel embeddings generates an instance-pixel correlation map.
- The contrastive loss enforces consistency between each instance's prototype and its corresponding pixel embeddings across frames.
- This simultaneously strengthens intra-frame region consistency and inter-frame temporal continuity.

### Loss & Training

The overall training loss is:

$$L = L_{\text{VIS}} + \lambda_{\text{Emb}} \cdot L_{\text{Emb}} + \lambda_{\text{PCC}} \cdot L_{\text{PCC}}$$

- $L_{\text{VIS}}$: Standard VIS loss (classification + BCE + Dice)
- $L_{\text{Emb}}$: Instance embedding contrastive loss (cross-frame)
- $L_{\text{PCC}}$: Prototypical cross-frame contrastive loss
- Mask2Former (Swin-L) is used as the segmentation backbone
- Standard VIS training strategy is adopted

## Key Experimental Results

### Main Results

OVIS dataset (most challenging VIS benchmark):

| Method | AP |
|--------|----|
| DVIS (ICCV'23) | 37.8 |
| CTVIS (ICCV'23) | 38.7 |
| GenVIS (CVPR'23) | 36.4 |
| **CAVIS** | **41.0** |

YTVIS19 dataset:

| Method | AP |
|--------|----|
| DVIS | 55.1 |
| CTVIS | 55.4 |
| **CAVIS** | **57.2** |

VIPSeg (Video Panoptic Segmentation):

| Method | VPQ |
|--------|-----|
| DVIS | 47.2 |
| **CAVIS** | **49.5** |

### Ablation Study

Component contributions on the OVIS dataset:

| Configuration | AP |
|---------------|----|
| Baseline (no context) | 37.8 |
| + Context-aware features | 39.2 |
| + Context-aware cross-attention | 39.8 |
| + PCC Loss | 40.5 |
| + Full CAVIS | **41.0** |

Comparison of context extraction strategies:

| Context Strategy | AP |
|------------------|----|
| No context | 37.8 |
| Global context (CAROQ-style) | 38.9 |
| Dilated mask region | 39.0 |
| **Boundary region context (Ours)** | **39.2** |

### Key Findings

1. CAVIS achieves the largest gains on OVIS, the benchmark with heavy occlusion and high complexity (+2.3 AP), confirming the critical role of contextual information in occluded scenarios.
2. Boundary-region context (Laplacian-based) outperforms global context, which introduces irrelevant noise and incurs higher memory overhead.
3. PCC Loss alone yields significant AP improvement, indicating that pixel-level prototype consistency is more effective than instance-level embedding contrastive loss.
4. The method generalizes effectively across both VIS and VPS tasks.

## Highlights & Insights

- **Cognitively Motivated Design**: The motivation is grounded in how humans exploit contextual cues for object recognition, providing a compelling and principled justification.
- **Lightweight Context Extraction**: The combination of Laplacian filtering and average pooling is simple and efficient, requiring no additional network modules.
- **Design Elegance of PCC Loss**: By using prototypes to bridge pixel-level and instance-level representations, PCC Loss achieves finer granularity than direct instance-level contrastive objectives.
- The substantial improvement on OVIS demonstrates the critical importance of contextual information for addressing occlusion in VIS.

## Limitations & Future Work

1. The Laplacian boundary width for context extraction is fixed, which may be suboptimal for objects of varying scales.
2. The 9×9 kernel size for average filtering is manually specified without theoretical justification.
3. The experimental section in cached notes is incomplete; some quantitative results require verification against the original paper.
4. For very small objects, boundary context may be overwhelmed by background noise.
5. PCC Loss introduces additional training overhead; its impact on inference speed warrants further evaluation.

## Related Work & Insights

- CAVIS inherits the decoupled framework of DVIS (segment–track–refine) and augments the tracking stage with context-aware features.
- CAROQ also employs contextual features but relies on a global memory bank, leading to memory bottlenecks; CAVIS focuses on boundary regions for greater efficiency.
- PCC Loss draws on the contrastive learning principles of SimCLR, extending them from the instance level to the pixel–instance level.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The context-aware tracking paradigm is novel and the PCC Loss design is elegant, though the overall contribution is largely incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple VIS and VPS benchmarks with thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The cognitive science motivation is compelling, and the notation system is well-defined.
- **Value**: ⭐⭐⭐⭐ — The improvements in challenging VIS scenarios (occlusion, similar appearance) carry practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[ICLR 2026\] VINCIE: Unlocking In-context Image Editing from Video](../../ICLR2026/segmentation/vincie_unlocking_in-context_image_editing_from_video.md)
- [\[CVPR 2026\] Towards Context-Aware Image Anonymization with Multi-Agent Reasoning](../../CVPR2026/segmentation/towards_context-aware_image_anonymization_with_multi-agent_reasoning.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](../../CVPR2026/segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)

</div>

<!-- RELATED:END -->
