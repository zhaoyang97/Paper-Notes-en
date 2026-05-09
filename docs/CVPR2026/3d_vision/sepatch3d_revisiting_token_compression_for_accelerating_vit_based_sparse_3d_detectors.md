---
title: >-
  [Paper Note] SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors
description: >-
  [CVPR 2026][3D Vision][3D object detection] This paper proposes SEPatch3D, which achieves 57% inference acceleration with comparable detection accuracy in ViT-based sparse multi-view 3D detection, via spatiotemporal-aware dynamic patch size selection and an entropy-based informative patch enhancement mechanism.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D object detection
  - token compression
  - patch size selection
  - multi-view detection
  - ViT acceleration
date: 2026-05-08
content_hash: 25f35f932956a7e8
---

# SEPatch3D: Revisiting Token Compression for Accelerating ViT-based Sparse Multi-View 3D Object Detectors

**Conference**: CVPR 2026
**arXiv**: [2604.14563](https://arxiv.org/abs/2604.14563)
**Code**: [github.com/Mingqj/SEPatch3D](https://github.com/Mingqj/SEPatch3D)
**Area**: 3D Vision
**Keywords**: 3D object detection, token compression, patch size selection, multi-view detection, ViT acceleration

## TL;DR

This paper proposes SEPatch3D, which achieves 57% inference acceleration with comparable detection accuracy in ViT-based sparse multi-view 3D detection, via spatiotemporal-aware dynamic patch size selection and an entropy-based informative patch enhancement mechanism.

## Background & Motivation

ViT-based sparse query-based multi-view 3D detectors (e.g., StreamPETR) deliver strong performance but suffer from high inference latency. Existing token compression strategies have the following limitations: (1) token pruning may discard informative background regions critical for learning hard negatives; (2) irregular aggregation in token merging disrupts contextual consistency; (3) naively increasing patch size beyond a threshold (e.g., >18) degrades performance due to loss of fine-grained semantic cues. The core observation is that increasing patch size reduces computation but must simultaneously preserve fine-grained information in semantically important regions.

## Method

### Overall Architecture

A two-stage strategy: (1) dynamic dual-patch embedding — the SPSS module adaptively selects patch size based on spatiotemporal cues; (2) selective cross-granularity feature enhancement — the IPS module identifies informative patches, and the CGFE module enhances coarse-grained patches with fine-grained counterparts.

### Key Designs

1. **Spatiotemporal-aware Patch Size Selection (SPSS)**: The average depth $\bar{D}^{T-1}$ and depth trend slope change $\Delta S^{T-1}$ of object queries from the previous frame are used to dynamically determine the patch size for the current frame. Distant objects with a receding trend lead to a larger patch to reduce computation; nearby objects with an approaching trend lead to a smaller patch to preserve detail; otherwise, the previous frame's setting is maintained to ensure temporal stability.

2. **Entropy-based Informative Patch Selection (IPS)**: After enhancing patch features via cross-attention with motion-aligned historical queries, the information entropy of L2-normalized features is computed. Patches whose entropy exceeds the scene mean are selected as informative regions, using an adaptive threshold rather than a fixed Top-K to accommodate varying scene complexity.

3. **Cross-Granularity Feature Enhancement (CGFE)**: Selected coarse-grained patches serve as queries, while the corresponding fine-grained patches from the original resolution serve as keys/values. Detail information is injected via position-encoding-augmented cross-attention, with residual connections preserving global structure.

### Loss & Training

The detection loss from StreamPETR is inherited. In the dual-patch embedding, the original 16×16 small patches provide fine-grained feature references, while the flexible large patches are used for efficient inference. End-to-end training is employed.

## Key Experimental Results

### Main Results

| Method | Backbone | NDS (%) | mAP (%) | Inference Time |
|--------|----------|---------|---------|----------------|
| StreamPETR (patch=16) | ViT | Baseline | Baseline | Baseline |
| ToC3D-faster | ViT | Slightly lower | Slightly lower | Faster |
| SEPatch3D-faster | ViT | **Comparable** | **Comparable** | **−57%** |

On nuScenes, inference is accelerated by 57% with less than 1-point performance drop; an additional 20% speedup over ToC3D-faster. Effectiveness is also validated on Argoverse 2.

### Ablation Study

- The joint depth-and-trend decision in SPSS outperforms using depth or trend alone.
- The adaptive entropy threshold outperforms fixed Top-K selection.
- The cross-granularity enhancement in CGFE is critical for maintaining detection accuracy.

### Key Findings

- Detection performance begins to degrade when patch size exceeds 18, but selective enhancement can extend the acceleration benefit beyond this point.
- Informative patches tend to correspond to texture-rich or edge regions — precisely where coarse patches incur the greatest information loss.
- Spatiotemporal-aware selection effectively prevents abrupt patch size changes between consecutive frames.

## Highlights & Insights

- The "enlarge patch + selective enhancement" paradigm is more suited to 3D detection than pruning or merging approaches, representing a novel perspective.
- Cross-layer interaction that uses detection query spatiotemporal information to guide backbone compression is an elegant design.
- Unlike the foreground-oriented pruning in ToC3D, this approach retains background information valuable for learning hard negatives.

## Limitations & Future Work

- The predefined patch size set ($P_s$, $P_l$) and depth threshold $\theta$ require manual specification.
- Fine-grained patches must always be computed, even though they do not participate in all ViT blocks.
- Validation is limited to the StreamPETR baseline; generalizability to other sparse detectors has not been tested.

## Related Work & Insights

- The dynamic patch size selection strategy is extensible to other ViT applications requiring an efficiency–accuracy trade-off.
- The paradigm of using spatiotemporal queries to guide backbone computation breaks the conventional unidirectional information flow.
- Cross-granularity feature enhancement can be applied to multi-scale representation learning.

## Rating

7/10 — Clear motivation, practical methodology, and significant acceleration, with practical value in autonomous driving scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Fast SceneScript: Fast and Accurate Language-Based 3D Scene Understanding via Multi-Token Prediction](fast_scenescript_fast_and_accurate_language-based_3d_scene_understanding_via_mul.md)
- [\[CVPR 2026\] VGGT-Det: Mining VGGT Internal Priors for Sensor-Geometry-Free Multi-View Indoor 3D Object Detection](vggt-det_mining_vggt_internal_priors_for_sensor-geometry-free_multi-view_indoor_.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](../../AAAI2026/3d_vision/stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[CVPR 2026\] LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates](ltgs_long-term_gaussian_scene_chronology_from_sparse_view_updates.md)

<!-- RELATED:END -->
