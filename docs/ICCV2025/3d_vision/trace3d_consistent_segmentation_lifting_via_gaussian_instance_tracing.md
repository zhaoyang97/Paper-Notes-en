---
title: >-
  [Paper Note] Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing
description: >-
  [ICCV 2025][3D Vision][3D Segmentation] This paper proposes the Gaussian Instance Tracing (GIT) mechanism, which maintains a per-Gaussian instance weight matrix across views via inverse rasterization. GIT jointly addresses two longstanding challenges—multi-view inconsistency in 2D segmentation and boundary Gaussian ambiguity—and yields significant improvements in 3D segmentation quality under both offline contrastive learning and online self-prompting settings.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Segmentation
  - Gaussian Splatting
  - 2D-to-3D Lifting
  - Multi-view Consistency
  - Instance Tracing
  - Density Control
date: 2026-05-08
content_hash: c48b3817b7938a47
---

# Trace3D: Consistent Segmentation Lifting via Gaussian Instance Tracing

**Conference**: ICCV 2025
**arXiv**: [2508.03227](https://arxiv.org/abs/2508.03227)
**Area**: 3D Vision
**Keywords**: 3D Segmentation, Gaussian Splatting, 2D-to-3D Lifting, Multi-view Consistency, Instance Tracing, Density Control

## TL;DR

This paper proposes the Gaussian Instance Tracing (GIT) mechanism, which maintains a per-Gaussian instance weight matrix across views via inverse rasterization. GIT jointly addresses two longstanding challenges—multi-view inconsistency in 2D segmentation and boundary Gaussian ambiguity—and yields significant improvements in 3D segmentation quality under both offline contrastive learning and online self-prompting settings.

## Background & Motivation

Lifting segmentation capabilities from 2D foundation models (e.g., SAM) to 3D has become a prominent paradigm, yet two core challenges remain:

**Multi-view segmentation inconsistency**: Models such as SAM produce masks of varying hierarchical granularity for the same object across different viewpoints, leading to contradictory 3D feature supervision—the same 3D region may be grouped together in some views but separated in others.

**Boundary Gaussian ambiguity**: Standard 3DGS reconstruction is semantics-agnostic, causing numerous Gaussian primitives to span multiple objects simultaneously. This results in severe artifacts when extracting 3D assets.

Existing approaches each carry notable shortcomings:
- GaussianGrouping / CoSSegGaussians associate masks via video trackers but suffer from track loss.
- FlashSplat / SAGS filter ambiguous Gaussians directly, sacrificing geometric detail.
- EgoLifter ignores ambiguous Gaussians altogether.
- SAGD decomposes boundary Gaussians post hoc, lacking global consistency.

**Core Insight**: Gaussian primitives are inherently consistent in 3D—the same Gaussian corresponds to the same 3D point across all views. Exploiting this property to inversely trace the instance membership of each Gaussian provides a principled solution to 2D mask inconsistency.

## Method

### Overall Architecture

Given a set of input images and SAM-generated masks, the pipeline proceeds in three stages: (1) GIT computes a global instance weight matrix; (2) inconsistent patches are merged based on the weight matrix; (3) GIT-guided adaptive density control handles ambiguous Gaussians. 3D segmentation lifting (via contrastive learning or self-prompting) is subsequently performed on the refined representation.

### Key Design 1: Gaussian Instance Tracing (GIT)

**Instance Patching**: SAM masks from each view are overlapped to form disjoint instance patches.

**Inverse Rasterization**: For each pixel in each view, the Gaussians contributing to that pixel's rendering are traced, and the pixel's instance label is back-propagated to those Gaussians proportionally to their rendering contribution.

**Weight Matrix**: Each Gaussian $G_i$ maintains a $T \times L$ weight matrix $\mathbf{W}_i$, where $T$ is the maximum number of patches and $L$ is the number of views. $\mathbf{W}_{i,j}^\nu$ denotes the probability that the $i$-th Gaussian belongs to the $j$-th instance in view $\nu$.

**Efficiency**: GIT computation runs in parallel with forward rendering, incurring overhead equivalent to a single forward pass.

### Key Design 2: Consistent Instance Map Generation

For two patches observed in the same view, local information alone is insufficient to determine whether they should be merged. GIT addresses this as follows:

1. Trace the sets of Gaussians associated with each patch.
2. Identify Gaussian pairs that are jointly visible in other views.
3. Compute multi-view similarity (using inner products over instance probability distributions):
$$\text{sim}(P_a, P_b) = \frac{1}{|\Omega|} \sum_{(G_a, G_b, \nu) \in \Omega} \langle W_{G_a}^\nu, W_{G_b}^\nu \rangle$$
4. Patches whose similarity exceeds a threshold $\theta=0.5$ are automatically merged.

This effectively performs **majority voting in 3D space**: the cross-view consistency of 3D Gaussians is exploited to rectify inconsistencies in 2D segmentation.

### Key Design 3: GIT-Guided Adaptive Density Control

**Ambiguous Gaussian Detection**: An ambiguity score is computed for each Gaussian:
$$As_i = \frac{1}{|\mathcal{V}_i|} \sum_{\nu \in \mathcal{V}_i} \mathbb{I}(\max_j(W_{i,j}^\nu) < \gamma)$$
A Gaussian is deemed ambiguous if its maximum instance probability falls below $\gamma=0.8$ in more than $\theta_{As}=0.5$ of its visible views.

**Adaptive Processing**:
- **Splitting**: Ambiguous Gaussians are split into two (scale halved), with new positions sampled from the original Gaussian's PDF.
- **Pruning**: Gaussians that remain ambiguous after splitting are removed.
- This process is repeated every 1,000 iterations, with the entire Gaussian scene retrained.

The key distinction from brute-force deletion is that splitting gives each Gaussian an opportunity to reassign its instance membership, thereby avoiding surface artifacts caused by abrupt removal.

### 3D Segmentation Lifting

**Offline Contrastive Learning**: A 16-dimensional feature vector is attached to each Gaussian and trained via a contrastive loss:
$$L_{contr} = -\frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \log \frac{\sum_{u' \in \mathcal{U}^+} \exp(\text{sim}(F[u], F[u']; \tau))}{\sum_{u' \in \mathcal{U}} \exp(\text{sim}(F[u], F[u']; \tau))}$$
Using consistent instance maps and the refined Gaussian set effectively eliminates contradictory training signals.

**Online Self-Prompting**: A user provides point prompts in a reference view → SAM generates an initial mask → GIT propagates point prompts to new views → iterative expansion → the complete 3D Gaussian set is retrieved from the weight matrix.

### Loss & Training

Standard 2DGS reconstruction loss combined with the contrastive loss $L_{contr}$ (in the offline setting).

## Key Experimental Results

### 3D Object Extraction (Table 2, Replica Dataset)

| Method | Avg. mIoU↑ | Avg. PSNR↑ |
|--------|:-:|:-:|
| Gaussian Grouping | 29.6 | 13.4 |
| FlashSplat | 39.3 | 16.9 |
| EgoLifter | 55.6 | 20.1 |
| **Ours** | **72.1** | **22.6** |

The proposed method outperforms the second-best, EgoLifter, by 16.5 mIoU, while also achieving superior rendering quality.

### Novel-View 2D Segmentation (Table 3, Replica Dataset)

| Method | Avg. mIoU↑ |
|--------|:-:|
| SA3D (NeRF) | 83.0 |
| OmniSeg3D (NeRF) | 84.4 |
| SA3D-GS | 79.1 |
| EgoLifter | 82.1 |
| **Ours** | **85.5** |

This is the first GS-based method to surpass all NeRF-based counterparts on this benchmark.

### NVOS Benchmark (Table 4)

| Method | mIoU↑ | mAcc↑ |
|--------|:-:|:-:|
| FlashSplat | 91.8 | 98.6 |
| GaussianCut | 92.5 | 98.4 |
| **Ours** | **92.5** | **98.6** |

State-of-the-art performance is achieved, with results that are best or tied-best on both metrics.

### Ablation Study (Table 5)

| Configuration | NVS mIoU | Object mIoU | PSNR |
|---------------|:-:|:-:|:-:|
| 2DGS + w/o GIT + w/o Density Control | 87.0 | 62.5 | 21.0 |
| 2DGS + Density Control only | 87.3 | 70.1 | 22.4 |
| 2DGS + Consistent Mask only | 89.2 | 63.6 | 21.2 |
| **2DGS + Both** | **89.1** | **72.1** | **22.6** |

- Density control contributes most to 3D object extraction (+13.4% object mIoU).
- Consistent masks contribute most to novel-view segmentation (+2.2 mIoU).
- The two components are complementary and jointly achieve the best results.

## Highlights & Insights

1. **Elegant use of inverse rasterization**: Forward rendering aggregates pixels from Gaussians; GIT inverts this process to map Gaussians to instances at negligible computational cost.
2. **3D consistency corrects 2D inconsistency**: The view-invariant nature of 3D Gaussians is leveraged to rectify 2D predictions, effectively turning a data-level problem into a geometric advantage.
3. **Splitting outperforms deletion**: Splitting ambiguous Gaussians rather than removing them provides an opportunity for re-assignment, preventing surface degradation.
4. **Unified online/offline framework**: The GIT mechanism applies equally to contrastive learning and self-prompting paradigms, demonstrating its generality.
5. **Hierarchical segmentation capability**: The method supports object extraction at varying granularities, including sub-object parts (e.g., Captain America's hammer in Figure 1).

## Limitations & Future Work

- GIT depends on the quality of initial SAM masks; very small or texture-less objects remain challenging.
- Reliable weight matrices require adequate view coverage; sparse input views may yield unreliable estimates.
- Contrastive learning and self-prompting each have distinct trade-offs; an optimal unified strategy has yet to be identified.
- Maintaining a global weight matrix for large-scale scenes introduces non-trivial memory overhead.

## Related Work & Insights

- **SA3D / SA3D-GS**: Online self-prompting methods that iteratively apply SAM across views but lack consistency guarantees.
- **OmniSeg3D**: Hierarchical contrastive learning on NeRF, but does not address mask inconsistency.
- **EgoLifter**: Combines contrastive learning with 3DGS but ignores ambiguous Gaussians.
- **FlashSplat**: Performs binary segmentation by filtering ambiguous Gaussians, at the cost of significant detail loss.
- **GaussianGrouping**: Associates masks via video trackers but is susceptible to track loss.
- **GaussianEditor**: Also employs inverse rendering for semantic editing, but targets single-view editing rather than global consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The GIT mechanism is original and general; the density control strategy is elegant.
- **Technical Depth**: ⭐⭐⭐⭐ — Method design is clear and each module is internally coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluation covers Replica, NVOS, and in-the-wild scenes, with complete ablations and fair comparisons.
- **Value**: ⭐⭐⭐⭐⭐ — Directly applicable to 3D asset extraction and scene editing.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ — A significant advance in segmentation lifting that resolves long-standing pain points in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [\[ICCV 2025\] CutS3D: Cutting Semantics in 3D for 2D Unsupervised Instance Segmentation](cuts3d_cutting_semantics_in_3d_for_2d_unsupervised_instance_segmentation.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[AAAI 2026\] UniC-Lift: Unified 3D Instance Segmentation via Contrastive Learning](../../AAAI2026/3d_vision/unic-lift_unified_3d_instance_segmentation_via_contrastive_learning.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)

</div>

<!-- RELATED:END -->
