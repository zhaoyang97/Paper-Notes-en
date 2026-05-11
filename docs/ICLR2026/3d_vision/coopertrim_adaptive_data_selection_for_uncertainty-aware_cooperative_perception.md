---
title: >-
  [Paper Note] COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception
description: >-
  [ICLR 2026][3D Vision][cooperative perception] CooperTrim is an adaptive feature selection framework that evaluates feature relevance via conformal temporal uncertainty estimation and dynamically determines the sharing v…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "cooperative perception"
  - "bandwidth optimization"
  - "temporal uncertainty"
  - "feature selection"
  - "conformal prediction"
date: 2026-05-08
content_hash: da53b49a88b4426c
---

# COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception

**Conference**: ICLR 2026
**arXiv**: [2602.13287](https://arxiv.org/abs/2602.13287)
**Code**: [https://cisl.ucr.edu/CooperTrim](https://cisl.ucr.edu/CooperTrim)
**Area**: 3D Vision
**Keywords**: cooperative perception, bandwidth optimization, temporal uncertainty, feature selection, conformal prediction

## TL;DR
CooperTrim is an adaptive feature selection framework that evaluates feature relevance via conformal temporal uncertainty estimation and dynamically determines the sharing volume through a data-driven mechanism. It achieves 80.28% bandwidth reduction with comparable performance on cooperative semantic segmentation, and is the first to apply selective sharing to cooperative segmentation tasks.

## Background & Motivation

**Background**: Cooperative perception enables autonomous vehicles to share encoded representations for enhanced situational awareness. Intermediate fusion is the dominant paradigm, yet the volume of transmitted features still strains wireless bandwidth (typically ~40 Mbps). Existing bandwidth optimization strategies include compression (lossy), selection (fixed threshold), and hybrid approaches.

**Limitations of Prior Work**: (a) Where2Comm employs confidence maps with fixed thresholds for feature selection, ignoring temporal context, resulting in persistently high bandwidth (~39.6 Mbps); (b) SwissCheese applies fixed-threshold channel/spatial selection without environmental adaptability; (c) all existing methods make per-frame decisions independently, repeatedly transmitting static information.

**Key Challenge**: The fundamental tension between limited bandwidth and rich sensory data — existing methods merely transmit less per frame rather than leveraging temporal continuity for demand-driven sharing.

**Goal**: (a) Utilize temporal context to identify dynamic features that genuinely require updating; (b) adaptively adjust the sharing volume according to scene complexity.

**Key Insight**: The ego vehicle can leverage its own temporal memory to determine which features carry "new information" (high temporal uncertainty) and request only those that have changed — transmitting less in simple scenes and more in complex ones.

**Core Idea**: Measure feature relevance through temporal uncertainty rather than static confidence scores, enabling environment-adaptive, demand-driven sharing.

## Method

### Overall Architecture
The ego vehicle computes conformal temporal uncertainty from the current-frame features $F_t$ and the previously fused features $F_{t-1}^{\text{fused}}$. A learnable quantile threshold $q$ and an attention mask threshold $\tau$ determine the subset of features to request. A request vector is broadcast, and the selected features received from collaborative vehicles are then fused.

### Key Designs

1. **Conformal Temporal Uncertainty**:

   - **Function**: Quantifies the degree of change in each feature channel relative to its temporal context.
   - **Mechanism**: Computes the L1 distance between the current frame and the previous fused frame, $S_t = |F_t - F_{t-1}^{\text{fused}}|$, and applies gating via a learnable quantile threshold $q$ (inspired by conformal prediction), retaining only features whose change exceeds $q$ as "uncertain."
   - **Design Motivation**: In static scenes, most features remain unchanged across frames and need not be retransmitted.

2. **Adaptive Volume Determination**:

   - **Function**: Dynamically adjusts the number of shared features according to scene complexity.
   - **Mechanism**: Cross-attention weighting is applied to uncertain features, followed by truncation via a learnable mask threshold $\tau$ — complex scenes (e.g., multiple intersections) yield high relevance scores, causing more features to exceed the threshold and thus more transmission.
   - **Design Motivation**: Realizes adaptive behavior of transmitting less in simple scenes and more in complex ones.

3. **$\epsilon$-Greedy Training Strategy**:

   - **Function**: Balances training with full features versus selected features.
   - **Mechanism**: With probability $\epsilon$, all features are used (exploration); with probability $(1-\epsilon)$, only selected features are used (exploitation). It is theoretically shown that this reduces both bias and variance of the gradient estimator.
   - **Design Motivation**: Training exclusively on partial features can introduce large gradient noise and unstable convergence.

### Loss & Training

Lagrangian-constrained optimization: $\theta^* = \arg\min_\theta L(C(\theta)) + \lambda \cdot (P(C(\theta)) - C_{1.6})$, which maximizes task performance subject to a bandwidth constraint of 1.6 Mbps. $\lambda$ is adjusted dynamically.

## Key Experimental Results

### Main Results

Cooperative semantic segmentation (OPV2V dataset, applied to CoBEVT / AttFuse / DiscoNet):

| Configuration | Dynamic IoU | Bandwidth Usage | Bandwidth Reduction |
|---|---|---|---|
| CoBEVT (original) | Baseline | 100% (40 Mbps) | — |
| CooperTrim-CoBEVT | **Comparable** | **27.9%** | **72.1%** |
| CooperTrim-AttFuse | Comparable | 21.07% | 78.93% |
| CooperTrim-DiscoNet | Comparable | 10.18% | 89.82% |

Comparison with other selection strategies:

| Method | Dynamic IoU | Bandwidth (Mbps) |
|---|---|---|
| Where2Comm | 8.62 | 39.6 |
| SwissCheese | 35.71 | 10.0 |
| **CooperTrim** | **54.03** | **11.16** |

### Ablation Study

| Analysis | Key Finding |
|---|---|
| + Compression (32×) | Bandwidth reduced to 1.46% with no IoU degradation |
| Localization error robustness | Performance degrades gracefully under positional noise |
| Communication latency robustness | Remains stable under transmission delays |
| Frame-level analysis | Dynamic scenes automatically allocated more bandwidth; static scenes exhibit very low bandwidth usage |

### Key Findings
- Average bandwidth reduction of 80.28% (segmentation) and 72.52% (detection) with comparable performance.
- CooperTrim outperforms Where2Comm by 45.41% in IoU while using 72% less bandwidth.
- Orthogonal to compression methods — combining both reduces bandwidth to 1.46%.
- Qualitative analysis confirms adaptive behavior: bandwidth usage increases when vehicles traverse intersections and decreases during straight-road travel.

## Highlights & Insights
- **Elegant exploitation of temporal information**: Using inter-frame variation directly as an uncertainty measure is simple yet effective, avoiding complex uncertainty modeling.
- **First selective perception for cooperative segmentation**: Segmentation demands pixel-level precision, posing greater bandwidth challenges than detection — achieving 80%+ reduction is highly impressive.
- **Orthogonality to compression**: Combining selection with compression achieves 1.46% bandwidth usage, demonstrating the complementarity of the two strategies.
- **Theoretical guarantees for $\epsilon$-Greedy training**: A rigorous scaling analysis of gradient bias for sparse feature training is provided.

## Limitations & Future Work
- Assumes accurate pose estimation — in practice, GPS/localization errors may affect spatial transformation.
- Validated on only two datasets (OPV2V and V2V4Real), limiting scene diversity.
- The conformal temporal uncertainty relies solely on L1 distance, without modeling semantic-level changes.
- Learnable thresholds $q$ and $\tau$ may require re-tuning under domain shift.
- Multi-hop communication and heterogeneous sensor configurations are not addressed.

## Related Work & Insights
- **vs. Where2Comm**: Where2Comm uses static confidence maps with fixed thresholds, ignoring temporal context. CooperTrim uses temporal uncertainty with adaptive thresholds, achieving 45%+ higher IoU and 72% lower bandwidth.
- **vs. SwissCheese**: SwissCheese applies fixed-threshold channel/spatial selection. CooperTrim's adaptive mechanism achieves 18%+ higher IoU at comparable bandwidth.
- **vs. UniSense**: UniSense employs uncertainty-driven selection but makes per-frame independent decisions. CooperTrim uses temporal contrast to reduce redundant transmission.
- **Implications for edge AI**: The paradigm of demand-driven transmission guided by temporal differences is transferable to any bandwidth-constrained distributed perception scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of temporal uncertainty and adaptive volume is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model / multi-task / multi-strategy comparisons, compression compatibility, and robustness analyses.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, though some equations could be more concise.
- Value: ⭐⭐⭐⭐ Substantially advances the practical deployment of cooperative perception systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](../../CVPR2026/3d_vision/long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](../../AAAI2026/3d_vision/domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](../../ICCV2025/3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](../../CVPR2026/3d_vision/varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)

</div>

<!-- RELATED:END -->
