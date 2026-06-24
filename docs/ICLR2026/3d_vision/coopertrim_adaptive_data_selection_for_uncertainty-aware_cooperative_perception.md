---
title: >-
  [Paper Note] COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception
description: >-
  [ICLR 2026][3D Vision][Cooperative Perception] The CooperTrim adaptive feature selection framework is proposed, which assesses feature relevance via conformal temporal uncertainty metrics and utilizes a data-driven mechanism to dynamically determine sharing quantities. It achieves an 80.28% bandwidth reduction with comparable performance in cooperative semantic segmentation, marking the first application of selective sharing to segmentation tasks.
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Cooperative Perception"
  - "Bandwidth Optimization"
  - "Temporal Uncertainty"
  - "Feature Selection"
  - "Conformal Prediction"
date: 2026-05-08
content_hash: 58d70fea2253d01e
---

# COOPERTRIM: Adaptive Data Selection for Uncertainty-Aware Cooperative Perception

**Conference**: ICLR 2026  
**arXiv**: [2602.13287](https://arxiv.org/abs/2602.13287)  
**Code**: [https://cisl.ucr.edu/CooperTrim](https://cisl.ucr.edu/CooperTrim)  
**Area**: 3D Vision  
**Keywords**: Cooperative Perception, Bandwidth Optimization, Temporal Uncertainty, Feature Selection, Conformal Prediction

## TL;DR
The CooperTrim adaptive feature selection framework is proposed, which assesses feature relevance via conformal temporal uncertainty metrics and utilizes a data-driven mechanism to dynamically determine sharing quantities. It achieves an 80.28% bandwidth reduction with comparable performance in cooperative semantic segmentation, marking the first application of selective sharing to segmentation tasks.

## Background & Motivation

**Background**: Cooperative perception allows autonomous vehicles to share encoded representations to enhance situational awareness. While intermediate fusion is the mainstream, the volume of transmitted features still pressures wireless bandwidth (typically ~40 Mbps). Existing bandwidth optimization methods include compression (lossy), selection (fixed thresholds), and hybrid strategies.

**Limitations of Prior Work**: (a) Where2Comm selects features using confidence maps with fixed thresholds, ignoring temporal context and maintaining high bandwidth (39.6 Mbps); (b) SwissCheese employs fixed thresholds for channel/spatial selection, lacking environmental adaptability; (c) All existing methods make per-frame independent decisions, repeatedly transmitting static information.

**Key Challenge**: The fundamental contradiction between limited bandwidth and abundant sensor information—existing methods only "transmit less per frame" rather than "transmit as needed" by leveraging temporal continuity.

**Goal**: (a) Utilize temporal context to identify dynamic features that truly require updates; (b) Adaptively adjust sharing volume according to environmental complexity.

**Key Insight**: The receiver (ego vehicle) can use its own temporal memory to determine which features constitute "new information" (high temporal uncertainty) and only request those that have changed. Less is transmitted in simple scenarios, while more is transmitted in complex ones.

**Core Idea**: Measure feature relevance using temporal uncertainty instead of static confidence to achieve environment-adaptive, on-demand sharing.

## Method

### Overall Architecture
CooperTrim addresses the waste of "transmitting all features every frame" in cooperative perception. Since static scenes change little between frames, repeated transmission is redundant. It delegates decision-making to the receiver (ego vehicle)—the ego vehicle encodes its sensor input to obtain current features $F_t$, compares them with the fused features of the previous frame $F_{t-1}^{\text{fused}}$ to identify "new information" relative to temporal memory (**Conformal Temporal Uncertainty**), assigns relevance scores to these uncertain features using cross-attention, and determines **sharing quantity** through a mask threshold. Only features exceeding the threshold are broadcast in a request vector. Collaborating vehicles perform spatial alignment and return only the requested feature subset, which the ego vehicle fuses and feeds into task heads. The sharing volume is not fixed but scales automatically with scene complexity; the training side utilizes an $\epsilon$-greedy strategy to stabilize optimization under sparse features.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Sensor Input X_t"] --> B["Feature Encoding<br/>Get current features F_t"]
    B --> C["Conformal Temporal Uncertainty<br/>L1 distance comparison with prev. fused features<br/>Quantile threshold q filters changed features"]
    C --> D["Adaptive Quantity Determination<br/>Cross-attention relevance score R_t<br/>Mask threshold τ truncates sharing volume"]
    D --> E["Ego Broadcasts Request Vector<br/>Requests only high-relevance features"]
    E --> F["Collaborators Send Back<br/>Spatial alignment of requested features"]
    F --> G["Feature Fusion"]
    G --> H["Task Head<br/>Segmentation / Detection output"]
    G -. Prev. Fused Feature F_{t-1} .-> C
```

### Key Designs

**1. Conformal Temporal Uncertainty: Using "inter-frame change" instead of static confidence to judge feature value**

Existing methods (e.g., Where2Comm) select features per-frame using confidence maps, ignoring what was sent previously, leading to redundant transmission of static backgrounds. CooperTrim adopts a different metric: the L1 distance $S_t = |F_t - F_{t-1}^{\text{fused}}|$ is calculated as temporal uncertainty. Only channels with significant changes are considered "uncertain and needing update." The gating threshold is not a manually tuned fixed value but a learnable quantile threshold $q$ inspired by conformal prediction—retaining only features where $S_t$ exceeds $q$. Thus, static features are naturally filtered, saving bandwidth for dynamic regions.

**2. Adaptive Quantity Determination: Scaling sharing volume with environmental complexity instead of a fixed threshold**

Fixed threshold methods (e.g., SwissCheese) treat simple and complex scenes identically, risking missing critical info at intersections or wasting bandwidth on empty roads. CooperTrim applies cross-attention weighting to the filtered uncertain features to obtain relevance scores, truncated by a learnable mask threshold $\tau$. This mechanism yields emergent adaptability: complex scenarios like multi-way intersections generate higher relevance scores, allowing more features to exceed $\tau$; open straight roads generate low scores, resulting in minimal transmission. "Less for simple, more for complex" emerges from data-driven logic.

**3. $\epsilon$-Greedy Training Strategy: Avoiding instability from training only on selected features**

Training exclusively on selected features can result in noisy gradients and unstable convergence due to input sparsity. CooperTrim adopts an exploration-exploitation approach from reinforcement learning: training with all features (exploration) with probability $\epsilon$, and with selected features (exploitation) with probability $1-\epsilon$. Theoretical analysis in the paper demonstrates that this hybrid sampling reduces both bias and variance of the gradient estimator, stabilizing training under sparse feature conditions.

### Loss & Training

The overall objective is formulated as a constrained optimization with a Lagrangian multiplier:

$$\theta^* = \arg\min_\theta L(C(\theta)) + \lambda \cdot (P(C(\theta)) - C_{1.6})$$

Where $L$ is the task loss, $P(C(\theta))$ is the bandwidth cost of the current strategy, and $C_{1.6}$ is a bandwidth budget of 1.6 Mbps. $\lambda$ is dynamically adjusted during training. Intuitively: maximize segmentation/detection performance while staying within bandwidth constraints; $\lambda$ increases penalties if transmission exceeds the budget.

## Key Experimental Results

### Main Results

Cooperative Semantic Segmentation (OPV2V dataset, applied to CoBEVT/AttFuse/DiscoNet):

| Configuration | Dynamic IoU | Bandwidth Usage | Bandwidth Reduction |
|------|---------|----------|---------|
| Original CoBEVT | Baseline | 100% (40Mbps) | — |
| CooperTrim-CoBEVT | **Comparable** | **27.9%** | **72.1%** |
| CooperTrim-AttFuse | Comparable | 21.07% | 78.93% |
| CooperTrim-DiscoNet | Comparable | 10.18% | 89.82% |

vs. other selection strategies:

| Method | Dynamic IoU | Bandwidth (Mbps) |
|------|---------|------------|
| Where2Comm | 8.62 | 39.6 |
| SwissCheese | 35.71 | 10.0 |
| **CooperTrim** | **54.03** | **11.16** |

### Ablation Study

| Analysis | Key Findings |
|------|---------|
| +Compression (32x)| Bandwidth drops to 1.46% with no IoU loss |
| Pose Error Robustness | Performance degrades gracefully under positional noise |
| Latency Robustness | Remains stable against communication delays |
| Frame-level Analysis | Automatically allocates more bandwidth to dynamic scenes and minimal for static scenes |

### Key Findings
- Achieves average bandwidth reduction of 80.28% (segmentation) and 72.52% (detection) with comparable performance.
- CooperTrim achieves 45.41% higher IoU and 72% lower bandwidth than Where2Comm.
- Orthogonal to compression—bandwidth drops to 1.46% when combined.
- Qualitative analysis confirms adaptive behavior: bandwidth usage increases at intersections and decreases on straightaways.

## Highlights & Insights
- **Clever use of temporal information**: Treating "inter-frame change" directly as an uncertainty metric is simple yet efficient, avoiding complex uncertainty modeling.
- **First selective perception for cooperative segmentation**: Segmentation requires pixel-level precision and is more bandwidth-intensive than detection—achieving 80%+ reduction is impressive.
- **Orthogonality with compression**: The combination of selection and compression reaching 1.46% bandwidth demonstrates that the strategies are complementary.
- **Theoretical guarantee for $\epsilon$-Greedy training**: Provides rigorous scaling analysis of gradient bias for training on sparse features.

## Limitations & Future Work
- Assumes precise pose—real-world GPS/localization errors may impact spatial transformation.
- Validated only on two datasets (OPV2V + V2V4Real); scene diversity is limited.
- Conformal temporal uncertainty uses only L1 distance, ignoring semantic-level changes.
- Learnable thresholds $q$ and $\tau$ may require recalibration for domain transfer.
- Does not consider multi-hop communication or heterogeneous sensor configurations.

## Related Work & Insights
- **vs. Where2Comm**: Where2Comm uses static confidence maps + fixed thresholds, ignoring time. CooperTrim uses temporal uncertainty + adaptive thresholds, resulting in 45%+ higher IoU and 72% lower bandwidth.
- **vs. SwissCheese**: SwissCheese uses fixed thresholds for channel/spatial selection. CooperTrim's adaptive mechanism achieves 18%+ higher IoU at similar bandwidth.
- **vs. UniSense**: UniSense uses uncertainty-driven selection but is frame-independent; CooperTrim uses temporal comparison to reduce redundant transmission.
- **Insights for Edge AI**: The logic of on-demand transmission driven by temporal differences is transferable to any bandwidth-constrained distributed sensing scenario.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of temporal uncertainty and adaptive quantity is novel, though individual components exist.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive multi-model/multi-task comparisons, compression compatibility, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, though some formulas could be further simplified.
- Value: ⭐⭐⭐⭐ Significant push toward practical deployment of cooperative perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction](peering_into_the_unknown_active_view_selection_with_neural_uncertainty_maps_for_.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](../../CVPR2026/3d_vision/long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)
- [\[CVPR 2026\] GSV2X: Geometry-Aware Uncertainty Modeling and Orthogonal Fusion for Robust Roadside Perception](../../CVPR2026/3d_vision/gsv2x_geometry-aware_uncertainty_modeling_and_orthogonal_fusion_for_robust_roads.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/3d_vision/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)

</div>

<!-- RELATED:END -->
