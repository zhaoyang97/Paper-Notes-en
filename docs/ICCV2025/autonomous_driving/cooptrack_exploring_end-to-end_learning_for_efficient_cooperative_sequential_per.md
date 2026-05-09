---
title: >-
  [Paper Note] CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception
description: >-
  [ICCV 2025][Autonomous Driving][Cooperative Perception] This paper proposes CoopTrack, the first fully instance-level end-to-end cooperative 3D multi-object tracking framework. It achieves cross-agent instance matching and fusion via a learnable graph attention association module and multi-dimensional feature extraction, reaching state-of-the-art performance on V2X-Seq.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Cooperative Perception
  - 3D Multi-Object Tracking
  - End-to-End Learning
  - V2X
  - Instance-Level Fusion
date: 2026-05-08
content_hash: 81e815feebda1f1e
---

# CoopTrack: Exploring End-to-End Learning for Efficient Cooperative Sequential Perception

**Conference**: ICCV 2025
**arXiv**: [2507.19239](https://arxiv.org/abs/2507.19239)
**Code**: [GitHub](https://github.com/zhongjiaru/CoopTrack)
**Area**: Autonomous Driving
**Keywords**: Cooperative Perception, 3D Multi-Object Tracking, End-to-End Learning, V2X, Instance-Level Fusion

## TL;DR

This paper proposes CoopTrack, the first fully instance-level end-to-end cooperative 3D multi-object tracking framework. It achieves cross-agent instance matching and fusion via a learnable graph attention association module and multi-dimensional feature extraction, reaching state-of-the-art performance on V2X-Seq.

## Background & Motivation

Single-vehicle perception is inherently limited by a single viewpoint—occlusions and restricted sensing range are common challenges. V2X (Vehicle-to-Everything) communication enables multi-agent cooperative perception. However, existing cooperative perception research has focused primarily on **single-frame tasks** (3D detection), while the more challenging **cooperative sequential perception task** (e.g., cooperative 3D multi-object tracking, MOT) remains largely underexplored.

**Limitations of Prior Work**:

**Tracking-by-Cooperative-Detection (TBCD)**: First performs cooperative detection, then applies a conventional tracker (e.g., AB3DMOT) as post-processing. The tracker cannot leverage fusion information, and the decoupled optimization of detection and tracking leads to suboptimal results.

**Existing end-to-end approach (UniV2X)**: While pioneering an end-to-end cooperative tracking framework, it suffers from two design issues:
   - Uses **rule-based association** (e.g., Euclidean distance matching), which is informationally limited and susceptible to pose noise.
   - Adopts a **fusion-before-decoding** pipeline—fusing queries from both agents before decoding with ego features—leading to ambiguity and conflicts.

**Core Idea**: Design a **fusion-after-decoding** pipeline—each agent decodes independently to obtain instance-level features, followed by cross-agent association and fusion. Association is shifted from rule-driven to **learnable graph attention-based**, leveraging multi-dimensional semantic and motion features for richer matching signals.

## Method

### Overall Architecture

CoopTrack comprises two sub-systems: a vehicle-side and a road-side agent. Each independently performs: image feature extraction → Transformer decoding → Multi-Dimensional Feature Extraction (MDFE). The road-side instance-level features are transmitted to the vehicle side via V2X communication at extremely low bandwidth. On the vehicle side, the received features are processed sequentially through: Cross-Agent Alignment (CAA) → Graph-Based Association (GBA) → feature aggregation → FFN for final output.

### Key Designs

1. **Multi-Dimensional Feature Extraction (MDFE)**:

    - **Decoupled semantic and motion features**: Existing query-based methods implicitly couple the two, causing decoding ambiguity.
    - Semantic features: extracted from query features via MLP.
    - Motion features: the relative coordinates of 8 corners of the coarse 3D bounding box are encoded into motion features via PointNet (4-layer MLP + max pooling).
    - **Temporal enhancement**: A dedicated temporal transformer block (2-layer decoder) with sinusoidal positional encoding captures temporal dependencies; short sequences are padded with zeros and handled with binary masks.
    - Historical features are updated in a sliding window via FIFO.

2. **Cross-Agent Alignment (CAA)**:

    - Differences in sensors, viewpoints, and spatial positions between the vehicle-side and road-side agents create a feature domain gap.
    - Core Idea: The domain gap can be approximated as a linear transformation, analogous to the rigid-body transformation of spatial coordinates.
    - Explicit spatial transformation: $\tilde{\mathcal{P}}^I = \mathcal{P}^I \cdot \mathbf{R}^\top + \mathbf{t}$
    - Implicit feature transformation: $\tilde{\mathcal{M}}^I = \mathcal{M}^I \cdot \hat{\mathbf{R}}^\top + \hat{\mathbf{t}}$
    - The latent rotation matrix $\hat{\mathbf{R}} \in \mathbb{R}^{d \times d}$ and translation $\hat{\mathbf{t}} \in \mathbb{R}^{1 \times d}$ are predicted by two MLPs from explicit pose parameters.
    - A 6D continuous rotation representation with piecewise mapping is used to reduce parameter count.

3. **Graph-Based Association (GBA)**:

    - A fully connected association graph $\mathcal{G} = \{\mathcal{N}, \mathcal{E}\}$ is constructed between vehicle-side and road-side instances.
    - Node features: extracted via MLP from the concatenation of motion and semantic features.
    - Edge features: Euclidean distance differences between reference points are encoded via MLP.
    - Graph attention computes the affinity matrix:
      $$\hat{A} = \frac{(\mathcal{N}^V W^V)(\mathcal{N}^I W^I)^T}{\sqrt{d}} + \mathcal{E} W^E$$
    - FFN + sigmoid generates the final affinity matrix $A$.
    - The Hungarian algorithm is applied to $1 - A$ to obtain matched pairs.

4. **Feature Aggregation and Tracking Propagation**:

    - Matched instance pairs fuse multi-dimensional features into a single representation (eliminating duplicate detections).
    - Unmatched instances are retained directly (extending the observation range).
    - Semantic features of active instances are propagated as query features to the next frame.
    - Reference points are predicted for the next frame using a constant velocity assumption.

### Loss & Training

**Two-stage training**:

- Stage 1: Vehicle-side and road-side end-to-end tracking models are trained independently.
    - $\mathcal{L}_{\text{stage1}} = 0.25 \cdot \mathcal{L}_{\text{bbx}} + 2.0 \cdot \mathcal{L}_{\text{cls}}$
    - Classification uses Focal Loss ($\alpha=0.25, \gamma=2.0$); regression uses L1 Loss.

- Stage 2: End-to-end cooperative tracking with association training.
    - $\mathcal{L}_{\text{stage2}} = 0.25 \cdot \mathcal{L}_{\text{bbx}} + 2.0 \cdot \mathcal{L}_{\text{cls}} + 10.0 \cdot \mathcal{L}_{\text{asso}}$
    - Association labels are automatically generated by matching predictions to GT via the Hungarian algorithm.
    - Association loss uses Focal Loss ($\alpha=0.5, \gamma=1.0$).

## Key Experimental Results

### Main Results

Comparison with cooperative perception state-of-the-art on V2X-Seq (ResNet101 backbone):

| Method | Paradigm | mAP↑ | AMOTA↑ | Transmission↓ |
|--------|----------|------|--------|---------------|
| V2X-ViT | TBCD | 0.268 | 0.287 | 2.56×10⁶ |
| Where2comm | TBCD | 0.162 | 0.106 | 5.40×10⁵ |
| Late Fusion | TBCD | 0.196 | 0.263 | 6.60×10² |
| UniV2X | E2EC | 0.295 | 0.239 | 6.96×10⁴ |
| **CoopTrack** | **E2EC** | **0.390** | **0.328** | **5.64×10⁴** |

CoopTrack outperforms UniV2X by +9.5% mAP and +8.9% AMOTA while requiring lower transmission volume.

### Ablation Study

Incremental contribution of each module (ResNet50 backbone):

| Pipeline | MDFE | CAA | GBA | mAP↑ | AMOTA↑ |
|----------|------|-----|-----|------|--------|
| ✗ | ✗ | ✗ | ✗ | 0.310 | 0.266 |
| ✓ | ✗ | ✗ | ✗ | 0.337 | 0.277 |
| ✓ | ✓ | ✗ | ✗ | 0.345 | 0.283 |
| ✓ | ✗ | ✓ | ✗ | 0.354 | 0.304 |
| ✓ | ✓ | ✓ | ✗ | 0.355 | 0.332 |
| ✓ | ✓ | ✓ | ✓ | **0.356** | **0.346** |

Effect of historical frame count: AMOTA improves from 0.100 (0 frames) to 0.346 (4 frames), validating the value of temporal modeling.

### Key Findings

- The fusion-after-decoding pipeline alone yields +2.7% mAP and +1.1% AMOTA over fusion-before-decoding.
- The CAA module learns implicit information: adding rotation noise only to the alignment module causes minor degradation, whereas global noise injection leads to significant performance drops.
- Higher frame rates benefit tracking: at 10 Hz, CoopTrack's AMOTA is 10.7% higher than at 2 Hz.
- The method generalizes effectively to the Griffin dataset (aerial-ground cooperation), demonstrating its broader applicability.

## Highlights & Insights

- Instance-level feature transmission incurs extremely low bandwidth overhead (5.64×10⁴ bytes/s), approximately one-thousandth of BEV feature fusion.
- Learnable association is more robust than rule-based matching: even when reference point positions are imprecise, semantic and motion features enable correct association.
- Decoupling multi-dimensional features (semantic vs. motion) resolves the decoding ambiguity caused by implicit coupling in query-based methods.
- The automatic generation of association labels cleverly leverages the prediction capability of the Stage 1 model.

## Limitations & Future Work

- The two-stage training pipeline is relatively complex; future work may explore single-stage end-to-end training.
- Validation is limited to vehicle-road (V2I) scenarios; multi-vehicle V2V settings remain to be explored.
- The communication delay compensation module, while effective, lacks fine granularity; performance still degrades under long delays.
- Pose noise has a considerable impact on the system (global noise), underscoring the importance of robust pose estimation.
- LiDAR input is not explored; only camera images are used.

## Related Work & Insights

- The temporal query propagation and prediction mechanism of PF-Track is extended in this work.
- ADA-Track's differentiable association module shares conceptual similarity with GBA, though applied at a different level.
- QUEST's instance-level feature fusion paradigm aligns with the instance-level transmission approach adopted here.
- UniV2X's pioneering work provides both the baseline and the direction for improvement in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ — First fully end-to-end learnable association framework for cooperative tracking.
- Technical Depth: ⭐⭐⭐⭐ — Well-motivated multi-module design with solid theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two datasets, comprehensive ablations, and qualitative analysis.
- Value: ⭐⭐⭐⭐ — Low bandwidth with high performance; strong practical prospects.
- Overall Recommendation: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] World4Drive: End-to-End Autonomous Driving via Intention-aware Physical Latent World Model](world4drive_end-to-end_autonomous_driving_via_intention-aware_physical_latent_wo.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)

</div>

<!-- RELATED:END -->
