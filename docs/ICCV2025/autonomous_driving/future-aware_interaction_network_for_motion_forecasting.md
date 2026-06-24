---
title: >-
  [Paper Note] Future-Aware Interaction Network For Motion Forecasting
description: >-
  [ICCV 2025][Autonomous Driving][Motion Forecasting] This paper proposes FINet, which incorporates latent future trajectories into the scene encoding stage for joint optimization, while introducing the Mamba architecture as a replacement for Transformers in spatiotemporal modeling, achieving efficient and accurate motion forecasting.
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Motion Forecasting"
  - "Mamba"
  - "State Space Model"
  - "Trajectory Prediction"
date: 2026-05-08
content_hash: 1a921c215a6636ab
---

# Future-Aware Interaction Network For Motion Forecasting

**Conference**: ICCV 2025
**arXiv**: [2503.06565](https://arxiv.org/abs/2503.06565)  
**Code**: Unavailable (authors indicate code will be released upon acceptance)  
**Area**: Autonomous Driving
**Keywords**: Motion Forecasting, Mamba, State Space Model, Autonomous Driving, Trajectory Prediction

## TL;DR

This paper proposes FINet, which incorporates latent future trajectories into the scene encoding stage for joint optimization, while introducing the Mamba architecture as a replacement for Transformers in spatiotemporal modeling, achieving efficient and accurate motion forecasting.

## Background & Motivation

Motion forecasting is a critical component of autonomous driving, requiring the prediction of multiple plausible future trajectories based on historical trajectories and map information. Existing methods fall into two main categories:

**MLP-based**: Directly generate future trajectories from the agent's current state via MLP.

**Query-based**: Use learnable queries to aggregate information from encoded representations before decoding trajectories.

A common limitation shared by both paradigms is that **future trajectories are absent during the scene encoding stage**, causing the optimization of historical and future states to be decoupled, which may lead to unreasonable predictions (e.g., incorrectly predicting a left turn). Furthermore, the quadratic complexity of Transformers is inefficient in multi-agent scenarios.

The motivation of this paper is twofold: (1) incorporate future trajectories into scene encoding to obtain more comprehensive traffic representations through joint optimization; (2) replace Transformers with Mamba (linear complexity) to improve efficiency.

## Method

### Overall Architecture

FINet consists of three main components:
- **Lightweight Scene Encoder (LSEnc)**: Converts the scene into token representations.
- **Future-Aware Interaction Mamba (FIM)**: Models future trajectories and jointly encodes them with scene elements.
- **Temporal Enhanced Decoder (TEDec)**: Decodes future trajectories.

### Key Designs

1. **Lightweight Scene Encoder (LSEnc)**:

    - Encodes agent historical trajectories using Mamba blocks (linear complexity), taking the last-timestep token as a representation of the full trajectory.
    - Encodes lane maps using a mini-PointNet (enabling efficient processing of more points).
    - Each trajectory/lane segment is encoded as a single token, augmented with semantic category embeddings (vehicle/pedestrian/lane type).
    - Formula: $\mathcal{ST}_i^A = \text{MambaBlocks}(\mathcal{T}_i^{hist})[0] + Cls_i^A$

2. **Future-Aware Interaction Mamba (FIM)**:

    - **Future trajectory modeling**: Future trajectories are represented as a combination of the current motion state, driving intent, and inductive bias: $\mathcal{T}^{fut} = \mathcal{T}_0^{hist} + \mathcal{T}^{bias} + \mathcal{T}^{DI}$
    - Driving intent is modeled with $K$ learnable tokens; the inductive bias is added only to the first trajectory and propagated via Mamba.
    - **Adaptive Reorder Strategy (ARS)**: Addresses Mamba's inability to directly handle unordered spatial data. A reference point is predicted, and scene elements are sorted by distance to this point, converting unordered data into an ordered sequence.
    - The focal agent token is placed at the end of the sorted sequence to maximize its influence on future trajectories.
    - Bidirectional Mamba blocks are used for spatial interaction modeling.
    - In the second stage, the reference point is predicted from the first future trajectory token and supervised with an auxiliary loss to align it with the GT endpoint.

3. **Temporal Enhanced Decoder (TEDec)**:

    - Future trajectory tokens are expanded into a temporal format via interpolation: $\mathcal{IDT}^{fut} = \frac{t}{T^{fut}} \cdot \mathcal{ST}^{fut}$
    - Scene information is aggregated and temporally refined via Cross-Attention + Mamba (CAMBlock).
    - Cross-attention aggregates scene context; Mamba processes tokens in temporal order to ensure temporal consistency.
    - A final MLP outputs trajectories and confidence scores.

### Loss & Training

The total loss comprises five terms:
$$\mathcal{L} = \mathcal{L}_{traj} + \mathcal{L}_{score} + \mathcal{L}_{traj}^{int} + \mathcal{L}_{score}^{int} + L_{align}$$

- $\mathcal{L}_{traj}$: Smooth L1 trajectory regression loss.
- $\mathcal{L}_{score}$: Cross-entropy classification loss.
- Trajectory and score losses are also applied to intermediate outputs.
- $L_{align}$: Reference point alignment loss (Smooth L1).
- A Winner-Take-All strategy is employed, optimizing only the best prediction.

## Key Experimental Results

### Main Results (Tables)

**Argoverse 2 Test Set**:

| Method | b-minFDE6↓ | minADE6↓ | minFDE6↓ | MR6↓ | minADE1↓ | minFDE1↓ | MR1↓ |
|--------|-----------|----------|----------|------|----------|----------|------|
| QCNet | 1.91 | 0.65 | 1.29 | 0.16 | 1.69 | 4.30 | 0.59 |
| ProphNet | 1.88 | 0.66 | 1.32 | 0.18 | 1.76 | 4.77 | 0.61 |
| **FINet** | **1.93** | **0.66** | **1.27** | **0.15** | **1.60** | **4.02** | **0.57** |

**Argoverse 1 Validation Set** (minADE6 reduced from 0.66 to **0.59**, approximately 10% improvement).

### Ablation Study (Tables)

**Effect of Decoder Type and Inductive Bias**:

| Method | b-minFDE6↓ | minADE6↓ | minFDE6↓ | minADE1↓ | minFDE1↓ |
|--------|-----------|----------|----------|----------|----------|
| MLP-based | 2.09 | 0.74 | 1.45 | 1.74 | 4.34 |
| Query-based | 2.08 | 0.73 | 1.43 | 1.73 | 4.28 |
| Interaction (w/o bias) | 1.99 | 0.66 | 1.32 | 1.60 | 4.03 |
| Interaction (all bias) | 1.98 | 0.66 | 1.35 | 1.60 | 4.02 |
| Interaction (t=0 bias) | **1.93** | **0.65** | **1.27** | **1.57** | **3.94** |

**Efficiency Comparison** (vs. QCNet):

| Metric | QCNet | FINet | Gain |
|--------|-------|-------|------|
| FLOPs (G) | 28.0 | 1.47 | **95%↓** |
| Latency (ms) | 54.55 | 17.72 | **68%↓** |
| Model Size (M) | 7.7 | 3.7 | **52%↓** |
| GPU Memory (G) | 2.92 | 0.55 | **81%↓** |

### Key Findings

- The interaction-based approach significantly outperforms both MLP-based and query-based methods, validating the effectiveness of incorporating future trajectories into scene encoding.
- Applying the inductive bias only to the first trajectory yields the best results, as Mamba's scanning mechanism propagates this information to the remaining trajectories.
- Performance gain with $K=1$ exceeds that with $K=6$, suggesting that joint optimization facilitates more accurate scoring and more diverse trajectory generation.
- FINet substantially outperforms pure Transformer-based methods across nearly all efficiency metrics.

## Highlights & Insights

- This work is the first to propose an interaction-based paradigm that integrates future trajectories into scene encoding for joint optimization, reformulating the objective from $P(\hat{\mathcal{T}}^{fut}|\mathcal{ST})$ to $P(\hat{\mathcal{T}}^{fut}, \mathcal{ST})$ from a probabilistic perspective.
- The ARS strategy elegantly addresses Mamba's inability to handle unordered spatial data.
- Using Mamba for temporal refinement to ensure trajectory temporal consistency is a natural and well-motivated design choice.
- The efficiency gains are impressive: 95% reduction in FLOPs and 68% reduction in actual latency.

## Limitations & Future Work

- Although Mamba offers theoretically lower FLOPs, its reliance on sequential computation limits GPU parallelism compared to Transformers.
- The reference point prediction in ARS relies on heuristic design, which may pose generalization challenges across diverse scenes.
- The method predicts trajectories only for the focal agent and does not address joint multi-agent prediction.
- Future work could consider incorporating scene flow or occupancy grids as additional inputs.

## Related Work & Insights

- The comparison with QCNet demonstrates that jointly optimizing historical and future states is indeed superior to decoupled optimization.
- This represents the first application of Mamba to spatiotemporal modeling in autonomous driving, introducing a new efficient backbone option for the field.
- The core idea behind ARS is generalizable to other Mamba-based applications that involve processing unordered sets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The interaction-based paradigm of incorporating future trajectories into scene encoding is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two standard benchmarks with thorough ablation studies and detailed efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is well-structured with clearly articulated motivation and methodology.
- **Value**: ⭐⭐⭐⭐ A motion forecasting method that achieves both high efficiency and high accuracy has strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Foresight in Motion: Reinforcing Trajectory Prediction with Reward Heuristics](foresight_in_motion_reinforcing_trajectory_prediction_with_reward_heuristics.md)
- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](../../CVPR2025/autonomous_driving/trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICCV 2025\] INSTINCT: Instance-Level Interaction Architecture for Query-Based Collaborative Perception](instinct_instance-level_interaction_architecture_for_query-based_collaborative_p.md)
- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
