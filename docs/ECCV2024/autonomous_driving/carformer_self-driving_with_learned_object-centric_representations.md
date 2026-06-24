---
title: >-
  [Paper Note] CarFormer: Self-Driving with Learned Object-Centric Representations
description: >-
  [ECCV2024][Autonomous Driving][object-centric learning] CarFormer is proposed to introduce self-supervised slot attention-learned object-centric representations into autonomous driving for the first time. On the CARLA Longest6 benchmark, it outperforms PlanT, which utilizes precise object attributes, while demonstrating the capability of a world model to predict future states.
tags:
  - "ECCV2024"
  - "Autonomous Driving"
  - "object-centric learning"
  - "slot attention"
  - "self-driving"
  - "bird's eye view"
  - "autoregressive transformer"
date: 2026-05-08
content_hash: 8006d4641e78e3b0
---

# CarFormer: Self-Driving with Learned Object-Centric Representations

**Conference**: ECCV2024  
**arXiv**: [2407.15843](https://arxiv.org/abs/2407.15843)  
**Code**: [https://kuis-ai.github.io/CarFormer/](https://kuis-ai.github.io/CarFormer/)  
**Area**: Autonomous Driving  
**Keywords**: object-centric learning, slot attention, self-driving, bird's eye view, autoregressive transformer

## TL;DR

CarFormer is proposed to introduce self-supervised slot attention-learned object-centric representations into autonomous driving for the first time. On the CARLA Longest6 benchmark, it outperforms PlanT, which utilizes precise object attributes, while demonstrating the capability of a world model to predict future states.

## Background & Motivation

- The choice of scene representation is crucial in autonomous driving. Bird's Eye View (BEV) has shown outstanding performance recently, but its dimensionality remains high—most pixels belong to road areas, and vehicles occupy only a small fraction of the BEV, yet they are the main cause of infractions.
- Existing object-centric methods (such as PlanT) use precise object attribute vectors (position, size, orientation, velocity), but these attributes are manually specified, potentially incomplete, and difficult to generalize to a wide variety of object types.
- Self-supervised methods such as Slot Attention have successfully decomposed scenes into objects in synthetic scenarios, but applying them to complex driving sequences remains challenging. BEV sequences, with their synthetic-like characteristics, provide a viable input space for slot extraction.
- Core Motivation: Can self-supervised slot representations replace hand-specified attribute vectors, allowing the model to automatically learn required object information (position, velocity, orientation, etc.) from spatial-temporal context?

## Core Problem

1. How to extract object-centric slot representations from BEV driving sequences in a self-supervised manner?
2. How to model scene dynamics and learn driving policies based on slot representations?
3. Can slot representations simultaneously support action prediction and future state prediction (world model)?

## Method

### Overall Architecture: Two-Stage Pipeline

**First Stage: Slot Extraction (SAVi)**

- A frozen SAVi (Slot Attention for Video) model is used to extract slot representations from BEV sequences.
- Given BEV frames of the past $T$ timesteps, a CNN encoder processes each frame to produce visual features $\mathbf{h}_i$.
- $K$ slot vectors are initialized and updated through the Slot Attention mechanism: $\mathcal{Z}_i = f_{SA}(\tilde{\mathcal{Z}_i}, \mathbf{h}_i)$.
- Temporal consistency is maintained via a predictor: $\mathcal{Z}_{i+1} = f_{pred}(\mathcal{Z}_i)$.
- Key techniques to improve slot extraction quality: (1) Assigning different colors to different vehicles; (2) Upscaling small vehicles (motorcycles, bicycles, etc.) to $4.9m \times 2.12m$; (3) Using a lightweight decoder to support a larger number of slots.

**Second Stage: CarFormer Behavior Learning**

- An autoregressive Transformer decoder based on the GPT-2 architecture.
- Trajectories are defined as a mixed-modality token sequence: $\tau_t = \{g_t^x, g_t^y, l_t, v_t, \mathbf{z}_t^1, \dots, \mathbf{z}_t^K, \mathbf{r}_t^1, \mathbf{r}_t^2, q_t^1, \dots, q_t^{2W}\}$
    - Goal $(g_t^x, g_t^y)$, traffic light status $l_t$, vehicle speed $v_t$ (discretized and quantized via k-means).
    - $K$ slot features $\mathbf{z}_t^i \in \mathbb{R}^{1 \times d}$ (continuous, projected via MLP).
    - Route vectors $\mathbf{r}_t^1, \mathbf{r}_t^2 \in \mathbb{R}^6$ (continuous, projected via MLP).
    - Quantized waypoints $q_t^i$ (discrete, retrieved via embedding lookup).

### Block Attention Mechanism

- The standard causal attention mask is replaced by a block triangular mask.
- Slot features and route vectors are grouped as a block, allowing bidirectional cross-attention within the block.
- This enables all objects and routes to interact fully, leading to better modeling of scene dynamics.

### Dual-Head Action Prediction

- **GRU Head**: Takes the final hidden vector of the backbone, concatenates it with the traffic light status, and autoregressively predicts $W$ continuous waypoints.
- **Quantization Head**: Discretizes waypoints into tokens and treats prediction as a next-token prediction problem.
- Experiments show that the GRU head significantly outperforms the quantization head.

### Loss & Training

- Total Loss: $\mathcal{L} = \mathcal{L}_{wp} + \alpha \mathcal{L}_{forecast}$
- $\mathcal{L}_{wp} = \mathcal{L}_{GRU} + \mathcal{L}_{LM}$: L1 loss of the GRU + cross-entropy of quantized waypoints.
- $\mathcal{L}_{forecast}$: MSE loss of predicting future slot representations, where $\alpha = 40$.
- A modality-aware encoder unifies both continuous and discrete inputs into the same hidden dimension $H = 768$.

## Key Experimental Results

### Main Results: CARLA Longest6 Benchmark Comparison

| Model | Representation Type | DS↑ | IS↑ | RC↑ |
|------|---------|-----|-----|-----|
| AIM-BEV | Scene (BEV) | 45.06±1.68 | 0.55±0.01 | 78.31±1.12 |
| ROACH | Scene (BEV) | 55.27±1.43 | 0.62±0.02 | 88.16±1.52 |
| PlanT | Attributes | 73.36±2.97 | **0.84±0.01** | 87.03±3.91 |
| CarFormer | Attributes | 71.53±3.52 | 0.78±0.06 | 90.01±1.60 |
| **CarFormer** | **Slots** | **74.89±1.44** | 0.79±0.02 | **92.90±1.28** |

- Slot representations achieve the highest DS with the lowest variance (±1.44 vs PlanT's ±2.97), demonstrating superior robustness.
- RC reaches up to 92.90%, far exceeding PlanT's 87.03%.

### Ablation Study

- Removing Block Attention: DS drops from 74.89 to 70.42.
- Removing Forecasting: IS drops drastically from 0.79 to 0.63, and DS drops to 57.25 (showing the largest impact).
- Removing Creeping: RC drops from 92.90 to 80.52.
- GRU vs Quantization Head: GRU head's DS = 74.89 is much higher than the quantization head's 66.87.

### Effect of Slot Count and Scaling

- 7 slots $\rightarrow$ 30 slots: DS increases from 48.17 to 71.48 (without scaling) / from 62.93 to 74.89 (with scaling).
- Scaling small vehicles is particularly effective for low slot configurations (7 slots: 48.17 $\rightarrow$ 62.93).

### Future State Prediction

- CarFormer predicting t+1: ARI=0.795, mIoU=0.702 (outperforming Input-Copy's 0.641/0.561).
- CarFormer predicting t+4: ARI=0.540, mIoU=0.454 (significantly outperforming Input-Copy's 0.412/0.375).

## Highlights & Insights

1. **First use of self-supervised slot representations in autonomous driving**: Without manually specifying object attributes, slots automatically and implicitly encode driving-critical information such as position, orientation, and velocity from spatial-temporal contexts.
2. **Joint training of driving and world models**: The forecasting auxiliary task not only yields extra supervision but also allows the agent to anticipate the intentions of other vehicles, improving IS by 0.17.
3. **Low variance and high robustness**: The cross-run variance of slot representations is only half that of PlanT, indicating better stability against scene variations.
4. **Solid engineering details**: Techniques such as vehicle coloring, small vehicle scaling, and a lightweight decoder effectively address the practical difficulties of slot extraction in driving scenarios.

## Limitations & Future Work

1. **Dependency on ground truth BEV**: The current setup assumes the availability of true BEV maps. Real-world deployment requires estimating the BEV from camera images, where cascading errors would impact slot quality.
2. **Perception bottleneck of SAVi**: The quality of slot extraction directly constrains downstream prediction; SAVi's blurry predictions in turning scenarios and missed detections in crowded scenarios propagate to CarFormer.
3. **Limited to imitation learning**: Currently designed as a single-step policy imitation learning, it does not utilize the autoregressive architecture for multi-step rollouts or introduce reinforcement learning reward signals.
4. **Hallucination issues**: The world model shows false positives (hallucinated vehicles) when predicting the future, and dynamic prediction remains challenging in complex multi-vehicle scenarios.
5. **Generalizability to be verified**: Experiments are validated only in the CARLA simulator. The complexity of real-world driving scenes vastly exceeds that of synthetic sequences in BEV.

## Related Work & Insights

| Dimension | PlanT | AIM-BEV / ROACH | CarFormer |
|------|-------|-----------------|-----------|
| Representation | Precise object attribute vectors | Scene-level BEV | Self-supervised slot representation |
| Information Acquisition | Requires precise position/velocity/orientation | Global pixels | Automatically learned from BEV sequences |
| Architecture | Transformer Encoder | CNN / RL | Autoregressive Transformer Decoder |
| World Model | Yes (attribute prediction) | No | Yes (slot prediction) |
| Scalability | Limited to predefined attribute sets | Too high-dimensional | Generalizable to any object catchable by slots |
| DS Performance | 73.36 | 45.06 / 55.27 | **74.89** |

### Inspirations & Connections

- **Potential of object-centric representations in other fields**: The success of slot attention suggests it could be applied to other tasks requiring object-level reasoning (robot manipulation, video understanding).
- **Bridging role of BEV as an intermediate representation**: The "synthetic-like" nature of BEV makes slot extraction methods, which are difficult to apply directly in real-world scenes, viable. This warrants exploring more self-supervised methods on BEV.
- **Joint Prediction and Planning**: The substantial performance gain from forecasting (IS +0.17) shows that jointly modeling prediction and planning is a key paradigm in autonomous driving.
- **Integration with End-to-End Methods**: Future work could explore extracting BEV slots directly from cameras, bypassing explicit BEV estimation to reduce cascading errors.

## Rating

- Novelty: 8/10 — First to use self-supervised slot representations in autonomous driving, presenting a clear and convincing approach.
- Experimental Thoroughness: 8/10 — Comprehensive ablations covering representation type, attention mechanism, action heads, slot quantities, etc.
- Writing Quality: 7/10 — Structurally clear, but some mathematical notation is dense; visualization analysis could be strengthened.
- Value: 7/10 — Validates the effectiveness of slots in a privileged setup, but the reliance on GT BEV limits its practical impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[AAAI 2026\] Towards 3D Object-Centric Feature Learning for Semantic Scene Completion](../../AAAI2026/autonomous_driving/towards_3d_object-centric_feature_learning_for_semantic_scene_completion.md)
- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
