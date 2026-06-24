---
title: >-
  [Paper Note] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM
description: >-
  [CVPR 2025][Autonomous Driving][Trajectory Prediction] Proposes Trajectory Mamba (Tamba), which redesigns the self-attention mechanism based on selective State Space Models (SSMs) to achieve linear-time-complexity trajectory forecasting. By utilizing a joint polyline encoding strategy and a cross-state space decoder, it maintains prediction accuracy while reducing parameters by over 40% and decreasing FLOPs by 4x.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Trajectory Prediction"
  - "State Space Model"
  - "Mamba"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: f3580a2d7923110a
---

# Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM

**Conference**: CVPR 2025  
**arXiv**: [2503.10898](https://arxiv.org/abs/2503.10898)  
**Code**: [GitHub](https://github.com/YiZhou-H/Trajectory-Mamba-CVPR)  
**Area**: Autonomous Driving  
**Keywords**: Trajectory Prediction, State Space Model, Mamba, Attention Mechanism, Autonomous Driving

## TL;DR

Proposes Trajectory Mamba (Tamba), which redesigns the self-attention mechanism based on selective State Space Models (SSMs) to achieve linear-time-complexity trajectory forecasting. By utilizing a joint polyline encoding strategy and a cross-state space decoder, it maintains prediction accuracy while reducing parameters by over 40% and decreasing FLOPs by 4x.

## Background & Motivation

Motion forecasting in autonomous driving requires predicting future vehicle motion based on historical trajectories, imposing strict demands on both real-time performance and accuracy. While Transformer models based on self-attention mechanisms exhibit superior prediction accuracy, their computational complexity grows quadratically with the number of agents, presenting a bottleneck in highly dynamic scenarios.

Although methods like LSTM are more efficient, their forecasting accuracy cannot rival that of Transformers. Sparse context encoding techniques (e.g., QCNet) improve efficiency, but the inherent design constraints of self-attention still limit the efficiency of recursive inference.

**Core Problem**: How can linear time complexity be achieved while maintaining Transformer-level prediction accuracy? State Space Models (SSM/Mamba) offer a promising avenue.

Furthermore, existing methods adopt uniform horizontal categorization when merging heterogeneous data (static scenes vs. dynamic agents), failing to fully address the deep correlations among scene elements (e.g., the joint constraints imposed by pedestrians and traffic lights on vehicles).

## Method

### Overall Architecture

Tamba adopts a multi-modal three-encoder, one-decoder architecture. Three parallel Tamba encoders process: (I) spatial-temporal attention (interactions among all elements within each timestep); (II) scene attention (interactions between all agents and scene elements); (III) traffic attention (the influence of pedestrians and traffic lights on other dynamic agents). The encoder outputs are concatenated and fed into the Cross-Tamba decoder to generate $K$ candidate trajectories. A secondary decoding stage refines the trajectories by merging scene context, and finally, an RNN assigns confidence scores to the trajectories.

### Key Design 1: Joint Polyline Encoding Strategy

**Function**: To fuse heterogeneous traffic data more effectively and enhance prediction accuracy.

**Mechanism**: Polyline types with strong informational correlations are jointly encoded using a shared embedder. Specifically, pedestrian trajectories and traffic light signals are jointly encoded using a shared embedder: $\mathcal{P}_{\text{joint}} = \text{Fusion}(\text{Embed}(\mathcal{P}_{\text{pedestrian}}), \text{Embed}(\mathcal{P}_{\text{traffic}}))$. Types with highly dissimilar features, such as lane lines and traffic signs, utilize independent embedders.

**Design Motivation**: Pedestrians and traffic lights exert similar behavioral constraints on vehicles (e.g., pedestrian-first rules apply even in the absence of traffic lights). Joint encoding facilitates feature sharing through a shared embedder, enabling these constraining factors to collaboratively influence vehicle trajectory reasoning. This design reduces model complexity while enhancing semantic understanding.

### Key Design 2: Selective SSM Replacing Multi-Head Attention

**Function**: To reduce the quadratic complexity of the self-attention mechanism to linear.

**Mechanism**: Standard multi-head attention is replaced with Mamba's selective State Space Model (SSM). The SSM selectively retains or discards sequence information via input-dependent parameterization: $\mathbf{h}_{t+1} = A(\mathcal{P}_t)\mathbf{h}_t + B(\mathcal{P}_t)\mathbf{u}_t$, $\mathbf{y}_t = C(\mathcal{P}_t)\mathbf{h}_t + D(\mathcal{P}_t)\mathbf{u}_t$. The matrices $A$, $B$, $C$, and $D$ are all functions of the input $\mathcal{P}_t$, yielding an overall complexity of $O(T \cdot n^2)$ (where $T$ represents sequence length and $n$ represents state dimension), which scales linearly with sequence length.

**Design Motivation**: Trajectory forecasting is a recursive task where each prediction step serves as the input for the next; thus, the quadratic complexity of Transformers progressively accumulates during recursion. SSMs naturally support sequential processing with linear complexity, while the selective mechanism enables the model to focus on critical information. One-dimensional convolution preprocessing extracts local, short-range features, ensuring a preliminary filtering before entering the SSM.

### Key Design 3: Cross-Tamba Decoder

**Function**: To resolve the "one-to-many" trajectory generation problem and allow all targets to share a unified scene representation.

**Mechanism**: Drawing inspiration from DETR, $K$ independent query vectors are utilized to represent different candidate trajectories. The multi-head attention in cross-attention is replaced with SSM: the query vector $Q$ interacts with the Key/Value pairs output by the encoder, where Key/Value fuse the current encoded features and the reasoning results from the prior recursive step. All target agents share a single static scene representation. A prediction weight reasoning module utilizes an RNN to generate trajectory confidences.

**Design Motivation**: The decoder and encoder adopt different architectural designs, where the encoder focuses on intra-frame feature extraction (SSM replacing self-attention) and the decoder focuses on cross-modal interaction (SSM replacing cross-attention), achieving a balance between efficiency and accuracy. The RNN provides cross-trajectory constraints to ensure a reasonable overall distribution.

### Loss & Training

A three-part joint loss is used: $L_{\text{total}} = L_{\text{proposal}} + L_{\text{refine}} + \lambda L_{\text{cls}}$. Here, $L_{\text{proposal}}$ denotes the MSE loss of the candidate trajectories; $L_{\text{refine}}$ employs the negative log-likelihood of a Laplace mixture distribution, using a "winner-take-all" strategy to backpropagate only the best trajectory; and $L_{\text{cls}}$ represents the classification loss of the mixture coefficients.

## Key Experimental Results

### Main Results on Argoverse 2

| Method | b-minFDE6↓ | minADE6↓ | minFDE6↓ | MR6↓ | #Params(M)↓ | FLOPs(G)↓ |
|------|-----------|---------|---------|------|------------|----------|
| MTR | 1.98 | 0.73 | 1.44 | 0.15 | 65.78 | — |
| GANet | 1.96 | 0.72 | 1.34 | 0.17 | 61.73 | — |
| QML | 1.95 | 0.69 | 1.39 | 0.19 | 9.39 | — |
| **Tamba** | **SOTA** | **SOTA** | **SOTA** | — | **~5.6** | **~4x fewer** |

### Efficiency Comparison

| Metric | Tamba vs. Existing Methods |
|------|-----------------|
| Parameters | **Reduced by 40%+** |
| FLOPs | **4x fewer** |
| Inference Speed | **SOTA** |

### Key Findings

1. **Win-Win of Efficiency and Accuracy**: Tamba reduces parameters by over 40% and decreases FLOPs by 4x, while its prediction accuracy surpasses the vast majority of existing methods.
2. **Effectiveness of Joint Encoding**: Joint pedestrian-traffic light encoding improves vehicle trajectory forecasting accuracy, validating the value of modeling traffic rule constraints.
3. **Consistency across Two Datasets**: Tamba achieves SOTA efficiency on both Argoverse 1 and 2, demonstrating the generalizability of the method.
4. **Feasibility of SSM Replacing Attention**: This is the first work to validate that Mamba/SSM can effectively replace Transformers in the field of trajectory forecasting, pioneering a new technical line.

## Highlights & Insights

- **First Systematic Application of Mamba in Trajectory Forecasting**: Complete utilization of SSM from encoder to decoder, featuring careful task-specific redesigns rather than simple replacements.
- **Heterogeneous Encoder-Decoder Design**: The encoder uses SSM to replace self-attention for intra-frame feature extraction, while the decoder uses SSM to replace cross-attention for cross-modal reasoning.
- **Traffic Semantic Insight of Joint Encoding**: Identifies that pedestrians and traffic lights, acting as "traffic control elements," exert a collaborative constraining effect on vehicles.

## Limitations & Future Work

- **Upper Bound of SSM Expressive Power**: SSMs may perform less effectively than full-attention Transformers when capturing extremely complex, long-range dependencies.
- **Pedestrian-Traffic Light Assumption**: The joint encoding is manually designed based on domain knowledge and may not generalize to all traffic scenarios.
- **Unverified Large-Scale Scenarios**: Performance in ultra-large-scale scenarios (e.g., 100+ agents) remains to be validated.
- Future directions include exploring adaptive aggregation strategies and end-to-end polyline group learning.

## Related Work & Insights

- **QCNet/MTR**: Transformer-based trajectory forecasting SOTAs; Tamba achieves comparable performance with significantly fewer parameters.
- **Motion Mamba**: An application of SSM in human motion prediction; Tamba extends this to more complex multi-agent traffic scenarios.
- **Insight**: The linear complexity advantage of SSM/Mamba in temporal recursive tasks is highly worth exploring in a broader range of autonomous driving sub-tasks.

## Rating

⭐⭐⭐⭐ — First systematic introduction of Mamba into trajectory forecasting, achieving impressive efficiency improvements of 4x fewer FLOPs and over 40% fewer parameters. The joint encoding design offers unique domain insights. The trade-off is highly reasonable, delivering substantial efficiency gains with only minor compromises in accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](../../ICCV2025/autonomous_driving/future-aware_interaction_network_for_motion_forecasting.md)
- [\[CVPR 2025\] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting](spatiotemporal_decoupling_for_efficient_vision-based_occupancy_forecasting.md)
- [\[CVPR 2026\] GEM: Generating LiDAR World Model via Deformable Mamba](../../CVPR2026/autonomous_driving/gem_generating_lidar_world_model_via_deformable_mamba.md)

</div>

<!-- RELATED:END -->
