---
title: >-
  [Paper Note] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting
description: >-
  [CVPR 2026][Autonomous Driving][Streaming motion prediction] This paper proposes SHARP, a motion prediction framework based on short-window streaming inference. It explicitly maintains and updates agent latent representations across time steps via an instance-aware context streamer module, and employs a dual-objective training strategy to achieve state-of-the-art streaming performance on the Argoverse 2 multi-agent benchmark while maintaining minimal latency.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Streaming motion prediction
  - heterogeneous observation lengths
  - instance-aware context streaming
  - short-window inference
  - multi-agent prediction
date: 2026-05-08
content_hash: 262918ce95dfde64
---

# SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting

**Conference**: CVPR 2026
**arXiv**: [2603.28091](https://arxiv.org/abs/2603.28091)
**Code**: N/A
**Area**: Autonomous Driving / Trajectory Prediction
**Keywords**: Streaming motion prediction, heterogeneous observation lengths, instance-aware context streaming, short-window inference, multi-agent prediction

## TL;DR

This paper proposes SHARP, a motion prediction framework based on short-window streaming inference. It explicitly maintains and updates agent latent representations across time steps via an instance-aware context streamer module, and employs a dual-objective training strategy to achieve state-of-the-art streaming performance on the Argoverse 2 multi-agent benchmark while maintaining minimal latency.

## Background & Motivation

1. **Background**: Trajectory prediction is a core component of the autonomous driving control stack, and state-of-the-art methods achieve high accuracy on large-scale benchmarks. However, these benchmarks assume fixed-length history and future windows, whereas in real-world driving, the available historical context is heterogeneous—newly entering agents may have only very short observation histories.

2. **Limitations of Prior Work**: (1) Methods that rely on long contexts must delay prediction for newly detected agents until sufficient observations accumulate. (2) Existing streaming approaches (e.g., RealMotion, DeMo) propagate information across time steps but are trained with a fixed number of streaming passes, causing performance degradation under varying streaming step counts. (3) Existing context streaming mechanisms rely solely on positional correspondence without explicitly modeling instance-level identity.

3. **Key Challenge**: Real-world driving scenes evolve continuously, and the available observation length varies significantly across agents—from a few frames to several seconds—yet most methods can only handle fixed-length inputs.

4. **Goal**: (1) How to make accurate predictions from short observation windows; (2) how to reliably propagate contextual information in continuously evolving scenes; (3) how to maintain a lightweight architecture that satisfies real-time inference requirements.

5. **Key Insight**: Incrementally processing the observation stream using short windows (1 second), while maintaining long-term agent memory through an instance-aware context streaming mechanism.

6. **Core Idea**: Short-window input + instance-aware cross-window context propagation + dual-objective training (streaming and single-chunk) = accurate and robust prediction under arbitrary observation lengths.

## Method

### Overall Architecture

The input consists of agent historical states $A^t \in \mathbb{R}^{N_a \times T_h \times D_a}$ and map information $L^t$ within a short window (e.g., 1 second). An encoder $f_E$ generates scene context $S_{\text{enc}}^t$ and target-centric features $C_{\text{enc}}^t$. During streaming inference, the previous scene context $S_{\text{enc}}^{t-1}$ and predictions $F^{t-1}$ are fused into the current representation; a DETR decoder then outputs multimodal trajectory predictions $(F^t, P^t)$. During training, prediction losses are jointly optimized with and without streaming context.

### Key Designs

1. **Instance-Aware Context Streamer**:

    - **Function**: Explicitly maintains temporal instance correspondence in agent encodings across streaming steps.
    - **Mechanism**: Cross-attention is used to integrate the previous scene context $S_{\text{enc}}^{t-1}$ into the current representation $S^t$. The key innovation lies in introducing an attention mask to explicitly model instance correspondence. Leveraging track IDs inherent in the input trajectories, the mask selectively modulates cross-attention weights by applying a learned bias parameter to matched instances of the same agent, enhancing feature consistency across time steps. By contrast, RealMotion and DeMo rely solely on positional correspondence (compensating coordinate shifts via motion-aware layer normalization), capturing instance relationships only implicitly.
    - **Design Motivation**: The short-window design inherently enables natural recovery from tracking interruptions—if an agent's track breaks at some step, it is simply treated as a new instance in the next step, rather than accumulating errors as in long-window methods.

2. **Target-Centric Context Encoding**:

    - **Function**: Extracts additional local contextual features using predicted endpoints from the previous step.
    - **Mechanism**: The $K$ trajectory endpoints predicted in the previous step are used as anchors. Scene tokens (agents and lanes) within a compact local region centered on each endpoint are aggregated to construct target-centric features $C_{\text{enc}}^t \in \mathbb{R}^{K \times (N_a + N_l)' \times D}$. An encoder $f_{TC}$ with the same architecture as the agent-centric encoder $f_S$ is used; the reduced token set keeps latency overhead low. Each endpoint serves as the origin of its local coordinate system, and the spatial relationship between the endpoint and the focal agent is additionally encoded.
    - **Design Motivation**: Predicted endpoints point to likely future positions; extracting fine-grained context in these regions helps improve prediction accuracy.

3. **Dual Training**:

    - **Function**: Improves model robustness across varying observation lengths.
    - **Mechanism**: Each training scenario is divided into non-overlapping short windows. For each new observation chunk, the model simultaneously produces: (1) a prediction with streaming context, $F^t$ ($\mathcal{L}_{\text{stream}}$); and (2) a prediction based solely on the current chunk, $F_{\text{chunk}}^t$ ($\mathcal{L}_{\text{chunk}}$). The total loss is $\mathcal{L}_{\text{dual}} = \mathcal{L}_{\text{stream}} + \mathcal{L}_{\text{chunk}}$. Each term combines cross-entropy classification loss (maximizing the probability of the optimal trajectory) and Smooth L1 regression loss (winner-takes-all strategy).
    - **Design Motivation**: Short windows yield more gradient update opportunities per training scenario and a longer effective prediction horizon. Jointly optimizing streaming and single-chunk predictions enables the model to leverage long-term context when available without depending on it—the chunk branch provides a fallback for newly detected agents.

### Loss & Training

A joint loss of cross-entropy and Smooth L1 regression under the winner-takes-all strategy is used. A 1-second short window is adopted for training (compared to the 3-second window + 1-second shift used by RealMotion/DeMo). Multi-agent prediction is incorporated through a cross-attention module: marginal predictions for individual agents are first generated, then joint predictions are produced via cross-agent and cross-modal interaction blocks.

## Key Experimental Results

### Main Results

AV2 multi-agent test set:

| Method | Streaming | avgMinADE₁ | avgMinFDE₁ | actorMR₆ | avgBrierMinFDE₆ |
|--------|-----------|-----------|-----------|---------|----------------|
| Forecast-MAE | ✗ | 1.30 | 3.33 | 0.19 | 2.24 |
| RealMotion | ✓ | 1.14 | 2.87 | 0.18 | 2.01 |
| DeMo | ✓ | 1.12 | 2.78 | 0.16 | 1.93 |
| **SHARP (Ours)** | ✓ | **1.03** | **2.53** | **0.15** | **1.80** |

AV2 single-agent test set:

| Method | MR₆ | mADE₆ | mFDE₆ | b-mFDE₆ |
|--------|-----|-------|-------|---------|
| QCNet | 0.16 | 0.65 | 1.29 | 1.91 |
| DeMo | 0.13 | 0.61 | 1.17 | 1.84 |
| **SHARP** | 0.14 | 0.64 | 1.19 | **1.83** |

### Ablation Study

Contribution of each component on AV2 Val ($T_{cl}=5s$):

| TCF | IA | DT | mADE₆ | mFDE₆ | b-mFDE₆ |
|-----|----|----|-------|-------|---------|
| ✗ | ✗ | ✗ | 0.74 | 1.28 | 1.91 |
| ✓ | ✗ | ✗ | 0.64 | 1.22 | 1.84 |
| ✓ | ✓ | ✗ | 0.63 | 1.19 | 1.81 |
| ✓ | ✓ | ✓ | 0.64 | 1.20 | 1.82 |

Critical role of Dual Training under short context $T_{cl}=1s$:

| TCF | IA | DT | mADE₆ | mFDE₆ | b-mFDE₆ |
|-----|----|----|-------|-------|---------|
| ✓ | ✗ | ✗ | 1.09 | 2.49 | 3.18 |
| ✓ | ✓ | ✗ | 1.11 | 2.55 | 3.25 |
| ✓ | ✓ | ✓ | **0.76** | **1.48** | **2.13** |

### Key Findings

- **Dual Training is critical under short context**: At $T_{cl}=1s$, adding DT reduces b-mFDE₆ from 3.25 to 2.13 (a 34% reduction), demonstrating that the chunk branch provides an essential short-window prediction capability.
- **SHARP remains stable across context lengths**: In the evolving-scene evaluation, RealMotion and DeMo exhibit significant performance degradation when deviating from their standard 3-step streaming configuration, whereas SHARP maintains competitive performance across context lengths from 1s to 8s.
- **Instance-aware mechanism is more valuable under short windows**: At long context ($5s$), the IA improvement is marginal (1.81→1.82), yet conceptually important—it enables the model to explicitly exploit tracking information.
- **Multi-agent prediction substantially outperforms Prev. SOTA**: A 6.7% reduction over DeMo (avgBrierMinFDE₆: 1.93→1.80) demonstrates the advantage of streaming methods in joint prediction.
- **Advantage grows more pronounced with extended context**: At $T_{cl}=8s$, $t_p=8s$, SHARP achieves b-mFDE₆=1.08, whereas DeMo degrades to 1.48.

## Highlights & Insights

- **Implicit advantages of the short-window design**: A 1-second window not only improves training efficiency (more gradient updates per scenario) but also naturally confers robustness to tracking interruptions—an easy-to-overlook but significant advantage over 3-second window methods.
- **Elegant design of dual-objective training**: By jointly training predictions with and without context, the model learns to exploit context when available without failing when it is absent—a more direct approach than alternatives such as masking or knowledge distillation.
- **Contribution of the evolving-scene evaluation protocol**: A comprehensive evaluation framework spanning different combinations of $t_p$ and $T_{cl}$ is proposed, which more closely reflects real deployment requirements than standard fixed-window evaluations. This protocol itself constitutes a meaningful contribution to the community.

## Limitations & Future Work

- **Perception noise not considered**: Clean detections and tracking results are assumed; perception noise in real deployments would affect performance.
- **Window size selection**: Whether a 1-second window is optimal across all scenarios is not thoroughly discussed.
- **Minor compromise on single-agent evaluation**: Not all metrics are best-in-class on the standard single-agent benchmark (e.g., MR₆=0.14 vs. DeMo's 0.13), reflecting a trade-off in favor of robustness to heterogeneous observation lengths.
- **Insufficient analysis of computational cost and real-time performance**: Specific latency figures are absent despite claims of "minimal latency."

## Related Work & Insights

- **vs. RealMotion**: RealMotion uses a 3-second window with position-based cross-attention streaming and degrades significantly when deviating from its standard configuration. SHARP, with its 1-second window, instance-aware streaming, and dual training, generalizes substantially better.
- **vs. DeMo**: DeMo performs strongly on standard benchmarks but similarly degrades under non-standard streaming step counts. SHARP's advantage becomes more pronounced at long contexts (8s).
- **vs. FLN/LaKD**: These methods handle varying input lengths via knowledge distillation but lack an information-flow mechanism; consecutive predictions are generated independently, making them unsuitable for the continuous operation required in real deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ The instance-aware context streaming and dual-objective training are novel and effective; the short-window design choice is well motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers three datasets (AV2/AV1/nuScenes); the evolving-scene evaluation is a major highlight; ablations are comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; experimental design and analysis are rigorous.
- Value: ⭐⭐⭐⭐⭐ Directly addresses real autonomous driving deployment needs; achieves state-of-the-art streaming multi-agent prediction; the proposed evaluation protocol offers long-term value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FlashCap: Millisecond-Accurate Human Motion Capture via Flashing LEDs and Event-Based Vision](flashcap_millisecond-accurate_human_motion_capture_via_flashing_leds_and_event-b.md)
- [\[ICCV 2025\] Future-Aware Interaction Network For Motion Forecasting](../../ICCV2025/autonomous_driving/future-aware_interaction_network_for_motion_forecasting.md)
- [\[CVPR 2026\] RESBev: Making BEV Perception More Robust](resbev_making_bev_perception_more_robust.md)
- [\[CVPR 2026\] ReMoT: Reinforcement Learning with Motion Contrast Triplets](remot_reinforcement_learning_with_motion_contrast_triplets.md)
- [\[CVPR 2026\] Look Before You Fuse: 2D-Guided Cross-Modal Alignment for Robust 3D Detection](look_before_you_fuse_2d-guided_cross-modal_alignment_for_robust_3d_detection.md)

</div>

<!-- RELATED:END -->
