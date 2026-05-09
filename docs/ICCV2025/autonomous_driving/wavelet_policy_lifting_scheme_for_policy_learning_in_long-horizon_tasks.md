---
title: >-
  [Paper Note] Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks
description: >-
  [ICCV 2025][Autonomous Driving][Policy Learning] Wavelet Policy is the first work to introduce wavelet analysis into embodied intelligence policy learning. It proposes a multi-scale policy network based on a learnable lifting scheme, decomposing observation sequences into different frequency components and synthesizing action sequences layer by layer. The method achieves superior or comparable performance to baselines across five long-horizon tasks, including autonomous driving (CARLA), robotic manipulation, and multi-robot collaboration.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Policy Learning
  - Wavelet Transform
  - Lifting Scheme
  - Long-Horizon Tasks
  - Imitation Learning
date: 2026-05-08
content_hash: ae9adf78bdf4009f
---

# Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks

**Conference**: ICCV 2025
**arXiv**: [2507.04331](https://arxiv.org/abs/2507.04331)
**Code**: [https://hhuang-code.github.io/wavelet_policy/](https://hhuang-code.github.io/wavelet_policy/)
**Area**: Autonomous Driving
**Keywords**: Policy Learning, Wavelet Transform, Lifting Scheme, Long-Horizon Tasks, Imitation Learning

## TL;DR

Wavelet Policy is the first work to introduce wavelet analysis into embodied intelligence policy learning. It proposes a multi-scale policy network based on a learnable lifting scheme, decomposing observation sequences into different frequency components and synthesizing action sequences layer by layer. The method achieves superior or comparable performance to baselines across five long-horizon tasks, including autonomous driving (CARLA), robotic manipulation, and multi-robot collaboration.

## Background & Motivation

**Background**: Policy learning aims to enable agents to generate optimal actions from observations. Recent progress spans behavioral cloning, reinforcement learning, and more sophisticated approaches such as Diffusion Policy (modeling multimodal action distributions via conditional denoising diffusion) and Behavior Transformer (processing discretized actions with Transformers).

**Limitations of Prior Work**: Complex long-horizon tasks present three key challenges: (1) maintaining consistent behavior across multiple steps and managing long-range temporal dependencies, as errors accumulate otherwise; (2) multimodal action patterns — multiple valid action sequences often exist to achieve the same goal; (3) precision requirements — even small control errors can lead to task failure.

**Key Challenge**: Existing policy learning methods directly learn action sequences in the raw time domain, making it difficult to simultaneously capture global trends (long-horizon consistency) and local details (precise manipulation). High-frequency noise in long sequences also interferes with the recognition of multimodal patterns.

**Goal**: How to construct a policy learning framework from a signal processing perspective that simultaneously captures global trends and fine-grained variations in action sequences?

**Key Insight**: The authors observe that applying wavelet decomposition to robot joint action sequences reveals that coarse-scale (low-frequency) components clearly exhibit several distinct action "modes" without noisy fluctuations, while fine-scale (high-frequency) components capture rapid changes. This motivates a coarse-to-fine action generation strategy, analogous to residual connections.

**Core Idea**: A wavelet policy network based on a learnable lifting scheme. In the analysis phase, observation sequences are recursively decomposed into multi-scale low-frequency approximations and high-frequency details. A converter maps these to action space, and a synthesis phase reconstructs the full action sequence from coarse to fine.

## Method

### Overall Architecture

Given an input observation sequence $S = \{s_t, \ldots, s_{t+N}\}$, the goal is to generate a corresponding action sequence $A = \{a_t, \ldots, a_{t+N}\}$. The framework consists of two stages: the **analysis phase** recursively decomposes observations into multi-scale approximation components $S_s^L$ and detail components $\{S_d^l\}$; a **converter** maps each component from observation space to action space, yielding $A_s^L, \{A_d^l\}$; and the **synthesis phase** reconstructs the full action sequence layer by layer from coarse to fine.

### Key Designs

1. **Learnable Lifting Scheme**:

    - Function: Replaces fixed wavelets (Haar, Daubechies) with an end-to-end learnable multi-scale signal decomposition and reconstruction.
    - Mechanism: In the analysis block, a splitter divides the sequence into two branches $S_e, S_o$; a prediction network $\mathcal{P}$ captures high-frequency details $S_d = S_o - \mathcal{P}(S_e)$; an update network $\mathcal{U}$ captures low-frequency approximations $S_s = S_e + \mathcal{U}(S_d)$. The synthesis block applies symmetric operations: $A_e = A_s - \hat{\mathcal{U}}(A_d)$ and $A_o = A_d + \hat{\mathcal{P}}(A_e)$. All network parameters are learnable, retaining the advantages of wavelets while adding flexibility.
    - Design Motivation: Manual selection of wavelet types (Haar, Daubechies, Morlet) is heuristic and non-learnable. Traditional wavelet transforms lack flexibility and generalization. Ablation experiments confirm that learnable wavelets (0.339/T4) significantly outperform fixed Haar (0.265/T4) and DB2 (0.219/T4).

2. **Causal Dilated Convolution Instantiation**:

    - Function: Instantiates $\mathcal{P}$, $\mathcal{U}$, $\hat{\mathcal{P}}$, $\hat{\mathcal{U}}$ as dilated convolutions that preserve temporal causality.
    - Mechanism: Causal convolutions ensure that the output at time $t$ depends only on inputs at $t$ and earlier, preventing future information leakage. Dilated convolutions allow the network to integrate information over wider temporal intervals without increasing parameters (adjacent samples in the lifting scheme are separated by one time step).
    - Design Motivation: Temporal causality is critical in policy learning — current actions must not depend on future observations. Ablation results show that replacing causal convolutions with non-causal ones causes a significant performance drop (T1: 0.494 vs. 0.953).

3. **Redundant Lifting + Transformer Merger**:

    - Function: Addresses input length constraints and merging issues in traditional lifting schemes.
    - Mechanism: A Transformer self-attention module serves as the splitter (duplicating output to two branches), eliminating the $2^L$ minimum input length requirement. A Transformer cross-attention module serves as the merger ($Q = A_s^l$, $K = V = A_d^l$), reconstructing from low-frequency and high-frequency components layer by layer, replacing the conventional position-interleaved merge.
    - Design Motivation: The even-odd split in traditional lifting schemes requires inputs of at least $2^L$ in length, and the merge operation doubles the sequence length. The Transformer merger flexibly recombines components via cross-attention, maintaining constant output length while conforming to the coarse-to-fine reconstruction concept.

4. **Converter**:

    - Function: Explicitly maps frequency components from observation space to action space.
    - Mechanism: A learnable subnetwork (instantiated with causal convolutions) is inserted between the analysis and synthesis stages to explicitly perform the observation-to-action space transformation.
    - Design Motivation: Observation and action distributions differ, and implicit transformation may be insufficient.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{task} + \alpha \mathcal{L}_{approx} + \beta \mathcal{L}_{detail}$:
- $\mathcal{L}_{task}$: Task-specific loss (MSE or cross-entropy).
- $\mathcal{L}_{approx}$: Approximation flow constraint ensuring low-frequency components preserve local mean consistency: $\sum \text{SmoothL}_1(\mathcal{C}(A_s^l) - A_s^{l+1})$, where $\mathcal{C}$ denotes causal moving average.
- $\mathcal{L}_{detail}$: Detail flow constraint preventing excessively large high-frequency components: $\sum \text{SmoothL}_1(A_d^l)$.

Hyperparameters are set to $\alpha = \beta = 0.1$. Results are reported as mean and standard deviation over three random seeds.

## Key Experimental Results

### Main Results — CARLA Autonomous Driving + Franka Kitchen

| Task | BeT | VQ-BeT | **Wavelet (Ours)** |
|------|-----|--------|-------------------|
| CARLA Success | 0.832±0.167 | 0.839±0.125 | **0.847±0.090** |
| Kitchen T1 | 0.948±0.034 | 0.950±0.021 | **0.953±0.020** |
| Kitchen T2 | 0.773±0.065 | 0.775±0.046 | **0.775±0.057** |
| Kitchen T3 | 0.562±0.095 | 0.559±0.105 | **0.563±0.063** |
| Kitchen T4 | 0.275±0.066 | 0.306±0.057 | **0.339±0.071** |
| Kitchen T5 | 0.027±0.023 | 0.029±0.020 | **0.041±0.027** |

Gains are most pronounced on long-horizon tasks (T4/T5), with T4 improving by +10.8% over BeT. Reduced standard deviations indicate greater stability.

### Ablation Study — Kitchen

| Configuration | T1 | T2 | T3 | T4 | T5 |
|------|-----|-----|-----|-----|-----|
| Non-causal convolution | 0.494 | 0.259 | 0.112 | 0.033 | 0.002 |
| **Causal convolution** | **0.953** | **0.775** | **0.563** | **0.339** | **0.041** |
| Haar wavelet | 0.884 | 0.668 | 0.535 | 0.265 | — |
| **Learnable wavelet** | **0.953** | **0.775** | **0.563** | **0.339** | **0.041** |

### Multimodal Behavior Analysis

| Method | CARLA Left/Right | Kitchen Entropy |
|------|-----------------|-----------------|
| Demonstrations | 0.50/0.50 | 2.96 |
| BeT | 0.293/0.699 | 2.506 |
| VQ-BeT | 0.315/0.674 | 2.508 |
| **Ours** | **0.337/0.662** | **2.511** |

Wavelet Policy more closely approximates the multimodal distribution of the demonstration data.

### Key Findings

- **Causality is decisive**: Non-causal convolutions cause performance collapse (T1 drops from 0.953 to 0.494), demonstrating that temporal causality is non-negotiable in policy learning.
- **Learnable wavelets >> fixed wavelets**: Haar wavelets achieve only 0.265 on T4 versus 0.339 for the learnable variant (+28%), indicating that data-driven wavelets better adapt to task-specific characteristics.
- **Greater gains on long-horizon tasks**: Methods perform comparably on T1–T3, but Wavelet Policy substantially outperforms on T4/T5, confirming the particular effectiveness of multi-scale decomposition for long sequences.
- **Integration with Diffusion Policy is also effective**: DP-Wavelet outperforms DP-Transformer on both Push-T (0.958 vs. 0.942) and Transport-mh (0.497 vs. 0.440), demonstrating the generality of the wavelet architecture.
- **Wins on all six D3IL tasks**: Wavelet Policy achieves higher success rates than both BeT and IBC on obstacle avoidance, aligning, pushing, sorting, stacking, and other tasks.

## Highlights & Insights

- **Viewing policy learning through a signal processing lens** is a genuinely novel perspective: treating action sequences as signals for frequency-domain analysis, where low frequencies correspond to global trends and action modes, and high frequencies capture fine adjustments and rapid changes. This perspective naturally accommodates hierarchical decision-making in long-horizon tasks.
- **Coarse-to-fine action generation**: The synthesis phase first generates the smoothest macro-level actions and progressively adds finer details, closely mirroring the hierarchical nature of human decision-making (first determining high-level direction, then refining). This shares the spirit of residual connections but is more structurally principled.
- **Generalizing the learnable lifting scheme**: The framework faithfully reconstructs the bidirectional structure of the lifting scheme through both analysis and synthesis stages, whereas prior work only employed the analysis stage. This design is transferable to any task requiring multi-scale temporal sequence modeling.

## Limitations & Future Work

- Performance gains are modest on some tasks (CARLA Success +1.5%), suggesting that the advantages of the wavelet architecture are most evident in long-sequence and multimodal scenarios.
- All experiments are conducted in simulation environments; real-robot validation is absent.
- The number of decomposition levels $L$ must be selected manually; adaptive scale mechanisms are not explored.
- Despite being framed as autonomous driving research, the CARLA experiments are conducted on a relatively simple dataset with a limited task setup (only left and right turns).
- Comparisons with recent strong baselines such as ACT and 3D Diffuser Actor are not included.

## Related Work & Insights

- **vs. Behavior Transformer (BeT)**: Wavelet Policy directly replaces the MinGPT module in BeT while keeping all other components unchanged. It comprehensively outperforms BeT on CARLA and Kitchen, particularly on long-horizon tasks and multimodal behavior capture.
- **vs. Diffusion Policy**: Wavelet Policy replaces the Transformer decoder in Diffusion Policy and achieves improvements on both Push-T and Transport, indicating that the two approaches are complementary.
- **vs. IBC (Implicit Behavioral Cloning)**: Wavelet Policy outperforms IBC across all six D3IL tasks; IBC uses an energy-based model to learn implicit policies.
- Wavelet analysis has been widely applied in vision tasks (denoising, 3D shape representation, generative models); this work is the first to introduce it into the policy learning domain.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of wavelet analysis to policy learning, proposing a complete learnable lifting scheme framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Five benchmark environments with multiple baseline comparisons and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Signal processing background is clearly explained and the method derivation is rigorous.
- Value: ⭐⭐⭐⭐ — Provides a new perspective on policy learning, though performance gains are limited on some tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations](beyond_one_shot_beyond_one_perspective_cross-view_and_long-horizon_distillation_.md)
- [\[NeurIPS 2025\] Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems](../../NeurIPS2025/autonomous_driving/causality_meets_locality_provably_generalizable_and_scalable_policy_learning_for.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
