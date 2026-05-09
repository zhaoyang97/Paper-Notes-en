---
title: >-
  [Paper Note] ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models
description: >-
  [ICCV 2025][Autonomous Driving][End-to-end autonomous driving] This paper proposes ETA, a dual-system framework that shifts large-model computation from the current frame to preceding time steps and applies batch inference, enabling large-model features to be available at every frame. ETA achieves a driving score of 69.53 on Bench2Drive with a 50 ms latency, improving the state of the art by 8%.
tags:
  - ICCV 2025
  - Autonomous Driving
  - End-to-end autonomous driving
  - dual-system architecture
  - asynchronous inference
  - real-time large model deployment
  - feature forecasting
date: 2026-05-08
content_hash: 2b56e0d7f71d47d2
---

# ETA: Efficiency through Thinking Ahead, A Dual Approach to Self-Driving with Large Models

**Conference**: ICCV 2025
**arXiv**: [2506.07725](https://arxiv.org/abs/2506.07725)
**Code**: [https://github.com/OpenDriveLab/ETA](https://github.com/OpenDriveLab/ETA)
**Area**: Autonomous Driving
**Keywords**: End-to-end autonomous driving, dual-system architecture, asynchronous inference, real-time large model deployment, feature forecasting

## TL;DR

This paper proposes ETA, a dual-system framework that shifts large-model computation from the current frame to preceding time steps and applies batch inference, enabling large-model features to be available at every frame. ETA achieves a driving score of 69.53 on Bench2Drive with a 50 ms latency, improving the state of the art by 8%.

## Background & Motivation

Large vision models (e.g., CLIP-ViT-L, VLMs) have demonstrated strong perception and reasoning capabilities in autonomous driving. However, autonomous driving systems impose strict real-time requirements—control frequency typically needs to reach 20 Hz (50 ms per frame)—whereas state-of-the-art models such as DriveTransformer and DriveAdapter incur latencies of several hundred milliseconds, far from satisfying real-time constraints.

**Dual-system approaches** (inspired by the System-1/System-2 dichotomy in cognitive science) represent a natural solution: a small model handles fast reactive decisions (System 1), while a large model performs deliberate analysis (System 2). However, existing dual-system designs suffer from fundamental limitations:

- **DriveVLM-Dual**: The large VLM runs asynchronously but requires 300 ms, making large-model inference results available only for a small fraction of frames.
- **AD-H**: The LLM outputs high-level planning commands to a smaller model for execution, but the large model similarly cannot respond at every frame.
- **LeapAD**: A memory bank stores large-model inference results for retrieval when needed, but memory bank management is difficult in dynamic environments and generalizes poorly.

**Key Challenge**: In existing dual-system methods, large-model inference results are available only at certain frames, and the system relies solely on the small model most of the time. This means that in frames where the large model is "absent," the system forfeits the performance gains the large model would otherwise provide.

**ETA's Key Insight**: Transfer the dense computation of the large model on the current frame **to preceding time steps**, and process multiple frames simultaneously via **batch inference**—so that large-model features are available at every frame. The core idea is to trade spatial complexity for temporal complexity.

## Method

### Overall Architecture

ETA is an asynchronous dual-system framework comprising four core components: (1) a large model processes preceding frames $\mathbf{I}_{t-\Delta}$ to extract high-quality features $\mathbf{f}_{t-\Delta}^l$; (2) a forecasting module propagates preceding-frame features to the current frame $\hat{\mathbf{f}}_t^l$; (3) a small model processes the current frame $\mathbf{I}_t$ to capture real-time changes that are difficult to predict $\mathbf{f}_t^s$; (4) an action model fuses the dual-stream features and predicts driving actions.

### Key Designs

#### 1. Asynchronous Batch Inference — Temporal Shifting

- **Function**: Defers large-model computation on the current frame to preceding time steps, and makes large-model features available at every frame by batch-processing multiple frames simultaneously.
- **Mechanism**: In conventional dual-system approaches, the large model runs at low frequency while the small model runs at high frequency, resulting in small-model-only outputs for most frames. ETA has the large model process preceding frame $\mathbf{I}_{t-\Delta}$ at time step $t-\Delta$:
  $\mathbf{f}_{t-\Delta}^l = f_{\text{large}}(\mathbf{I}_{t-\Delta})$

  Large-model inference for multiple frames is then parallelized (simultaneously inferring future multi-frame features corresponding to $\mathbf{I}_{t-\Delta}$), ensuring that relevant large-model features are available at every frame.
- **Design Motivation**: The temporal continuity of autonomous driving scenarios implies that visual features of adjacent frames are highly correlated. Propagating preceding-frame features to the current frame incurs some information loss, but is far preferable to having no large-model features at all. Batch inference further amortizes the per-frame cost of the large model.

#### 2. Feature Forecasting Module

- **Function**: Predicts current-frame large-model features based on large-model features extracted from preceding frames.
- **Mechanism**: A forecasting model $f_{\text{forecast}}$ takes as input the preceding-frame features $\mathbf{f}_{t-\Delta}^l$, predicted actions $\hat{\mathbf{a}}_{t-\Delta}$, and conditional inputs $\mathbf{c}_{t-\Delta}$ (velocity and target waypoints):
  $\hat{\mathbf{f}}_t^l = f_{\text{forecast}}(\mathbf{f}_{t-\Delta}^l, \hat{\mathbf{a}}_{t-\Delta}, \mathbf{c}_{t-\Delta})$

  The predicted features are supervised via a forecasting loss $\mathcal{L}_{\text{forecast}} = |\mathbf{f}_t^l - \hat{\mathbf{f}}_t^l|$ (ground-truth large-model features $\mathbf{f}_t^l$ are used during training only and are not required at inference).
- **Design Motivation**: This follows a world-model-like paradigm—predicting future states from previous states and actions. By conditioning on actions and velocity, the model learns to reason about how features should change (e.g., "if the preceding frame involved a left turn at a given speed, how should the current-frame features look?"). The supervision signal is needed only during training and does not affect inference speed.

#### 3. Small Model — Real-Time Complement

- **Function**: A lightweight model processes the current frame to capture abrupt changes that the forecasting module cannot anticipate.
- **Mechanism**: $\mathbf{f}_t^s = f_{\text{small}}(\mathbf{I}_t)$. The first 8 layers of CLIP-ViT-L are used as the small model (reusing the shallow weights of the large model), which is sufficient for extracting basic visual features and runs at high speed (occupying only a small fraction of the full model's inference time).
- **Design Motivation**: Certain critical changes cannot be predicted from preceding frames—such as traffic light state changes or pedestrians appearing suddenly. The small model directly processes the current frame to capture these unpredictable real-time changes, complementing the forecasted large-model features.

#### 4. Action Mask

- **Function**: Guides the model to focus on image regions relevant to the driving action.
- **Mechanism**: Attention between patch features and action-encoding queries is computed to generate a mask $\hat{\mathbf{m}}_t$ that marks image regions toward which the ego vehicle is about to travel. Ground-truth masks are obtained by projecting expert actions onto the image—patches containing trajectory points or waypoints are labeled 1, and the rest are labeled 0. Binary cross-entropy (BCE) loss is used for supervision.
- **Design Motivation**: The model was observed to struggle with focusing on action-relevant image regions. The mask mechanism explicitly guides attention toward image patches along the planned path and waypoints, promoting alignment between observation features and predicted actions in the feature space.

### Loss & Training

- **Base model loss**: $\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{action}} + \frac{1}{16}\mathcal{L}_{\text{mask}}$
- **Async model loss**: $\mathcal{L}_{\text{async}} = \mathcal{L}_{\text{action}} + \frac{1}{16}\mathcal{L}_{\text{mask}} + 0.5 \mathcal{L}_{\text{forecast}}$
- Action loss uses L1 loss, predicting residuals rather than absolute positions (for training stability).
- A stop-gradient operation is applied to ground-truth features in the forecasting loss to prevent collapse.
- The large model is CLIP-ViT-L-336px; the small model uses its first 8 layers; the action decoder is a 12-layer Transformer.
- Training is conducted on 8×A100 GPUs for 40 epochs with $\Delta=0.5\text{s}$.

## Key Experimental Results

### Main Results — Bench2Drive Leaderboard

| Method | Driving Score (DS)↑ | Success Rate (SR)↑ | Efficiency↑ | Latency (ms)↓ |
|--------|--------------------|--------------------|-------------|---------------|
| AD-MLP (sanity check) | 18.05 | 0.00% | 48.45 | **3** |
| UniAD-Base | 45.81 | 16.36% | 129.21 | 663 |
| VAD | 42.35 | 15.00% | 157.94 | 278 |
| DriveTransformer-L | 63.46 | 35.01% | 100.64 | 212 |
| DriveAdapter | 64.22 | 33.08% | 70.22 | 931 |
| **ETA Base** | **74.33** | **48.33%** | **186.04** | 102 |
| **ETA Async** | **69.53** | **38.64%** | **184.51** | **50** |

The Base model achieves across-the-board best performance (DS 74.33) at 102 ms latency, but does not satisfy real-time requirements. The Async model reduces latency to 50 ms (20 Hz) while still achieving a DS of 69.53, surpassing the previous best DriveTransformer by 63.46 (+8%).

### Ablation Study

| Configuration | DS↑ | SR↑ | Latency↓ | Note |
|---------------|-----|-----|----------|------|
| Async (full model) | 69.53 | 38.64% | 50 ms | All components |
| w/o forecasting module | 54.92 | 30.45% | 50 ms | DS drops 21%; forecasting is critical |
| w/o small model | 42.49 | 17.27% | 31 ms | DS drops 39%; real-time information is indispensable |
| w/o action mask | 42.67 | 17.27% | 50 ms | DS drops 39%; mask is critical for attention alignment |
| Small model only | 61.30 | 35.00% | 50 ms | No large-model features; DS lower than Async |
| GT feature forecasting (train+test) | 74.12 | 48.18% | 124 ms | Approximate upper bound of Base |

Capability-level ablations show that removing the small model causes the steepest degradation in emergency braking (66.67→16.36) and traffic sign recognition (59.43→30.56)—precisely the scenarios that are unpredictable and require real-time perception.

### Key Findings

- **Synergy between forecasting and the small model is essential**: neither alone is sufficient. The forecasting module provides global understanding (e.g., scene layout, vehicle positions), while the small model supplies real-time details (e.g., traffic light states, suddenly appearing pedestrians).
- **Experiment (F) with GT feature forecasting** reveals that providing GT features at inference time without supervising the forecasting module with GT during training actually degrades performance to 60.72, demonstrating that the forecasting loss is critical for learning effective feature propagation.
- **Pareto-optimal latency–performance trade-off**: ETA Async lies on the Pareto frontier in the DS–latency plane; no other method achieves a DS of 69.53 at 50 ms latency.

## Highlights & Insights

- **Paradigm innovation through "thinking ahead"**: Shifting large-model computation from the current frame to preceding frames is an elegant strategy of trading time complexity for spatial complexity. This differs fundamentally from model compression or distillation—the full large model is retained; only its usage pattern changes.
- **Rigorous experimental design**: Experiments are conducted under the highly challenging closed-loop Bench2Drive evaluation; all models report the mean and standard deviation over three runs; ablation studies comprehensively cover individual components and capability dimensions.
- **Action mask** is a simple yet effective design: by projecting expert actions as binary masks in image space, it provides explicit spatial attention guidance for the end-to-end model.

## Limitations & Future Work

- The forecasting module assumes a fixed time interval of $\Delta=0.5\text{s}$; longer intervals may substantially degrade forecasting accuracy.
- In scenarios with abrupt changes (e.g., sudden vehicle cut-ins), the forecasting module may produce severely erroneous feature estimates.
- The small model reuses the first 8 layers of the large model, introducing parameter redundancy and precluding independent optimization.
- Evaluation is conducted exclusively in the CARLA simulation environment; validation on real-world data is absent.
- The current framework uses single-camera input; extension to multi-view settings has not been explored.

## Related Work & Insights

- ETA follows the same spirit as DriveVLM-Dual, but achieves large-model availability at every frame through feature forecasting rather than sparse sampling of large-model outputs.
- The forecasting module conceptually relates to world models (e.g., GAIA-1, DriveDreamer)—it effectively learns a lightweight latent-space world model.
- The framework generalizes to any embodied AI system requiring real-time deployment of large models (e.g., robotic manipulation, UAV navigation), provided the scene exhibits temporal continuity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving](../../NeurIPS2025/autonomous_driving/futuresightdrive_thinking_visually_with_spatiotemporal_cot_f.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)
- [\[ICCV 2025\] AdaDrive: Self-Adaptive Slow-Fast System for Language-Grounded Autonomous Driving](adadrive_self-adaptive_slow-fast_system_for_language-grounded_autonomous_driving.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

</div>

<!-- RELATED:END -->
