---
title: >-
  [Paper Note] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation
description: >-
  [CVPR 2026][Robotics][Normalizing Flow] This paper proposes RC-NF, a real-time anomaly detection model based on conditional normalizing flows that decouples the processing of robot state and object trajectory features. Trained in an unsupervised manner using only normal demonstrations, RC-NF detects OOD anomalies during VLA model execution within 100ms, outperforming state-of-the-art methods (including VLM baselines such as GPT-5 and Gemini 2.5 Pro) by approximately 8% AUC and 10% AP on LIBERO-Anomaly-10.
tags:
  - CVPR 2026
  - Robotics
  - Normalizing Flow
  - Robotic Anomaly Detection
  - VLA Models
  - OOD Detection
  - Real-Time Monitoring
date: 2026-05-08
content_hash: cfc7f84f1d20a935
---

# RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

**Conference**: CVPR 2026
**arXiv**: [2603.11106](https://arxiv.org/abs/2603.11106)
**Code**: [https://heikaishuizz.github.io/RC-NF/](https://heikaishuizz.github.io/RC-NF/)
**Area**: Embodied Intelligence / Robotic Manipulation / Anomaly Detection
**Keywords**: Normalizing Flow, Robotic Anomaly Detection, VLA Models, OOD Detection, Real-Time Monitoring

## TL;DR

This paper proposes RC-NF, a real-time anomaly detection model based on conditional normalizing flows that decouples the processing of robot state and object trajectory features. Trained in an unsupervised manner using only normal demonstrations, RC-NF detects OOD anomalies during VLA model execution within 100ms, outperforming state-of-the-art methods (including VLM baselines such as GPT-5 and Gemini 2.5 Pro) by approximately 8% AUC and 10% AP on LIBERO-Anomaly-10.

## Background & Motivation

Vision-Language-Action (VLA) models trained on expert demonstration data via imitation learning are capable of executing complex manipulation tasks. However, they frequently encounter OOD scenarios when deployed in dynamic real-world environments, leading to task failures. Existing runtime monitoring approaches suffer from two key limitations: (1) **state classification methods** (e.g., behavior trees) require exhaustive enumeration of anomalous conditions or manual precondition definitions, making them ill-suited for combinatorial variations in practice; (2) **VLM-based methods** (e.g., Sentinel using GPT-5/Gemini for inference) involve multi-step reasoning with latencies on the order of seconds, precluding timely intervention. There is therefore a need for a lightweight monitor that is both accurate and fast (<100ms).

## Core Problem

How to design a **plug-and-play** real-time monitoring module for VLA models that, without modifying the VLA architecture: (1) trains exclusively on normal demonstration data (no failure data collection required); (2) determines in real time whether the joint distribution of robot state and object trajectory deviates from the normal task distribution; and (3) supports OOD detection and correction at both the task level and the state level?

## Method

### Overall Architecture

The RC-NF pipeline proceeds as follows: a video stream is processed by SAM2 to extract object segmentation masks, from which point sets are obtained via grid sampling; robot joint/gripper/pose states and task text descriptions are simultaneously acquired. These inputs are fed into a conditional normalizing flow model, which maps them to a Gaussian latent space through $K$ invertible transformations, and computes an anomaly score via negative log-likelihood. When the score exceeds a threshold, either task-level replanning or state-level rollback is triggered.

### Key Designs

1. **Spherical Uniform Encoding for Task Embeddings**: Different task instructions are mapped to uniformly distributed vectors on a $T$-dimensional hypersphere, ensuring maximal separation of task embeddings in the latent space. This design enables the model to detect not only dataset-level OOD but also task-specific anomalies (e.g., the robot moving in the wrong direction under Spatial Misalignment). Ablation results show that removing the task embedding causes AUC for Spatial Misalignment to drop sharply from 0.97 to 0.81.

2. **RCPQNet (Robot-Conditioned Point Query Network)**: This is the core affine coupling layer design in RC-NF. Robot states are linearly projected and modulated together with task embeddings via FiLM to produce a Task-aware Robot-Conditioned Query token. Object point sets are encoded through a dual-branch encoder (a Dynamic Shape branch that performs centroid normalization to extract shape features, and a Positional Residual branch to compensate for positional information), processed by MLP+GRU+Transformer into memory tokens. Cross-attention is then applied to fuse the two, generating the scale and shift parameters of the affine transformation. This decoupled design avoids the feature interference caused by direct concatenation in FailDetect.

3. **Dual-Branch Point Feature Encoding**: The Dynamic Shape branch centers and normalizes each frame's point set to remove translation/scale effects and capture shape variations of the object. The Positional Residual branch retains raw coordinate information to compensate for normalization losses. Both branches model temporal dependencies with GRU before being encoded by a Transformer. Ablation results show that the Dynamic Shape branch is the most critical component (removing it reduces AUC from 0.93 to 0.68), while the Positional Residual branch provides complementary information (removing it reduces AUC to 0.89).

4. **Anomaly Detection & Handling**: At inference time, negative log-likelihood serves as the anomaly score, and the threshold is estimated on a calibration set via conformal prediction (mean + quantile). Upon anomaly detection, two cases are distinguished: **task-level OOD** (environmental changes that invalidate the current instruction, e.g., a drawer already closed) → pause and trigger high-level replanning; **state-level OOD** (task remains valid but trajectory deviates, e.g., object slippage) → activate a homing routine to return to the initial state before resuming execution.

### Loss & Training

- **Training Objective**: Maximize the conditional log-likelihood $\log p_{X|C}(x|c)$, where the Gaussian prior mean is set to the task embedding $\mu_{\text{task}}$
- **BalancedHardSampler Debiasing**: High trajectory similarity during the robot's initial motion phase causes sample imbalance. A staged training strategy is designed: standard sampling is used until NextStageEpoch, after which BalancedHardSampler balances the sample distribution to reduce redundancy.
- Training configuration: $K=12$ flow steps, 100 epochs.

## Key Experimental Results

| Dataset | Anomaly Type | Metric | RC-NF | GPT-5 | FailDetect | Gain (vs. best) |
|--------|----------|------|-------|-------|------------|--------------|
| LIBERO-Anomaly-10 | Gripper Open | AUC | 0.9312 | 0.9137 | 0.7883 | +1.9% |
| LIBERO-Anomaly-10 | Gripper Slippage | AUC | 0.9195 | 0.8941 | 0.6665 | +2.8% |
| LIBERO-Anomaly-10 | Spatial Misalign. | AUC | 0.9676 | 0.5292 | 0.6557 | +31.2% |
| LIBERO-Anomaly-10 | Average | AUC/AP | 0.9309/0.9494 | 0.8500/0.8507 | 0.7181/0.7700 | +8%/+10% |

**Real-Time Performance**: RC-NF achieves a total per-frame inference latency of 86.7ms on an RTX 3090 (SAM2 50ms + grid sampling 1.7ms + RC-NF 30ms + other 5ms), satisfying real-time requirements.

### Ablation Study

- **Task embedding is critical for distinguishing task-specific anomalies**: Removing it causes Spatial Misalignment AUC to drop from 0.97 to 0.81, limiting the model to dataset-level OOD detection only.
- **Robot state is essential for Gripper Open anomalies**: Removing it causes AUC for this category to plummet from 0.93 to 0.63, since an unclosed gripper does not alter object position and the anomaly is only reflected in the relative motion between robot and object.
- **The Dynamic Shape branch contributes the most** (removing it reduces AUC to 0.68); the Positional Residual branch is complementary (removing it reduces AUC to 0.89).
- VLM baselines nearly fail on Spatial Misalignment (AUC ≈ 0.5), demonstrating that spatial reasoning is a weakness of VLMs.

## Highlights & Insights

- **Elegance of the probabilistic framework**: Using normalizing flows for anomaly detection is a natural fit — normal behaviors correspond to high probability density, while anomalies correspond to low density, requiring no negative samples.
- **Sophistication of the decoupled design**: RCPQNet treats robot state as the query and object features as memory, achieving decoupled interaction via cross-attention, which is more effective than the simple concatenation in FailDetect.
- **Dual-granularity anomaly handling**: The distinction between task-level and state-level OOD, and their respective handling strategies (replanning vs. rollback), is highly practical from an engineering standpoint.
- **Plug-and-play compatibility**: RC-NF runs in parallel with VLA models such as π0 without modifying their architecture, representing a more principled design philosophy than VLM-dependent approaches.

## Limitations & Future Work

- The benchmark is built solely on LIBERO-10, with limited task and scene diversity; validation on a broader range of real-world tasks is needed.
- Threshold calibration depends on a calibration dataset, and different tasks may require different $\alpha$ parameters.
- The quality of SAM2's object segmentation affects point set representations; performance may be limited in occlusion or fast-motion scenarios.
- The current framework assumes that anomalies can be recovered via homing or replanning; irreversible anomalies (e.g., object damage) are not addressed.
- The scalability of spherical uniform encoding to larger numbers of tasks has not been verified (currently only 10 tasks).

## Related Work & Insights

- **vs. FailDetect**: Both are flow-based methods, but FailDetect directly concatenates image features and robot states as flow matching inputs, causing feature entanglement and imbalance. RC-NF achieves fundamental improvements in both feature selection (point sets instead of raw images) and processing (decoupled design instead of concatenation) via RCPQNet and cross-attention fusion.
- **vs. VLM-based monitoring (Sentinel + GPT-5/Gemini)**: VLM-based approaches nearly fail on spatial reasoning (Spatial Misalignment AUC ≈ 0.5) and incur multi-step inference latencies on the order of seconds. RC-NF replaces semantic understanding with probabilistic density estimation, achieving superior performance on spatial anomalies (AUC = 0.97) with latency below 100ms.
- **vs. Behavior tree methods**: Behavior trees require explicit design of anomaly conditions and rollback steps, whereas RC-NF learns in an unsupervised manner from normal demonstration data alone.

The conditional probability density estimation approach via normalizing flows can be generalized to other scenarios requiring real-time anomaly detection (e.g., autonomous driving, surgical robotics). The decoupled query-memory cross-attention design has broad reference value for multimodal feature fusion. Point set representations as intermediate object state descriptors are more robust than raw images, a principle applicable to other embodied AI tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Conditional normalizing flows for robot monitoring are not entirely novel, but the RCPQNet decoupled design and dual-branch encoding make substantive contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ A new benchmark is introduced; comparisons against VLM and flow-based baselines are comprehensive; ablations are complete; real-robot validation is included.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Structure is clear, figures are well-crafted, and the motivation–method–experiment narrative is logically coherent.
- **Value**: ⭐⭐⭐⭐ Addresses a critical safety issue in VLA deployment; the plug-and-play design has significant engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation](language-grounded_decoupled_action_representation_for_robotic_manipulation.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)

<!-- RELATED:END -->
