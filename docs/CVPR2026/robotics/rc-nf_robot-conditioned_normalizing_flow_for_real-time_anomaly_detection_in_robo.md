---
title: >-
  [Paper Note] RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation
description: >-
  [CVPR2026][Robotics][Anomaly Detection] This paper proposes Robot-Conditioned Normalizing Flow (RC-NF), which models the joint distribution of robot states and object motion trajectories via a conditional normalizing flo…
tags:
  - "CVPR2026"
  - "Robotics"
  - "Anomaly Detection"
  - "Normalizing Flow"
  - "VLA Monitoring"
  - "Robotic Manipulation"
  - "Out-of-Distribution"
date: 2026-05-08
content_hash: fb281f0a38ba6964
---

# RC-NF: Robot-Conditioned Normalizing Flow for Real-Time Anomaly Detection in Robotic Manipulation

**Conference**: CVPR2026
**arXiv**: [2603.11106](https://arxiv.org/abs/2603.11106)  
**Code**: None  
**Area**: Robotics
**Keywords**: Anomaly Detection, Normalizing Flow, VLA Monitoring, Robotic Manipulation, Out-of-Distribution

## TL;DR

This paper proposes Robot-Conditioned Normalizing Flow (RC-NF), which models the joint distribution of robot states and object motion trajectories via a conditional normalizing flow, enabling real-time anomaly detection at <100ms latency. RC-NF serves as a plug-and-play monitoring module for VLA models (e.g., π₀), supporting task-level replanning and state-level trajectory rollback (homing).

## Background & Motivation

VLA (Vision-Language-Action) models learn from expert demonstrations via imitation learning, mapping natural language instructions to low-level control actions. However, real-world deployment faces severe OOD (Out-of-Distribution) challenges:

**Task-level OOD**: Environmental changes render the current instruction inapplicable (e.g., the drawer closes unexpectedly during "place the ball into the drawer").

**State-level OOD**: The instruction remains valid but the robot's physical state deviates from the training distribution (e.g., an object slips from the gripper).

Limitations of existing runtime monitoring approaches:

- **State classification methods** (behavior trees, etc.): Rely on exhaustive enumeration of anomaly conditions or manually defined preconditions, making it difficult to cover the combinatorial variability in real manipulation.
- **VLM reasoning methods** (e.g., Sentinel's dual-system architecture): Require chain-of-thought reasoning, incurring second-level latency that precludes timely intervention.
- **FailDetect** (unsupervised flow matching): Directly concatenates image features with robot states, leaving room for improvement in feature selection and processing.

**Core Motivation**: A positive-only trained, real-time (<100ms), plug-and-play anomaly detection module is needed — one that requires neither enumerating all anomaly types nor multi-step reasoning.

## Method

### Overall Architecture

RC-NF is built on the Glow normalizing flow architecture. Its key contribution is a novel affine coupling layer — **RCPQNet (Robot-Conditioned Point Query Network)** — which injects robot state and task information as conditions into the normalizing flow.

The overall pipeline consists of three stages:

1. **Input Processing**: SAM2 extracts object segmentation masks from the video stream → grid-sampled into point sets; task prompts are encoded as uniformly distributed vectors on a hypersphere; robot proprioception provides joint/gripper/pose states.
2. **RC-NF Inference**: Conditioned on robot states and task embeddings, K=12 invertible transformation steps compute the probability density of the current configuration under the normal task distribution; the negative log-likelihood serves as the anomaly score.
3. **Anomaly Detection and Response**: When the anomaly score exceeds a threshold, corrective action is triggered — task-level OOD triggers replanning; state-level OOD triggers trajectory rollback (homing).

### Key Designs

#### Conditional Normalizing Flow

RC-NF extends a standard normalizing flow to a conditional form, with condition $c = (s, \tau)$, where $s$ denotes the robot state (T-dimensional joint states, gripper states, Cartesian pose) and $\tau$ is the task embedding. The task embedding is obtained by mapping task prompts to T-dimensional vectors on the surface of a hypersphere, where the uniform distribution ensures maximal separation between task embeddings.

The point set $\mathcal{X}$ is mapped to a Gaussian latent distribution $\mathcal{Z} \sim \mathcal{N}(\mu_{\text{task}}, I)$, with mean $\mu_{\text{task}}$ broadcast from the task embedding. The conditional likelihood is computed as:

$$\log p_{X|C}(x|c) = \log p_{Z|C}(z|c) + \sum_{i=1}^{K} \log \left| \det \frac{\partial f_{i,c}(y_{i-1})}{\partial y_{i-1}} \right|$$

#### RCPQNet: Robot-Conditioned Point Query Network

RCPQNet serves as the affine coupling layer and contains two core components:

**Task-aware Robot-Conditioned Query (query generation)**: Robot states are projected into the latent space via a linear layer, then modulated by a FiLM mechanism using the task embedding $\tau$, producing task-aware query tokens. These query tokens jointly encode robot state context and task objectives.

**Dual-Branch Point Feature Encoding (memory generation)**:
- **Dynamic Shape Branch**: Centers and normalizes each frame's point set to remove translation and scale effects, extracting shape features; all object point sets are treated as a unified whole, with shape changes representing relative motion among target objects.
- **Positional Residual Branch**: Compensates for position information lost during shape normalization, preserving average displacement features in robot–object motion.

Each branch is processed as follows: MLP dimensionality expansion → average pooling to obtain frame-level representations → GRU for temporal dependency modeling → Transformer Encoder to generate memory vectors. The query and memory vectors then interact via cross-attention in a Transformer to produce affine transformation parameters $\gamma, \beta$.

#### Core Idea of the Decoupled Design

Unlike FailDetect, which directly concatenates image and robot features, RC-NF treats robot states as queries and object point features as memory, achieving **decoupled yet interactive** feature processing. This avoids feature entanglement and feature imbalance issues.

### Loss & Training

- **Training Objective**: Maximize the conditional log-likelihood of normal demonstrations (Eq. 5), equivalent to minimizing $\frac{1}{2}\|z - \mu_{\text{task}}\|_2^2$ plus the Jacobian determinant term.
- **Positive-only Unsupervised Training**: Only successful demonstration data are required; no anomalous samples are needed.
- **Debiasing**: A debiasing operation is applied during training to ensure temporal smoothness of the anomaly score.
- **Static Threshold**: The upper threshold is estimated from a calibration set as $\text{Upper}_\mathcal{T} = \mu_\mathcal{T} + Q_{1-\alpha}(D_\mathcal{T})$, with $\alpha = 0.05$.
- **Training Setup**: K=12 flow steps, trained for 100 epochs with 50 demonstrations per task.

## Key Experimental Results

### Main Results: LIBERO-Anomaly-10 Benchmark

| Method | Gripper Open AUC | Gripper Open AP | Gripper Slippage AUC | Gripper Slippage AP | Spatial Misalign AUC | Spatial Misalign AP | Avg AUC | Avg AP |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-5 | 0.914 | 0.964 | 0.894 | 0.872 | 0.490 | 0.402 | 0.850 | 0.851 |
| Gemini 2.5 Pro | 0.864 | 0.933 | 0.863 | 0.851 | 0.517 | 0.427 | 0.819 | 0.831 |
| Claude 4.5 | 0.875 | 0.940 | 0.855 | 0.829 | 0.529 | 0.429 | 0.821 | 0.825 |
| FailDetect | 0.788 | 0.903 | 0.667 | 0.693 | 0.656 | 0.582 | 0.718 | 0.770 |
| **RC-NF (Ours)** | **0.931** | **0.978** | **0.920** | **0.918** | **0.968** | **0.959** | **0.931** | **0.949** |

### Ablation Study: RCPQNet Components

| Configuration | Gripper Open AUC | Gripper Slippage AUC | Spatial Misalign AUC | Avg AUC | Avg AP |
|------|:---:|:---:|:---:|:---:|:---:|
| RC-NF (Full) | 0.931 | 0.920 | 0.968 | 0.931 | 0.949 |
| w/o Task Embedding | 0.877 | 0.867 | 0.814 | 0.864 | 0.901 |
| w/o Robot State | 0.633 | 0.744 | 0.893 | 0.715 | 0.840 |
| w/o Pos. Residual Branch | 0.905 | 0.897 | 0.854 | 0.895 | 0.923 |
| w/o Dyn. Shape Branch | 0.767 | 0.776 | 0.102 | 0.684 | 0.790 |

### Key Findings

1. **RC-NF substantially outperforms VLM-based approaches**: On Spatial Misalignment, VLMs degrade to near-random performance (AUC ≈ 0.5), while RC-NF achieves 0.968, demonstrating that trajectory-based density estimation far surpasses semantic reasoning for this task.
2. **The Dynamic Shape branch is critical**: Removing it causes the average AUC to drop from 0.931 to 0.684, with a catastrophic collapse to 0.102 on Spatial Misalignment, confirming that temporal shape evolution is the strongest evidence for anomaly detection.
3. **Robot state conditioning is indispensable**: Removing it causes the Gripper Open AUC to drop sharply from 0.931 to 0.633, since an unclosed gripper does not directly displace objects — the anomaly manifests in the relative motion between the robot and the object.
4. **Real-time performance**: Inference latency on an RTX 3090 is <100ms, far faster than the second-level latency of VLM-based approaches.
5. **Successful sim-to-real transfer**: RC-NF transfers effectively from simulation to physical environments, and in combination with π₀, successfully handles both task-level OOD (drawer closing unexpectedly) and state-level OOD (ball slipping from gripper).

## Highlights & Insights

1. **Elegant decoupled conditioning design**: Treating robot states as queries and object point sets as memory achieves decoupled yet interactive feature processing, avoiding feature entanglement — a fundamental improvement over FailDetect's naive concatenation strategy.
2. **Positive-only training**: Unsupervised training on successful demonstrations alone avoids the difficulty of enumerating anomaly types, better suiting real-world deployment.
3. **Two-level OOD response mechanism**: Distinguishing task-level from state-level OOD and responding accordingly (replanning vs. trajectory rollback) is more granular and practical than a monolithic failure detection approach.
4. **Plug-and-play**: RC-NF operates as a parallel monitoring module without modifying the VLA architecture, making it straightforward to deploy in practice.
5. **Hyperspherical task embedding**: Mapping task prompts to a uniform distribution on a hypersphere ensures maximal inter-task separation, providing a favorable geometric structure for density estimation.

## Limitations & Future Work

1. **Dependency on SAM2 segmentation quality**: The first frame requires a bounding box prompt (obtained via graphics methods in simulation, and Gemini 2.5 Pro in the real world); segmentation failures degrade point set quality.
2. **Per-task training and threshold calibration**: New tasks require re-collecting demonstrations and re-calibrating thresholds, limiting scalability.
3. **Static threshold**: Although debiasing ensures temporal smoothness, a fixed threshold may be insufficiently robust in long-tail distribution scenarios.
4. **Single third-person camera only**: RC-NF relies on a single third-person viewpoint for monitoring; multi-view fusion could further improve performance.
5. **Coarse anomaly categorization**: Only task-level and state-level OOD are distinguished, without finer-grained anomaly typing to guide specific recovery strategies.
6. **Limited scale of LIBERO-Anomaly-10**: The benchmark covers only 10 tasks and 3 anomaly types; larger-scale and more diverse benchmarks are warranted.

## Related Work & Insights

- **FailDetect**: The most direct baseline, also a flow-based unsupervised method but relying on naive feature concatenation; the decoupled conditioning design in RC-NF is the core differentiator.
- **Sentinel / VLM monitoring**: VLMs excel at semantic understanding but struggle with spatial reasoning and incur high latency, underscoring the importance of low-level geometric and trajectory features for manipulation anomaly detection.
- **Pedestrian anomaly detection methods**: RC-NF's normalizing flow approach is inspired by pedestrian anomaly detection and adapted to robotic manipulation.
- **VLA models (e.g., π₀)**: RC-NF is positioned as an auxiliary monitoring module for VLAs — augmenting rather than replacing them.
- Insight: **Decoupled design + probabilistic density estimation** may generalize to other robotic tasks requiring real-time monitoring (e.g., navigation, multi-arm collaboration).

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying conditional normalizing flows to robot anomaly detection is a novel combination; the RCPQNet decoupled design has both engineering and academic value.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative evaluation (simulation benchmark, multi-baseline comparison, ablation) and qualitative validation (real-world π₀ integration) are both thorough; ablation analysis is in-depth.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, method description is complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core pain point in VLA deployment; the plug-and-play design is highly practical; <100ms latency meets real-time requirements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Real-Time Robot Execution with Masked Action Chunking](../../ICLR2026/robotics/real-time_robot_execution_with_masked_action_chunking.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
