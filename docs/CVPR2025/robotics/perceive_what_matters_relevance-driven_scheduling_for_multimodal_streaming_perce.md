---
title: >-
  [Paper Note] Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception
description: >-
  [CVPR2025][Robotics][perception scheduling] Proposes a perception scheduling framework for human-robot collaboration that selectively activates perception modules (object detection/pose estimation) based on the trade-off between information gain and computational cost. Under streaming perception scenarios, it reduces computational latency by up to 27.52% while improving MMPose activation recall by 72.73%.
tags:
  - "CVPR2025"
  - "Robotics"
  - "perception scheduling"
  - "human-robot collaboration"
  - "streaming perception"
  - "information theory"
  - "multimodal perception"
date: 2026-05-08
content_hash: b6d36e234accbf3a
---

# Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception

**Conference**: CVPR2025  
**arXiv**: [2603.13176](https://arxiv.org/abs/2603.13176)  
**Code**: None  
**Area**: Robotics  
**Keywords**: perception scheduling, human-robot collaboration, streaming perception, information theory, multimodal perception

## TL;DR
Proposes a perception scheduling framework for human-robot collaboration that selectively activates perception modules (object detection/pose estimation) based on the trade-off between information gain and computational cost. Under streaming perception scenarios, it reduces computational latency by up to 27.52% while improving MMPose activation recall by 72.73%.

## Background & Motivation

### Core Problem

**Key Challenge**: **Background**: 1. Human-robot collaboration (HRC) requires continuous execution of multiple perception modules to achieve accurate scene understanding.
2. Activating all modules frame-by-frame guarantees offline perception quality, but accumulates latency in streaming scenarios, leading to performance degradation.
3. Existing parallel perception pipelines periodically activate modules based on readiness (rather than information needs), which may cause heavy modules to miss critical frames.
4. Critical frame methods typically require the full video sequence, and their selection criteria do not align with the information needs of the perception system.
5. Efficient perception research mainly focuses on frame-level optimization (resolution/configuration) without considering module-level activation necessity.
6. Adaptive sampling and sensor scheduling assume homogeneous information sources, making them inapplicable to perception modules that produce different types of outputs.

## Method

### Overall Architecture
The perception scheduling framework consists of four phases: (1) perception region segmentation $\rightarrow$ (2) perception reward estimation $\rightarrow$ (3) perception module selector $\rightarrow$ (4) result feedback to the Relevance framework. For each frame, the information gain of each module is estimated using the output of the previous frame to select the optimal activation set.

### Key Designs

**1. Perception Region Segmentation**
- Divides the scene into three types of regions: background, objects, and humans.
- Assigns two attributes to each region: motion state (detected via frame-differencing) and relevance (predicted by the Relevance framework).
- Motion detection: Grayscale difference $\Delta L^i = Y \cdot \Delta P^i$; if the change ratio $\text{CR}^i$ exceeds the threshold $\epsilon$, it is classified as motion.

**2. Perception Reward Estimation**
- General reward formulation: $\rho_k^j = \Phi_R(S_k, m^j) - C^j$ (Information Gain - Computational Cost Penalty)

**Object Detection Reward**:
- Scene composition change detection (frame-differencing + color histogram Chi-Square distance $D_H$).
- State update information gain of tracked objects: entropy reduction based on Kalman filter prediction covariance.
- $G_2^{\text{yolo}}[k] = \sum_{p=1}^{n} \frac{1}{2} r_k^p \log\left(\frac{\det(H \bar{\mathcal{P}}_k^p H^\top)}{\det(\mathcal{R})}\right)$
- Weighted by relevance $r_k^p$.

**Pose Estimation Reward**:
- Pre-execution uncertainty: Entropy of each keypoint within the bounding box assuming a uniform distribution $\mathcal{H}_k^{\text{pre}} = D \sum_{s=1}^{N} r^s \ln[(w_k^s + \sigma_w^s)(h_k^s + \sigma_h^s)]$.
- Post-execution uncertainty: Extrapolated confidence score, estimating standard deviation via negative log mapping $\sigma_k^d = -\sigma_{base}^d \log(\hat{s}_k^d)$.
- Information gain: $G_2^{\text{pose}}[k] = \mathcal{H}_k^{\text{pre}} - \mathcal{H}_k^{\text{pose}}$.

**3. Perception Module Selector**
- Maximizes cumulative reward: $\pi^*[k] = \arg\max_{a \in \mathcal{A}} \sum_j a^j \cdot \rho_k^j$.
- The reward of each module is estimated independently, simplifying to independent decision-making per module: activate if $\rho_k^j > 0$.
- Supports preset activation indicators $G_1^j[k]$ (e.g., forcing YOLO activation when scene composition changes).

### Loss & Training
No training loss—purely inference-time framework, utilizing information-theory-based online decision making.

## Key Experimental Results

### Main Results

| Domain | Method | Latency (ms) | YOLO Recall | Pose Recall |
|----|------|---------|-------------|-------------|
| Indoor Reading | Parallel | 98.81 | 1.00 | 0.16 |
| Indoor Reading | **Scheduled** | **71.62** | 0.97 | 0.20 |
| Eating | Parallel | 94.99 | 1.00 | 0.16 |
| Eating | **Scheduled** | **86.44** | 0.98 | 0.20 |
| Walking | Parallel | 93.63 | 1.00 | 0.22 |
| Walking | **Scheduled** | **75.15** | 0.93 | 0.38 |

- Latency reduced by up to 27.52% (Indoor Reading).
- MMPose activation recall increased by up to 72.73% (Walking: 0.22 $\rightarrow$ 0.38).

### Keyframe Identification Accuracy

| Domain | YOLO Keyframe Accuracy | MMPose Keyframe Accuracy |
|----|----------------|-----------------|
| Indoor Reading | 0.97 | 0.89 |
| Eating | 0.98 | 0.97 |
| Walking | 0.93 | 0.92 |

### Key Findings
- For static scenes (Reading), latency reduction is most significant; low information gain leads to conservative scheduling.
- For dynamic scenes (Walking), MMPose recall shows the largest improvement, because critical frames are more dense.
- YOLO recall decreases only marginally (0.93-0.98); the efficiency gain is well worth this cost.
- Overall recall for MMPose is still relatively low, primarily because inference latency causes intermediate important frames to be skipped.

## Highlights & Insights
1. **First to propose the concept of perception scheduling**: Systematizes module activation decisions as an optimization problem of information gain vs. computational cost.
2. **Solid information-theoretic foundation**: Models module-level information gain using Kalman filter prediction covariance entropy and keypoint confidence entropy.
3. **Scalable modules**: The framework is designed as a general Perception Toolkit, extensible to more modules such as VLMs.
4. **Relevance-aware**: Task-relevance weighting provided by the Relevance framework ensures resource allocation to important areas.
5. **Practical and lightweight**: The scheduling logic runs on the CPU, consuming no GPU resources.

## Limitations & Future Work
1. MMPose recall absolute value is still very low (0.20-0.38); the framework cannot fundamentally solve the inference latency of heavy modules.
2. Experiments are only validated on 3 self-recorded videos, lacking standard datasets and large-scale evaluations.
3. The Lagrange multiplier $\lambda$ in the reward needs to be manually tuned, and the optimal value may vary across different scenes.
4. Motion detection relies on a simple frame-differencing method, which can cause false triggers in scenes with sudden lighting changes.
5. It does not consider resource contention between modules and scheduling constraints of shared GPUs.

## Related Work & Insights
- The Relevance concept mimics the human Reticular Activating System (RAS), providing a task-aware foundation for perception scheduling.
- Difference from adaptive sampling: Object detection and pose estimation yield outputs of different qualities, which are not homogeneous measurements.
- The modeling approach for information gain (Kalman + keypoint confidence) can be generalized to other perception modules.
- Transitioning this to super-large modules such as VLMs is particularly valuable, as their inference cost is much higher.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (First systematic perception scheduling framework)
- **Experimental Thoroughness**: ⭐⭐⭐ (Only 3 self-recorded videos, lacks standard benchmarks)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear information-theoretic modeling)
- **Value**: ⭐⭐⭐⭐ (Important direction, framework has extensibility potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](../../AAAI2026/robotics/touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[NeurIPS 2025\] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies](../../NeurIPS2025/robotics/act_to_see_see_to_act_diffusion-driven_perception-action_interplay_for_adaptive_.md)
- [\[CVPR 2025\] Decision SpikeFormer: Spike-Driven Transformer for Decision Making](decision_spikeformer_spike-driven_transformer_for_decision_making.md)
- [\[CVPR 2025\] Magma: A Foundation Model for Multimodal AI Agents](magma_a_foundation_model_for_multimodal_ai_agents.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)

</div>

<!-- RELATED:END -->
