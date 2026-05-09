---
title: >-
  [Paper Note] Language-Grounded Decoupled Action Representation for Robotic Manipulation
description: >-
  [CVPR 2026][Robotics] This paper proposes LaDA, a framework that decouples continuous 7-DoF robotic actions into interpretable, language-described motion primitives (translation, rotation, gripper state), and unifies the visual-language-action representation space via semantically guided soft-label contrastive learning to achieve cross-task generalization.
tags:
  - CVPR 2026
  - Robotics
date: 2026-05-08
content_hash: 29f0a0411cfc6a72
---

# Language-Grounded Decoupled Action Representation for Robotic Manipulation

**Conference**: CVPR 2026
**arXiv**: [2603.12967](https://arxiv.org/abs/2603.12967)
**Code**: None
**Area**: Robotics

## TL;DR

This paper proposes LaDA, a framework that decouples continuous 7-DoF robotic actions into interpretable, language-described motion primitives (translation, rotation, gripper state), and unifies the visual-language-action representation space via semantically guided soft-label contrastive learning to achieve cross-task generalization.

## Background & Motivation

1. **Heterogeneity gap between high-level semantics and low-level control**: Current VLA models lack an effective bridge between visual-language understanding and fine-grained action control; high-level semantic instructions (e.g., "pour water") are difficult to map directly to precise motion parameters.
2. **Unexploited shared motion primitives**: Tasks with distinct semantics (e.g., "pour water" and "place bottle") often share underlying motion primitives (reaching, grasping, rotating), yet existing models cannot reuse these shared structures, leading to redundant learning and poor cross-task generalization.
3. **Inherent limitations of existing paradigms**:
   - End-to-end VLAs: perception and control are tightly coupled, lacking interpretability and motion structure reuse.
   - Implicit action learning: latent spaces are defined by visual differences, lacking explicit semantics, which limits cross-task transfer.
   - Language-conditioned policies: rely on coarse-grained discrete primitives (e.g., "move forward") and lack fine-grained motion parameters (translation magnitude, rotation angle).
4. **Absence of a semantic grounding layer**: The root cause is the lack of a semantic grounding layer between symbolic intent and continuous execution—a role that language is naturally suited to fulfill.

## Method

### 3.1 Overall Architecture

LaDA uses language as a semantic bridge to unify visual, linguistic, and action representations in a shared embedding space. The core pipeline consists of: action decomposition → semantic contrastive learning → adaptive weighting → fine-tuning and inference.

### 3.2 Language-Grounded Action Decomposition

Each 7-DoF end-effector action $\mathbf{a}_t$ is projected into three categories of interpretable motion primitives via $\Pi: \mathbf{a}_t \mapsto \mathbf{p}_t$:

| Primitive Type | Symbol | Example Language Template |
|---|---|---|
| Translation Primitive | $\Delta T$ | "Move [dist] meters along [dir]" |
| Rotation Primitive | $\Delta R$ | "Rotate [mag] degrees around [axis]" |
| Gripper Primitive | $G$ | "Open" / "Close" |

Each primitive is discretized into language-aligned categories, converting continuous control trajectories into interpretable semantic classes. This decomposition bridges low-level kinematics and high-level semantics, enabling cross-task alignment and compositional generalization.

### 3.3 Semantically Guided Contrastive Learning

#### Soft-Label Similarity Construction

A soft-label similarity matrix $S \in [0,1]^{N \times N}$ is constructed to encode primitive-level semantic affinity:

$$S = \frac{w_t M_t + w_r M_r + w_g M_g}{w_t + w_r + w_g}$$

where $M_t$, $M_r$, and $M_g$ are binary matching matrices for translation, rotation, and gripper primitives, respectively, and $(w_t, w_r, w_g)$ are hyperparameters. Each entry $S_{ij}$ represents the fine-grained primitive-level semantic similarity between actions $i$ and $j$.

#### Dual-Path Soft-Label Contrastive Learning

Visual tokens $v_i = f_v(V_i)$ and language tokens $l_i = f_l(L_i)$ are extracted using a pretrained CLIP encoder, fused via FiLM, and projected by an MLP: $A_i = \text{MLP}(\text{FiLM}(v_i, l_i))$.

**Path 1: Action–Action Alignment**, which brings actions sharing primitive attributes closer in the embedding space:

$$\mathcal{L}_a = -\sum_{i=1}^N \sum_{j=1}^N S_{ij} \log \frac{\exp(\text{sim}(A_i, A_j) / \tau)}{\sum_{k=1}^N \exp(\text{sim}(A_i, A_k) / \tau)}$$

**Path 2: Action–Primitive Alignment**, which anchors each action to its primitive language description $P_j = f_l(\mathcal{D}(p_j))$:

$$\mathcal{L}_m = -\sum_{i=1}^N \sum_{j=1}^N S_{ij} \log \frac{\exp(\text{sim}(A_i, P_j) / \tau)}{\sum_{k=1}^N \exp(\text{sim}(A_i, P_k) / \tau)}$$

Total contrastive loss: $\mathcal{L}_{\text{CL}} = \mathcal{L}_a + \lambda \mathcal{L}_m$

### 3.4 Adaptive Loss Weighting

The imitation learning loss $\mathcal{L}_{\text{IL}}$ (predicting discretized primitive categories) and the contrastive loss $\mathcal{L}_{\text{CL}}$ exhibit different convergence characteristics. A moving-average-based adaptive weighting scheme is employed:

$$w_{\text{IL}} = \frac{\text{MA}(\mathcal{L}_{\text{IL}})}{\text{MA}(\mathcal{L}_{\text{IL}}) + \text{MA}(\mathcal{L}_{\text{CL}})}, \quad w_{\text{CL}} = \frac{\text{MA}(\mathcal{L}_{\text{CL}})}{\text{MA}(\mathcal{L}_{\text{IL}}) + \text{MA}(\mathcal{L}_{\text{CL}})}$$

Final objective: $\mathcal{L}_{total} = w_{\text{CL}} \mathcal{L}_{\text{CL}} + w_{\text{IL}} \mathcal{L}_{\text{IL}}$

### 3.5 Fine-Tuning and Inference

After pretraining, a lightweight MLP action head is used for fine-tuning on 7-DoF action prediction (via $\mathcal{L}_1$ trajectory regression loss). At inference time, continuous actions are generated directly from $(V_t, L_t)$ without requiring explicit primitive labels.

## Key Experimental Results

### Pretraining Data

The Open X-Embodiment (OXE) dataset is used, comprising approximately 22.5 million visual frames across 22 robot embodiments, with each action represented as a 7-DoF control vector.

### LIBERO Benchmark

| Model | Params | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|---|
| UniACT | 0.5B | 65.0 | 78.0 | 68.0 | 47.0 | 64.5 |
| OpenVLA | 7.5B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π-FAST | 2B | 96.4 | 96.8 | 88.6 | 60.2 | 85.5 |
| FlowVLA | 8.5B | 93.2 | 95.0 | 91.6 | 72.6 | 88.1 |
| CLIP-RT | 1.3B | 95.2 | 99.2 | 94.2 | 83.8 | 93.1 |
| **LaDA** | **0.6B** | **95.2** | **99.2** | **93.6** | **86.4** | **93.6** |

With only 0.6B parameters, LaDA achieves a state-of-the-art average success rate of 93.6%, with particularly strong performance on LIBERO-Long (86.4%), surpassing models with several times more parameters.

### MimicGen Benchmark

| Model | C_D0 | C_D1 | S_D0 | S_D1 | ST_D0 | ST_D1 | T_D0 | TPA_D0 | TPA_D1 | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|
| OpenVLA | 42% | 18% | 84% | 86% | 36% | 20% | 20% | 28% | 8% | 38% |
| Phoenix | 94% | 48% | 96% | 86% | 50% | 20% | 68% | 52% | 6% | 58% |
| CLIP-RT* | 77% | 34% | 93% | 87% | 68% | 52% | 32% | 11% | 4% | 51% |
| **LaDA** | **94%** | **46%** | **96%** | **95%** | **76%** | **71%** | **48%** | **50%** | **25%** | **67%** |

LaDA achieves an average success rate of 67%, outperforming Phoenix by 9% and CLIP-RT* by 16%, with particularly pronounced advantages on long-horizon tasks (StackThree_D1: 71%).

### Ablation Study

| Method | Spatial | Object | Goal | Long | Avg. |
|---|---|---|---|---|---|
| w/o SCL | 79.2 | 82.8 | 76.6 | 63.4 | 75.5 |
| w/o AW | 93.6 | 94.4 | 87.2 | 74.4 | 87.4 |
| **LaDA** | **95.2** | **99.2** | **93.6** | **86.4** | **93.6** |

Removing soft-label contrastive learning (SCL) causes an 18.1% performance drop, validating the critical role of fine-grained semantic alignment. Removing adaptive weighting (AW) reduces performance by 6.2%.

### Generalization

In cross-task generalization tests, CLIP-RT* achieves 0% success on unseen "push" instructions, whereas LaDA reaches 12.3%. In multi-task training settings, LaDA benefits substantially from joint multi-task training, while CLIP-RT shows only marginal improvement.

## Highlights & Insights

- **Novel perspective of language as a semantic bridge**: Language is elevated from a task instruction signal to a general-purpose interface connecting perception and control, enabling semantic grounding of actions.
- **Fine-grained interpretable primitives**: Unlike coarse-grained "move forward" primitives, LaDA's primitives encode precise motion parameters (distance, angle), achieving genuine semantic-to-control alignment.
- **Exceptional parameter efficiency**: With 0.6B parameters, LaDA surpasses large models exceeding 7B parameters (OpenVLA, CoT-VLA), demonstrating a highly favorable performance-to-parameter ratio.
- **Soft-label contrastive learning**: Overcomes the limitations of hard positive/negative sample pairs by capturing fine-grained motion correspondences via continuous affinity weights.
- **Comprehensive multi-benchmark validation**: Evaluation spans LIBERO (language-conditioned multi-task), MimicGen (contact-rich manipulation), and real-robot deployment.

## Limitations & Future Work

- **Reliance on predefined primitive discretization**: The discretization granularity for translation and rotation is a hyperparameter and may not adequately cover all fine-grained motion requirements.
- **Limited scale of real-robot experiments**: Validation is conducted only on a single pick-and-place task, without demonstrating more complex real-robot manipulation scenarios.
- **Dependence on pretraining data**: Pretraining relies on the large-scale OXE dataset, which incurs substantial data acquisition costs.
- **Bounded generalization ceiling**: Although cross-task generalization outperforms baselines, the absolute success rate remains low (12.3%), indicating that zero-shot generalization remains an open problem.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The language-grounded action decomposition paradigm is novel; soft-label contrastive learning for robotic action representation is introduced for the first time.
- ⭐⭐⭐⭐ **Practicality**: Achieving SOTA with 0.6B parameters and strong framework generality make this approach deployment-friendly.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Two simulation benchmarks, real-robot experiments, comprehensive ablations, and generalization tests provide broad coverage.
- ⭐⭐⭐ **Writing Quality**: The structure is clear, though some notation and formulas could be further unified and simplified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models](adaptive_action_chunking_at_inference-time_for_vision-language-action_models.md)

</div>

<!-- RELATED:END -->
