---
title: >-
  [Paper Note] VGGT-DP: Generalizable Robot Control via Vision Foundation Models
description: >-
  [AAAI 2026][3D Vision][Visuomotor Policy] The paper proposes VGGT-DP, a bio-inspired visuomotor policy framework that combines the pre-trained 3D-aware foundation model VGGT as a vision encoder with Diffusion Policy. Through three key designs—frame-wise token reuse, random token pruning, and proprioception-guided vision learning—it significantly outperforms DP and DP3 baselines on high-precision manipulation tasks in MetaWorld.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Visuomotor Policy"
  - "Diffusion Policy"
  - "VGGT"
  - "Proprioception-Guided"
  - "Robot Manipulation"
date: 2026-05-08
content_hash: 9dba2e632ba974d4
---

# VGGT-DP: Generalizable Robot Control via Vision Foundation Models

**Conference**: AAAI 2026  
**arXiv**: [2509.18778](https://arxiv.org/abs/2509.18778)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Visuomotor Policy, Diffusion Policy, VGGT, Proprioception-Guided, Robot Manipulation

## TL;DR

The paper proposes VGGT-DP, a bio-inspired visuomotor policy framework that combines the pre-trained 3D-aware foundation model VGGT as a vision encoder with Diffusion Policy. Through three key designs—frame-wise token reuse, random token pruning, and proprioception-guided vision learning—it significantly outperforms DP and DP3 baselines on high-precision manipulation tasks in MetaWorld.

## Background & Motivation

### From Biological Vision to Robot Perception

Visuomotor policy lies at the core of robot manipulation. Current dominant research paradigms fall into two categories:

**Vision-Action (VA)** paradigm: Small vision encoders + large policy heads, such as Diffusion Policy.

**Vision-Language-Action (VLA)** paradigm: Utilizing large-scale vision-language models to provide language priors to enhance generalization.

However, the authors present a profound biological insight: **many non-verbal organisms also exhibit outstanding manipulation capabilities**. Insects, fruit flies, and even single-celled organisms can perceive, navigate, and manipulate their environments exceptionally well without any language or symbolic reasoning. Biological research shows that animals allocate massive neural resources specifically to **visual processing**.

Therefore, the core problem does not lie in language priors, but in the **capacity and quality of visual representations**. Visual encoders used in current robotic systems are often too simple to capture complex spatial and geometric relationships.

### Why VGGT

VGGT (Visual Geometry Grounded Transformer) is a visual foundation model pre-trained on large-scale 3D reconstruction tasks, which can unifiedly predict camera poses, dense depth maps, 3D point clouds, and visual features. Unlike semantic-focused models such as CLIP/DINOv2, VGGT provides **geometry-aware spatial representations**, making it more suitable for manipulation tasks requiring precise spatial reasoning.

## Method

### Overall Architecture

VGGT-DP consists of three core components:
1. **VGGT Encoder + Token Pruning**: Extracts geometry-aware visual features.
2. **Frame-Wise Token Reuse (FTR)**: Reduces inference latency.
3. **Proprioception-Guided Diffusion Policy**: Integrates visual and proprioceptive signals for action prediction.

### Key Designs

#### 1. **VGGT as a Feature Projector**

Instead of using the low-level visual outputs (depth maps, point clouds) of VGGT, the proposed method leverages the **token outputs of its aggregator**, which provide compact, semantically rich 3D scene representations.

Given input images of $B \cdot T$ frames and $V$ views, VGGT outputs visual tokens:
$$\mathcal{T}_{vggt} = \text{VGGT}_{agg}(\mathcal{I}) \in \mathbb{R}^{B \cdot T \times V \times (N_p+1) \times D}$$

These are then further processed by a Transformer Encoder, and projected to a conditional embedding $\mathcal{C} \in \mathbb{R}^{B \cdot T \times d_c}$ via average pooling and an MLP, acting as the conditioned input for the diffusion policy.

Design Motivation: The VGGT aggregator already fuses spatial and appearance cues from multiple views to generate geometry-aware tokens with global context, which are stronger than raw image features.

#### 2. **Frame-Wise Token Reuse (FTR)**

Existing methods recompute visual embeddings for all observed frames in each inference step, even when frames overlap across temporal windows. For large models like VGGT, this incurs a massive computational overhead.

Core Idea: **Reuse already computed tokens for overlapping frames**. The VGGT features are only computed for the latest frame, while the tokens of older frames are cached on the CPU:
$$\mathcal{T}_{vggt}^{(t)} = \text{Concat}(\mathcal{T}_{cache}^{(t-1)}, \text{VGGT}_{agg}(\mathcal{I}_t))$$

Design Motivation: In the sliding window of robot control, most observed frames are shared between adjacent time steps. FTR reduces the inference overhead from $O(T)$ to $O(1)$.

#### 3. **Random Token Pruning**

Before feeding the VGGT tokens into the Transformer Encoder, a certain ratio $r_{prune}$ of patch tokens is randomly discarded.

Design Motivation: Introducing token-level randomness prevents overfitting, reduces computational load to accelerate inference, and encourages the model to learn representations invariant to partial observational loss.

#### 4. **Proprioception-Guided Vision Learning**

An auxiliary decoder $D$ is designed to predict the robot's proprioceptive state (joint angles + end-effector position) from visual features:
$$\hat{p}_t = D(f_t), \quad \mathcal{L}_{proprio} = \mathbb{E}_t[\|p_t - \hat{p}_t\|^2]$$

Design Motivation: This forces the visual encoder to learn spatial features relevant to manipulation, improving the quality of closed-loop feedback control.

### Loss & Training

- Architecture: 1D U-Net diffusion model, using FiLM conditioning.
- Scheduler: DDIM with 100 training timesteps and 10 inference denoising steps.
- Prediction horizon: 16 steps, observation horizon: 2 steps.
- Optimizer: AdamW, lr=$1 \times 10^{-4}$, weight decay $1 \times 10^{-6}$.
- Trained for 3000 epochs, batch size 128, EMA decay 0.9999.

## Key Experimental Results

### Main Results

**Success Rate (%) on 10 Selected MetaWorld Tasks**:

| Task | DP | DP3 | **VGGT-DP** | Type |
|------|-----|-----|-------------|------|
| Disassemble | 43±7 | **69±4** | 55±2.5 | Simple |
| Peg Unplug Side | 74±3 | **75±5** | 63±6 | Simple |
| Pick out of Hole | 0±0 | 14±9 | **55±6** | Complex Spatial |
| Shelf Place | 11±3 | **17±10**| 10±0 | Placement |
| Reach | 18±2 | 24±1 | **42±8** | Complex Spatial |
| Soccer | 14±4 | 18±3 | **30±7** | Complex Spatial |
| Sweep Into | 10±4 | 15±5 | **44±4** | Complex Spatial |
| Hand Insert | 10±4 | 15±5 | **19±4** | Complex Spatial |
| Pick Place | 0±0 | **12±4** | 0±0 | Placement |
| Stick Pull | 11±2 | 27±8 | **48±5** | Complex Spatial |
| **Average** | 19.1 | 28.6 | **36.6** | — |

VGGT-DP achieves an average success rate of 36.6%, outperforming DP by 17.5 percentage points and DP3 by 8.0 percentage points.

### Ablation Study

**Robustness Test under Viewpoint Perturbations (Stick Pull Task)**:

| Perturbation Angle δ | Success Rate | Description |
|----------------------|--------------|-------------|
| 0° | 39% | Normal viewpoint |
| 5° | 5% | Drastic decrease even with minor perturbation |
| 10° | 0% | Complete failure |
| 15° | 0% | Complete failure |

**Effectiveness of FTR**: FTR significantly reduces inference latency under large batch sizes and long temporal windows, which is highly valuable for deploying large vision models to real-time systems.

### Key Findings

1. **VGGT-DP performs outstandingly on complex spatial reasoning tasks**: Pick out of Hole (0 -> 55%), Sweep Into (10 -> 44%), Stick Pull (11 -> 48%).
2. **No advantage on simple tasks**: Small encoders are sufficient.
3. **Failure in placement tasks**: Fails to locate small, slender, or partially occluded target objects precisely.
4. **Poor viewpoint robustness**: A perturbation of just 5° triggers a drastic drop in success rate (39% -> 5%), indicating severe overfitting to the training camera pose.

## Highlights & Insights

1. **Profound bio-inspired insight**: Bypassing language priors and returning to the essence of visual perception is highly compelling.
2. **Simple yet effective FTR mechanism**: Executing token reuse by leveraging temporal redundancy is useful for deploying large vision models to real-time robotic systems.
3. **Proprioception-guided vision learning**: Utilizing internal robot states as auxiliary supervision inputs to guide visual feature learning holds great promise in embodied AI.

## Limitations & Future Work

1. **Severe lack of viewpoint robustness**: The primary bottleneck. A 5° perturbation causes failure, requiring the introduction of equivariant encoders or viewpoint domain randomization.
2. **Evaluation only in simulation (MetaWorld)**: Lacking real-world robot experiments.
3. **High computational overhead of VGGT**: Large parameter size limits deployment for real-time control.
4. **Failure in placement tasks**: Lacks sufficient support for small objects and fine-grained manipulation.

## Related Work & Insights

- **Diffusion Policy (DP)**: The policy baseline for this paper; VGGT-DP primarily improves the vision encoder.
- **DP3**: Utilizes 3D point cloud information; this paper replaces point clouds with VGGT.
- **VLA Models**: Language-driven control paradigms; this paper provides justification for language-free alternatives.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing a 3D reconstruction pre-trained model into robot control is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐ — Evaluated only in the MetaWorld simulation, lacking real-world experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Arguments regarding biological motivation are compelling.
- **Value**: ⭐⭐⭐ — Poor viewpoint robustness and high computational overhead limit practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](../../ECCV2024/3d_vision/sapiens_foundation_for_human_vision_models.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava-bench_atomic_visual_ability_benchmark_for_vision_foundation_models.md)

</div>

<!-- RELATED:END -->
