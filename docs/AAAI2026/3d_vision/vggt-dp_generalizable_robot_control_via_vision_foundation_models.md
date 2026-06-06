---
title: >-
  [Paper Note] VGGT-DP: Generalizable Robot Control via Vision Foundation Models
description: >-
  [AAAI 2026][3D Vision][visuomotor policy] This paper proposes VGGT-DP, a biologically inspired visuomotor policy framework that integrates the pretrained 3D-aware foundation model VGGT as a visual encoder with Diffusion…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "visuomotor policy"
  - "diffusion policy"
  - "VGGT"
  - "proprioception guidance"
  - "robot manipulation"
date: 2026-05-08
content_hash: 5deff3b123ee299a
---

# VGGT-DP: Generalizable Robot Control via Vision Foundation Models

**Conference**: AAAI 2026
**arXiv**: [2509.18778](https://arxiv.org/abs/2509.18778)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: visuomotor policy, diffusion policy, VGGT, proprioception guidance, robot manipulation

## TL;DR

This paper proposes VGGT-DP, a biologically inspired visuomotor policy framework that integrates the pretrained 3D-aware foundation model VGGT as a visual encoder with Diffusion Policy. Through three key designs — frame-wise token reuse (FTR), random token pruning, and proprioception-guided visual learning — VGGT-DP substantially outperforms DP and DP3 baselines on high-precision manipulation tasks in MetaWorld.

## Background & Motivation

### From Biological Vision to Robot Perception

Visuomotor policies are central to robot manipulation. Two dominant research paradigms exist:

**Vision-Action (VA)**: lightweight visual encoder + large policy head, e.g., Diffusion Policy.

**Vision-Language-Action (VLA)**: leverages large-scale vision-language models to provide language priors for improved generalization.

The authors offer a compelling biological insight: **many non-linguistic organisms exhibit remarkable manipulation capabilities**. Insects, fruit flies, and even unicellular organisms navigate and manipulate their environments with no linguistic or symbolic reasoning. Biological research indicates that a large proportion of neural resources in animals is dedicated to **visual processing**.

The core issue, therefore, is not linguistic priors but rather the **capacity and quality of visual representations**. Visual encoders commonly used in current robotic systems are often too simple to capture complex spatial and geometric relationships.

### Why VGGT

VGGT (Visual Geometry Grounded Transformer) is a vision foundation model pretrained on large-scale 3D reconstruction tasks, capable of jointly predicting camera poses, dense depth maps, 3D point clouds, and visual features. Unlike semantics-focused models such as CLIP or DINOv2, VGGT provides **geometry-aware spatial representations** that are better suited to manipulation tasks requiring precise spatial reasoning.

## Method

### Overall Architecture

VGGT-DP consists of three core components:
1. **VGGT encoder + token pruning**: extracts geometry-aware visual features
2. **Frame-Wise Token Reuse (FTR)**: reduces inference latency
3. **Proprioception-guided diffusion policy**: fuses visual and proprioceptive signals for action prediction

### Key Designs

#### 1. **VGGT as a Feature Projector**

Rather than using VGGT's low-level visual outputs (depth maps, point clouds), the method leverages the **tokens output by VGGT's aggregator**, which are compact, semantically rich 3D scene representations.

Given $B \cdot T$ frames and $V$ camera views, VGGT outputs visual tokens:
$$\mathcal{T}_{vggt} = \text{VGGT}_{agg}(\mathcal{I}) \in \mathbb{R}^{B \cdot T \times V \times (N_p+1) \times D}$$

These tokens are further processed by a Transformer Encoder, then average-pooled and projected via an MLP to produce the conditioning embedding $\mathcal{C} \in \mathbb{R}^{B \cdot T \times d_c}$, which serves as the conditional input to the diffusion policy.

**Design Motivation**: VGGT's aggregator already fuses multi-view spatial and appearance cues, yielding geometry-aware tokens with global context that are more expressive than raw image features.

#### 2. **Frame-Wise Token Reuse (FTR)**

Existing methods recompute visual embeddings for all observation frames at every inference step, even when frames overlap across temporal windows. For large models such as VGGT, this incurs substantial computational cost.

FTR's core idea: **reuse precomputed tokens for overlapping frames**. Only the most recent frame triggers new VGGT computation; tokens for older frames are cached on CPU:
$$\mathcal{T}_{vggt}^{(t)} = \text{Concat}(\mathcal{T}_{cache}^{(t-1)}, \text{VGGT}_{agg}(\mathcal{I}_t))$$

**Design Motivation**: In the sliding window used for robot control, most observation frames are shared between adjacent time steps. FTR reduces inference cost from $O(T)$ to $O(1)$.

#### 3. **Random Token Pruning**

Before feeding VGGT tokens into the Transformer Encoder, a fraction $r_{prune}$ of patch tokens is randomly discarded.

**Design Motivation**: Introducing token-level stochasticity prevents overfitting, reduces computation to accelerate inference, and encourages the model to learn representations invariant to partial observation loss.

#### 4. **Proprioception-Guided Visual Learning**

An auxiliary decoder $D$ is trained to predict the robot's proprioceptive state (joint angles + end-effector position) from visual features:
$$\hat{p}_t = D(f_t), \quad \mathcal{L}_{proprio} = \mathbb{E}_t[\|p_t - \hat{p}_t\|^2]$$

**Design Motivation**: This forces the visual encoder to learn spatially grounded, manipulation-relevant features, thereby improving closed-loop feedback control quality.

### Loss & Training

- Architecture: U-Net-1D diffusion model with FiLM conditioning
- Scheduler: DDIM, 100 training timesteps, 10 denoising steps at inference
- Prediction horizon: 16 steps; observation window: 2 steps
- Optimizer: AdamW, lr=$1 \times 10^{-4}$, weight decay $1 \times 10^{-6}$
- Training: 3000 epochs, batch size 128, EMA decay 0.9999

## Key Experimental Results

### Main Results

**Success rate (%) on 10 selected MetaWorld tasks**:

| Task | DP | DP3 | **VGGT-DP** | Type |
|------|-----|-----|-------------|------|
| Disassemble | 43±7 | **69±4** | 55±2.5 | Simple |
| Peg Unplug Side | 74±3 | **75±5** | 63±6 | Simple |
| Pick out of Hole | 0±0 | 14±9 | **55±6** | Complex Spatial |
| Shelf Place | 11±3 | **17±10** | 10±0 | Placement |
| Reach | 18±2 | 24±1 | **42±8** | Complex Spatial |
| Soccer | 14±4 | 18±3 | **30±7** | Complex Spatial |
| Sweep Into | 10±4 | 15±5 | **44±4** | Complex Spatial |
| Hand Insert | 10±4 | 15±5 | **19±4** | Complex Spatial |
| Pick Place | 0±0 | **12±4** | 0±0 | Placement |
| Stick Pull | 11±2 | 27±8 | **48±5** | Complex Spatial |
| **Average** | 19.1 | 28.6 | **36.6** | — |

VGGT-DP achieves an average success rate of 36.6%, outperforming DP by 17.5 percentage points and DP3 by 8.0 percentage points.

### Ablation Study

**Viewpoint perturbation robustness (Stick Pull task)**:

| Perturbation δ | Success Rate | Note |
|----------------|--------------|------|
| 0° | 39% | Normal viewpoint |
| 5° | 5% | Sharp drop under minor perturbation |
| 10° | 0% | Complete failure |
| 15° | 0% | Complete failure |

**FTR mechanism**: FTR significantly reduces inference latency under large batch sizes and long temporal windows, offering practical value for deploying large vision models in real-time systems.

### Key Findings

1. **VGGT-DP excels on complex spatial reasoning tasks**: Pick out of Hole (0→55%), Sweep Into (10→44%), Stick Pull (11→48%)
2. **No advantage on simple tasks**: lightweight encoders suffice for these
3. **Failure on placement tasks**: the model cannot precisely localize small, elongated, or partially occluded target objects
4. **Severe viewpoint sensitivity**: a mere 5° perturbation causes a catastrophic drop in success rate (39%→5%), indicating strong overfitting to training camera poses

## Highlights & Insights

1. **Biologically grounded insight**: the argument for bypassing language priors and returning to the essence of visual perception is compelling
2. **FTR is simple yet effective**: exploiting temporal redundancy for token reuse provides a practical reference for deploying large vision models in real-time robotic systems
3. **Proprioception-guided visual learning**: using the robot's internal state as auxiliary supervision to guide visual feature learning is a promising direction in embodied AI

## Limitations & Future Work

1. **Severely insufficient viewpoint robustness**: the most critical weakness. Failure at 5° perturbation calls for equivariant encoders or camera-pose domain randomization
2. **Evaluation limited to simulation (MetaWorld)**: no real-world robot experiments are provided
3. **High computational cost of VGGT**: the large parameter count constrains deployment for real-time control
4. **Failure on placement tasks**: insufficient support for small objects and fine-grained manipulation

## Related Work & Insights

- **Diffusion Policy (DP)**: the policy foundation of this work; VGGT-DP primarily improves the visual encoder
- **DP3**: exploits 3D information from point clouds; this work substitutes VGGT representations for point clouds
- **VLA models**: language-driven control paradigm; this work provides a reasoned case for a language-free alternative

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing a 3D-reconstruction pretrained model into robot control is an inspiring contribution
- **Experimental Thoroughness**: ⭐⭐⭐ — Limited to MetaWorld simulation; real-world experiments are absent
- **Writing Quality**: ⭐⭐⭐⭐ — The biological motivation is articulated persuasively
- **Value**: ⭐⭐⭐ — Practical applicability is constrained by poor viewpoint robustness and high computational overhead

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[AAAI 2026\] Adapt-As-You-Walk Through the Clouds: Training-Free Online Test-Time Adaptation of 3D Vision-Language Foundation Models](adapt-as-you-walk_through_the_clouds_training-free_online_te.md)
- [\[ICLR 2026\] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](../../ICLR2026/3d_vision/generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)
- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](../../ICCV2025/3d_vision/vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)

</div>

<!-- RELATED:END -->
