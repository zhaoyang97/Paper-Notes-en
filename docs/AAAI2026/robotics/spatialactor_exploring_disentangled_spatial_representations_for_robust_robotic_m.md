---
title: >-
  [Paper Note] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation
description: >-
  [AAAI 2026][Robotics][Semantic-geometric disentanglement] This paper proposes SpatialActor, a framework that explicitly disentangles semantic and geometric representations. It introduces a Semantic-Guided Geometry Module…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Semantic-geometric disentanglement"
  - "depth estimation prior"
  - "spatial Transformer"
  - "robust manipulation"
  - "RLBench"
date: 2026-05-08
content_hash: 3bb0259524ff78f2
---

# SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation

**Conference**: AAAI 2026
**arXiv**: [2511.09555](https://arxiv.org/abs/2511.09555)  
**Authors**: Hao Shi, Bin Xie, Yingfei Liu, Yang Yue, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, Gao Huang
**Code**: [GitHub](https://github.com/shihao1895/SpatialActor)  
**Area**: Robotic Manipulation / Spatial Representation Learning
**Keywords**: Semantic-geometric disentanglement, depth estimation prior, spatial Transformer, robust manipulation, RLBench

## TL;DR

This paper proposes SpatialActor, a framework that explicitly disentangles semantic and geometric representations. It introduces a Semantic-Guided Geometry Module (SGM) that adaptively fuses noisy depth features with a pretrained depth estimation expert prior, and a Spatial Transformer (SPT) that encodes low-level spatial position cues. SpatialActor achieves 87.4% success rate on 50+ RLBench tasks (SOTA +6.0%) and outperforms RVT-2 by 19.4% under heavy-noise conditions.

## Background & Motivation

**3D spatial understanding is critical for robotic manipulation**: Real-world manipulation tasks occur in three-dimensional space, requiring precise spatial reasoning, occlusion handling, and fine-grained object interaction. Pure 2D visual methods are insufficient for these scenarios.

**Sparsity issues in point cloud methods**: Point cloud-based methods (PolarNet, PerAct, etc.) can explicitly represent 3D geometry, but sparse sampling leads to loss of fine-grained semantic information, and the high cost of 3D annotation limits pretraining scale.

**Semantic-geometric entanglement in image-based methods**: RGB-D image methods (RVT, RVT-2, etc.) feed RGB and depth into a shared feature space, where entanglement of semantics and geometry makes the model highly sensitive to depth noise — RVT-2's success rate drops by 8.9% under mild noise.

**Inevitable real-world depth noise**: Sensor noise, illumination variation, and surface reflectance ensure that depth measurements in the real world always contain noise, severely limiting the practical deployment of existing methods.

**Lack of low-level spatial cue modeling**: Existing joint modeling approaches primarily retain high-level geometric information while neglecting low-level spatial cues — such as precise 2D-3D correspondences — that are critical for accurate interaction.

**Core Problem**: How to construct robust spatial representations that simultaneously support fine-grained spatial understanding, robustness to sensor noise, and low-level spatial cues?

## Method

### Overall Architecture

SpatialActor takes as input $V$-view RGB images $I^v \in \mathbb{R}^{H \times W \times 3}$, depth maps $D^v \in \mathbb{R}^{H \times W}$, proprioceptive state $P$, and a language instruction $L$. The framework adopts a **three-branch disentangled** design:

- **Semantic branch**: RGB + language instruction → CLIP vision-language encoder → semantic features $F_{\text{sem}}^v$ and text features $F_{\text{text}}$
- **Geometric branch**: Raw depth $D^v$ → depth encoder (ResNet-50) → noisy geometric features $F_{\text{geo}}^v$, enhanced via SGM to yield fused geometric features $F_{\text{fuse-geo}}^v$
- **Spatial branch**: Semantic and fused geometric features are concatenated into $H^v$, then processed by SPT for spatial position encoding and multi-layer interaction

An action head finally predicts the 3D end-effector pose $A = (x, y, z, \theta_x, \theta_y, \theta_z, g)$.

### Semantic-Guided Geometry Module (SGM)

The core idea of SGM is to fuse **two complementary geometric representations**:

| Source | Characteristic | Advantage | Disadvantage |
|--------|---------------|-----------|--------------|
| Pretrained depth estimation expert (Depth Anything v2) | Infers geometry from RGB | Robust, noise-resistant | Coarse-grained |
| Raw depth encoder (ResNet-50) | Extracts geometry from $D^v$ | Fine-grained, pixel-level | Sensitive to noise |

The fusion mechanism employs **multi-scale gating**:

$$G^v = \sigma\bigl(\text{MLP}(\text{Concat}(\hat{F}_{\text{geo}}^v, F_{\text{geo}}^v))\bigr)$$

$$F_{\text{fuse-geo}}^v = G^v \odot F_{\text{geo}}^v + (1 - G^v) \odot \hat{F}_{\text{geo}}^v$$

The gate $G^v$ is adaptively learned: it preserves raw fine-grained details in reliable depth regions and falls back to the expert prior in highly noisy regions, achieving a balance between granularity and robustness.

### Spatial Transformer (SPT)

SPT assigns precise 3D positional information to each spatial token:

1. **3D coordinate computation**: Camera intrinsics $K^v$ and extrinsics $E^v$ are used to back-project pixel $(x', y')$ and corresponding depth $d$ into 3D coordinates $[x,y,z]^\top$ in the robot frame.
2. **Rotary Position Encoding (RoPE)**: $D/3$ dimensions are allocated to each axis of the 3D coordinates, generating sinusoidal/cosine position embeddings so that tokens at different spatial positions carry unique spatial indices.
3. **View-level interaction**: Self-attention + FFN refines token representations within each view.
4. **Scene-level interaction**: Tokens from all views are concatenated with language features $F_{\text{text}}$, and cross-view, cross-modal information is fused via self-attention + FFN.

### Action Prediction and Supervision

- A decoder (ConvexUp) generates per-view 2D heatmaps; argmax retrieves the target 2D location, which is then lifted to 3D via the camera model.
- An MLP regresses rotation angles $(\theta_x, \theta_y, \theta_z)$ and gripper state $g$.
- Loss functions: cross-entropy on 2D heatmaps (translation) + cross-entropy on discretized Euler angles (rotation) + binary classification loss (gripper).

## Key Experimental Results

### Experiment 1: RLBench 18-Task Performance Comparison

**Setup**: RLBench benchmark, 18 tasks with 249 variations, 4 fixed RGB-D cameras (128×128 resolution), 100 expert demonstrations per task for training and 25 unseen episodes for testing. Trained on 8 GPUs for ~40k iterations with batch size 192.

| Method | Avg. Success Rate ↑ | Avg Rank ↓ | Insert Peg | Sort Shape | Drag Stick |
|--------|---------------------|-----------|------------|------------|------------|
| PerAct | 49.4% | 7.1 | 5.6% | 16.8% | 70.4% |
| RVT | 62.9% | 5.3 | 11.2% | 36.0% | 88.0% |
| 3D Diffuser Actor | 81.3% | 2.8 | 65.6% | 44.0% | 96.8% |
| RVT-2 | 81.4% | 2.8 | 40.0% | 35.0% | 99.0% |
| **SpatialActor** | **87.4%** | **2.3** | **93.3%** | **73.3%** | **98.7%** |

**Key Findings**: SpatialActor achieves an average success rate of 87.4%, surpassing RVT-2 by 6.0%. On tasks requiring high spatial precision — Insert Peg and Sort Shape — it outperforms RVT-2 by 53.3% and 38.3%, respectively, demonstrating the advantage of disentangled spatial representations for fine-grained manipulation.

### Experiment 2: Noise Robustness Evaluation

**Setup**: Gaussian noise is injected into the reconstructed point cloud at three levels — Light (20% points, std=0.05), Middle (50% points, std=0.1), and Heavy (80% points, std=0.1).

| Method | Light ↑ | Middle ↑ | Heavy ↑ |
|--------|---------|----------|---------|
| RVT-2 | 72.5% | 68.4% | 57.0% |
| **SpatialActor** | **86.4%** (+13.9%) | **85.3%** (+16.9%) | **76.4%** (+19.4%) |

**Key Findings**: As noise severity increases, SpatialActor's advantage grows (13.9% → 16.9% → 19.4%). On the Insert Peg task under heavy noise, it surpasses RVT-2 by 61.3%, validating the noise resistance of the SGM gating fusion mechanism.

### Additional Results

- **Few-shot generalization** (19 new tasks, 10 demonstrations each): SpatialActor 79.2% vs. RVT-2 46.9% (+32.3%), indicating that disentangled representations significantly improve transfer capability.
- **ColosseumBench spatial perturbation** (20 tasks): SpatialActor achieves best results under object size, container size, and camera pose perturbations (baseline 57.4%; camera perturbation 54.2%).
- **Ablation study**: Disentanglement alone +3.7% (85.1%); +SGM +1.3% (86.4% / Heavy noise 73.9%); +SPT +1.0% (87.4% / Heavy noise 76.4%).
- **Real-world experiment**: WidowX arm + RealSense D435i, 8 tasks with 15 variations; SpatialActor 63% vs. RVT-2 43% (+20%).

## Highlights & Insights

1. **Explicit semantic-geometric disentanglement**: This breaks the paradigm of sharing a feature space for semantics and geometry in image-based methods, preventing depth noise from interfering with semantic understanding and improving robustness at the root level.
2. **Complementary geometric fusion**: SGM cleverly combines the complementary strengths of a pretrained depth estimation expert (noise-resistant but coarse) and raw depth features (fine-grained but noisy); the gating mechanism adaptively modulates their relative trust.
3. **Low-level spatial cue modeling**: SPT encodes true 3D coordinates into the Transformer via RoPE, establishing precise 2D-3D correspondences so that interactions among spatial tokens carry geometric meaning.
4. **Overwhelming advantage under noisy conditions**: 76.4% success rate under heavy noise (vs. 57.0%), and 79.2% in few-shot settings (vs. 46.9%), demonstrating substantial practical deployment value.

## Limitations & Future Work

1. **Increased computational overhead**: Incorporating the frozen Depth Anything v2 expert model increases inference-time computation and memory cost, which may become a bottleneck on resource-constrained robotic platforms.
2. **Dependence on pretrained depth estimation quality**: The robustness of SGM relies on the generalization ability of the depth estimation expert; when the scene distribution deviates significantly from the pretraining data, the quality of the expert prior may degrade.
3. **Single-arm tabletop scenarios**: Experiments are primarily validated on tabletop platforms (Franka / WidowX) and do not cover more complex settings such as bimanual coordination, dexterous hands, or mobile manipulation.
4. **Fixed viewpoint assumption**: Simulation uses 4 fixed RGB-D cameras, and the real-world setup uses only 1 static camera; applicability to dynamic viewpoints or eye-in-hand configurations has not been verified.

## Related Work & Insights

- **Point cloud methods**: PolarNet (Chen 2023), PerAct (Shridhar 2023) — explicit 3D structure but sparse.
- **Image-based methods**: RVT (Goyal 2023), RVT-2 (Goyal 2024), SAM-E (Zhang 2024) — dense semantics but with semantic-geometric entanglement.
- **3D diffusion policy**: 3D Diffuser Actor (Ke 2024) — 3D scene representation + diffusion policy.
- **Visual foundation models**: CLIP (Radford 2021) for semantic priors; Depth Anything v2 (Yang 2025) for geometric priors.
- **Voxel-based methods**: C2F-ARM-BC (James 2022) — coarse-to-fine voxelization with high computational cost.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐ |
| Practicality | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Recommended Reading | ⭐⭐⭐⭐ |

**Overall Recommendation**: ⭐⭐⭐⭐
**Rationale**: This paper proposes a spatially disentangled representation framework with clear engineering intuition and strong empirical results in robotic manipulation. The design of semantic-geometric disentanglement combined with complementary geometric fusion demonstrates a pronounced advantage under noisy conditions. Comprehensive evaluation across 50+ tasks and real-robot experiments substantially enhance credibility. The core idea — disentanglement + complementary fusion — also serves as a valuable reference for other tasks requiring robust multimodal fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)
- [\[AAAI 2026\] Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems](shadows_in_the_code_exploring_the_risks_and_defenses_of_llm-.md)
- [\[AAAI 2026\] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception](touchformer_a_robust_transformer-based_framework_for_multimodal_material_percept.md)
- [\[AAAI 2026\] EvoEmpirBench: Dynamic Spatial Reasoning with Agent-ExpVer](evoempirbench_dynamic_spatial_reasoning_with_agent-expver.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)

</div>

<!-- RELATED:END -->
