---
title: >-
  [Paper Note] Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics
description: >-
  [CVPR 2026][3D Vision][SE(3) Equivariance] This paper proposes E3Flow, the first equivariant flow matching policy framework based on spherical harmonic representations. It introduces a Feature Enhancement Module (FEM) to…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "SE(3) Equivariance"
  - "Spherical Harmonics"
  - "Rectified Flow"
  - "Robot Policy Learning"
  - "Multi-Modal Fusion"
date: 2026-05-08
content_hash: 64acaca5baf6e407
---

# Efficient Hybrid SE(3)-Equivariant Visuomotor Flow Policy via Spherical Harmonics

**Conference**: CVPR 2026
**arXiv**: [2603.23227](https://arxiv.org/abs/2603.23227)
**Code**: [https://github.com/zql-kk/E3Flow](https://github.com/zql-kk/E3Flow)
**Area**: 3D Vision / Robot Manipulation
**Keywords**: SE(3) Equivariance, Spherical Harmonics, Rectified Flow, Robot Policy Learning, Multi-Modal Fusion

## TL;DR

This paper proposes E3Flow, the first equivariant flow matching policy framework based on spherical harmonic representations. It introduces a Feature Enhancement Module (FEM) to dynamically fuse point cloud and image modalities, and combines rectified flow for efficient equivariant action generation. E3Flow achieves an average success rate 3.12% higher than the strongest baseline SDP across 8 MimicGen tasks while delivering a 7× speedup in inference.

## Background & Motivation

Diffusion Policy has demonstrated strong performance in robot policy learning, yet faces two core challenges:

**Low data efficiency**: It relies on large amounts of high-quality expert demonstrations, which are costly to collect.

**Bottlenecks in equivariant methods**: Embedding symmetry priors can substantially improve data efficiency and generalization, but existing equivariant diffusion policies suffer from:
- **Computational intensity**: Requiring many iterative denoising steps, which compounds the already high cost of equivariant networks.
- **Unimodal dependency**: Relying solely on point clouds or images, lacking fine-grained visual detail.
- **Incompatibility with fast sampling**: Directly applying one-step diffusion or flow matching to equivariant policies leads to instability and performance degradation.

Core gap: **No existing method unifies the data efficiency of equivariant learning with the inference speed of flow matching.**

As illustrated in Figure 1(a), when a tabletop object is rotated to an unseen pose, non-equivariant DP fails to grasp it, whereas the equivariant E3Flow succeeds by correspondingly rotating the original trajectory.

## Method

### Overall Architecture

The E3Flow pipeline (Figure 2) proceeds as follows:

1. **Multimodal encoding**: Eye-in-hand camera images → ResNet extracts invariant features; single-view point cloud → EquiformerV2 extracts equivariant features.
2. **FEM fusion**: Image semantics are injected into the spherical harmonic representation of the point cloud.
3. **Spherical harmonic conditioning**: Fused visual features + proprioceptive state are mapped into spherical harmonic space.
4. **Rectified flow generation**: An ODE solver efficiently generates equivariant action sequences.

### Key Designs

1. **Spherical harmonic visual representation**: Strict SO(3) equivariance is achieved via spherical harmonic functions.

   - Spherical harmonics $Y_l^m(\theta,\phi)$ form an orthogonal basis on the unit sphere. Under rotation $R$, harmonics of the same order $l$ undergo linear mixing: $Y_l^m(R^{-1}\hat{\mathbf{r}}) = \sum_{m'} D_{mm'}^{(l)}(R) Y_l^{m'}(\hat{\mathbf{r}})$
   - EquiformerV2 encodes point clouds into multi-order spherical harmonic features (scalar $f_{pcd}^{(0)}$ + higher-order $f_{pcd}^{(>0)}$), preserving fine-grained directional and rotational detail.
   - Compared to EquiBot's vector neuron approach, EquiformerV2's higher-order coefficient encoding captures more precise directional information.
   - Compared to SDP's sparse point cloud only approach, this work introduces hybrid visual input.

2. **Feature Enhancement Module (FEM)**: Cross-modal dynamic fusion.

   - Image features are injected exclusively into the scalar component (Type-0), preserving the equivariance of higher-order components.
   - Core formula: $f_{fused} = \Pi[\Lambda(\mathcal{A}(f_{pcd}^{(0)}, f_{img}), f_{pcd}^{(0)}) \| f_{pcd}^{(>0)}]$
   - $\mathcal{A}$: cross-modal attention, using point cloud scalar features as queries and image features as keys/values.
   - $\Lambda$: gating mechanism that adaptively balances image contribution — contrasting with the performance drop caused by naive concatenation (79.00% → 72.36%).
   - $\Pi$: projection back into spherical harmonic space.
   - **Design Motivation**: Naively concatenating features from different modalities disrupts the equivariant structure. FEM operates exclusively in the invariant subspace (Type-0), elegantly preserving equivariance.

3. **Equivariant rectified flow**: Efficient action generation.

   - Learns a straight-line interpolation path from a noise distribution to the action distribution: $x_t = (1-t)x_0 + ta$
   - Training loss: $\mathcal{L}_{RF}(\theta) = \mathbb{E}_{t,x_0,x_1}[\|v_\theta(x_t,t,s,v) - (a-x_0)\|^2]$
   - Since the velocity field network is equivariant, it satisfies $v_\theta(\rho*x_t, t, \rho*s, \rho*v) = \rho * v_\theta(x_t, t, s, v)$
   - The training target is a linear transformation of equivariant actions, and the loss is invariant under group actions; therefore, an equivariant optimal solution exists.
   - Default 10-step sampling yields an inference time of 0.51s, 7× faster than SDP (DDPM) at 3.73s.

### Loss & Training

- Optimizer: AdamW, learning rate $1 \times 10^{-4}$, batch size 64.
- EMA decay rate: 0.95.
- Trained for 500 epochs on a single NVIDIA H20 GPU.
- Evaluated every 20 epochs over 50 episodes; maximum success rate is reported.
- Action representation: 3D position + 6D rotation + 1D gripper state (position and rotation are equivariant quantities).

## Key Experimental Results

### Main Results

**MimicGen 8-task success rate** (Table 1, 100 expert demonstrations)

| Method | Equivariance | Coffee_D2 | Nut_Asm | Square | Stack3 | **Avg.** |
|--------|-------------|-----------|---------|--------|--------|----------|
| DP | None | 44 | 54 | 10 | 32 | 47.50 |
| EquiDiff (voxel) | SO(2) | 65 | 67 | 39 | 76 | 68.50 |
| SDP (DDPM) | SE(3) | 63 | 92 | 64 | 98 | 75.88 |
| **E3Flow** | **SE(3)** | **64** | **94** | **70** | **100** | **79.00** |

**Inference time** (Table 2)

| Method | Avg. Inference Time (s) | Relative to E3Flow |
|--------|------------------------|-------------------|
| EquiBot | 2.03 | 4.0× |
| DP | 0.95 | 1.9× |
| EquiDiff (img) | 2.51 | 4.9× |
| EquiDiff (voxel) | 1.10 | 2.2× |
| SDP (DDPM) | 3.73 | **7.3×** |
| SDP (DDIM) | 0.46 | 0.9× |
| **E3Flow** | **0.51** | **1.0×** |

Note: Accelerating SDP with DDIM reduces its success rate by 6.13% (75.88 → 69.75), whereas E3Flow achieves higher success at comparable speed.

### Ablation Study

**Component analysis** (Table 4)

| Input | Fusion | Generation | Avg. Success Rate |
|-------|--------|------------|------------------|
| PCD | — | RF | 75.88 |
| PCD | — | Diffusion | 75.23 |
| PCD+Img | cat | RF | 72.36 |
| PCD+Img | FEM | Diffusion | 77.58 |
| **PCD+Img** | **FEM** | **RF** | **79.00** |

**Flow method comparison** (Table 5)

| Method | Steps | Inference Time | Avg. Success Rate |
|--------|-------|---------------|------------------|
| MeanFlow | 1 | 0.17s | 54.50 |
| AlphaFlow | 1 | 0.17s | 64.62 |
| RF-1 | 1 | 0.16s | 69.00 |
| RF-5 | 5 | 0.28s | 71.00 |
| **RF-10** | **10** | **0.51s** | **79.00** |

### Key Findings

- Naive multimodal feature concatenation (cat) degrades performance (79.00 → 72.36%), underscoring the importance of modality alignment.
- FEM elegantly resolves the tension between equivariance and multimodal fusion by operating exclusively in the invariant subspace (Type-0 features).
- Equivariant learning offers a pronounced advantage on complex tasks: DP achieves only 10% on Square_D2, while E3Flow reaches 70%.
- One-step sampling is unsuitable for equivariant models (MeanFlow: 54.50%), as a single forward pass is insufficient for highly abstract equivariant features to guide fine-grained actions.
- SE(3) generalization experiments (Table 3): E3Flow comprehensively outperforms SDP in zero-shot tests with 10° tilting perturbations.
- Data efficiency: E3Flow with 100 demonstrations matches the performance of other methods using 200 demonstrations (Figure 5).

## Highlights & Insights

1. **First successful unification of equivariant learning and flow matching**: The work demonstrates that rectified flow naturally accommodates equivariant networks — since the training target is a linear transformation of equivariant actions, the loss is invariant under group actions.
2. **Elegant design of FEM**: Image semantics are injected only into the Type-0 invariant subspace, leaving higher-order equivariant features intact and resolving the dilemma between multimodal fusion and equivariance preservation.
3. **In-depth analysis of one-step methods**: The paper reveals why equivariant models require multi-step sampling — the highly abstract equivariant features require more steps to be decoded into fine-grained actions.
4. **End-to-end equivariance proof**: The complete equivariant chain from input to output is supported by rigorous mathematical guarantees.
5. **Practical deployment potential**: 0.51s inference time combined with 100-demonstration data efficiency makes the approach suitable for real-world robotic scenarios.

## Limitations & Future Work

- Although a single EquiformerV2 forward pass is faster than ET-SEED, it remains the inference bottleneck.
- Equivariance is validated only for SE(3); more general symmetry groups (e.g., SIM(3) with scale) are unexplored.
- Downsampling point clouds to 1,024 points may discard geometric detail, potentially insufficient for precision assembly tasks.
- Real-world experiments cover only 4 tasks, limiting the scale of evaluation.
- The sim-to-real gap and domain randomization are not discussed.
- The image encoder (ResNet) in FEM does not use pretrained weights (e.g., CLIP), which may constrain semantic understanding.

## Related Work & Insights

- E3Flow extends the SDP framework: SDP employs spherical harmonics + diffusion, while E3Flow replaces diffusion with rectified flow and adds image input.
- The comparison with EquiDiff highlights the difference between continuous equivariance (SO(3) spherical harmonics) and discrete equivariance (SO(2) convolution).
- The FEM design paradigm is generalizable to other settings that require injecting invariant information into equivariant representations.
- The application of rectified flow in robot policy learning warrants further investigation, including fewer sampling steps and distillation approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first unification of equivariant learning and flow matching is a significant contribution, though core components build upon existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 8 simulation tasks + 4 real-world tasks, with extensive ablations and baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and figures are professionally presented.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses the inference efficiency bottleneck of equivariant policies, offering direct practical value to the robot learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [\[ICCV 2025\] Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](../../ICCV2025/3d_vision/spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [\[CVPR 2026\] From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection](from_pairs_to_sequences_track-aware_policy_gradients_for_keypoint_detection.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)

</div>

<!-- RELATED:END -->
