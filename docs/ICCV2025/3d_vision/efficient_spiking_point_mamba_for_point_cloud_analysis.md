---
title: >-
  [Paper Note] Efficient Spiking Point Mamba for Point Cloud Analysis
description: >-
  [3D Vision] SPM (Spiking Point Mamba) proposes the first Mamba-based 3D spiking neural network framework. Through Hierarchical Dynamic Encoding (HDE) and a Spiking Mamba Block (SMB)…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: bdb4aebf97d55380
---

# Efficient Spiking Point Mamba for Point Cloud Analysis

## Meta Information
- **Conference**: ICCV 2025
- **arXiv**: [2504.14371](https://arxiv.org/abs/2504.14371)
- **Code**: Not released
- **Area**: 3D Vision / Point Cloud Analysis
- **Keywords**: SNN, Mamba, point cloud analysis, spiking neural network, energy efficiency

## TL;DR

SPM (Spiking Point Mamba) proposes the first Mamba-based 3D spiking neural network framework. Through Hierarchical Dynamic Encoding (HDE) and a Spiking Mamba Block (SMB), it achieves over 3.5× energy reduction while improving accuracy by 6–7% over the previous state-of-the-art SNN method on ScanObjectNN.

## Background & Motivation

Spiking Neural Networks (SNNs) offer significant energy efficiency advantages due to their event-driven nature, yet they face three key challenges in 3D point cloud analysis:

**Difficulty modeling long-range dependencies**: Existing MLP- and Transformer-based SNN architectures struggle to capture long-range dependencies in irregular point sequences.

**Static temporal encoding**: Conventional direct encoding simply repeats inputs along the temporal dimension, failing to leverage the temporal feature extraction capability of SNNs.

**Spike information degradation**: Spike-driven computation yields substantially lower information density than ANNs (approximately 30% fewer activations per timestep), a problem that is exacerbated when integrating sequence models such as Mamba.

Although Mamba provides linear-complexity sequence modeling, directly transferring it to SNNs introduces **temporal complexity mismatches** (continuous state transitions vs. discrete spike events) and **information density discrepancies**.

## Method

### Overall Architecture

SPM consists of three components:
1. **HDE (Hierarchical Dynamic Encoding)**: Converts a static point cloud into a hierarchical dynamic event representation.
2. **SEL (Spiking Embedding Layer)**: Maps tokens to high-dimensional semantic features.
3. **SMB (Spiking Mamba Block)**: The core component, stacked $N=12$ times for feature interaction.

### Key Design 1: Hierarchical Dynamic Encoding (HDE)

Conventional SNNs apply direct encoding by repeating inputs along the temporal dimension, which renders temporal feature extraction ineffective. HDE introduces dynamic variation across three stages of Farthest Point Sampling (FPS):

- **Early stage** (unstable): Instability arises from random initial point selection.
- **Middle stage** (stable): Effectively captures the skeletal structure of the point cloud.
- **Late stage** (redundant): May introduce noise.

HDE strategies:
- **Finite Forward Sliding**: Dynamically slides to select $F$ points during the early and middle stages, with stride $l$ decreasing over time.
- **Infinite Backward Extension**: Dynamically samples $r$ points from the remaining points as a memory pool during the late stage.

### Key Design 2: Spiking Mamba Block (SMB)

The SMB is the core of SPM and contains two branches:

**SSM branch**:
- Input spikes pass through FC layers and spiking neurons (SN) to extract intermediate features.
- Only the **temporal dimension** is flipped (not the token dimension), since token flipping of a sparse spike matrix is semantically meaningless.
- Bidirectional SSM learns dynamic relationships across timesteps.

**Gate branch**:
- Maps spikes to high-dimensional gating matrices.
- Rather than applying direct element-wise multiplication (which causes severe information loss), Element-wise Average Pooling (EAP) is first applied along the token dimension.
- This preserves important feature dimensions while maintaining inter-token relationships.

The mathematical formulation of SMB:

$$\mathbf{S}_n'' = \mathcal{SN}(\mathbf{U}_n' + \mathbf{U}_t') \circ \mathcal{SN}(\text{EAP}(\mathbf{Z}_n))$$
$$\mathbf{U}_{n+1} = \text{MLP}(\mathbf{S}_n'') + \mathbf{U}_n$$

### Key Design 3: Spike-Based Pre-training

An asymmetric SNN–ANN heterogeneous encoder–decoder architecture is adopted:
- **Encoder**: $N$ stacked SMBs (SNN).
- **Decoder**: $N_d$ unidirectional SSM blocks (ANN, $N_d < N$).
- Chamfer Distance is used as the reconstruction loss.
- The SNN–ANN decoupled design leverages the modeling capacity of the ANN decoder to enhance the SNN encoder, while inference uses only the SNN, preserving low energy consumption.

### Energy Consumption Analysis

SPM converts multiply-accumulate operations (MAC, 4.6 pJ) in ANNs into sparse accumulate operations (AC, 0.9 pJ). Energy consumption is primarily attributed to the SEL and SMB modules.

## Key Experimental Results

### Main Results: 3D Classification (ScanObjectNN + ModelNet40)

| Architecture | Method | Params | T | OBJ-BG | OBJ-ONLY | PB-T50-RS | ModelNet40 |
|---|---|---|---|---|---|---|---|
| ANN | PointMamba | 12.3 | - | 90.2 | 89.8 | 85.4 | 92.4 |
| SNN | Spiking PointNet | 3.5 | 4 | 72.2 | 76.4 | 64.1 | 88.2 |
| SNN | P2SResLNet-B | 14.3 | 1 | 78.6 | 80.2 | 74.5 | 88.7 |
| SNN | SPT | 10.2 | 4 | 82.8 | 83.4 | 78.0 | 91.4 |
| **SNN** | **SPM (Ours)** | **12.8** | **4** | **90.2 (+7.4)** | **89.5 (+6.1)** | **84.2 (+6.2)** | **92.3 (+0.9)** |

### Ablation Study: SMB Design

| Model | Gate Branch | SSM Branch | OBJ-BG | PB-T50-RS | ModelNet40 |
|---|---|---|---|---|---|
| Model I (vanilla Gate + unidirectional) | Gate I | uni. | 88.3 | 83.1 | 91.4 |
| Model II (vanilla Gate + bidirectional) | Gate I | bi. | 89.3 | 83.8 | 91.8 |
| Model III (SNN Gate + unidirectional) | Gate II | uni. | 88.9 | 83.5 | 91.6 |
| **SMB (full design)** | **Gate II** | **bi.** | **90.2** | **84.2** | **92.3** |

### Timesteps and Energy Consumption

| Timesteps | Energy (mJ) | OBJ-BG | PB-T50-RS | ModelNet40 |
|---|---|---|---|---|
| ANN | 18.9 | 90.2 | 85.4 | 92.4 |
| 1 | 1.5 | 88.9 | 83.3 | 91.6 |
| 4 | 5.4 | 90.2 | 84.2 | 92.3 |
| 6 | 7.6 | 90.0 | 84.3 | 92.3 |

### Key Findings

1. **Substantial SNN SOTA improvement**: SPM outperforms the previous best SNN method (SPT) by 6–7% across all three ScanObjectNN variants.
2. **Near-ANN performance**: SPM matches the performance of its ANN counterpart, PointMamba, on classification tasks.
3. **Energy consumption at 1/3.5 of ANN**: At $T=4$, SPM achieves comparable performance with only 5.4 mJ vs. 18.9 mJ.
4. **Effective pre-training**: Spike-based pre-training yields an additional 2.3% gain on PB-T50-RS (84.2 → 86.5).
5. **Temporal flipping outperforms token flipping**: In the bidirectional strategy, flipping the temporal dimension is more effective than flipping the token dimension.

## Highlights & Insights

1. **First 3D Mamba SNN**: SPM successfully introduces Mamba into the 3D SNN domain, resolving the compatibility issue between spikes and continuous states.
2. **Elegant HDE design**: The three-stage FPS characteristic is exploited to naturally introduce temporal dynamics without additional parameters.
3. **EAP gating**: Averaging pooling along the token dimension prevents information loss from spike multiplication in a simple yet effective manner.
4. **Heterogeneous pre-training**: The asymmetric SNN encoder + ANN decoder design requires only the SNN at inference, preserving low energy consumption.

## Limitations & Future Work

1. The maximum number of timesteps evaluated is limited to 6 due to computational cost.
2. A performance gap with ANNs persists on the ShapeNetPart segmentation task (84.8 vs. 85.8 instance mIoU).
3. Validation is conducted only on relatively small-scale datasets; large-scale real-world scenarios remain untested.
4. The method relies on the Leaky Integrate-and-Fire (LIF) neuron model; applicability to other neuron models has not been explored.

## Related Work & Insights

- **PointMamba**: The ANN counterpart upon which SPM builds its SNN version.
- **Spiking Point Transformer (SPT)**: The previous SNN state-of-the-art, based on a Transformer architecture.
- **TA-TiTok / S4D / Mamba**: SPM integrates the efficient sequence modeling of SSMs with the low energy consumption of SNNs.

## Rating

⭐⭐⭐⭐ — Highly pioneering with thorough experiments, representing a leap forward in the SNN domain; however, practical application scenarios and validation at larger scales remain to be addressed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[ICCV 2025\] ResGS: Residual Densification of 3D Gaussian for Efficient Detail Recovery](resgs_residual_densification_of_3d_gaussian_for_efficient_detail_recovery.md)
- [\[ICCV 2025\] RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)
- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] StrandHead: Text to Hair-Disentangled 3D Head Avatars Using Human-Centric Priors](strandhead_text_to_hair-disentangled_3d_head_avatars_using_human-centric_priors.md)

</div>

<!-- RELATED:END -->
