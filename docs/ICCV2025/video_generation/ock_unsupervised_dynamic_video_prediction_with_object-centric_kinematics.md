---
title: >-
  [Paper Note] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics
description: >-
  [ICCV 2025][Video Generation][Object-centric learning] This paper proposes OCK (Object-Centric Kinematics), which augments object-centric video prediction by introducing explicit kinematic attributes (position, velocity…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Object-centric learning"
  - "video prediction"
  - "kinematic modeling"
  - "Slot Attention"
  - "autoregressive Transformer"
date: 2026-05-08
content_hash: d6ddc950f7a65882
---

# OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics

**Conference**: ICCV 2025
**arXiv**: [2404.18423](https://arxiv.org/abs/2404.18423)  
**Code**: None  
**Area**: Video Generation
**Keywords**: Object-centric learning, video prediction, kinematic modeling, Slot Attention, autoregressive Transformer

## TL;DR

This paper proposes OCK (Object-Centric Kinematics), which augments object-centric video prediction by introducing explicit kinematic attributes (position, velocity, acceleration) as complements to slot representations. Two Transformer variants — Joint-OCK and Cross-OCK — are designed to fuse appearance and motion information, achieving significant improvements in dynamic video prediction quality across complex synthetic and real-world scenarios.

## Background & Motivation

Human perception decomposes complex multi-object scenes into **time-invariant appearance** (size, shape, color) and **time-varying motion** (position, velocity, acceleration). Existing object-centric Transformer-based video prediction methods (e.g., SlotFormer, OCVP) primarily rely on appearance representations extracted by Slot Attention, and suffer from the following limitations:

**Lack of explicit motion dynamics**: Motion changes are learned only implicitly, making it difficult to accurately model dynamic interactions such as collisions and acceleration/deceleration.

**Poor performance in complex scenes**: Prediction quality degrades or even diverges in scenes with diverse object appearances, motion patterns, and backgrounds (e.g., MOVi-C/D/E).

**Poor long-term generalization**: The absence of explicit kinematic priors leads to rapid error accumulation.

## Method

### Overall Architecture

OCK consists of three main modules:
1. **Slot Encoder**: A pretrained SAVi model that decomposes video frames into object slots $\mathcal{S}_t \in \mathbb{R}^{N \times D_{\text{slot}}}$.
2. **Kinematics Encoder**: Extracts object kinematics $\mathbf{K}_t \in \mathbb{R}^{N \times D_{\text{kin}}}$ from video frames.
3. **Autoregressive OCK Transformer**: Fuses slot and kinematic information to predict slots at the next timestep.

### Key Designs

1. **Object Kinematics**: A CNN extracts low-level image features to localize the 2D centroid coordinates of each object, constructing a three-level kinematic state:
    $$\mathbf{K}_t = \begin{bmatrix} \mathbf{x}_t^{\text{pos}} \\ \mathbf{x}_t^{\text{vel}} \\ \mathbf{x}_t^{\text{acc}} \end{bmatrix} = \begin{bmatrix} \phi(\mathbf{o}_t) \\ \lambda(\mathbf{x}_t^{\text{pos}} - \mathbf{x}_{t-1}^{\text{pos}}) \\ \mathbf{x}_t^{\text{vel}} - \mathbf{x}_{t-1}^{\text{vel}} \end{bmatrix}$$
   where $\lambda$ is a learnable scaling parameter. Kinematics are modeled in 2D image space (avoiding the computational overhead of 3D depth estimation) and require no task-specific loss functions.

2. **Two Kinematic Integration Strategies**:

    - **Analytical**: Predicts the next-frame position from current kinematics as $\mathbf{x}_{t+1}^{\text{pos}'} = \mathbf{x}_t^{\text{pos}} + \mathbf{x}_t^{\text{vel}} \times \delta$, then feeds both current and predicted kinematics into the Transformer.
    - **Empirical**: Uses only current-frame kinematics, allowing the Transformer to implicitly learn motion patterns.

3. **Two OCK Transformer Architectures**:

    - **Joint-OCK**: Concatenates slots and kinematics and feeds them jointly into a standard Transformer encoder for self-attention.
    - **Cross-OCK**: Employs a cross-attention mechanism where slots serve as queries and kinematics as keys/values, with a temperature parameter $\tau$ for attention calibration: $\text{Cross-OCK}(\mathbf{v}, \mathbf{k}, \mathbf{q}; \tau) = \mathbf{v} \cdot \text{softmax}(\frac{\mathbf{k}^\top \mathbf{q}}{\tau})$

### Loss & Training

Training proceeds in two stages: SAVi is first trained to decompose video frames into slots, followed by training of the OCK Transformer.

The total loss is $\mathcal{L} = \mathcal{L}_{\text{object}} + \alpha \mathcal{L}_{\text{image}}$:
- **Object reconstruction loss**: L2 distance between predicted slots and ground-truth slots.
- **Image reconstruction loss**: L2 distance between frames decoded from predicted slots via the frozen SAVi decoder and ground-truth frames.

The model is trained with 6 input frames to predict 8 future frames, using temporal positional encodings that preserve permutation equivariance across objects.

## Key Experimental Results

### Main Results (Tables)

Video prediction quality on six synthetic datasets (from simple to complex):

| Model | OBJ3D PSNR↑ | MOVi-A PSNR↑ | MOVi-C PSNR↑ | MOVi-D PSNR↑ | MOVi-E PSNR↑ |
|------|-------------|-------------|-------------|-------------|-------------|
| SlotFormer | 33.08 | 25.18 | 19.48 | 20.68 | 21.27 |
| OCVP-Seq | 33.10 | 26.24 | 17.95 | Diverge | Diverge |
| Joint-OCK | **35.13** | 27.26 | **21.04** | **22.09** | **22.39** |
| Cross-OCK | 34.10 | **27.58** | 21.04 | 22.34 | 22.34 |

Real-world Waymo Open Dataset:

| Model | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| SlotFormer | 19.13 | 0.330 | 0.714 |
| OCVP-Seq | 18.98 | 0.329 | 0.718 |
| Joint-OCK | 25.02 | 0.798 | 0.251 |
| Cross-OCK | **25.98** | **0.728** | **0.220** |

### Ablation Study (Tables)

Ablation of Transformer components (MOVi-A):

| Setting | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Cross-OCK(A) default | **27.58** | **0.812** | **0.123** |
| Input frames = 4 | 27.01 | 0.801 | 0.125 |
| Input frames = 8 | 27.12 | 0.806 | 0.125 |
| Transformer layers = 6 | 26.92 | 0.796 | 0.130 |
| Transformer layers = 8 | 26.50 | 0.784 | 0.133 |
| Standard positional encoding | 23.60 | 0.591 | 0.205 |
| Teacher Forcing | 23.58 | 0.589 | 0.207 |

### Key Findings

- **Kinematics are critical for complex scenes**: OCVP completely diverges on MOVi-D/E, whereas OCK maintains stable predictions.
- On the Waymo real-world dataset, OCK improves PSNR by ~6.9 dB and reduces LPIPS by ~0.49 compared to SlotFormer.
- **Temporal positional encodings** (preserving permutation equivariance) are crucial; replacing them with standard positional encodings reduces PSNR by 4 dB.
- **Teacher Forcing is harmful**: Training the model to handle its own imperfect predictions is more beneficial for long-term generalization.
- The analytical strategy marginally outperforms the empirical strategy, as explicitly predicting the next-frame kinematic state provides more accurate guidance.
- Six input frames suffice to capture object dynamics; increasing to 8 frames yields a slight performance drop.

## Highlights & Insights

- **Physics-inspired design**: Incorporating classical kinematics (position–velocity–acceleration) into object-centric learning is an intuitive and effective innovation.
- **Elegant Cross-OCK design**: Using slots as queries and kinematics as keys/values achieves a favorable balance between computational efficiency and performance.
- **Long-term generalization**: Trained on only 6 frames, the model generalizes to 18-frame prediction with slow error accumulation.
- Strong performance is demonstrated on real-world autonomous driving data (Waymo).

## Limitations & Future Work

- Kinematics are modeled only in 2D image space, offering limited handling of 3D occlusions and depth variations.
- The method depends on the quality of SAVi-pretrained slot encoders; slot decomposition may be imperfect in complex scenes.
- Rotational and scale kinematics are not considered, limiting the modeling of complex motion types.
- Handling of object appearance and disappearance is not discussed.
- Incorporating object interaction graphs (GNNs) to explicitly model collision events is a promising future direction.

## Related Work & Insights

- SlotFormer is the most direct baseline; OCK extends it by incorporating a kinematic dimension.
- OCVP explores the separation of temporal and relational attention but diverges in complex scenes.
- G-SWM models interactions via graph neural networks but underperforms Transformer-based methods.
- The proposed approach has potential implications for physics simulator learning and robotic visual understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing classical kinematics into object-centric learning is a natural yet effective contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 7 datasets including real-world scenarios, with comprehensive ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured; the systematic comparison between two strategies (analytical/empirical) and two architectures (Joint/Cross) is clearly presented.
- Value: ⭐⭐⭐⭐ Addresses a key bottleneck in object-centric video prediction under complex scenes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DreamRelation: Relation-Centric Video Customization](dreamrelation_relation-centric_video_customization.md)
- [\[ICCV 2025\] FuXi-RTM: A Physics-Guided Prediction Framework with Radiative Transfer Modeling](fuxi-rtm_a_physics-guided_prediction_framework_with_radiative_transfer_modeling.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](../../NeurIPS2025/video_generation/training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)
- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](../../ACL2026/video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)

</div>

<!-- RELATED:END -->
