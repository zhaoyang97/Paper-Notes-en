---
title: >-
  [Paper Note] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation
description: >-
  [3D Vision] This paper proposes a cross-modal distillation paradigm that leverages a vision foundation model (VFM) from the image domain (Depth Anything v2) to generate pseudo-labels for training event-based depth estima…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: 23ce695089900ad4
---

# Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2509.15224](https://arxiv.org/abs/2509.15224)
- **Code**: [bartn8/depthanyevent](https://bartn8.github.io/depthanyevent)
- **Area**: 3D Vision / Event Camera Depth Estimation
- **Keywords**: Event Camera, Monocular Depth Estimation, Cross-Modal Distillation, Vision Foundation Model, Recurrent Architecture

## TL;DR

This paper proposes a cross-modal distillation paradigm that leverages a vision foundation model (VFM) from the image domain (Depth Anything v2) to generate pseudo-labels for training event-based depth estimation networks. It further introduces a VFM-based recurrent architecture, DepthAnyEvent-R, achieving state-of-the-art performance in event-based monocular depth estimation without requiring costly depth annotations.

## Background & Motivation

- **Background**: Event cameras capture brightness changes at microsecond-level temporal resolution with high dynamic range, making them particularly suitable for high-speed motion and challenging illumination scenarios (autonomous driving, UAVs, robotics, etc.)
- **Limitations of Prior Work**: Event data lacks large-scale datasets with dense depth annotations; the prohibitive annotation cost severely constrains learning-based depth estimation methods.
- **Key Challenge**: VFMs in the image domain (e.g., Depth Anything v2) achieve strong depth estimation capabilities through large-scale pretraining, yet no equivalent large-scale dataset exists in the event domain.
- **Goal**: DAVIS cameras can simultaneously output spatially aligned RGB frames and event streams, providing a natural condition for cross-modal knowledge transfer.

## Method

### Overall Architecture

The paper presents two main contributions:
1. **Cross-Modal Distillation Paradigm**: A pretrained VFM serves as the teacher model to process RGB frames and generate pseudo depth labels $\mathbf{D}^*$, which supervise event-domain student models (e.g., E2Depth, EReFormer).
2. **VFM Adaptation to the Event Domain**: Directly fine-tuning DAv2 to the event domain (DepthAnyEvent), or extending it into a recurrent architecture (DepthAnyEvent-R).

### Cross-Modal Distillation

- **Teacher Model**: DAv2 ViT-Large, fine-tuned on EventScape for 10K steps.
- **Student Model**: Any event-based depth estimation network.
- **Alignment Condition**: Frames and events are spatially and temporally aligned (naturally satisfied by DAVIS cameras).
- **Training Loss**:

$$\mathcal{L} = \mathcal{L}_{si} + \lambda \mathcal{L}_{reg}$$

Scale-invariant loss:

$$\mathcal{L}_{si}(\hat{\mathbf{D}}, \hat{\mathbf{D}}^*) = \frac{1}{2|\mathbf{M}|} \sum_{(x,y) \in \mathbf{M}} (\hat{\mathbf{D}} - \hat{\mathbf{D}}^*)^2$$

Scale and shift are solved via least squares: $(s,t) = \arg\min_{s,t} \sum (\mathbf{sD}+t - \mathbf{D}^*)^2$

Gradient regularization term:

$$\mathcal{L}_{reg} = \sum_{k=1}^{K} \frac{1}{|\mathbf{M}_k|} \sum (|\nabla_x \mathbf{R}_k| + |\nabla_y \mathbf{R}_k|)$$

### Event Representation — Tencode

To minimize modifications to the pretrained VFM, **Tencode** is adopted to encode events as RGB images:

$$\mathbf{E}(x_k, y_k) = \begin{cases} (1, \frac{t_d - t_k}{\Delta T}, 0) & \text{if } p_k = 1 \\ (0, \frac{t_d - t_k}{\Delta T}, 1) & \text{if } p_k = -1 \end{cases}$$

The R/B channels encode positive/negative polarity, and the G channel encodes relative timestamps, preserving both spatial and temporal information.

### DepthAnyEvent (Vanilla VFM Adaptation)

DAv2 ViT-Small is directly fine-tuned using the Tencode representation without any architectural modification.

### DepthAnyEvent-R (Recurrent VFM Architecture)

**ConvLSTM** modules are inserted after the multi-scale feature maps of the DAv2 encoder to incorporate temporal information from historical event stacks:

- Encoder $\mathcal{G}$ splits Tencode images into patches → multi-layer Transformer → multi-scale feature maps $\mathbf{F}_s$
- At each scale $s$, ConvLSTM module $\mathcal{R}_s$ receives $\mathbf{F}_s$ and hidden state $\mathbf{H}_s^i$, outputting enhanced features $\hat{\mathbf{F}}_s$ and updated hidden state $\mathbf{H}_s^{i+1}$
- Hierarchical fusion → decoder $\mathcal{D}$ → final depth map
- This design addresses degraded prediction quality in static scenes where events are sparse.

## Key Experimental Results

### Main Results: Zero-Shot Generalization (Trained Only on EventScape Synthetic Data)

| Model | Dataset | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|--------|----------|-------|---------|
| E2Depth | MVSEC | 0.527 | 7.894 | 0.363 |
| EReFormer | MVSEC | 0.518 | 8.423 | 0.361 |
| **DepthAnyEvent** | MVSEC | **0.466** | **7.824** | 0.408 |
| **DepthAnyEvent-R** | MVSEC | 0.469 | 8.064 | **0.428** |
| E2Depth | DSEC | 0.395 | 13.258 | 0.409 |
| EReFormer | DSEC | 0.297 | 11.608 | 0.524 |
| **DepthAnyEvent** | DSEC | 0.297 | 11.072 | 0.519 |
| **DepthAnyEvent-R** | DSEC | **0.276** | **10.942** | **0.555** |

### Ablation Study: Distillation vs. Full Supervision (After MVSEC Fine-tuning)

| Model | Supervision | Abs Rel↓ | RMSE↓ | δ<1.25↑ |
|------|----------|----------|-------|---------|
| E2Depth | Synth | 0.527 | 7.894 | 0.363 |
| E2Depth | **Distilled** | 0.400 | **6.786** | **0.479** |
| E2Depth | Supervised | 0.420 | 7.268 | 0.432 |
| DepthAnyEvent | Synth | 0.466 | 7.824 | 0.408 |
| DepthAnyEvent | **Distilled** | 0.397 | 6.910 | 0.461 |
| DepthAnyEvent | Supervised | **0.373** | **6.627** | **0.471** |
| DepthAnyEvent-R | **Distilled** | 0.399 | 6.830 | 0.462 |
| DepthAnyEvent-R | Supervised | **0.365** | **6.465** | **0.489** |

### Key Findings

- **Distillation vs. Full Supervision**: On E2Depth, the distilled model **surpasses** the fully supervised counterpart on several metrics (RMSE 6.786 vs. 7.268), indicating that the density of VFM pseudo-labels compensates for the sparsity of LiDAR annotations.
- **DSEC Dataset**: DepthAnyEvent-R achieves Abs Rel 0.226 under distillation vs. 0.191 under full supervision — a manageable gap.
- **Tencode vs. Voxel Grid**: Ablation experiments (C) vs. (D) demonstrate that Tencode outperforms Voxel Grid.
- **Importance of Pretraining**: Without pretraining (E), Abs Rel degrades to 0.446, significantly worse than 0.365 with pretraining (C).
- **Mixed Supervision (F)**: Training jointly with ground truth and distilled labels yields the best performance on several metrics.

## Highlights & Insights

1. **Paradigm Innovation**: This work is the first to systematically distill knowledge from an image-domain VFM into the event domain, elegantly exploiting the natural alignment property of DAVIS cameras.
2. **Practical Value**: The distillation scheme entirely eliminates the need for costly depth annotation, requiring only aligned RGB frames and event streams for training.
3. **VFM Pseudo-Labels Outperform Sparse LiDAR**: Dense pseudo-labels generated by the VFM outperform LiDAR annotations in certain scenarios, as LiDAR ground truth is itself semi-sparse.
4. **Principled Recurrent Architecture**: The ConvLSTM modules naturally integrate temporal information into the VFM, improving depth estimation quality in static scenes and continuous sequences.

## Limitations & Future Work

- **Dependency on RGB Frame Alignment**: The distillation approach requires DAVIS cameras or similar frame-event aligned devices; it cannot be directly applied in pure event-camera settings.
- **Fixed VFM Backbone**: Experiments are limited to DAv2 ViT-Small/Large; larger models or alternative VFMs (e.g., Metric3D) remain unexplored.
- **Information Loss in Tencode**: Three-channel RGB encoding inevitably discards fine-grained temporal information.
- **Inference Speed Not Thoroughly Analyzed**: The overhead of recurrent unrolling and ConvLSTM in DepthAnyEvent-R is not discussed in detail.

## Related Work & Insights

- The large-scale pretraining strategy of Depth Anything v2 (DAv2) forms the foundation of the proposed distillation approach.
- Ablation experiment (B) comparing teacher models shows that teacher model quality directly impacts distillation effectiveness.
- Self-supervised event-based depth estimation (Zhu et al.) avoids annotations but sacrifices accuracy; the distillation scheme achieves a favorable balance between the two.
- This cross-modal distillation paradigm could be generalized to other event-camera tasks, such as optical flow and semantic segmentation.

## Rating ⭐⭐⭐⭐

The method is concise and elegant, with thorough experiments and clearly articulated contributions. The cross-modal distillation paradigm demonstrates strong generalizability and practical value; distillation performance approaches or even surpasses full supervision, validating the feasibility of transferring VFM knowledge to the event domain. The recurrent architecture design is principled and natural. A limitation lies in insufficient exploration of alternative VFMs and event representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[CVPR 2026\] No Calibration, No Depth, No Problem: Cross-Sensor View Synthesis with 3D Consistency](../../CVPR2026/3d_vision/no_calibration_no_depth_no_problem_cross-sensor_view_synthesis_with_3d_consisten.md)
- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](hort_monocular_hand-held_objects_reconstruction_with_transformers.md)

</div>

<!-- RELATED:END -->
