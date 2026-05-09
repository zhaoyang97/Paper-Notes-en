---
title: >-
  [Paper Note] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation
description: >-
  [ICCV 2025][3D Vision][robotic manipulation] This paper proposes RoboTron-Mani, a multimodal large model for robotic manipulation, together with the comprehensive dataset RoboData. By enhancing 3D perception via camera parameters and occupancy supervision, and enabling flexible multimodal fusion through a Modality-Isolation-Mask (MIM), RoboTron-Mani is the first generalist policy to simultaneously surpass specialist models across multiple datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - robotic manipulation
  - multimodal large model
  - 3D perception
  - cross-embodiment generalization
  - data alignment
date: 2026-05-08
content_hash: c33a74a20f2a2121
---

# RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation

**Conference**: ICCV 2025
**arXiv**: [2412.07215](https://arxiv.org/abs/2412.07215)
**Code**: [GitHub](https://github.com/EmbodiedAI-RoboTron/RoboTron-Mani)
**Area**: 3D Vision
**Keywords**: robotic manipulation, multimodal large model, 3D perception, cross-embodiment generalization, data alignment

## TL;DR

This paper proposes RoboTron-Mani, a multimodal large model for robotic manipulation, together with the comprehensive dataset RoboData. By enhancing 3D perception via camera parameters and occupancy supervision, and enabling flexible multimodal fusion through a Modality-Isolation-Mask (MIM), RoboTron-Mani is the first generalist policy to simultaneously surpass specialist models across multiple datasets.

## Background & Motivation

Applying large models to robotic manipulation faces two core challenges:

1. **The 2D-to-3D gap**: Existing multimodal large models (e.g., LLaVA, Flamingo) focus primarily on 2D image understanding, whereas robots must interact with physical 3D space. Directly applying 2D multimodal models to embodied intelligence is suboptimal—robots require understanding of spatial depth, occlusion, and 3D geometry for precise manipulation.

2. **High data collection cost**: Collecting approximately 130,000 episodes for RT-1 took 17 months. While cross-platform datasets such as Open X-Embodiment aggregate multiple datasets, they lack critical 3D information (multi-view images, camera parameters, depth maps), and inconsistent coordinate systems and action spaces across datasets mean that naive fusion degrades performance (RT-1-X underperforms RT-1).

These two problems are interrelated: enabling a generalist model to learn effectively from heterogeneous data requires both a unified 3D input representation (to eliminate 2D feature discrepancies caused by differing camera parameters) and an aligned output space (to unify action representations across different robots).

## Method

### Overall Architecture

RoboTron-Mani is built upon the OpenFlamingo architecture. It takes multi-view images $I$, text instructions $T$, and camera parameters $Cam$ as input, and outputs actions $O_A$ along with optional image $O_I$ and occupancy map $O_O$. The pipeline consists of four core components in series: Vision Encoder → 3D Perception Adapter → Feature Fusion Decoder → Multimodal Decoders.

### Key Designs

1. **3D Perception Adapter (UVFormer)**: This module addresses multi-view feature unification and 3D spatial perception. UVFormer transforms image features $X^h$ from $H$ timesteps and $N$ viewpoints, together with corresponding camera parameters $Cam^h$, into a unified view representation:

$$U_I^h = \text{UVFormer}(Q, X^h, Cam^h)$$

where $Q = \{Pos, Emb\}$ is a learnable query, and $Pos \in \mathbb{R}^{L \times B \times 3P}$ defines the positions of a 3D grid within the robot's manipulation space. The key advantage of this design is that the unified view representation $U_I^h$ of a given 3D scene remains consistent regardless of changes in camera parameters, thereby achieving input-space alignment.

2. **Modality-Isolation-Mask (MIM)**: A KQ mask is introduced in the self-attention layers of the Feature Fusion Decoder to control attention interactions among tokens of different modalities (text, image, action, occupancy). Dark cells indicate permitted attention connections; white cells indicate blocked connections. MIM's core value lies in enabling flexible modality fusion—auxiliary modality supervision (image reconstruction, occupancy prediction) can be used during training, while unnecessary modality outputs can be omitted at inference, substantially improving modality composition flexibility.

3. **Multimodal Decoders**: Three distinct decoders are designed to accommodate different output modalities:
    - **Image Decoder**: A 2-layer attention decoder that outputs image patches assembled into complete images (static view or wrist view).
    - **Occupancy Decoder**: Generates features $U_{occ}^h$, then reconstructs the full 3D occupancy $O_o^h = \{o_{pos}^h, o_{rgb}^h\}$ via reshape, upsampling, and 3D convolution.
    - **Action Decoder**: An MLP or DiT block that outputs delta 6D pose $a_{pose}^h$ and 1-DoF gripper action $a_g^h$.

4. **RoboData Data Alignment**: Ten datasets including CALVIN, Meta-World, LIBERO, and RT-1 are integrated with three-dimensional alignment:
    - **3D spatial alignment**: A unified world coordinate system (X→right, Y→forward, Z→up), with the workspace constrained to $[-0.5, -0.5, 0]$ to $[0.5, 0.5, 1]$.
    - **Action representation alignment**: Actions are uniformly regenerated using the Composite Rotation Matrix Method (CRMM).
    - **Missing data completion**: Original simulation environments are reconstructed to supplement missing camera intrinsics and extrinsics.

### Loss & Training

The composite loss function is:

$$l = l_a + \lambda_{\text{image}}(l_{simg} + l_{gimg}) + \lambda_{\text{occ}} l_o$$

- **Action loss** $l_a$: MSE for pose, BCE for gripper.
- **Image loss** $l_{simg}, l_{gimg}$: L2 loss between predicted next frame and ground truth.
- **Occupancy loss** $l_o$: Position MSE + RGB MSE (weighted by $\lambda_{rgb}$).

Training details: 4B parameters (bf16), 32×A100 GPUs, ~50 hours, 2.1M samples, 10 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | RoboTron-Mani | Prev. SOTA | Gain |
|---------|--------|---------------|------------|------|
| LIBERO | Success Rate | **91.7%** | QueST 89.8% | +1.9% |
| RoboCasa | Success Rate | **47.4%** | GR00T-N1 40.9% | +6.5% |
| CALVIN | Avg Len | **3.51** | MDT 93.7% (Task1) | Competitive |
| Meta-World | Success Rate | **80.1%** | PRISE 80.4% | On par |
| RT-1 | Success Rate | **60.0%** | RT-2-X (55B) 60.7% | On par (far fewer params) |

Note: RoboTron-Mani is the only generalist policy evaluated simultaneously across all five datasets; all baselines are specialist models optimized for individual datasets.

### Ablation Study

| Configuration | Task1 | Task2 | Task3 | Task4 | Task5 | Avg Len | Note |
|---------------|-------|-------|-------|-------|-------|---------|------|
| Baseline | 81.0% | 48.1% | 25.7% | 14.5% | 8.6% | 1.77 | Last-frame action only |
| +FFA | 85.0% | 63.3% | 42.0% | 28.7% | 18.8% | 2.37 | Per-frame action output |
| +FFA+Image | 88.5% | 74.7% | 60.7% | 49.1% | 39.6% | 3.13 | With image prediction |
| +FFA+UVFormer | 94.2% | 74.7% | 55.1% | 38.3% | 25.8% | 2.88 | 3D perception |
| +All (MLP) | 94.7% | 80.3% | 65.1% | 51.4% | 39.0% | 3.31 | Full framework |
| +All (DiT) | **96.9%** | **83.0%** | **68.1%** | **56.5%** | **46.8%** | **3.51** | DiT action head |

### Key Findings

- UVFormer yields the most significant gain on the first task (81% → 94.2%), demonstrating that 3D spatial understanding is critical for task initiation.
- Auxiliary modality supervision substantially improves action performance even when the generated images and occupancy maps are of low quality.
- Data alignment is crucial for cross-dataset training: without alignment, LIBERO accuracy is only 64.2%; after alignment it reaches 90.7%.
- The DiT action head holds a clear advantage over the MLP head on long-horizon tasks (Avg Len: 3.31 → 3.51).

## Highlights & Insights

- **First generalist policy to comprehensively surpass specialist models**: Joint training and evaluation across five heterogeneous datasets challenges the conventional wisdom that generalist policies underperform specialists.
- **3D perception is key to cross-embodiment generalization**: While the same 3D scene yields different 2D features under different camera parameters, UVFormer's 3D features remain consistent.
- **Elegant MIM design**: Auxiliary modality supervision enhances learning during training while allowing flexible pruning at inference—a cost-free performance boost.
- **Deep investment in data engineering**: Hundreds of person-days were spent aligning data and completing missing modalities, an investment that is empirically validated as worthwhile.

## Limitations & Future Work

- The data alignment strategy is currently validated only in simulation; aligning heterogeneous real-world data presents greater challenges.
- Training costs remain high (32×A100, 50 hours) for a 4B-parameter model.
- Auxiliary modality generation quality is low; improving generation fidelity may yield further performance gains.
- Online learning and adaptive capabilities are not explored.

## Related Work & Insights

- Comparison with Open X-Embodiment demonstrates that careful spatial alignment combined with architectural design outperforms naive data fusion.
- The cross-attention mechanism in OpenFlamingo naturally accommodates multi-frame and video inputs, offering advantages over the autoregressive mechanism in LLaVA.
- The alignment methodology of RoboData is generalizable to the unification of a broader range of robotic datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of 3D perception and MIM is novel; the data alignment scheme is systematic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across five datasets with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and well-formatted equations, though some sections are verbose.
- **Value**: ⭐⭐⭐⭐⭐ Provides a complete data-plus-model solution for cross-embodiment robot learning.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](../../CVPR2026/3d_vision/hyperbolic_multiview_pretraining_for_robotic_manipulation.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](../../CVPR2026/3d_vision/ada3drift_adaptive_training-time_drifting_for_one-step_3d_visuomotor_robotic_man.md)
- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](../../NeurIPS2025/3d_vision/dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[NeurIPS 2025\] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent](../../NeurIPS2025/3d_vision/sceneweaver_all-in-one_3d_scene_synthesis_with_an_extensible_and_self-reflective.md)

<!-- RELATED:END -->
