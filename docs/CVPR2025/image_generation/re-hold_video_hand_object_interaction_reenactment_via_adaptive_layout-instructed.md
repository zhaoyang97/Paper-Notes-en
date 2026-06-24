---
title: >-
  [Paper Note] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Hand-Object Interaction] This work proposes Re-HOLD, the first human-centric hand-object interaction (HOI) video reenactment framework, which decouples hand and object modeling through a decoupled layout representation, and combines an interactive texture enhancement module with an adaptive layout adjustment strategy to achieve high-fidelity HOI video generation across different objects.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Hand-Object Interaction"
  - "Video Reenactment"
  - "Layout Guidance"
  - "Diffusion Models"
  - "Texture Enhancement"
date: 2026-05-08
content_hash: 9d1c600656f37954
---

# Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2503.16942](https://arxiv.org/abs/2503.16942)  
**Code**: [Project Page](https://fyycs.github.io/Re-HOLD)  
**Area**: Image Generation  
**Keywords**: Hand-Object Interaction, Video Reenactment, Layout Guidance, Diffusion Models, Texture Enhancement

## TL;DR

This work proposes Re-HOLD, the first human-centric hand-object interaction (HOI) video reenactment framework, which decouples hand and object modeling through a decoupled layout representation, and combines an interactive texture enhancement module with an adaptive layout adjustment strategy to achieve high-fidelity HOI video generation across different objects.

## Background & Motivation

The advancement of digital human video technology drives the demand for generating hand-object interaction (HOI) scenes. However, HOI video synthesis faces three core challenges:

- **Entanglement caused by hand-object occlusion**: Physical contact between the hand and the object creates complex occlusions, which easily lead to artifacts at the hand-object interface.
- **Difficulty in detailed texture recovery**: Hands and objects have high degrees of freedom but occupy only limited pixels in the frame, making precise texture recovery extremely challenging.
- **Cross-object size discrepancies**: Variations in the shape and size of different objects affect the interaction positions; if the motion sequence remains unchanged, it results in unnatural grasping.
- **Limitations of Prior Work**: HOI-Swap only supports object-centric single-hand grasping video editing, failing to handle two-handed interaction scenarios.

## Method

### Overall Architecture

Re-HOLD adopts a dual-branch architecture: the Reference U-Net processes the target object image to extract texture information, while the Denoising U-Net receives noisy latent variables and layout guidance for diffusion processing. On top of this, an HOI restoration module is superimposed for hand structure recovery and texture refinement. A two-stage training strategy is adopted: image-level HOI modeling in the first stage, and temporal consistency modeling in the second stage.

### Key Design 1: Decoupled Layout Representation — Decoupling Hand and Object Positional Information

**Function**: Achievement of hand-object decoupling through bounding box representations with different properties, enabling the model to adapt to cross-object reenactment for different objects.

**Mechanism**: The layout representation of each frame consists of three detection bounding boxes: two **fixed-size square** hand boxes (providing only positional information, independent of pose and size) and one **variable-size** object box (varying based on object and depth). A 4-layer lightweight convolutional encoder extracts layout features $\mathbf{F}_l$, which are combined with Gaussian noise $\epsilon_t$ and input into the Denoising U-Net.

**Design Motivation**: The pose/size-independence of the hand boxes decouples positional information from motion signals, allowing the model to focus on hand locations rather than binding to specific gestures; the variable size of the object box provides shape and positional guidance. This sparse yet effective decoupling lays the foundation for subsequent finer HOI modeling.

### Key Design 2: HOI Interaction Texture Enhancement Module — Dual Memory Banks for Recovering Hand and Object Details

**Function**: Recovery of fine details in hand poses and object textures via independent learnable memory banks.

**Mechanism**: Hand memory bank $\mathbf{B}_h \in \mathbb{R}^{N_h \times C_h}$ and object memory bank $\mathbf{B}_o \in \mathbb{R}^{N_o \times C_o}$ are constructed, integrating Hand-Attention and Object-Attention layers into the U-Net architecture. The attention computation uses the corresponding hand/object mask $M$ to restrict the region of effect:

$$\mathbf{F}^a = \text{Att}(\mathbf{F}, \mathbf{B}, \mathbf{B}) * M + \mathbf{F}$$

Simultaneously, ControlNet is used to encode 3D hand meshes $V^h$ (reconstructed by HaMeR) for structural guidance, and random offset augmentation is applied to hand positions during training to eliminate over-dependence.

**Design Motivation**: Relying solely on layouts and reference images is insufficient to recover finger details and object textures. The global memory banks accumulate rich priors of hand poses and object textures during training, which can effectively supplement missing information during inference.

### Key Design 3: Adaptive Layout Adjustment Strategy — Handling Cross-Object Size Discrepancies

**Function**: Adaptive layout adjustment during the inference stage to avoid unreasonable physical contact caused by size discrepancies between target and source objects.

**Mechanism**: A four-step adjustment process: (1) Initialize the four side centers of the object box as potential contact points, and determine contact relationships via H2O distance (the distance from the hand box center to the nearest contact point); (2) Keep the center of the object box unchanged while adjusting width and height according to target object dimensions; (3) Adjust hand box positions horizontally to maintain the original H2O distance; (4) Move the object box to align its bottom with the original box bottom, preventing floating effects.

**Design Motivation**: When there is a significant size discrepancy between the target and source objects, directly using the source motion sequence causes failure in proper hand-object contact or results in unreasonable grasping positions. Adaptive adjustment ensures the plausibility of physical interactions.

### Loss & Training

Based on the standard noise prediction loss of Stable Diffusion $\mathbf{L} = \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, c, t)\|_2^2]$, during the late stage of training, the $L_1$ loss of only hand and object regions is computed every 10 iterations to emphasize HOI areas.

## Key Experimental Results

### Main Results: Cross-Object Reenactment (Our dataset)

| Method | Hand Fid. ↑ | Subj. Cons. ↑ | Mot. Smth. ↑ |
|------|------------|--------------|-------------|
| AnyV2V | 0.934 | 0.829 | 0.983 |
| VideoSwap | 0.936 | 0.922 | 0.992 |
| HOI-Swap | 0.994 | 0.944 | 0.994 |
| **Re-HOLD** | **0.994** | **0.955** | **0.994** |

### Self-Reenactment

| Method | PSNR ↑ | FID ↓ | Hand Agr. ↑ |
|------|--------|-------|------------|
| HOI-Swap | 31.634 | 30.932 | 0.725 |
| RealisDance | 32.784 | 26.337 | 0.749 |
| **Re-HOLD** | **33.451** | **19.021** | **0.773** |

### User Study (Scale 1-5)

| Method | HOI Consistency | Object Consistency | Temporal Consistency |
|------|----------|----------|----------|
| HOI-Swap | - | - | - |
| **Re-HOLD** | **Highest** | **Highest** | **Highest** |

### Key Findings

- Re-HOLD significantly outperforms all methods in FID (19.021 vs. 26.337 of the second best), substantially improving image quality.
- Hand fidelity and hand agreement both achieve optimal performance, demonstrating the effectiveness of the HOI texture enhancement module.
- Adaptive layout adjustment enables the generation of plausible hand-object interactions even with massive discrepancies in object sizes.

## Highlights & Insights

1. **Simplicity and elegance of decoupled layout representation**: Decoupling hands and objects using three bounding boxes, where the pose/size-independence of the hand boxes is a key innovation.
2. **Pragmatic and efficient two-stage training strategy**: Modeling image-level HOI first followed by temporal consistency, reducing training complexity.
3. **Adaptive layout adjustment strategy solves the core challenge of cross-object generalization.**
4. **First human-centric two-handed HOI video reenactment framework**, filling a research gap.

## Limitations & Future Work

- The training data consists of only 9 subjects and 14 objects, potentially limiting generalization to more diverse scenarios.
- It depends on the 3D hand reconstruction quality of HaMeR; failure in estimation affects the generation results.
- Object segmentation relies on the LISA model, which may be inaccurate under complex backgrounds.
- Future work could explore incorporating physical engine constraints for more realistic grasp mechanics.

## Related Work & Insights

- **HOI-Swap**: Video object replacement via diffusion models, but only supports single-handed, object-centric scenarios.
- **AnimateAnyone / RealisDance**: Human animation methods, which do not handle hand-object interactions.
- **HaMeR**: Provides 3D hand mesh estimation, establishing the foundation for structural hand guidance.
- **InteractDiffusion**: The source of inspiration for capturing interaction relationships through layout inputs.

## Rating

⭐⭐⭐⭐ — Successfully addresses the core technical challenges of HOI video reenactment, with clever designs for layout decoupling and adaptive adjustment strategies. The experimental results are comprehensive, significantly surpassing existing methods across multiple metrics. However, the limited scale of training data requires larger-scale validation for generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)

</div>

<!-- RELATED:END -->
