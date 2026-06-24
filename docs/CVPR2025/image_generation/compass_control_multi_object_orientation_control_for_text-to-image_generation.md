---
title: >-
  [Paper Note] Compass Control: Multi Object Orientation Control for Text-to-Image Generation
description: >-
  [CVPR 2025][Image Generation][Orientation Control] Proposes Compass Control, which achieves precise 3D orientation control for multiple objects in text-to-image diffusion models by introducing a lightweight orientation encoder to predict compass tokens combined with a Coupled Attention Localization (CALL) mechanism. Requiring only synthetic data for training, it generalizes to unseen categories and multi-object scenes.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Orientation Control"
  - "Text-to-Image Generation"
  - "Diffusion Models"
  - "Multi-Object Scenes"
  - "Attention Constraints"
date: 2026-05-08
content_hash: 6aca600f18f11902
---

# Compass Control: Multi Object Orientation Control for Text-to-Image Generation

**Conference**: CVPR 2025  
**arXiv**: [2504.06752](https://arxiv.org/abs/2504.06752)  
**Code**: None  
**Area**: 3D Vision / Controllable Image Generation  
**Keywords**: Orientation Control, Text-to-Image Generation, Diffusion Models, Multi-Object Scenes, Attention Constraints

## TL;DR

Proposes Compass Control, which achieves precise 3D orientation control for multiple objects in text-to-image diffusion models by introducing a lightweight orientation encoder to predict compass tokens combined with a Coupled Attention Localization (CALL) mechanism. Requiring only synthetic data for training, it generalizes to unseen categories and multi-object scenes.

## Background & Motivation

### Background

**Background**: Current text-to-image (T2I) diffusion models struggle to precisely control the 3D orientation of objects through text (e.g., "facing right" is semantically ambiguous), requiring tedious prompt engineering.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing 3D control methods either require dense 3D information (multi-view images, 3D bounding boxes) or are limited to simple single-object scenes.

### Key Challenge

**Key Challenge**: The lack of a user-friendly interface that allows specifying the orientation angle for each object while generating multi-object scenes.

### Core Idea

**Core Idea**: Core Problem: How to achieve object-level decoupled orientation control while preserving the original generation capability of the T2I model?

## Method

### Overall Architecture

Compass Control introduces orientation-aware compass tokens $\mathbf{c}_n$ into the text embedding space of Stable Diffusion, which are predicted by a lightweight MLP encoder $\mathcal{P}$ based on the orientation angle $\theta_n$. The compass tokens are inserted into the prompt text (e.g., "A photo of $\mathbf{c}_1$ jeep and $\mathbf{c}_2$ sedan"), passed through the text encoder, and then used to condition the denoising U-Net. Meanwhile, LoRA is used to fine-tune the U-Net, which is trained on a synthetic dataset.

### Key Designs

1. **Compass Token Orientation Encoding**:
    - **Function**: Encodes the object orientation angle into a token in the text embedding space.
    - **Core Idea**: Uses a three-layer MLP (with ReLU) to map the orientation angle $\theta$ to the embedding of the text encoder's input space, prepended before the corresponding object token.
    - **Design Motivation**: Embeds orientation as an object attribute within the text conditioning, keeping the original T2I model interface intact and supporting multi-object scenes.

2. **Coupled Attention Localization (CALL)**:
    - **Function**: Constrains the cross-attention regions of compass tokens and object tokens, achieving decoupled object orientation.
    - **Core Idea**: During both training and inference, a relaxed 2D bounding box is used to generate a binary mask $m$ (0 inside the box, $-\infty$ outside) to perform a masking operation on cross-attention: $\Psi(\mathbf{c}_n) = \text{softmax}(m + QK(\mathbf{c}_n)^T / \sqrt{d_K})$.
    - **Design Motivation**: Direct training causes the compass token to attend to irrelevant regions, leading to orientation control failure. CALL binds the compass token with its corresponding object token to the same region, achieving decoupled control.

3. **Multi-Stage Training Strategy**:
    - **Function**: Ensures objects are generated within the bounding boxes, laying the foundation for CALL.
    - **Core Idea**: First trains on single-object scenes to learn bounding box compliance, then continues training with a mixture of single- and dual-object scenes.
    - **Design Motivation**: Single-stage training fails to make objects comply with bounding box constraints, causing objects to leak into neighboring regions in multi-object layouts.

### Loss & Training

- Standard diffusion loss is used to train the encoder $\mathcal{P}$ and LoRA weights.
- Synthetic dataset: 10 types of 3D assets × multiple layout orientations, rendered in Blender (1,000 single-object + 7,900 dual-object scenes).
- ControlNet augmentation: Canny edge maps are used to condition ControlNet to generate diverse backgrounds, avoiding overfitting to solid black backgrounds.
- Training configuration: SD v2.1, LoRA rank 4, batch size 4, learning rate $10^{-4}$, 25K steps, relaxation coefficient $\lambda = 1.2$.

## Key Experimental Results

### Main Results

| Method | Text Align. ↑ | % Obj. Generated ↑ | Angular Err. ↓ |
|------|--------------|-------------------|----------------|
| ViewNeTI | 22.12 | 0.920 | 0.596 |
| Cont-3D-Words | 29.88 | 0.732 | 0.509 |
| LooseControl | 31.60 | 0.656 | 0.385 |
| **Ours (Single Object)** | **32.98** | **0.968** | **0.198** |
| LooseControl (Multi-Object) | 31.73 | 0.778 | 0.372 |
| **Ours (Multi-Object)** | **33.93** | **0.964** | **0.215** |

### Ablation Study

| Configuration | Effect | Explanation |
|------|------|------|
| Without CALL | Poor orientation control, object entanglement | compass token attends to irrelevant regions |
| Single-stage Training | Fewer multi-object generations | Objects fail to comply with bounding boxes, leading to leakage |
| Without ControlNet Augmentation | Overfitting to black backgrounds | Synthetic data lacks background diversity |

### Key Findings

- Trained only on 1-2 object scenes, yet generalizes to complex scenes with 3-5 objects.
- Generalizes to unseen categories from the training set (e.g., strollers, boats, humans, etc.).
- Combined with DreamBooth, personalized orientation control is achieved with only about 10 pose-free images.
- Outperforms baseline methods across all dimensions in a user study (57 participants).

## Highlights & Insights

- Encodes 3D orientation control as attribute tokens in the text embedding space, providing an elegant design that preserves the original capabilities of the T2I model.
- The CALL mechanism is simple and effective: it achieves both orientation binding and object decoupling simultaneously through attention masking.
- Requires synthetic data from only 10 types of 3D assets to train a model with strong generalization capability.
- Discovers that T2I diffusion models inherently possess some level of 3D understanding.

## Limitations & Future Work

- Control fails under severe occlusion or overlap, potentially leading to missing objects or mixed attributes.
- Single-angle orientation parameterization is overly simplified for non-rigid bodies (e.g., humans).
- Requires 2D bounding boxes at inference time (although they can be generated heuristically).
- Can be extended to 3-DoF orientation control and more complex 3D attributes.

## Related Work & Insights

- Follows the orientation control ideas of ViewNeTI, Continuous 3D Words, etc., but scales up to multi-object scenes.
- The CALL mechanism relates to attention constraint works such as Attend-and-Excite but is specifically designed for object-orientation token binding.
- Personalized extension uses the DreamBooth framework, demonstrating the generalizability of the conditioning mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever framework design combining orientation tokens and attention localization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete ablation, sufficient user study, and rich extended experiments such as personalization.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Practical application value in creative design and 3D content generation fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multitwine: Multi-Object Compositing with Text and Layout Control](multitwine_multi-object_compositing_with_text_and_layout_control.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)
- [\[CVPR 2025\] MVPortrait: Text-Guided Motion and Emotion Control for Multi-View Vivid Portrait Animation](mvportrait_text-guided_motion_and_emotion_control_for_multi-view_vivid_portrait_.md)
- [\[CVPR 2025\] GPS as a Control Signal for Image Generation](gps_as_a_control_signal_for_image_generation.md)

</div>

<!-- RELATED:END -->
