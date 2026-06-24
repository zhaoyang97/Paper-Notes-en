---
title: >-
  [Paper Note] PromptHMR: Promptable Human Mesh Recovery
description: >-
  [CVPR 2025][3D Vision][Human Mesh Recovery] PromptHMR proposes a Transformer-based promptable human pose and shape estimation method. By combining spatial prompts (bounding boxes, segmentation masks) and semantic prompts (language descriptions, interaction labels), it flexibly guides full-image 3D human reconstruction, achieving SOTA performance on multiple benchmarks and supporting video-based motion estimation in world coordinates.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Mesh Recovery"
  - "Promptable Estimation"
  - "Multi-modal Prompts"
  - "SMPL-X"
  - "Human Interaction"
date: 2026-05-08
content_hash: f3809a13eefa6a12
---

# PromptHMR: Promptable Human Mesh Recovery

**Conference**: CVPR 2025  
**arXiv**: [2504.06397](https://arxiv.org/abs/2504.06397)  
**Code**: [https://yufu-wang.github.io/phmr-page](https://yufu-wang.github.io/phmr-page)  
**Area**: 3D Vision / Human Pose Estimation  
**Keywords**: Human Mesh Recovery, Promptable Estimation, Multi-modal Prompts, SMPL-X, Human Interaction

## TL;DR

PromptHMR proposes a Transformer-based promptable human pose and shape estimation method. By combining spatial prompts (bounding boxes, segmentation masks) and semantic prompts (language descriptions, interaction labels), it flexibly guides full-image 3D human reconstruction, achieving SOTA performance on multiple benchmarks and supporting video-based motion estimation in world coordinates.

## Background & Motivation

**Background**: The classic paradigm of 3D human pose and shape (HPS) estimation is "pixel-to-parameters"—regressing SMPL parameters from tightly cropped human images. Crop-based methods offer high precision but lose scene context, while full-image methods retain context but often suffer from poor detection and accuracy.

**Limitations of Prior Work**: (1) Crop-based methods cannot exploit scene context, leading to limited performance in scenes with occlusions, dense crowds, or human-human interactions. (2) Full-image methods face the dual challenges of missed detections and insufficient precision. (3) Recent language-based methods (e.g., ChatPose) attempt to integrate Vision-Language Models (VLMs), but their 3D accuracy is far below SOTA. (4) Body shape estimation suffers from severe perspective ambiguities in monocular views.

**Key Challenge**: VLMs have strong semantic understanding of humans in images but lack 3D comprehension, whereas metric regressors have deep understanding of 3D human bodies but lack semantic awareness. How can these two approaches complement each other?

**Goal**: Design a unified HPS framework capable of accepting various types of "side information" as prompts. This framework should handle full-image context while leveraging different prompts to enhance robustness and accuracy across diverse scenes.

**Key Insight**: Redefine HPS as a prompt-guided regression task—given an image and multi-modal prompts (bounding boxes, masks, text, interaction labels), regress the SMPL-X parameters for each prompted individual.

**Core Idea**: Replace traditional detection with spatial prompts, supplement insufficient visual information with semantic prompts (e.g., shape descriptions to improve body shape estimation), and design switchable cross-person attention layers to handle interaction scenarios, achieving flexible, robust, and high-precision multi-person HPS.

## Method

### Overall Architecture

The input consists of a full 896×896 image and a set of prompts for each person. The image is processed through a DINOv2 ViT encoder to extract features (executed only once per frame, independent of the number of people). Optional segmentation masks are downsampled via a convolutional encoder and added to the image tokens. Various types of prompts (bounding boxes, text descriptions, interaction labels) are mapped into tokens of a unified dimension using a prompt encoder. The SMPL-X decoder feeds the image features and prompt tokens into a Transformer decoder to regress the SMPL-X parameters (orientation, pose, shape, translation) for each person.

### Key Designs

1. **Multi-modal Prompt Encoder**:

    - **Function**: Encode different types of input prompts into token vectors of the same dimension.
    - **Mechanism**: Bounding boxes are generated as two tokens $T_{bi} \in \mathbb{R}^{2 \times d}$ using positional encodings + learned embeddings. During training, full-body boxes, face boxes, and truncated boxes are simulated with Gaussian noise, enabling the model to adaptively accept arbitrary box types at inference time. Text descriptions (e.g., "tall muscular male") generate a token $T_{ti} \in \mathbb{R}^d$ through a CLIP text encoder; training descriptions are automatically generated using the body shape-to-attributes method from SHAPY. Interaction labels $k_i$ control the activation of the cross-person attention layer. Missing prompts are replaced with learned empty tokens, with different prompt modalities randomly masked during training.
    - **Design Motivation**: Unified encoding allows the model to flexibly accept different combinations of inputs, and random-mask training ensures the model can work with any subset of prompts during testing.

2. **SMPL-X Decoder and Position Regression**:

    - **Function**: Regress SMPL-X parameters and 3D locations for each person from image features and prompt tokens.
    - **Mechanism**: The decoder consists of three attention blocks, each containing self-attention, cross-person attention (optional), and bidirectional cross-attention. Separate query tokens $T_{smpl}$ and $T_{depth}$ are used to regress pose/shape and position, respectively. Instead of directly regressing translation $\tau$, the model predicts normalized 2D translation $p_{xy}$ and inverse depth $p_z$, which are then converted using focal length: $t_{xy} = p_{xy}/p_z$, $t_z = (1/p_z) \cdot (f/f_c)$.
    - **Design Motivation**: Since the representation spaces of position and pose differ significantly, separating the tokens prevents the position representation from being influenced by the 3D pose. Inverse depth is linearly related to the scale of the person in the image, making its prediction more intuitive and stable.

3. **Promptable Cross-Person Interaction**:

    - **Function**: Model two-person interactions within the decoder to improve estimation accuracy in close-contact scenes.
    - **Mechanism**: Implemented as flow control with residual connections. When the interaction label $k_i$ is active, the query tokens of both individuals undergo information exchange through an additional self-attention layer; otherwise, this layer is bypassed. Tokens are augmented with positional encodings to distinguish individual identities, and the attention output is merged via residual connections. The CHI3D and HI4D two-person interaction datasets are used during training.
    - **Design Motivation**: Applying attention to all individuals can create unnecessary dependencies in crowded scenes. The switchable design activates attention only when interaction annotations are available, avoiding issues caused by limited data diversity and enhancing flexibility.

### Loss & Training

A joint 2D and 3D loss is used: $\mathcal{L} = \lambda_1\mathcal{L}_{2D} + \lambda_2\mathcal{L}_{3D} + \lambda_3\mathcal{L}_{SMPL} + \lambda_4\mathcal{L}_V + \lambda_5\mathcal{L}_t$. $\mathcal{L}_{2D}$ is the reprojected 2D joint error, $\mathcal{L}_{3D}$ is the 3D joint error, $\mathcal{L}_{SMPL}$ is the SMPL parameter error, $\mathcal{L}_V$ is the vertex error, and $\mathcal{L}_t$ is the normalized translation and inverse depth error.

Training uses standard datasets such as BEDLAM, AGORA, 3DPW, COCO, and MPII, along with CHI3D and HI4D interactive datasets. The model is optimized using AdamW with a batch size of 96 and image resolution of 896×896, converging after 350K steps.

## Key Experimental Results

### Main Results (Camera Space Reconstruction PA-MPJPE mm)

| Method | 3DPW | EMDB | RICH | Type |
|------|------|------|------|------|
| CLIFF | 43.0 | 68.3 | 68.1 | Crop |
| HMR2.0a | 44.4 | 61.5 | 60.7 | Crop |
| CameraHMR | 35.1 | 43.3 | 34.0 | Crop |
| Multi-HMR | 45.9 | 50.1 | 46.3 | Full |
| **PromptHMR** | **36.6** | **41.0** | **37.3** | Full |
| **PromptHMR-Vid** | **35.5** | **40.1** | **37.0** | Video |

### Ablation Study (Body Shape Prompt, HBW Dataset)

| Train w/ text | Test w/ text | Height | Chest | Waist | Hip |
|-------------|-------------|--------|-------|-------|-----|
| ✗ | ✗ | 69 | 51 | 88 | 63 |
| ✓ | ✗ | 69 | 48 | 86 | 60 |
| ✓ | ✓ | **62** | **43** | **76** | **58** |

### Key Findings

- The full-image method PromptHMR matches or even outperforms crop-based methods (e.g., CameraHMR) in PA-MPJPE, demonstrating that promptable methods can effectively leverage scene context without sacrificing precision.
- Using text prompts only during training already improves body shape accuracy, and providing text at test time further enhances performance (reducing Height error from 69 to 62 mm).
- The interaction layer improves two-person metrics (Pair-PA-MPJPE from 87.2 to 73.0 mm) even when HI4D training data is not utilized, showing strong cross-domain generalization.
- Mask prompts outperform bounding boxes in close-contact scenarios (due to box ambiguity, where masks offer better precision).
- PromptHMR-Vid (video version) combined with TRAM's metric SLAM achieves SOTA in world-coordinate motion estimation.

## Highlights & Insights

- **Paradigm Shift**: Redefining HPS from "pixel-to-parameters" to "pixel + prompt-to-parameters" opens up new opportunities for collaboration with VLMs.
- **Practical Multi-modal Synergy**: Different prompt types (boxes, masks, text, interactions) excel in different scenarios, and training with random masking allows the model to adapt flexibly.
- **Separating Position and Pose**: This design is simple but highly effective, and the inverse depth representation successfully borrows insights from monocular depth estimation literature.

## Limitations & Future Work

- Shape descriptions and interaction prompts currently need to be provided manually; future work should integrate VLMs to generate prompts automatically.
- Monocular regression methods still inevitably suffer from interpenetration during close physical interactions.
- The model only focuses on body pose and does not model facial or hand parameters.
- Incorporating other types of side information (e.g., action descriptions, 3D scene context, physical body measurements) could provide additional benefits in different scenarios.

## Related Work & Insights

- Sharing the same philosophy as SAM's "promptable segmentation," this work proves that the prompting paradigm is also effective for regression tasks.
- VLM-based methods like ChatPose point in the right direction but lack precision. PromptHMR provides a more practical path for "VLM + metric regressor" collaboration.
- Optimization-based methods like BUDDI handle interactions in a post-processing stage; in contrast, PromptHMR models interactions directly during the regression stage in a more end-to-end manner.

## Rating

- **Novelty**: 8/10 — The design of the promptable HPS framework is novel, and the multi-modal synergy is simple yet effective.
- **Experimental Thoroughness**: 9/10 — Covered multiple datasets with extensive ablation studies on different prompt types.
- **Writing Quality**: 9/10 — Clear motivation and highly informative experimental presentations.
- **Value**: 9/10 — Substantially advances the field of human perception, and the prompting paradigm can be generalized to other regression tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MEGA: Masked Generative Autoencoder for Human Mesh Recovery](mega_masked_generative_autoencoder_for_human_mesh_recovery.md)
- [\[CVPR 2025\] HeatFormer: A Neural Optimizer for Multiview Human Mesh Recovery](heatformer_a_neural_optimizer_for_multiview_human_mesh_recovery.md)
- [\[ICCV 2025\] AJAHR: Amputated Joint Aware 3D Human Mesh Recovery](../../ICCV2025/3d_vision/ajahr_amputated_joint_aware_3d_human_mesh_recovery.md)
- [\[ICCV 2025\] Fish2Mesh Transformer: 3D Human Mesh Recovery from Egocentric Vision](../../ICCV2025/3d_vision/fish2mesh_transformer_3d_human_mesh_recovery_from_egocentric_vision.md)
- [\[ECCV 2024\] Global-to-Pixel Regression for Human Mesh Recovery](../../ECCV2024/3d_vision/global-to-pixel_regression_for_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
