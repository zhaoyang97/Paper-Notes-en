---
title: >-
  [Paper Note] Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine
description: >-
  [AAAI 2026][3D Vision][3D-aware image editing] FFSE is proposed—an autoregressive 3D-aware image editing framework based on video diffusion models. Combined with a hybrid dataset 3DObjectEditor (real + synthetic), it enables multi-round object translation, scaling, and rotation on real images, similar to a 3D engine. It simultaneously generates realistic background effects such as shadows, reflections, and occlusions, maintaining consistency across editing rounds. It signific…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D-aware image editing"
  - "multi-round editing"
  - "autoregressive generation"
  - "diffusion models"
  - "object manipulation"
date: 2026-05-08
content_hash: 65a133ff9b7f5f2a
---

# Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine

**Conference**: AAAI 2026  
**arXiv**: [2511.13713](https://arxiv.org/abs/2511.13713)  
**Code**: [https://github.com/FudanCVL/FFSE](https://github.com/FudanCVL/FFSE)  
**Area**: 3D Vision  
**Keywords**: 3D-aware image editing, multi-round editing, autoregressive generation, diffusion models, object manipulation

## TL;DR
FFSE is proposed—an autoregressive 3D-aware image editing framework based on video diffusion models. Combined with a hybrid dataset 3DObjectEditor (real + synthetic), it enables multi-round object translation, scaling, and rotation on real images, similar to a 3D engine. It simultaneously generates realistic background effects such as shadows, reflections, and occlusions, maintaining consistency across editing rounds. It significantly outperforms existing methods in both single-round and multi-round editing.

## Background & Motivation
**Background**: Text-driven image editing (e.g., InstructPix2Pix) excels at semantic editing (appearance/style modification), while drag-based methods (e.g., DragDiffusion) utilize source-target point pairs for non-rigid deformation. A few approaches attempt 3D-aware object manipulation. Image-space methods (e.g., 3DIT, Zero-1-to-3) learn 3D priors from synthetic data, whereas 3D-space methods (e.g., Diffusion Handles, 3DitScene) reconstruct 3D structures (point clouds/3DGS) from a single image before performing manipulation.

**Limitations of Prior Work**: (1) **Poor object quality**: Image-space methods support limited manipulation types (e.g., 3DIT only supports translation and z-axis rotation) and generalize poorly to real images. 3D-space methods suffer from noisy geometric estimation, leading to low quality in complex operations like rotation. (2) **Lack of background effects**: Existing methods can barely generate environmental interaction effects (shadow movement, reflection changes, occlusion relations) caused by object manipulation. (3) **Inconsistent multi-round editing**: Due to a lack of awareness regarding scene structure changes, accumulated errors over multiple editing rounds lead to severe quality degradation. (4) **Cumbersome user interface**: 3D-space methods require a time-consuming reconstruction process.

**Key Challenge**: Achieving high-quality 3D-aware editing requires understanding the 3D scene structure, but reconstructing 3D structures from a single image is both time-consuming and unreliable. Furthermore, multi-round editing requires the model to track changes in scene states, whereas existing methods are stateless, single-round operations.

**Goal**: Without performing 3D reconstruction, how can a diffusion model directly learn 3D-aware object manipulation? How can scene consistency be maintained in multi-round editing? How can physically plausible background effects be generated?

**Key Insight**: Model the editing process as a learned sequence of 3D transformations, leveraging the motion priors of pre-trained video diffusion models (SVD). Construct a hybrid dataset (real domain + synthetic domain) to provide multi-round editing training sequences, and use Domain LoRA to isolate domain-specific content.

**Core Idea**: Model 3D-aware editing as autoregressive sequence generation. Encode editing history through frame buffers and operation buffers to learn the joint generation of object transformations, background effects, and scene consistency on a video diffusion model.

## Method

### Overall Architecture
FFSE formalizes the editing process as a state transition problem: scene state space $S$, operation space $O = \{o^T, o^S, o^X, o^Y, o^Z\}$ (translation/scaling/xyz rotation), and state transition function $p_{tf}(s'|s,o)$. Given the editing history $h_r = \{(x_i, o_i)\}_{i=0}^{r-1}$, the goal is to model the observation distribution $p(x_r|h_r)$. Based on a pre-trained SVD video generation model, it introduces a frame encoder (to encode historical observations), an operation encoder (to encode operation sequences), context self-attention (to maintain object consistency), and Domain LoRA (to isolate domain content), learned from a hybrid dataset through multi-stage training.

### Key Designs

1. **Operation Encoder**:

    - **Function**: Encodes the source region localization and operation parameters of each round into conditional features, injecting them into the diffusion model to guide the editing behavior.
    - **Mechanism**: The source region (centroid $l_i^p$ + bounding box $l_i^b$) and operation values (normalized pixel offset $o_i^T$, scaling factor $o_i^S$, rotation angle $o_i^{X/Y/Z}$) are encoded into $c_i^{\text{src}}$ and $c_i^{\text{opt}}$ respectively via Fourier embedding + MLP. The encodings of all rounds are concatenated along the sequence dimension and injected into the main branch through operation self-attention: $\hat{v} = \bar{v} + \beta \cdot \tanh(\gamma) \cdot \text{TS}(\text{SelfAttn}([\bar{v}, \text{repeat}([c_{\text{src}}, c_{\text{opt}}])]))$, where $\gamma$ is initialized to 0 (zero initialization) and $\beta$ controls the manipulation intensity during inference.
    - **Design Motivation**: Fourier embedding + MLP can precisely represent continuous operation parameters. Operation self-attention is placed between context self-attention and cross-attention to ensure that operation conditions are injected after spatial features but before text conditions. The zero-initialized $\gamma$ avoids disrupting the generative capacity of the pre-trained model during the early stages of training.

2. **Frame Encoder + Context Self-Attention**:

    - **Function**: Encodes historical observations to capture scene dynamics, and maintains appearance consistency of the edited object through cross-frame attention.
    - **Mechanism**: The frame encoder is a lightweight residual-block network that takes historical observations $\{x_j\}_{j=0}^{r-1}$ and a target region binary mask $M_{\text{tgt}}$ (derived from the target position bounding box) as inputs, adding its output to the downsampled block features. Context Self-Attention (CSA) enhances standard self-attention: $\bar{v}_r = v_r + \lambda M_{\text{tgt}} \text{softmax}(A_{r,r-1} + \frac{Q'_r(K'_{r-1})^T}{\sqrt{d}})V'_{r-1}$. The attention mask $A_{r,r-1}$ restricts the calculation of cross-frame correspondence specifically within the object region.
    - **Design Motivation**: The frame encoder captures structural changes in the overall scene (such as occlusion relationships), while CSA ensures that the object's appearance remains unchanged after manipulation. The target mask $M_{\text{tgt}}$ limits the scope of CSA's influence, preventing interference with non-object regions. During training, $M_{\text{tgt}}$ is randomly omitted (as an all-zero mask) to enable the model to implicitly infer the target location from operations.

3. **Hybrid Dataset 3DObjectEditor + Domain LoRA**:

    - **Function**: Constructs a hybrid dataset supporting multi-round 3D manipulation training while preventing overfitting to domain-specific content.
    - **Mechanism**: **Real Domain** $D_{\text{real}}$—RGBA objects and backgrounds are obtained from the MULAN dataset, composed using the painter's algorithm sorted by depth, supporting translation and scaling (40K sequences of 32 frames). **Synthetic Domain** $D_{\text{syn}}$—rendered using Blender's Cycles ray tracer with over 6,000 3D assets (Objaverse), supporting all 5 types of 3D operations (46K sequences of 32 frames), naturally producing physical effects such as shadows and reflections. Domain LoRAs $DL_{\text{real}}/DL_{\text{syn}}$ are injected exclusively into the CSA layers; the corresponding LoRA is selected based on the training sample's domain, and **all are removed** during inference to preserve the base model's quality.
    - **Design Motivation**: Training with $D_{\text{real}}$ alone lacks rotation support and physical effects, while using only $D_{\text{syn}}$ leads to severe overfitting to synthetic styles (oversaturated colors) on real images. Joint training enables the operation encoding module to learn shared 3D transformation knowledge across domains, while Domain LoRA absorbs domain-specific visual styles so they do not interfere with each other during training.

### Loss & Training
Two-stage training: **Stage 1** jointly trains $\theta$ (parameters of new modules) and $DL_{\text{real}}, DL_{\text{syn}}$ on the full $D_{\text{real}} \cup D_{\text{syn}}$ (80K iterations), using standard diffusion reconstruction loss. **Stage 2** fine-tunes $\theta$ solely on $D_{\text{syn}}$ (loading $DL_{\text{syn}}$, 10K iterations) to enhance the quality of generated background effects. All Domain LoRAs are removed during inference. Sequence length $r$ is uniformly sampled from $[r_{\min}=1, r_{\max}=12]$. Training utilizes the Adam optimizer, 4×A800 GPUs, 512×512 resolution, and a batch size of 8.

## Key Experimental Results

### Main Results

| Setup | Method | PSNR↑ | SSIM×10²↑ | DINO↑ | CLIP↑ |
|------|------|-------|-----------|-------|-------|
| Single-round | 3DIT | 20.12 | 68.76 | 61.38 | 80.96 |
| Single-round | Zero-1-to-3 | 23.84 | 71.97 | 65.42 | 83.27 |
| Single-round | Diffusion Handles | 18.83 | 58.33 | 71.33 | 88.53 |
| Single-round | 3DitScene | 17.67 | 53.39 | 73.69 | 89.11 |
| Single-round | **FFSE (Ours)** | **26.31** | **79.54** | **82.39** | **91.67** |
| Multi-round | Zero-1-to-3 | 19.81 | 64.77 | 61.67 | 82.38 |
| Multi-round | Diffusion Handles | 13.79 | 50.47 | 59.06 | 78.24 |
| Multi-round | **FFSE (Ours)** | **24.96** | **74.99** | **79.51** | **90.42** |

### Ablation Study

| Configuration | PSNR↑ | SSIM×10²↑ | DINO↑ | CLIP↑ | Description |
|------|-------|-----------|-------|-------|------|
| w/ $D_{\text{real}}$ only | 25.86 | 79.31 | 81.92 | 91.11 | Lacks rotation and background effects |
| w/ $D_{\text{syn}}$ only | 24.37 | 74.51 | 73.31 | 86.43 | Overfits to synthetic style |
| w/o Stage 2 | 25.92 | 79.33 | 78.77 | 89.82 | Degraded shadow quality |
| w/o Domain LoRA (a) Remove LoRA | 25.37 | 76.54 | 79.53 | 89.75 | Manipulation fails |
| w/o Domain LoRA (b) Keep LoRA | 24.53 | 73.25 | 74.92 | 88.13 | Generates artifacts |
| w/o CSA | 24.81 | 75.17 | 75.65 | 88.71 | Degraded object appearance consistency |
| **FFSE (Full)** | **26.31** | **79.54** | **82.39** | **91.67** | Full model |

### Key Findings
- FFSE's PSNR outperforms the best baseline by 2.47 (26.31 vs 23.84) in single-round editing. The gap is even wider in multi-round editing (24.96 vs 19.81), indicating that the autoregressive framework is far more robust to cumulative edits than stateless approaches.
- User Study: Background effects score 0.98 (vs only 0.59 for 3DIT), and scene consistency scores 0.91 (vs only 0.12 for Diffusion Handles), validating FFSE's comprehensive superiority in subjective quality.
- CSA contributes the most to object consistency: its removal drops DINO from 82.39 to 75.65 (-8.2%), demonstrating that cross-frame attention is critical for maintaining object identity.
- Clever design of Domain LoRA: Without LoRAs, the operation encoding couples with domain styles, leading to inference failure. Using a single set of LoRAs mixes domain styles, causing artifacts. The dual-LoRA setup with removal during inference is the optimal solution.
- Stage 2 is crucial for background effects: fine-tuning for only 10K iterations significantly improves shadow/reflection quality because the ray-tracing of the synthetic domain provides physically accurate training signals.

## Highlights & Insights
- **Editing as Sequence Generation**: Modeling 3D editing as autoregressive state transitions cleverly utilizes the temporal consistency of pre-trained video models to ensure scene consistency across multi-round editing. This paradigm can be generalized to other image editing tasks that require multi-step operations.
- **Domain LoRA Isolation Strategy**: During training, per-domain LoRAs absorb domain-specific content, and all are removed during inference to preserve the base model's quality. This "multi-LoRA during training, zero-LoRA during inference" design can be transferred to multi-domain training scenarios.
- **Hybrid Dataset Design**: The real domain provides visual diversity and generalization ability, whereas the synthetic domain provides physically accurate lighting effects and rotation manipulation—the two are complementary rather than a simple combination.
- **Occlusion Relationship Recovery**: FFSE can correctly recover previously occluded objects during multi-round editing (e.g., a cup that was hidden reappears once the teapot is moved away), which other methods fail to achieve completely.

## Limitations & Future Work
- **No Support for Non-rigid Deformation**: The method only processes rigid transformations such as translation, scaling, and rotation, and cannot perform non-rigid manipulations like bending or stretching.
- **Limited Sequence Length**: Too many editing steps result in unacceptable memory and computational overhead. Although keeping only a subset of history frames is possible, it sacrifices consistency.
- **512×512 Resolution Limitation**: The current training resolution is relatively low, and high-resolution scenarios require further adaptation.
- **Limited Precision of Manipulation**: The precision of operation parameters (such as rotation angles) is limited by the 3D priors learned from 2D images. Large-angle rotations of complex geometries still exhibit distortion.

## Related Work & Insights
- **vs 3DIT**: 3DIT uses text prompts to control editing, supporting only translation and z-axis rotation with poor generalization on real images. FFSE uses precise operation parameters + 2D bounding boxes as input, supporting all 3D operations.
- **vs Diffusion Handles/3DitScene**: 3D-space methods obtain 3D controllability through reconstructing point clouds/3DGS, but are limited by time-consuming reconstruction and noisy geometric estimation. FFSE is completely reconstruction-free, and Diffusion Handles suffers from severe cumulative errors in multi-round editing (multi-round PSNR of only 13.79).
- **vs Neural Assets**: Neural Assets requires 3D bounding box input and training data is limited to restricted classes. FFSE is more user-friendly by using 2D bounding boxes and covers a much wider range of classes (6000+ 3D assets).
- **Inspiration from Video Generation**: Leveraging the motion priors of SVD is key—video models naturally understand object motion and environmental responses (such as shadow following). Learning the editing sequence as "video frames" is crucial.

## Rating
- Novelty: ⭐⭐⭐⭐ The paradigm of modeling 3D editing as autoregressive sequence generation is novel. The design of Domain LoRA and the hybrid dataset is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compares with 4 methods, includes a user study with 30 participants, and provides comprehensive ablations (data/LoRA/CSA/stages). However, it lacks comparison with more of the latest approaches.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear (4 key challenges), with rich visualization results and a detailed appendix.
- Value: ⭐⭐⭐⭐⭐ Resolves the practical pain point of multi-round 3D editing. The method offers high practicality (reconstruction-free + user-friendly interface), and its capability to generate background effects is a unique contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Ctrl&Shift: High-Quality Geometry-Aware Object Manipulation in Visual Generation](../../ICLR2026/3d_vision/ctrlshift_high-quality_geometry-aware_object_manipulation_in_visual_generation.md)
- [\[ICLR 2026\] SpatialHand: Generative Object Manipulation from 3D Perspective](../../ICLR2026/3d_vision/spatialhand_generative_object_manipulation_from_3d_prespective.md)
- [\[CVPR 2026\] FE2E: From Editor to Dense Geometry Estimator](../../CVPR2026/3d_vision/from_editor_to_dense_geometry_estimator.md)
- [\[ICCV 2025\] DriveX: Driving View Synthesis on Free-form Trajectories with Generative Prior](../../ICCV2025/3d_vision/driving_view_synthesis_on_free-form_trajectories_with_generative_prior.md)
- [\[AAAI 2026\] Multi-Modal Assistance for Unsupervised Domain Adaptation on Point Cloud 3D Object Detection](multi-modal_assistance_for_unsupervised_domain_adaptation_on_point_cloud_3d_obje.md)

</div>

<!-- RELATED:END -->
