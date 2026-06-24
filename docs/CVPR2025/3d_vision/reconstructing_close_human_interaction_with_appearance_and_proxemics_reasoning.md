---
title: >-
  [Paper Note] Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning
description: >-
  [CVPR 2025][3D Vision][Human Interaction Reconstruction] This paper proposes a dual-branch optimization framework that reconstructs accurate 3D poses, natural interaction relationships, and plausible body contacts of close human interactions from monocular in-the-wild videos by combining human appearance constraints (3D Gaussian Splatting), a proxemics diffusion prior, and physical constraints, achieving state-of-the-art (SOTA) performance on Hi4D and 3DPW.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Human Interaction Reconstruction"
  - "3D Gaussian Splatting"
  - "Proxemics Prior"
  - "Dual-Branch Optimization"
  - "Close Interaction"
date: 2026-05-08
content_hash: b5914f3aa0ddc1fe
---

# Reconstructing Close Human Interaction with Appearance and Proxemics Reasoning

**Conference**: CVPR 2025  
**arXiv**: [2507.02565](https://arxiv.org/abs/2507.02565)  
**Code**: [https://www.buzhenhuang.com/works/CloseApp.html](https://www.buzhenhuang.com/works/CloseApp.html)  
**Area**: 3D Vision / Human Reconstruction  
**Keywords**: Human Interaction Reconstruction, 3D Gaussian Splatting, Proxemics Prior, Dual-Branch Optimization, Close Interaction

## TL;DR

This paper proposes a dual-branch optimization framework that reconstructs accurate 3D poses, natural interaction relationships, and plausible body contacts of close human interactions from monocular in-the-wild videos by combining human appearance constraints (3D Gaussian Splatting), a proxemics diffusion prior, and physical constraints, achieving state-of-the-art (SOTA) performance on Hi4D and 3DPW.

## Background & Motivation

**Background**: 3D human pose estimation has made tremendous progress in recent years, but existing methods still struggle when dealing with close human interactions. Single-person methods focus only on pose accuracy and image-to-mesh alignment, while multi-person methods focus on interpenetration and the rationality of spatial distribution. Both neglect the critical physical contacts and proxemic relationships in close interactions. The few methods dedicated to close-range interactions (such as BUDDI, CloseInt) rely either on detected 2D human semantics or high-quality indoor interaction data.

**Limitations of Prior Work**: Close human interaction scenes present severe visual ambiguity and inter-person occlusion. Even state-of-the-art large foundation models (such as SAM, ViTPose++) fail to accurately segment or keypoint human semantics of two people in close contact—keypoint detectors give unreliable 2D keypoints, and segmentation models cannot correctly separate the two closely touching individuals. Regression-based methods (such as CloseInt) rely on indoor scene data for training and experience a significant drop in performance when generalizing to in-the-wild scenes.

**Key Challenge**: Depth ambiguity and visual ambiguity make it extremely difficult to infer the 3D configuration of close interactions from 2D observations. Traditional optimization methods easily fall into local optima, while feedforward regression methods lack generalization capability to in-the-wild scenes.

**Goal**: To design a framework capable of working in diverse in-the-wild environments that simultaneously reconstructs accurate human poses, natural interactive spatial relationships, and plausible physical contacts.

**Key Insight**: The authors find that human appearance can provide direct cues to resolve visual ambiguity and occlusion. By modeling human appearance and rendering it back to the image plane, depth ordering relationships and image-to-mesh alignment can be directly inferred using raw RGB images without relying on unreliable 2D semantic detection.

**Core Idea**: Simultaneously reconstruct human motion and appearance, substituting unreliable 2D semantic constraints with appearance rendering constraints, and cooperating with interaction priors learned by diffusion models to achieve robust reconstruction of close interactions.

## Method

### Overall Architecture

Given a monocular in-the-wild video showing a close interaction between two individuals, the framework operates in two stages. First, a diffusion model is trained to learn prior knowledge of human interaction behaviors and poses. Then, the trained diffusion model and two optimizable tensors are integrated into a dual-branch optimization framework: the Motion Branch utilizes the diffusion model to generate and fine-tune interactive motions, while the Appearance Branch decodes the optimizable tensors into Gaussian UV maps via a U-Net to reconstruct the human appearance. The two branches are jointly optimized, constrained by appearance, 2D keypoints, interpenetration penalties, and smoothness constraints.

### Key Designs

1. **Proxemic Prior**:

    - **Function**: Provides pose and interaction prior knowledge, mitigating depth ambiguity and local optima issues.
    - **Mechanism**: Adopt a diffusion model to iteratively denoise from noise to generate clean two-person motion. The model is conditioned on 2D keypoints and image features, using Transformer blocks to process features, where features of the two individuals share information via cross-attention. Two masking strategies are used during training: (1) randomly masking partial frames to learn temporal dependencies; (2) completely masking the input of one person to force the model to generate reactive actions from the other. The loss function includes reprojection loss, SMPL parameter loss, 3D joint position loss, velocity loss, and interaction distance loss $\mathcal{L}_{int} = \||J^a_{3D} - J^b_{3D}| - |\hat{J}^a_{3D} - \hat{J}^b_{3D}|\|^2$.
    - **Design Motivation**: Unlike existing interaction priors (such as BUDDI), this model simultaneously receives 2D observations and utilizes temporal information to infer 3D interactions, making it robust to occlusions through masking strategies. Fine-tuning the network parameters (instead of directly modifying the motion itself) during the optimization phase allows for better control of the output and leverages the pretrained knowledge.

2. **Appearance Branch (3D Gaussian Splatting)**:

    - **Function**: By reconstructing human appearance and rendering it to the image, it provides dense RGB constraints to resolve depth ordering and alignment issues.
    - **Mechanism**: Design two sets of optimizable tensors as latent codes, which are decoded by a U-Net into Gaussian UV maps (14 channels: offset $\mu$, color $c$, opacity $\sigma$, rotation $q$, scale $s$, identity $d$). The Gaussians are mapped onto the 3D human surface using UV coordinates and then rendered onto 2D images via splatting. A key innovation is rendering the Gaussians of both people simultaneously, using the identity channel to differentiate the two individuals. The appearance constraint is compared with the original image via a combined loss of L1 + SSIM + LPIPS.
    - **Design Motivation**: Unlike traditional methods that rely on 2D keypoints and segmentation masks, directly using raw RGB images as constraints is more reliable—since 2D semantic detection itself is inaccurate in close-contact scenarios. Rendering both individuals simultaneously naturally reflects occlusion relationships without requiring separate semantic parsing for each person.

3. **Joint Motion-Appearance Optimization**:

    - **Function**: Simultaneously optimizes motion and appearance parameters, utilizing multiple constraints to find the global optimum.
    - **Mechanism**: The total objective function is $\mathcal{L} = \mathcal{L}_{app} + \mathcal{L}_{reproj} + \mathcal{L}_{pen} + \mathcal{L}_{smooth} + \mathcal{L}_{reg}$. The physical penetration constraint uses a differentiable 3D distance field to detect colliding triangles and penalize penetration depth. The smoothness constraint encourages minimizing joint displacement between adjacent frames. The regularization constraint prevents the optimized parameters from deviating too far from the initial predictions. Using the Adam optimizer, with a motion branch learning rate of 0.00002 and an appearance branch learning rate of 0.003, processing a 128-frame video takes about 3-5 minutes.
    - **Design Motivation**: Optimizing motion alone may yield correct poses but incorrect depth ordering; optimizing appearance alone lacks physical plausibility. Jointly optimizing both branches allows appearance constraints to guide depth ordering while physical constraints guarantee realistic contact.

### Loss & Training

The diffusion model is pretrained on large-scale interaction datasets such as Inter-X and InterHuman. During the optimization phase, AutoTrackAnything is used to obtain bounding boxes and whole-body masks (note that only the joint mask of the two people is required, without separate segmentation), and ViTPose is used to obtain 2D keypoints. The overall pipeline is: Pretraining priors $\rightarrow$ Initial prediction $\rightarrow$ Joint dual-branch optimization.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | CloseInt | BUDDI | GroupRec |
|--------|------|------|----------|-------|---------|
| Hi4D | MPJPE↓ | **59.1** | 63.1 | 96.8 | 82.4 |
| Hi4D | PA-MPJPE↓ | **44.3** | 47.5 | 70.6 | 51.6 |
| Hi4D | MPVPE↓ | **72.0** | 76.4 | 116.0 | 88.6 |
| Hi4D | Inter↓ | **80.2** | 81.4 | 102.6 | 98.8 |
| 3DPW | MPJPE↓ | **64.5** | 70.6 | 83.6 | 73.3 |
| 3DPW | PA-MPJPE↓ | **45.6** | 51.4 | 53.6 | 48.7 |

### Ablation Study

| Configuration | MPJPE↓ | PA-MPJPE↓ | MPVPE↓ | Inter↓ | A-PD↓ |
|------|--------|-----------|--------|--------|-------|
| Initial Prediction | 65.05 | 48.54 | 78.35 | 86.20 | 1.16 |
| w/o Appearance | 60.68 | 45.86 | 73.52 | 81.01 | 0.83 |
| w/o Proxemics | 61.52 | 47.13 | 74.84 | 87.13 | 0.85 |
| w/o Physics | **57.01** | **42.67** | **69.57** | **78.50** | 1.30 |
| Full Model | 59.06 | 44.29 | 71.99 | 80.18 | **0.81** |

### Key Findings

- **Appearance constraints are crucial for depth ordering**: Removing the appearance branch still yields acceptable poses, but depth ordering becomes prone to errors. Even coarse textures are sufficient to constrain the depth relations of motion.
- **There is a trade-off between physical constraints and accuracy**: Without the physics constraint, MPJPE is actually the lowest (57.01), but the penetration depth A-PD increases from 0.81 to 1.30. This indicates that physical constraints sacrifice some joint accuracy for physical plausibility.
- **WildCHI dataset can improve regression methods**: Training CloseInt with pseudo-labels generated by the proposed method improves its performance on both Hi4D and 3DPW.
- **Fine-tuning network parameters is more robust than directly optimizing SMPL parameters**: Leveraging the prior knowledge in pretrained weights effectively mitigates depth ambiguity.

## Highlights & Insights

- **Using appearance as a constraint signal is highly inspiring**: When 2D semantics are unreliable in close interactions, directly using raw RGB images as constraints acts as a powerful alternative—rendering both individuals' Gaussians simultaneously eliminates the dependence on individual segmentation.
- **Diffusion model as a fine-tunable optimization prior**: Instead of using the diffusion model for feedforward inference, embedding it in the optimization loop to fine-tune parameters is a valuable "prior as an optimizable component" paradigm.
- **The mutually beneficial relationship of simultaneously reconstructing motion and appearance**: Motion provides geometric support for appearance, and appearance in turn constrains the depth and alignment of motion, forming a virtuous cycle. This approach can be transferred to other reconstruction tasks requiring joint multi-signal optimization.

## Limitations & Future Work

- High-quality complete textures cannot be reconstructed under illumination changes or heavy occlusion, which could be improved by introducing illumination embeddings or large vision foundation models.
- Currently only supports two-person interactions, and lacks multi-person interaction datasets to train the prior.
- The input video needs to contain some frames with minimal or no contact to constrain the appearance.
- Processing 128 frames takes 3-5 minutes, leaving a gap before real-time application.

## Related Work & Insights

- **vs BUDDI**: BUDDI is also an optimization framework but relies solely on 2D keypoint fitting, lacking temporal information and appearance constraints. This work introduces appearance rendering constraints and temporal diffusion priors, making it more robust in visually ambiguous scenes.
- **vs CloseInt**: CloseInt is a regression method that relies on indoor data for training, leading to poor generalization. The proposed optimization framework can work in diverse environments, and the generated pseudo-labels can in turn improve CloseInt's training.
- **vs BEV/GroupRec**: These multi-person methods consider the spatial distribution of crowds but do not handle close contact, failing to reconstruct accurate interactive relationships.

## Rating

- Novelty: ⭐⭐⭐⭐ Creatively integrates Gaussian splatting and diffusion models into an optimization framework for human interaction reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple datasets, with insightful ablation analysis, and introduces the WildCHI dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-described methodology, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a practical solution for close human interaction reconstruction, and the WildCHI dataset is valuable to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] InteractVLM: 3D Interaction Reasoning from 2D Foundational Models](interactvlm_3d_interaction_reasoning_from_2d_foundational_models.md)
- [\[CVPR 2025\] Reconstructing Animals and the Wild](reconstructing_animals_and_the_wild.md)
- [\[CVPR 2025\] EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](envgs_modeling_view-dependent_appearance_with_environment_gaussian.md)
- [\[CVPR 2025\] Textured Gaussians for Enhanced 3D Scene Appearance Modeling](textured_gaussians_for_enhanced_3d_scene_appearance_modeling.md)

</div>

<!-- RELATED:END -->
