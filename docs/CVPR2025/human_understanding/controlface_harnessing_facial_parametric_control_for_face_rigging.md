---
title: >-
  [Paper Note] ControlFace: Harnessing Facial Parametric Control for Face Rigging
description: >-
  [Human Understanding] Proposes ControlFace, which utilizes a dual-branch U-Net (FaceNet + denoising U-Net) combined with 3DMM rendering conditions to achieve flexible editing of facial pose, expression, and illumination without fine-tuning, while precisely preserving identity and semantic details.
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: 1badf937324822b7
---

# ControlFace: Harnessing Facial Parametric Control for Face Rigging

| Property | Value |
|------|------|
| Conference | CVPR 2025 |
| arXiv | [2412.01160](https://arxiv.org/abs/2412.01160) |
| Code | [Project Page](https://cvlab-kaist.github.io/ControlFace/) |
| Area | Human Understanding / Face Editing |
| Keywords | face rigging, 3DMM, dual-branch U-Net, diffusion model, reference control guidance |

## TL;DR

Proposes ControlFace, which utilizes a dual-branch U-Net (FaceNet + denoising U-Net) combined with 3DMM rendering conditions to achieve flexible editing of facial pose, expression, and illumination without fine-tuning, while precisely preserving identity and semantic details.

## Background & Motivation

### Background

Face rigging is a fundamental task in computer vision, aiming to modify facial images according to user-specified control signals such as pose, expression, and illumination, while maintaining identity consistency. Recently, diffusion models have demonstrated powerful capabilities in face generation, which, when combined with 3D Morphable Models (3DMMs, e.g., FLAME), can achieve parametric, explicit control.

### Limitations of Prior Work

1. **Reconstruction training dilemma caused by reliance on image datasets**: Existing methods (e.g., DiffusionRig, CapHuman) are trained in a reconstruction paradigm on single-image datasets such as FFHQ. To prevent the model from directly copying the reference image and ignoring control signals, they have to compress the reference image into a single vector (e.g., face recognition features), which discards fine-grained semantic information such as hairstyle and background.
2. **Requirement for subject-specific fine-tuning**: For each new identity, additional fine-tuning data and training are required, limiting practical utility.
3. **Trade-off between control accuracy and identity preservation**: Encoding too much information from the reference image causes the model to ignore control signals, while encoding too little results in the loss of identity details.

### Goal

How to simultaneously achieve **fine-grained identity preservation** (including hairstyle, background, etc.) and **precise parametric control** (pose, expression, illumination) **without any fine-tuning**?

### Key Insight & Core Idea

Leverage facial video datasets to construct paired quadruplets, avoiding the reconstruction training dilemma; use a dual-branch U-Net to fully encode the rich representation of the reference image; propose a Control Mixer Module (CMM) and Reference Control Guidance (RCG) to enhance control accuracy.

## Method

### Overall Architecture

ControlFace adopts a dual-branch U-Net architecture: FaceNet encodes the identity and semantic details of the reference image, while the denoising U-Net is responsible for generation. The two are integrated via Augmented Self-Attention layers. Control signals are injected through the Face Controller and Control Mixer Module, and Reference Control Guidance is further applied during inference to enhance control accuracy.

### Key Designs

#### Key Design 1: Dual-branch U-Net + Video Data Training

- **Function**: Captures the complete identity and semantic details of the reference image without ignoring control signals.
- **Mechanism**: FaceNet and the denoising U-Net share the same architecture (both initialized from Stable Diffusion v1.5). FaceNet encodes the reference image, and its key and value are concatenated with those of the denoising U-Net to perform augmented self-attention.
- **Design Motivation**: It is trained on the CelebV-HQ facial video dataset, where two frames from the same video are randomly selected as the reference image $X_R$ and target image $X_T$, forming a paired quadruplet $\{X_R, X_T, D_R, D_T\}$. This avoids the issue in reconstruction training where $X_R = X_T$ leads to control signals being ignored.
- **Loss**: Standard diffusion model denoising loss $\mathcal{L} = \mathbb{E}[\|\epsilon_\theta(z_{T,t}; t, z_R, D_T, D_R) - \epsilon\|^2_2]$

#### Key Design 2: Control Mixer Module (CMM)

- **Function**: Encodes correlation features between the target control $D_T$ and reference control $D_R$, enhancing control alignment.
- **Mechanism**: Two weight-sharing control encoders (containing convolutional layers and cross-attention layers) encode $D_R$ and $D_T$ respectively, outputting correlation embeddings $E_R$ and $E_T$. These embeddings are added to the queries and keys of the augmented self-attention to guide the model's attention.
- **Design Motivation**: Encoding only the target control $D_T$ prevents the model from understanding the "delta" between the reference image and the target. CMM models their correlation to provide guidance on the direction of change.
- **Modified Self-Attention**: $\text{Aug-Attn}^* = \text{Softmax}\left(\frac{(Q+E_T)[K+E_T, K^{\text{face}}+E_R]^T}{\sqrt{d}}\right)[V, V^{\text{face}}]$

#### Key Design 3: Reference Control Guidance (RCG)

- **Function**: Enhances adherence to target control signals during inference.
- **Mechanism**: Unlike standard CFG which uses an empty condition as the null condition, RCG replaces the null condition with the reference control $D_R$: $\hat{\epsilon}_\theta(\cdot, D_T) = \epsilon_\theta(\cdot, D_R) + w(\epsilon_\theta(\cdot, D_T) - \epsilon_\theta(\cdot, D_R))$
- **Design Motivation**: The null condition in standard CFG (e.g., empty input) provides poor grounding, resulting in noisy difference signals. In contrast, since $D_R$ and $D_T$ share the same identity, their difference is concentrated in the facial region, precisely indicating the areas that need modification. Visualizations demonstrate that the difference of RCG serves as a clean, face-aligned estimation across all timesteps.

## Key Experimental Results

### Main Results

**Control Accuracy (DECA Re-inference Error ↓)**:

| Method | Light | Shape | Exp. | Pose | Avg. |
|------|-------|-------|------|------|------|
| GIF | 17.04 | 2.29 | 8.16 | 8.17 | 8.91 |
| CapHuman | 15.16 | 2.65 | 6.68 | 19.03 | 10.40 |
| DiffusionRig | 6.31 | 2.11 | 5.58 | 6.26 | 5.06 |
| **ControlFace** | **3.75** | 2.56 | **5.43** | 7.67 | **4.85** |

**Image Quality and Identity Preservation**:

| Method | ID ↑ | FID ↓ | LPIPS ↓ |
|------|------|-------|---------|
| Arc2Face | 0.7825 | 17.82 | 0.5253 |
| DiffusionRig | 0.2042 | 23.05 | 0.3758 |
| **ControlFace** | 0.7586 | **15.50** | **0.1429** |

### Ablation Study

| Configuration | Re-Infer. ↓ | ID ↑ | FID ↓ |
|------|-------------|------|-------|
| FaceNet only | 7.13 | 0.8234 | 32.45 |
| +CMM | 5.78 | 0.7520 | 15.35 |
| +CMM+RCG | **4.85** | **0.7586** | **15.50** |

Face Controller (~1M parameters) achieves better performance than ControlNet (~360M) and ControlNeXt (~3M).

### Key Findings

- In user studies, ControlFace far outperforms baselines in semantic consistency (0.875) and perceptual quality (0.861).
- It achieves successful control even on out-of-domain images, such as anime styles.
- It outperforms DiffusionRig, which requires fine-tuning, without any fine-tuning.

## Highlights & Insights

1. **Ingenious training strategy using video datasets**: Utilizing paired frames to circumvent the reconstruction training dilemma is the core insight.
2. **Simple yet effective RCG concept**: Replacing the null condition with reference control is plug-and-play, requiring no additional training.
3. **LPIPS of 0.1429 far outperforms all baselines**: Indicating exceptional preservation of semantic details (hairstyle/background).
4. **Lightweight Face Controller**: With only ~1M parameters, it outperforms ControlNet's ~360M, showing that simplicity is beauty.

## Limitations & Future Work

1. Reliance on DECA to extract 3DMM renderings places an upper bound on control accuracy based on DECA's precision.
2. Trained solely on the CelebV-HQ video dataset, which limits identity diversity (~15K individuals).
3. Currently only supports 256×256 resolution, requiring expansion for high-resolution scenarios.

## Related Work & Insights

- **DiffusionRig** (CVPR 2023): Requires subject-specific fine-tuning and has limited encoding capability; ControlFace outperforms it without any fine-tuning.
- **IP-Adapter / ReferenceNet Series**: Share similar dual-branch structural ideas, but ControlFace is tailored with fine designs specifically for face tasks.
- **Video dataset training paradigm**: Can be extended to other generative tasks that require paired training but lack corresponding annotations.

## Rating

⭐⭐⭐⭐ — The method is ingeniously designed, with clear motivations for each component. The video training strategy and RCG are key highlights, though resolution limitations and dependency on 3DMMs remain unresolved issues.

## Related Papers

- [\[CVPR 2025\] CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)
- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](../../ICCV2025/human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CryptoFace: End-to-End Encrypted Face Recognition](cryptoface_end-to-end_encrypted_face_recognition.md)
- [\[CVPR 2025\] CRISP: Object Pose and Shape Estimation with Test-Time Adaptation](crisp_object_pose_and_shape_estimation_with_test-time_adaptation.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](../../ICCV2025/human_understanding/openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](../../ICCV2025/human_understanding/sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)

</div>

<!-- RELATED:END -->
