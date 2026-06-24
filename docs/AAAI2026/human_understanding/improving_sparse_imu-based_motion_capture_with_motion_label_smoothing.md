---
title: >-
  [Paper Note] Improving Sparse IMU-based Motion Capture with Motion Label Smoothing
description: >-
  [AAAI 2026][Human Understanding][Sparse IMU] This paper proposes Motion Label Smoothing, adapting classical label smoothing from classification tasks to sparse IMU-based motion capture. By incorporating skeleton-structure-aware Perlin noise as smoothed labels, the method improves accuracy across three state-of-the-art methods on four datasets in a plug-and-play manner without modifying model architectures. GlobalPose achieves a 20.41% reduction in SIP error on TotalCapture.
tags:
  - "AAAI 2026"
  - "Human Understanding"
  - "Sparse IMU"
  - "Human Motion Capture"
  - "Label Smoothing"
  - "Perlin Noise"
  - "Regularization"
date: 2026-05-08
content_hash: d14aa3b45ffd405e
---

# Improving Sparse IMU-based Motion Capture with Motion Label Smoothing

**Conference**: AAAI 2026
**arXiv**: [2511.22288](https://arxiv.org/abs/2511.22288)  
**Authors**: Zhaorui Meng, Lu Yin, Yangqing Hou, Anjun Chen, Shihui Guo, Yipeng Qin (Xiamen University, Cardiff University)
**Code**: Not released  
**Area**: Human Understanding
**Keywords**: Sparse IMU, Human Motion Capture, Label Smoothing, Perlin Noise, Regularization

## TL;DR

This paper proposes Motion Label Smoothing, adapting classical label smoothing from classification tasks to sparse IMU-based motion capture. By incorporating skeleton-structure-aware Perlin noise as smoothed labels, the method improves accuracy across three state-of-the-art methods on four datasets in a plug-and-play manner without modifying model architectures. GlobalPose achieves a 20.41% reduction in SIP error on TotalCapture.

## Background & Motivation

Sparse IMU motion capture systems use only six IMU sensors placed at the wrists, ankles, head, and pelvis to achieve real-time human motion reconstruction. Compared to optical motion capture systems, this approach offers portability, low cost, and occlusion robustness, with broad application prospects in film production, gaming, and medical rehabilitation.

Recent research in this area has focused primarily on **model architecture design**: TransPose introduced the Transformer architecture to improve accuracy, PIP combined physics-based optimization to enhance motion plausibility, PNP uses autoregressive MLP to calibrate acceleration signals, and GlobalPose enables translation estimation in full 3D space. However, these works have largely overlooked **regularization methods**—an equally critical component in deep learning—leaving a notable gap in the AI toolkit for sparse IMU motion capture.

Label smoothing is a widely used regularization technique in classification tasks that prevents model overconfidence by mixing one-hot labels with a uniform distribution. However, directly transferring it to motion capture is non-trivial: the "uniform vector" in classification degenerates to a static pose (e.g., T-pose) in motion space, which reduces label entropy rather than increasing it, fundamentally contradicting the purpose of label smoothing.

## Core Problem

How to design a label smoothing regularization method applicable to continuous motion representations—without modifying model architectures—that increases label entropy while preserving three intrinsic properties of human motion data: (1) temporal smoothness, (2) joint correlation, and (3) low-frequency dominance?

## Method

### Overall Architecture

Motion Label Smoothing replaces the motion label $R$ (24 SMPL joint rotations) in the classical label smoothing formula $y' = (1-\epsilon)y + \epsilon u$, with the core challenge being the design of noise $u$ that satisfies motion properties:

$$R' = (1-\epsilon)R + \epsilon u$$

where $u$ is realized using the proposed skeleton-based Perlin noise.

### Key Design 1: Analysis of Three Motion Label Properties

**Property 1 — Temporal Smoothness**: Human motion is constrained by muscle forces and joint range of motion, bounding the angular velocity of joints:

$$\omega(t) \approx \frac{R(t+\Delta t) - R(t)}{\Delta t}, \quad \|\omega(t)\| \leq M$$

**Property 2 — Joint Correlation**: The human body is a rigid skeletal chain coupling system in which the rotation range of child joints is constrained by parent joints:

$$R_{\text{child}}(t) \in \mathcal{A}(R_{\text{parent}}(t))$$

**Property 3 — Low-Frequency Dominance**: Motion signals are dominated by low-frequency components (experiments show $\alpha=0.7$ at $f_c=5\text{Hz}$), with power spectral density:

$$P_c(f) = \left|\int_0^T R_c(t) e^{-i2\pi ft} dt\right|^2$$

### Key Design 2: Skeleton-based Perlin Noise

Standard i.i.d. noise (Gaussian/Uniform) has a flat power spectrum and independent sampling characteristics that violate the three properties above. This paper proposes skeleton-structure-based Perlin noise:

$$u = \texttt{sk-Perlin}(JC, \mathcal{H}, size)$$

where $JC$ denotes the 6 joint chains of the SMPL skeleton and $\mathcal{H} = \{S_b, S_t, S_s, p, oct, l\}$ are the Perlin noise parameters.

Key characteristics:
- **Amplitude decoupling**: The amplitude of Perlin noise is controlled by $S_b$ and the smoothness by the interpolation function, independently—unlike i.i.d. noise where amplitude and smoothness are coupled
- **Satisfying Properties 1 & 2**: Interpolation along the time axis ensures temporal continuity; a base noise is first generated for each joint chain, and single-layer octave offsets are then added per joint within the chain, ensuring intra-chain joint correlation while maintaining distinguishability
- **Satisfying Property 3**: Amplitudes decay exponentially as $1/2^i$ across octave summations, with persistence $p=0.5$, octave count $oct=5$, guaranteeing low-frequency dominance

## Key Experimental Results

### Main Results: Error Reduction Across Three Methods × Four Datasets

| Method | Dataset | SIP Error↓ | Joint Error↓ | Mesh Error↓ |
|--------|---------|-----------|-------------|------------|
| TransPose+Ours | TotalCapture | 12.49 (↓12.54%) | 5.00 (↓5.84%) | 5.55 (↓5.77%) |
| PIP+Ours | TotalCapture | 10.54 (↓5.56%) | 4.38 (↓3.74%) | 5.07 (↓3.61%) |
| **GlobalPose+Ours** | **TotalCapture** | **7.84 (↓20.41%)** | **3.26 (↓17.68%)** | **3.75 (↓13.79%)** |
| TransPose+Ours | DIP-IMU | 13.57 (↓3.35%) | 4.64 (↓4.53%) | 5.50 (↓5.17%) |
| PIP+Ours | DIP-IMU | 11.62 (↓3.81%) | 4.18 (↓3.46%) | 4.88 (↓3.56%) |
| GlobalPose+Ours | DIP-IMU | 13.50 (↓1.96%) | 4.27 (↓2.06%) | 4.98 (↓1.97%) |

### Ablation Study: Incrementally Adding Motion Properties

| Configuration | SIP(°)↓ | Ang(°)↓ | Joint(cm)↓ | Mesh(cm)↓ |
|---------------|---------|---------|-----------|----------|
| Baseline (GlobalPose) | 9.85 | 9.55 | 3.96 | 4.35 |
| +Label Smoothing (Gaussian) | 8.82 | 8.65 | 3.82 | 4.43 |
| +Temporal Smoothness (T) | 8.59 | 8.30 | 3.77 | 4.37 |
| +Joint Correlation (T+J) | 8.22 | 8.02 | 3.52 | 4.12 |
| +Low-Freq Dominance (T+J+L, Ours) | **7.84** | **7.87** | **3.26** | **3.75** |

### Alternative Strategy Comparison

| Strategy | SIP↓ | Joint↓ | Mesh↓ |
|----------|------|--------|-------|
| T-pose Vector | 8.97 | 3.96 | 4.73 |
| AMASS Mean Vector | 8.75 | 3.87 | 4.60 |
| Uniform Noise | 8.72 | 3.82 | 4.44 |
| Gaussian Noise | 8.82 | 3.82 | 4.43 |
| Temporal Smoothing | 8.23 | 3.57 | 4.15 |
| Knowledge Distillation | 8.46 | 3.59 | 4.17 |
| **Ours** | **7.84** | **3.26** | **3.75** |

## Highlights & Insights

- **First regularization method for sparse IMU-based motion capture**: Fills the gap in the regularization module of the AI toolkit for this domain with a plug-and-play design requiring no modification to any model architecture
- **Deep insight into label smoothing**: Reveals that the core mechanism of label smoothing is increasing label entropy rather than mere uniformization, correcting a long-standing misconception—an insight equally applicable to other continuous regression tasks
- **Systematic analysis of motion properties**: For the first time rigorously defines and validates three key properties of motion labels (temporal smoothness, joint correlation, low-frequency dominance), providing theoretical grounding for the Perlin noise design
- **Strong generalization across methods and datasets**: Consistently improves performance across 3 state-of-the-art methods × 4 datasets, with GlobalPose achieving a 20.41% SIP error reduction on TotalCapture

## Limitations & Future Work

- **Inherits limitations of underlying methods**: The method relies on the SMPL template body model, which ignores the effect of body shape variation on IMU data, limiting generalization for children or users with extreme body proportions
- **Limited motion types in datasets**: Although large-scale, the training data (AMASS) covers a restricted range of motion types, and reconstruction of complex motions such as falls or street dance remains challenging
- **Lack of theoretical convergence analysis**: While an intuitive explanation is provided from an entropy-increase perspective, no rigorous theoretical analysis of the convergence speed or generalization bounds of motion label smoothing is presented

## Related Work & Insights

- **TransPose/PIP/GlobalPose, etc.**: Focus on model architecture innovations (RNN→Transformer→physics-based optimization); this paper provides complementary gains from a regularization perspective
- **Classical Label Smoothing (Szegedy et al. 2016)**: Designed for classification tasks using a uniform vector; this paper demonstrates that direct transfer to regression tasks is ineffective
- **Knowledge Distillation (Yuan et al. 2020)**: Knowledge distillation can be viewed as implicit label smoothing, but underperforms the proposed method on motion capture (SIP 8.46 vs. 7.84)
- **Label Relaxation (Lienen & Hüllermeier 2021)**: Represents labels as probability sets but does not account for the structured properties of motion data

**Inspiration and Connections**:
- **Novel application of Perlin noise**: Originally used for texture generation in computer graphics, this paper cleverly applies it to motion label regularization, demonstrating the value of cross-domain technology transfer
- **Regularization transfer from classification to regression**: The core insight that label smoothing increases entropy suggests that similarly structured noise regularization can be designed for other continuous signal regression tasks (e.g., speech synthesis, trajectory prediction)
- **Property-aware data augmentation**: The identification of three motion properties provides a methodological reference for designing task-specific data augmentation strategies in other domains

## Rating

- Novelty: ⭐⭐⭐⭐ — First adaptation of label smoothing to motion capture tasks; the application of Perlin noise is novel and theoretically grounded
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 3 methods × 4 datasets with extensive ablation and alternative strategy comparisons
- Writing Quality: ⭐⭐⭐⭐ — Clear logical chain: problem definition → property analysis → method design → experimental validation, with progressive reasoning
- Value: ⭐⭐⭐⭐ — A plug-and-play regularization tool with practical value for the sparse IMU domain, though theoretical depth could be further strengthened

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] IMU-HOI: A Symbiotic Framework for Coherent Human-Object Interaction and Motion Capture via Contact-Conscious Inertial Fusion](../../CVPR2026/human_understanding/imu-hoi_a_symbiotic_framework_for_coherent_human-object_interaction_and_motion_c.md)
- [\[CVPR 2026\] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors](../../CVPR2026/human_understanding/ground_reaction_inertial_poser_physics-based_human_motion_capture_from_sparse_im.md)
- [\[AAAI 2026\] KineST: A Kinematics-guided Spatiotemporal State Space Model for Human Motion Tracking from Sparse Signals](kinest_a_kinematics-guided_spatiotemporal_state_space_model_for_human_motion_tra.md)
- [\[ICCV 2025\] MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances](../../ICCV2025/human_understanding/magshield_towards_better_robustness_in_sparse_inertial_motion_capture_under_magn.md)
- [\[CVPR 2026\] MoBind: Motion Binding for Fine-Grained IMU-Video Pose Alignment](../../CVPR2026/human_understanding/mobind_motion_binding_for_fine-grained_imu-video_pose_alignment.md)

</div>

<!-- RELATED:END -->
