---
title: >-
  [Paper Note] PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing
description: >-
  [CVPR 2026][Video Generation][Portrait video editing] PerformRecast presents a GAN-based portrait video editing method built upon a corrected 3DMM keypoint transformation formulation. By applying expression deformation before head rotation — consistent with the FLAME model — the method achieves precise disentanglement of expression and head pose. A Boundary Alignment Module (BAM) is further introduced to address stitching misalignment between facial and non-facial regions. The approach substantially outperforms existing methods under both expression replacement and expression enhancement modes.
tags:
  - CVPR 2026
  - Video Generation
  - Portrait video editing
  - expression disentanglement
  - 3DMM
  - keypoint transformation
  - GAN
date: 2026-05-08
content_hash: 48bad9b79a333db7
---

# PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing

**Conference**: CVPR 2026
**arXiv**: [2603.19731](https://arxiv.org/abs/2603.19731)
**Code**: [https://youku-aigc.github.io/PerformRecast](https://youku-aigc.github.io/PerformRecast)
**Area**: Video Generation
**Keywords**: Portrait video editing, expression disentanglement, 3DMM, keypoint transformation, GAN

## TL;DR

PerformRecast presents a GAN-based portrait video editing method built upon a corrected 3DMM keypoint transformation formulation. By applying expression deformation before head rotation — consistent with the FLAME model — the method achieves precise disentanglement of expression and head pose. A Boundary Alignment Module (BAM) is further introduced to address stitching misalignment between facial and non-facial regions. The approach substantially outperforms existing methods under both expression replacement and expression enhancement modes.

## Background & Motivation

**Background**: The portrait animation field has seen extensive development, encompassing GAN-based warping methods (LivePortrait, Face Vid2Vid, etc.) and diffusion-based methods (SkyReels, Hunyuan-Portrait, Wan-Animate, etc.). These methods typically generate animations from static portrait images guided by a driving video.

**Limitations of Prior Work**: The central challenge lies in **expression and head pose disentanglement**. Portrait video expression editing requires modifying only facial expressions while strictly preserving the subject's facial identity, head pose, camera motion, and background — any unintended change is considered a failure. Existing methods suffer from: (1) diffusion-based methods are inherently ill-suited for disentangling expression from head rotation, exhibit slow inference, and suffer from temporal inconsistency; (2) GAN-based warping methods (e.g., LivePortrait) offer better controllability but rely on implicit keypoints that lack explicit physical meaning and direct supervision, resulting in incomplete disentanglement.

**Key Challenge**: LivePortrait's keypoint transformation formula is $x = s \cdot (x_c R + \delta) + t$, which applies head rotation $R$ to the canonical keypoints before adding expression deformation $\delta$. This ordering is inconsistent with the 3DMM forward process — in FLAME, expression blendshapes are added to the template mesh before joint rotation is applied. This incorrect ordering forces the learned $\delta$ to absorb residual head pose information, preventing true disentanglement.

**Goal**: (1) Correct the keypoint transformation formula to align with the FLAME forward process; (2) directly supervise the motion extractor using explicit 3D keypoints; (3) resolve boundary misalignment between facial and non-facial regions during expression editing.

**Key Insight**: 3DMMs inherently represent identity, expression, and head pose with independent parameters. The authors leverage the physically grounded ordering of the FLAME forward process — expression first, then rotation — to improve upon LivePortrait.

**Core Idea**: Revise the keypoint transformation to apply expression deformation before rotation, matching the FLAME forward process, and directly supervise the motion extractor with explicit keypoints derived from 3D face tracking.

## Method

### Overall Architecture

PerformRecast is built on LivePortrait's warping architecture. Given a source frame and a driving frame, an appearance feature extractor $\mathcal{F}$ (DINOv2-based) extracts a 3D feature volume, and a motion extractor $\mathcal{M}$ (ConvNext-V2-Tiny) predicts canonical keypoints $x_c$, head rotation $R$, expression deformation $\delta$, scale $s$, and translation $t$. A corrected keypoint transformation formula produces source and driving keypoints; a warping module generates optical flow to deform the feature volume; and a SPADE decoder synthesizes the output image. Training incorporates FLAME-based supervision via Pixel3DMM 3D face tracking.

### Key Designs

1. **FLAME-Based Keypoint Transformation (Core Contribution)**

   - **Function**: Achieve complete disentanglement of expression and head pose.
   - **Mechanism**: Replace LivePortrait's formula $x = s \cdot (x_c R + \delta) + t$ with $x = s \cdot ((x_c + \delta) R) + t$, i.e., **expression deformation is applied before head rotation**. This aligns with the FLAME forward process, in which expression blendshapes are added to the template mesh prior to joint rotation. Explicit 3D supervision is provided by selecting 49 keypoints from FLAME face mesh vertices, organized into three groups: canonical keypoints $V_c$ (identity only), expression keypoints $V_{exp}$ (expression plus eye/jaw rotations, excluding head rotation), and full keypoints $V_{kp}$ (all parameters). Wing loss is used to compute the FLAME loss.
   - **Design Motivation**: Under the original ordering, $\delta$ is forced to compensate for residual head rotation, causing incomplete disentanglement. The corrected ordering ensures $\delta$ is added to canonical keypoints prior to rotation, physically aligned with the 3DMM process and eliminating head pose leakage from the expression representation. With strong FLAME supervision, auxiliary constraints used in LivePortrait — such as keypoint equivariance loss and keypoint prior loss — can be removed.

2. **Boundary Alignment Module (BAM)**

   - **Function**: Mitigate stitching misalignment between facial and non-facial regions.
   - **Mechanism**: A teacher-student two-stage training scheme is adopted. In stage one, a teacher model $M_t$ is trained with a global animation loss, producing accurate facial expressions but with boundary artifacts. In stage two, a student model $M_s$ learns two objectives simultaneously: (a) for the facial region, supervision is provided by the teacher's intermediate output $\hat{I}_s^t$ (with only $\delta$ replaced); (b) for the non-facial region, supervision is provided by the original source frame $I_s$. This enables the student to learn precise expression from the teacher while preserving the non-facial region.
   - **Design Motivation**: The 3D keypoint warping field inevitably affects non-facial regions, causing misalignment at hairlines, ears, and the neck. Applying separate supervision for facial and non-facial regions provides an intuitive and effective solution. The design also eliminates the need for LivePortrait's stitching and retargeting modules, simplifying overall training.

3. **Dual Inference Modes (Replacement + Enhancement)**

   - **Function**: Provide flexible editing modes for practical production scenarios.
   - **Mechanism**: In **Replacement** mode, the driving frame's $\delta_d$ directly replaces the source frame's $\delta_s$, while all other source parameters are preserved. In **Enhancement** mode, the expression delta from the driving sequence $\delta_{d,i} - \delta_{d,0}$ is added to the source expression, achieving expression amplification rather than full replacement.
   - **Design Motivation**: These modes address distinct needs in film production — Replacement suits cases where overall performance is satisfactory but expressions require substitution, while Enhancement suits cases where only the magnitude of existing expressions needs to be amplified.

### Loss & Training

The overall training loss is $\mathcal{L}_{animate} = \mathcal{L}_{FLAME} + \mathcal{L}_{P,cascade} + \mathcal{L}_{1,cascade} + \mathcal{L}_{G,cascade} + \mathcal{L}_{faceid}$, where the FLAME loss applies Wing loss supervision over three keypoint groups; cascade losses are cascaded perceptual, L1, and GAN losses; and the face identity loss maintains identity consistency. The BAM stage additionally introduces separate supervision losses for facial and non-facial regions. Training data includes public datasets (VFHQ, MEAD, Nersemble) as well as high-quality internet animation and film video clips, totaling approximately 600K video clips.

## Key Experimental Results

### Main Results

Evaluation is conducted on an expression editing benchmark constructed using MetaHuman digital humans (20 subjects, 18 expressions), under Replacement mode:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | AED↓ | APD↓ | FVD↓ |
|---|---|---|---|---|---|---|
| LivePortrait | 27.73 | 0.899 | 0.059 | 0.610 | 0.016 | 165.1 |
| SkyReels-A1 | 24.91 | 0.859 | 0.162 | 0.716 | 0.016 | 1249.7 |
| Hunyuan-Portrait | 22.43 | 0.792 | 0.169 | 0.661 | 0.035 | 1925.1 |
| Wan-Animate | 22.82 | 0.802 | 0.132 | 0.700 | 0.024 | 849.2 |
| **PerformRecast** | **29.27** | **0.914** | **0.047** | **0.499** | **0.012** | **103.0** |

Under Enhancement mode, PerformRecast similarly leads across all metrics (PSNR 30.27, FVD 90.2).

### Ablation Study

| Configuration | PSNR↑ | AED↓ | FVD↓ | Notes |
|---|---|---|---|---|
| Ours (Full) | 29.27 | 0.499 | 103.0 | Complete model |
| Ours (KT of LP) | 27.06 | 0.573 | 288.8 | Original LivePortrait keypoint transformation |
| Ours (w/o FLAME loss) | 24.99 | 0.663 | 188.1 | FLAME loss removed |
| Ours (w/o T-S) | 27.73 | 0.575 | 136.4 | Teacher-student (BAM) removed |

### Key Findings

- **The corrected keypoint transformation is the most critical design**: reverting to the original LivePortrait formula drops PSNR from 29.27 to 27.06 and raises FVD from 103.0 to 288.8, confirming that transformation ordering is essential for disentanglement.
- Explicit FLAME supervision reduces AED from 0.663 to 0.499, demonstrating that explicit 3D keypoint constraints are more effective than implicit learning.
- BAM primarily improves boundary region quality (FVD from 136.4 to 103.0), with notable gains in non-facial region fidelity.
- On standard portrait animation benchmarks, PerformRecast also outperforms LivePortrait across both self-driven and cross-identity settings.

## Highlights & Insights

- **The keypoint transformation reordering appears minor but has substantial impact** — simply changing $x_c R + \delta$ to $(x_c + \delta) R$ yields a PSNR gain exceeding 2.2 and a near 3× reduction in FVD. This demonstrates that aligning model architecture with the underlying physical model constitutes a powerful inductive bias in warping-based animation.
- **The region-separated teacher-student supervision** elegantly resolves the conflict between global optimization and local fidelity: the teacher focuses on accurate facial expression, while the student learns facial behavior from the teacher and non-facial behavior from the ground truth. This region-separated supervision paradigm is transferable to any generation task requiring local editing with global consistency.
- The substantial advantage over diffusion-based methods (FVD reduced by 10×+) suggests that for fine-grained controllable editing tasks, GAN-based warping methods remain the superior choice.

## Limitations & Future Work

- Input resolution is fixed at 512×512, which may be insufficient for cinematic high-resolution applications.
- Facial and non-facial region separation still relies on 2D face segmentation, which may be insufficiently robust under heavy occlusion or extreme head angles.
- Enhancement mode achieves expression amplification via simple deformation accumulation, lacking fine-grained control over expression intensity.
- The accuracy of 3D face tracking (Pixel3DMM) imposes an upper-bound constraint — tracking failures introduce erroneous FLAME supervision.
- Performance under large-angle profile views or extreme lighting conditions has not been evaluated.

## Related Work & Insights

- **vs. LivePortrait**: PerformRecast builds directly on LivePortrait and improves upon it. The key differences are the corrected keypoint transformation ordering and explicit FLAME supervision. LivePortrait requires additional stitching and retargeting modules, which PerformRecast eliminates by virtue of superior disentanglement.
- **vs. Diffusion-based methods (SkyReels-A1, Hunyuan-Portrait, Wan-Animate)**: Diffusion-based methods perform poorly on expression editing (PSNR 4–7 points lower, FVD 8–19× higher), as their motion representations cannot precisely disentangle expression from pose.
- **vs. Act-Two (Runway commercial product)**: Act-Two fails to accurately preserve the source video's head pose and loses fine-grained expression detail (PSNR 20.83 vs. 29.27).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The keypoint transformation reordering appears straightforward but carries deep insight; the region-separated BAM supervision is also a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Includes a self-constructed MetaHuman benchmark, comparisons against diverse baselines, comprehensive ablation studies, and multi-task evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation; some formula notation could be further streamlined.
- **Value**: ⭐⭐⭐⭐ — Direct practical value for film and post-production; disentangled expression editing addresses a genuine industry need.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation](pam_a_pose-appearance-motion_engine_for_sim-to-real_hoi_video_generation.md)
- [\[CVPR 2026\] PoseGen: In-Context LoRA Finetuning for Pose-Controllable Long Human Video Generation](posegen_in-context_lora_finetuning_for_pose-controllable_long_human_video_genera.md)
- [\[CVPR 2026\] NOVA: Sparse Control, Dense Synthesis for Pair-Free Video Editing](nova_sparse_control_dense_synthesis_for_pair-free_video_editing.md)

</div>

<!-- RELATED:END -->
