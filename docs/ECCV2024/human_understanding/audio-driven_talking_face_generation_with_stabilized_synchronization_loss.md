---
title: >-
  [Paper Note] Audio-Driven Talking Face Generation with Stabilized Synchronization Loss
description: >-
  [ECCV2024][Human Understanding][talking face generation] This work proposes three improvements—AVSyncNet, stabilized synchronization loss, and a silent-lip generator—to systematically address the two core issues of SyncNet instability and lip leaking in audio-driven talking face generation, achieving SOTA performance in both lip synchronization and visual quality.
tags:
  - "ECCV2024"
  - "Human Understanding"
  - "talking face generation"
  - "lip synchronization"
  - "SyncNet"
  - "lip leaking"
  - "GAN"
date: 2026-05-08
content_hash: d79d458f671fd49e
---

# Audio-Driven Talking Face Generation with Stabilized Synchronization Loss

**Conference**: ECCV2024  
**arXiv**: [2307.09368](https://arxiv.org/abs/2307.09368)  
**Code**: [yamand16/TalkingFaceGeneration](https://yamand16.github.io/TalkingFaceGeneration/)  
**Area**: Human Understanding  
**Keywords**: talking face generation, lip synchronization, SyncNet, lip leaking, GAN

## TL;DR

This work proposes three improvements—AVSyncNet, stabilized synchronization loss, and a silent-lip generator—to systematically address the two core issues of SyncNet instability and lip leaking in audio-driven talking face generation, achieving SOTA performance in both lip synchronization and visual quality.

## Background & Motivation

Audio-driven talking face generation aims to generate realistic videos with accurate lip synchronization and high visual quality from a given audio and reference video while preserving identity and visual features. This task has a wide range of applications in film dubbing, online education, and video conference enhancement.

Current methods face two fundamental challenges:

1. **SyncNet Instability**: The SyncNet introduced by Wav2Lip is widely used to calculate lip-sync loss, but its cosine similarity performance on real data is highly unstable—even ground-truth (GT) audio-lip pairs often receive low scores. This unstable training signal leads to unstable training, poor lip-sync, and degraded visual quality, especially in high-resolution scenarios.
2. **Lip Leaking**: The current standard practice masks the lower half of the face as a pose reference, and randomly selects another frame from the same video as an identity reference to maintain identity information. However, the randomly selected identity reference may have lip shapes similar to the target frame, leading the model to directly copy the lip movements of the identity reference to exploit faster convergence, rather than learning the correct lips from the audio.

## Core Problem

- SyncNet yields wildly fluctuating cosine similarity scores for GT audio-lip pairs, providing incorrect training gradients.
- Lip-sync loss conflicts with reconstruction loss, trapping the model in a dilemma between poor lip-sync or poor visual quality.
- When a randomly appearing lip shape in the identity reference is similar to the target, the model tends to "plagiarize" rather than learn the correct mapping.
- SyncNet has poor shift-invariance; slight displacements significantly affect prediction results.

## Method

### 1. Silent-Lip Generator ($G_S$)

To address the lip leaking issue, a pre-processing module $G_S$ is proposed to modify the lips of the identity reference to a closed state ("silent lip shape") before feeding it into the main generator.

- **Architecture**: The same U-Net architecture as the main generator $G_L$.
- **Training Strategy**: Trained separately on LRS2 **without any synchronization loss**, using only GAN loss + pixel loss + perceptual loss.
- **Key Insight**: Under the absence of sync loss supervision, the model implicitly learns to generate closed lips when silent audio is input.
- **During Inference**: Only silent audio is input to $G_S$, causing it to change the lips of any identity reference into a closed state.
- **Effect**: Eliminates the diversity of lip movements in the identity reference, cutting off the path of lip leaking at the source.

### 2. AVSyncNet

Based on a ResNet-50 image encoder + ResNetSE-34 audio encoder, SyncNet is re-designed to replace the original simple CNN architecture:

- **Image Encoder**: Uses ResNet-50, taking the lower half of the face (112×224) as input, which exhibits stronger shift-invariance.
- **Audio Encoder**: Uses ResNetSE-34 (a ResNet-34 variant specifically designed for spectrograms).
- **Training**: Computes cosine similarity + BCE loss on audio-lip features on LRS2.
- Inputs 5 consecutive frames and corresponding audio at each step, with negative samples randomly selected from non-overlapping parts of the video.
- Experiments show that AVSyncNet significantly reduces cosine similarity fluctuations on GT data and greatly improves shift-invariance.

### 3. Stabilized Synchronization Loss ($L_{ss}$)

Even though AVSyncNet is more stable, instability cannot be completely eliminated. Therefore, a stabilized synchronization loss is proposed:

$$L_{ss} = -\log\left(1 - \frac{|x - y| + \epsilon}{|x - y| + |y - d| + \epsilon}\right)$$

where $x = \text{AVsim}(I', A)$ (generated image-audio similarity), $y = \text{AVsim}(I^{GT}, A)$ (GT-audio similarity), and $d = \text{AVsim}(I^R, A)$ (identity reference-audio similarity).

- **Core Idea**: Instead of directly using the absolute similarity between the generated lips and the audio, it computes the **relative gap** between the GT pair and the generated pair—guiding the model to generate lips with a synchronization score similar to GT.
- **Identity Reference Penalty Term**: Increases the penalty when $d$ is high (high similarity between the identity reference and audio) to further suppress lip leaking.
- Similar to the distillation concept: ignores absolute scores and only leverages the score difference to provide gradients.

### 4. Main Generator Architecture and Training

- **Architecture**: U-Net design, including independent identity encoder and pose encoder (each being CNN + residual connections), and the face decoder uses transposed convolutions + skip connections.
- **Audio Encoder**: Uses the pre-trained and frozen audio encoder of AVSyncNet to obtain better audio embeddings.
- **Adaptive Triplet Loss ($L_{at}$)**: Minimizes the distance between the generated image and GT, maximizes the distance between the generated image and the identity reference, and adaptively adjusts based on the similarity between GT and the identity reference.
- **Total Loss**: $L = L_{GAN} + 10 L_{pixel} + L_{per} + 2 L_{ss} + 0.5 L_{at}$
- **Post-processing**: Uses VQFR to improve output visual quality and resolution.

## Key Experimental Results

Main results on LRS2 test set (without post-processing / with VQFR post-processing vs. previous SOTA):

| Metric | Ours (w/o FR) | Ours (w/ VQFR) | Prev. Best |
|------|:---:|:---:|:---:|
| SSIM ↑ | **0.952** | 0.905 | 0.87 (IPLAP) |
| PSNR ↑ | **32.64** | 31.80 | 29.67 (IPLAP) |
| FID ↓ | **3.83** | 5.23 | 4.10 (IPLAP) |
| LMD ↓ | **1.13** | 1.36 | 2.11 (IPLAP) |
| LSE-C ↑ | 8.41 | **8.52** | 8.53 (TalkLip) |
| LSE-D ↓ | 6.03 | **5.83** | 6.08 (TalkLip) |
| IFC ↓ | **0.16** | 0.27 | 0.20 (IPLAP) |

- The LMD metric leads by a large margin (1.13 vs 2.11), showing a significant improvement in lip accuracy.
- Visual quality (SSIM/PSNR/FID) comprehensively outperforms all methods.
- Achieves SOTA on unseen LRW and HDTF datasets as well.

Key findings from the ablation study:
- Silent-lip generator reduces LMD from 2.325 to 1.741, and increases LSE-C from 7.271 to 7.752.
- Stabilized sync loss improves PSNR from 27.18 to 31.17, and SSIM from 0.872 to 0.925.
- Adaptive triplet loss further improves PSNR to 32.75, and reduces FID to 4.02.
- Replacing SyncNet with AVSyncNet reduces LSE-C variance from 1.16 to 0.97.

## Highlights & Insights

- **Thorough problem analysis**: Systematically identifies two fundamental problems—SyncNet instability and lip leaking—and supports them with visual and quantitative analyses.
- **Ingenious Silent-lip generator design**: Implicitly learns closed-mouth generation through zero-sync-loss training + silent audio input, without requiring additional labeled data.
- **Elegant formulation of Stabilized sync loss**: Replaces absolute scores with relative gaps while incorporating an identity reference penalty, solving both training instability and lip leaking with a single formula.
- **Comprehensive ablation study**: Step-by-step validation of each component's contribution, including different post-processing methods, sync loss variants, and silent face generation strategies.
- The version without post-processing already significantly outperforms SOTA on most metrics, demonstrating the effectiveness of the core method.

## Limitations & Future Work

- The input resolution is only 96×96, relying on VQFR post-processing to increase resolution, which introduces extra artifacts and degrades some metrics.
- Silent-lip generator requires separate pre-training, increasing model complexity and training costs.
- Only trained on LRS2, without exploring the effects on larger-scale datasets.
- 3D facial priors (such as 3DMM) are not considered, which may limit performance under extreme poses.
- Inference requires two forward passes ($G_S$ + $G_L$), potentially posing a performance bottleneck for real-time applications.
- The potential of newer paradigms like diffusion models is not explored.

## Related Work & Insights

| Method | Core Strategy | Limitations |
|------|----------|------|
| Wav2Lip | SyncNet + lip-sync loss | SyncNet instability, degradation at high resolutions |
| TalkLip | Global audio encoder + lip reading loss | LSE-C is slightly better but LSE-D is worse, artifacts in generalization |
| IPLAP | Intermediate landmark + motion field | Good visual quality but insufficient lip synchronization |
| DINet | Deformation module feature alignment | Low FID but poor LSE metrics |
| VideoReTalking | canonical expression pre-processing | Similar approach but inferior identity preservation and efficiency compared to ours |
| SIDGAN | shift-invariant APS-SyncNet | Analyzes similar problems, but our solution is cleaner and more effective |

This work is closest to SIDGAN in problem awareness, but proposes a different path (stabilized loss vs. pyramid model), and our silent-lip generator is a unique contribution.

## Insights & Connections

- The idea of **relative loss design** can be extended to other scenarios with "unreliable teachers"—when the pre-trained evaluation model itself is noisy, replacing absolute scores with relative gaps is a general strategy.
- The concept of "weakly conditional training -> exploiting degenerate behavior during inference" in the Silent-lip generator can inspire other tasks requiring the removal of conditional information.
- The lip leaking problem is widespread in all conditional generation tasks (e.g., information leakage in image editing and style transfer); the analysis and solutions in this paper are highly valuable reference points.
- The independent identity encoder and pose encoder design supports better task decoupling, a paradigm worth promoting in multi-conditional generation.

## Rating
- Novelty: ⭐⭐⭐⭐ (In-depth problem analysis; the three contributions are complementary and highly novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-dataset, comprehensive ablation, and complete qualitative/quantitative analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete mathematical derivation)
- Value: ⭐⭐⭐⭐ (Provides a systematic solution to the core pain points in the talking face generation domain)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation](../../CVPR2026/human_understanding/pc-talk_precise_facial_animation_control_for_audio-driven_talking_face_generatio.md)
- [\[CVPR 2026\] AudioAvatar: Personalized Audio-driven Whole-body Talking Avatars](../../CVPR2026/human_understanding/audioavatar_personalized_audio-driven_whole-body_talking_avatars.md)
- [\[ECCV 2024\] EDTalk: Efficient Disentanglement for Emotional Talking Head Synthesis](edtalk_efficient_disentanglement_for_emotional_talking_head_synthesis.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)
- [\[ECCV 2024\] ScanTalk: 3D Talking Heads from Unregistered Scans](scantalk_3d_talking_heads_from_unregistered_scans.md)

</div>

<!-- RELATED:END -->
