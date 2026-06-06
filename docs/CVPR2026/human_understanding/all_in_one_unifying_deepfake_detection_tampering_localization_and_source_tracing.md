---
title: >-
  [Paper Note] All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark
description: >-
  [CVPR2026][Human Understanding][deepfake detection] This paper proposes LIDMark, the first proactive forensics framework that unifies deepfake detection, tampering localization…
tags:
  - "CVPR2026"
  - "Human Understanding"
  - "deepfake detection"
  - "watermarking"
  - "tampering localization"
  - "source tracing"
  - "proactive forensics"
  - "facial landmark"
date: 2026-05-08
content_hash: 8584c9c212f8b791
---

# All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark

**Conference**: CVPR2026
**arXiv**: [2602.23523](https://arxiv.org/abs/2602.23523)  
**Code**: [GitHub](https://github.com/vpsg-research/LIDMark)  
**Area**: Human Understanding
**Keywords**: deepfake detection, watermarking, tampering localization, source tracing, proactive forensics, facial landmark

## TL;DR

This paper proposes LIDMark, the first proactive forensics framework that unifies deepfake detection, tampering localization, and source tracing within a single watermarking scheme. By embedding a 152-dimensional Landmark-Identity watermark (136D facial landmarks + 16D source ID) and leveraging intrinsic/extrinsic consistency, LIDMark achieves three-in-one forensics while surpassing existing methods in both PSNR/SSIM and detection accuracy.

## Background & Motivation

The rapid advancement of deepfake technology poses severe security threats. Existing forensic methods fall into two broad categories:

**Passive forensics**: Forgery traces are extracted directly from images for detection. Key limitations include: (a) only binary classification (real/fake) is supported, with no ability to localize tampered regions or trace sources; (b) poor generalization, with performance degrading sharply on unseen forgery methods.

**Proactive forensics (watermarking)**: Watermarks are embedded in images in advance, and forensic analysis is performed by monitoring watermark preservation or destruction. Limitations of existing watermarking methods (e.g., FaceSigns, MBRS, PIMoG) include:
   - Most support detection only, without tampering localization
   - Limited watermark capacity (typically 30 bits), insufficient for encoding multiple types of information simultaneously
   - Detection and localization require different watermark designs, making unification difficult

**Core Insight**: Facial landmarks inherently possess two complementary properties — (1) sensitivity to tampering (landmark distributions shift after face swapping), making them suitable for localization; and (2) the source identity ID needs to be robust against forgery, making it suitable for source tracing. Encoding both into a unified watermark simultaneously addresses all three forensic tasks.

## Core Problem

How to design a unified proactive forensics framework that achieves the following within a single watermark:
- **Deepfake Detection**: Determining whether an image has been tampered with
- **Tampering Localization**: Precisely identifying the tampered facial regions
- **Source Tracing**: Tracing the original source identity of the image

## Method

### LIDMark Watermark Design (152-dimensional)

The watermark consists of two components:

- **Landmark Component (136D)**: The $(x, y)$ coordinates of 68 facial landmarks, normalized to $[0,1]$. This component is sensitive to tampering — after face swapping, the recovered watermark landmarks become inconsistent with those detected on the current image.
- **Identity Component (16D)**: Binary encoding of the source identity ID. This component is designed to be robust against various forgery operations and is used for source tracing.

### Encoder: Dual-Stream Fusion Network

The encoder adopts a dual-stream architecture to embed the watermark into the image:

- **Image Stream**: SEResNet-based feature extraction capturing image content features
- **Watermark Stream**: DiffusionNet-based watermark processing, mapping the 152-dimensional vector to a feature map of the same spatial size as the image
- **Fusion Mechanism**: Features from both streams are fused via concatenation, with skip connections added to preserve image quality

The embedding process is formulated as: $I_w = E(I, m)$, where $I$ is the original image and $m = [m_{\text{land}}, m_{\text{id}}]$ is the 152-dimensional watermark.

### FHD: Factorized-Head Decoder

The decoder adopts a unified Factorized-Head Decoder (FHD) design:

- **Shared Backbone**: Extracts common feature representations from the watermarked image
- **Regression Head**: Outputs 136-dimensional landmark coordinates $\hat{m}_{\text{land}}$, trained with L1 loss
- **Classification Head**: Outputs 16-dimensional source ID $\hat{m}_{\text{id}}$, trained with BCE loss

FHD is more unified and parameter-efficient than a dual-decoder design, and the shared backbone enables mutual benefit between the two tasks.

### Intrinsic–Extrinsic Consistency Detection

This is the key mechanism enabling three-in-one forensics:

1. **Intrinsic Landmarks**: Landmarks $\hat{m}_{\text{land}}$ recovered by FHD from the watermarked image (i.e., the original landmarks embedded at encoding time)
2. **Extrinsic Landmarks**: Landmarks $m_{\text{ext}}$ re-detected on the current image using an external facial landmark detector (e.g., dlib)

**Detection Logic**:

- **Global Detection**: The Average Euclidean Distance (AED) is computed as $\text{AED} = \frac{1}{68}\sum_{i=1}^{68} \| \hat{p}_i - p_i^{\text{ext}} \|_2$. If AED exceeds threshold $\tau$, the image is classified as forged.
- **Region-Level Localization**: Spatial analysis of per-landmark displacement identifies regions with large deviations as tampered areas.
- **Source Tracing**: The 16D ID decoded by the classification head identifies the original source.

### Two-Stage Training Strategy

- **Stage 1 – Pre-training**: The encoder–decoder is trained with standard image distortions (JPEG compression, Gaussian noise, cropping, etc.) to establish basic watermark embedding and extraction capabilities.
- **Stage 2 – Fine-tuning**: The model is fine-tuned using forged images generated by deepfake methods (SimSwap, UniFace, CSCS, StarGAN-v2) to enhance robustness in deepfake scenarios.

### Loss & Training

The total training loss is:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{image}} + \lambda_2 \mathcal{L}_{\text{land}} + \lambda_3 \mathcal{L}_{\text{id}}$$

where $\mathcal{L}_{\text{image}}$ is the image quality loss (L2 + LPIPS), $\mathcal{L}_{\text{land}}$ is the L1 regression loss for landmarks, and $\mathcal{L}_{\text{id}}$ is the BCE classification loss for the identity component.

## Key Experimental Results

### Main Results

#### Image Quality

| Resolution | PSNR ↑ | SSIM ↑ | Capacity |
|------------|--------|--------|----------|
| 128×128 | **40.22** | **0.98** | 152 bits |
| 256×256 | **44.31** | **0.99** | 152 bits |
| Best Baseline (MBRS) | 38.76 | 0.97 | 30 bits |

Despite substantially higher capacity (152 bits vs. 30 bits), LIDMark achieves superior image quality compared to all baselines.

#### Deepfake Detection Performance

| Dataset | Method | AUC ↑ |
|---------|--------|-------|
| CelebA-HQ | LIDMark | **Best** |
| LFW | LIDMark | **Best** |

LIDMark achieves the best detection AUC on both CelebA-HQ and LFW compared to existing proactive forensics methods.

#### Tampering Localization Accuracy

Through region-level landmark displacement analysis, LIDMark generates tampering heatmaps that closely correspond to face-swapped regions, with IoU significantly outperforming methods based on global watermark differences.

#### Source Tracing Accuracy

The 16D identity component achieves over 95% recovery accuracy after various forgery attacks, validating the effectiveness of the robustness design for the ID component.

### Ablation Study

| Component | PSNR | Detection AUC | Notes |
|-----------|------|---------------|-------|
| Full LIDMark | 40.22 | Best | — |
| w/o skip connection | 38.5 | Degraded | Significant image quality drop |
| Dual decoder instead of FHD | 39.8 | Comparable | More parameters, slightly lower quality |
| Stage 1 training only | 40.1 | Degraded | Not robust to deepfakes |

## Highlights & Insights

1. **Three-in-One Unified Framework**: For the first time, detection, localization, and tracing are unified within a single watermarking scheme, eliminating the need for separate watermark designs per task.
2. **The 152-Dimensional Watermark Design is Highly Elegant**: It exploits the natural dual properties of facial landmarks (tamper-sensitivity + semantic richness), elevating watermarking from "bit encoding" to "semantic encoding."
3. **Intrinsic–Extrinsic Consistency** is the core innovation — reformulating watermark recovery as a consistency verification problem naturally extends detection to localization.
4. **FHD Factorized Decoding**: A shared backbone with task-specific heads is more efficient than a decoupled design and enables mutual task reinforcement.
5. **High Image Quality at High Capacity**: At 152 bits — far exceeding existing methods (30 bits) — LIDMark still achieves superior PSNR/SSIM.

## Limitations & Future Work

1. **Fundamental Constraint of Proactive Forensics**: Watermarks must be embedded before image distribution; the framework is inapplicable to existing un-watermarked images.
2. **Resolution Limitation**: Experiments are conducted only at 128×128 and 256×256; scalability to high resolutions (1024+) remains unclear.
3. **Coverage of Forgery Methods**: Fine-tuning uses only four deepfake methods; generalization to emerging forgery techniques (e.g., diffusion model-based generation) warrants further investigation.
4. **Dependency on Landmark Detectors**: Extrinsic consistency relies on the accuracy of detectors such as dlib; detector failures adversely affect the framework.
5. **Adversarial Robustness**: Robustness against adversarial watermark removal attacks is not discussed.

## Related Work & Insights

- Compared to **FaceSigns** (Neekhara et al., 2022): FaceSigns supports detection only (watermark present/absent), whereas LIDMark extends to localization and tracing.
- Compared to **MBRS** (Jia et al., 2021): MBRS has only 30-bit capacity and no tampering localization; LIDMark provides 152 bits with localization.
- Compared to passive methods (**Xception**, **Face X-ray**): Passive methods require no preprocessing but generalize poorly; LIDMark trades deployment convenience for reliable three-in-one capability.
- **Insight**: Watermarks need not be arbitrary bit strings — encoding watermarks using domain semantics (e.g., landmarks) can achieve functionality far beyond conventional watermarking. This idea is transferable to other domains (e.g., anatomical keypoints in medical imaging, landmark encoding in remote sensing).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First three-in-one proactive forensics framework with an elegant watermark design
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset, multi-baseline comparisons are thorough, but high-resolution and broader forgery method evaluations are lacking
- **Value**: ⭐⭐⭐⭐ — Clear practical value for proactive forensics scenarios, contingent on pre-embedding requirements
- **Writing Quality**: ⭐⭐⭐⭐ — Framework is described clearly with well-motivated rationale
- **Overall**: ⭐⭐⭐⭐ (4.0/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unleashing Vision-Language Semantics for Deepfake Video Detection](unleashing_vision-language_semantics_for_deepfake_video_detection.md)
- [\[CVPR 2026\] Face Time Traveller: Travel Through Ages Without Losing Identity](face_time_traveller_travel_through_ages_without_losing_identity.md)
- [\[CVPR 2026\] A2P: From 2D Alignment to 3D Plausibility for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_hete.md)
- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2026\] Beyond the Fold: Quantifying Split-Level Noise and the Case for Leave-One-Dataset-Out AU Evaluation](beyond_the_fold_quantifying_split-level_noise_and_the_case_for_leave-one-dataset.md)

</div>

<!-- RELATED:END -->
