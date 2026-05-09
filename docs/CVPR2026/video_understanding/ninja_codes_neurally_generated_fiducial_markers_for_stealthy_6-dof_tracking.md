---
title: >-
  [Paper Note] Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking
description: >-
  [CVPR 2026][Video Understanding][Fiducial Markers] Ninja Codes leverages deep steganography to transform arbitrary images into visually inconspicuous fiducial markers via an end-to-end trained encoder. The resulting markers can be printed with standard printers and detected using RGB cameras, enabling stealthy 6-DoF pose tracking.
tags:
  - CVPR 2026
  - Video Understanding
  - Fiducial Markers
  - 6-DoF Tracking
  - Deep Steganography
  - Stealthy Markers
  - Neural Encoding
date: 2026-05-08
content_hash: af8984d3668fb7a6
---

# Ninja Codes: Neurally Generated Fiducial Markers for Stealthy 6-DoF Tracking

**Conference**: CVPR 2026
**arXiv**: [2510.18976](https://arxiv.org/abs/2510.18976)
**Code**: [https://sento.net/research/ninjacodes](https://sento.net/research/ninjacodes)
**Area**: Video Understanding
**Keywords**: Fiducial Markers, 6-DoF Tracking, Deep Steganography, Stealthy Markers, Neural Encoding

## TL;DR
Ninja Codes leverages deep steganography to transform arbitrary images into visually inconspicuous fiducial markers via an end-to-end trained encoder. The resulting markers can be printed with standard printers and detected using RGB cameras, enabling stealthy 6-DoF pose tracking.

## Background & Motivation
1. **Background**: Traditional fiducial markers (ArUco, AprilTag, etc.) are widely adopted for their low cost, easy deployment, and robust performance, but their conspicuous appearance limits their use in aesthetics-sensitive environments.
2. **Limitations of Prior Work**: The black-and-white grid appearance of conventional fiducial markers makes them unsuitable for domestic, exhibition, or other visually demanding settings, hindering the adoption of indoor localization and AR technologies in everyday life.
3. **Key Challenge**: Markers must be sufficiently salient for reliable detection while remaining visually unobtrusive enough to blend into their surroundings — an apparently contradictory requirement.
4. **Goal**: To create stealthy fiducial markers that blend naturally into diverse real-world textures while maintaining reliable 6-DoF tracking capability.
5. **Key Insight**: Drawing inspiration from deep steganography (hiding information within images imperceptibly to the human eye), the paper frames marker generation as an information encoding problem.
6. **Core Idea**: An encoder, decoder, region detector, corner detector, and adversarial network are jointly trained end-to-end to embed a 36-bit ID into environmental texture images through subtle visual modifications.

## Method

### Overall Architecture
The training pipeline proceeds as follows: square patches are cropped from training images → the encoder generates a Ninja Code → print noise is added → the patch is composited back into the original image → camera noise is added → a region detector localizes the marker → a corner detector refines the localization → the decoder recovers the embedded ID. Five network modules (encoder, decoder, region detector, corner detector, and discriminator) are trained jointly in an end-to-end manner.

### Key Designs

1. **Encoder Network**:
    - **Function**: Transforms an RGB cover image and a 36-bit ID into a visually inconspicuous Ninja Code.
    - **Mechanism**: The ID is linearly projected into a tensor matching the spatial dimensions of the cover image, concatenated with the cover image to form a 6-channel input, and processed through a U-Net to produce the encoded Ninja Code. An adversarial loss minimizes the perceptual difference between the encoded and original images.
    - **Design Motivation**: The multi-scale nature of U-Net is well suited for embedding local information while preserving global visual coherence.

2. **Two-Stage Differentiable Noise Simulation**:
    - **Function**: Enhances the robustness of Ninja Codes against real-world perturbations.
    - **Mechanism**: (1) Print noise: color shift and specular reflection simulation; (2) Camera noise: color shift, Gaussian blur, Gaussian noise, and JPEG compression. All noise functions are designed to be differentiable, enabling end-to-end backpropagation.
    - **Design Motivation**: Markers undergo a full pipeline of printing → surface attachment → camera capture, each stage introducing perturbations that must be simulated during training.

3. **Two-Stage Training Strategy**:
    - **Function**: Ensures stable training convergence.
    - **Mechanism**: In the first stage, only detection capability is trained (20 epochs), during which the encoder spontaneously generates colored stripe markers. In the second stage, all losses are introduced (60 epochs), with the image loss weight $w_i$ gradually increased from 1.0 to 100–300, allowing the encoder to progressively learn to generate more stealthy markers. The degree of stealthiness can be controlled by adjusting $w_i$.
    - **Design Motivation**: Direct joint training of all objectives is difficult to converge. Establishing a detection foundation first, then progressively imposing stealthiness constraints, yields greater training stability.

### Loss & Training
The total loss is $L = w_i L_i + w_r L_r + w_c L_c + w_k L_k + w_m L_m + w_a L_a$, comprising an image loss (pixel MSE + chrominance L1 + LPIPS), region detection regression/classification loss, corner MSE loss, message MSE loss, and adversarial loss.

## Key Experimental Results

### Main Results

| Configuration | Corner Error (px) | Miss Rate (%) | Notes |
|---|---|---|---|
| NC₁₀₀ | 0.994 | 3.20 | Low stealthiness |
| NC₂₀₀ | 1.057 | 7.30 | Medium stealthiness |
| NC₃₀₀ | 1.145 | 11.10 | High stealthiness |
| ArUco | 0.586 | 0.00 | Traditional marker baseline |
| NC₃₀₀ + Error Correction | — | 6.00 | Reed-Solomon error correction |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| After removing high-contrast textures | NC₃₀₀ miss rate drops to 8.15% | High-contrast environments are the primary challenge |
| Fine-tuned detector | Approaches dedicated detector performance | Supports a shared detector across multiple encoders |
| 6-DoF position error | ~2.42 cm (NC₂₀₀) | Close to ArUco's 2.18 cm |

### Key Findings
- A trade-off exists between stealthiness and detection reliability: higher $w_i$ yields more stealthy markers but also higher miss rates.
- Most detection failures stem from message recovery errors rather than region localization failures; Reed-Solomon error correction effectively mitigates this issue.
- High-contrast textures (e.g., tiles, grass) pose the greatest challenge, with detection failures concentrated on such images.
- A fine-tuned detector can handle markers generated by different encoders, supporting scene-specific customization.

## Highlights & Insights
- **Addresses a long-standing yet underexplored practical problem**: deploying fiducial markers in aesthetics-sensitive environments.
- The **end-to-end training pipeline** enables collaborative optimization across all modules without manual tuning.
- The two-stage noise simulation design is highly engineering-oriented, separately modeling perturbations introduced during printing and image capture.

## Limitations & Future Work
- Detection reliability degrades notably on high-contrast textures.
- Validation is limited to indoor environments under standard lighting; extreme illumination conditions (e.g., direct intense sunlight) remain untested.
- The encoder and detector are tightly coupled, and markers generated in different training sessions are not mutually compatible.
- Future work may explore applications on non-planar surfaces (drawing on DeepFormableTag).

## Related Work & Insights
- **vs. ArUco/AprilTag**: Traditional markers achieve higher detection accuracy and reliability but cannot blend into their surroundings. Ninja Codes trade a moderate performance degradation for stealthiness.
- **vs. HiDDeN/StegaStamp**: These deep steganography works focus solely on encoding and decoding; this paper additionally incorporates localization capability (region and corner detection).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Applying deep steganography to fiducial marker generation represents a novel cross-domain combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Digital and physical print testing are included, though scene diversity could be further increased.
- Writing Quality: ⭐⭐⭐⭐ Method description is thorough with sufficient experimental detail.
- Value: ⭐⭐⭐⭐ Addresses a genuine practical pain point with clearly defined application scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Mamba-VMR: Multimodal Query Augmentation via Generated Videos for Precise Temporal Grounding](mamba-vmr_multimodal_query_augmentation_via_generated_videos_for_precise_tempora.md)
- [\[CVPR 2026\] Drift-Resilient Temporal Priors for Visual Tracking](drift-resilient_temporal_priors_for_visual_tracking.md)
- [\[CVPR 2026\] FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT](rethinking_twostage_referringbytracking_in_referri.md)
- [\[CVPR 2026\] Event6D: Event-based Novel Object 6D Pose Tracking](event6d_event-based_novel_object_6d_pose_tracking.md)
- [\[CVPR 2026\] UETrack: A Unified and Efficient Framework for Single Object Tracking](uetrack_a_unified_and_efficient_framework_for_single_object_tracking.md)

<!-- RELATED:END -->
