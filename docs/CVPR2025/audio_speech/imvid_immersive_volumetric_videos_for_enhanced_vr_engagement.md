---
title: >-
  [Paper Note] ImViD: Immersive Volumetric Videos for Enhanced VR Engagement
description: >-
  [CVPR 2025][Audio & Speech][Volumetric Video] This work constructs the first immersive volumetric video dataset by capturing 7 indoor/outdoor scenes using a mobile multi-view system with 46 synchronized GoPros. It proposes STG++, which introduces learnable affine color transformations to resolve cross-camera color inconsistency, achieving rendering at 110.47 FPS with 387MB of storage, and integrates HRTF spatial audio.
tags:
  - "CVPR 2025"
  - "Audio & Speech"
  - "Volumetric Video"
  - "VR Immersion"
  - "Multi-view GoPro"
  - "Temporal 3DGS"
  - "Spatial Audio"
date: 2026-05-08
content_hash: 2f28da8add86c1bf
---

# ImViD: Immersive Volumetric Videos for Enhanced VR Engagement

**Conference**: CVPR 2025  
**arXiv**: [2503.14359](https://arxiv.org/abs/2503.14359)  
**Code**: To be released  
**Area**: Audio & Speech / VR  
**Keywords**: Volumetric Video, VR Immersion, Multi-view GoPro, Temporal 3DGS, Spatial Audio

## TL;DR

This work constructs the first immersive volumetric video dataset by capturing 7 indoor/outdoor scenes using a mobile multi-view system with 46 synchronized GoPros. It proposes STG++, which introduces learnable affine color transformations to resolve cross-camera color inconsistency, achieving rendering at 110.47 FPS with 387MB of storage, and integrates HRTF spatial audio.

## Background & Motivation

### Background

**Background**: VR experiences require realistic free-viewpoint rendering. Existing volumetric video datasets are either based on fixed camera arrays (limited spatial range) or lack sufficient resolution and frame rate to support immersive experiences.

**Limitations of Prior Work**: (1) Lack of high-resolution (5K+), high-frame-rate (60FPS), synchronized multi-view dynamic scene data; (2) fixed arrays have limited coverage angles and do not support free movement; (3) existing methods do not address cross-camera color discrepancies (due to lighting occlusion causing inconsistent exposure among GoPros).

**Key Challenge**: Mobile capturing provides larger spatial coverage, but camera pose estimation is challenging (COLMAP fails on video sequences).

**Key Insight**: A dual-strategy capture approach—fixed-point capturing (dense temporal sequences calibratable with COLMAP) + mobile capturing (large-scale coverage, with poses to be resolved). STG++ incorporates learnable color transformations to resolve cross-camera color inconsistency.

**Core Idea**: A mobile array of 46 GoPros + STG++ color correction + HRTF spatial audio = the first immersive volumetric video dataset.

### Proposed Solution

**Goal**: ### Key Designs

1. **Capture System**: 46 synchronized GoPros, 5312×2988@60FPS.


## Method

### Key Designs

1. **Capture System**: 46 synchronized GoPros, 5312×2988@60FPS. Dual strategy: fixed-point (dense temporal) + mobile (large-scale)

2. **STG++**: Incorporates a learnable per-camera affine color transformation $C'_i = WC_i + T$ to standard STG (Spacetime Gaussians)—eliminates cross-camera color inconsistencies caused by lighting occlusions

3. **HRTF Spatial Audio**: Converts monaural audio to binaural stereo based on HRTF (Head-Related Transfer Function), dynamically adjusted according to the listener-source direction $\theta_s$ and distance $\lambda$

### Loss & Training

$\mathcal{L} = (1-\lambda_1)L_1 + \lambda_1 D_{SSIM}$. Segmented training on 60-frame sequences.

## Key Experimental Results

| Scene | STG++ PSNR | FPS | Memory |
|------|-----------|-----|------|
| Opera | 31.24% | 110.47 | 387MB |
| Lab | 27.58% | — | — |
| 4DRotor (Comparison) | — | 46.22% | 5818MB |

User Study (21 experts): Spatial perception 61.9% Excellent, overall immersion 90.46% ≥ Good.

### Ablation Study
- STG++ color correction eliminates color flickering between segments.
- Joint modeling of direction + distance for spatial audio outperforms single-factor modeling.
- 4DRotor performs better in high-motion areas but requires 15x more memory.

### Key Findings
- **Color inconsistency is a core challenge in segmented training**—STG++'s affine transformation is simple yet effective.
- **High user immersion**: Over 90% of experts rated it as Good or Excellent.
- **Mobile capture pose estimation remains unresolved**—this is a key technological challenge left for future work.

## Highlights & Insights
- **First high-quality volumetric video dataset for VR**—7 scenes, 46 synchronized cameras, 60FPS 5K resolution.
- **A trinity of dataset + method + evaluation**—not only provides data, but also offers improved rendering methods and subjective evaluations.

## Limitations & Future Work
- **Uncalibrated mobile capture poses**—COLMAP fails on video sequences.
- Local flickering still exists (despite global color alignment).
- Sound field model assumes a single static omnidirectional sound source.
- Continuous shooting is limited by heat dissipation (~30 minutes).

## Rating
- Novelty: ⭐⭐⭐⭐ The first immersive volumetric video + spatial audio dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rendering comparisons + user study + audio evaluation.
- Writing Quality: ⭐⭐⭐⭐ Detailed dataset descriptions.
- Value: ⭐⭐⭐⭐ Provides key resources for VR content creation and free-viewpoint rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Action2Sound: Ambient-Aware Generation of Action Sounds from Egocentric Videos](../../ECCV2024/audio_speech/action2sound_ambientaware_generation_of_action_sounds_from_e.md)
- [\[ECCV 2024\] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](../../ECCV2024/audio_speech/latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)
- [\[ICCV 2025\] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation](../../ICCV2025/audio_speech/align_your_rhythm_generating_highly_aligned_dance_poses_with_gating-enhanced_rhy.md)
- [\[ACL 2026\] VAPO: End-to-end Slide-Enhanced Speech Recognition with Omni-modal Large Language Models](../../ACL2026/audio_speech/vapo_end-to-end_slide-enhanced_speech_recognition_with_omni-modal_large_language.md)
- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)

</div>

<!-- RELATED:END -->
