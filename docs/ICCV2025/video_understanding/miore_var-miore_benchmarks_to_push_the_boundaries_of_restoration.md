---
title: >-
  [Paper Note] MIORe & VAR-MIORe: Benchmarks to Push the Boundaries of Restoration
description: >-
  [ICCV 2025][Video Understanding][Motion Deblurring] This paper introduces MIORe and VAR-MIORe, two multi-task motion restoration benchmark datasets captured using a 1000fps industrial-grade camera and a professional lens…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Motion Deblurring"
  - "Video Frame Interpolation"
  - "Optical Flow Estimation"
  - "High Frame Rate Dataset"
  - "Multi-Task Benchmark"
date: 2026-05-08
content_hash: 5b33c91e3fdc0e32
---

# MIORe & VAR-MIORe: Benchmarks to Push the Boundaries of Restoration

**Conference**: ICCV 2025
**arXiv**: [2509.06803](https://arxiv.org/abs/2509.06803)  
**Code**: [https://github.com/george200150/MIORe](https://github.com/george200150/MIORe)  
**Area**: Image/Video Restoration
**Keywords**: Motion Deblurring, Video Frame Interpolation, Optical Flow Estimation, High Frame Rate Dataset, Multi-Task Benchmark

## TL;DR

This paper introduces MIORe and VAR-MIORe, two multi-task motion restoration benchmark datasets captured using a 1000fps industrial-grade camera and a professional lens array. The benchmarks span a full motion magnitude spectrum from near-static to extreme motion, employ an adaptive frame-averaging mechanism to generate consistent motion blur, and provide a unified evaluation platform for deblurring, frame interpolation, and optical flow estimation.

## Background & Motivation

Existing motion restoration datasets suffer from several critical limitations:

**Limited Frame Rate**: Datasets such as GoPro (240fps) offer insufficient frame rates, constraining the controllable range of motion blur and the quality of optical flow ground truth.

**Narrow Motion Diversity**: Most datasets cover only a restricted set of motion types and fixed motion magnitudes, lacking continuous variation from subtle to extreme motion.

**Task Fragmentation**: Deblurring (GoPro), frame interpolation (Vimeo90K), and optical flow estimation (SINTEL/KITTI) each rely on separate datasets, precluding unified multi-task evaluation.

**Absence of Optical Degradations**: Existing datasets do not incorporate real optical degradations such as defocus blur, lens aberrations, or vignetting.

**Insufficient Environmental Diversity**: Comprehensive coverage of seasonal, meteorological, and temporal variations is lacking.

## Method

### Overall Architecture

The dataset construction pipeline proceeds as follows: high-speed capture at 1000fps → optical flow computation → adaptive frame averaging to synthesize motion blur → retention of sharp left/center/right frames as ground truth → assignment to three restoration tasks.

### Key Designs

1. **Adaptive Frame-Averaging Mechanism**: Unlike conventional fixed-count averaging (e.g., GoPro's fixed 7–13 frames), the number of averaged frames is determined dynamically based on the mean and maximum optical flow of each scene. High-flow scenes (~10 px/frame) average only 3 frames to prevent over-blurring, while sub-pixel motion scenes average up to 30 frames to amplify blur. The objective is to normalize motion blur across all scenes to a moderate level of approximately 30-pixel optical flow. A critical constraint is that the leftmost and rightmost boundary frames remain fully sharp, serving as inputs for frame interpolation and optical flow estimation with no blur cues—substantially increasing task difficulty.

2. **Variable Motion Magnitude in VAR-MIORe**: This is the first dataset to explicitly incorporate "motion magnitude" as a benchmark parameter. By controlling the frame-averaging offset (1–249ms), the dataset provides continuous coverage from static to extreme motion, with maximum optical flow reaching 1932 pixels—far exceeding SINTEL (414) and KITTI (355). Seven intensity buckets (1/5/13/29/61/125/249 frames) form a gradient evaluation scheme that precisely identifies the performance "breaking point" of each algorithm.

3. **Diverse Acquisition Settings**:

    - **Camera**: CHRONOS 2.1-HD industrial high-speed camera, 1920×1080 @ 1000fps, up to 5516 frames per capture.
    - **Lens Array**: Four professional lenses — Tamron 15–30mm zoom (wide to standard), Canon 24mm (wide-angle), Sigma 85mm (telephoto), Laowa 100mm (macro) — all shot at maximum aperture to maximize light intake.
    - **Motion Types**: Ego-motion (translation/rotation/push-pull/tilt/roll/tracking pan) + scene motion (multi-subject/parallax/deformable objects) + complex effects such as dolly zoom.
    - **Environmental Coverage**: Four seasons, all-weather conditions (cloud/fog/rain/snow), and varying times of day (color temperature variation from sunrise to sunset).
    - **Optical Degradations**: Defocus blur (bokeh from shallow depth of field), overexposure, underexposure, lens aberrations, and vignetting.

### Loss & Training

As a dataset paper, no model training is involved. Evaluation metrics are as follows:
- **Deblurring**: PSNR, SSIM (blurry image vs. corresponding sharp center frame)
- **Frame Interpolation**: PSNR, SSIM (predicted intermediate frame vs. GT intermediate frame)
- **Optical Flow Estimation**: Standard EPE (End-Point Error)

## Key Experimental Results

### Main Results (MIORe Deblurring)

| Method | XF (Extreme) | F (Fast) | MF (Med-Fast) | M (Med) | MS (Med-Slow) | S (Slow) | Overall PSNR |
|--------|-------------|---------|--------------|---------|--------------|---------|--------------|
| AdaRevD | 31.79 | 31.92 | 32.22 | 32.04 | 30.09 | 31.30 | **31.70** |
| LoFormer | 31.71 | 31.77 | 31.81 | 32.20 | 30.12 | 31.32 | 31.61 |
| FFTformer | 27.14 | 27.80 | 28.11 | 28.52 | 26.29 | 26.89 | 27.48 |
| UFPNet | 31.48 | 31.98 | **32.66** | **32.53** | **30.62** | **31.64** | 31.78 |
| NAFNet | 26.73 | 28.98 | 31.12 | 30.63 | 29.36 | 27.59 | 28.56 |

MIORe Frame Interpolation Results (PSNR):

| Method | XF | F | MF | M | MS | S | Overall |
|--------|----|---|----|---|----|---|---------|
| VFIMamba | 35.67 | 34.55 | 34.53 | 34.32 | 31.86 | 30.29 | **34.41** |
| SGM-VFI | 35.60 | 34.14 | 34.29 | 34.25 | 31.81 | 30.43 | 34.24 |
| EMA-VFI | 35.60 | 34.30 | 34.43 | 34.26 | 32.09 | 30.68 | 34.35 |
| BiFormer | 33.47 | 32.57 | 32.73 | 32.79 | 31.05 | 29.42 | 32.60 |
| PerVFI | 31.69 | 30.77 | 30.86 | 31.17 | 29.43 | 28.31 | 30.88 |

### Ablation Study (VAR-MIORe Deblurring — Variable Motion Magnitude)

| Method | 1 frame (no blur) | 5 frames | 13 frames | 29 frames | 61 frames | 125 frames | 249 frames (extreme) |
|--------|------------------|----------|-----------|-----------|-----------|------------|----------------------|
| AdaRevD | 39.61 | 34.03 | 31.39 | 28.42 | 25.83 | 23.27 | 21.39 |
| LoFormer | 41.41 | 34.09 | 31.36 | 28.43 | 25.71 | 23.23 | 21.42 |
| FFTformer | 35.33 | 29.17 | 27.34 | 25.72 | 24.20 | 22.41 | 20.91 |
| UFPNet | 37.83 | 33.63 | 31.44 | 28.39 | 25.65 | 22.98 | 21.15 |
| NAFNet | 25.69 | 28.25 | 27.99 | 26.47 | 24.19 | 21.87 | 20.03 |
| Baseline (input) | ∞ | 34.32 | 30.05 | 27.13 | 24.88 | 23.03 | 21.52 |

VAR-MIORe Frame Interpolation (PSNR):

| Method | 1 frame | 5 frames | 13 frames | 29 frames | 61 frames | 125 frames | 249 frames |
|--------|---------|----------|-----------|-----------|-----------|------------|------------|
| VFIMamba | 67.79 | 37.18 | 34.24 | 28.68 | 23.56 | 20.58 | 18.91 |
| EMA-VFI | **81.16** | 37.06 | 34.08 | 29.07 | 24.00 | 20.78 | 18.91 |
| SGM-VFI | 56.80 | 37.06 | 33.70 | 28.69 | 23.94 | 20.68 | 18.71 |

### Key Findings

1. **Asymmetric Fast/Slow Performance of Frequency-Based Methods**: FFTformer performs relatively well in fast-motion scenes by exploiting contrast cues, but produces hallucinations in slow-motion scenes, yielding substantially lower PSNR. Frequency priors become detrimental when mismatched to actual blur characteristics.
2. **Counter-Intuitive Behavior in Frame Interpolation**: All VFI methods perform better under extreme motion (XF) than under medium-slow motion (MS/S), presumably because extreme-motion scenes feature simpler or more predictable backgrounds.
3. **Decay Curves Revealed by VAR-MIORe**: From 1 to 249 frames, all methods exhibit a continuous PSNR decline of approximately 18–20 dB, but at differing rates — AdaRevD and LoFormer degrade uniformly across all intensities, whereas NAFNet collapses under extreme motion.
4. **Reference Value of the Input Baseline**: At high motion magnitudes (61–249 frames), most methods achieve only marginally higher PSNR than the blurry input itself, demonstrating that extreme-motion deblurring remains an open problem.
5. **Superior Dataset Coverage**: MIORe simultaneously encompasses all four motion types and achieves a maximum optical flow of 1932 pixels, far surpassing existing datasets.

## Highlights & Insights

- This is the first dataset to explicitly introduce variable motion magnitude as a benchmark parameter, enabling precise identification of algorithm performance breaking points.
- The adaptive frame-averaging design is elegant and effective: dynamically adjusting based on optical flow ensures cross-scene blur consistency.
- The constraint that left/right sharp frames contain no blur information substantially raises the difficulty and realism of the VFI and optical flow tasks.
- The diverse optical degradations introduced by four professional lenses (defocus/aberration/vignetting) constitute a distinctive contribution, closely approximating real-world shooting conditions.
- The 333 curated sequences spanning four seasons, all-weather conditions, and diverse scenes offer unmatched environmental diversity among comparable datasets.

## Limitations & Future Work

- Hardware constraints of a single high-speed camera (RAM limits each capture to 5516 frames) preclude coverage of long-duration scenes.
- Optical flow ground truth is computed via algorithms (pseudo-labels) rather than physical measurement, introducing potential errors.
- Only single-image deblurring benchmarks are currently provided; video deblurring (multi-frame input) scenarios are not explored.
- The total dataset size (52K frames), while substantial, falls short of the million-scale data that large data-driven models may require.
- No semantic annotations (object categories, scene segmentation, etc.) are provided, limiting research on semantically guided restoration.

## Related Work & Insights

- **GoPro/RealBlur**: Classic deblurring datasets with limited frame rates and motion range.
- **Vimeo90K**: Standard VFI dataset, but extreme-motion sequences are filtered out.
- **SINTEL/FlyingChairs**: Synthetic optical flow datasets lacking the coupling of real motion blur and defocus.
- **X4K1000FPS**: Also captured at 1000fps but at lower resolution (768×768) with frequent defocus regions.
- **AdaRevD/LoFormer**: Current state-of-the-art deblurring methods, achieving top performance on MIORe.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The variable motion magnitude concept is novel and the adaptive frame-averaging design is well-crafted, though the core contribution remains a dataset.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive benchmarking across three tasks, six motion intensity groups, and seven gradient buckets in VAR-MIORe.
- **Writing Quality**: ⭐⭐⭐ Content is thorough but somewhat lengthy; the motion type classification section could be condensed.
- **Value**: ⭐⭐⭐⭐ Fills the gap in unified multi-task motion restoration evaluation; the extreme-motion data in VAR-MIORe offers meaningful value for pushing the boundaries of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)

</div>

<!-- RELATED:END -->
