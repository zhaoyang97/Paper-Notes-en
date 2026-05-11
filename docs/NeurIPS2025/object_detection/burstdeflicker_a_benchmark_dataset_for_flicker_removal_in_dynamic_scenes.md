---
title: >-
  [Paper Note] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes
description: >-
  [NeurIPS 2025][Object Detection][flicker removal] This paper introduces BurstDeflicker, the first large-scale benchmark dataset for multi-frame flicker removal (MFFR)…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "flicker removal"
  - "rolling shutter"
  - "burst imaging"
  - "dataset"
  - "Retinex"
date: 2026-05-08
content_hash: 11b69d5cfb8cbeae
---

# BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes

**Conference**: NeurIPS 2025
**arXiv**: [2510.09996](https://arxiv.org/abs/2510.09996)
**Code**: [qulishen/BurstDeflicker](https://github.com/qulishen/BurstDeflicker)
**Area**: Object Detection
**Keywords**: flicker removal, rolling shutter, burst imaging, dataset, Retinex

## TL;DR

This paper introduces BurstDeflicker, the first large-scale benchmark dataset for multi-frame flicker removal (MFFR), comprising three complementary subsets — Retinex-based synthetic data, real-world static data, and green-screen dynamic data — systematically addressing the core bottleneck of obtaining aligned flickering–clean image pairs in dynamic scenes.

## Background & Motivation

**Flicker artifacts are pervasive**: Under artificial lighting powered by alternating current (AC), images captured with short exposures suffer from non-uniform brightness distributions and prominent banding artifacts, arising from the coupling between the rolling shutter's row-by-row exposure mechanism and the periodic luminance variation of light sources. These artifacts significantly degrade image quality and downstream tasks such as detection and tracking.

**Hardware-based solutions are limited**: Traditional methods rely on flicker-detection circuits to extend exposure time, which introduces motion blur, making it difficult to simultaneously achieve deflickering and sharpness. Neuromorphic sensors offer high temporal resolution but are costly and complex to calibrate, precluding deployment on consumer devices.

**Single-image flicker removal (SIFR) has inherent ambiguity**: Flicker artifacts are spatially localized and temporally dynamic; single-frame methods struggle to distinguish flickering dark regions from visually similar phenomena such as shadows, and the lack of pixel context renders restoration unreliable.

**Multi-frame approaches are more promising**: Modern handheld devices routinely capture multiple frames in a single burst, naturally providing rich temporal cues and inter-frame correlations that facilitate accurate flicker localization and removal. However, large-scale, high-quality MFFR datasets are lacking.

**Existing synthetic data is poorly modeled**: Prior flicker synthesis methods by Wong et al. and Lin et al., originally designed for geo-tagging, model flicker removal as the complete elimination of flickering illumination rather than adjustment to an effective value, resulting in training data that lacks diversity and physical realism.

**Paired data acquisition in dynamic scenes is nearly impossible**: Dynamic scenes are non-repeatable, making it infeasible to obtain aligned flickering–clean image pairs. Consequently, models tend to misinterpret motion-induced pixel changes as flicker artifacts, producing ghosting.

## Method

### Overall Architecture

The BurstDeflicker dataset consists of three complementary subsets, each addressing data acquisition challenges from the dimensions of scale, realism, and dynamics:

- **Retinex-based synthetic data**: Models the interaction between ambient light and flickering light based on Retinex theory, supporting unlimited generation of diverse flicker patterns for pretraining.
- **BurstDeflicker-S (4,000 real static images)**: Captures flickering frames with short exposures and clean reference frames with long exposures across 369 real-world scenes.
- **BurstDeflicker-G (3,690 green-screen dynamic images)**: Composites green-screen foregrounds onto real flickering backgrounds to simulate dynamic scenes.

The training pipeline proceeds as follows: synthetic data pretraining → fine-tuning on real data (S + G). During training, 3 frames are randomly sampled (with intervals of 1–3 frames), concatenated as input, and a single clean image is produced as output.

### Key Design 1: Retinex-Based Flicker Synthesis

- **Function**: Redefines the flicker removal objective by adjusting flickering illumination to its effective value rather than eliminating it entirely.
- **Mechanism**: Based on Retinex theory, the flickering image is modeled as $I_{flicker} = R \odot (L_a + L_f)$ and the clean image as $I_{clean} = R \odot (L_a + \overline{L_f})$, where $\overline{L_f}$ denotes the effective value of the flickering light. Through algebraic transformation, this yields $I_{flicker} = I_{clean} \cdot (1 + \frac{L_f/\overline{L_f} - 1}{k+1})$, where $k$ is the ratio of ambient to flickering light intensity.
- **Design Motivation**: Prior methods equate deflickering with the complete removal of flickering illumination ($I_{clean} = R \odot L_a$), which is physically inaccurate — flickering lights do provide effective illumination, with only their instantaneous values fluctuating. The proposed formulation more accurately reflects the underlying physical process.

### Key Design 2: Modeling Multiple Rectification Modes

- **Function**: Models the flicker patterns of three major light source types — full-wave rectification (fluorescent lamps), half-wave rectification (incandescent bulbs), and PWM rectification (LEDs).
- **Mechanism**: AC sinusoidal waveforms produce distinct luminance distribution patterns under different rectification schemes. Each mode's row-wise flicker intensity distribution is precisely modeled via parameterized equations (Eq. 4), with controllable parameters including grid frequency (50/60 Hz), row scan frequency, initial phase, and duty cycle.
- **Design Motivation**: Flicker patterns vary substantially across different lamp types; a simple sinusoidal superposition fails to capture the diversity encountered in the real world.

### Key Design 3: Green-Screen Synthesis for Dynamic Data

- **Function**: Addresses the core challenge of obtaining paired data in dynamic scenes.
- **Mechanism**: Semantically and spatially compatible green-screen foreground clips are selected from the VideoMatte240K dataset and composited onto real flickering backgrounds using alpha masks. The clean ground-truth images are replicated 10 times with the same foreground overlaid, ensuring consistency except for background flicker.
- **Design Motivation**: Dynamic scenes are non-repeatable and cannot be directly captured as aligned pairs. Models trained exclusively on static data tend to misidentify motion-induced pixel changes as flicker, producing ghosting artifacts. The green-screen approach exposes the model to flickering image pairs with motion, enabling it to learn to distinguish motion from flicker.

### Key Design 4: Real-World Data Acquisition Pipeline

- **Function**: Constructs 4,000 real-world flickering–clean paired images.
- **Mechanism**: A tripod-mounted camera with a remote shutter and electronic shutter is used to eliminate mechanical vibration. Short exposures (1/1000–1/2000 s) capture flickering frames, while long exposures (1/50 or 1/60 s) integrate over multiple flicker cycles to obtain clean reference frames. Ten burst frames are captured per scene.
- **Design Motivation**: The 369 scenes span indoor (offices, supermarkets, subway stations) and outdoor (LED billboards, parking lots) environments to ensure diversity. Manual mode is used throughout to maintain consistent imaging conditions.

## Loss & Training

- Synthetic data pretraining → fine-tuning on BurstDeflicker real data.
- Input consists of 3 randomly sampled frames (intervals of 1–3), concatenated along the channel dimension after data augmentation.
- Random rotation ($[-3°, 3°]$) and translation ($[-5, 5]$ pixels) are applied to simulate handheld camera shake.
- Images are resized rather than cropped during training to preserve the periodicity of flicker artifacts along the scan direction.
- Train/test split follows an 8:2 ratio.

## Key Experimental Results

### Table 1: Performance Comparison of Different Methods on Static and Dynamic Test Sets

| Method | FLOPs (G) | Params (M) | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|--------|-----------|------------|--------|--------|---------|---------|--------|-----------|
| Retinexformer* (w/o proposed data) | 69.23 | 1.61 | 15.704 | 0.707 | 0.213 | 53.596 | 50.269 | 30.242 |
| Lin et al.* (original) | 509.80 | 92.08 | 20.358 | 0.838 | 0.134 | 55.228 | 43.875 | 25.121 |
| Lin et al. (retrained on proposed data) | 509.80 | 92.08 | 26.408 | 0.875 | 0.102 | 58.131 | 35.710 | 22.102 |
| Retinexformer (retrained) | 69.23 | 1.61 | 27.212 | 0.885 | 0.081 | 58.249 | 35.942 | 21.648 |
| Burstormer | 141.05 | 0.17 | 29.439 | 0.910 | 0.056 | 58.527 | 37.014 | 20.451 |
| HDRTransformer | 272.12 | 1.04 | 30.031 | 0.914 | 0.054 | 59.069 | 37.292 | 21.588 |
| **Restormer** | **149.01** | **7.92** | **30.634** | **0.918** | **0.045** | **59.097** | **34.896** | **19.324** |

**Key Findings**: After retraining on BurstDeflicker, Lin et al.'s PSNR improves from 20.358 to 26.408 (+6.05 dB), and Retinexformer's from 15.704 to 27.212 (+11.51 dB), validating the effectiveness of the dataset. Restormer with 3-frame input achieves the best performance at 30.634 dB.

### Table 2: Ablation Study on Dataset Subsets (Restormer)

| Synthetic | BurstDeflicker-S | BurstDeflicker-G | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|-----------|-----------------|-----------------|--------|--------|---------|---------|--------|-----------|
| ✓ | | | 24.483 | 0.862 | 0.122 | 57.096 | 39.726 | 23.755 |
| ✓ | ✓ | | 30.481 | 0.915 | 0.053 | 58.011 | 37.498 | 21.523 |
| ✓ | | ✓ | 30.645 | 0.916 | 0.052 | 58.431 | 36.259 | 20.338 |
| ✓ | ✓ | ✓ | **30.634** | **0.918** | **0.045** | **59.097** | **34.896** | **19.324** |

**Key Findings**: The three subsets are mutually complementary — synthetic data pretraining enhances robustness; the green-screen dynamic subset yields substantial improvements on the dynamic test set (BRISQUE reduced from 21.523 to 19.324), effectively mitigating ghosting artifacts.

### Table 3: Ablation Study on Number of Input Frames (Restormer)

| Input | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------|--------|--------|---------|
| Single frame | 27.310 | 0.891 | 0.069 |
| 2 frames | 30.264 (+2.954) | 0.915 | 0.048 |
| 3 frames | 30.634 (+3.324) | 0.918 | 0.045 |

The gain from 1 to 2 frames is 2.954 dB, while the gain from 2 to 3 frames is only 0.37 dB, reflecting diminishing marginal returns due to inter-frame information redundancy.

## Highlights & Insights

1. **Problem redefinition**: The paper corrects the erroneous modeling in prior work that equates deflickering with complete removal of flickering illumination, arguing instead that the target should be adjustment to the effective value. This physically grounded correction offers meaningful guidance for future work.
2. **Comprehensive three-stage data construction strategy**: Synthetic data (scale) → real static data (realism) → green-screen dynamic data (dynamic scenes). Each subset addresses a distinct bottleneck, yielding a well-structured design.
3. **Clever and practical use of green-screen compositing**: Borrowing a mature technique from the film industry to solve an academic dataset construction challenge, enabling models to learn to distinguish motion-induced from flicker-induced pixel changes.
4. **Strong generalizability**: The dataset is architecture-agnostic, benefiting diverse models including Restormer, Burstormer, and HDRTransformer.

## Limitations & Future Work

1. When the flickering source is the sole light source, degradation is severe and the restored output may exhibit noticeable color shifts.
2. Under severe inter-frame misalignment (e.g., significant handheld camera shake), multi-frame method performance degrades to the single-frame level.
3. Green-screen compositing remains semi-synthetic, and a domain gap with real dynamic scenes persists.
4. No architecture specifically tailored to flicker priors is proposed; general-purpose restoration networks are adapted instead.

## Related Work & Insights

- **Comparison with DeflickerCycleGAN (Lin et al.)**: This single-frame method performs poorly under severe flicker and low-light conditions; after retraining on the proposed dataset, PSNR reaches only 26.408, far below multi-frame approaches.
- **Comparison with Retinexformer**: Low-light enhancement methods globally brighten images and introduce color shifts, making them unsuitable for flicker removal.
- **Connection to burst super-resolution**: The paper draws on BurstSR's multi-frame pretraining-then-fine-tuning strategy and the training technique of using resize instead of crop.
- **Broader implications**: The three-stage dataset construction paradigm (synthetic pretraining → real-world collection → semi-synthetic dynamic data) is generalizable to other restoration tasks where paired data is difficult to obtain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First MFFR dataset; physically grounded Retinex reformulation of the deflickering objective; creative use of green-screen compositing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple baselines, comprehensive ablations (data subsets, frame count), and a real dynamic test set; comparisons with more recent methods are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem formulation is clear, the three-stage structure is logically progressive, and figures and tables are well-presented.
- **Value**: ⭐⭐⭐⭐ — Fills the gap in MFFR datasets and substantively advances research on flicker removal; dataset is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[NeurIPS 2025\] Ascent Fails to Forget](ascent_fails_to_forget.md)

</div>

<!-- RELATED:END -->
