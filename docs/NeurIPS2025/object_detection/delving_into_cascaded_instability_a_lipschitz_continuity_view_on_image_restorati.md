---
title: >-
  [Paper Note] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes
description: >-
  [NeurIPS 2025][Object Detection][flicker removal] This paper introduces BurstDeflicker, the first benchmark dataset for multi-frame flicker removal (MFFR) in dynamic scenes. It is constructed through three complementary strategies—Retinex-based synthesis, real-world static scene capture, and green-screen compositing—enabling large-scale training and evaluation that significantly improves the generalization of flicker removal models to real-world dynamic scenes.
tags:
  - NeurIPS 2025
  - Object Detection
  - flicker removal
  - rolling shutter
  - Retinex theory
  - burst image restoration
  - benchmark dataset
date: 2026-05-08
content_hash: 7b4a953800dfed19
---

# BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes

**Conference**: NeurIPS 2025  
**arXiv**: [2510.09996](https://arxiv.org/abs/2510.09996)  
**Code**: [qulishen/BurstDeflicker](https://github.com/qulishen/BurstDeflicker)  
**Area**: Object Detection  
**Keywords**: flicker removal, rolling shutter, Retinex theory, burst image restoration, benchmark dataset  

## TL;DR

This paper introduces BurstDeflicker, the first benchmark dataset for multi-frame flicker removal (MFFR) in dynamic scenes. It is constructed through three complementary strategies—Retinex-based synthesis, real-world static scene capture, and green-screen compositing—enabling large-scale training and evaluation that significantly improves the generalization of flicker removal models to real-world dynamic scenes.

---

## Background & Motivation

- **Ubiquity of flicker artifacts**: Short-exposure images captured under AC-powered light sources are highly susceptible to flicker artifacts, manifesting as non-uniform luminance distribution and visible dark bands. This is particularly pronounced in high-speed photography, HDR imaging, and slow-motion video.
- **Impact on downstream vision tasks**: Flicker degrades not only perceptual image quality but also the performance of high-level tasks such as object detection, tracking, and recognition, especially in scenarios requiring reliable illumination.
- **Side effects of hardware solutions**: Conventional hardware approaches mitigate flicker by increasing exposure time, but at the cost of introducing motion blur, making it impossible to simultaneously eliminate flicker and preserve sharpness.
- **Inherent limitations of single-frame methods**: Single-image flicker removal (SIFR) methods struggle to distinguish flicker regions from visually similar dark areas such as shadows, and the lack of pixel context in severely degraded regions leads to unreliable restoration.
- **Unrealistic prior synthesis methods**: Existing flicker synthesis methods, originally designed for geotagging, model flicker as the complete removal of flicker illumination, which is inconsistent with the true physical process and leads to poor model generalization.
- **Absence of dynamic scene datasets**: Dynamic scenes are non-repeatable, making it infeasible to capture aligned flicker/flicker-free image pairs. The lack of large-scale MFFR datasets severely hinders progress in this field.

---

## Method

### Overall Architecture

The BurstDeflicker dataset is constructed from three complementary perspectives: (1) large-scale synthetic data based on Retinex theory; (2) 4,000 paired images captured from real-world static scenes; and (3) 3,690 dynamic image pairs generated via green-screen compositing. Training follows a two-stage pipeline of synthetic pre-training followed by real-data fine-tuning.

### Key Design 1: Retinex-Based Flicker Synthesis

- **Function**: Synthesizes training images with diverse flicker patterns based on Retinex theory, supporting unlimited data generation.
- **Mechanism**: Scene illumination is decomposed into ambient light $L_a$ and flicker light $L_f$, with the flickered image represented as $I_{flicker} = R \odot (L_a + L_f)$. Unlike prior work that defines the clean target as $I_{clean} = R \odot L_a$ (i.e., completely removing flicker illumination), this paper argues the correct target is $I_{clean} = R \odot (L_a + \overline{L_f})$, which adjusts the instantaneous flicker light to its effective value rather than eliminating it entirely.
- **Design Motivation**: Prior methods treat flicker illumination as entirely harmful and remove it completely, which contradicts physical reality—a flickering lamp does provide illumination over a full cycle, and the correct approach is to retain its effective value. By varying the ambient-to-flicker ratio $k$, rectification mode (full-wave, half-wave, PWM), and initial phase $\varphi$, a rich diversity of flicker patterns can be synthesized.

### Key Design 2: Real-World Static Data Capture

- **Function**: Captures high-quality flicker/flicker-free paired sequences from 369 real-world scenes, forming the BurstDeflicker-S subset.
- **Mechanism**: A professional camera mounted on a tripod captures 10 consecutive frames at a shutter speed of 1/1000–1/2000s to record flicker artifacts, followed by a long-exposure reference image at 1/50 or 1/60s (matched to the power grid frequency) as the flicker-free ground truth, with ISO adjusted to maintain consistent exposure.
- **Design Motivation**: Synthetic data suffers from a domain gap; real data helps the model learn the spatiotemporal characteristics of genuine flicker artifacts, improving generalization to in-the-wild scenes. A remote shutter release and electronic shutter are used to eliminate mechanical vibration and ensure spatial alignment.

### Key Design 3: Green-Screen Composited Dynamic Data

- **Function**: Composites green-screen foreground subjects (people or objects in motion) onto real flickering backgrounds to generate paired flickering image sequences with motion, forming the BurstDeflicker-G subset.
- **Mechanism**: Foreground clips are selected from the VideoMatte240K dataset and composited onto real flickering backgrounds using alpha mattes. For clean ground truth, 10 identical foreground frames are overlaid onto the corresponding clean background to maintain consistency.
- **Design Motivation**: The non-repeatability of dynamic scenes makes direct capture of paired flickering data infeasible. Training without dynamic data causes models to misinterpret motion-induced pixel changes as flicker, producing ghosting artifacts. Green-screen compositing elegantly resolves this challenge.

### Key Design 4: Multi-Frame Training Pipeline

- **Function**: Designs a training strategy for multi-frame flicker removal that is compatible with various network architectures.
- **Mechanism**: At each iteration, 3 frames are randomly sampled (with intervals of 1–3 frames to simulate different frame rates), augmented, concatenated along the channel dimension, and fed into the network, which outputs the deflickered result for one of the frames. Resize rather than crop is used during training to preserve the row-directional periodicity of flicker.
- **Design Motivation**: Cropping destroys the periodic pattern of flicker along the row-scanning direction; resizing preserves this critical structural characteristic.

---

## Loss & Training

- A two-stage training strategy is adopted: pre-training on Retinex-synthesized data (providing a strong initialization), followed by fine-tuning on BurstDeflicker-S and BurstDeflicker-G
- Random camera jitter augmentation (rotation ±3°, translation ±5 pixels) is applied to simulate handheld capture
- Multiple backbones are evaluated, including Restormer, Burstormer, and HDRTransformer

---

## Key Experimental Results

### Table 1: Quantitative Comparison of Different Methods on Static and Dynamic Test Sets

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MUSIQ ↑ | PIQE ↓ | BRISQUE ↓ |
|------|--------|--------|---------|---------|--------|-----------|
| Retinexformer* (untrained) | 15.70 | 0.707 | 0.213 | 53.60 | 50.27 | 30.24 |
| Lin et al.* (original) | 20.36 | 0.838 | 0.134 | 55.23 | 43.88 | 25.12 |
| Lin et al. (retrained) | 26.41 | 0.875 | 0.102 | 58.13 | 35.71 | 22.10 |
| Retinexformer (retrained) | 27.21 | 0.885 | 0.081 | 58.25 | 35.94 | 21.65 |
| Burstormer | 29.44 | 0.910 | 0.056 | 58.53 | 37.01 | 20.45 |
| HDRTransformer | 30.03 | 0.914 | 0.054 | 59.07 | 37.29 | 21.59 |
| **Restormer** | **30.63** | **0.918** | **0.045** | **59.10** | **34.90** | **19.32** |

All methods achieve substantial performance gains after retraining on the proposed dataset, validating its effectiveness. Restormer achieves the best performance across all metrics.

### Table 2: Ablation Study on Dataset Composition

| Synthetic Data | BurstDeflicker-S | BurstDeflicker-G | PSNR ↑ | SSIM ↑ | MUSIQ ↑ | BRISQUE ↓ |
|:---:|:---:|:---:|--------|--------|---------|-----------|
| ✓ | | | 24.48 | 0.862 | 57.10 | 23.76 |
| ✓ | ✓ | | 30.48 | 0.915 | 58.01 | 21.52 |
| ✓ | | ✓ | 30.65 | 0.916 | 58.43 | 20.34 |
| ✓ | ✓ | ✓ | **30.63** | **0.918** | **59.10** | **19.32** |

The green-screen dynamic data (BurstDeflicker-G) contributes significantly to robustness in dynamic scenes, reducing BRISQUE from 21.52 to 19.32.

---

## Highlights & Insights

- **First multi-frame flicker removal dataset**: Fills the data gap in the MFFR field; three complementary data sources cover synthetic, real static, and dynamic scenes
- **Corrected physical modeling of flicker removal**: Identifies the physical implausibility of completely removing flicker illumination in prior methods and proposes a corrected formulation based on effective values, better reflecting physical reality
- **Green-screen compositing approach**: Elegantly resolves the challenge of obtaining paired data in dynamic scenes, effectively reducing model misinterpretation of motion-induced pixel changes
- **Comprehensive experimental validation**: Covers multiple baselines, metrics, and scenarios, including 50 real-world dynamic test sequences

---

## Limitations & Future Work

- Green-screen compositing requires manual selection of semantically compatible foreground clips, which is labor-intensive and difficult to scale
- Data collection is limited to specific camera models; rolling shutter characteristics may differ across devices
- The current work targets only row-scan flicker from rolling shutter cameras and does not address other exposure mechanisms such as global shutter
- The foreground motion patterns in green-screen compositing are constrained by VideoMatte240K and may not cover all real-world dynamic scenarios

---

## Related Work & Insights

- **Hardware methods**: Photodiode-based flicker detection with exposure adjustment, and linear comb filters for event cameras, but these are costly and difficult to deploy
- **Single-frame methods**: DeflickerCycleGAN (Lin et al.) employs a CycleGAN framework with Flicker Loss and Gradient Loss, but single-frame approaches lack temporal context
- **Synthetic data**: The sinusoidal intensity overlay method of Wong et al. was designed for geotagging and performs poorly when used for flicker removal training
- **Multi-frame image restoration**: Burstormer (burst restoration), HDRTransformer (HDR imaging), and Restormer (general restoration) can all be adapted for multi-frame input

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The Retinex-based synthesis corrects prior formulations; the green-screen dynamic data construction approach is novel and practical
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-angle ablations (dataset composition, frame count, method comparison) with real dynamic test sequences
- **Writing Quality**: ⭐⭐⭐⭐ — Problem definition is clear, physical modeling derivations are complete, and figures and tables are informative
- **Value**: ⭐⭐⭐⭐ — Fills the MFFR data gap; the dataset and baseline experiments lay a solid foundation for future research

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](overlaybench_a_benchmark_for_layout-to-image_generation_with_dense_overlaps.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[NeurIPS 2025\] Generalizable Insights for Graph Transformers in Theory and Practice](generalizable_insights_for_graph_transformers_in_theory_and_practice.md)

<!-- RELATED:END -->
