---
title: >-
  [Paper Note] SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras
description: >-
  [CVPR2026][Object Detection][RGB-to-RAW] This paper proposes SpiralDiff, a diffusion framework for RGB-to-RAW conversion that employs a signal-dependent noise weighting strategy to accommodate varying reconstruction difficulty across pixel intensity regions, and introduces a CamLoRA module for lightweight cross-camera adaptation within a single unified model.
tags:
  - CVPR2026
  - Object Detection
  - RGB-to-RAW
  - diffusion model
  - signal-dependent noise
  - LoRA
  - cross-camera adaptation
date: 2026-05-08
content_hash: edf28b77d6d415e3
---

# SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras

**Conference**: CVPR2026  
**arXiv**: [2603.14885](https://arxiv.org/abs/2603.14885)  
**Code**: [Chuancy-TJU/SpiralDiff](https://github.com/Chuancy-TJU/SpiralDiff)  
**Area**: Object Detection / Image Signal Processing  
**Keywords**: RGB-to-RAW, diffusion model, signal-dependent noise, LoRA, cross-camera adaptation, object detection

## TL;DR

This paper proposes SpiralDiff, a diffusion framework for RGB-to-RAW conversion that employs a signal-dependent noise weighting strategy to accommodate varying reconstruction difficulty across pixel intensity regions, and introduces a CamLoRA module for lightweight cross-camera adaptation within a single unified model.

## Background & Motivation

1. **RAW images carry richer information**: RAW preserves linear radiometric response and high dynamic range, making it more suitable for downstream tasks such as denoising, low-light enhancement, and object detection; however, RAW datasets are far smaller and less diverse than RGB datasets.
2. **Demand for RGB-to-RAW conversion**: Synthesizing RAW from abundant RGB data avoids costly sensor-level data collection, yet existing methods perform poorly in high-intensity regions (overexposure / nonlinear tone mapping).
3. **Reconstruction difficulty varies with pixel intensity**: In low-intensity regions, the RGB–RAW residual is small and stable, enabling high-fidelity recovery; in high-intensity or overexposed regions, residuals are large and highly variable due to multiplicative digital gain and value clipping, making a uniform strategy inadequate.
4. **Large ISP discrepancies across cameras**: Different camera ISP pipelines (demosaicing, white balance, tone mapping, etc.) vary significantly; naïve mixed training leads to performance degradation.
5. **Limitations of metadata-based methods**: Methods that rely on ISP parameters or sampled RAW pixels are typically inapplicable in real-world scenarios where metadata is unavailable.
6. **Shortcomings of existing metadata-free methods**: CycleISP, InvISP, and ReRAW adopt globally uniform reconstruction strategies without adapting to signal-dependent characteristics, and lack cross-camera adaptation capability.

## Method

### Overall Architecture

SpiralDiff is built upon ResShift, an efficient residual-shift diffusion framework, and requires only **4 sampling steps** for inference. Given an RGB image and a camera label, a U-Net (with Swin Transformer layers) iteratively denoises a noisy RAW estimate to produce the target RAW output. The model operates directly in **pixel space** (the VQGAN latent compression used in ResShift is removed, as it was trained on RGB and is unsuitable for RAW).

### Spiral Diffusion — Signal-Dependent Noise Weighting

- **Core observation**: RGB pixel intensity is positively correlated with the RGB–RAW residual; high-intensity regions exhibit large residuals with high uncertainty, while low-intensity regions exhibit small, stable residuals.
- **Time-varying weight map** $\mathbf{w}_t = \mathbf{x}_0 + \eta_t \mathbf{e}_0$: evolves with diffusion step $t$, approaching RGB $\mathbf{y}_0$ at $t=T$ and RAW $\mathbf{x}_0$ at $t=0$, enabling a smooth transition from RGB to RAW.
- **Forward process**: Building on the isotropic Gaussian noise of ResShift, the noise variance is modulated pixel-wise by $\mathbf{w}_t^2$ — low noise in dark regions to preserve fidelity, high noise in bright regions to grant the model greater generative freedom.
- **Reverse process**: The mean $\boldsymbol{\mu}_{t-1}$ remains a convex combination of the denoised and clean terms, but the mixing coefficient $\boldsymbol{\gamma}_t$ depends on the spatial weight map, enabling **pixel-adaptive fusion**. Setting $\mathbf{w}_t \equiv 1$ recovers standard ResShift.

### CamLoRA — Camera-Aware Low-Rank Adaptation

- Camera-specific low-rank updates $\Delta \mathbf{W}_i = \mathbf{B}_i \mathbf{A}_i$ (rank $r=8$) are appended to the $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$ matrices of the Swin Transformer layers in the U-Net.
- During training, the shared backbone weights are updated with all data, while only the LoRA branch corresponding to the current camera label participates in gradient computation.
- Additional parameters account for only **2.7%** (1.05M), with one set of LoRA adapters per camera across four cameras.
- **Few-shot extension**: After pretraining the unified model, only one LoRA branch needs to be fine-tuned for a new camera; 1-shot adaptation achieves 42.85 dB PSNR, compared to only 39.83 dB when training from scratch.

### Loss & Training

The training objective follows ResShift: the network $f_\theta(\mathbf{x}_t, \mathbf{y}_0, t)$ is trained to predict $\mathbf{x}_0$, optimized jointly with the diffusion loss.

## Key Experimental Results

### Main Results — Quantitative Comparison on Four Datasets

| Method | FiveK Canon | FiveK Nikon | NOD Nikon | NOD Sony |
|--------|------------|------------|-----------|----------|
| CycleISP | 37.93 / 0.9913 | 40.18 / 0.9920 | 50.11 / 0.9985 | 46.57 / 0.9975 |
| InvISP | 36.81 / 0.9814 | 34.30 / 0.9163 | 48.29 / 0.9954 | 44.76 / 0.9922 |
| RAW-Diffusion | 39.96 / 0.9890 | 39.68 / 0.9866 | 50.52 / 0.9954 | 47.31 / 0.9908 |
| **SpiralDiff** | **42.82 / 0.9936** | **41.72 / 0.9925** | **53.64 / 0.9990** | **50.46 / 0.9980** |
| +CamLoRA (merged) | 42.46 / 0.9934 | 43.82 / 0.9950 | 52.62 / 0.9988 | 50.08 / 0.9977 |

> SpiralDiff comprehensively surpasses the state of the art under the independent training setting, improving PSNR over RAW-Diffusion by **+2.86 dB** (FiveK Canon) and **+3.12 dB** (NOD Nikon).

### Overexposure Test Set Comparison

| Method | FiveK Canon PSNR | NOD Nikon PSNR |
|--------|-----------------|----------------|
| RAW-Diffusion | 30.60 | 40.05 |
| **SpiralDiff** | **31.10** | **40.79** |

### Ablation Study

| Noise Weighting Strategy | FiveK Canon PSNR | NOD Nikon PSNR |
|--------------------------|-----------------|----------------|
| Baseline (uniform noise) | 41.40 | 53.48 |
| Static $\mathbf{y}_0$ weighting | 40.06 | 53.42 |
| **Time-varying $\mathbf{w}_t$ weighting** | **42.82** | **53.64** |

- Static RGB weighting even underperforms the baseline, validating the necessity of the time-varying weight map.
- CamLoRA vs. direct camera embedding: the embedding approach falls below the unconditional baseline, whereas CamLoRA yields an effective gain of approximately +0.5 dB.
- Plug-in experiment: replacing DDPM in RAW-Diffusion with SpiralDiff improves PSNR by +1.57 dB (FiveK Canon).

### Downstream Object Detection

| Training Data | NOD Nikon AP | NOD Sony AP |
|--------------|-------------|-------------|
| RGB-only | 19.1 | 19.7 |
| RAW-only (100 images) | 18.4 | 17.6 |
| RAW + BDD-RAW (synthetic) | **26.7** | **29.0** |

> Synthesized RAW data significantly improves object detection performance in low-data regimes (AP gain: +8.3 / +11.4).

## Highlights & Insights

1. The **signal-dependent noise scheduling** concept is novel and physically intuitive — less noise in dark regions preserves detail, while more noise in bright regions increases generative flexibility, aligning closely with sensor physics.
2. **CamLoRA** achieves a unified cross-camera model with only 2.7% additional parameters and supports few-shot rapid adaptation to new cameras.
3. **4-step sampling** offers high inference efficiency and strong practical applicability.
4. Experiments are comprehensive: 4 datasets + overexposure testing + real ISP + downstream detection + thorough ablations + plug-in experiments.
5. The plug-in replacement on RAW-Diffusion yields substantial gains, demonstrating the generalizability of SpiralDiff.

## Limitations & Future Work

1. The weight map $\mathbf{w}_t$ is defined using ground-truth $\mathbf{x}_0$; at inference time a network prediction must substitute, potentially introducing cumulative error.
2. CamLoRA is validated on only 4 cameras; scalability to larger camera pools (tens of sensor types) remains unexplored.
3. Pixel-space diffusion incurs high computational cost at high resolution; latent-space acceleration is not investigated.
4. The downstream detection experiments use a simple setup of 100 real RAW images with synthetic augmentation; effectiveness at larger scale requires further validation.
5. Noise weighting considers only pixel intensity, without incorporating structural information such as spatial texture complexity.

## Related Work & Insights

| Dimension | CycleISP / InvISP | RAW-Diffusion | SpiralDiff |
|-----------|-------------------|---------------|------------|
| Noise type | No diffusion | DDPM isotropic | Signal-dependent time-varying weighting |
| Cross-camera support | None | None | CamLoRA |
| Sampling steps | — | ~1000 | 4 |
| Overexposure handling | Poor | Moderate | Good (adaptive noise) |
| Few-shot | Not supported | Not supported | Supported (LoRA fine-tuning) |

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel combination of signal-dependent noise, time-varying weight maps, and CamLoRA
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets + overexposure / real ISP / downstream detection / ablations / plug-in experiments
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete mathematical derivations, intuitive illustrations
- Value: ⭐⭐⭐⭐ — Practical significance for RAW-domain data augmentation and cross-camera adaptation

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Evolutionary Negative Module Pruning for Better LoRA Merging](../../ACL2026/object_detection/evolutionary_negative_module_pruning_for_better_lora_merging.md)
- [\[AAAI 2026\] SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements](../../AAAI2026/object_detection/simrod_a_simple_baseline_for_raw_object_detection_with_global_and_local_enhancem.md)
- [\[CVPR 2026\] ABRA: Teleporting Fine-Tuned Knowledge Across Domains for Open-Vocabulary Object Detection](abra_teleporting_finetuned_knowledge_across_domain.md)
- [\[ICCV 2025\] Diffusion Curriculum: Synthetic-to-Real Data Curriculum via Image-Guided Diffusion](../../ICCV2025/object_detection/diffusion_curriculum_synthetic-to-real_data_curriculum_via_image-guided_diffusio.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)

<!-- RELATED:END -->
