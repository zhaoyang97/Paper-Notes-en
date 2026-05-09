---
title: >-
  [Paper Note] Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels
description: >-
  [CVPR 2026][Remote Sensing][hyperspectral video] This paper presents Lumosaic, an active hyperspectral video system that synchronizes an array of 12 narrowband LEDs with a coded-exposure pixel (CEP) camera at microsecond precision. Within 158 sub-frames per video frame, the system jointly encodes spatial, temporal, and spectral information, achieving motion-robust hyperspectral video reconstruction at 30 fps, VGA resolution, and 31 spectral channels (400–700 nm), with PSNR exceeding passive snapshot systems by more than 10 dB.
tags:
  - CVPR 2026
  - Remote Sensing
  - hyperspectral video
  - coded-exposure pixel
  - active illumination
  - motion-robust
  - spectral reconstruction
date: 2026-05-08
content_hash: ad00bd6fdb3cbebb
---

# Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels

**Conference**: CVPR 2026
**arXiv**: [2602.22140](https://arxiv.org/abs/2602.22140)
**Code**: N/A
**Area**: Computational Imaging / Hyperspectral Video
**Keywords**: hyperspectral video, coded-exposure pixel, active illumination, motion-robust, spectral reconstruction

## TL;DR

This paper presents Lumosaic, an active hyperspectral video system that synchronizes an array of 12 narrowband LEDs with a coded-exposure pixel (CEP) camera at microsecond precision. Within 158 sub-frames per video frame, the system jointly encodes spatial, temporal, and spectral information, achieving motion-robust hyperspectral video reconstruction at 30 fps, VGA resolution, and 31 spectral channels (400–700 nm), with PSNR exceeding passive snapshot systems by more than 10 dB.

## Background & Motivation

**Background**: Hyperspectral imaging (HSI) captures multi-band reflectance and finds wide application in material classification, physiological monitoring, and spectral relighting. Traditional scanning-based HSI achieves high spectral fidelity but is slow. Snapshot HSI systems (CASSI, DOE, MSFA) enable single-frame acquisition but suffer from low light efficiency and severe motion artifacts. Active HSI exploits programmable light sources to encode spectra along temporal or spatial dimensions, improving photon utilization.

**Limitations of Prior Work**:

1. Passive snapshot systems distribute light across multiple spectral channels, incurring heavy photon loss and amplifying noise through ill-posed inversion.
2. Existing active systems (e.g., LED time-division multiplexing, structured light projection) apply fine control along only a single dimension, leading to inter-frame spectral misalignment in dynamic scenes.
3. Even when rolling shutters enable within-frame spectral multiplexing, fast motion still produces rolling shutter distortion.

**Key Challenge**: Hyperspectral video demands simultaneous spectral resolution, light efficiency, and temporal sampling density — requirements that neither passive nor active prior systems can jointly satisfy.

**Goal**: To achieve compact, motion-robust, real-time hyperspectral video acquisition.

**Key Insight**: Coupling the per-pixel, high-speed modulation capability of CEP sensors with time-varying narrowband LED illumination to jointly encode spatial, temporal, and spectral information within a single frame.

**Core Idea**: By combining per-pixel exposure control from a CEP camera with time-varying LED illumination, the system constructs dense spatial–spectral–temporal mosaic codes within each frame, with all signal integration performed entirely on-chip.

## Method

### Overall Architecture

**Hardware**: 12 narrowband LEDs (20–30 nm FWHM, Lumileds Luxeon C) + VGA CEP camera (640×480, 12,500 sub-frames/sec) + ESP32 microcontroller for microsecond-level synchronization. Each frame consists of $S=158$ sub-frames (170 µs/sub-frame), with ~27 ms total integration and ~6 ms readout/synchronization. **Software pipeline**: spectral demosaicking → RIFE optical flow temporal alignment → HAN network reconstruction of 31-channel hyperspectral video.

### Key Designs

1. **Joint Illumination–Exposure Coding Scheme**

   - **Function**: Creates dense spatial–spectral–temporal codes within each frame.
   - **Mechanism**: Pixels are partitioned into $T=16$ tiles (4×4 mosaic), each with a unique exposure code $\mathbf{C}_{\text{tile}} \in \{0,1\}^{T \times S}$ and illumination code $\mathbf{I}_{\text{tile}} \in \{0,1\}^{T \times S \times L}$. Each sub-frame activates one LED, so neighboring pixels observe different spectral bands at different times. The forward model is: $Y_p = \sum_{s=1}^{S} C_{p,s} \cdot \mathbf{a}_{p,s}^\top \mathbf{r}_p + \eta_p$, where $\mathbf{a}_{p,s} = \mathcal{S} \odot \boldsymbol{\mathcal{I}}_{p,s}$ is the effective spectral sensing vector.
   - **Design Motivation**: Active illumination ensures the full narrowband output of each LED contributes to the effective signal without attenuation by filters; CEP per-pixel control provides dense spatial coding.

2. **CEP Per-Pixel Coded Exposure**

   - **Function**: Each pixel switches at high speed between two charge buckets according to a binary control code within a frame.
   - **Mechanism**: Each pixel contains a 1-bit writable memory that governs the active bucket for each sub-frame. At the frame level, the two buckets are read out independently, yielding complementary integrated signals. The modulation rate exceeds 39 kHz at VGA resolution.
   - **Design Motivation**: Breaks the constraint of conventional cameras where all pixels share the same exposure, making each pixel an independent spectral–temporal sampling unit.

3. **Temporal Alignment and Learning-Based Reconstruction**

   - **Function**: Compensates for sub-millisecond motion differences between LED sub-images, followed by neural network reconstruction of hyperspectral output.
   - **Mechanism**: The lime-LED sub-image is selected as the temporal reference (central wavelength + median exposure time). RIFE optical flow estimates inter-frame motion between sub-images of the same LED, and each sub-image is warped to the reference timestamp. The HAN network (18 residual blocks, 10 residual groups, 128 channels) takes 12-channel LED sub-images as input and produces 33 channels, of which the middle 31 channels (400–700 nm) are retained.
   - **Design Motivation**: Different LED sub-images correspond to different temporal intervals within a frame; direct fusion would introduce spectral–spatial aliasing. Sub-images from the same LED maintain photometric consistency, making them suitable for optical flow estimation.

### Loss & Training

$\mathcal{L}_1$ loss, Adam optimizer (lr=1e-4), batch size 14 with 2-step gradient accumulation, 50,000 iterations, ~24 h on an RTX A6000. Data augmentation includes 0–15% Gaussian noise injection. Training set: CAVE (32 scenes) + KAUST (409 scenes) + ARAD (949 scenes), resampled to 31 channels (400–700 nm, 10 nm interval), with an 80/10/10 train/val/test split.

## Key Experimental Results

### Main Results

**Simulated Reconstruction Quality (Noise-Free)**

| Method | Type | PSNR (dB)↑ | SSIM↑ | SAM↓ |
|---|---|---|---|---|
| MST++ | Passive RGB→HSI | ~30 | ~0.92 | ~0.25 |
| QDO | Passive DOE Snapshot | ~32 | ~0.93 | ~0.22 |
| Lumosaic + SRNet | Active CEP | ~42 | ~0.98 | ~0.06 |
| Lumosaic + MCAN | Active CEP | ~43 | ~0.98 | ~0.05 |
| **Lumosaic + HAN** | **Active CEP** | **~44.0** | **~0.99** | **~0.04** |

### Ablation Study

**Noise Robustness (Lumosaic + HAN)**

| Noise Level σ | PSNR (dB) | Notes |
|---|---|---|
| 0% | 44.0 | Best under noise-free conditions |
| 5% | ~38 | Mild noise; still far exceeds passive systems |
| 10% | ~35 | High fidelity maintained |
| 20% | 32.0 | Under heavy noise, still outperforms passive systems at 0% noise |

**Reconstruction Backbone Comparison**

| Backbone | PSNR↑ | Inference Speed | Notes |
|---|---|---|---|
| HAN | 44.0 dB | 4.7 s/frame | Highest accuracy |
| MCAN | Slightly lower | 52 ms/frame | Accuracy–speed trade-off |
| SRNet | Lowest | 27 ms/frame | Near real-time |

### Key Findings

- Lumosaic comprehensively outperforms passive snapshot systems (by 10+ dB PSNR), validating the fundamental advantage of active illumination combined with coded exposure.
- All three backbones surpass passive baselines; the performance gains are primarily attributable to the hardware coding scheme rather than network complexity.
- In ColorChecker experiments, reconstructed spectra closely match ground truth measured by a Konica Minolta CS-2000 spectroradiometer.
- Metameric disambiguation experiments demonstrate the system's ability to distinguish visually similar but spectrally distinct materials (genuine objects vs. printed replicas).
- Dynamic scene reconstructions at 30 fps (rotating globe, hand gestures, liquid diffusion, bubbles) are temporally coherent and spectrally accurate.

## Highlights & Insights

- This work pioneers the use of CEP sensors for hyperspectral video; all signal encoding is performed entirely on-chip, yielding a compact system that requires no complex optical calibration.
- The system co-design is elegant: illumination codes, exposure codes, and the reconstruction network are tightly coupled.
- The coding density of 158 sub-frames × 12 LEDs × 16 tiles achieves extremely high information capacity within a single frame.
- RIFE optical flow alignment addresses the inherent inter-sub-frame motion problem of active illumination systems and is the key step that enables hyperspectral "video" reconstruction.

## Limitations & Future Work

- Reconstruction inference is slow (HAN: 4.7 s/frame vs. 30 fps acquisition); real-time deployment requires a lightweight backbone (SRNet at 27 ms is feasible but at reduced accuracy).
- Only one CEP bucket (Bucket 1) is utilized; joint modeling of both buckets could further improve dynamic range and light efficiency.
- Active illumination restricts applicable scenarios (requires a controllable light source); outdoor or long-range scenes are not supported.
- Frames are processed independently, leaving inter-frame temporal redundancy unexploited, partly due to the scarcity of hyperspectral video training data.
- The coding pattern is fixed; adaptive or randomized mosaics could potentially yield further improvements.

## Related Work & Insights

- **vs. CASSI and other passive systems**: Active illumination fundamentally changes photon utilization efficiency — all LED output contributes to the effective signal, whereas passive filtering discards the majority of photons.
- **vs. Verma et al.**: Both exploit time-varying LED illumination, but Verma et al. rely on rolling shutter row-level multiplexing, which still introduces distortion under fast motion; Lumosaic's per-pixel coding offers greater flexibility.
- **vs. Yu et al. (event camera)**: That approach uses an event camera with rainbow illumination but depends on mechanically rotating optics, limiting compactness and robustness.
- **Inspiration**: The CEP + time-varying illumination paradigm is extensible to fluorescence imaging, Raman spectroscopy, and other domains requiring active excitation combined with spectral resolution.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The hyperspectral video system combining CEP sensors with active illumination is unprecedented; a system-level innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers simulation, real prototype, static/dynamic scenes, and metameric disambiguation; lacks quantitative real-scene comparisons against more recent systems.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The forward model is developed progressively from pixel to system level; hardware–software co-design logic is clearly articulated.
- **Value**: ⭐⭐⭐⭐ — Exceptionally high system-level innovation, though active illumination narrows the scope of applicable scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[CVPR 2026\] No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)
- [\[CVPR 2026\] MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)
- [\[ICCV 2025\] GeoExplorer: Active Geo-Localization with Curiosity-Driven Exploration](../../ICCV2025/remote_sensing/geoexplorer_active_geo-localization_with_curiosity-driven_exploration.md)
- [\[AAAI 2026\] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](../../AAAI2026/remote_sensing/perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)

</div>

<!-- RELATED:END -->
