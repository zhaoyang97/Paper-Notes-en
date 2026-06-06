---
title: >-
  [Paper Note] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting
description: >-
  [NeurIPS 2025][3D Vision][Event camera] EF-3DGS is the first work to introduce event cameras into free-trajectory scene reconstruction. It employs an Event Generation Model (EGM) to reconstruct latent inter-frame images…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Event camera"
  - "3D Gaussian Splatting"
  - "free-trajectory"
  - "pose estimation"
  - "novel view synthesis"
date: 2026-05-08
content_hash: ea4dba27297bf7a2
---

# EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting

**Conference**: NeurIPS 2025
**arXiv**: [2410.15392](https://arxiv.org/abs/2410.15392)  
**Code**: To be confirmed  
**Area**: 3D Vision
**Keywords**: Event camera, 3D Gaussian Splatting, free-trajectory, pose estimation, novel view synthesis

## TL;DR

EF-3DGS is the first work to introduce event cameras into free-trajectory scene reconstruction. It employs an Event Generation Model (EGM) to reconstruct latent inter-frame images for continuous supervision, Contrast Maximization (CMax) combined with a Linear Event Generation Model (LEGM) to extract motion information for pose calibration, and a photometric BA + Fixed-GS strategy to resolve color inconsistency. The method achieves a 3 dB PSNR improvement and a 40% reduction in ATE in high-speed scenarios.

## Background & Motivation

- 3DGS optimizes scene representations from posed image collections, achieving remarkable progress in novel view synthesis.
- Free-trajectory video reconstruction faces two major challenges: (1) inaccurate camera poses, and (2) insufficient inter-frame overlap in high-speed scenarios leading to under-constrained optimization.
- Existing pose-free methods (LocalRF, CF-3DGS) suffer severe performance degradation in high-speed or low-frame-rate settings.
- Event cameras offer high temporal resolution and low latency, providing rich brightness and motion information in the inter-frame blind zone.
- Seamlessly integrating event data into 3DGS is technically challenging: events are differential signals, whereas 3DGS renders absolute luminance.

## Core Problem

How to exploit the high temporal resolution of event cameras to jointly optimize camera poses and 3DGS scene reconstruction quality in high-speed free-trajectory scenarios?

## Method

### 1. EGM-Driven Optimization

The inter-frame time interval is divided uniformly into $N$ sub-intervals, over which event frames are accumulated. The latent brightness image at intermediate timestamps is reconstructed from the nearest RGB frame and the accumulated events:

$$I_{i,j} = I_{i,0} \cdot \exp\left(\sum_{n=0}^{j-1} E_{i,n} \cdot C\right)$$

This serves as a supervision signal, extending 3DGS optimization from discrete frames to the continuous event stream:

$$\mathcal{L}_{EGM} = (1-\lambda)\mathcal{L}_1(\hat{I}_t, I_t) + \lambda\mathcal{L}_{D\text{-}SSIM}(\hat{I}_t, I_t)$$

### 2. CMax + LEGM Joint Optimization

The CMax framework estimates the motion field by exploiting the spatiotemporal correlation of events. Event frames from the first $r$ sub-intervals are warped back to the reference frame via optical flow, which is derived from the 3DGS-rendered depth and relative poses.

The contrast maximization loss maximizes the variance of the Image of Warped Events (IPWE):

$$\mathcal{L}_{cm} = -\text{Var}(\text{IPWE}_{i,j})$$

The LEGM gradient loss connects the IPWE to rendered luminance changes via the linear event model:

$$\mathcal{L}_{grad} = \|C \cdot \text{IPWE}_{i,j} - (\hat{L}(\mathbf{u}) - \hat{L}(\mathbf{u} + F_{i,j \to j+1}))\|^2$$

$$\mathcal{L}_{LEGM} = \lambda_{cm}\mathcal{L}_{cm} + \lambda_{grad}\mathcal{L}_{grad}$$

### 3. Photometric BA + Fixed-GS Strategy

Photometric BA (PBA) establishes photometric reprojection errors at randomly sampled timestamps, projecting rendered pixels onto the nearest RGB frame to compute consistency.

The Fixed-GS two-stage training strategy:
- **Stage 1**: Full-parameter optimization (position, opacity, rotation, scale, SH) using combined event and frame losses.
- **Stage 2**: Only SH coefficients (color) are optimized; all other parameters are fixed, trained exclusively on RGB frames.

The two stages follow a 4:1 ratio, effectively resolving color distortion caused by the absence of color information in the event stream.

### Total Loss

$$\mathcal{L}_{event} = \mathcal{L}_{EGM} + \mathcal{L}_{LEGM} + \lambda_{PBA}\mathcal{L}_{PBA}$$

## Key Experimental Results

### Tanks and Temples Benchmark (Various Frame Rates)

| Method | Pose-Free | 6FPS PSNR↑ | 2FPS PSNR↑ | 1FPS PSNR↑ |
|--------|-----------|------------|------------|------------|
| CF-3DGS | Yes | 26.05 | 22.08 | 20.53 |
| Event-3DGS (E+F) | No | 26.32 | 23.44 | 22.41 |
| EvCF-3DGS | Yes | 26.07 | 22.81 | 21.73 |
| **EF-3DGS** | **Yes** | **26.66** | **24.43** | **23.96** |

At the extreme high-speed setting of 1FPS, EF-3DGS outperforms CF-3DGS by **3.43 dB**.

### Pose Estimation

EF-3DGS achieves the lowest ATE across all frame rates, with a reduction of approximately 40% in high-speed scenarios. The method also demonstrates significant superiority on the newly collected RealEv-DAVIS real-world event dataset.

## Highlights & Insights

- The first work to introduce event cameras into free-trajectory 3DGS scene reconstruction.
- Three complementary loss functions (EGM / CMax+LEGM / PBA) are rigorously derived from the event camera imaging model.
- The Fixed-GS two-stage training strategy elegantly decouples geometry and color optimization, addressing the fundamental issue that event streams carry no color information.
- PSNR improvement exceeds 3 dB in high-speed scenarios (1FPS), demonstrating significant practical value.

## Limitations & Future Work

- Requires hardware with synchronized event and frame output (DAVIS camera), resulting in relatively high deployment costs.
- The event noise model is simplified (fixed contrast threshold $C$), whereas real-world noise is considerably more complex.
- The progressive scene expansion is inherited from LocalRF/CF-3DGS; efficiency on very long sequences remains to be verified.
- The ability to handle dynamic scenes is not explicitly discussed.

## Related Work & Insights

- vs **CF-3DGS**: A frame-only method that degrades severely in high-speed scenarios; EF-3DGS supplements inter-frame information via the event stream.
- vs **Event-3DGS**: Requires known poses; EF-3DGS is pose-free.
- vs **E-NeRF/EventNeRF**: NeRF-based event methods that require known poses and are limited to small-scale scenes.
- vs **EvCF-3DGS**: Naively incorporates event loss into CF-3DGS, lacking the CMax motion constraint and Fixed-GS strategy.

EF-3DGS's application of the CMax framework within 3DGS can be extended to other tasks requiring sub-frame-level motion estimation. The Fixed-GS two-stage training strategy generalizes to other multimodal scene modeling settings (e.g., thermal infrared + visible light). Event cameras hold a natural advantage in high-speed applications such as VR/AR, FPV drones, and autonomous driving.

## Rating

- ⭐ **Novelty**: 4.5/5 — First to introduce event-aided free-trajectory 3DGS; three loss functions are elegantly designed.
- ⭐ **Experimental Thoroughness**: 4/5 — Evaluated on public benchmarks and a newly collected real-world dataset with comprehensive multi-frame-rate comparisons.
- ⭐ **Writing Quality**: 4/5 — The methodology is derived systematically from event camera imaging principles, with clear logical structure.
- ⭐ **Value**: 4.5/5 — Provides an effective solution to a key pain point in high-speed scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](../../ICCV2025/3d_vision/evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](../../CVPR2026/3d_vision/e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[NeurIPS 2025\] ZPressor: Bottleneck-Aware Compression for Scalable Feed-Forward 3DGS](zpressor_bottleneck-aware_compression_for_scalable_feed-forward_3dgs.md)
- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](../../ICCV2025/3d_vision/robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
