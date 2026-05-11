---
title: >-
  [Paper Note] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision
description: >-
  [CVPR 2026][3D Vision][Egocentric Vision] This paper presents Ego-1K, a large-scale temporally synchronized egocentric multiview video dataset comprising 956 short clips (12+4 cameras, 60Hz)…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Egocentric Vision"
  - "Multiview Dataset"
  - "Dynamic Scene Reconstruction"
  - "Novel View Synthesis"
  - "Hand-Object Interaction"
date: 2026-05-08
content_hash: f0979773b182d9e1
---

# Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision

**Conference**: CVPR 2026
**arXiv**: [2603.13741](https://arxiv.org/abs/2603.13741)
**Code**: [Dataset](https://huggingface.co/datasets/facebook/ego-1k)
**Area**: 3D Vision
**Keywords**: Egocentric Vision, Multiview Dataset, Dynamic Scene Reconstruction, Novel View Synthesis, Hand-Object Interaction

## TL;DR

This paper presents Ego-1K, a large-scale temporally synchronized egocentric multiview video dataset comprising 956 short clips (12+4 cameras, 60Hz), addressing the data gap in egocentric dynamic 3D reconstruction, and demonstrates that stereo depth guidance can substantially improve 4D novel view synthesis quality.

## Background & Motivation

Mixed reality devices and egocentric world modeling require realistic 4D reconstruction from the wearer's perspective. However, existing datasets exhibit critical gaps:

- **NVS datasets** (e.g., Neural 3D Video, DiVA360): provide multiview coverage but from exocentric perspectives, lacking egocentric viewpoints
- **Egocentric datasets** (e.g., Ego4D, EPIC-KITCHENS): large-scale but predominantly monocular/binocular, focused on activity recognition rather than 3D reconstruction
- **Multiview egocentric datasets** (e.g., EgoExo4D, HOT3D): only 2–3 egocentric cameras, insufficient in viewpoint count

Core requirement: a dynamic scene dataset that simultaneously satisfies **large scale, high camera count, egocentric perspective, and precise synchronization**. Unique challenges include large disparities from close-range hand motion, rapid image motion, and frequent occlusions.

## Method

### Overall Architecture

Ego-1K is a dataset and benchmark paper rather than an algorithmic contribution. Core contributions include: (1) design and construction of a multi-camera head-mounted capture system; (2) a stereo consistency evaluation protocol; (3) a 4D NVS evaluation protocol; and (4) a stereo depth-guided 3DGS baseline.

### Key Designs

1. **Multi-camera capture system**: A custom head-mounted device integrates a Quest 3 headset (4 forward-facing cameras) and 12 external fisheye cameras (8MP global shutter, 190° FOV, f/2.8). All 16 cameras are hardware-synchronized at 60Hz via a wireless synchronizer. The 12 external cameras stream via USB 3.1 to a backpack computer (dual 8-port USB adapters) in 8-bit raw Bayer format. The system also includes 2 iToF sensors (30Hz alternating) and an IMU (800Hz), with a total raw data throughput of approximately 15 GB/s. **Design Motivation**: existing head-mounted devices support at most 2–3 cameras, which is insufficient for dense 3D reconstruction.

2. **Calibration system (offline + online)**:
   - **Offline calibration**: conducted in a laboratory environment using 5 large Calibu calibration boards to solve intrinsic and extrinsic parameters for all cameras
   - **Online calibration**: compensates for 0.1–0.2° rotational drift caused by lens motion and focal length variation induced by temperature changes (corresponding to 1–3 pixel shifts), optimizing camera orientation and focal length while keeping other parameters fixed
   - Online calibration reduces the median MAD score by 35%

3. **Research-release dataset**: The 12 fisheye cameras are undistorted into 6 rectified stereo pairs (1280×1280, 130° horizontal FOV) for ease of processing. Quest 3 RGB cameras are excluded due to rolling shutter, and differing resolution and color configurations relative to the external cameras. Each recording clip is approximately 19 GB; the full dataset totals 17.5 TB.

4. **Stereo depth-guided 3DGS baseline**: A core finding is that existing NVS methods are severely insufficient on this dataset, while stereo foundation models provide reasonably accurate depth estimates. The proposed baseline:
   - Runs Foundation Stereo bidirectionally (L→R and R→L) to obtain depth maps
   - Fuses all stereo depth maps via TSDF to obtain a watertight surface
   - Samples points (with normals and colors) from the fused surface to initialize 3D Gaussians
   - Fine-tunes with a small number of iterations, minimizing the photometric loss $\mathcal{L}=(1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_{\text{D-SSIM}}$ ($\lambda=0.1$)
   - Optimizes each frame independently to form a dense 4D reconstruction

### Loss & Training

- Evaluation does not involve model training; only 3DGS is fine-tuned
- Train/test split: 10 training viewpoints + 2 test viewpoints (target stereo pair 3–4)
- Experimental subset: 10% of the dataset (96 clips)

## Key Experimental Results

### Main Results

4D NVS reconstruction evaluation (target pair 3–4 as test views; remaining 10 views used for training):

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| 3DGS (per-frame) | 21.22 | 0.709 | 0.260 |
| K-Planes | 16.46 | 0.597 | 0.443 |
| Spacetime Gaussians | 24.76 | 0.780 | 0.270 |
| **3DGS + Stereo Guidance** | **29.12** | **0.830** | **0.115** |

Stereo guidance improves PSNR by 7.9 dB over vanilla 3DGS and by 4.4 dB over Spacetime Gaussians.

### Ablation Study

Stereo method consistency evaluation (warping disparity maps from 5 stereo pairs to the target pair and computing consistency):

| Stereo Method | MAD ↓ (mm) | MAD<1mm ↑ | SD ↓ (mm) |
|---------------|-----------|-----------|----------|
| **Foundation Stereo** | **1.6** | **74.0%** | 42.5 |
| Selective-Stereo | 8.0 | 0.0% | 46.2 |
| BiDAStereo | 2.2 | 3.1% | **8.3** |
| StereoAnywhere | 1.7 | 29.5% | 10.4 |

Foundation Stereo achieves the best overall consistency (lowest MAD), while BiDAStereo produces the fewest extreme outliers (lowest SD).

### Key Findings

- Existing NVS methods (3DGS, K-Planes) are severely insufficient for egocentric dynamic scenes; K-Planes achieves only 16.46 dB
- Dynamic models (K-Planes, Spacetime Gaussians) are designed for object-centric or fixed-pose multiview videos and cannot effectively handle the combination of ego-motion, close-range hand motion, and large disparities
- Performance gaps are larger for close-range dynamic objects (hands) and smaller for distant objects (bystanders)
- Online calibration is critical for stereo estimation accuracy, reducing MAD by 35%

## Highlights & Insights

- Addresses a well-defined data gap: the first dataset to simultaneously satisfy large scale, high camera count, egocentric perspective, and precise synchronization for dynamic scenes
- The proposed stereo consistency evaluation protocol (requiring no ground-truth depth) is practically valuable and transferable to other multiview systems
- The core finding is insightful: per-frame initialization outperforms end-to-end dynamic models, indicating that the critical bottleneck lies in geometric initialization rather than temporal modeling
- Dataset design details merit attention: online calibration, global shutter selection, and fisheye undistortion parameter choices are all supported by thorough engineering rationale

## Limitations & Future Work

- The 4 Quest 3 cameras are excluded from the research-release dataset due to rolling shutter differences, leaving room for improved utilization
- iToF data is unused due to motion artifacts and phase ambiguity; future work could explore multimodal fusion
- The current baseline is per-frame 3DGS and lacks temporal consistency modeling; spatiotemporal regularization or scene flow priors could be explored
- The dataset focuses on hand-object interaction; scene diversity could be further expanded (e.g., outdoor settings, multi-person collaboration)
- The raw dataset is 88 TB, posing significant storage and bandwidth requirements
- The absence of semantic annotations (hand keypoints, object categories, etc.) limits evaluation on downstream tasks

## Related Work & Insights

- **Ego4D / EgoExo4D**: large-scale egocentric datasets with few cameras, focused on activity recognition
- **Neural 3D Video / DiVA360**: multiview NVS datasets but from exocentric perspectives
- **Foundation Stereo**: the best-performing stereo foundation model, highly effective as a geometric prior
- **3DGS**: the backbone for novel view synthesis; substantially improved by stereo-guided initialization
- Insight: with the proliferation of smart glasses, egocentric multiview reconstruction is an important research direction; geometric priors are more reliable than purely learned approaches

## Rating

- Novelty: ⭐⭐⭐ Primarily a dataset contribution; the stereo guidance idea on the method side is intuitive but well-validated
- Experimental Thoroughness: ⭐⭐⭐⭐ Stereo evaluation + NVS evaluation + multi-baseline comparison with rigorously designed evaluation protocols
- Writing Quality: ⭐⭐⭐⭐ Detailed dataset description, comprehensive tabular comparisons, and thorough engineering details
- Value: ⭐⭐⭐⭐ Fills a clear data gap and will advance egocentric 3D/4D reconstruction research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2026\] Indoor Asset Detection in Large Scale 360° Drone-Captured Imagery via 3D Gaussian Splatting](indoor_asset_detection_in_large_scale_360_drone-captured_imagery_via_3d_gaussian.md)
- [\[ICLR 2026\] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](../../ICLR2026/3d_vision/egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](hyperbolic_multiview_pretraining_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
