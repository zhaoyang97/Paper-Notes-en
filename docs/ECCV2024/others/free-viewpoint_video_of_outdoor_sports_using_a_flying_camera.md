---
title: >-
  [Paper Note] Free-Viewpoint Video of Outdoor Sports Using a Flying Camera
description: >-
  [ECCV 2024][Free-viewpoint video] Proposes a system based on a UAV-mounted RGB camera capable of reconstructing 4D dynamic humans and 3D unbounded backgrounds in outdoor sports scenes, enabling free-viewpoint video rendering at any timestamp.
tags:
  - "ECCV 2024"
  - "Free-viewpoint video"
  - "UAV"
  - "outdoor sports"
  - "4D reconstruction"
  - "Neural Radiance Fields"
date: 2026-05-08
content_hash: 6df681264a7579b0
---

# Free-Viewpoint Video of Outdoor Sports Using a Flying Camera

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Free-viewpoint video, UAV, outdoor sports, 4D reconstruction, Neural Radiance Fields

## TL;DR

Proposes a system based on a UAV-mounted RGB camera capable of reconstructing 4D dynamic humans and 3D unbounded backgrounds in outdoor sports scenes, enabling free-viewpoint video rendering at any timestamp.

## Background & Motivation

1. **Background**: Free-Viewpoint Video (FVV) technology has received widespread attention in recent years in fields such as live sports broadcasting, sports analysis, and virtual reality. Existing methods mainly rely on dense camera arrays (e.g., multi-camera broadcasting systems) or handheld mobile cameras to acquire multi-view data for free-viewpoint rendering. However, outdoor sports scenes are characterized by large-scale human motion and large-scale scene structures, making this task extremely challenging.

2. **Limitations of Prior Work**: 
    - **Dense camera array schemes**: Require a large number of camera resources, are expensive, and are complex to deploy, making them difficult to apply to open outdoor sports scenes.
    - **Handheld camera schemes**: A single handheld camera struggles to track fast-moving athletes while covering a sufficient range of the scene, performing poorly in real sports scenarios.
    - **Joint dynamic-static reconstruction**: Outdoor sports involve both dynamic athletes and large-scale static backgrounds (360° environments). Existing methods struggle to reconstruct both with high quality simultaneously.

3. **Key Challenge**: How to simultaneously reconstruct high-quality 4D dynamic humans and 3D large-scale outdoor background scenes using a single low-cost RGB camera (rather than expensive multi-camera setups), and achieve spatiotemporally free-viewpoint rendering.

4. **Goal**: Design a complete UAV-based free-viewpoint video system that uses a single RGB camera mounted on a UAV to achieve 4D human reconstruction and 3D unbounded scene reconstruction in outdoor sports scenarios.

5. **Key Insight**: Leverage a UAV (drone/flying camera) as the capture platform—the UAV can flexibly fly around the athlete to capture image sequences from multiple angles, thus compensating for the limited viewpoint of a single static camera. Additionally, system-level calibration and motion capture sub-modules are proposed to enhance overall robustness and efficiency.

6. **Core Idea**: Use a single UAV equipped with an RGB camera to fly and shoot around the athlete, combined with specially designed calibration, motion capture, and neural rendering sub-modules, to achieve low-cost free-viewpoint video generation for outdoor sports scenes.

## Method

### Overall Architecture

The system pipeline consists of the following main stages:
1. **Data Acquisition**: The UAV flies around the athlete to capture RGB video sequences.
2. **Camera Calibration**: Accurately estimate camera intrinsic and extrinsic parameters for each frame from the UAV video.
3. **Human Motion Capture**: Recover the 3D human pose and motion of the athlete from the monocular video.
4. **4D Dynamic Human Reconstruction**: Reconstruct a deformable 4D human model based on the human motion sequence.
5. **3D Background Reconstruction**: Reconstruct a 360° unbounded outdoor scene using multi-view images captured by the UAV.
6. **Free-Viewpoint Rendering**: Combine the reconstructed dynamic human and static background to render images at any viewpoint and any timestamp.

### Key Designs

1. **Drone Calibration**: 
    - **Function**: Estimates precise camera parameters from the UAV flight video.
    - **Mechanism**: Combines UAV IMU data with visual SfM (Structure-from-Motion) technology, utilizing UAV trajectory constraints and ground feature points to achieve precise calibration.
    - **Design Motivation**: Outdoor scenes lack structured calibration targets, and UAVs experience jitter and non-uniform motion during flight, necessitating a robust calibration scheme.

2. **Human Motion Capture**: 
    - **Function**: Recovers the athlete's 3D skeletal pose and shape parameters from the monocular UAV video.
    - **Mechanism**: Adopts a SMPL-based monocular motion capture method, combining temporal consistency constraints and motion smoothness priors to ensure the stability of long-sequence motions.
    - **Design Motivation**: Outdoor sports involve large-scale, fast human movements (such as running and jumping), where traditional single-frame pose estimation is prone to jitter and inconsistency.

3. **4D Human + 3D Scene Reconstruction**: 
    - **Function**: Simultaneously reconstructs a deformable 4D dynamic human and a 360° unbounded outdoor background.
    - **Mechanism**: Separates the processing of the human body and the background—the human body part is reconstructed deformably based on a parametric human model, while the background part is reconstructed using an unbounded NeRF from multi-view images, finally combined during rendering.
    - **Design Motivation**: The dynamic human body and the static background possess distinct motion characteristics and scale features; separate processing allows optimization tailored to their respective properties.

### Loss & Training

- **Photometric Reconstruction Loss**: Measures the pixel-level difference between the rendered image and the ground truth input image.
- **Perceptual Loss**: Uses pre-trained networks to extract features, measuring the perceptual difference in rendering quality.
- **Regularization Loss**: Imposes smoothness constraints on the human deformation field and background geometry to prevent overfitting.
- **Stage-wise Training**: Trains the background NeRF first, followed by the dynamic human model, and finally performs joint fine-tuning.

## Key Experimental Results

### Main Results

The authors collected a real-world outdoor sports dataset named AerialRecon, which contains multiple outdoor sports scenes (such as running, throwing, etc.).

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|---------|------|
| AerialRecon | PSNR (Human) | Outperforms baselines | Existing SOTA systems | Significant improvement |
| AerialRecon | SSIM (Human) | Outperforms baselines | Existing SOTA systems | Significant improvement |
| AerialRecon | LPIPS (Human) | Outperforms baselines | Existing SOTA systems | Significant improvement |
| AerialRecon | Background Rendering Quality | Outperforms baselines | Existing SOTA systems | Significant improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Without calibration optimization | PSNR drops | Precise calibration is critical to reconstruction quality |
| Without temporal constraints | Motion smoothness drops | Temporal consistency constraints improve motion capture stability |
| Without human-background separation | Overall quality drops | The separated processing strategy effectively improves the reconstruction quality of each part |

### Key Findings

- The UAV platform possesses a significant advantage over handheld cameras in outdoor sports scenarios, covering a wider range of viewpoints.
- The system-level co-design (calibration -> motion capture -> reconstruction -> rendering) is more effective than optimizing individual modules in isolation.
- The system demonstrates superior performance and applicability compared to existing SOTA systems in real outdoor sports scenarios.
- The AerialRecon dataset fills the data gap in research on free-viewpoint video for outdoor sports.

## Highlights & Insights

- **High Practicality**: Achieves free-viewpoint video using a single consumer-grade UAV, significantly reducing costs and deployment complexity.
- **Systematic Design**: Rather than simply stacking existing methods, the framework performs end-to-end system-level design spanning data acquisition, calibration, motion capture, and rendering, with all modules working collaboratively.
- **Real-world Validation**: Evaluated under real outdoor sports scenarios rather than being restricted to controlled laboratory environments.
- **New Dataset Contribution**: Provides the AerialRecon dataset, establishing a benchmark for future research.

## Limitations & Future Work

- The video frame rate and coverage angle of a single UAV are limited, which may affect the reconstruction quality of fast motion.
- Outdoor lighting variations (e.g., weather changes, shadows) may affect rendering consistency.
- Current human reconstruction assumes a single athlete in the scene; extending to multi-person scenarios requires further investigation.
- Real-time rendering capability is limited, making it currently difficult to meet the demands of real-time broadcasting.
- UAV flights are restricted by battery life and flight regulations, limiting practical application scenarios.

## Related Work & Insights

- **Neural Body / HumanNeRF / NeuralMan**: NeRF-based human reconstruction methods; this work extends them to outdoor scenarios.
- **Instant-NGP / Mip-NeRF 360**: Unbounded scene NeRF reconstruction methods, which this work adopts for background reconstruction.
- **SMPL / SMPL-X**: Parametric human models, which provide priors for human motion capture.
- **Insight**: Utilizing low-cost mobile platforms such as UAVs for scene acquisition represents a highly promising and valuable research direction.

## Rating

- Novelty: ⭐⭐⭐⭐ The first to utilize a UAV platform for free-viewpoint video generation of outdoor sports, with a novel system design.
- Experimental Thoroughness: ⭐⭐⭐ Validated in real-world scenarios, but quantitative comparisons might be limited by the dataset scale.
- Writing Quality: ⭐⭐⭐⭐ The system is clearly described with a reasonable design across its modules.
- Value: ⭐⭐⭐⭐ High practical value, significantly lowering the technical barrier to free-viewpoint video.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[ECCV 2024\] Real-Data-Driven 2000 FPS Color Video from Mosaicked Chromatic Spikes](real-data-driven_2000_fps_color_video_from_mosaicked_chromatic_spikes.md)
- [\[CVPR 2026\] Computer Vision with a Superpixelation Camera](../../CVPR2026/others/computer_vision_with_a_superpixelation_camera.md)
- [\[ECCV 2024\] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)
- [\[ECCV 2024\] ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting](abc_easy_as_123_a_blind_counter_for_exemplar-free_multi-class_class-agnostic_cou.md)

</div>

<!-- RELATED:END -->
