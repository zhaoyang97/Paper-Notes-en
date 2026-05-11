---
title: >-
  [Paper Note] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving
description: >-
  [AAAI 2026][Autonomous Driving][3D Object Detection] DriveFlow is a rectified flow adaptation method built upon pretrained T2I Flow models. Through frequency decomposition…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "3D Object Detection"
  - "Data Augmentation"
  - "Rectified Flow"
  - "Image Editing"
  - "Robustness"
date: 2026-05-08
content_hash: 1b9485f26e13b51c
---

# DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving

**Conference**: AAAI 2026
**arXiv**: [2511.18713](https://arxiv.org/abs/2511.18713)
**Code**: [Available](https://github.com/Hongbin98/DriveFlow)
**Area**: Autonomous Driving
**Keywords**: 3D Object Detection, Data Augmentation, Rectified Flow, Image Editing, Robustness

## TL;DR

DriveFlow is a rectified flow adaptation method built upon pretrained T2I Flow models. Through frequency decomposition, it preserves high-frequency foreground content while applying dual-frequency optimization to the background, enabling training-free driving scene image editing for data augmentation and significantly improving the OOD robustness of vision-based 3D detectors.

## Background & Motivation

Vision-centric 3D object detection relies on RGB images to detect and localize three-dimensional objects, and is central to autonomous driving perception. However, training data rarely covers all possible test scenarios (e.g., fog, snow), leading to severe performance degradation under out-of-distribution (OOD) conditions.

Limitations of existing approaches:

- **Test-time adaptation methods** (MonoTTA, MonoWAD): introduce additional computational overhead during inference.
- **DriveGEN** (prior work): relies on inversion-based editing with a U-Net architecture based on SD 1.5, suffering from inversion inaccuracies and low computational efficiency.
- **General image editing methods** (FlowEdit, FreeControl): cannot preserve the precise 3D geometric structure of foreground objects, resulting in object misalignment and omission even with detailed text descriptions.

The core motivation is: can stronger pretrained T2I Flow models (e.g., SD3) be leveraged, without any additional training, to simultaneously achieve efficient scene style transfer and precise preservation of 3D object geometry?

## Method

### Overall Architecture

DriveFlow is built upon a pretrained T2I Flow model (e.g., Stable Diffusion 3) and operates in a training-free controllable image editing paradigm. The core pipeline is as follows:

1. Given a source image $X_0^{src}$, the initial latent $Z_0^{src}$ is obtained via a VAE encoder.
2. Source/target latent–text pairs are prepared: the source prompt describes the original scene (e.g., "sunny urban"), and the target prompt describes the desired scene (e.g., "rainy").
3. Based on the noise-free editing path of FlowEdit, frequency decomposition and optimization strategies are applied to learn an appropriate target velocity field $V'^{tar}_t$.
4. The editing latent is updated using the optimized velocity difference $\Delta V'_t$, and the augmented image is obtained by decoding.

The editing process builds on FlowEdit's core ODE formulation. Averaging over multiple random pairings eliminates the mismatch from fixed pairing and realizes a noise-free editing path, avoiding the stochastic perturbations inherent in inversion-based methods.

### Key Designs

**1. Frequency Decomposition**

The source and target velocity fields $V_t^{src}$ and $V_t^{tar}$ are decomposed into frequency components:

- Low-frequency component $V_{L,t}$: obtained via Gaussian blur $G_\sigma^{(k)}$, corresponding to slowly varying background structure.
- High-frequency component $V_{H,t} = V - V_{L,t}$: the high-frequency residual captures rapidly varying content, typically corresponding to objects within 2D bounding boxes.

**2. High-Frequency Foreground Preservation**

An L2 alignment loss is imposed between the source and target high-frequency components within all object regions, ensuring that the 3D geometric structure of foreground objects remains unchanged:

$$\mathcal{L}_{obj} = \frac{1}{|\mathbf{M}|}\|\mathbf{M} \odot (V_{H,t}^{tar} - V_{H,t}^{src})\|_2^2$$

where $\mathbf{M}$ is a binary mask downsampled from the image layout (2D bounding boxes). DriveFlow requires only target scene conditions and image layout (bboxes), without detailed text descriptions.

**3. Dual-Frequency Background Optimization**

To fully exploit the editing capacity of the pretrained model, the background regions require sufficient editing intensity:

- **Diversity loss**: maximizes the discrepancy between source and target low-frequency components in the background region:
$$\mathcal{L}_{div} = \frac{1}{|\bar{\mathbf{M}}|}\sum_{\bar{\mathbf{M}}} \cos(V_{L,t}^{tar}, V_{L,t}^{src})$$
Cosine similarity guides the model to focus on regions most similar to the original, encouraging more comprehensive background editing.

- **Background regularization**: prevents semantic collapse caused by the diversity loss alone:
$$\mathcal{L}_{bg} = \frac{1}{|\bar{\mathbf{M}}|}\|\bar{\mathbf{M}} \odot (V_{H,t}^{tar} - V_{H,t}^{src})\|_2^2$$
High-frequency regularization ensures that background editing strikes a balance between diversity and semantic consistency.

### Loss & Training

The overall optimization objective is:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{obj} + \lambda_2 \mathcal{L}_{div} + \lambda_3 \mathcal{L}_{bg}$$

For each diffusion step of each training image, the target velocity field $V'^{tar}_t$ is optimized through $N_n$ inner iterations, followed by updating the editing latent via the velocity difference. DriveFlow is a training-free method that does not modify pretrained model parameters; it performs online optimization of the velocity field only during data augmentation.

## Key Experimental Results

### Main Results

**Table 1: Car mAP on KITTI-C (MonoGround detector)**

| Method | Noise | Blur | Weather | Digital | Avg. |
|--------|-------|------|---------|---------|------|
| MonoGround (baseline) | 17.90 | 27.85 | 25.20 | 32.56 | 26.36 |
| + DriveGEN (Snow) | 22.54 | 36.49 | 33.60 | 39.63 | 33.38 |
| + DriveGEN (6×Aug.) | 28.91 | 39.99 | 36.36 | 43.39 | 37.21 |
| + FlowEdit (Snow) | 6.63 | 27.58 | 26.84 | 32.71 | 23.87 |
| + **DriveFlow (Snow)** | **29.67** | **40.70** | **41.79** | **44.16** | **39.75** |
| + DriveFlow (6×Aug.) | 33.22 | 44.82 | 43.11 | 46.34 | 42.27 |

**Table 2: Car mAP on KITTI-C (MonoCD detector)**

| Method | Avg. mAP |
|--------|----------|
| MonoCD (baseline) | 26.54 |
| + DriveGEN (Snow) | 35.79 |
| + DriveGEN (6×Aug.) | 38.67 |
| + **DriveFlow (Snow)** | **39.75** |
| + DriveFlow (6×Aug.) | 42.27 |

Key conclusion: **DriveFlow with only Snow augmentation surpasses DriveGEN with six augmentation types**, achieving an average improvement of approximately 14.54 mAP on KITTI-C.

### Ablation Study

- Removing high-frequency foreground preservation → severe degradation of foreground object geometry, significant drop in detection performance.
- Removing the diversity loss → insufficient background editing, reduced scene diversity.
- Removing background regularization → semantic drift in the background, particularly harmful for temporal multi-view detection.
- DriveFlow achieves a generation speed **23.8× faster** than DriveGEN (on KITTI).
- ControlNet and FreeControl as augmentation methods perform poorly or even degrade performance (ControlNet 6×Aug. reduces MonoCD Avg. to 0.00).

### Key Findings

1. Frequency decomposition is essential for balancing foreground preservation and background editing — high frequencies correspond to object edges and structure, while low frequencies correspond to overall scene style.
2. DriveFlow achieves comprehensive OOD performance gains on minority classes (Pedestrian), addressing the insufficient improvement exhibited by DriveGEN on such classes.
3. Background semantic consistency constraints are especially important for temporal 3D detectors (e.g., BEVDet4D), ensuring inter-frame background consistency.
4. The rectified flow editing path is more stable and efficient than inversion-based approaches, fundamentally eliminating inversion inaccuracies.

## Highlights & Insights

- **First application of rectified flow editing to robust 3D object detection**, opening a new direction for deploying pretrained T2I Flow models in autonomous driving.
- The frequency decomposition idea is elegant — high/low-frequency components of the velocity field naturally correspond to object/background structures in the image.
- Editing requires only 2D bboxes and a target scene description, enabling full annotation reuse with zero additional annotation cost.
- The 23.8× speedup is of significant practical value for large-scale data augmentation.

## Limitations & Future Work

- The method depends on the quality of pretrained T2I Flow models; insufficient understanding of driving scenes may limit editing effectiveness.
- Gaussian blur kernel parameters require manual tuning and may not represent the optimal frequency decomposition strategy (learnable frequency decomposition is worth exploring).
- Only 2D bboxes are used as foreground constraints without leveraging depth information; 3D geometric preservation could be further refined.
- The approach can be extended to additional modalities (e.g., LiDAR augmentation) and more complex editing tasks (object insertion/removal).

## Related Work & Insights

- **FlowEdit** provides the theoretical foundation for noise-free editing; DriveFlow builds upon it by incorporating foreground preservation constraints.
- **DriveGEN** is the most direct comparison baseline; DriveFlow achieves improvements in both quality and efficiency by replacing the editing paradigm.
- The frequency decomposition idea can inspire other image editing tasks requiring "local preservation + global modification."

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Exploring Surround-View Fisheye Camera 3D Object Detection](exploring_surround-view_fisheye_camera_3d_object_detection.md)
- [\[AAAI 2026\] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection](fq-petr_fully_quantized_position_embedding_transformation_fo.md)
- [\[AAAI 2026\] Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving](invisible_triggers_visible_threats_road-style_adversarial_creation_attack_for_vi.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](../../ICCV2025/autonomous_driving/robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)
- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)

</div>

<!-- RELATED:END -->
