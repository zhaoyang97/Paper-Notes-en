---
title: >-
  [Paper Note] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions
description: >-
  [ICCV 2025][Object Detection][Human Pose Estimation] This paper presents VOccl3D, a large-scale synthetic video dataset (250K frames, 400 video sequences) rendered via 3DGS, targeting 3D human pose and shape (HPS) estimation under realistic occlusion scenarios. Models fine-tuned on VOccl3D demonstrate significant performance improvements in occluded settings.
tags:
  - ICCV 2025
  - Object Detection
  - Human Pose Estimation
  - Occlusion
  - Synthetic Dataset
  - 3D Gaussian Splatting
  - SMPL-X
date: 2026-05-08
content_hash: 7e33883b4ad89ebc
---

# VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions

**Conference**: ICCV 2025
**arXiv**: [2508.06757](https://arxiv.org/abs/2508.06757)
**Code**: [https://yashgarg98.github.io/VOccl3D-dataset/](https://yashgarg98.github.io/VOccl3D-dataset/)
**Area**: Object Detection
**Keywords**: Human Pose Estimation, Occlusion, Synthetic Dataset, 3D Gaussian Splatting, SMPL-X

## TL;DR

This paper presents VOccl3D, a large-scale synthetic video dataset (250K frames, 400 video sequences) rendered via 3DGS, targeting 3D human pose and shape (HPS) estimation under realistic occlusion scenarios. Models fine-tuned on VOccl3D demonstrate significant performance improvements in occluded settings.

## Background & Motivation

While 3D human pose and shape (HPS) estimation methods perform well in standard settings, they remain challenging under **severe occlusion**. The core issues are:

**Existing occlusion datasets lack realism**: Most datasets simulate occlusion with random color patches or copy-paste artifacts (e.g., 3DPW-AdvOcc), which are far from real-world occlusion patterns.

**Natural occlusion datasets have insufficient occlusion coverage**: Datasets such as 3DPW and OCMotion contain only sparse and infrequent occlusions.

**Lack of large-scale, diverse, heavily occluded training data**: This prevents models from learning robust estimation under occlusion.

VOccl3D aims to address this gap by providing a large-scale synthetic video dataset with realistic occlusions.

## Method

### Overall Architecture

The dataset construction pipeline proceeds as follows:
1. **3D Gaussian Splatting** is applied to real-world videos from the DL3DV dataset to learn 3D representations of 40 background scenes.
2. Approximately 400 SMPL-X human motion sequences are sampled from the **AMASS** motion capture dataset.
3. Around 200 human textures are selected from the **SMPLitex** dataset.
4. All components are integrated and rendered in the **Unity** engine: 3DGS background + human animation + texture → synthetic video.

### Key Designs

1. **3DGS Background Scenes**: Unlike methods such as BEDLAM that rely on graphic assets, VOccl3D leverages 3D Gaussian Splatting to learn 3D scene representations from real-environment videos. This offers two advantages:

    - Rendered scenes more closely match the real data distribution.
    - Domain-specific datasets (e.g., agricultural, street, indoor) can be flexibly created from raw RGB video alone.
    - Scenes from DL3DV are selected to include natural occluders (garlands, chairs, benches, sculptures, cars, trash cans, etc.).

2. **Motion Sequence Processing**:

    - VPoser is used to assess pose difficulty, filtering for challenging poses ($\|\epsilon_\theta\|_2 > 40$).
    - Each sequence is required to contain at least 180 frames (6 seconds at 30 fps).
    - Boundary constraints are enforced to prevent subjects from moving outside occluded regions.
    - Random rotations and translations are applied to increase diversity.

3. **Dataset Properties and Annotations**:

    - 250,000+ frames, 400 video sequences, total duration > 2.5 hours.
    - 40 background scenes, 10 videos per scene.
    - Full annotations are provided: camera intrinsics/extrinsics, SMPL-X pose/shape parameters, global orientation, translation, gender, and 2D keypoints.
    - **Keypoint-level occlusion labels**: binary occlusion flags per keypoint.
    - Three-tier occlusion classification: hard occlusion (4–9 visible keypoints), medium occlusion (10–15), and low occlusion (16–20).

4. **Multi-modal Annotations**: In addition to 3D pose, the dataset supports human silhouette, body-part segmentation, 2D keypoints, and bounding box annotations.

### Loss & Training

- CLIFF and BEDLAM-CLIFF are fine-tuned on VOccl3D, yielding VOccl3D-CLIFF and VOccl3D-B-CLIFF.
- Approximately 200K training images and 50K test images are used.
- Five-fold cross-validation is employed to report average performance.
- YOLO11 is also fine-tuned as a human detector → VOccl3D-YOLO11.

## Key Experimental Results

### Main Results (Tables)

**VOccl3D Test Set (Ground-truth bounding boxes):**

| Method | Hard-Occ MPJPE | Hard-Occ PA-MPJPE | Med-Occ MPJPE | Med-Occ PA-MPJPE | Low-Occ MPJPE | Low-Occ PA-MPJPE |
|--------|---------------|-------------------|---------------|-------------------|---------------|-------------------|
| CLIFF | 192.22 | 114.35 | 121.70 | 78.56 | 98.82 | 67.64 |
| BEDLAM-CLIFF | 154.86 | 99.53 | 90.97 | 65.03 | 74.95 | 52.65 |
| HMR2.0 | 169.71 | 100.49 | 113.88 | 71.78 | 88.53 | 59.08 |
| WHAM | 152.15 | 102.14 | 110.97 | 76.81 | 93.90 | 66.68 |
| **VOccl3D-B-CLIFF** | **136.34** | **89.94** | **82.48** | **58.78** | **69.46** | **46.32** |

VOccl3D-B-CLIFF reduces Hard-Occlusion MPJPE by approximately **18.5 mm** compared to BEDLAM-CLIFF.

### Ablation Study (Tables)

**Real Dataset Evaluation (Effect of Detector):**

| Method | 3DPW MPJPE | 3DPW PA-MPJPE | OCMotion MPJPE | OCMotion PA-MPJPE |
|--------|-----------|---------------|---------------|-------------------|
| VOccl3D-CLIFF w/ GT Box | 71.10 | 45.98 | 64.29 | 39.64 |
| VOccl3D-CLIFF w/ YOLO11 | 116.52 | 63.35 | 67.16 | 41.30 |
| VOccl3D-CLIFF w/ VOccl3D-YOLO11 | **114.85** | **62.66** | **66.65** | **41.15** |

**Occlusion-Augmented 3DPW Variants:**

| Method | 3DPW MPJPE | OcclType1-3DPW MPJPE | OcclType2-3DPW MPJPE |
|--------|-----------|---------------------|---------------------|
| CLIFF | 73.9 | 98.15 | 99.49 |
| BEDLAM-CLIFF | 72.0 | 98.71 | 96.80 |
| **VOccl3D-B-CLIFF** | **72.0** | **95.89** | **93.74** |

### Key Findings

- **Performance gap increases with occlusion severity**: All methods degrade substantially from low to hard occlusion, but VOccl3D fine-tuned models show the largest advantage under hard occlusion.
- **Effective transfer from synthetic to real data**: Competitive or superior performance is achieved on real-world datasets 3DPW and OCMotion.
- **Detector is a bottleneck for HPS**: GT boxes vs. YOLO11 detections lead to an MPJPE gap of ~45 mm on 3DPW, highlighting the critical role of the detector under occlusion.
- **STRIDE + VOccl3D pseudo-labels synergy**: Pseudo-labels generated by VOccl3D-B-CLIFF outperform those from the original BEDLAM-CLIFF.

## Highlights & Insights

- **3DGS as a replacement for traditional graphic assets**: Learning real-scene representations via 3D Gaussian Splatting yields more realistic backgrounds at lower production cost.
- **Keypoint-level occlusion annotations**: More fine-grained than image-level occlusion labels, enabling deeper occlusion analysis.
- **Multi-task benchmark**: The same dataset supports evaluation of HPS estimation, detection, and segmentation tasks.
- **End-to-end analysis**: A full pipeline analysis from detector to pose estimator reveals the critical impact of detection quality on HPS performance.

## Limitations & Future Work

- A domain gap between synthetic and real data remains (lighting, texture, physics).
- Only the SMPL-X body model is used; fine-grained hand and facial estimation is not addressed.
- The number of background scenes (40) is relatively limited; expansion to more diverse environments is desirable.
- Dynamic occluders (e.g., moving vehicles or other pedestrians) are not considered.
- Physical interactions between human bodies and occluders (e.g., contact and collision) are not modeled.

## Related Work & Insights

- VOccl3D follows in the tradition of BEDLAM but focuses specifically on occlusion, filling an important gap in occlusion-specific datasets.
- The 3DGS-based background generation approach is generalizable to other synthetic data needs (e.g., autonomous driving, robotics).
- Cascaded improvements from detector fine-tuning suggest that full-pipeline optimization is necessary for occluded downstream tasks.
- Integration with diffusion models could enable generation of more diverse occlusion scenarios.

## Rating

- **Novelty**: ⭐⭐⭐ Methodological novelty is limited; the primary contribution lies in dataset construction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple models, datasets, occlusion levels, and detector configurations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with sufficient detail on dataset construction.
- **Value**: ⭐⭐⭐⭐ Fills an important gap in occlusion datasets for HPS research and can serve as a standard benchmark.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes](../../NeurIPS2025/object_detection/delving_into_cascaded_instability_a_lipschitz_continuity_view_on_image_restorati.md)
- [\[ICCV 2025\] YOLOE: Real-Time Seeing Anything](yoloe_realtime_seeing_anything.md)
- [\[ICLR 2026\] ForestPersons: A Large-Scale Dataset for Under-Canopy Missing Person Detection](../../ICLR2026/object_detection/forestpersons_a_large-scale_dataset_for_under-canopy_missing_person_detection.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)

<!-- RELATED:END -->
