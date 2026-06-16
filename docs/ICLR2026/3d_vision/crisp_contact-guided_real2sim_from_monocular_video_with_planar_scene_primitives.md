---
title: >-
  [Paper Note] CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives
description: >-
  [ICLR 2026][3D Vision][Real2Sim] This paper proposes CRISP, a method for recovering simulatable human motion and scene geometry from monocular video. By fitting planar primitives to obtain clean…
tags:
  - "ICLR 2026"
  - "3D Vision"
  - "Real2Sim"
  - "monocular video"
  - "planar scene primitives"
  - "human-scene interaction"
  - "RL humanoid control"
date: 2026-05-08
content_hash: 8760de245ca6f93f
---

# CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives

**Conference**: ICLR 2026
**arXiv**: [2512.14696](https://arxiv.org/abs/2512.14696)  
**Code**: Available (project page)  
**Area**: 3D Vision / Real2Sim
**Keywords**: Real2Sim, monocular video, planar scene primitives, human-scene interaction, RL humanoid control

## TL;DR
This paper proposes CRISP, a method for recovering simulatable human motion and scene geometry from monocular video. By fitting planar primitives to obtain clean, simulation-ready geometry and leveraging human-scene contact modeling to reconstruct occluded regions, CRISP reduces the motion tracking failure rate of a humanoid controller from 55.2% to 6.9%.

## Background & Motivation

Real2Sim (converting real environments into simulation-ready representations) is a central problem in robotics and AR/VR. Recovering physically simulatable human motion and scene geometry from monocular video has significant value for robot policy training, motion retargeting, and virtual reality content creation.

**Limitations of Prior Work**:

**Joint optimization methods with data-driven priors**: These approaches rely on learned priors for joint human-scene reconstruction but operate with no physics in the loop, resulting in physically implausible outputs such as body-object interpenetration.

**Direct geometric reconstruction methods**: Although capable of recovering scene geometry, the results typically contain noise and artifacts. Such unclean geometry causes failures in scene interaction when fed into motion tracking policies — for example, an uneven chair surface can trigger abnormal physical collisions when a humanoid controller attempts to sit.

**Key Challenge**: Existing methods either lack physical plausibility or produce geometry that is insufficiently clean to be directly used for interactive physical simulation.

**Core Idea**: Fit planar primitives to the scene point cloud to obtain convex, clean, simulation-ready geometry, and use human-scene contact modeling to recover geometrically occluded regions during interaction.

## Method

### Overall Architecture
**Input**: Monocular RGB video. **Output**: Simulatable human motion sequences and clean scene geometry. The pipeline consists of three main stages: (1) scene geometry reconstruction via planar primitive fitting, (2) occluded region recovery via contact guidance, and (3) physical validation via a humanoid controller trained with RL.

### Key Designs

1. **Planar Primitive Fitting**:

    - Dense point cloud reconstruction is first obtained from the video using existing methods.
    - A clustering pipeline segments the point cloud based on three features: depth, surface normals, and optical flow.
    - A planar primitive is fit to each cluster.
    - The result is a compact scene representation composed of convex planar surfaces.
    - **Design Motivation**: Planar primitives are inherently convex and noise-free, making them well-suited for physics simulation engines. Compared to meshes or implicit representations, they introduce no noisy artifacts and support efficient, stable collision detection.
    - **Additional Benefit**: Simulation throughput improves by 43%, as convex geometry enables much faster collision detection than complex meshes.

2. **Contact-Guided Occlusion Recovery**:

    - During human-scene interaction, portions of the scene geometry are occluded by the human body (e.g., a chair seat is hidden when a person sits).
    - Human-scene contact modeling is used to infer the occluded geometry.
    - **Core Idea**: Human pose implicitly encodes scene geometry — for instance, a sitting pose can be used to infer the position and shape of a chair seat.
    - By estimating contact points between body joints and the scene, the method back-projects the planar positions of occluded scene regions.
    - This approach requires no prior CAD models or scene category templates.

3. **Physical Validation via Humanoid Controller + RL**:

    - The recovered human motion and scene geometry are used to drive a humanoid controller.
    - An RL policy is trained to track the original video motion within the reconstructed scene.
    - This step serves both as a validation mechanism (poor reconstruction quality leads to RL failure) and as a final output (producing simulatable human motion).
    - Physical simulation enforces physical plausibility in the final results: no penetration, maintained balance, and reasonable contact.

### Loss & Training
- **Clustering stage**: Unsupervised clustering using distance metrics over depth, surface normal, and optical flow features.
- **Plane fitting**: Least-squares fitting of plane parameters for each cluster.
- **RL controller training**: Standard PPO or similar policy gradient method; reward function includes motion tracking error and physical plausibility penalties (e.g., penetration, loss of balance).

## Key Experimental Results

### Main Results
Evaluated on the human-centric video benchmarks EMDB and PROX:

| Method | Motion Tracking Failure Rate ↓ | RL Simulation Throughput | Notes |
|--------|-------------------------------|--------------------------|-------|
| Prior methods (noisy geometry) | 55.2% | Baseline | Geometric artifacts cause frequent failures |
| **CRISP (Ours)** | **6.9%** | **+43% faster** | Clean geometry drastically reduces failures |

### In-the-Wild Video Validation

| Video Type | Result | Notes |
|------------|--------|-------|
| Casually captured daily videos | Success | Generalizes to uncontrolled environments |
| Internet videos | Success | Generalizes to diverse scenes |
| Sora-generated videos | Success | Applicable even to AI-generated content |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Without planar primitives (raw mesh) | Failure rate increases substantially | Validates the critical role of planar primitives |
| Without contact-guided occlusion recovery | Poor performance on interaction scenes | Occlusion recovery is necessary for interactions |
| Without RL validation (direct output) | Physically implausible penetration | RL ensures physical realism |
| Different clustering feature combinations | Depth + normals + flow is optimal | Three features are complementary |

### Key Findings
- Planar primitives are an ideal scene representation for Real2Sim: clean, convex, and computationally efficient.
- Human pose is a powerful signal for inferring occluded scene geometry.
- The motion tracking failure rate drops from 55.2% to 6.9%, representing an approximately 88% relative improvement.
- The 43% throughput gain stems from more efficient collision detection with convex geometry.
- The method generalizes well to in-the-wild videos, including AI-generated content from Sora.
- The entire pipeline requires no CAD model libraries or scene category priors.

## Highlights & Insights
- The insight of replacing complex meshes with planar primitives is simple yet powerful — at a modest cost in geometric detail, simulation compatibility is substantially improved.
- Contact-guided occlusion recovery is the key innovation: human pose serves as a "mold" for inferring hidden geometry.
- Using the RL humanoid controller as a physical plausibility validator forms a meaningful closed loop.
- Successful validation on Sora-generated videos demonstrates strong generalization potential and forward-looking applicability.
- The depth + normals + optical flow feature combination for clustering is concise yet effective.
- The method can generate physically valid human motion and interaction environments at scale, with direct applications in robotics and AR/VR.

## Limitations & Future Work
- The planar primitive assumption limits representational capacity for curved objects (e.g., spheres, cylinders).
- The pipeline depends on the quality of upstream point cloud reconstruction; inaccurate depth estimation propagates errors throughout.
- Contact-based inference relies on human pose, and cannot handle occlusions caused by distant, non-contact objects.
- Clustering hyperparameters (e.g., number of clusters, distance thresholds) may require scene-specific tuning.
- Dynamic scenes involving moving objects are not currently handled.
- Training the RL controller itself requires substantial computational resources.
- Future work may extend to multi-person interaction scenarios and more complex object manipulation tasks.

## Related Work & Insights
- Related to human-scene interaction reconstruction methods such as PROX and LEMO, but introduces physical simulation as a validation step.
- Complementary to physics-constrained motion generation approaches such as PhysDiff.
- The planar primitive fitting idea connects to classical computational geometry work on plane detection, but its application to Real2Sim is novel.
- **Insights**: (1) Simple geometric representations often outperform noisy but detailed ones in simulation contexts; (2) human pose as an implicit encoder of scene geometry is a direction worth deeper exploration; (3) the use of RL as a validation tool is transferable to other reconstruction tasks.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Illumination-Consistent Human-Scene Reconstruction from Monocular Video](../../CVPR2026/3d_vision/illumination-consistent_human-scene_reconstruction_from_monocular_video.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](../../ICCV2025/3d_vision/vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[CVPR 2026\] Elastic3D: Controllable Stereo Video Conversion with Guided Latent Decoding](../../CVPR2026/3d_vision/elastic3d_controllable_stereo_video_conversion_with_guided_latent_decoding.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2026/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[AAAI 2026\] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](../../AAAI2026/3d_vision/mobgs_motion_deblurring_dynamic_3d_gaussian_splatting_for_blurry_monocular_video.md)

</div>

<!-- RELATED:END -->
