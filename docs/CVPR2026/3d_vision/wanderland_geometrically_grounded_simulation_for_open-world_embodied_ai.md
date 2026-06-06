---
title: >-
  [Paper Note] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
description: >-
  [CVPR 2026][3D Vision][real-to-sim] This paper proposes Wanderland, a real-to-sim framework that uses a handheld multi-sensor scanner (LiDAR+IMU+RGB) to capture open-world indoor and outdoor scenes. It employs LIV-SLAM t…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "real-to-sim"
  - "3D Gaussian Splatting"
  - "LiDAR-Inertial-Visual SLAM"
  - "navigation simulation"
  - "geometric grounding"
  - "novel view synthesis"
date: 2026-05-08
content_hash: 42f66cb1c0b3ff2b
---

# Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI

**Conference**: CVPR 2026
**arXiv**: [2511.20620](https://arxiv.org/abs/2511.20620)  
**Code**: [https://ai4ce.github.io/wanderland/](https://ai4ce.github.io/wanderland/)  
**Area**: 3D Vision / Embodied AI / Simulation Environments
**Keywords**: real-to-sim, 3D Gaussian Splatting, LiDAR-Inertial-Visual SLAM, navigation simulation, geometric grounding, novel view synthesis

## TL;DR

This paper proposes Wanderland, a real-to-sim framework that uses a handheld multi-sensor scanner (LiDAR+IMU+RGB) to capture open-world indoor and outdoor scenes. It employs LIV-SLAM to obtain metric-accurate geometry and camera poses, combines 3DGS for photorealistic rendering with geometrically grounded collision simulation, and constructs a large-scale dataset of 530 scenes / 420K frames / 3.8M m². The work systematically demonstrates that purely vision-based reconstruction falls significantly short of LiDAR-enhanced approaches in metric accuracy, mesh quality, and reliability for navigation policy training and evaluation.

## Background & Motivation

- **Background**: Open-world embodied navigation is expanding from indoor settings to urban streets, campuses, and commercial areas, demanding simulation environments with large spatial scale, mixed indoor-outdoor coverage, high-fidelity sensor simulation, and reliable physical interaction.

- **Limitations of Prior Work**: Classic RGB-D datasets such as Matterport3D, ScanNet, and HM3D rely on tripod-mounted RGB-D capture, making them unsuitable for outdoor use due to sunlight interference and limited depth range; pose drift becomes severe in large-scale, low-texture environments. Video-based 3DGS approaches (e.g., Vid2Sim, GaussGym) suffer from three fundamental deficiencies: (a) purely RGB SfM/depth estimation yields non-metric poses, (b) collision meshes extracted from 3DGS opacity are fragmented and lack metric grounding, and (c) single-trajectory video leads to severe degradation in novel view rendering quality upon extrapolation. Furthermore, existing outdoor datasets lack high-accuracy geometric ground truth, hindering evaluation of reconstruction and NVS methods.

- **Key Challenge**: Video-based 3DGS environments may suffice for training but cannot serve as standardized, reproducible benchmarks for closed-loop evaluation due to unreliable geometry.

- **Goal**: To construct a scalable real-to-sim pipeline and large-scale dataset that provides metric-accurate geometry and photorealistic rendering for open-world embodied AI research.

## Method

### Overall Architecture

Multi-sensor capture → LIV-SLAM reconstruction (globally consistent metric point cloud + accurate camera poses) → 3DGS initialization from point cloud + depth-regularized training → collision mesh extraction from point cloud → 3DGS model + mesh integrated into USD scene → loaded into Isaac Sim for navigation training/evaluation. The core design principle is: geometry originates from LiDAR (ensuring metric accuracy) and appearance from 3DGS (ensuring photorealism), with both sharing a unified coordinate system for seamless integration.

### 1. Data Capture (Section 3.1)

- **Hardware**: MetaCam Air handheld 3D scanner, integrating a Livox Mid-360 non-repetitive LiDAR (with built-in IMU), RTK-GNSS antenna, and two synchronized 4K fisheye cameras (>180° FOV), all factory-calibrated.
- **LiDAR Mounting**: Tilt angle optimized to maximize ground detail capture and camera FOV overlap.
- **Real-time Monitoring**: A companion mobile app displays colored point cloud previews in real time, enabling operators to actively fill coverage gaps.
- **Capture Protocol**:
    - Scene scale: 5,000–10,000 m², balancing complexity and coverage.
    - Trigger strategy: RGB capture triggered by displacement/angle thresholds rather than fixed frame rates, ensuring uniform viewpoint distribution.
    - Trajectory strategy: **Training trajectories** follow closed-loop paths with dense multi-view coverage of all navigable areas; **extrapolation trajectories** simulate natural navigation paths with minimal overlap with training trajectories. This contrasts sharply with the unidirectional urban walk videos used by Vid2Sim/GaussGym.
    - Quality control: Dynamic obstacles and reflective surfaces are minimized; lighting is kept consistent within a single capture session; real-time point cloud monitoring verifies coverage completeness.
- **Diversity**: Data collected in New York City and Jersey City, covering residential buildings, commercial districts, streets, plazas, and university campuses, under varying times of day (morning/midday/evening) and weather conditions (sunny/cloudy/light rain).

### 2. Data Processing Pipeline (Section 3.2)

**a) LIV-SLAM Mapping and Reconstruction**

- Raw sensor data processed using MetaCam Studio, implementing LiDAR-inertial-visual-GNSS multi-sensor fusion.
- Extended from established multi-sensor fusion methods (VINS-Mono, GVINS, FAST-LIO2, etc.).
- Output: dense, metric-scale point cloud (millimeter-level spacing) + globally consistent camera trajectory.
- This geometric foundation directly addresses the metric accuracy and completeness limitations of purely vision-based approaches.

**b) Image Preprocessing**

- **Privacy masking**: EgoBlur is used to mask faces and license plates for anonymized data release.
- **Dynamic object masking**: YOLOv11 detects and masks pedestrians, animals, vehicles, and other dynamic objects; masked pixels are filtered during 3DGS training.
- **Undistortion**: A 120° perspective crop is extracted from raw fisheye images and undistorted, motivated by the low-order distortion approximation limitations of 3DGS and the predominant training of detection models on pinhole images.

### 3. 3DGS Training Strategy (Section 3.3)

**a) Metric Point Cloud Initialization**

- 3D Gaussians are initialized directly from the dense colored point cloud output by LIV-SLAM (raw spacing 5–10 mm, downsampled to ~5M points per scene).
- Each point yields one Gaussian; initial scale is set via KNN heuristics.
- Opacity is parameterized as the inverse of volumetric density to suppress the dominance of large Gaussians and floating noise in early training steps.
- Training resolution: 800×800; framework: gsplat; 15,000 steps.

**b) Depth Regularization**

- Rather than using monocular depth estimates as pseudo-GT, depth maps are obtained by projecting the initialized Gaussians onto each camera pose and used as GT depth.
- Joint optimization combines photometric loss and depth loss.
- Freezing Gaussian centers was found to preserve multi-view geometric consistency but produces suboptimal visual quality and degenerate behavior; pure image supervision improves single-view fidelity at the cost of unseen-view consistency; depth regularization achieves a balance between the two.

**c) Training View Augmentation**

- Pretrained Difix3D+ is used to generate clean, geometrically accurate novel views for augmentation.
- Augmented views are progressively shifted farther from training views over the course of training.
- This is critical for stabilizing large-scale 3DGS training and improving sensor simulation quality for extrapolated viewpoints.

### 4. Geometrically Grounded Simulation Construction (Section 3.4)

**a) Mesh Extraction**

- The global point cloud is voxelized into an occupancy grid; Marching Cubes is applied to extract a geometric mesh.
- Post-processing removes regions far from the capture trajectory and filters out fragments with too few faces.
- Compared to mesh extraction from 3DGS opacity (Vid2Sim/GaussGym), meshes extracted directly from LiDAR point clouds are clean, complete, and metrically accurate.

**b) Scene Integration**

- The mesh and 3DGS model share the same global point cloud coordinate system and are directly integrated into USD format.
- The mesh provides a lightweight physics/collision layer; 3DGS serves as the primary renderer.
- Scenes can be directly loaded into Isaac Sim for navigation system training and evaluation.

### 5. Navigation Task Definition (Section 3.5)

- **Expert trajectories**: The mesh is imported into Unity; navigable triangles are extracted via the NavMesh baking API, and a pathfinding module generates collision-free shortest paths. Start and goal points are sampled near captured camera positions, supporting both point-goal and image-goal navigation.
- **Language instructions**: Trajectory playback generates egocentric video → Gemini VLM automatically generates natural language navigation instructions → human verification ensures reliability. This approach is more scalable than fully manual annotation (R2R, RxR) and yields more consistent instruction quality across scenes.

## Dataset Scale

| Metric | Value |
|:---|:---|
| Number of Scenes | 530 |
| Total Frames | 420K+ |
| Total Capture Duration | 100+ hours |
| Coverage Area | 3.8M+ m² |
| Scene Types | Mixed indoor/outdoor (residential / commercial / streets / plazas / campus) |
| Per-scene Data | Fisheye RGB + calibration parameters + global poses + colored point cloud + 3DGS model + collision mesh + USD scene |
| Expansion Plan | Continuously maintained toward 1,000+ scenes |

## Key Experimental Results

### 1. 3D Reconstruction Accuracy (Q1: Pure Vision vs. LIV-SLAM)

Using LIV-SLAM poses as ground truth, the camera pose accuracy of 8 vision-only methods is evaluated:

| Method | T-ATE_raw (m) ↓ | T-ATE_scaled (m) ↓ | R-ATE (°) ↓ | AUC@30 ↑ | Success Rate ↑ |
|:---|:---|:---|:---|:---|:---|
| DUSt3R | 15/14 | 20/18 | 73/60 | 0.12/0.07 | 0.39 |
| MUSt3R | 7.8/5.7 | 10/3.7 | 26/13 | 0.53/0.61 | 0.81 |
| VGGT | 15/14 | 9.9/4.5 | 33/15 | 0.44/0.52 | 0.80 |
| π³ | 15/14 | 4.7/1.4 | 21/6.9 | 0.64/0.76 | 0.89 |
| COLMAP | 16/10 | 8.1/2.3 | 42/10 | 0.50/0.64 | 0.64 |
| COLMAP_calib | 10/9.7 | 4.8/0.30 | 15/5.0 | 0.73/0.83 | 0.87 |
| **Best across all methods** | **2.8** | **0.30** | **5.0** | **0.83** | **0.89** |

**Key Findings**: Even taking the best combination across all methods, raw metric error remains at the meter level in scenes under 100 meters; after scale alignment, average error is 30 cm / 5°. The fundamental modal limitations of vision-only methods in metric accuracy have not been bridged by the latest foundation models.

### 2. Photorealistic Simulation (Q2: NVS Quality)

Novel view synthesis quality is evaluated on Wanderland using a unified train/val split:

| Method | Interp. PSNR ↑ | Interp. SSIM ↑ | Interp. LPIPS ↓ | Extrap. PSNR ↑ | Extrap. SSIM ↑ | Extrap. LPIPS ↓ |
|:---|:---|:---|:---|:---|:---|:---|
| 3DGS (COLMAP) | 18.27 | 0.658 | 0.510 | 16.90 | 0.624 | 0.559 |
| 2DGS (COLMAP) | 17.98 | 0.593 | 0.550 | 16.81 | 0.631 | 0.508 |
| Vid2Sim | 17.20 | 0.549 | 0.399 | 16.49 | 0.573 | 0.371 |
| GaussGym | 12.17 | 0.440 | 0.738 | 12.63 | 0.436 | 0.725 |
| **Wanderland (Ours)** | **20.37** | **0.688** | **0.327** | **17.92** | 0.591 | 0.445 |

**Key Findings**: Wanderland achieves the best performance across all interpolation metrics (PSNR lead of 2+ dB) and the highest extrapolation PSNR. GaussGym performs worst due to inaccurate VGGT reconstruction. Vid2Sim introduces additional noise from unreliable monocular depth estimation. Semantic consistency experiments show that GaussGym's fragmented rendering prevents Grounded SAM 2 from detecting key environmental elements, and Vid2Sim's DINOv3 features deviate significantly from GT.

### 3. Navigation Policy Training and Evaluation (Q3: Simulation Reliability)

**RL Training Comparison**: Models trained via RL in Vid2Sim environments generally degrade (CityWalker SR decreases by 21%), as inaccurate geometry encourages locally short but globally unreliable behaviors; all models improve significantly when trained in Wanderland environments (CityWalker SR increases by 14%, intervention rate decreases by 23%).

**Evaluation Reliability**: The same models evaluated in Vid2Sim environments show lower success rates and higher intervention rates, demonstrating that geometrically unreliable environments cannot support faithful evaluation.

**Navigation Benchmark**: Benchmarking 5 pretrained models on the full Wanderland dataset, no model achieves an outdoor success rate above 31%, highlighting the vast research space remaining for open-world navigation. NaVILA (a VLN model) performs best (indoor SR=0.47, outdoor SR=0.31), benefiting from LLM-based semantic understanding.

## Limitations & Future Work

1. **Capture Frame Rate Limitation**: Current hardware is limited to 1 FPS, resulting in insufficient viewpoint sampling density that affects rendering quality in complex scenes.
2. **Static Scene Assumption**: Dynamic urban elements (pedestrians, vehicles, traffic patterns) are not modeled; future work should integrate behavior prediction and interactive simulation.

## Highlights & Insights

1. **Scalability Trade-off of LiDAR Dependency**: The framework relies on specialized hardware (MetaCam ≈ commercial 3D scanner), constraining dataset scale by capture cost. This contrasts with the GaussGym approach of building environments at zero cost from large-scale online video—the two directions may ultimately converge toward a hybrid strategy where LiDAR provides a small number of high-quality anchor scenes for calibration and evaluation, while vision-based methods scale coverage broadly.
2. **Quantifying the Necessity of Geometric Grounding**: The paper provides a rare quantitative analysis of how simulation geometry quality affects downstream navigation policies—RL training in geometrically unreliable environments not only fails but degrades models, a finding with important implications for the broader sim-to-real community.
3. **Connection to NeRF/3DGS Scene Understanding**: Wanderland's multi-sensor data is naturally suited for evaluating recent feed-forward 3D reconstruction models (DUSt3R, VGGT, etc.), and its value as a GT benchmark may extend well beyond navigation simulation.
4. **Dynamic Scene Extension**: The current static scene assumption is a significant limitation. Injecting dynamic agents via 4D Gaussian Splatting or video generation models is a promising next direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MotionAnymesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physicsgrounded_articulation_for_sim.md)
- [\[CVPR 2026\] OnlineHMR: Video-based Online World-Grounded Human Mesh Recovery](onlinehmr_video-based_online_world-grounded_human_mesh_recovery.md)
- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)
- [\[CVPR 2026\] PhysHead: Simulation-Ready Gaussian Head Avatars](physhead_simulation-ready_gaussian_head_avatars.md)
- [\[CVPR 2026\] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)

</div>

<!-- RELATED:END -->
