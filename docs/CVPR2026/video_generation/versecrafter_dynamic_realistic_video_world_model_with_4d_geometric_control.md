---
title: >-
  [Paper Note] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control
description: >-
  [CVPR 2026][Video Generation][Video World Model] This paper presents VerseCrafter, a video world model based on a unified 4D geometric control representation (static background point cloud + per-object 3D Gaussian trajec…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Video World Model"
  - "4D Geometric Control"
  - "3D Gaussian Trajectory"
  - "Camera and Object Motion"
  - "Video Diffusion Model"
date: 2026-05-08
content_hash: f6c527265fe5caef
---

# VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control

**Conference**: CVPR 2026
**arXiv**: [2601.05138](https://arxiv.org/abs/2601.05138)  
**Code**: [https://sixiaozheng.github.io/VerseCrafter_page/](https://sixiaozheng.github.io/VerseCrafter_page/)  
**Area**: 3D Vision / Video Generation
**Keywords**: Video World Model, 4D Geometric Control, 3D Gaussian Trajectory, Camera and Object Motion, Video Diffusion Model

## TL;DR
This paper presents VerseCrafter, a video world model based on a unified 4D geometric control representation (static background point cloud + per-object 3D Gaussian trajectories). A lightweight GeoAdapter injects 4D control signals into a frozen Wan2.1-14B video diffusion model, enabling precise and disentangled control over camera and multi-object motion. The authors also construct VerseControl4D, a real-world dataset containing 35K training samples.

## Background & Motivation

1. **Background**: Video world models aim to simulate dynamic real-world environments. Recent approaches condition video generation on text, actions, or camera trajectories. Camera control methods such as CameraCtrl and MotionCtrl achieve viewpoint control via Plücker embeddings or 3D prior injection. Object motion control has primarily relied on 2D cues including trajectory points, optical flow, masks, and 2D bounding boxes.

2. **Limitations of Prior Work**: (a) 2D control signals are not robust under large viewpoint changes and lack 3D awareness; (b) more advanced 3D signals such as 3D bounding boxes are overly rigid, SMPL-X is limited to human bodies, and sparse 3D trajectories are often noisy and incomplete; (c) existing control spaces are fragmented—camera and object motion are not defined in a unified coordinate system and cannot be jointly controlled.

3. **Key Challenge**: An ideal world model should simulate the complete 4D spatiotemporal space, yet videos only capture 2D projections. A compact, editable, and category-agnostic 4D geometric state representation is needed to unify camera and multi-object motion control.

4. **Goal**: (a) Design a unified 4D geometric control representation; (b) achieve disentangled control of camera and multi-object motion in a shared world coordinate system; (c) construct large-scale training data.

5. **Key Insight**: 3D Gaussian distributions are used to describe the probabilistic 3D occupancy of objects—the mean defines the motion path, and the covariance captures spatial extent and orientation—naturally supporting soft, flexible, and category-agnostic object modeling.

6. **Core Idea**: A unified 4D geometric state is constructed from a static background point cloud and per-object 3D Gaussian trajectories in a shared world coordinate system. These are rendered into multi-channel control maps and fed through a GeoAdapter to drive a frozen video diffusion model.

## Method

### Overall Architecture
**Input**: A single reference image and a text prompt. The pipeline first estimates depth and camera intrinsics (MoGe-2) and obtains user-specified object masks (Grounded SAM2) to construct the 4D geometric control (background point cloud + 3D Gaussian trajectories). The user specifies camera trajectories in the shared world coordinate system. All control signals are rendered into per-frame 4D control maps (background RGB/depth + trajectory RGB/depth + soft fusion mask). The control maps are encoded by the Wan Encoder and fed into the GeoAdapter, which injects geometric features into the frozen Wan2.1-14B DiT backbone via residual modulation, conditioned jointly with umT5 text embeddings to generate the output video.

### Key Designs

1. **4D Geometric Control Representation**:
    - **Function**: Unified representation of static scene structure and dynamic objects in a shared world coordinate system.
    - **Mechanism**: The background point cloud $P^{\text{bg}}$ is obtained by back-projecting non-object pixels of the input image into 3D. Each object's 3D Gaussian trajectory $\{\mathcal{G}_o^t\}_{t=1}^T$ consists of a mean $\boldsymbol{\mu}_o^t$ (position) and covariance $\mathbf{\Sigma}_o^t$ (shape/orientation), initialized by fitting a full-covariance Gaussian to the object point cloud. Users can visualize the Gaussians as ellipsoids in Blender and specify trajectories via drag-and-drop keyframing.
    - **Design Motivation**: Compared to 3D bounding boxes (overly rigid), SMPL-X (human-only), and sparse trajectory points (no shape information), 3D Gaussians softly encode object occupancy in a probabilistic manner, are category-agnostic, and admit low-dimensional editing.

2. **4D Control Map Rendering**:
    - **Function**: Convert the 4D geometric state into model-consumable control signals.
    - **Mechanism**: Three types of maps are rendered per frame: (i) background RGB/depth—project $P^{\text{bg}}$ under the target camera; (ii) trajectory RGB/depth—project per-object Gaussians as soft elliptical footprints; (iii) soft fusion mask—derived by inverting background visibility and merging Gaussian footprints, indicating regions where the diffusion model should synthesize or inpaint content. The first frame retains the original input image. Background and trajectory maps are rendered through decoupled channels, separating camera motion from object motion.
    - **Design Motivation**: Decoupled rendering ensures control maps do not interfere with each other—background changes arise solely from camera motion while trajectory changes arise solely from object motion—while maintaining geometric consistency.

3. **GeoAdapter Architecture**:
    - **Function**: Inject 4D geometric control signals into the frozen video diffusion model.
    - **Mechanism**: Four groups of RGB/depth control maps are encoded by a frozen Wan Encoder. Soft fusion masks are reshaped to the latent resolution. All geometric latents are concatenated channel-wise to form a spatiotemporal geometry tensor. The GeoAdapter is a lightweight DiT branch that shares the hidden dimension of the Wan-DiT but has far fewer layers. Each GeoAdapter block is paired with every $k=5$ DiT blocks; its output is projected through a zero-initialized linear layer and added as a residual to the corresponding DiT block. GeoAdapter blocks are initialized from the weights of their paired DiT blocks to stabilize training.
    - **Design Motivation**: The adapter design introduces only a small number of additional parameters while keeping the backbone frozen, inheriting the strong video prior of Wan2.1. Zero initialization ensures the backbone is not perturbed during the early stages of training.

4. **VerseControl4D Dataset Construction**:
    - **Function**: Provide large-scale 4D geometric control annotations for real-world videos.
    - **Mechanism**: 81-frame clips are extracted from Sekai-Real-HQ and SpatialVID-HQ, filtered by object criteria (1–6 controllable objects detected by Grounded SAM2) and quality criteria (aesthetic/brightness scores). Text descriptions are generated by Qwen2.5-VL-72B. Depth and camera trajectories are estimated using MoGe-2, UniDepth V2, and MegaSAM; 3D point clouds are reconstructed and Gaussian trajectories are fitted; control maps are rendered. The final dataset contains 35K training samples and 1K validation samples.
    - **Design Motivation**: Addresses the bottleneck of scarce 4D control data; the fully automated pipeline enables large-scale training.

### Loss & Training
- Optimizer: Adam, learning rate 2e-5, constant schedule with 100-step warmup
- Two-stage training: 2,500 steps at 480P followed by 2,500 fine-tuning steps at 720P
- Classifier-free guidance (CFG) training: text condition dropped with probability 0.1
- Inference: 50 denoising steps, CFG scale 5.0
- Hardware: 16 × 96GB GPUs, total training time approximately 380 hours

## Key Experimental Results

### Main Results (Joint Camera + Object Motion Control)

| Method | Overall Score↑ | Imaging Quality↑ | RotErr↓ | TransErr↓ | ObjMC↓ |
|---|---|---|---|---|---|
| Perception-as-Control | 83.66 | 66.81 | 5.006 | 8.767 | 6.556 |
| Yume | 85.47 | 71.16 | 7.560 | 8.735 | 7.959 |
| Uni3C | 83.55 | 68.06 | 1.361 | 7.731 | 5.883 |
| **VerseCrafter** | **88.10** | **72.70** | **0.890** | **3.103** | **2.507** |

### Ablation Study (3D Representation and Control Design)

| Configuration | Overall Score | RotErr | TransErr | ObjMC |
|---|---|---|---|---|
| Full (3D Gaussian) | **88.10** | **0.890** | **3.103** | **2.507** |
| 3D Bounding Box | 85.45 | 1.350 | 3.805 | 4.520 |
| 3D Point Trajectory | 85.57 | 1.298 | 3.281 | 6.896 |
| w/o depth | 85.64 | 1.177 | 3.900 | 4.929 |
| BG & FG Merged | 85.72 | 1.080 | 3.803 | 3.726 |

### Key Findings
- **3D Gaussian trajectories consistently outperform 3D bounding boxes and point trajectories**: ObjMC drops from 4.520 and 6.896 to 2.507, respectively, as Gaussians encode shape and orientation information. Point trajectories yield the worst ObjMC (6.896) because object size cannot be encoded.
- **Depth information is critical**: Removing depth causes incorrect foreground–background ordering (e.g., a lamppost pulled in front of a building), raising TransErr from 3.103 to 3.900.
- **Disentangled control outperforms merged control**: Merging background and foreground control maps significantly degrades object motion accuracy (ObjMC rises from 2.507 to 3.726) because the model cannot distinguish camera motion from object motion.
- **Static-scene camera control**: VerseCrafter achieves substantially lower RotErr (0.650) and TransErr (2.587) than FlashWorld (1.792 / 3.257) and ViewCrafter (2.101 / 9.868), demonstrating the precision of 4D geometric control.

## Highlights & Insights
- **3D Gaussian trajectories as a universal object motion representation**: Replacing rigid geometric primitives with probability distributions reconciles shape encoding with flexibility, and naturally handles objects of arbitrary categories. This representation is transferable to domains such as autonomous driving and robotics that require object motion prediction.
- **Elegant decoupled rendering design**: Camera motion (background variation) and object motion (foreground variation) are separated into independent control channels, allowing the model to learn the two motion modes independently without confusion. This principle is broadly applicable to any generative model with multi-signal conditioning.
- **Zero-initialization + weight inheritance adapter training strategy**: GeoAdapter blocks are initialized from their paired DiT block weights, and outputs are injected through zero-initialized linear layers, ensuring training stability while enabling rapid task adaptation.

## Limitations & Future Work
- **High inference cost**: Generating a single 81-frame 720P video requires approximately 1,152 seconds on 8 × 96GB GPUs, far from real-time applicability.
- **Dependence on monocular depth estimation**: Background point clouds and initial Gaussians are reconstructed from a single image; occluded regions may be missing under large-baseline viewpoint changes.
- **Dataset primarily covers outdoor/urban scenes**: VerseControl4D is sourced from Sekai-Real-HQ and SpatialVID-HQ, and coverage of complex indoor scenes may be insufficient.
- **Object interactions are not modeled**: Physical interactions among multiple objects—such as collisions and occlusions—are not explicitly constrained, potentially leading to unrealistic interpenetration artifacts.

## Related Work & Insights
- **vs. Yume**: Yume controls 4D generation via text/action tokens but lacks precise camera and object motion control (RotErr = 7.560 vs. 0.890). VerseCrafter replaces implicit control with an explicit geometric state, yielding substantially higher accuracy.
- **vs. Uni3C**: Uni3C uses SMPL-X to control object motion, limiting it to human bodies and single-person scenarios. VerseCrafter's 3D Gaussian trajectories are category-agnostic and support multiple objects simultaneously.
- **vs. ControlNet**: The GeoAdapter draws inspiration from ControlNet's adapter-style injection but extends it to 4D spatiotemporal control with decoupled multi-channel control maps.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The 4D geometric control representation and 3D Gaussian trajectories as motion control signals constitute an entirely novel design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Joint control, camera control, and ablation experiments are comprehensive, with thorough quantitative and qualitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear, though notation density is high.
- Value: ⭐⭐⭐⭐⭐ Provides a unified 4D control interface for video world models; both the dataset and the method have high reuse value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation](../../AAAI2026/video_generation/3d4d_an_interactive_editable_4d_world_model_via_3d_video_generation.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] Towards Realistic and Consistent Orbital Video Generation via 3D Foundation Priors](orbital_video_3d_foundation_priors.md)
- [\[CVPR 2026\] Diff4Splat: Repurposing Video Diffusion Models for Dynamic Scene Generation](diff4splat_controllable_4d_scene_generation_with_latent_dynamic_reconstruction_m.md)
- [\[CVPR 2026\] From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning](from_static_to_dynamic_exploring_self-supervised_image-to-video_representation_t.md)

</div>

<!-- RELATED:END -->
