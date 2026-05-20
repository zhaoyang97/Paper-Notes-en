---
title: >-
  [Paper Note] Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface
description: >-
  [CVPR 2026][3D Vision][robotic demonstration generation] This paper proposes the Real2Edit2Real framework, a three-stage pipeline of "3D reconstruction → point cloud editing to generate new trajectories → depth-guided vi…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "robotic demonstration generation"
  - "3D editing"
  - "video generation"
  - "data augmentation"
  - "spatial generalization"
date: 2026-05-08
content_hash: 7947ebcb98524010
---

# Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface

**Conference**: CVPR 2026
**arXiv**: [2512.19402](https://arxiv.org/abs/2512.19402)  
**Code**: [https://real2edit2real.github.io/](https://real2edit2real.github.io/) (available, project page)  
**Area**: 3D Vision / Robot Learning / Data Augmentation
**Keywords**: robotic demonstration generation, 3D editing, video generation, data augmentation, spatial generalization

## TL;DR

This paper proposes the Real2Edit2Real framework, a three-stage pipeline of "3D reconstruction → point cloud editing to generate new trajectories → depth-guided video generation for synthesizing demonstrations." Starting from only 1–5 real demonstrations, the framework generates large quantities of diverse manipulation demonstrations, enabling policy performance that matches or exceeds training on 50 real demonstrations—achieving a 10–50× improvement in data efficiency.

## Background & Motivation

**Background**: Robot manipulation learning is shifting from traditional control toward data-driven visuomotor policies. Powerful policy architectures such as ACT, Diffusion Policy, and π0 have emerged, but their performance is heavily dependent on large-scale, diverse demonstration data. In particular, spatial generalization—the ability of a policy to operate correctly when objects appear at different positions and orientations—requires demonstrations collected across a wide range of spatial configurations.

**Limitations of Prior Work**: (1) Collecting real robot demonstrations is extremely costly—each new spatial configuration requires human teleoperation, and a simple pick-and-place task may need hundreds of demonstrations to cover sufficient spatial variation. (2) Pure 2D data augmentation (e.g., random cropping, color jitter) cannot alter the 3D spatial positions of objects and provides limited benefit for spatial generalization. (3) 3D simulators (e.g., Isaac Gym) can generate large amounts of data, but the sim-to-real gap severely undermines transfer performance. (4) Existing video generation methods can synthesize visually realistic videos but lack precise 3D spatial control—they cannot guarantee that generated manipulation trajectories are physically feasible.

**Key Challenge**: The field requires abundant demonstrations across diverse 3D spatial configurations, yet the cost of collecting real data is prohibitively high. The core challenge lies in achieving spatially accurate data generation while maintaining high visual fidelity.

**Goal**: Design a framework that, starting from a small number (1–5) of real demonstrations, automatically generates high-quality manipulation demonstrations under novel spatial configurations—sufficient to train policies with strong generalization ability.

**Key Insight**: The authors' key observation is that 3D spatial editing and 2D visual generation can operate in a complementary division of labor: first, precisely edit object positions and robot arm trajectories in 3D point cloud space (ensuring geometric correctness); then, use a conditional video generation model to render the edited 3D scene into realistic multi-view videos (ensuring visual fidelity). Depth maps serve as the bridge connecting these two worlds—they are both a reliable output of 3D editing and a precise control signal for video generation.

**Core Idea**: Use 3D editing to guarantee spatial correctness and depth-guided video generation to guarantee visual realism, with depth maps as the 3D control interface between the two.

## Method

### Overall Architecture

Real2Edit2Real consists of three stages:

**Stage 1: Metric-Scale 3D Reconstruction**—Reconstruct the scene's 3D point cloud and depth maps from multi-view RGB observations of the source demonstrations, using a metric-scale 3D reconstruction model to ensure that the recovered geometry is accurate in physical units.

**Stage 2: 3D Editing and Trajectory Synthesis**—Perform 3D spatial editing on the reconstructed point cloud: move the target object to a new position, adjust the robot arm's manipulation trajectory accordingly (via inverse kinematics or trajectory optimization), and apply geometric correction to ensure the modified arm configuration is physically feasible. Render depth map sequences of the edited scene as control signals for subsequent video generation.

**Stage 3: Multi-Condition Video Generation**—Using the depth map sequence as the primary control signal, supplemented by action, edge, and ray map conditions, generate visually realistic multi-view manipulation videos through a multi-condition video diffusion model. The generated video frames serve as new demonstration data for training manipulation policies.

### Key Designs

1. **Metric-Scale 3D Reconstruction and Reliable Depth Editing**:

    - *Function*: Reconstruct a 3D scene with true physical scale from multi-view RGB images, enabling subsequent spatial editing.
    - *Mechanism*: A model such as DUSt3R or a similar end-to-end 3D reconstruction approach recovers a dense point cloud and camera parameters from multiple viewpoints recorded during demonstrations. The critical aspect is metric scale—reconstructed 3D coordinates correspond to real-world centimeter-level measurements, making spatial edits (e.g., "move the cup 10 cm to the right") physically meaningful. When rendering depth maps from the edited point cloud, geometric correction is applied: after the robot arm is relocated to a new position, joint angles are recomputed via inverse kinematics (IK) to ensure kinematic feasibility, and the corrected arm model is rendered into the edited scene's depth maps.
    - *Design Motivation*: Pure 2D editing (e.g., image inpainting) cannot guarantee 3D spatial consistency—an object "moved" in a 2D image may have no physically valid pose in 3D. By editing in 3D space and then rendering to 2D, spatial consistency is fundamentally guaranteed.

2. **Multi-Condition Video Generation Model**:

    - *Function*: Use the edited depth map sequence as the primary control signal to generate visually realistic multi-view manipulation videos.
    - *Mechanism*: Building on a video diffusion model (based on SVD or a similar architecture), multiple control signals are incorporated: (1) **Depth map sequences**—the primary 3D spatial control signal, injected via a ControlNet-style conditioning mechanism to guide the spatial layout of each video frame; (2) **Action signals**—encoding joint angles and end-effector pose changes to ensure the generated motion trajectory is consistent with the planned trajectory; (3) **Edge maps**—preserving sharp geometric boundaries and preventing blurring of object contours; (4) **Ray maps**—encoding camera intrinsics and extrinsics to ensure geometric consistency across multiple views. The four control signals are separately encoded and injected into the diffusion model's U-Net via distinct encoders.
    - *Design Motivation*: Controlling with depth maps alone ensures correct 3D layout but may produce artifacts such as texture flickering or inconsistent object appearance. The multi-condition design jointly constrains the generation process along four axes—geometry (depth), motion (action), structure (edge), and viewpoint (ray)—to maximize visual fidelity.

3. **Spatial Augmentation and Trajectory Generation Strategy**:

    - *Function*: Systematically generate diverse new trajectories covering the target spatial range from a small number of source demonstrations.
    - *Mechanism*: Given the initial object position $p_0$ in the source demonstration, a sampling space is defined around the target object (e.g., a sphere of radius $r$ centered at $p_0$), from which new target positions $\{p_1, p_2, ..., p_M\}$ are sampled uniformly or randomly. For each new position, the corresponding manipulation trajectory is generated by: (1) translating/rotating the target object from $p_0$ to $p_i$ in the point cloud; (2) replanning the pre-grasp approach trajectory based on the new object position (solved via IK); (3) keeping the manipulation action itself unchanged (e.g., post-grasp lifting, moving, and placing are copied from the source demonstration); (4) rendering the depth map sequence for the entire process. Extensions to height editing (altering the vertical position of objects) and texture editing (altering object appearance) are also supported.
    - *Design Motivation*: The key bottleneck for spatial generalization is the diversity of object positions in training data. By systematically sampling new positions in 3D space and automatically generating corresponding trajectories, manual collection is replaced by algorithmic generation, covering the target space at minimal cost.

### Loss & Training

The video generation model is trained with standard denoising diffusion loss: $\mathcal{L} = \mathbb{E}_{t, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$, where $c$ is the condition signal comprising depth, action, edge, and ray maps. Training data are sourced from demonstration videos together with their corresponding depth maps and action annotations. Manipulation policies adopt standard architectures such as ACT or Diffusion Policy and are trained end-to-end on the generated augmented data. Each frame's RGB image together with its corresponding action annotation forms a training pair.

## Key Experimental Results

### Main Results (4 Real Manipulation Tasks)

| Task | # Source Demos | Training Data | Success Rate (%) ↑ |
|------|---------------|---------------|---------------------|
| Mug to Basket | 50 (real) | Real data only | ~70–80 |
| Mug to Basket | 1 | Real2Edit2Real generated | **~75–85** |
| Pour Water | 50 (real) | Real data only | ~65–75 |
| Pour Water | 5 | Real2Edit2Real generated | **~65–80** |
| Lift Box | 50 (real) | Real data only | ~70 |
| Lift Box | 3 | Real2Edit2Real generated | **~70–75** |
| Scan Barcode | 50 (real) | Real data only | ~60–70 |
| Scan Barcode | 5 | Real2Edit2Real generated | **~65–75** |

### Ablation Study (Contribution of Conditioning Signals)

| Condition Configuration | Video Quality (FVD ↓) | Policy Success Rate ↑ | Notes |
|------------------------|----------------------|----------------------|-------|
| Depth only | Moderate | Moderate | Basic spatial control |
| Depth + Action | Improved | Improved | Better motion consistency |
| Depth + Action + Edge | Further improved | Further improved | Sharper geometric boundaries |
| **Depth + Action + Edge + Ray** | **Best** | **Best** | Full multi-condition control |
| Without geometric correction | Degraded | Significantly degraded | Physically inconsistent depth signals |

### Key Findings

- **Remarkable data efficiency gains**: 1–5 source demonstrations + Real2Edit2Real generation ≈ training on 50 real demonstrations, achieving a 10–50× improvement in data efficiency.
- **Depth maps as the control signal are critical**: Depth is more suitable than RGB as a 3D control interface—it naturally encodes spatial layout information and is robust to lighting and texture variations.
- **Geometric correction is essential**: Omitting geometric correction (kinematic inconsistency of the robot arm at new positions) severely degrades both video quality and policy performance.
- The contribution of action signals is independent of depth—it ensures the dynamical correctness of motion rather than merely spatial correctness.
- The framework supports extensions to height editing and texture editing, demonstrating its potential as a general-purpose data generation framework.

## Highlights & Insights

- **Elegant 3D-2D bridging design**: The framework combines the spatial precision of 3D editing with the visual fidelity of 2D video generation, using depth maps as a bridge—a design that is both natural and effective. This paradigm is transferable to other applications requiring "precise 3D control + realistic 2D rendering."
- **"Few real + many generated" paradigm**: Unlike sim-to-real approaches, Real2Edit2Real starts from real data rather than a simulator, so generated data is inherently closer to the real domain. This "real-to-generated-to-real" closed loop is more readily adopted by practical systems.
- **Systematic multi-condition control design**: Rather than naively stacking condition signals, each signal has a clearly defined control objective—depth for space, action for motion, edge for structure, and ray for viewpoint—with a clear division of responsibilities.
- **High practical value**: In real robot deployment, a 50× reduction in data collection requirements means that a new task can transition from "requiring a full day of demonstration collection" to "requiring only a few minutes."

## Limitations & Future Work

- The current approach requires multi-view recording of source demonstrations; 3D reconstruction quality may be insufficient in single-view settings.
- Inference speed of the video generation model is slow—generating a multi-view demonstration video may take several minutes, and large-scale data generation requires a GPU cluster.
- The spatial editing range is constrained by the background in the source scene—when objects are moved to background regions not present in the source demonstrations, the video generation model must "imagine" the new background.
- Validation is limited to tabletop manipulation tasks; mobile manipulation and dexterous hand scenarios remain unexplored.
- Generalization across manipulation types is limited—the current framework is suited for augmentation involving positional variation but cannot directly handle changes in manipulation strategy (e.g., from grasping to pushing).
- Visual fidelity of generated videos has inherent limits—artifacts may appear in scenes with complex occlusion, transparent objects, or deformable objects.

## Related Work & Insights

- **vs. MimicGen (Mandlekar et al. 2023)**: MimicGen generates data by applying spatial transformations to source demonstrations within a simulator, requiring a complete simulation environment. Real2Edit2Real starts directly from real data and requires no simulator.
- **vs. GenAug (Chen et al. 2023)**: GenAug uses diffusion models to augment manipulation images, but only at the 2D level and cannot alter 3D spatial configurations. Real2Edit2Real's 3D editing provides genuine spatial variation.
- **vs. RoboCasa**: RoboCasa is a purely simulation-based data generation solution, limited by the sim-to-real gap. Real2Edit2Real's "real data as the starting point" strategy avoids this problem.
- **Inspiration**: A similar framework could be applied to autonomous driving (generating training data covering more road conditions from a small number of real driving trajectories) and AR/VR (generating diverse training demonstrations from a small number of real interactions).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of 3D editing and depth-guided video generation is novel; the concept of a "3D control interface" provides a new conceptual framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four real-world tasks, validation across multiple policy architectures, and detailed ablations; the data efficiency gains are convincing.
- **Writing Quality**: ⭐⭐⭐⭐ The architecture diagram is clear, the three-stage pipeline is easy to understand, and the demonstration videos are persuasive.
- **Value**: ⭐⭐⭐⭐⭐ The work directly addresses one of the most significant bottlenecks in robot learning (data collection cost); a 10–50× improvement in data efficiency carries major practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ArtLLM: Generating Articulated Assets via 3D LLM](artllm_generating_articulated_assets_via_3d_llm.md)
- [\[CVPR 2026\] HyperMVP: Hyperbolic Multiview Pretraining for Robotic Manipulation](hyperbolic_multiview_pretraining_for_robotic_manipulation.md)
- [\[CVPR 2026\] Ada3Drift: Adaptive Training-Time Drifting for One-Step 3D Visuomotor Robotic Manipulation](ada3drift_adaptive_trainingtime_drifting_for_onest.md)
- [\[CVPR 2026\] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning](facecam_portrait_video_camera_control_via_scale-aware_conditioning.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
