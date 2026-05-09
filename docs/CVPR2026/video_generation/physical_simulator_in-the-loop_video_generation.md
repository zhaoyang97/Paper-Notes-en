---
title: >-
  [Paper Note] Physical Simulator In-the-Loop Video Generation
description: >-
  [CVPR 2026][Video Generation][physical simulator in-the-loop] This paper proposes PSIVG — the first training-free inference-time framework that embeds a physical simulator into the video diffusion generation loop. It reconstructs a 4D scene and object meshes from a template video, generates physically consistent trajectories via an MPM simulator, guides video generation using optical flow, and enforces texture consistency of moving objects through Test-Time Consistency Optimization (TTCO), achieving a user preference rate of 82.3%.
tags:
  - CVPR 2026
  - Video Generation
  - physical simulator in-the-loop
  - video diffusion model
  - MPM simulation
  - test-time optimization
  - physically consistent generation
date: 2026-05-08
content_hash: a726a98c0373f0f1
---

# Physical Simulator In-the-Loop Video Generation

**Conference**: CVPR 2026
**arXiv**: [2603.06408](https://arxiv.org/abs/2603.06408)
**Code**: [https://vcai.mpi-inf.mpg.de/projects/PSIVG](https://vcai.mpi-inf.mpg.de/projects/PSIVG)
**Area**: Video Generation / Physical Consistency
**Keywords**: physical simulator in-the-loop, video diffusion model, MPM simulation, test-time optimization, physically consistent generation

## TL;DR
This paper proposes PSIVG — the first training-free inference-time framework that embeds a physical simulator into the video diffusion generation loop. It reconstructs a 4D scene and object meshes from a template video, generates physically consistent trajectories via an MPM simulator, guides video generation using optical flow, and enforces texture consistency of moving objects through Test-Time Consistency Optimization (TTCO), achieving a user preference rate of 82.3%.

## Background & Motivation

**State of the Field**: Diffusion-based video generation models (e.g., CogVideoX, HunyuanVideo) have achieved impressive visual realism, yet frequently violate fundamental physical laws such as gravity, inertia, and collision — objects disappear arbitrarily, motion trajectories are unreasonable, and physical interactions are implausible.

**Limitations of Prior Work**: (1) Modern video generation models are trained on denoising/reconstruction objectives, which optimize pixel- or patch-level reconstruction and lack explicit physical constraint mechanisms. (2) Early physics-aware methods couple 2D rigid-body simulators with image generators but are constrained by simplified 2D assumptions. (3) Methods such as PhysAnimator focus on 2D mesh simulation for cartoon animation, while PhysGen3D requires an input image for 3D reconstruction. (4) LLM-based prompting approaches are orthogonal explorations that do not directly impose physical constraints within the generator.

**Root Cause**: The training objectives of video diffusion models (denoising/reconstruction) contain no physical constraints and provide no mechanism to enforce the learning of physical laws. Achieving physical consistency while preserving visual quality requires introducing physics-based guidance into the generation process.

**Paper Goals**: How can information from a physical simulator be effectively integrated into the video diffusion process to achieve physically consistent video generation?

**Starting Point**: The paper proposes a *simulation-in-the-loop* paradigm, in which a physical simulator acts as a physics-aware constraint that guides the model to maintain spatiotemporal consistency within the diffusion generation loop.

**Core Idea**: A pretrained video model first generates a template video; a 4D scene and object meshes are then reconstructed from it and fed into a physical simulator; the physically consistent trajectories output by the simulator guide video re-generation; and test-time optimization further improves texture consistency of moving objects.

## Method

### Overall Architecture
PSIVG is a multi-stage pipeline: (1) A pretrained video generator produces a **template video** from a text prompt, providing scene composition, camera motion, and object appearance, albeit with physical inconsistencies. (2) A **perception pipeline** extracts 3D foreground object meshes, a 4D background scene reconstruction, and camera trajectories from the template video. (3) The scene is initialized in an **MPM physical simulator** for forward simulation to obtain physically consistent trajectories. (4) The simulator's rendered outputs (RGB, segmentation masks, pixel correspondences) serve as guidance signals to condition video generation via optical flow. (5) Optionally, **Test-Time Consistency Optimization (TTCO)** further improves texture consistency of moving objects.

### Key Designs

1. **Perception Pipeline (from Template Video to Simulator Assets)**

    - **Function**: Converts the 2D generated video into 3D assets usable by the simulator.
    - **Mechanism**: Information is extracted through three parallel branches. (a) *Foreground object geometry*: SAM/GroundedSAM detects and segments dynamic objects; a cropped object image from the first frame is fed into InstantMesh for single-image 3D mesh reconstruction (more reliable than multi-frame reconstruction due to geometric inconsistencies across generated video frames). (b) *Background scene geometry*: ViPE (with foreground masked out) performs 4D reconstruction; bundle adjustment recovers camera trajectories; per-frame metric depth point maps are aggregated into a global 3D background point cloud. (c) *Foreground object dynamics*: Two keyframes are selected; linear velocity is computed as 3D displacement divided by $\Delta t$; rotational velocity is estimated from 2D flow fields relative to the object centroid via SuperGlue feature matching.
    - **Design Motivation**: Although the template video is physically inconsistent, it encodes the overall scene composition. The perception pipeline bridges generated video and the physical simulator, serving as the critical link for realizing the simulation-in-the-loop paradigm.

2. **Physical Simulator Scene Initialization**

    - **Function**: Reproduces the template video's scene within the MPM simulator.
    - **Mechanism**: (a) *Simulation domain definition*: The simulation cube $[0,2]^3$ is determined by enclosing the foreground dynamics and background geometry using a spatial offset factor, establishing the metric-to-simulation scaling ratio. (b) *Physical property estimation*: GPT-4 infers physical properties such as density and Young's modulus from the first frame of the template video, using a hierarchical prompting strategy — first querying material composition, elastic characteristics, and surface roughness, then mapping these to numerical physical parameters, which is more reliable than directly estimating numerical values. (c) *Simulation and rendering*: MPM forward simulation yields high-resolution particle trajectories, which are rendered by Mitsuba into RGB frames, segmentation masks, and pixel correspondences.
    - **Design Motivation**: Although simulator renderings lack photorealism (artificial style, missing lighting, potential mesh artifacts), they faithfully encode physically correct motion information, which is sufficient as a guidance signal.

3. **Physically Consistent Video Generation (Optical Flow Conditioning)**

    - **Function**: Uses simulator outputs to guide the video diffusion model toward physically consistent generation.
    - **Mechanism**: Go-with-the-Flow (GwtF) is used for optical-flow-conditioned video generation. A blended optical flow is computed: foreground flow is derived from simulator-rendered RGB (ensuring physically consistent motion), while background flow is taken from the template video (preserving scene motion and camera dynamics); the two are fused via segmentation masks. The optical flow is used to warp noise latents as model input.
    - **Design Motivation**: Directly conditioning on simulator outputs is insufficient due to poor visual quality. Optical flow conditioning jointly encodes trajectory and rotation information and facilitates straightforward modeling of camera motion.

4. **TTCO: Test-Time Consistency Optimization**

    - **Function**: Optimizes learnable parameters at test time to enforce inter-frame texture consistency of moving objects in the generated video.
    - **Mechanism**: The first frame of the template video $\hat{I}_1$ is warped to each subsequent frame using pixel-to-pixel correspondences from the simulator, serving as a texture-consistent target. Learnable zero-initialized embeddings — added to text tokens corresponding to foreground objects and used for feature modulation in DiT layers — are optimized so that generated video pixels follow the simulator's foreground motion:

$$\mathcal{L}_{\text{TTCO}} = \sum_t \sum_j \|[De(h_0(\hat{L}_\tau))]_{q_{t,j}} - [W_t(\hat{I}_1)]_{q_{t,j}}\|_2^2$$

Optimization focuses on earlier (noisier) diffusion timesteps (700–1000) to guide texture generation and converges within 50 iterations.
    - **Design Motivation**: Optical flow conditioning only guides motion direction and does not guarantee texture consistency (flickering may occur under rotation or occlusion). Optimizing foreground text tokens achieves localized adaptation — affecting only foreground objects without degrading the background.

### Loss & Training
PSIVG requires no additional training data. TTCO uses the AdamW optimizer at test time with a learning rate of 2e-4 for 50 iterations. Template videos are generated using SD3 for image synthesis followed by CogVideoX-I2V-5B or HunyuanVideo-I2V for video generation.

## Key Experimental Results

### Main Results

| Method Type | Method | SAM mIoU↑ | Corr. Pixel MSE↓ | CLIP Text↑ | Subj. Consis.↑ |
|-------------|--------|-----------|-----------------|------------|----------------|
| Text-based | CogVideoX | 0.47 | 0.032 | 0.34 | 0.93 |
| Text-based | HunyuanVideo | 0.46 | 0.017 | 0.35 | 0.95 |
| Physics | PISA-Seg | 0.50 | 0.012 | 0.35 | 0.95 |
| Controllable | SG-I2V | 0.75 | 0.021 | 0.34 | 0.95 |
| Controllable | MotionClone | 0.68 | 0.019 | 0.35 | 0.87 |
| **Ours** | **PSIVG** | **0.84** | **0.007** | **0.35** | **0.95** |

### User Study

| Method | Preference Rate (%) |
|--------|---------------------|
| CogVideoX | 7.2 |
| HunyuanVideo | 4.5 |
| PISA-Seg | 2.6 |
| SG-I2V | 2.5 |
| MotionClone | 0.9 |
| **PSIVG (Ours)** | **82.3** |

32 participants unanimously rated PSIVG-generated videos as the most physically plausible.

### Ablation Study

| Configuration | SAM mIoU↑ | Corr. Pixel MSE↓ | Subj. Consis.↑ |
|---------------|-----------|-----------------|----------------|
| w/o TTCO | 0.82 | 0.009 | 0.93 |
| **w/ TTCO** | **0.84** | **0.007** | **0.95** |

### Key Findings
- **PSIVG achieves top performance across all motion controllability metrics** — SAM mIoU of 0.84 (0.09 above the second-best, SG-I2V) and Corr. Pixel MSE of only 0.007 (lowest among all methods).
- Methods such as PISA-Seg exhibit high temporal stability metrics but generate nearly static videos (minimal inter-frame variation), lacking genuine motion.
- **The benefit of TTCO is primarily reflected in texture consistency** — Corr. Pixel MSE decreases from 0.009 to 0.007, and Subject Consistency improves from 0.93 to 0.95.
- Prompt-based optimization outperforms LoRA-based designs — LoRA frequently degrades background quality and introduces artifacts, whereas prompt optimization is more lightweight and spatially localized.
- Directly optimizing spatio-temporal tokens (rather than text tokens) introduces grid-like artifacts.

## Highlights & Insights
- The **simulation-in-the-loop paradigm** is the primary contribution — physical constraints are introduced at inference time without modifying the generation model or requiring additional training. This decoupled design enables plug-and-play integration with arbitrary video generation models.
- **The perception pipeline's design is noteworthy**: single-frame 3D reconstruction via InstantMesh (rather than multi-frame reconstruction) is adopted because geometric inconsistencies across generated video frames make multi-frame reconstruction unreliable — reflecting a deep understanding of the characteristics of generated video.
- **GPT-based hierarchical physical property estimation** — inferring material descriptions (composition, elasticity, roughness) before mapping to numerical physical parameters — is more reliable than directly prompting LLMs for numerical values. This coarse-to-fine LLM usage paradigm generalizes to other scenarios requiring visual estimation of physical quantities.
- **The finding that "text tokens = object control"**: modifying the text embeddings corresponding to foreground objects primarily affects object appearance without disrupting the background, consistent with findings from other diffusion research and further confirming the spatial correspondence of text tokens.

## Limitations & Future Work
- Reliance on the MPM simulator precludes handling of complex agents (humans, vehicles) and articulated structures.
- The quality of initial object reconstruction in the perception pipeline directly impacts downstream stages — reconstruction errors propagate into simulation and generation.
- The method inherits limitations of the GwtF video model, making it difficult to generate very small or thin objects.
- The overall pipeline is substantially more complex than end-to-end approaches (template video → perception → simulation → re-generation → TTCO), resulting in higher latency.
- Only object interactions describable by rigid bodies or material point models are supported; complex materials such as fluids and cloth are not handled.

## Related Work & Insights
- **vs. PhysAnimator**: PhysAnimator targets 2D cartoon animation using 2D mesh simulation; PSIVG operates in 3D and enables training-free open-vocabulary video generation.
- **vs. PhysGen3D**: PhysGen3D obtains a 3D representation from an input image, runs MPM simulation, and renders directly; PSIVG additionally employs a video diffusion model to compensate for the visual shortcomings of simulator rendering (low resolution, absent lighting, unnatural style).
- **vs. WonderPlay**: WonderPlay first generates a 3DGS surfel scene and then updates it with video supervision; PSIVG directly performs video refinement via TTCO, which is simpler and more efficient.
- **vs. PISA**: PISA fine-tunes a diffusion model to learn physical interactions, requiring large amounts of training data; PSIVG is entirely training-free.
- **vs. Phantom**: Phantom internalizes physical reasoning into the model (requiring training), whereas PSIVG injects physical constraints externally at inference time (requiring no training); the two approaches are complementary.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The first training-free framework to embed a 3D physical simulator into a text-to-video diffusion pipeline; the TTCO design is also creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive quantitative comparisons; the user study (82.3% preference rate) is highly convincing; ablations cover key components.
- **Writing Quality**: ⭐⭐⭐⭐ — The method pipeline is clearly described and easy to follow, with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Proposes a general paradigm that can be plugged into any video generation model; practical applicability is nonetheless limited by pipeline complexity and MPM constraints.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Phantom: Physics-Infused Video Generation via Joint Modeling of Visual and Latent Physical Dynamics](phantom_physics-infused_video_generation_via_joint_modeling_of_visual_and_latent.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] Generative Neural Video Compression via Video Diffusion Prior](generative_neural_video_compression_via_video_diffusion_prior.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[CVPR 2026\] DriveLaW: Unifying Planning and Video Generation in a Latent Driving World](drivelaw_unifying_planning_and_video_generation_in_a_latent_driving_world.md)

<!-- RELATED:END -->
