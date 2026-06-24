---
title: >-
  [Paper Note] 3D4D: An Interactive Editable 4D World Model via 3D Video Generation
description: >-
  [AAAI 2026][Video Generation][4D scene] This paper proposes 3D4D, an interactive 4D visualization framework integrating WebGL and Supersplat rendering. A four-module backend pipeline converts static images and text prompts into editable 4D scenes, while a VLM-guided foveated rendering strategy enables 60fps real-time interaction, achieving state-of-the-art performance on both CLIP Consistency and CLIP Score.
tags:
  - "AAAI 2026"
  - "Video Generation"
  - "4D scene"
  - "WebGL"
  - "Gaussian Splatting"
  - "foveated rendering"
  - "VLM-guided"
date: 2026-05-08
content_hash: 139e4a836891f5a0
---

# 3D4D: An Interactive Editable 4D World Model via 3D Video Generation

**Conference**: AAAI 2026
**arXiv**: [2511.08536](https://arxiv.org/abs/2511.08536)  
**Code**: [Project Page](https://yunhonghe1021.github.io/NOVA/)  
**Area**: 4D Scene Generation / Interactive Visualization
**Keywords**: 4D scene, WebGL, Gaussian Splatting, foveated rendering, VLM-guided

## TL;DR

This paper proposes 3D4D, an interactive 4D visualization framework integrating WebGL and Supersplat rendering. A four-module backend pipeline converts static images and text prompts into editable 4D scenes, while a VLM-guided foveated rendering strategy enables 60fps real-time interaction, achieving state-of-the-art performance on both CLIP Consistency and CLIP Score.

## Background & Motivation

**Background**: Advances in generative models and multimodal learning have made text-driven 4D content generation feasible; however, a significant gap remains between generating 4D content and interactively exploring it.

**Limitations of Prior Work**:
- Traditional WebGL frameworks are constrained by high computational cost, high latency, and poor scalability when handling real-time 4D rendering and fine-grained temporal navigation.
- Existing 4D generation systems (e.g., SV4D, 4D-fy) produce 4D content but do not support real-time interactive editing.
- No system seamlessly integrates high-performance rendering with user interaction capabilities.

**Key Challenge**: A fundamental tension exists between rendering quality and real-time interactivity in 4D scenes — high-quality rendering demands substantial computational resources, while interactivity requires low latency.

**Key Insight**: Drawing inspiration from human peripheral vision, the paper employs a VLM to identify semantically important regions and adaptively allocate rendering resources, thereby reducing computational overhead while maintaining perceptual quality.

## Method

### Overall Architecture

3D4D adopts a decoupled frontend–backend architecture. The backend consists of a four-module pipeline for 4D content generation (3D reconstruction → image-to-video synthesis → video frame decomposition → 4D scene generation), while the frontend provides real-time interactive rendering based on WebGL and Supersplat. User-provided images and text prompts are processed by the backend to produce a sequence of Gaussian Splat point clouds, which are then rendered sequentially by the frontend to form a continuous 4D video.

### Key Designs

**1. WebGL + Supersplat Interactive Frontend**

- **Function**: High-performance 4D environment visualization and real-time editing.
- **Design Motivation**: Standard WebGL does not support fine-grained temporal interaction and requires custom extensions.
- **Mechanism**:
    - The backend outputs multiple PLY-format Gaussian Splat point clouds; the frontend renders them sequentially or in a loop to produce continuous 4D video.
    - A custom timeline control is developed to allow users to adjust camera pose, playback speed, and frame rate.
    - Five selection tools — rectangular, brush, polygon, lasso, and sphere — are provided for precise scene editing.
    - Users can define keyframes, with camera trajectories automatically interpolated.
    - All interactions are synchronized with the backend via API.
- **Novelty**: Unlike 4D-fy and SV4D, which only output static 4D content, 3D4D supports real-time interactive editing.

**2. VLM-Guided Foveated Rendering**

- **Function**: Efficient rendering that preserves perceptual quality under constrained computational resources.
- **Design Motivation**: The human visual system is sensitive to regions of fixation but insensitive to the periphery, a property that can be exploited to reduce computational cost.
- **Mechanism**:
    - A VLM (e.g., Qwen2.5-VL) analyzes each frame to generate a semantic importance map identifying salient regions such as persons and moving objects.
    - WebGL shaders adaptively allocate rendering resources: full-precision rendering for focal regions and low-cost blurred shading for background areas.
    - Output is captured via framebuffer, temporally smoothed, and encoded client-side by the browser's MediaRecorder API into .webm/.mp4.
- **Novelty**: Unlike conventional foveated rendering, which relies on eye-tracking hardware or fixed heuristics, this approach is driven by VLM-based semantic understanding.

**3. Four-Module Backend Generation Pipeline**

- 3D scene reconstruction: reconstructs a 3D Gaussian Splat scene from a single image.
- Image-to-video synthesis: generates temporally coherent video frames.
- Video frame decomposition: decomposes video into individual frames.
- 4D scene generation: assembles multi-frame outputs into a temporally coherent 4D scene.

### Loss & Training

This work is a system-level contribution. The backend combines existing pretrained models (e.g., DreamGen) without introducing new training procedures. The frontend adopts a fully client-side pipeline, requiring no server-side rendering.

## Key Experimental Results

### Main Results

Evaluation metrics: CLIP Consistency (CC, cross-view consistency) and CLIP Score (CS, semantic alignment).

| Method | CLIP Consistency ↑ | CLIP Score ↑ |
|--------|-------------------|-------------|
| Text2Room | 24.50 | 0.9035 |
| LucidDreamer | 26.72 | 0.8972 |
| WonderJourney | 27.34 | 0.9544 |
| WonderWorld | 29.47 | 0.9948 |
| SV4D | 30.29 | 0.8856 |
| 4D-fy | 11.23 | 0.6147 |
| **3D4D (Ours)** | **30.40** | **0.9951** |

### Rendering Efficiency

| Method | FPS | Real-Time Interaction |
|--------|-----|-----------------------|
| 4D-fy | 16 | ✗ |
| SVD-4D | 40 | ✗ |
| **3D4D (Ours)** | **60** | **✓** |

### Key Findings
- 3D4D achieves state-of-the-art on both CC (30.40, marginally surpassing SV4D's 30.29) and CS (0.9951, with a substantial margin over all baselines).
- It is the only system supporting real-time interaction, achieving 60fps — 3.75× faster than 4D-fy.
- 4D-fy performs worst on both metrics (CC: 11.23, CS: 0.6147), indicating limited generation quality and consistency.
- SV4D achieves relatively high CC but low CS (0.8856), suggesting cross-view consistency without adequate semantic alignment.

## Highlights & Insights
- **System-level innovation**: 3D4D is the first end-to-end system integrating 4D content generation with real-time interactive editing, bridging the generation–interaction gap.
- **VLM-guided foveated rendering** is the key enabler of 60fps performance — through semantically aware adaptive resource allocation rather than naive resolution reduction.
- **Fully client-side pipeline**: Browser-based MediaRecorder handles real-time encoding, eliminating the need for server-side rendering and substantially lowering deployment barriers.

## Limitations & Future Work
- The paper is presented primarily as a system/demo; technical depth is limited — all four backend modules are based on existing methods, with no novel training procedure.
- Evaluation relies solely on CLIP-based metrics and FPS; user studies and perceptual quality assessments are absent.
- Ablation studies are missing: the specific contributions of foveated rendering to FPS and quality are not quantified.
- The applicability of the system to different scene types (indoor, outdoor, object-level) is not discussed.

## Related Work & Insights
- **vs. SV4D**: SV4D achieves a comparable CC (30.29 vs. 30.40) but does not support interaction; 3D4D holds an exclusive advantage in interactivity.
- **vs. 4D-fy**: 4D-fy is substantially inferior on both CC and CS, and operates at only 16fps without interaction, highlighting the limitations of pure generation pipelines.
- **vs. WonderWorld**: WonderWorld achieves a near-identical CS (0.9948 vs. 0.9951) but trails noticeably on CC (29.47 vs. 30.40) and does not support the temporal dimension.

## Rating
- Novelty: ⭐⭐⭐⭐ — Primarily a system integration contribution; no novel algorithms in individual modules; VLM-guided foveated rendering is the main highlight.
- Experimental Thoroughness: ⭐⭐⭐ — Only two tables; ablation studies and user evaluations are absent; evaluation dimensions are narrow.
- Writing Quality: ⭐⭐⭐⭐ — System architecture is described clearly, but the paper is relatively short and lacks technical depth.
- Value: ⭐⭐⭐⭐ — As the first interactive 4D editing system, it has practical application value; academic contribution is limited.

---

## Related Papers

- [\[CVPR 2026\] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](../../CVPR2026/video_generation/neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](../../CVPR2025/video_generation/world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[ICML 2025\] How Far is Video Generation from World Model: A Physical Law Perspective](../../ICML2025/video_generation/how_far_is_video_generation_from_world_model_a_physical_law_perspective.md)
- [\[CVPR 2025\] SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](../../CVPR2025/video_generation/saw_toward_a_surgical_action_world_model_via_controllable_and_scalable_video_gen.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Yume1.5: A Text-Controlled Interactive World Generation Model](../../CVPR2026/video_generation/yume15_a_text-controlled_interactive_world_generation_model.md)
- [\[CVPR 2026\] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](../../CVPR2026/video_generation/versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)
- [\[CVPR 2026\] SeeU: Seeing the Unseen World via 4D Dynamics-aware Generation](../../CVPR2026/video_generation/seeu_seeing_the_unseen_world_via_4d_dynamics-aware_generation.md)
- [\[CVPR 2026\] Stereo World Model: Camera-Guided Stereo Video Generation](../../CVPR2026/video_generation/stereo_world_model_camera-guided_stereo_video_generation.md)
- [\[ICLR 2026\] Vid2World: Crafting Video Diffusion Models to Interactive World Models](../../ICLR2026/video_generation/vid2world_crafting_video_diffusion_models_to_interactive_world_models.md)

</div>

<!-- RELATED:END -->
