---
title: >-
  [Paper Note] Lighting-grounded Video Generation with Renderer-based Agent Reasoning
description: >-
  [CVPR 2026][Video Generation][Lighting-controllable video generation] LiVER proposes a lighting-driven video generation framework that employs a renderer-based agent to translate textual descriptions into explicit 3D sce…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "Lighting-controllable video generation"
  - "3D scene agent"
  - "physical rendering"
  - "diffusion model"
  - "scene agent"
date: 2026-05-08
content_hash: 1e2b314127377e50
---

# Lighting-grounded Video Generation with Renderer-based Agent Reasoning

**Conference**: CVPR 2026
**arXiv**: [2604.07966](https://arxiv.org/abs/2604.07966)  
**Code**: None  
**Area**: 3D Vision / Video Generation
**Keywords**: Lighting-controllable video generation, 3D scene agent, physical rendering, diffusion model, scene agent

## TL;DR

LiVER proposes a lighting-driven video generation framework that employs a renderer-based agent to translate textual descriptions into explicit 3D scene proxies (encompassing layout, lighting, and camera trajectories). Physical rendering is then used to produce diffuse/glossy/rough GGX scene proxies, which are injected into a video diffusion model to achieve physically accurate lighting effects and precise scene control.

## Background & Motivation

Diffusion models have made remarkable progress in video generation, yet controllability remains a core bottleneck. Existing methods primarily improve visual quality through data-driven approaches but fall short in explicitly modeling scene factors such as layout, lighting, and camera trajectories. Although prior works have explored 3D-aware conditioning (e.g., CameraCtrl for camera control and MotionCtrl for object trajectories), these methods largely neglect physically accurate lighting modeling—effects such as shadows, reflections, and ambient occlusion remain unrealistic on real-world materials (skin, metal, glass).

The root cause lies in the fact that existing methods either focus on geometric/motion control while ignoring lighting, or entangle lighting with other attributes in a manner that precludes decoupled control. The paper's starting point is to explicitly disentangle the lighting properties of a scene via a physical renderer into controllable 2D rendering channels (diffuse, glossy GGX, rough GGX), thereby preserving the physical lighting information of the 3D scene while injecting it into the video diffusion model as image sequences.

**Core Idea**: Use a renderer agent to automatically convert text into a 3D scene → obtain lighting-aware scene proxies via PBR rendering → inject physical lighting signals into a video diffusion model through a lightweight encoder and three-stage training.

## Method

### Overall Architecture

The LiVER pipeline consists of three steps: (1) the renderer agent parses user text to extract object categories, spatial relationships, lighting cues, and camera motion, constructing a structured scene graph and retrieving 3D models from an asset library; (2) a physical renderer (Blender) renders scene proxies $y \in \mathbb{R}^{F \times 9 \times H \times W}$ from the 3D scene, HDR environment maps, and camera trajectories, comprising three RGB rendering channels; (3) a lightweight proxy encoder maps scene proxies into the video latent space and guides the Wan2.2-5B video diffusion model via residual injection to generate videos with physically accurate lighting.

### Key Designs

1. **Renderer Agent Reasoning System**:
    - Function: Converts high-level textual descriptions into structured 3D control signals.
    - Mechanism: Executed in three steps — a Scene Agent parses objects and spatial relationships to construct a scene graph $\mathcal{G}=(V,E)$ and retrieves assets from Objaverse-XL; a Lighting Agent selects HDR environment maps from the Poly Haven library based on textual lighting cues (e.g., "warm mood"); a Camera Agent parses motion semantics (e.g., "orbit," "dolly zoom") and generates smooth camera trajectories via spline interpolation.
    - Design Motivation: Automates the generation of 3D control signals to lower the barrier for end users, while supporting manual editing for professional use cases.

2. **Physical Lighting Scene Proxy**:
    - Function: Encodes 3D scene lighting information into 2D signals that can be injected into the diffusion model.
    - Mechanism: A PBR renderer generates three rendering channels — diffuse (low-frequency ambient), rough GGX (mid-frequency broad reflections), and glossy GGX (high-frequency specular highlights) — stacked into a 9-channel image sequence $y = [x^{\text{DIFF}}, x^{\text{GGX1}}, x^{\text{GGX2}}] = R(s^i, l^i, c^i)$.
    - Design Motivation: Directly using full 3D representations is too complex for video diffusion models, whereas 2D rendering channels preserve physical lighting information while remaining compatible with image processing pipelines.

3. **Lightweight Proxy Encoder and Adapter**:
    - Function: Aligns scene proxy features with the video latent space.
    - Mechanism: A 2D convolutional encoder downsamples the 9-channel input to the same resolution as the VAE latent $z^y \in \mathbb{R}^{F \times C \times H' \times W'}$, with residual injection via a zero-initialized learnable scalar $\alpha$: $z' = z + \alpha \cdot z^y$.
    - Design Motivation: Zero initialization ensures that the original generative capability is preserved at the start of training, while proxy features gradually guide the latent space toward lighting control.

### Loss & Training

A three-stage training scheme is adopted:
- **Stage 1 — Conditioning Pathway Training**: The video diffusion backbone is frozen; only the proxy encoder and adapter are trained (10 epochs) to learn coarse control signal conversion.
- **Stage 2 — Joint LoRA Fine-tuning**: LoRA layers in the backbone are unfrozen and jointly trained with the encoder/adapter (10 epochs) to refine semantic alignment.
- **Stage 3 — Lighting Diversity Expansion**: Joint training continues with a 1:1 mixture of real and synthetic data to improve generalization across diverse lighting phenomena.

The training loss follows the standard flow matching objective: $\mathcal{L} = \mathbb{E}_{z,\epsilon,t}[|u_\theta(z_t, y, c^{\text{txt}}, t) - v_t|^2]$

## Key Experimental Results

### Main Results

| Method | FVD ↓ | FID ↓ | CLIP ↑ | ATE ↓ | LE ↓ | mIoU ↑ |
|--------|-------|-------|--------|-------|------|--------|
| CameraCtrl | 48.03 | 98.29 | 28.75 | 2.15 | 0.06 | 0.68 |
| MotionCtrl | 63.13 | 97.21 | 26.67 | 3.42 | 0.07 | 0.66 |
| VideoFrom3D | 36.94 | 157.89 | 24.51 | 17.55 | 0.05 | 0.74 |
| **LiVER** | **32.56** | **129.56** | **30.97** | **2.48** | **0.04** | **0.87** |

In the 16-frame comparison against CameraCtrl/MotionCtrl, LiVER achieves FVD=32.45, FID=42.32, and CLIP=29.62, outperforming all baselines across the board.

### Ablation Study

| Configuration | Key Effect | Description |
|---------------|------------|-------------|
| w/o synthetic data | Uniform, erroneous lighting | Lack of dynamic lighting diversity causes overfitting to the limited lighting patterns in real data |
| w/o staged training | Nearly static outputs | Simultaneously learning control signals and adapting the pretrained model leads to optimization difficulties |
| Full model | Best performance | The three-stage scheme ensures stable convergence and high-quality generation |

### Key Findings

- Synthetic data is critical for lighting diversity: training on real data alone results in uniformly flat lighting.
- Staged training is essential for stable convergence: end-to-end training leads to nearly static generation results.
- A user study (25 participants × 20 groups) shows that LiVER outperforms competing methods across all four dimensions: video quality (83.4%), scene control (83.3%), camera control (72.1%), and lighting control (59.3%).

## Highlights & Insights

- Decomposing physically rendered lighting (diffuse/glossy/rough) into conditioning signals for video generation is an elegant design — it preserves physical meaning while remaining compatible with 2D processing pipelines.
- The renderer agent design enables both automated use and manual editing, catering to a spectrum of users ranging from general audiences to professional film and media production.
- The zero-initialized residual injection strategy is a mature engineering choice that ensures the pretrained model's generative capability is not compromised.

## Limitations & Future Work

- Initial 3D reconstruction is coarse; geometric detail and material quality are sensitive to the precision of textual descriptions.
- The quality and coverage of 3D asset retrieval are constrained by existing asset libraries.
- Only HDR environment maps are supported as global illumination; fine-grained control of local light sources is not yet available.
- Scene proxy rendering requires engines such as Blender, adding complexity to the inference pipeline.

## Related Work & Insights

- **vs. CameraCtrl**: CameraCtrl controls only camera motion, whereas LiVER additionally achieves decoupled control of lighting and layout.
- **vs. VideoFrom3D**: VideoFrom3D requires training a style LoRA for each test sample (~40 minutes) and neglects lighting; LiVER is end-to-end and lighting-aware.
- **vs. Light-A-Video/LumiSculpt**: These methods focus on video relighting but entangle lighting with other physical attributes; LiVER achieves decoupling by starting from 3D scene proxies.
- **Insight**: Using intermediate PBR rendering results as conditioning signals for generative models represents a generalizable paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — The idea of using PBR rendering channels as conditioning signals for video generation is novel; the integrated design combining agent, renderer, and diffusion model is cohesive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative, qualitative, user study, and ablation experiments are comprehensive, though the dataset scale is relatively small (11K videos).
- Writing Quality: ⭐⭐⭐⭐ — The paper is well-structured with detailed method descriptions and high-quality illustrations.
- Value: ⭐⭐⭐⭐ — The work has practical value for film production and virtual content creation, advancing the development of controllable video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MotiMotion: Motion-Controlled Video Generation with Visual Reasoning](../../ICML2026/video_generation/motimotion_motion-controlled_video_generation_with_visual_reasoning.md)
- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](../../NeurIPS2025/video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[ICCV 2025\] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent](../../ICCV2025/video_generation/motionagent_fine-grained_controllable_video_generation_via_motion_field_agent.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] Physical Simulator In-the-Loop Video Generation](physical_simulator_in-the-loop_video_generation.md)

</div>

<!-- RELATED:END -->
