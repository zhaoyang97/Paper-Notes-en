---
title: >-
  [Paper Note] pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning
description: >-
  [ICLR 2026][3D Vision][Visual Programming] pySpatial is a visual programming framework that enables MLLMs to generate Python code that automatically invokes 3D spatial tools (3D reconstruction, camera pose estimation, novel view synthesis, etc.), transforming limited 2D image inputs into interactively explorable 3D scenes. The framework achieves zero-shot, plug-and-play explicit 3D spatial reasoning, attaining an overall accuracy of 58.56% on the MindCube benchmark—surpassing GPT-4.1-mini by 12.94% and VLM-3R by 16.5%—while also successfully driving a real quadruped robot to perform indoor navigation.
tags:
  - ICLR 2026
  - 3D Vision
  - Visual Programming
  - 3D Reconstruction
  - Spatial Reasoning
  - Zero-Shot
  - Robot Navigation
date: 2026-05-08
content_hash: 86a1855ecc3baa48
---

# pySpatial: Generating 3D Visual Programs for Zero-Shot Spatial Reasoning

**Conference**: ICLR 2026
**arXiv**: [2603.00905](https://arxiv.org/abs/2603.00905)
**Code**: [Project Page](https://pySpatial.github.io)
**Area**: 3D Vision
**Keywords**: Visual Programming, 3D Reconstruction, Spatial Reasoning, Zero-Shot, Robot Navigation

## TL;DR

pySpatial is a visual programming framework that enables MLLMs to generate Python code that automatically invokes 3D spatial tools (3D reconstruction, camera pose estimation, novel view synthesis, etc.), transforming limited 2D image inputs into interactively explorable 3D scenes. The framework achieves zero-shot, plug-and-play explicit 3D spatial reasoning, attaining an overall accuracy of 58.56% on the MindCube benchmark—surpassing GPT-4.1-mini by 12.94% and VLM-3R by 16.5%—while also successfully driving a real quadruped robot to perform indoor navigation.

## Background & Motivation

**Background**: MLLMs (e.g., GPT-4o, Claude) excel at image captioning and video understanding, yet remain severely limited in 3D spatial reasoning. Recent studies show that MLLMs perform only marginally above chance on multi-view spatial reasoning tasks such as "How should one move from viewpoint 1 to viewpoint 2?"

**Limitations of Prior Work**:

1. **Training data bottleneck**: MLLMs are pretrained on massive image–text pairs, but explicit 3D spatial supervision data is extremely scarce and costly to annotate, making it difficult for models to establish reliable correspondences between language and 3D spatial structure.
2. **Unreliable implicit reasoning**: Existing methods (e.g., cognitive maps, chain-of-thought) rely on the "implicit imagination" of MLLMs to construct spatial models, yielding limited and uncontrollable results.
3. **Single-view limitation**: Methods such as SpatialVLM and SpatialRGPT address only single-view spatial understanding and cannot handle multi-view reasoning.
4. **Requirement for fine-tuning**: Specialized spatial models (e.g., VLM-3R) require fine-tuning on synthetic data and lack plug-and-play flexibility.

**Key Challenge**: MLLMs lack explicit geometric understanding of the 3D world, and implicit reasoning alone cannot reliably solve spatial problems.

**Goal**: Rather than having MLLMs implicitly imagine 3D structure, pySpatial adopts a visual programming paradigm in which MLLMs generate Python code to invoke 3D tools, explicitly constructing, exploring, and reasoning about 3D scenes—transforming "imagination" into "computation."

## Method

### Overall Architecture

pySpatial operates in three stages:

$$\text{Image Sequence} + \text{Language Query} \xrightarrow{\text{Code Agent } \mathcal{F}} \text{Python Program } z \xrightarrow{\text{Interpreter } \mathcal{E}} \text{Intermediate Output } O \xrightarrow{\text{MLLM } \mathcal{M}} \text{Final Answer } r$$

1. **Program generation**: A code agent (GPT-4o by default) generates a Python program $z = \mathcal{F}(q)$ that calls the pySpatial API in response to query $q$.
2. **Program execution**: A Python interpreter executes the program, invoking 3D tools to produce intermediate results $O = \mathcal{E}(z, \mathcal{I})$ (text, images, or rendered views).
3. **Final reasoning**: An MLLM integrates the original images, program outputs, and query to generate the final answer $r = \mathcal{M}(\mathcal{I}, O, q)$.

### Key Design 1: Modular Spatial Tool API

pySpatial defines a concise Python API that encapsulates complex low-level implementations as high-level semantic operations:

| API Function | Function | Default Parameters |
|---|---|---|
| `reconstruct(scene)` | 3D reconstruction from an image sequence | — |
| `describe_camera_motion(recon)` | Describes camera poses in natural language | — |
| `synthesize_novel_view(recon, pose)` | Renders a novel view from an arbitrary viewpoint | — |
| `rotate_right/left(ext, angle)` | Rotates the camera pose left/right | 45° |
| `move_forward/backward(ext, dist)` | Moves the camera forward/backward | 0.3 |
| `turn_around(ext)` | Rotates the camera 180° | — |

**3D Reconstruction Tool**: CUT3R (metric scale, for real-world navigation) or VGGT (normalized space, for benchmark evaluation) is selected based on the task, back-projecting pixels to world coordinates via:

$$\mathbf{X}_i = \mathbf{G}_n^{-1} \pi^{-1}(\mathbf{p}_i, D_n(\mathbf{p}_i), K^{-1})$$

**Camera Motion Description**: Camera pose matrices are converted into egocentric motion labels in natural language (forward, backward, turn left, etc., across eight directions) by computing the yaw angle $\theta = \text{atan2}(d_x, d_z) \cdot 180/\pi$ of the displacement in the world coordinate system projected onto the first camera frame, followed by discretization.

**Novel View Synthesis**: Rasterization rendering is performed based on the reconstructed point cloud $\mathcal{P}$ and target camera pose; high-level commands such as `rotate_left` and `turn_around` are automatically converted to yaw rotations before rendering.

### Key Design 2: Zero-Shot Visual Program Generation

A core advantage of pySpatial is its zero-shot nature—no gradient-based training is required:

- The code agent requires only API documentation and a small number of query–code examples (in-context learning).
- The agent has no access to model weights, file I/O, or rendering backend internals.
- Structured output is used: the agent first performs natural language reasoning, then synthesizes Python code.
- The generated program itself constitutes an interpretable reasoning trace that can be directly inspected, debugged, or modified.

### Key Design 3: Plug-and-Play Framework

- Compatible with both open-source and closed-source MLLMs (GPT-4o, GPT-4.1-mini, Claude, etc., can serve as either the code agent or the final reasoner).
- The 3D reconstruction module is interchangeable (CUT3R / VGGT / DUSt3R, etc.).
- All experiments are conducted on a single NVIDIA A6000 Ada GPU.

## Key Experimental Results

### Main Results: MindCube Full Set (21K+ Questions)

| Method | Type | Overall | Rotation | Among | Around |
|---|---|:---:|:---:|:---:|:---:|
| Random (chance) | — | 32.35 | 36.36 | 32.29 | 30.66 |
| LLaVA-OneVision-7B | Open-source MLLM | 47.43 | 36.45 | 48.42 | 44.09 |
| DeepSeek-VL2-Small | Open-source MLLM | 47.62 | 37.00 | 50.38 | 26.91 |
| GPT-4o | Commercial MLLM | 38.81 | 32.65 | 40.17 | 29.16 |
| GPT-4.1-mini | Commercial MLLM | 45.62 | 37.84 | 47.22 | 34.56 |
| Claude-4-Sonnet | Commercial MLLM | 44.75 | 48.42 | 44.21 | 47.62 |
| VLM-3R | Specialized spatial model | 42.09 | 36.73 | 44.22 | 24.45 |
| **pySpatial (Ours)** | Visual programming | **58.56** | **43.20** | **60.54** | **48.10** |

pySpatial achieves an overall accuracy of 58.56%, decisively outperforming all baselines: +10.94% over the strongest open-source MLLM (DeepSeek-VL2-Small), +12.94% over GPT-4.1-mini, and +16.47% over the fine-tuned VLM-3R. On the most challenging Among category (reasoning about the spatial relationships between a central object and all surrounding objects), pySpatial reaches 60.54%, while no other method exceeds 50%.

### MindCube-1k Subset Comparison

| Method | Overall | Rotation | Among | Around |
|---|:---:|:---:|:---:|:---:|
| GPT-4o | 42.29 | 35.00 | 43.00 | 46.40 |
| Chain-of-Thought | 40.48 | 32.00 | 36.00 | 58.00 |
| Cognitive Map | 41.43 | 37.00 | 41.67 | 44.40 |
| ViperGPT | 36.95 | 20.50 | 41.00 | 40.40 |
| VADAR | 40.76 | 33.50 | 40.67 | 46.80 |
| VADAR + 3D Reconstruction | 35.62 | 31.00 | 36.83 | 36.40 |
| **pySpatial** | **62.35±1.18** | **41.83±2.34** | **64.89±2.60** | **72.67±3.30** |

pySpatial outperforms all mental-model-based methods and visual programming baselines by approximately 20%. Notably, augmenting VADAR with a 3D reconstruction module actually degrades performance (40.76→35.62), demonstrating that access to 3D information alone is insufficient—effective API design is essential for leveraging 3D geometry.

### Omni3D-Bench Single-View Generalization

| Method | numeric(ct) | numeric(other) | y/n | multi-choice | Total |
|---|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 28.1 | 35.5 | 66.7 | 57.2 | 42.9 |
| VADAR | — | — | — | — | 41.5 |
| ViperGPT | — | — | — | — | 27.8 |
| **pySpatial** | — | — | — | — | **45.3** |

Even in the single-view setting, pySpatial surpasses GPT-4o and all visual programming methods, validating the framework's generalization across settings.

## Highlights & Insights

### Strengths

1. **Paradigm innovation**: The conceptual shift from "implicit imagination" to "explicit computation" is exceptionally clear; visual programming serves as an effective bridge between MLLMs and the 3D world.
2. **Zero-shot state-of-the-art**: Without any training, the framework substantially outperforms fine-tuned specialized models across multiple benchmarks, demonstrating strong generalization.
3. **Interpretability**: The generated Python programs serve as precise records of the reasoning process, facilitating debugging and auditing.
4. **Real-world validation**: Indoor navigation experiments with a quadruped robot demonstrate the feasibility of transferring from academic benchmarks to real-world deployment.

### Limitations & Future Work

1. The framework relies on GPT-4o as the code agent, incurring high API costs and dependence on the availability of commercial models.
2. The quality of 3D reconstruction directly affects downstream reasoning; performance may degrade in texture-poor or repetitively textured scenes.
3. Novel view synthesis is based on point cloud rasterization, which produces holes in occluded regions and may adversely affect subsequent MLLM reasoning.

## Rating

⭐⭐⭐⭐ — An elegant application of the visual programming paradigm to 3D spatial reasoning. The approach is concise and effective; its zero-shot performance is impressive, and it offers a practical pathway toward grounding MLLMs in embodied intelligence.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](../../CVPR2026/3d_vision/context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)
- [\[CVPR 2026\] Masking Matters: Unlocking the Spatial Reasoning Capabilities of LLMs for 3D Scene-Language Understanding](../../CVPR2026/3d_vision/masking_matters_unlocking_the_spatial_reasoning_capabilities_of_llms_for_3d_scen.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](../../ICCV2025/3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](../../CVPR2026/3d_vision/learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](../../ICCV2025/3d_vision/monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)

<!-- RELATED:END -->
