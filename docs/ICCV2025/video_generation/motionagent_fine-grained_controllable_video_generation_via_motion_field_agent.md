---
title: >-
  [Paper Note] MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent
description: >-
  [ICCV 2025][Video Generation][Image-to-video generation] This paper proposes MotionAgent, which leverages a Motion Field Agent to parse motion descriptions from text into object trajectories and camera extrinsics…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Image-to-video generation"
  - "motion field agent"
  - "optical flow"
  - "fine-grained motion control"
  - "diffusion models"
date: 2026-05-08
content_hash: 7eadf0a06d7a10d2
---

# MotionAgent: Fine-grained Controllable Video Generation via Motion Field Agent

**Conference**: ICCV 2025
**arXiv**: [2502.03207](https://arxiv.org/abs/2502.03207)  
**Code**: [GitHub](https://github.com/leoisufa/MotionAgent)  
**Area**: Video Generation
**Keywords**: Image-to-video generation, motion field agent, optical flow, fine-grained motion control, diffusion models

## TL;DR

This paper proposes MotionAgent, which leverages a Motion Field Agent to parse motion descriptions from text into object trajectories and camera extrinsics, then unifies them into optical flow maps via an analytical flow synthesis module, enabling fine-grained and precise control over both object motion and camera motion in I2V generation using only text input.

## Background & Motivation

Existing I2V (Image-to-Video) generation models have achieved remarkable visual quality, yet precise motion control from text alone remains an open problem. Current approaches exhibit three categories of limitations:

**Text-encoder-based control** (e.g., DynamiCrafter, CogVideoX): Motion embeddings are injected into visual features via text encoders, enabling only coarse global control and failing to achieve fine-grained per-element motion control. High-quality text–motion aligned training data is also extremely scarce.

**Dedicated motion control modules** (e.g., DragNUWA, CameraCtrl): These design specialized modules for object motion or camera motion separately, but typically handle only one motion type at a time; moreover, their control inputs (e.g., point trajectories, camera extrinsic matrices) require domain expertise and are unfriendly to general users.

**Intermediate representation methods** (e.g., Motion-I2V, MOFA-Video): These use intermediate representations such as optical flow to control multiple motion types, but still require users to manually provide trajectories or camera parameters, resulting in a high usage barrier.

The core motivation of MotionAgent is: **Can users control both object motion and camera motion simultaneously and precisely in generated videos using only natural language descriptions?** By combining the reasoning capability of LLM agents with geometrically grounded optical flow synthesis, the paper establishes an end-to-end pipeline of "text → motion field → controllable video."

## Method

### Overall Architecture

MotionAgent consists of two major components:

1. **Motion Field Agent**: Parses motion information from text and converts it into two explicit intermediate representations — object trajectories and camera extrinsics.
2. **Controllable I2V Generation Model**: Comprises an analytical optical flow synthesis module and a flow adapter, which transform the intermediate representations into unified optical flow to condition a base I2V diffusion model (SVD).

### Key Designs of the Motion Field Agent

#### Step 1: Video Motion Decomposition
The agent analyzes motion information in the text and decomposes it into **object motion descriptions** and **camera motion descriptions**, achieving decoupled and independently controllable motion types.

#### Step 2: Object Trajectory Drawing
This step involves two sub-tasks:

- **Object Identification**: The agent extracts descriptions of dynamic objects from the text and invokes Grounded-SAM for open-world object detection and segmentation. Detection results are overlaid on the input image as semi-transparent masks and fed back to the agent for confirmation.
- **Trajectory Drawing**: The agent determines trajectory starting points based on detection results and adopts a **grid selection strategy** — dividing the image into an $N \times M$ grid with integer-labeled cells — whereby the agent selects grid indices to define trajectory waypoints, which are then connected sequentially. This approach is more stable and robust than directly generating continuous coordinates.

#### Step 3: Camera Extrinsics Generation
The agent generates camera extrinsics $E$ directly from the camera motion text description and the input image. The text specifies the camera path, while the image helps the agent estimate appropriate motion magnitude (e.g., expansive scenes warrant large motion, close-up shots require subtle adjustment). The translation $T$ is constrained to $(-1, 1)$ and subsequently rescaled using the depth map.

#### Step 4: Rethinking Mechanism (Optional)
The agent analyzes the generated video, reviews decisions made at each prior step, and forms a closed feedback loop. It corrects object trajectories and camera extrinsics based on discrepancies between the text and the generated video, further improving generation quality.

### Analytical Optical Flow Synthesis Module

The core of this module is the geometric composition of object motion and camera motion in 3D space:

1. **Depth Estimation and 3D Lifting**: Metric3D is used to estimate a depth map $D$ from the input image; each pixel $I^0$ is back-projected into 3D space to obtain its initial 3D position $P^0$.
2. **Object Motion Flow**: CMP estimates dense optical flow $F_{obj}$ from object trajectories; combined with the depth map, the 3D positional offset $O$ induced by object motion is computed, yielding the updated 3D position $P^1 = P^0 + O$.
3. **Camera Motion Reprojection**: $P^1$ is reprojected into image coordinates according to camera extrinsics $E$: $I^1 = \Pi(E P^1)$.
4. **Unified Optical Flow**: Pixel offsets are computed as the unified optical flow: $F = I^1 - I^0$, which encapsulates both object motion and camera motion.

### Loss & Training

- The base I2V model uses frozen SVD (Stable Video Diffusion).
- The flow adapter follows the architecture proposed in MOFA-Video but requires **fine-tuning** to accommodate the unified optical flow, which differs in domain from real optical flow.
- Fine-tuning data preparation: Unimatch is used to estimate real optical flow; DROID-SLAM computes camera extrinsics; camera-induced flow is subtracted to obtain pure object motion flow; sparse sampling yields object trajectories, which are then re-processed via the analytical synthesis method to produce unified optical flow as training input.
- Training configuration: 32 × A800 GPUs, AdamW optimizer, learning rate $2 \times 10^{-5}$, resolution $512 \times 512$, randomly sampled 24 frames (stride 4).

## Key Experimental Results

### Main Results: General I2V Generation (VBench)

| Method | I2V Score | Video-Text Camera Motion | Subject Consistency | Motion Smoothness | Dynamic Degree |
|--------|-----------|--------------------------|---------------------|-------------------|----------------|
| VideoCrafter | 88.95 | 33.60 | 97.86 | 98.00 | 22.60 |
| DynamiCrafter | 97.98 | 35.81 | 95.69 | 97.38 | 47.40 |
| SVD (baseline) | 96.93 | – | 95.42 | 98.12 | 43.17 |
| **MotionAgent** | **97.51** | **81.91** | **96.10** | **98.93** | 16.67 |

**Key Findings**: MotionAgent achieves **81.91%** on the Video-Text Camera Motion metric, far surpassing all other methods (second-best DynamiCrafter at 35.81%), demonstrating precise camera motion control capability.

### Main Results: Controllable I2V Generation (Custom Benchmark)

| Method | Object Movement Q&A | Complex Camera Motion | Total Scores |
|--------|---------------------|-----------------------|--------------|
| DynamiCrafter | 29.38 | 8.22 | 16.58 |
| CogVideoX | 26.47 | 20.62 | 22.93 |
| Pyramid Flow | 30.96 | 6.18 | 15.97 |
| **MotionAgent** | **45.69** | **77.76** | **65.10** |
| **MotionAgent (Rethinking)** | **49.58** | **89.04** | **73.45** |

**Key Findings**: MotionAgent achieves a total score of 65.10%, substantially outperforming the second-best method CogVideoX (22.93%); with Rethinking, the score further improves to 73.45%.

### Ablation Study

| Variant | Object Movement Q&A | Complex Camera Motion | Dynamic Degree |
|---------|---------------------|-----------------------|----------------|
| w/o detection tool (multi-turn dialogue) | 34.33 | 75.11 | 20.53 |
| w/o object motion | 10.51 | 75.95 | 8.42 |
| w/o camera motion | 38.20 | 0.30 | 29.95 |
| w/o flow synthesis (direct addition) | 30.07 | 64.92 | 27.89 |
| w/o adapter fine-tuning | 40.56 | 48.27 | 28.47 |
| **Full model** | **45.69** | **77.76** | **32.11** |

**Key Findings from Ablation**:
- Grounded-SAM-assisted detection outperforms pure multi-turn dialogue object identification (+11.36 on Q&A).
- Analytical optical flow synthesis surpasses direct flow addition by 15.62 and 12.84 percentage points on the two metrics respectively.
- Flow adapter fine-tuning yields a significant gain on complex camera motion (+29.49).
- The Rethinking mechanism notably improves complex camera motion control (+11.28).

### Robustness Across Different LLMs and Base Models

| Agent LLM | Base Model | Total Scores |
|-----------|------------|--------------|
| GPT-4o | SVD | 65.10 |
| Qwen2 | SVD | 61.73 |
| Llama3 | SVD | 63.80 |
| GPT-4o | Motion-I2V | 61.00 |

Performance variation across different LLM backends is minimal, demonstrating the robustness of the proposed method to the choice of agent backbone.

## Highlights & Insights

1. **Elegant fusion of agent reasoning and geometry**: Rather than having an LLM directly generate video control signals end-to-end, the agent produces structured intermediate representations (trajectories + extrinsics), which are then converted to optical flow via geometric computation — combining the semantic understanding of LLMs with the precision of geometric calculation.
2. **Grid selection instead of coordinate generation**: Reformulating trajectory drawing as a grid-index selection problem reduces the difficulty of generating continuous coordinates for LLMs and improves the robustness of trajectory generation.
3. **Rethinking mechanism**: A closed-loop "generate → evaluate → correct" feedback cycle, analogous to agent self-reflection, yields particularly notable gains on complex camera motion (+11.28).
4. **Decoupled design**: Decoupling object motion and camera motion prevents error propagation between components (e.g., object identification errors do not affect camera motion control).
5. **Custom evaluation benchmark**: To address the lack of motion-semantic alignment evaluation in existing benchmarks, a dedicated controllable I2V evaluation set is constructed, comprising 432 object motion prompts and 662 complex camera motion prompts.

## Limitations & Future Work

1. **Decreased Dynamic Degree**: Precise motion control causes objects not mentioned in the text to remain static, leading to lower Dynamic Degree scores on VBench (16.67 vs. 43.17 for SVD), which may result in less lively videos in scenarios requiring rich natural motion.
2. **Heavy reliance on external toolchains**: The pipeline involves multiple external models including Grounded-SAM, Metric3D, CMP, Unimatch, and DROID-SLAM, resulting in high system complexity where errors from any component may accumulate and propagate.
3. **High inference cost**: The system requires multi-step LLM agent reasoning, optional Rethinking iterations, and inference across multiple external models, incurring substantially higher latency and computational cost than end-to-end approaches.
4. **Inherent limitations of optical flow representation**: Optical flow as a 2D motion representation has fundamental limitations in handling occlusion, newly appearing content, and large-displacement motion.
5. **Self-constructed evaluation metrics**: The Object Movement Q&A metric relies on GPT-4o scoring, which may introduce scoring bias; the coverage and generalizability of the custom benchmark remain to be validated.

## Related Work & Insights

- **MOFA-Video**: Provides the foundational flow adapter architecture; MotionAgent extends this with a fine-tuning strategy to accommodate unified optical flow.
- **Motion-I2V**: Uses an optical flow diffusion model to generate intermediate representations, but achieves lower precision than MotionAgent's analytical synthesis approach.
- **ChatCam**: Enables camera motion control via dialogue, but is limited to camera motion only and does not address object motion.
- **ObjCtrl-2.5D**: Extends 2D trajectories to 2.5D but requires manual input of trajectories and depth.
- **Future directions**: The agent framework can be generalized to broader generation tasks (3D scene generation, long video generation); the Rethinking mechanism can be combined with RLHF for further optimization; optical flow representation can be upgraded to 3D scene flow to handle more complex motion scenarios.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | ⭐⭐⭐⭐ | First to combine LLM agents with geometric optical flow synthesis for purely text-driven controllable I2V |
| Technical Depth | ⭐⭐⭐⭐ | Analytical flow synthesis module is elegantly designed with a clear 3D composition rationale |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | VBench evaluation + custom benchmark + ablation + user study + robustness validation — highly comprehensive |
| Practical Value | ⭐⭐⭐⭐ | Lowers the barrier for controllable video generation, though system complexity poses deployment challenges |
| Writing Quality | ⭐⭐⭐⭐ | Well-structured with detailed pipeline illustrations |
| Overall | ⭐⭐⭐⭐ | Addresses an important problem with a novel approach and solid experiments; a strong contribution to controllable I2V generation |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MotionCharacter: Fine-Grained Motion Controllable Human Video Generation](../../AAAI2026/video_generation/motioncharacter_fine-grained_motion_controllable_human_video_generation.md)
- [\[ICCV 2025\] ETVA: Evaluation of Text-to-Video Alignment via Fine-Grained Question Generation and Answering](etva_evaluation_of_text-to-video_alignment_via_fine-grained_question_generation_.md)
- [\[ICCV 2025\] DH-FaceVid-1K: A Large-Scale High-Quality Dataset for Face Video Generation](dh-facevid-1k_a_large-scale_high-quality_dataset_for_face_video_generation.md)
- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](../../AAAI2026/video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)

</div>

<!-- RELATED:END -->
