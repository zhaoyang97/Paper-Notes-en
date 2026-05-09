---
title: >-
  [Paper Note] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation
description: >-
  [CVPR 2026][Autonomous Driving][adversarial scenario generation] This paper proposes CompoSIA, a framework that achieves composable adversarial driving scene generation via disentangled control over three factors — Structure, Identity, and Action — built upon a video diffusion model. The approach reduces FVD for identity editing by 17% and increases the collision rate of downstream planners by 173%, effectively exposing hidden failure modes in autonomous driving systems.
tags:
  - CVPR 2026
  - Autonomous Driving
  - adversarial scenario generation
  - video diffusion model
  - disentangled control
  - identity injection
  - action control
  - safety testing
date: 2026-05-08
content_hash: 587f8a18b770aa70
---

# CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**Conference**: CVPR 2026
**arXiv**: [2603.12864](https://arxiv.org/abs/2603.12864)
**Code**: [Yifever20002/CompoSIA](https://github.com/Yifever20002/CompoSIA)
**Area**: Autonomous Driving
**Keywords**: adversarial scenario generation, video diffusion model, disentangled control, identity injection, action control, safety testing

## TL;DR

This paper proposes CompoSIA, a framework that achieves composable adversarial driving scene generation via disentangled control over three factors — Structure, Identity, and Action — built upon a video diffusion model. The approach reduces FVD for identity editing by 17% and increases the collision rate of downstream planners by 173%, effectively exposing hidden failure modes in autonomous driving systems.

## Background & Motivation

Safety validation of autonomous driving systems demands large quantities of test data covering rare, "long-tail" hazardous scenarios, which are extremely uncommon in the real world and prohibitively expensive to collect. Generative methods have become a key avenue for constructing adversarial scenarios, whose core requirement is **independently controllable editing of individual scene factors** — the ability to "replace vehicle appearance without altering trajectories," "change motion while preserving identity," and "introduce novel viewpoints without breaking consistency."

Existing methods exhibit clear limitations in controllability:

- **MagicDrive-V2**: Structure, identity, and action are processed through a shared pathway, causing control signals to be entangled. Editing one factor inevitably affects others, making precise single-factor editing difficult.
- **DriveEditor**: Supports partial editing operations but lacks novel-view synthesis capability; it can only operate on existing viewpoints and cannot generate scenes from new camera angles.
- **General video generation models (e.g., SVD)**: Weak identity preservation — the appearance features of reference objects tend to be lost or drift during generation.

The core insight of CompoSIA is that **structure, identity, and action are fundamentally characterized by different types of information** — structure encodes global spatial layout, identity encodes local appearance texture, and action encodes temporal motion trajectories — and therefore each requires a distinct injection mechanism rather than being handled by a unified mixed representation.

## Method

### Overall Architecture

CompoSIA is built upon Stable Video Diffusion (SVD) and controls structure, identity, and action through three purpose-designed modules:

1. **Structure Condition**: A sequence of 3D bounding boxes → defines the spatial layout of the scene.
2. **Noise-level Identity Injection**: A single-frame reference image → injects the appearance of the target object.
3. **Hierarchical Dual-Branch Action Control**: Trajectory conditions → controls object motion.

The three signals are injected at different layers and via different mechanisms within the diffusion model, naturally avoiding inter-signal interference.

### Module 1: Noise-level Identity Injection

**Problem**: How to extract identity information from a single reference image and maintain appearance consistency throughout video generation, while remaining unaffected by the pose in the reference image?

**Solution**: A "replaced-clean" training pair mechanism:

- **Training data construction**: A frame is sampled from the training video; the target object region is replaced with an object of the same category but different appearance → forming a *replaced clip*; the original video serves as the *clean clip*. The model learns to "accept identity information from the replaced clip and produce the clean clip as output."
- **Noise-level injection strategy**: At each denoising timestep $t$, the reference frame is corrupted with a corresponding level of noise before injection. At high noise levels (large $t$), identity information is blurred → the model relies on structural conditions to determine layout; at low noise levels (small $t$), identity details are clear → the model faithfully reproduces appearance textures.
- **Pose invariance**: By training across varying noise levels, the model automatically learns to extract pose-agnostic identity representations from the reference image — high noise retains only low-frequency features such as color and texture, naturally filtering out pose information.

This design is more elegant than approaches such as IP-Adapter, as it requires no additional encoder and achieves information disentanglement directly through the diffusion model's own noise schedule.

### Module 2: Hierarchical Dual-Branch Action Control

**Problem**: Object action encompasses both local deformation (e.g., body lean when turning) and global trajectory (straight/left/right), so how can both be precisely controlled simultaneously?

**Solution**: A dual-branch hierarchical control scheme:

**Branch 1 — AdaLN Local Residual Control**:
- Trajectory conditions are injected into intermediate layers of the diffusion model via Adaptive Layer Normalization (AdaLN).
- AdaLN operates in a residual manner: $\gamma \cdot \text{LayerNorm}(x) + \beta$, where $\gamma$ and $\beta$ are computed from the trajectory condition.
- Advantages: low parameter count, fast convergence; well-suited to capturing frame-level local motion variations.
- Analogous to the "small-magnitude correction" paradigm of LoRA.

**Branch 2 — Camera Attention with PRoPE Global Trajectory Control**:
- Positional Rotary Position Embedding (PRoPE) is introduced to embed 3D spatial position encoding into the attention computation.
- Camera Attention handles multi-view consistency by accounting for spatial relationships between different camera perspectives within the self-attention mechanism.
- Advantages: strong long-range dependency modeling; well-suited to temporal consistency of global trajectories.
- PRoPE encodes camera extrinsic matrices as rotary position embeddings, enabling the model to perceive relative positions in 3D space.

**Hierarchical fusion**: AdaLN handles *how* the object moves (local deformation), while Camera Attention handles *where* it goes (global trajectory); the two branches are complementary.

### Module 3: Structure Condition

- Input consists of 3D bounding box sequences spanning the full temporal extent of the video.
- Each bounding box defines the position, size, and orientation of an object in 3D space.
- Injected into the diffusion model via a ControlNet-like mechanism to provide coarse-level spatial constraints.
- The structure condition constrains only object locations, without restricting appearance or motion details.

### Loss & Training

- Trained on the nuScenes dataset, leveraging its multi-view annotations and 3D bounding box ground truth.
- The three modules are trained in stages: SVD weights are first frozen while training the structure condition → identity injection is added → action control is incorporated last.
- The loss function follows the standard diffusion denoising objective: $\mathcal{L} = \mathbb{E}_{t,\epsilon}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|^2]$, where $c$ encompasses all three conditions.

## Key Experimental Results

### Generation Quality

| Method | FVD↓ | FID↓ | Identity Preservation↑ |
|--------|------|------|------------------------|
| MagicDrive-V2 | baseline | baseline | baseline |
| DriveEditor | — | — | — |
| **CompoSIA** | **−17%** | best | best |

### Controllability Metrics

| Control Dimension | Metric | Improvement |
|-------------------|--------|-------------|
| Identity editing | FVD | −17% vs. strongest baseline |
| Rotation control | Rotation error | −30% |
| Translation control | Translation error | −47% |

### Downstream Safety Testing

Adversarial scenarios generated by CompoSIA are used for stress-testing an autonomous driving planner:

| Metric | Normal Scenarios | CompoSIA Adversarial Scenarios | Change |
|--------|-----------------|-------------------------------|--------|
| 3-second collision rate | baseline | +173% | **Substantially exposes safety vulnerabilities** |

This result demonstrates that the generated adversarial scenarios effectively expose planner failure modes that cannot be revealed by normal data.

### Ablation Study

| Ablation Condition | FVD Change | Observation |
|--------------------|-----------|-------------|
| Remove noise-level strategy (direct concat) | degrades | Identity–pose coupling; generation pose is locked to reference image |
| Remove AdaLN branch (Camera Attention only) | degrades | Local motion details are lost |
| Remove Camera Attention (AdaLN only) | degrades | Long-range trajectory inconsistency |
| Fixed noise level (no hierarchy) | degrades | Unable to balance identity detail and pose flexibility |

## Highlights & Insights

- **Philosophy of three-factor disentanglement**: Structure (spatial), identity (appearance), and action (temporal) each have an optimal injection mechanism — 3D bounding boxes for spatial constraints, noise level for appearance injection, and dual branches for temporal control. This "factor–mechanism alignment" is far more elegant than unified processing.
- **Noise-level Identity Injection is particularly ingenious**: It introduces no additional encoder whatsoever, relying solely on the diffusion model's own noise schedule to control information granularity. High noise = low-frequency information (color/texture); low noise = high-frequency information (details/edges) — pose-agnostic appearance extraction is achieved naturally.
- **Tangible safety value**: The +173% collision rate is not merely a metric improvement — it reflects the genuine discovery of planner bugs that would not be exposed under normal testing, which is a real and pressing need in autonomous driving safety validation.
- **3D awareness via PRoPE**: Encoding camera extrinsics as positional embeddings elegantly resolves multi-view consistency.

## Limitations & Future Work

1. **Dependence on nuScenes**: nuScenes is limited in scale (~1,000 scenes), constraining generation diversity to the training distribution. Transferring to larger datasets (e.g., Waymo Open) could further improve performance.
2. **Vehicle-only support**: The current approach primarily targets identity editing and action control for vehicles; support for other road participants such as pedestrians and cyclists requires additional validation.
3. **Dependency on 3D bounding box annotations**: The structure condition requires precise 3D annotations, limiting applicability to unannotated data. Automatic 3D layout estimation from 2D detections could be explored.
4. **Temporal length constraint**: The SVD backbone limits the temporal length of generated videos; longer sequences (e.g., a complete intersection traversal) may require segmented generation and stitching.
5. **Physical plausibility of adversarial scenarios**: The increased collision rate demonstrates scenario effectiveness, but some generated scenarios may contain physically implausible dynamics (e.g., impossible accelerations) that warrant further constraints.
6. **Multi-agent interaction**: The current framework is oriented primarily toward single-agent editing; modeling the interaction dynamics among multiple objects (e.g., yielding behavior) remains insufficiently addressed.

## Related Work & Insights

- **MagicDrive series**: Representative work on driving scene generation → CompoSIA significantly outperforms it in controllability.
- **DriveEditor**: Supports editing but lacks novel-view synthesis → CompoSIA's Camera Attention fills this gap.
- **SVD (Stable Video Diffusion)**: Foundation video generation model → CompoSIA demonstrates how to build fine-grained control on top of it.
- **IP-Adapter**: An alternative route to identity preservation → the noise-level approach is simpler and requires no additional encoder.
- **Insights**: The disentangled control paradigm is generalizable to broader scene editing tasks (indoor environments, robotics). The shift from random generation to purposeful adversarial generation is the central demand in safety validation.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4.0 | The three-factor disentangled framework is elegantly designed; Noise-level Identity Injection demonstrates originality |
| Practicality | 4.5 | Directly serves safety validation; +173% collision rate substantiates real-world value |
| Experimental Thoroughness | 4.0 | Three-level validation covering generation quality, controllability, and downstream tasks; ablations are comprehensive |
| Writing Quality | 4.0 | Clear structure; the three-factor disentanglement narrative flows naturally |
| **Overall** | **4.1** | A strong contribution to autonomous driving safety validation, with high alignment between technical design and application scenario |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](composing_driving_worlds_through_disentangled_cont.md)
- [\[ICLR 2026\] Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)](../../ICLR2026/autonomous_driving/steerable_adversarial_scenario_generation_through_test-time_preference_alignment.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](ttsg_text_to_traffic_scene_generation_from_natural_language.md)

</div>

<!-- RELATED:END -->
