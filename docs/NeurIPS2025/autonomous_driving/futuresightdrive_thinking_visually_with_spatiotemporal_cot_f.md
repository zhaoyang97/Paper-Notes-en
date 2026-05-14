---
title: >-
  [Paper Note] FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving
description: >-
  [NeurIPS 2025][Autonomous Driving][Visual CoT] FSDrive enables VLAs to "think visually" — first acting as a world model to generate a unified visual CoT frame that integrates future lane lines, 3D detection boxes…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Visual CoT"
  - "Trajectory Planning"
  - "World Model"
  - "VLA"
  - "Future Frame Prediction"
date: 2026-05-08
content_hash: 942ab4d39fea4705
---

# FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving

**Conference**: NeurIPS 2025
**arXiv**: [2505.17685](https://arxiv.org/abs/2505.17685)
**Code**: [GitHub](https://github.com/MIV-XJTU/FSDrive)
**Area**: Autonomous Driving / VLA
**Keywords**: Visual CoT, Trajectory Planning, World Model, VLA, Future Frame Prediction

## TL;DR

FSDrive enables VLAs to "think visually" — first acting as a world model to generate a unified visual CoT frame that integrates future lane lines, 3D detection boxes, and scene predictions, then acting as an inverse dynamics model to perform trajectory planning based on current observations and the visual CoT. This approach activates the visual generation capability of MLLMs using only a minimal amount of data (~0.3%).

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: Most existing VLA-based autonomous driving models employ textual CoT (e.g., scene descriptions, coordinate strings) as intermediate reasoning steps, but this introduces critical issues:

1. **Modality Gap**: Compressing continuous visual information into discrete text is lossy, discarding fine-grained spatio-temporal relationships.
2. **Semantic Discontinuity**: Coordinates and scene relations expressed in text form a modality gap with the original visual input.
3. **Insufficient Information**: Textual CoT struggles to capture the temporal evolution and spatial structure of dynamic scenes.

Human drivers reason more by "mentally simulating future scenes" than by "verbally describing them." Accordingly, models should also "think" in a visual manner.

## Method

### Overall Architecture

Two-stage training:
- **Pre-training Stage**: Unified visual understanding (VQA) + visual generation (future frame prediction), progressively from structural priors to complete scenes.
- **Fine-tuning Stage**: Scene understanding + visual CoT-based trajectory planning.

### Key Designs

1. **Visual Spatio-Temporal CoT**:
    - Function: Generates a unified image frame that integrates multiple types of future information as an intermediate reasoning step.
    - Mechanism: Fuses future lane lines (annotated in red), 3D detection boxes, and predicted scenes into a single image; lane lines encode spatially drivable regions, detection boxes encode key object motion, and the scene image encodes temporal evolution.
    - Design Motivation: Unifying everything into image format avoids semantic loss from cross-modal conversion, while encoding both spatial (lane lines + detection boxes) and temporal (scene evolution) dimensions of future information.

2. **Unified Pre-training Paradigm**:
    - Function: Simultaneously activates visual understanding and visual generation capabilities on top of an existing MLLM.
    - Mechanism: Extends the VQ-VAE image codebook into the MLLM's text vocabulary, enabling the model to autoregressively predict visual tokens.
    - Design Motivation: Requires only ~0.3% of the data needed by methods trained from scratch, without modifying the MLLM architecture, directly activating latent visual generation capabilities.
    - Progressive Generation: Lane line tokens $Q_l$ (static physical constraints) → 3D detection box tokens $Q_d$ (dynamic physical constraints) → complete future frame tokens $Q_f$.

### Loss & Training

- **Pre-training**: Joint training with VQA cross-entropy loss and visual token autoregressive prediction loss.
- **Fine-tuning**: DriveLM GVQA scene understanding + nuScenes trajectory planning, using unified visual CoT.
- Initialized from Qwen2-VL-2B; pre-trained for 32 epochs, fine-tuned for 12 epochs.
- VQ-VAE uses the MoVQGAN encoder-decoder.

## Key Experimental Results

### Main Results (Table)

nuScenes Trajectory Planning (ST-P3 metrics):

| Method | LLM | L2 (1s) ↓ | L2 (2s) ↓ | L2 (3s) ↓ | Col. (1s) ↓ | Col. (2s) ↓ | Col. (3s) ↓ |
|--------|-----|-----------|-----------|-----------|-------------|-------------|-------------|
| VAD | - | 0.54 | 1.15 | 1.98 | 0.04 | 0.39 | 1.17 |
| OmniDrive | ✓ | 0.51 | 1.04 | 1.70 | - | - | - |
| **FSDrive** | ✓ | **superior** | **superior** | **superior** | **lower** | **lower** | **lower** |

FSDrive outperforms baselines on both L2 displacement error and collision rate, while achieving competitive performance on DriveLM scene understanding and future frame FID.

### Ablation Study

- Visual CoT vs. Textual CoT vs. No CoT: Visual CoT significantly outperforms textual CoT, which in turn outperforms no CoT.
- Progressive generation vs. direct generation: Progressively generated future frames better conform to physical constraints.
- Necessity of VQA pre-training: Removing VQA pre-training leads to a substantial drop in scene understanding performance.
- MoVQGAN codebook size: Larger codebooks improve generation quality at the cost of increased inference overhead.

### Key Findings

- A competitive trajectory planning performance is achievable with only Qwen2-VL-2B, demonstrating the reasoning gains provided by visual CoT.
- Visual CoT supplies richer spatio-temporal information than textual CoT, directly reducing collision rates.
- Progressive generation is essential — directly generating complete future frames tends to violate physical constraints.
- MLLM visual generation capabilities can be activated with a very small amount of data, without training from scratch.

## Highlights & Insights

- **Paradigm Innovation**: Enabling VLAs to "think visually" rather than "think verbally" more closely mirrors human driving cognition.
- **Low Cost**: Visual generation is activated with only 0.3% of the data, requiring no modification to the MLLM architecture.
- **Progressive Physical Priors**: Reasoning over lane lines and detection boxes as physical constraints before generating the complete scene ensures physical feasibility.
- Unifies the world model (future prediction) and inverse dynamics model (trajectory planning) within a single VLA.

## Limitations & Future Work

- Validation is limited to nuScenes, which contains relatively homogeneous scenarios.
- Visual tokens from VQ-VAE lack semantic information, potentially affecting understanding tasks.
- Only a single future frame is predicted; multi-frame or long-horizon visual reasoning remains unexplored.
- Whether the inference speed of the 2B model meets real-time requirements has not been verified.

## Related Work & Insights

- Distinction from EMMA (pure textual CoT) and CoT-VLA (mixed text-visual CoT): FSDrive employs a purely visual unified CoT.
- Draws on ideas from visual prompt engineering (e.g., red circles to guide attention) for annotating lane lines and detection boxes.
- The dual-purpose VLA combining a world model (generation) and an inverse dynamics model (planning) offers a new paradigm for autonomous driving VLMs.

## Rating

⭐⭐⭐⭐ — The visual CoT concept is novel and well-motivated; the progressive physical prior design is elegant; and the low-cost approach to activating visual generation is practically valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning](how_different_from_the_past_spatio-temporal_time_series_forecasting_with_self-su.md)
- [\[NeurIPS 2025\] Spatio-Temporal Graphs Beyond Grids: Benchmark for Maritime Anomaly Detection](spatio-temporal_graphs_beyond_grids_benchmark_for_maritime_anomaly_detection.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
