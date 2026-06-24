---
title: >-
  [Paper Note] SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving
description: >-
  [CVPR 2025][Autonomous Driving][VLM Synergy] The paper proposes SOLVE, which achieves feature-level synergy between VLM and end-to-end (E2E) driving models via a shared SQ-Former vision encoder. By employing Trajectory Chain-of-Thought (T-CoT), it utilizes the long-range trajectories from the VLM as prior initialization for the E2E model, achieving a state-of-the-art average L2 error of 0.28m on nuScenes.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "VLM Synergy"
  - "End-to-End Driving"
  - "Trajectory Chain-of-Thought"
  - "Temporal Decoupling"
  - "Shared Encoder"
date: 2026-05-08
content_hash: d3965fda0795a62f
---

# SOLVE: Synergy of Language-Vision and End-to-End Networks for Autonomous Driving

**Conference**: CVPR 2025  
**arXiv**: [2505.16805](https://arxiv.org/abs/2505.16805)  
**Code**: None  
**Area**: Autonomous Driving / End-to-End Driving  
**Keywords**: VLM Synergy, End-to-End Driving, Trajectory Chain-of-Thought, Temporal Decoupling, Shared Encoder

## TL;DR

The paper proposes SOLVE, which achieves feature-level synergy between VLM and end-to-end (E2E) driving models via a shared SQ-Former vision encoder. By employing Trajectory Chain-of-Thought (T-CoT), it utilizes the long-range trajectories from the VLM as prior initialization for the E2E model, achieving a state-of-the-art average L2 error of 0.28m on nuScenes.

## Background & Motivation

**Background**: End-to-end (E2E) autonomous driving directly outputs planning trajectories from sensor inputs but lacks semantic understanding capabilities. Vision-Language Models (VLMs) possess strong scene-understanding and reasoning capabilities but suffer from high inference latency and are incapable of directly outputting precise control signals.

**Limitations of Prior Work**: VLMs and E2E models possess complementary strengths: VLMs excel at understanding "what to do," whereas E2E models excel at executing "how to do it specifically." However, they are typically designed independently, where information transfer is restricted to linguistic intermediate representations (e.g., "decelerate"), thus discarding crucial spatial information.

**Key Challenge**: VLMs are slow to infer but deep in understanding, whereas E2E models are fast to respond but shallow in comprehension. The challenge lies in enabling deep synergy between the two at the feature level, rather than merely cascading them through linguistic interfaces.

**Key Insight**: A shared vision encoder allows both models to "see the same things." The long-range trajectories of the VLM are asynchronously stored in memory for real-time querying by the E2E model, preventing the VLM from becoming a latency bottleneck.

**Core Idea**: Shared SQ-Former + T-CoT trajectory prior + Asynchronous temporal decoupling = Deep synergy between VLM and E2E.

## Method

### Key Designs

1. **共享 Sequential Q-Former (SQ-Former)**:

    - **Function**: Provides uniform visual encoding for both VLM and E2E models.
    - **Mechanism**: 384 learnable queries perform cross-attention sequentially with image features, detection tokens, and lane tokens, following a Q→Img→Det→Lane encoding sequence. The encoded results are simultaneously fed into both the VLM and E2E branches.
    - **Design Motivation**: The shared encoder allows the semantic understanding capability of the VLM to enhance the E2E visual features through gradient backpropagation (the VLM branch improves E2E by 1.5cm L2, and conversely improves VLM by 0.6cm).

2. **Trajectory Chain-of-Thought (T-CoT)**:

    - **Function**: Transforms trajectory planning from discrete textual reasoning to continuous spatial reasoning.
    - **Mechanism**: Maintains a bank of 36 candidate trajectories clustered via K-means from the training data. The VLM first selects the optimal coarse trajectory and then refines it using trajectory tokens. Compared to direct coordinate regression, this coarse-to-fine paradigm yields more reliable VLM inference.
    - **Design Motivation**: The performance is optimal with 6 reference trajectories (4 are too few and cause confusion, while 8 are too many and introduce distraction).

3. **异步时序解耦 (Asynchronous Temporal Decoupling)**:

    - **Function**: Addresses the issue of VLM inference latency making it unsuitable for real-time control.
    - **Mechanism**: The VLM generates long-range trajectories (e.g., 3 seconds) at a low frequency and stores them in global memory. The E2E model operates at a high frequency, retrieving the VLM trajectory from memory at each step to use as an initialization prior.
    - **Design Motivation**: The VLM does not need to perform inference at every frame—its value lies in long-range comprehension, while the E2E model handles real-time refinement.

### Loss & Training

Three-stage training: QA training (cross-entropy) $\rightarrow$ trajectory adapter (MSE) $\rightarrow$ joint VLM+E2E training (L2 trajectory loss).

## Key Experimental Results

### Main Results

nuScenes open-loop planning L2 error ↓:

| Method | 1s | 2s | 3s | Average | Collision Rate |
|------|-----|-----|-----|------|--------|
| OmniDrive | 0.17 | 0.30 | 0.55 | 0.33 | 0.25% |
| **SOLVE-VLM** | **0.15** | **0.23** | **0.47** | **0.28** | **0.20%** |

### Ablation Study

| Configuration | Average L2 | Description |
|------|---------|------|
| W/o shared SQ-Former | 0.30 | Shared encoder contributes 0.02m |
| W/o T-CoT | 0.295 | T-CoT contributes 0.015m |
| Full SOLVE | **0.28** | — |

### Key Findings
- The shared encoder is mutually beneficial: the semantic cues from the VLM enhance E2E representation features, while the spatial precision of the E2E model improves the VLM's trajectory prediction.
- The optimal number of reference trajectories for T-CoT is 6; selecting from candidates is more stable than direct regression.
- Asynchronous decoupling makes real-time deployment feasible.

## Highlights & Insights
- **Feature-Level Synergy > Linguistic Cascading**: The shared encoder realizes deep fusion of the two models in the feature space, which is more efficient than transferring information via language.
- **Practicality of Asynchronous Design**: The VLM does not need to run in real-time; its long-range comprehension is stored in memory and retrieved by the E2E model on demand.

## Limitations & Future Work
- Evaluation is restricted to nuScenes open-loop testing; lacks closed-loop validation.
- VLM inference still incurs latency overhead.
- Relies on the OmniDrive-nuScenes QA dataset.

## Rating
- Novelty: ⭐⭐⭐⭐ The visual-linguistic feature-level synergy design for E2E is inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation studies, but limited to open-loop testing.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured.
- Value: ⭐⭐⭐⭐ Provides a practical framework for VLM and E2E integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/drivemoe_mixture-of-experts_for_vision-language-action_model_in_end-to-end_auton.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[CVPR 2025\] RC-AutoCalib: An End-to-End Radar-Camera Automatic Calibration Network](rc-autocalib_an_end-to-end_radar-camera_automatic_calibration_network.md)
- [\[CVPR 2026\] E3AD: An Emotion-Aware Vision-Language-Action Model for Human-Centric End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/e3ad_an_emotion-aware_vision-language-action_model_for_human-centric_end-to-end_.md)

</div>

<!-- RELATED:END -->
