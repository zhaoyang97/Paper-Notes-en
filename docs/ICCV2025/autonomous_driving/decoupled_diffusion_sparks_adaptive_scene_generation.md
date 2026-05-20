---
title: >-
  [Paper Note] Decoupled Diffusion Sparks Adaptive Scene Generation
description: >-
  [ICCV2025][Autonomous Driving][Scene Generation] This paper proposes Nexus, an adaptive driving scene generation framework based on decoupled diffusion. By assigning independent noise states to individual tokens…
tags:
  - "ICCV2025"
  - "Autonomous Driving"
  - "Scene Generation"
  - "Decoupled Diffusion"
  - "Autonomous Driving Simulation"
  - "Safety-Critical Scenes"
  - "Traffic Layout"
date: 2026-05-08
content_hash: 5bff3410beffce5b
---

# Decoupled Diffusion Sparks Adaptive Scene Generation

**Conference**: ICCV2025
**arXiv**: [2504.10485](https://arxiv.org/abs/2504.10485)  
**Code**: [opendrivelab.com/Nexus](https://opendrivelab.com/Nexus)  
**Area**: Autonomous Driving
**Keywords**: Scene Generation, Decoupled Diffusion, Autonomous Driving Simulation, Safety-Critical Scenes, Traffic Layout

## TL;DR

This paper proposes Nexus, an adaptive driving scene generation framework based on decoupled diffusion. By assigning independent noise states to individual tokens, Nexus unifies goal-oriented and reactive generation, reduces displacement error by 40%, and introduces Nexus-Data, a dataset comprising 540 hours of safety-critical driving scenarios.

## Background & Motivation

- Diversity is essential in autonomous driving datasets, yet safety-critical long-tail scenarios remain extremely scarce.
- Existing scene generation methods face two fundamental tensions:
    - **Full-sequence diffusion** (e.g., SceneDiffuser): supports goal-oriented generation via inpainting, but cannot respond promptly to environmental changes.
    - **Autoregressive prediction** (e.g., GUMP): enables real-time environmental feedback, but lacks awareness of goal states for precise control.
- No existing method simultaneously provides **Reactivity** and **Goal Orientation**.
- Publicly available datasets predominantly capture safe driving behaviors, with insufficient coverage of high-risk scenarios.

## Method

### Core Idea: Noise as Soft Masks

- Key insight: different noise levels are interpreted as masks of varying intensity.
    - Low-noise tokens → confirmed goal/historical states (analogous to hard masks).
    - High-noise tokens → future states to be generated.
- This formulation unifies diffusion models (noise-axis masking) and autoregressive prediction (time-axis masking) into a **tri-axis masked modeling** framework.
- Each token is assigned an independent noise level, forming a noise matrix $\mathbf{k} \in (0,1]^{A \times \mathcal{T}}$.

### Scene Encoding and Representation

- **Agent Tensor**: $\mathbf{x} \in \mathbb{R}^{A \times \mathcal{T} \times D}$, encoding coordinates, heading, velocity, and size.
- **Map Tensor**: $\mathbf{c} \in \mathbb{R}^{L \times N \times D'}$, encoding road topology.
- Perceiver IO is used to compress map information into fixed-length tokens.
- Agent tokens are perturbed with independently sampled noise and encoded with 2D rotary positional embeddings (physical time + denoising step).

### Noise-Masked Training (Goal Orientation)

- During training, noise levels are sampled independently per token rather than uniformly across the full sequence.
- The model learns to reconstruct complete sequences from partially soft-masked inputs, conditioned on low-noise guidance tokens.
- Optimization objective: $\forall \mathbf{k} \in (0,1]^{A \times \mathcal{T}}, \min_\theta \mathbb{E} \|(\epsilon - \epsilon_\theta(g(\mathbf{x}^0, \mathbf{k}); \mathbf{c}, \mathbf{k}))\|_2^2$
- At inference: history and goal tokens are assigned low noise; remaining tokens receive high noise, enabling conditional generation.

### Diffusion Transformer Architecture

- Built upon DiT with multiple attention interaction types:
    - **Map cross-attention**: agents query map tokens to model agent–map interactions.
    - **Temporal attention**: captures trajectory continuity.
    - **Spatial attention**: models spatial interactions (e.g., car-following, yielding).
- Validity masks exclude invalid or skipped tokens.
- AdaLN is used to condition transformer blocks.

### Noise-Aware Scheduling (Reactivity)

- A scheduling matrix $\mathcal{K} \in [\mathbf{k}]^M$ encodes the noise level progression of each agent at each timestep.
- **Chunk mechanism**:
    - Each chunk contains historical frames, frames to be denoised, and optional goal tokens.
    - After each denoising step, the lowest-noise token is popped (generation complete) and a high-noise frame is pushed in.
    - Environmental changes can directly overwrite agent states with reduced noise levels.
- **Scheduling strategies**:
    - **Pyramid scheduling**: tokens enter and exit from one end of the chunk, enabling sequential generation.
    - **Trapezoid scheduling**: tokens enter and exit from both ends, supporting bidirectional goal conditioning.
    - Both strategies achieve a response latency of only 0.16 seconds (vs. 4.96 seconds for full-sequence methods).

### Behavior-Aligned Classifier Guidance

- A correction function is applied at each denoising step:
    - Separates overlapping agents along the centerline in the reverse direction (collision avoidance).
    - Smooths trajectories.
    - Attracts agents toward the nearest lane (road-keeping).

### Nexus-Data: Safety-Critical Scene Dataset

- Generated using the MetaDrive simulator with ScenarioNet-formatted scenes.
- High-risk interactions (cut-ins, hard braking, collisions) are produced via CAT adversarial learning.
- Automated filtering retains only 36.9% of scenarios with valid collisions, further excluding off-road cases and invalid trajectories.
- The final dataset comprises 540 hours of high-quality safety-critical driving scenarios.

## Key Experimental Results

### Main Generation Performance (nuPlan Dataset, 8-second Prediction)

| Method | ADE↓ | Off-road Rate↓ | Collision Rate↓ | Instability↓ | Time (s) |
|--------|------|----------------|-----------------|--------------|----------|
| IDM | 10.52 | 9.85 | 10.17 | 6.30 | 2.16 |
| Diffusion Policy | 7.80 | 13.9 | 14.92 | 12.71 | 6.59 |
| SceneDiffuser | 5.99 | 8.53 | 11.78 | 9.64 | 5.34 |
| GUMP | 1.93 | 7.73 | 7.85 | 16.18 | 5.59 |
| **Nexus** | **1.28** | 6.89 | **1.62** | **4.63** | 2.79 |
| **Nexus-Full** | **1.12** | **6.25** | **1.56** | **3.17** | 2.93 |

### Scheduling Strategy Comparison

| Scheduling Strategy | ADE↓ | Response Time (s) | Total Time (s) |
|---------------------|------|-------------------|----------------|
| Autoregressive | 1.48 | 4.96 | 79.36 |
| Full-sequence | 1.28 | 4.96 | 4.96 |
| Pyramid | 1.53 | **0.16** | 7.68 |
| Trapezoid | 1.39 | **0.16** | 6.20 |
| Trapezoid + Feedback | **1.17** | **0.16** | 6.20 |

### Ablation Study

| Component | ADE↓ (Conditional Generation) |
|-----------|-------------------------------|
| Baseline (Diffusion Policy) | 7.53 |
| + Noise-masked training | 3.42 |
| + Positional encoding | 1.44 |
| + Nexus-Data | 1.32 |
| + Classifier guidance | 1.25 |

### Closed-Loop Driving World Generator Evaluation

| Method | Reactive Score↑ | Collision Score↑ | Progress Score↑ |
|--------|----------------|------------------|-----------------|
| Oracle (GT) | 82.8 | 89.5 | 97.0 |
| Diffusion Policy | 61.6 | 81.9 | 90.2 |
| SceneDiffuser | 57.2 | 74.7 | 91.6 |
| **Nexus** | **73.0** | **84.9** | **95.0** |

### Data Augmentation Effect

Augmenting planning model training with synthetic data generated by Nexus improves the closed-loop score from 48.11 to 57.86 (+20%).

## Highlights & Insights

**Highlights**:
- First framework to unify goal-oriented generation and real-time reactivity in scene generation.
- Noise-masked training elegantly unifies diffusion and autoregressive prediction under a single formulation.
- Response latency of only 0.16 seconds, suitable for online closed-loop simulation.
- Collision rate of only 1.56%, substantially lower than all baselines.
- Nexus-Data provides large-scale safety-critical scenario data for model training.

**Limitations & Future Work**:
- Generates only structured traffic layouts; does not directly synthesize video.
- Closed-loop training of end-to-end driving models still requires video synthesis support.
- Insufficient synthetic data volume (3×) degrades performance; adequate scale (30×+) is necessary.

## Personal Reflections

- Treating noise states as soft masks is an elegant unifying framework that fundamentally addresses the question of whether diffusion and autoregressive models can be reconciled.
- The chunk sliding window design endows diffusion models with genuine online generation capability for the first time.
- The construction methodology of Nexus-Data (adversarial learning + automated filtering) offers a paradigm for large-scale acquisition of safety-critical scenarios.
- The data augmentation experiment (60× synthetic data → +20% closed-loop score) compellingly demonstrates the practical value of scene generation.
- Integration with NeRF illustrates a complete pipeline from layout generation to visual rendering, representing an important component of world models.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](../../NeurIPS2025/autonomous_driving/cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)

</div>

<!-- RELATED:END -->
