---
title: >-
  [Paper Note] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation
description: >-
  [CVPR 2025][Autonomous Driving][Closed-loop simulation] Proposes a high-fidelity closed-loop driving simulation framework based on 4D occupancy grids. It generates static scene occupancy from BEVs using OccDreamer, composes dynamic objects using Actor Bank, and generates multi-view videos conditioned on occupancy using VideoDreamer, reducing FVD by 44% and improving object detection mAP by 33%.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Closed-loop simulation"
  - "4D world model"
  - "occupancy grid diffusion"
  - "video generation"
date: 2026-05-08
content_hash: 524b4179d025f5c4
---

# DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation

**Conference**: CVPR 2025  
**arXiv**: [2411.11252](https://arxiv.org/abs/2411.11252)  
**Code**: [https://yanty123.github.io/DrivingSphere/](https://yanty123.github.io/DrivingSphere/)  
**Area**: Autonomous Driving  
**Keywords**: Closed-loop simulation, 4D world model, occupancy grid diffusion, video generation, autonomous driving

## TL;DR
Proposes a high-fidelity closed-loop driving simulation framework based on 4D occupancy grids. It generates static scene occupancy from BEVs using OccDreamer, composes dynamic objects using Actor Bank, and generates multi-view videos conditioned on occupancy using VideoDreamer, reducing FVD by 44% and improving object detection mAP by 33%.

## Background & Motivation

**Background**: Autonomous driving simulation requires high-fidelity visual rendering to test planning algorithms. Existing methods (MagicDrive, DriveArena) use 2D layouts or 3D BBoxes as scene conditions, which lack geometric precision.

**Limitations of Prior Work**: (1) 2D layout conditions cannot accurately represent 3D geometric relationships (occlusion, distance). (2) The generation quality of background scenes (buildings, vegetation) is poor. (3) The temporal/view consistency of dynamic objects (vehicles, pedestrians) is insufficient. (4) Text-guided scene editing is not supported.

**Key Challenge**: Closed-loop simulation requires pixel-level accurate multi-view videos, but existing conditional controls (2D/3D BBox) do not provide sufficient information to constrain high-fidelity generation.

**Goal**: To use 4D occupancy grids as an intermediate representation to provide richer geometric conditions than BBoxes, driving high-fidelity multi-view video generation.

**Key Insight**: The simulation is decoupled into two steps: first generating the 4D occupancy grid (composition of static background + dynamic objects), and then generating multi-view videos conditioned on the occupancy grid.

**Core Idea**: OccDreamer generates static scene occupancy + Actor Bank inserts dynamic objects $\rightarrow$ VideoDreamer generates spatio-temporally consistent multi-view videos from 4D occupancy conditions.

## Method

### Overall Architecture
BEV map $\rightarrow$ OccDreamer (VQVAE + CLIP conditional diffusion) generates static occupancy $\rightarrow$ Actor Bank provides dynamic object occupancy $\rightarrow$ 4D occupancy composition $\rightarrow$ VideoDreamer (ST-DiT + ControlNet) generates multi-view videos.

### Key Designs

1. **OccDreamer (Static Scene Generation)**:

    - **Function**: Generates the complete static scene occupancy grid from the BEV map.
    - **Mechanism**: VQVAE is first used to discretize the occupancy grid, followed by diffusion generation using CLIP conditioning + ControlNet (with BEV as the control signal). It supports scene expansion by performing extrapolation through overlapping regions.
    - **Design Motivation**: FID of 274 vs SemCity 634, demonstrating that diffusion generation yields significantly higher quality than traditional methods.

2. **VideoDreamer (Multi-view Video Generation)**:

    - **Function**: Generates spatio-temporally consistent multi-view videos from 4D occupancy conditions.
    - **Mechanism**: Spatio-temporal consistency is achieved via ST-DiT (Spatial-Temporal Diffusion Transformer) + VSSA (View-aware Spatial Self-Attention). ControlNet-DiT takes the rendered semantic map from the occupancy grid as a spatial condition. ID-aware actor encoding uses Fourier features to encode position/ID + T5 to encode descriptions, ensuring consistency of the same vehicle across frames.
    - **Design Motivation**: Standard diffusion models cannot guarantee multi-view and temporal consistency; VSSA + ID encoding addresses both issues.

3. **Autoregressive Temporal Generation**:

    - **Function**: Ensures the temporal continuity of long videos.
    - **Mechanism**: After generating a video segment, the final few frames are used as the conditioning input to generate the next segment.
    - **Design Motivation**: The quality of generating long videos in a single run degrades, making segmented autoregressive generation the most reliable solution currently.

### Loss & Training
Standard training for diffusion models. Tested on the nuScenes dataset. Supports text-guided occupancy generation.

## Key Experimental Results

### Main Results

| Method | FVD↓ | mAP↑ | NDS↑ | Lanes↑ |
|------|------|------|------|--------|
| MagicDrive | 218.12 | 12.92 | 28.36 | 21.95 |
| DriveArena | 185.32 | 16.06 | 30.03 | 26.14 |
| **DrivingSphere** | **103.42** | **21.45** | **34.16** | **27.99** |

### Ablation Study

| Component | Effect |
|------|------|
| OccDreamer FID | 274 vs SemCity 634 |
| Open-loop PDMS | 0.742 vs DriveArena 0.698 |
| Closed-loop ADS | 0.0851 vs DriveArena 0.0508 |

### Key Findings
- **4D occupancy conditions >> 2D/3D BBox conditions**: mAP 21.45 vs DriveArena 16.06 (+33%), FVD 103 vs 185 (-44%).
- **First to support text-guided occupancy generation**: Enables generating scenes of different styles using textual descriptions.
- **Greater advantage in closed-loop simulation** (ADS 0.085 vs 0.051), as 4D geometric consistency is more critical for continuous decision-making.

## Highlights & Insights
- The idea of using **4D occupancy grids as an intermediate representation for simulation** is highly promising—it provides richer geometry than BBoxes while remaining more controllable than raw rendering.
- The **two-step generation pipeline** (occupancy $\rightarrow$ video) decouples geometry and appearance, allowing each to be improved independently.

## Limitations & Future Work
- The resolution of the occupancy grid limits the precision of fine details.
- Evaluated only on nuScenes; not tested on larger-scale and more diverse scenarios.
- The number of closed-loop evaluation scenarios is limited.

## Related Work & Insights
- **vs MagicDrive**: MagicDrive uses 2D layout conditions, yielding poor geometric accuracy. DrivingSphere's occupancy condition provides a fundamental improvement.
- **vs DriveArena**: Also performs closed-loop simulation but relies on BBox conditions. DrivingSphere comprehensively outperforms it in both visual quality and downstream tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The 4D occupancy conditions and the two-step generation pipeline are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across open-loop and closed-loop evaluations, as well as multiple downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology.
- Value: ⭐⭐⭐⭐⭐ Significant engineering value for autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)
- [\[ECCV 2024\] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](../../ECCV2024/autonomous_driving/safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)
- [\[CVPR 2025\] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model](scenediffuser_city-scale_traffic_simulation_via_a_generative_world_model.md)
- [\[ECCV 2024\] NeuroNCAP: Photorealistic Closed-Loop Safety Testing for Autonomous Driving](../../ECCV2024/autonomous_driving/neuroncap_photorealistic_closed-loop_safety_testing_for_autonomous_driving.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
