---
title: >-
  [Paper Note] Open-World Drone Active Tracking with Goal-Centered Rewards
description: >-
  [NeurIPS 2025][Video Understanding][Drone Active Tracking] This paper introduces DAT, the first open-world drone active tracking benchmark comprising 24 city-scale scenes with high-fidelity dynamics simulation, along with GC-VAT, a reinforcement learning tracking method based on goal-centered rewards and curriculum learning, achieving approximately 72% tracking success rate in simulation.
tags:
  - NeurIPS 2025
  - Video Understanding
  - Drone Active Tracking
  - Reinforcement Learning
  - Goal-Centered Reward
  - Curriculum Learning
  - Open-World Benchmark
date: 2026-05-08
content_hash: 06c350d701458037
---

# Open-World Drone Active Tracking with Goal-Centered Rewards

**Conference**: NeurIPS 2025
**arXiv**: [2412.00744](https://arxiv.org/abs/2412.00744)
**Code**: [DAT_Benchmark](https://github.com/SHWplus/DAT_Benchmark)
**Area**: Video Understanding / Drone Tracking
**Keywords**: Drone Active Tracking, Reinforcement Learning, Goal-Centered Reward, Curriculum Learning, Open-World Benchmark

## TL;DR

This paper introduces DAT, the first open-world drone active tracking benchmark comprising 24 city-scale scenes with high-fidelity dynamics simulation, along with GC-VAT, a reinforcement learning tracking method based on goal-centered rewards and curriculum learning, achieving approximately 72% tracking success rate in simulation.

## Background & Motivation

**Background**: Visual Active Tracking (VAT) aims to autonomously follow targets by controlling a motion system, with broad applications in drone surveillance and security. RL-based VAT methods integrate visual tracking and control into a unified framework, eliminating the need for manual annotation and additional tuning required by pipeline-based approaches.

**Limitations of Prior Work**:
- **Lack of unified benchmarks**: Existing scenarios exhibit low complexity, neglect tracker dynamics, or rely on overly simplified models, making them insufficient for validating algorithm performance. Prior methods employ rule-based target management, which is far from human-like behavior.
- **Complex interference in open-world settings**: Large-scale dynamic environments involve frequent occlusions and distractors. Prior methods capture images only from a fixed horizontal viewpoint, limiting perceptual and motion range.

**Key Challenge**: Real-world drone tracking must operate in complex open-world environments, yet neither existing simulation environments nor reward designs meet this requirement—distance-based rewards become misleading under top-down viewpoints.

**Goal**: (1) Construct a realistic and comprehensive drone active tracking benchmark; (2) Design reward functions and efficient training strategies applicable to non-fixed viewpoints.

**Key Insight**: Drawing from projective geometry, a deviation metric-based reward function is designed to replace Euclidean distance rewards, combined with curriculum learning.

**Core Idea**: Replace Euclidean distance with a normalized deviation metric based on image-center projection, enabling rewards to correctly reflect target position under arbitrary viewpoints.

## Method

### Overall Architecture

Drone active tracking is formulated as a Markov Decision Process $\langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \gamma, \mathcal{T} \rangle$:
- **State**: $84 \times 84$ RGB image
- **Action**: Discrete action set (forward, backward, left, right, rotate left, rotate right, stop)
- **Network**: CNN + GRU Drone Agent
- **Algorithm**: PPO (Proximal Policy Optimization)

### Key Designs

#### 1. DAT Benchmark

- **24 city-scale scenes**: 6 outdoor scene types $\times$ 4 weather conditions (daytime, fog, night, snow)
- **Digital twin toolchain**: Automatically generates 3D scenes from arbitrary OpenStreetMap regions
- **High-fidelity drone dynamics**: Simulates the mass, inertia, aerodynamic properties, and gimbal response of DJI Matrice 100 using Webots
- **Human-like target behavior**: Integrates the SUMO traffic simulator to manage behaviors of 24 tracking targets (cars, motorcycles, pedestrians, wheeled/legged robots)
- **7-dimensional scene complexity**: Scene area, building density, color richness, road density, terrain density, tree density, tunnel density

#### 2. Goal-Centered Reward (Core of GC-VAT)

**Problem**: When the drone pitches downward, the image plane is no longer parallel to the ground and the projection becomes trapezoidal. The Euclidean distance between the target and image center fails to accurately reflect the actual spatial relationship in the image plane.

**Deviation metric**:

$$\phi(P_g, C_g) = \frac{|P_g - C_g|}{|E_g(P_g, C_g) - C_g|}$$

where $P_g$ is the target point, $C_g$ is the image-center projection, and $E_g$ is the intersection of the connecting line with the projection boundary. This metric maps all points at the same relative position to the same value.

**Reward function**:

$$r_{gc}(P_g) = \begin{cases} \tanh(\alpha(1-\phi(P_g, C_g))^3), & P_g \in \mathcal{I}_{clip} \\ 0, & \text{otherwise} \end{cases}$$

where $\alpha = 4$ and $\lambda_{clip} = 0.7$. The $\tanh$ function provides a strong signal at the image center, and the clipping range prevents the target from remaining near the edges.

**Theoretical guarantee (Proposition 1)**: When the camera is not in a fixed horizontal forward-facing orientation, Euclidean distance-based rewards may assign lower values to targets that are actually closer to the center, causing training failure.

#### 3. Curriculum-Based Training (CBT)

Training proceeds in two stages:
- **Stage 1**: Simplified environment (linear target trajectory + no obstacles), learning basic goal-centering capability
- **Stage 2**: Complex environment (diverse target motion + obstacles/occlusions), enhancing generalization
- **Transition condition**: Automatically switches when average reward reaches threshold $\eta$

#### 4. Domain Randomization

The drone's initial position and orientation relative to the target, as well as the gimbal pitch angle, are randomized to promote diverse behavioral exploration.

### Loss & Training

The PPO algorithm is used with reward $r_{gc}$ (Eq. 3), combined with domain randomization and curriculum learning strategies.

## Key Experimental Results

### Main Results: In-Scene Performance

| Method | Avg. CR | Avg. TSR |
|------|---------|----------|
| AOT | ~44 | ~0.22 |
| D-VAT | ~35 | ~0.19 |
| **GC-VAT (Ours)** | **~242** | **~0.72** |

Compared to D-VAT, CR improves by **591%** and TSR by **279%**.

### Cross-Scene / Cross-Domain Generalization

| Test Condition | GC-VAT CR | GC-VAT TSR | TSR Gain vs. D-VAT |
|----------|-----------|------------|---------------------|
| Cross-scene | 176 (avg) | 0.57 | +200% |
| Cross-domain (night) | 217 | 0.64 | — |
| Cross-domain (fog) | 243 | 0.76 | — |
| Cross-domain (snow) | 178 | 0.60 | — |
| Cross-domain avg. | 213 | 0.67 | +253% |

### Ablation Study

| Configuration | In-Scene TSR | Cross-Scene TSR | Cross-Domain TSR |
|------|----------|----------|---------|
| With D-VAT reward | 0.06 | 0.05 | 0.06 |
| Without CBT | 0.23 | 0.26 | 0.23 |
| Without angle randomization (AR) | 0.44 | 0.37 | 0.36 |
| Without height randomization (HR) | 0.49 | 0.48 | 0.57 |
| Without vertical randomization (VR) | 0.63 | 0.54 | 0.60 |
| Without pitch randomization (PR) | 0.61 | 0.48 | 0.52 |
| **Full GC-VAT** | **0.68** | **0.54** | **0.65** |

### Key Findings

1. **D-VAT/AOT reward functions completely fail under top-down viewpoints**: Training curves rapidly degrade or plateau, validating the theoretical analysis.
2. **Angle randomization (AR) contributes most**: Removing it causes a substantial drop in TSR, indicating that multi-angle initialization is critical for policy exploration.
3. **Robustness**: TSR decreases by $< 0.06$ under wind disturbance and by $< 0.07$ under raindrop blur.
4. **Distractors and novel targets**: With similar vehicle distractors, TSR = 0.91 (vs. 0.94); for an unseen bus target, TSR = 0.79.

## Highlights & Insights

1. **Integration of theory and practice**: Beyond designing a new reward function, the paper provides a mathematical proof of distance metric failure, which is highly convincing.
2. **Comprehensive benchmark contribution**: With 24 scenes, a digital twin toolchain, and human-like behavior simulation, the benchmark itself may be of greater value than the method.
3. **Elegance of the deviation metric**: Normalizing to the projection boundary effectively eliminates the influence of viewpoint variation, yielding an intuitive and elegant design.
4. **Significant impact of curriculum learning**: Without CBT, the policy completely fails to learn (TSR only 0.23), illustrating the training difficulty of open-world scenarios.

## Limitations & Future Work

1. **Discrete action space**: Continuous action spaces may enable finer-grained control.
2. **RGB-only input**: Depth and other multimodal information are not utilized.
3. **Single-target tracking**: Multi-target scenarios are not considered.
4. **Reward design relies on geometric priors**: A flat ground assumption is made, which may require adjustment in complex terrain.
5. **Limitations of end-to-end methods**: An explicit target re-detection mechanism is absent when the target is fully occluded.

## Related Work & Insights

- **AD-VAT+/D-VAT/AOT**: Primary baselines, all using distance-based rewards in simple scenes.
- **PPO**: Leveraged as the base RL algorithm for its stability in control tasks.
- **Curriculum learning**: Inspired by progressive training paradigms, particularly effective in complex RL environments.
- **Implications for visual tracking**: In sim-to-real frameworks, well-designed rewards are more important than complex network architectures.

## Rating

⭐⭐⭐⭐

A systematic contribution with value in both the benchmark and the method. The reward design is theoretically grounded and thoroughly validated, though the tracking method itself (CNN + GRU + PPO) is relatively standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels](trackingworld_world-centric_monocular_3d_tracking_of_almost_all_pixels.md)
- [\[ICCV 2025\] Attention to Trajectory: Trajectory-Aware Open-Vocabulary Tracking](../../ICCV2025/video_understanding/attention_to_trajectory_trajectory-aware_open-vocabulary_tracking.md)
- [\[NeurIPS 2025\] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity](lattice_boltzmann_model_for_learning_real-world_pixel_dynamicity.md)
- [\[CVPR 2026\] CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](../../CVPR2026/video_understanding/cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)

</div>

<!-- RELATED:END -->
