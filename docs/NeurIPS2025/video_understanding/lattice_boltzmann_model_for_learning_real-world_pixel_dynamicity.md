---
title: >-
  [Paper Note] Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity
description: >-
  [NeurIPS 2025][Video Understanding][point tracking] Inspired by the Lattice Boltzmann Method from fluid dynamics, this work proposes LBM (Lattice Boltzmann Model) for online real-time pixel tracking. It models video pixels as fluid lattices and solves motion states via collision-streaming processes, achieving SOTA online tracking performance with 18M parameters while enabling real-time inference on edge devices.
tags:
  - NeurIPS 2025
  - Video Understanding
  - point tracking
  - online tracking
  - lattice Boltzmann
  - real-time
  - object tracking
date: 2026-05-08
content_hash: 3c669cb2a544c2b0
---

# Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity

**Conference**: NeurIPS 2025
**arXiv**: [2509.16527](https://arxiv.org/abs/2509.16527)
**Code**: [Project Page](https://george-zhuang.github.io/lbm)
**Area**: Video Understanding
**Keywords**: point tracking, online tracking, lattice Boltzmann, real-time, object tracking

## TL;DR

Inspired by the Lattice Boltzmann Method from fluid dynamics, this work proposes LBM (Lattice Boltzmann Model) for online real-time pixel tracking. It models video pixels as fluid lattices and solves motion states via collision-streaming processes, achieving SOTA online tracking performance with 18M parameters while enabling real-time inference on edge devices.

## Background & Motivation

- **Practical limitations of offline/semi-online methods**: Mainstream point tracking methods (TAPIR, CoTracker, LocoTrack) require full video or temporal window inputs, resulting in high memory consumption, unavoidable latency, inability to respond instantly to newly appearing pixels, and privacy/storage risks.
- **Dependency on spatiotemporal completeness is the fundamental bottleneck**: Offline methods rely on bidirectional temporal optimization, and semi-online methods rely on multiple iterations — LocoTrack's throughput drops by over 60% when increasing from 1 to 4 iterations.
- **Edge deployment requirements**: Applications such as robotic manipulation and medical vision require real-time inference on resource-constrained embedded devices.
- **Fragility of open-world object tracking**: Conventional MOT treats targets as holistic entities, leading to severe performance degradation under deformation, self-occlusion, and fast motion.

## Method

### Theoretical Foundation: Lattice Boltzmann Method

Classical LBM discretizes fluid into lattices, where the distribution function $\mathbf{f}$ undergoes collision and streaming processes:

$$\mathbf{f}(\mathbf{x}, t) = \sum_i [f_i(\mathbf{x} - \mathbf{c}_i \Delta t, t - \Delta t) + \Omega_i(\mathbf{x} - \mathbf{c}_i \Delta t, t - \Delta t)]$$

where $\mathbf{c}_i$ is the discrete velocity in the $i$-th direction and $\Omega$ is the collision operator describing relaxation toward equilibrium. Density and velocity are recovered as:

$$\rho(\mathbf{x},t) = \sum_i f_i(\mathbf{x},t), \quad \rho \mathbf{u}(\mathbf{x},t) = \sum_i \mathbf{c}_i f_i(\mathbf{x},t)$$

### LBM for Point Tracking

Given image $\mathbf{I} \in \mathbb{R}^{3 \times H \times W}$ and $N$ query points $\mathbf{q} \in \mathbb{R}^{N \times 2}$, the model estimates positions $\mathbf{p} \in \mathbb{R}^{N \times 2}$ and visibility $\mathbf{v} \in \mathbb{R}^N$ in subsequent frames.

**Visual Encoding**: The first three layers of an ImageNet-pretrained ResNet18 are used; all feature maps are upsampled to stride-4 and concatenated, yielding $\mathbf{o} \in \mathbb{R}^{d \times H/4 \times W/4}$. The design prioritizes efficiency.

**Distribution Initialization**: Bilinear sampling at query points: $\mathbf{f}_{init} = \text{BilinearSample}(\mathbf{o}, \mathbf{q}) \in \mathbb{R}^{N \times d}$

**Predict**: Unlike classical LBM with fixed neighborhoods, LBM employs learnable neighborhoods $\delta$ to compute collision interactions:

$$\mathbf{f}(x, t | \delta) = \mathbf{f}(x, t - \Delta t | \delta) + \Omega(x, t - \Delta t | \delta)$$

The collision operator $\Omega$ is implemented via deformable attention. Temporal context extends over $N_s$ historical steps, maintaining streaming distribution $\mathbf{f}_s$ and collision distribution $\mathbf{f}_c$:

$$\mathbf{f} = \phi_c(\phi_s(\mathbf{f}_{init}, \mathbf{f}_s), \mathbf{f}_c)$$

where $\phi_s, \phi_c$ are cross-attention modules.

**Update**: A correlation map is computed between pixel distributions and visual features; top-$k$ responses are selected as reference points $\mathbf{r}$, and the distribution is updated via deformable attention $\psi$: $\psi(\mathbf{f}, \mathbf{o}, \mathbf{r})$.

**Multi-layer Predict-Update Transformer**: Multiple layers are stacked, each comprising a predict step and an update step. The number of reference points decreases progressively across layers, with the final layer retaining a single deterministic reference point $\mathbf{r}_{last}$.

**Output Heads**: The tracking head predicts offset $\Delta \mathbf{p} = \mathcal{H}_{track}(\mathbf{f}, \mathbf{o}, \mathbf{r}_{last})$; the visibility head predicts confidence and visibility $\{\rho, \mathbf{v}\} = \mathcal{H}_{vis}(\mathbf{f}, \mathbf{o}, \mathbf{r}_{last})$.

### Loss & Training

$$\mathcal{L} = \lambda_{cls} \mathcal{L}_{cls} + \mathcal{L}_{reg} + \mathcal{L}_{vis} + \mathcal{L}_{conf}$$

- $\mathcal{L}_{cls}$: Cross-entropy loss on correlation maps (per layer)
- $\mathcal{L}_{reg}$: L1 loss on offsets (visible points only)
- $\mathcal{L}_{vis}$: Visibility cross-entropy
- $\mathcal{L}_{conf}$: Confidence cross-entropy (positive when $\|\mathbf{p} - \mathbf{p}_{gt}\| < 8$)

### LBM for Object Tracking

Targets are decomposed into fine-grained pixel sets, and object associations are established via pixel tracking:
- **Initialization**: $N$ pixels are randomly sampled within detection bounding boxes.
- **Matching**: After predicting pixel positions and visibility, spatial correspondences between predicted pixels and pixels within detection boxes in the new frame are evaluated.
- **Dynamic Update**: Pixels persistently outside the box are removed (outlier pruning), while new pixels within the current box are supplemented (inlier replenishment) to maintain a robust representation.

## Key Experimental Results

### Table 1: Real-World Point Tracking Performance (TAP-Vid DAVIS / Kinetics / RoboTAP)

| Model | Params | Type | DAVIS AJ↑ | DAVIS $\delta_{avg}^x$↑ | DAVIS OA↑ | Kinetics AJ↑ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| TAPIR | 31M | Offline | 56.2 | 70.0 | 86.5 | 49.6 |
| LocoTrack | 12M | Offline | 62.9 | 75.3 | 87.2 | 52.9 |
| CoTracker3 | 25M | Window Online | 64.5 | 76.7 | 89.7 | 54.1 |
| Track-On | 49M | Online | 65.0 | 78.0 | 90.8 | 53.9 |
| **LBM** | **18M** | **Online** | **65.1** | **77.5** | **89.5** | **53.4** |

LBM achieves SOTA online performance (DAVIS AJ 65.1) with only 18M parameters — 37% of Track-On's parameter count — while surpassing most offline and window-online methods.

### Edge Device Efficiency

On NVIDIA Jetson Orin NX Super: LBM achieves **14.3 FPS** real-time inference, **3.9×** faster than Track-On.

### Table 2: Open-World Object Tracking (TAO Validation Set)

| Model | Extra Training | TETA↑ | LocA↑ | AssocA↑ |
|:---|:---:|:---:|:---:|:---:|
| MASA | Yes | 37.1 | 51.8 | 35.8 |
| NetTrack | No | 36.1 | 50.2 | 31.0 |
| **LBM** | **No** | **37.4** | **51.7** | **35.1** |

Without training on any object tracking domain data, LBM achieves SOTA performance (TETA 37.4), outperforming MASA which requires additional training.

### BFT and OVT-B Datasets

LBM achieves OWTA 50.3 on BFT and TETA 41.2 on OVT-B, both best among methods without additional training.

## Highlights & Insights

- **Elegant physical analogy**: Modeling pixel motion as collision-streaming processes over fluid lattices yields a theoretically coherent and computationally efficient formulation.
- **Extreme efficiency**: 18M parameters with real-time edge inference (14.3 FPS), achieving the highest parameter efficiency among all online methods.
- **Unified framework across levels**: The same LBM handles both point tracking and object tracking, seamlessly bridging the two via pixel decomposition and dynamic pruning.
- **Training-free transfer**: Achieves zero-shot superiority over task-specifically trained methods on TAO, BFT, and OVT-B.

## Limitations & Future Work

- **Inherent limitations of online methods**: Future frame information cannot be exploited for bidirectional optimization; recovery from severe occlusion may be weaker than offline methods.
- **Limited encoder capacity of ResNet18**: The lightweight encoder chosen for efficiency may have insufficient representational power in scenes with complex textures.
- **Slightly below Track-On on TAP-Vid Kinetics**: AJ 53.4 vs. 53.9, indicating room for improvement on more diverse video scenarios.
- **Fixed history length $N_s$**: The temporal context window is fixed, potentially limiting adaptability to extremely long occlusions.

## Related Work & Insights

- **Offline point tracking** (TAPIR [Doersch+ 2023], LocoTrack [Cho+ 2024], CoTracker3 [Karaev+ 2024]): Strong performance but high latency and incompatible with real-time requirements.
- **Online point tracking** (MFT [Neoral+ 2024], Track-On [Aydemir+ 2025], DOT [Le Moing+ 2024]): LBM substantially outperforms all prior methods in efficiency.
- **Open-world object tracking** (MASA [Li+ 2024], NetTrack [Zheng+ 2024]): LBM generalizes point tracking advantages to object tracking.
- **Lattice Boltzmann Method** ([Mohamad 2011]): A classical fluid simulation technique; LBM represents its first application to visual tracking.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Introduces the Lattice Boltzmann analogy from fluid mechanics; the online tracking paradigm is genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both point tracking and object tracking tasks, with edge device efficiency evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Physical analogy is clearly articulated; framework diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Provides an efficient and unified solution for real-time online tracking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[CVPR 2026\] Real-World Point Tracking with Verifier-Guided Pseudo-Labeling](../../CVPR2026/video_understanding/realworld_point_tracking_with_verifierguided_pseud.md)
- [\[AAAI 2026\] UVLM: Benchmarking Video Language Model for Underwater World Understanding](../../AAAI2026/video_understanding/uvlm_benchmarking_video_language_model_for_underwater_world_understanding.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](pass_path-selective_state_space_model_for_event-based_recognition.md)

</div>

<!-- RELATED:END -->
