---
title: >-
  [Paper Note] SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic
description: >-
  [ECCV 2024][Autonomous Driving][Diffusion Transformer] SLEDGE proposes the first generative-model-based driving simulator. By utilizing a Raster-to-Vector Autoencoder to encode driving scenes into Rasterized Latent Maps (RLMs), and subsequently using a Diffusion Transformer to generate high-quality lane graphs and agents, it creates a simulation environment with 500x less storage (<4GB) than nuPlan. Meanwhile, it supports 500m long route evaluations…
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Diffusion Transformer"
  - "Driving Simulation"
  - "Lane Graph Generation"
  - "Latent Diffusion Model"
  - "Motion Planning"
date: 2026-05-08
content_hash: 4d45fc8e6616a45c
---

# SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic

**Conference**: ECCV 2024  
**arXiv**: [2403.17933](https://arxiv.org/abs/2403.17933)  
**Code**: [GitHub](https://github.com/autonomousvision/sledge)  
**Area**: Autonomous Driving  
**Keywords**: Diffusion Transformer, Driving Simulation, Lane Graph Generation, Latent Diffusion Model, Motion Planning

## TL;DR

SLEDGE proposes the first generative-model-based driving simulator. By utilizing a Raster-to-Vector Autoencoder to encode driving scenes into Rasterized Latent Maps (RLMs), and subsequently using a Diffusion Transformer to generate high-quality lane graphs and agents, it creates a simulation environment with 500x less storage (<4GB) than nuPlan. Meanwhile, it supports 500m long route evaluations, exposing a failure rate of over 40% in the SOTA planner PDM-Closed.

## Background & Motivation

**Background**: Data-driven driving simulators (such as nuPlan, Waymax) are key tools for evaluating autonomous driving planning algorithms. These simulators initialize the simulation environment by replaying abstract BEV representations (lanes, traffic lights, dynamic/static objects) from real-world driving logs.

**Limitations of Prior Work**:
   - **Huge Storage Requirements**: nuPlan contains 1300 hours of driving logs, requiring more than 2TB of storage space, which severely raises the barrier to research.
   - **Restricted Routes**: Log-replay-based simulators can only perform tests during short time frames (about 15 seconds) and along restricted routes, because once a planner deviates from the recorded route, the coverage of the simulation environment is no longer guaranteed.
   - **Poor Controllability**: It is difficult to flexibly adjust parameters such as traffic density and route difficulty for more comprehensive stress testing.

**Key Challenge**: Generative models have achieved grand success in the image synthesis domain, but the abstract representations of driving scenes (variable-length vector sets, topological connectivity, geometric precision requirements) are fundamentally different from the uniform grid structure of images, preventing modern generative models from being directly applied.

**Goal**: How to use generative models to synthesize driving scenes (including lane graphs and traffic agents) suitable for simulation, substituting replay methods that rely on massive logs.

**Key Insight**: Design a unified, rasterized representation that encodes all scene entities (lanes, traffic lights, vehicles, pedestrians, obstacles) into a fixed-size 2D latent space, making standard Diffusion Transformers directly applicable.

**Core Idea**: Map variable-length vectorized driving scenes to fixed-size Rasterized Latent Maps (RLMs) using a Raster-to-Vector Autoencoder (RVAE), and then perform latent diffusion generation with a DiT.

## Method

### Overall Architecture

The generation pipeline of SLEDGE consists of two primary stages: (1) training a Raster-to-Vector Autoencoder (RVAE) to learn the encoding and decoding of scenes; (2) training a Diffusion Transformer (DiT) on the frozen RLMs produced by the encoder. During inference, the DiT generates RLMs from noise, which are then reconstructed into vectorized scene entities by the RVAE's decoder, and finally utilized to initialize rule-based traffic simulations.

### Key Designs

1. **nuPlan Vector Representation (Scene State $\mathcal{S}$)**:

    - **Function**: Defines the complete scene state required for simulation.
    - **Mechanism**: The scene is composed of multiple entities:
        - Lanes $\mathbf{L} \in \mathbb{R}^{20 \times 2}$: Polylines consisting of 20 BEV points, combined with an adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ to encode topological connectivity.
        - Traffic lights: Red lights $\mathcal{R}$ and green lights $\mathcal{G}$, formatted as $20 \times 2$ polylines identical to lanes.
        - Traffic agents: Pedestrians $\mathcal{P}$, vehicles $\mathcal{V}$, and static obstacles $\mathcal{O}$ described by 2D centers, headings, extents, and velocities.
        - Ego velocity $\mathbf{v} \in \mathbb{R}^2$.
        - Complete scene state: $\mathcal{S} = \{\mathcal{M}, \mathcal{R}, \mathcal{G}, \mathcal{P}, \mathcal{V}, \mathcal{O}, \mathbf{v}\}$.
    - **Design Motivation**: Although this vectorized representation is the standard input format for simulators, direct modeling is difficult due to the variable number of entities and complex topological constraints.

2. **Rasterized State Image (RSI) and Raster-to-Vector Autoencoder (RVAE)**:

    - **Function**: Uniformly encodes variable-length vectorized scenes into fixed-size 2D Rasterized Latent Maps (RLMs).
    - **Mechanism**:
        - **Rasterization function** $\rho: \mathcal{S} \rightarrow \mathbf{I}$: Encodes the scene into a 12-channel Rasterized State Image $\mathbf{I} \in \mathbb{R}^{W \times H \times 12}$, where each entity type occupies 2 channels. Polylines are encoded with delta coordinates $\Delta = [dx, dy]$, dynamic objects are filled with 2D velocities, and static obstacles with heading vectors.
        - **Raster encoder** $\pi$: A ResNet-50 downsamples the RSI into a compact RLM $\mathbf{M} = \pi(\mathbf{I})$ with a shape of $8 \times 8 \times 64$.
        - **Channel Grouping**: RLM channels are divided into two groups $C = C_L + C_A$: a lane group ($8 \times 8 \times 32$) and an agent group ($8 \times 8 \times 32$).
        - **Vector decoder** $\phi$: A DETR-like Transformer decoder that uses the $1 \times 1$ spatially tokenized RLM as keys/values, decoding polyline coordinates, bounding box attributes, and existence probabilities $p \in [0,1]$ using learnable entity queries.
        - **Channel Group Masking**: Implements a binary mask in cross-attention—lane queries can only attend to lane tokens, and agent queries can only attend to agent tokens. This enables generating agents in a conditional manner given the lanes.
    - **Design Motivation**: Unifying heterogeneous entity types into a fixed-size 2D representation ensures compatibility with mainstream diffusion architectures, while channel grouping enables flexible conditional generation.

3. **Diffusion Transformer (DiT)**:

    - **Function**: Learns the data distribution in the RLM latent space to generate new driving scenes.
    - **Mechanism**:
        - **Training**: Adopts the DDPM framework. For the RLM $\mathbf{M}$ of each scene, noise is added as $\hat{\mathbf{M}} = \mathbf{M} + \sigma \boldsymbol{\mathcal{E}}$, and the DiT predicts the noise $\delta(\hat{\mathbf{M}}; \mathbf{c}, \sigma)$. The conditioning vector $\mathbf{c}$ is the city one-hot label (distinguishing right-hand traffic in US vs left-hand traffic in SG), injected via the AdaLN-Zero mechanism, optimizing the L2 reconstruction loss.
        - **Inference**: Generates RLMs by iteratively denoising from noise $\hat{\mathbf{M}} \sim \mathcal{N}(0, \sigma_{\max}^2 \mathbf{I})$. After decoding, entities with an existence probability $> \tau$ are kept, and overlapping bounding boxes are filtered via non-maximum suppression based on probability.
        - **Adjacency Matrix Recovery**: Extracts topological connectivity by matching lanes with endpoint distance < 1.5m and heading difference < 60°.
        - **Conditional Generation (Inpainting)**: Leverages the natural inpainting capability of diffusion models for two tasks: (a) generating agents given lanes (encoding lane tokens, denoising agent tokens); (b) route extrapolation (iteratively sampling new poses along the route, applying affine transformation to the previous tile's RSI, and filling unknown areas using the known regions as context).
    - **Design Motivation**: The DiT architecture is simple, scalable, and contains no down/upsampling layers, inherently compatible with RLMs of arbitrary spatial resolutions. The inpainting mechanism allows scenes to extend limitlessly.

4. **SLEDGE Simulation Environment**:

    - **Function**: Initializes reactive simulation using generated scenes.
    - **Mechanism**:
        - **Hard Routes**: Extracts multiple valid routes from the lane graph, choosing those with the most turns as "hard" routes.
        - **Hard Traffic**: Generates multiple traffic configurations for the same route, choosing those with the highest agent density as "hard" traffic.
        - **Behavior Simulation**: Non-ego vehicles are projected onto the nearest lane centerline and travel along it, with longitudinal control governed by the Intelligent Driver Model (IDM); pedestrians walk at a constant linear speed; traffic lights cycle every 15 seconds.
        - **Simulation Radius**: Simulates only those agents within a radius of $\alpha = 64m$ from the ego vehicle (distant agents remain stationary), enabling scalable simulation over 500m long routes (150 seconds).
    - **Design Motivation**: Overcomes route length limits of traditional simulators via a dynamic simulation radius, while hard routes/traffic configurations provide more challenging evaluations.

### Loss & Training

- **RVAE Training**:
    - Reconstruction loss: L1 error across all attributes computed after Hungarian matching.
    - Existence loss: Binary cross-entropy determining whether a query matches a ground-truth entity.
    - KL divergence loss: Regularizes the latent distribution of RLMs.
- **DiT Training**:
    - L2 noise reconstruction loss: $\|\boldsymbol{\mathcal{E}} - \delta(\hat{\mathbf{M}}; \mathbf{c}, \sigma)\|_2^2$
    - Noise scale $\sigma$ is sampled from a log-normal distribution.
- **Model Size**: DiT-L (138M params) and DiT-XL (487M params), patch size $1 \times 1$.
- **Data**: nuPlan dataset, 450k training frames + 50k validation frames, four cities, 64m × 64m FOV.

## Key Experimental Results

### Main Results: Lane Graph Reconstruction Quality

| Representation | Fixed Size | Channel Grouping | Size (KB) | GEO F1↑ | TOPO F1↑ | TOPO Chamfer↓ |
|---------|---------|---------|---------|---------|---------|---------------|
| RSI | ✓ | ✓ | 524.3 | 0.933 | 0.851 | 64.824 |
| RLM (No mask) | ✓ | ✗ | 16.0 | 0.981 | 0.945 | 20.096 |
| RLM (With mask) | ✓ | ✓ | 8.0 | 0.980 | 0.944 | 20.624 |
| Vector (Upper bound) | ✗ | ✓ | 4.8 | 0.997 | 0.990 | 4.174 |

### Main Results: Lane Graph Generation Quality

| Method | Representation | Route Length↑ | Precision(RVEnc)↑ | Recall(RVEnc)↑ | Reach↓ | Convenience↓ |
|------|------|----------|-------------------|----------------|--------|-------------|
| VAE | RSI | 2.68±3.66 | 0.00 | 0.16 | 2.86 | 13.06 |
| HDMapGen | Vector | 28.17±14.81 | 7.48 | 12.45 | 2.49 | 18.10 |
| DiT-L | RSI | 24.78±10.38 | 19.20 | 5.94 | 1.90 | 3.95 |
| DiT-L | RLM | 32.51±9.93 | **63.99** | **61.60** | 0.88 | 3.10 |
| DiT-XL | RLM | **35.37±10.28** | **78.07** | **72.63** | **0.20** | **0.47** |

### Simulation Results: PDM-Closed Planner Failure Rate

| Task | Route Length | Route/Traffic Difficulty | No. of Turns | No. of Vehicles | Failure Rate (PFR) |
|------|---------|-------------|--------|--------|------------|
| Replay | 100m | - | 0.89 | 57.40 | 0.06 |
| Lane→Agent | 100m | Easy/Easy | 0.89 | 44.61 | 0.07 |
| Lane→Agent | 500m | Hard/Hard | 4.20 | 170.87 | **0.44** |
| Lane&Agent | 100m | Easy/Easy | 0.61 | 27.30 | 0.22 |
| Lane&Agent | 500m | Hard/Hard | 3.82 | 169.66 | **0.49** |

### Key Findings

- The RLM representation achieves reconstruction quality near the upper bound (F1=0.980) at only 8KB, significantly outperforming the 524KB RSI representation in topological metrics.
- Channel grouping masks have virtually no impact on reconstruction quality while successfully enabling conditional generation.
- DiT-XL significantly outperforms other methods across all generation metrics, showing substantial scaling with compute while being less sensitive to data quantity (suggesting diversity matters more than quantity).
- 500m long route simulations reveal a fatal weakness in PDM-Closed: its inability to change lanes and overtake, which is rarely captured in current 15-second short simulations.
- Under hard routes and dense traffic, the failure rate of the SOTA planner surges from 6% to 49%.

## Highlights & Insights

1. **Extreme Compression**: Requires <4GB to completely configure the simulation environment, which is nearly a 500x compression compared to nuPlan's 2TB, dramatically lowering the entry barrier for research.
2. **Exquisite Unified Representation Design**: The 12-channel encoding scheme of RSI (direction vectors encoding polylines, velocity vectors encoding dynamic agents) is natural and compact; channel group masking elegantly switches between conditional and joint generation.
3. **Inpainting-based Route Extrapolation**: Leverages the natural inpainting capability of diffusion models to extend scenes limitlessly without extra training, which is an elegant concept.
4. **Evaluation Value**: Beyond being a generation tool, it exposes the vulnerabilities of current planning algorithms—namely, their fragility in long-distance driving and complex traffic scenarios.

## Limitations & Future Work

1. **Small FOV and Simulation Radius**: The 64m × 64m FOV and 64m simulation radius limit its applicability in high-speed scenarios.
2. **Over-simplified Lane Representation**: Uses only centerlines and encoding constant lane widths, lacking details like lane boundaries and markings.
3. **Simple Traffic Behaviors**: IDM and constant-velocity pedestrian models are idealized and lack realistic interaction behaviors.
4. **Insufficient Evaluation**: Lacks validation in downstream tasks such as reinforcement learning.
5. **High Computational Overhead**: Inference for diffusion models is computationally expensive; acceleration techniques such as Consistency Distillation could be considered.

## Related Work & Insights

- **Scenario Diffusion**: The closest pioneer work, using latent diffusion + raster decoders to generate vehicles, but does not support lane graph generation and long-range simulations.
- **HDMapGen**: Autoregressively generates lane graphs node-by-node, falling short of this paper's parallel generation scheme in quality and scalability.
- **DriveSceneGen**: Concurrent work that diffuses in image space to generate lanes and vehicles, but is more heuristic-heavy and less efficient.
- **Insights**: The "vector $\rightarrow$ raster $\rightarrow$ vector" encoding-decoding paradigm of RLM can be generalized to other structured scene generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first complete generative driving simulator; the RVAE+DiT architecture design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic representation comparisons, generation quality evaluations, scaling analysis, and downstream simulation validations compose a comprehensive evaluation suite.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definitions, well-structured methodology, and elegant, intuitive figures/tables.
- **Value**: ⭐⭐⭐⭐⭐ 500x storage compression is highly practical; the open-source code significantly lowers the research barrier.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DriveCombo: Benchmarking Compositional Traffic Rule Reasoning in Autonomous Driving](../../CVPR2026/autonomous_driving/drivecombo_benchmarking_compositional_traffic_rule_reasoning_in_autonomous_drivi.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[NeurIPS 2025\] SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration](../../NeurIPS2025/autonomous_driving/simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](../../ICCV2025/autonomous_driving/recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)

</div>

<!-- RELATED:END -->
