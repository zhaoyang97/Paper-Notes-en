---
title: >-
  [Paper Note] Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network
description: >-
  [AAAI 2026][3D Vision][Garment Simulation] This paper proposes Pb4U-GNet, which decouples message propagation from feature update (Propagation-before-Update) and incorporates resolution-aware propagation depth control and update scaling mechanisms, enabling garment simulation models trained solely on low-resolution meshes to generalize to high-resolution meshes.
tags:
  - AAAI 2026
  - 3D Vision
  - Garment Simulation
  - Graph Neural Network
  - Cross-Resolution Generalization
  - Message Propagation
  - Resolution Adaptation
date: 2026-05-08
content_hash: 9c988527d6a849eb
---

# Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network

**Conference**: AAAI 2026
**arXiv**: [2601.15110](https://arxiv.org/abs/2601.15110)
**Code**: [github.com/adam-lau709/PB4U-GNet](https://github.com/adam-lau709/PB4U-GNet)
**Area**: 3D Vision
**Keywords**: Garment Simulation, Graph Neural Network, Cross-Resolution Generalization, Message Propagation, Resolution Adaptation

## TL;DR

This paper proposes Pb4U-GNet, which decouples message propagation from feature update (Propagation-before-Update) and incorporates resolution-aware propagation depth control and update scaling mechanisms, enabling garment simulation models trained solely on low-resolution meshes to generalize to high-resolution meshes.

## Background & Motivation

Garment simulation is a core technology for applications such as virtual try-on and digital human modeling. Traditional physics-based methods (e.g., mass-spring systems) incur prohibitive computational costs, particularly on high-resolution meshes that require repeated constraint solving. Graph neural networks (GNNs) have emerged as effective acceleration alternatives, yet **existing GNN methods suffer from severe cross-resolution generalization failures**—performance degrades sharply on meshes outside the training resolution, especially at higher resolutions.

The authors identify two fundamental causes of cross-resolution failure:

**Fixed message propagation depth**: Existing GNNs use a fixed number of propagation layers, restricting each vertex to perceiving neighbors within a predefined hop count. On fine meshes, fixed depth leads to insufficient receptive field coverage; on coarse meshes, it causes over-smoothing.

**Resolution-dependent displacement magnitude**: In higher-resolution meshes, the same global motion is distributed across more vertices, resulting in smaller per-vertex displacements. This intrinsic resolution dependency causes models trained at low resolution to overestimate displacements at high resolution.

These two issues reveal a core contradiction: training directly on high-resolution meshes is computationally prohibitive, yet models trained at low resolution fail to generalize. This is the central challenge this paper addresses.

## Method

### Overall Architecture

The core innovation of Pb4U-GNet is **decoupling message propagation from feature update**. In conventional GNNs, each layer simultaneously performs message aggregation and feature update. Pb4U-GNet instead conducts $K$ steps of pure message propagation to expand the receptive field, followed by a single unified feature update. This decoupled design allows the propagation depth $K$ to be flexibly adjusted according to resolution without affecting the update frequency.

The overall pipeline is: input current mesh state $\mathbf{X}_t$ → vertex/edge encoders → $K$-step message propagation → feature update → 15-layer MeshGraphNet refinement → vertex decoder predicting acceleration → resolution-aware scaling → forward Euler integration to obtain the next-timestep positions.

### Key Designs

#### 1. **Propagation-before-Update (PbU): Decoupled Propagation and Update**

**Core Idea**: Pure message aggregation is first performed to accumulate neighborhood information, followed by a unified feature update.

During the propagation phase, each vertex maintains an aggregated feature vector $\mathbf{h}_{t,i}$, initialized to the vertex embedding $\mathbf{v}_{t,i}$. At each step $k$, neighborhood information is aggregated via a learnable message function $f_m(\cdot)$ (implemented as an MLP):

$$\tilde{\mathbf{h}}^k_{t,i} = \text{LayerNorm}\left(\sum_{j \in \mathcal{N}(i)} f_m(\mathbf{h}^{k-1}_{t,i}, \mathbf{h}^{k-1}_{t,j}, \mathbf{e}_{t,ij})\right)$$

New and historical information are then fused via decay accumulation:

$$\mathbf{h}^k_{t,i} = \gamma \cdot \mathbf{h}^{k-1}_{t,i} + \tilde{\mathbf{h}}^k_{t,i}$$

where $\gamma$ is a decay factor controlling the influence of historical messages. After $K$ propagation steps, an update function $f_u$ (MLP) integrates the original embedding with the accumulated features:

$$\mathbf{v}'_{t,i} = f_u(\mathbf{v}_{t,i}, \mathbf{h}^K_{t,i})$$

**Design Motivation**: Decoupling ensures that receptive field size is determined solely by the propagation step count $K$, independent of update frequency, enabling flexible adaptation to different resolutions.

#### 2. **Resolution-Aware Propagation Control**

**Core Idea**: Dynamically adjust propagation steps $K$ based on mesh density to maintain a consistent physical propagation distance.

The effective physical propagation distance is defined as $D = K_{\text{base}} \times \bar{L}_{\text{base}}$, where $K_{\text{base}}$ is the number of propagation steps at the reference resolution and $\bar{L}_{\text{base}}$ is the mean edge length at the reference resolution. For meshes at any resolution:

$$K = \lfloor D \times \bar{L}^{-1} \rfloor$$

As resolution increases (i.e., $\bar{L}$ decreases), $K$ scales proportionally to maintain consistent physical receptive field coverage.

**Design Motivation**: Grounded in the physical intuition that elastic wave propagation distance in cloth is independent of mesh discretization; thus, the same physical propagation distance should be maintained across different resolutions.

#### 3. **Resolution-Aware Update Scaling**

**Core Idea**: Scale predicted accelerations according to the local geometric scale of each vertex.

Based on the geometric similarity principle from continuum mechanics (displacement fields scale linearly with element size), a per-vertex scaling factor is computed as:

$$\mathbf{s}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} l_{ij}$$

i.e., the mean length of all edges connected to vertex $i$ in the rest state. The final acceleration is $\mathbf{A}_{g,t} = \mathbf{S} \odot \tilde{\mathbf{A}}_{g,t}$.

**Design Motivation**: In higher-resolution meshes, each vertex represents a smaller area and mass; thus, per-vertex acceleration should be smaller for the same global deformation. This scaling restores physically consistent displacement magnitudes.

### Loss & Training

The model is trained in a **fully self-supervised** manner without requiring ground-truth simulation data. Six physics-based loss terms are employed:

- **Stretch loss** $\mathcal{L}_{\text{stretch}}$: measures stretch/compression energy based on the St. Venant-Kirchhoff model
- **Bending loss** $\mathcal{L}_{\text{bending}}$: penalizes curvature changes between adjacent faces
- **Collision loss** $\mathcal{L}_{\text{collision}}$: quantifies garment-body interpenetration depth
- **Gravity loss** $\mathcal{L}_{\text{gravity}}$: encourages natural draping
- **Friction loss** $\mathcal{L}_{\text{friction}}$: penalizes tangential sliding at contact points
- **Inertia loss** $\mathcal{L}_{\text{inertia}}$: maintains temporal coherence

Total loss: $\mathcal{L} = \mathcal{L}_{\text{stretch}} + \mathcal{L}_{\text{bending}} + \mathcal{L}_{\text{collision}} + \mathcal{L}_{\text{gravity}} + \mathcal{L}_{\text{friction}} + \mathcal{L}_{\text{inertia}}$

Training details: training is conducted exclusively at the lowest resolution (11K triangles), with a 128-dimensional latent space. Both message propagation and update functions are 2-layer MLPs with 128 units, followed by 15 MeshGraphNet layers. Training runs for 100K iterations, taking approximately 36 hours on an RTX 4070 Ti.

## Key Experimental Results

### Main Results

Evaluated on the VTO dataset across four garment types (T-shirt, tank top, long-sleeve shirt, long skirt). All methods are trained only at Level 1 (11K) and tested at higher resolutions.

| Resolution | Metric | Pb4U-GNet | CCRAFT | ESLR | HOOD | MGN |
|--------|------|-----------|--------|------|------|-----|
| Lv.1 (11K) | Total Loss | **-1.66E-02** | 4.24E-02 | -2.56E-02 | 9.45E-03 | 4.70E-03 |
| Lv.2 (18K) | Total Loss | **8.13E-03** | 1.10E-01 | 6.06E-02 | 2.49E-01 | 4.32E-01 |
| Lv.3 (25K) | Total Loss | **6.34E-02** | 1.72E-01 | 1.73E-01 | 2.78E-01 | 1.44E+03 |
| Lv.4 (38K) | Total Loss | **2.22E-01** | 2.82E-01 | 1.07E+05 | 2.57E+00 | 1.24E+06 |

At low resolution, all methods perform comparably; however, as resolution increases, competing methods degrade drastically (MGN reaches a total loss of $1.24\times10^6$ at 38K), while Pb4U-GNet remains stable.

### Ablation Study

| Configuration | Lv.1 (11K) | Lv.3 (25K) | Lv.4 (38K) | Notes |
|------|-----------|-----------|-----------|------|
| Pb4U-GNet (full) | -1.66E-02 | 6.34E-02 | 2.22E-01 | Best |
| w/o propagation control | -1.61E-03 | 1.08E+06 | 1.08E+09 | Collapse at high resolution |
| w/o update scaling | -5.78E-03 | 1.55E+13 | 7.34E+13 | Severe collapse at high resolution |
| w/o both | 4.70E-03 | 1.44E+03 | 1.24E+06 | Degrades to baseline |

Ablation results strongly demonstrate that both modules are indispensable for high-resolution generalization.

### Key Findings

1. **Propagation depth scales with resolution**: Figure 5 shows that, across different resolutions, physics loss converges only with more propagation steps for finer meshes.
2. **Adaptive computational efficiency**: At low resolution, fewer propagation steps improve efficiency (50ms vs. MGN's 46.4ms); at high resolution, more steps ensure accuracy (196.4ms, yet with far superior total loss compared to baselines).
3. **Generalization to unseen garment types**: Tested on a fitted dress and a cardigan, the total loss of 0.155 substantially outperforms CCRAFT's 0.264.

## Highlights & Insights

1. **Rigorous problem analysis**: Two independent factors underlying cross-resolution failure (receptive field and displacement scaling) are clearly identified, each addressed by a targeted solution.
2. **Elegant design**: The Propagation-before-Update decoupling is simple yet effective, achieving resolution adaptation naturally through architectural design.
3. **Physical consistency**: Both propagation distance control and update scaling carry clear physical interpretations (elastic wave propagation distance and the geometric similarity principle from continuum mechanics).
4. **Fully self-supervised**: No expensive physics simulation data is required as supervision.

## Limitations & Future Work

1. The current approach validates only the fixed-$\gamma$ decay accumulation scheme; more flexible multi-hop information fusion strategies warrant exploration.
2. The resolution scaling relies on a simple linear relationship based on mean edge length, which may be insufficiently precise for non-uniform meshes.
3. The method currently depends on predefined world-space distance thresholds for garment-body interaction handling, potentially limiting performance in extreme collision scenarios.
4. Temporal adaptation (e.g., fast motions may require different propagation strategies) has not been investigated.

## Related Work & Insights

- **Relationship to HOOD (Grigorev 2023)**: HOOD employs hierarchical graph structures for long-range interaction but does not address cross-resolution generalization. The decoupling approach proposed here is more direct.
- **Limitations of super-resolution methods**: Methods such as Zhang & Li 2024 rely on fixed-resolution coarse simulations; the proposed method performs simulation directly at arbitrary resolution, offering greater flexibility.
- The concept of decoupling propagation and update may offer broader inspiration for graph network tasks requiring cross-scale generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The decoupled propagation-update design is novel; both resolution-aware modules are grounded in solid physical motivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive multi-resolution quantitative comparisons, unseen garment generalization, ablation studies, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem analysis is clear, method motivation is well-articulated, and the structure is logical.
- **Value**: ⭐⭐⭐⭐ — Addresses a critical bottleneck in practical deployment (train at low resolution, deploy at high resolution).

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](../../CVPR2026/3d_vision/reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)
- [\[NeurIPS 2025\] MaNGO: Adaptable Graph Network Simulators via Meta-Learning](../../NeurIPS2025/3d_vision/mango_-_adaptable_graph_network_simulators_via_meta-learning.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)

<!-- RELATED:END -->
