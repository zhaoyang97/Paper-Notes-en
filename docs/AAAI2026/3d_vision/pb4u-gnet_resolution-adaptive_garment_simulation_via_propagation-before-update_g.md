---
title: >-
  [Paper Note] Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network
description: >-
  [AAAI 2026][3D Vision][Garment Simulation] Pb4U-GNet is proposed, which decouples message propagation from feature updating (Propagation-before-Update). By combining resolution-aware propagation depth control and an update scaling mechanism, it achieves garment simulation that generalizes to high-resolution meshes after training only on low-resolution meshes.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Garment Simulation"
  - "Graph Neural Networks"
  - "Cross-Resolution Generalization"
  - "Message Propagation"
  - "Resolution Adaptivity"
date: 2026-05-08
content_hash: fdc707ef045e79a2
---

# Pb4U-GNet: Resolution-Adaptive Garment Simulation via Propagation-before-Update Graph Network

**Conference**: AAAI 2026  
**arXiv**: [2601.15110](https://arxiv.org/abs/2601.15110)  
**Code**: [github.com/adam-lau709/PB4U-GNet](https://github.com/adam-lau709/PB4U-GNet)  
**Area**: 3D Vision  
**Keywords**: Garment Simulation, Graph Neural Networks, Cross-Resolution Generalization, Message Propagation, Resolution Adaptivity

## TL;DR

Pb4U-GNet is proposed, which decouples message propagation from feature updating (Propagation-before-Update). By combining resolution-aware propagation depth control and an update scaling mechanism, it achieves garment simulation that generalizes to high-resolution meshes after training only on low-resolution meshes.

## Background & Motivation

Garment simulation is a core technology for applications such as virtual try-on and digital human modeling. Traditional physics-based methods (such as mass-spring systems) are computationally expensive, particularly when repeatedly solving constraints on high-resolution meshes. As an accelerating alternative, Graph Neural Networks (GNNs) have achieved promising results. However, **existing GNN methods suffer from severe limitations in cross-resolution generalization**—their performance degrades drastically when applied to meshes outside the training resolution, especially higher resolutions.

The authors deeply analyze two fundamental causes of cross-resolution failure:

**Fixed Message Propagation Depth Issue**: Existing GNNs employ a fixed number of message propagation layers, allowing each vertex to only perceive neighbors within a preset number of hops. On fine meshes, a fixed depth leads to insufficient receptive field coverage; on coarse meshes, it causes over-smoothing.

**Resolution-Dependency of Displacement Magnitude**: On higher-resolution meshes, the same global movement is allocated across more vertices, reducing the displacement magnitude of individual vertices. This inborn resolution-dependency causes models trained at low resolutions to overestimate displacements at high resolutions.

These two issues reveal a key challenge: directly training on high-resolution meshes is computationally prohibitive, yet low-resolution trained models fail to generalize. This is precisely the critical problem this work addresses.

## Method

### Overall Architecture

The core innovation of Pb4U-GNet is **decoupling message propagation from feature updating**. While conventional GNNs perform message aggregation and feature updating simultaneously inside each layer, Pb4U-GNet first executes $K$ steps of pure message propagation to expand the receptive field, followed by a single consolidated feature update. This decoupled design allows the propagation depth $K$ to be adaptively adjusted according to the resolution without affecting the update frequency.

Overall pipeline: Input current mesh state $\mathbf{X}_t$ $\rightarrow$ node/edge encoder $\rightarrow$ $K$-step message propagation $\rightarrow$ feature update $\rightarrow$ refinement via 15-layer MeshGraphNet $\rightarrow$ node decoder predicting acceleration $\rightarrow$ resolution-aware scaling $\rightarrow$ forward Euler integration to yield the mesh state for the next time-step.

### Key Designs

#### 1. **Propagation-before-Update (PbU)**

Mechanism: Perform pure message aggregation to accumulate neighborhood information first, followed by a unified feature update.

In the propagation phase, each vertex maintains an aggregated feature vector $\mathbf{h}_{t,i}$, initialized with the vertex embedding $\mathbf{v}_{t,i}$. At each step $k$, neighborhood information is aggregated via a learnable message function $f_m(\cdot)$ (implemented as an MLP):

$$\tilde{\mathbf{h}}^k_{t,i} = \text{LayerNorm}\left(\sum_{j \in \mathcal{N}(i)} f_m(\mathbf{h}^{k-1}_{t,i}, \mathbf{h}^{k-1}_{t,j}, \mathbf{e}_{t,ij})\right)$$

Then, the old and new information are fused via decaying accumulation:

$$\mathbf{h}^k_{t,i} = \gamma \cdot \mathbf{h}^{k-1}_{t,i} + \tilde{\mathbf{h}}^k_{t,i}$$

where $\gamma$ is a decay factor controlling the influence of historical messages. After $K$ propagation steps, an update function $f_u$ (MLP) fuses the raw embedding and the accumulated features:

$$\mathbf{v}'_{t,i} = f_u(\mathbf{v}_{t,i}, \mathbf{h}^K_{t,i})$$

**Design Motivation**: Decoupling makes the receptive field size solely dependent on the propagation steps $K$ rather than being entangled with the update frequency, enabling flexible adaptation to different resolutions.

#### 2. **Resolution-Aware Propagation Control**

Mechanism: Dynamically adjust the propagation steps $K$ according to the mesh density to maintain a consistent physical propagation distance.

Define the effective physical propagation distance $D = K_{\text{base}} \times \bar{L}_{\text{base}}$, where $K_{\text{base}}$ is the number of propagation steps at the baseline resolution, and $\bar{L}_{\text{base}}$ is the average edge length of the baseline resolution. For any arbitrary resolution mesh:

$$K = \lfloor D \times \bar{L}^{-1} \rfloor$$

As the resolution increases ($\bar{L}$ decreases), $K$ increases proportionally, keeping the physical receptive field coverage consistent.

**Design Motivation**: Grounded in physical intuition—the propagation distance of elastic waves on cloth is independent of mesh discretization. Therefore, the same physical propagation distance should be maintained across different resolutions.

#### 3. **Resolution-Aware Update Scaling**

Mechanism: Scale the predicted acceleration based on the local geometric scale of each vertex.

Based on the geometric similarity principle in continuum mechanics (the displacement field scales linearly with element size), the scaling factor for each vertex is formulated as:

$$\mathbf{s}_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} l_{ij}$$

which represents the average length of all incident edges of vertex $i$ in the rest state. The final acceleration is obtained as $\mathbf{A}_{g,t} = \mathbf{S} \odot \tilde{\mathbf{A}}_{g,t}$.

**Design Motivation**: Vertices in higher-resolution meshes represent smaller areas and masses; thus, under the same global deformation, the acceleration of each vertex should be smaller. This scaling restores physically consistent displacement magnitudes.

### Loss & Training

The framework adopts **fully self-supervised training** without requiring ground-truth simulation data. It incorporates six physics-based loss terms:

- **Stretch loss** $\mathcal{L}_{\text{stretch}}$: Measures stretching/compression energy based on the St. Venant-Kirchhoff model
- **Bending loss** $\mathcal{L}_{\text{bending}}$: Penalizes curvature changes between adjacent faces
- **Collision loss** $\mathcal{L}_{\text{collision}}$: Quantifies the garment-body penetration depth
- **Gravity loss** $\mathcal{L}_{\text{gravity}}$: Encourages natural draping
- **Friction loss** $\mathcal{L}_{\text{friction}}$: Penalizes tangential sliding at contact points
- **Inertia loss** $\mathcal{L}_{\text{inertia}}$: Maintains temporal coherence

The total loss is $\mathcal{L} = \mathcal{L}_{\text{stretch}} + \mathcal{L}_{\text{bending}} + \mathcal{L}_{\text{collision}} + \mathcal{L}_{\text{gravity}} + \mathcal{L}_{\text{friction}} + \mathcal{L}_{\text{inertia}}$

Training details: Trained exclusively on the lowest resolution (11K triangular faces) with a 128-dimensional latent space. Both message propagation and update functions are 2-layer MLPs with 128 units, followed by a 15-layer MeshGraphNet. The overall training takes 100K iterations, taking approximately 36 hours (on an RTX 4070 Ti).

## Key Experimental Results

### Main Results

Evaluated on the VTO dataset across 4 garment categories (T-shirt, Vest, Long-sleeve, Long-dress). All methods are trained only on Level 1 (11K) and tested on higher resolutions.

| Resolution | Metric | Pb4U-GNet | CCRAFT | ESLR | HOOD | MGN |
|--------|------|-----------|--------|------|------|-----|
| Lv.1 (11K) | Total Loss | **-1.66E-02** | 4.24E-02 | -2.56E-02 | 9.45E-03 | 4.70E-03 |
| Lv.2 (18K) | Total Loss | **8.13E-03** | 1.10E-01 | 6.06E-02 | 2.49E-01 | 4.32E-01 |
| Lv.3 (25K) | Total Loss | **6.34E-02** | 1.72E-01 | 1.73E-01 | 2.78E-01 | 1.44E+03 |
| Lv.4 (38K) | Total Loss | **2.22E-01** | 2.82E-01 | 1.07E+05 | 2.57E+00 | 1.24E+06 |

As observed, while the performance of different methods is comparable at low resolutions, other methods degrade drastically as physical resolution increases (e.g., the total loss of MGN reaches $1.24\times10^6$ at 38K), whereas Pb4U-GNet remains robust and stable.

### Ablation Study

| Configuration | Lv.1 (11K) | Lv.3 (25K) | Lv.4 (38K) | Notes |
|------|-----------|-----------|-----------|------|
| Pb4U-GNet (Full) | -1.66E-02 | 6.34E-02 | 2.22E-01 | Best |
| w/o Propagation Control | -1.61E-03 | 1.08E+06 | 1.08E+09 | Crashes at high res |
| w/o Update Scaling | -5.78E-03 | 1.55E+13 | 7.34E+13 | Severely crashes at high res |
| w/o Both | 4.70E-03 | 1.44E+03 | 1.24E+06 | Degrades to baseline |

The ablation results strongly demonstrate that both modules are indispensable for generalizing to high resolutions.

### Key Findings

1. **Positive Correlation between Propagation Depth and Resolution**: Figure 5 illustrates the changes in physical loss across different propagation steps at various resolutions; higher-resolution meshes require more propagation steps to converge.
2. **Adaptive Computational Efficiency**: Under low resolution, reducing the propagation steps enhances efficiency (50ms vs 46.4ms for MGN); under high resolution, increasing the steps secures accuracy (196.4ms, while yielding a significantly superior total loss compared to other baselines).
3. **Generalization to Unseen Garment Categories**: Testing on tight dresses and cardigans yields a total loss of 0.155, significantly outperforming CCRAFT's 0.264.

## Highlights & Insights

1. **In-depth Problem Analysis**: The work clearly identifies two independent factors for cross-resolution failure (receptive field and displacement scaling) and formulates tailored solutions for each.
2. **Elegant Design**: The decoupling concept of Propagation-before-Update is simple yet effective, naturally achieving resolution adaptivity through structural design.
3. **Physical Consistency**: Both the propagation distance and update scaling possess clear physical interpretations (elastic wave propagation distance and the geometric similarity principle in continuum mechanics).
4. **Fully Self-Supervised**: Avoids the requirement of expensive physical simulation data as ground truth.

## Limitations & Future Work

1. Currently, only the decaying accumulation scheme (with fixed $\gamma$) is validated; more flexible multi-hop information fusion strategies could be explored.
2. The resolution scaling relies on a simple linear relationship based on average edge length, which may lack precision for non-uniform meshes.
3. The method currently relies on a predefined world-space distance threshold to handle garment-body interactions, which may limit execution in extreme collision scenarios.
4. Temporal adaptivity has not been explored (e.g., high-speed motion might warrant distinct propagation strategies).

## Related Work & Insights

- **Relation to HOOD (Grigorev et al. 2023)**: HOOD utilizes hierarchical graph structures to model long-range interactions but does not address cross-resolution generalization. The decoupling idea presented in this work is more direct.
- **Limitations of Super-Resolution Methods**: Methods such as Zhang & Li (2024) rely on coarse-grained simulations at a fixed resolution, whereas this work directly simulates on arbitrary resolutions, offering greater flexibility.
- The concept of decoupling propagation and updating could inspire other graph network tasks that require cross-scale generalization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The design of decoupling propagation and updating is novel, and the two resolution-aware modules have solid physical motivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluations including multi-resolution quantitative comparison, unseen garment generalization, ablation study, and efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem analysis, well-defined method motivations, and a well-structured layout.
- **Value**: ⭐⭐⭐⭐ — Resolves a crucial bottleneck in practical deployment (training at low resolution and deploying at high resolution).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ReWeaver: Towards Simulation-Ready and Topology-Accurate Garment Reconstruction](../../CVPR2026/3d_vision/reweaver_towards_simulation-ready_and_topology-accurate_garment_reconstruction.md)
- [\[ICML 2026\] Adaptive Volumetric Mechanical Property Fields Invariant to Resolution](../../ICML2026/3d_vision/adaptive_volumetric_mechanical_property_fields_invariant_to_resolution.md)
- [\[CVPR 2026\] SAQN: Semantic-based Adaptive Query Network for 3D Referring Expression Segmentation](../../CVPR2026/3d_vision/saqn_semantic-based_adaptive_query_network_for_3d_referring_expression_segmentat.md)
- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](../../NeurIPS2025/3d_vision/mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)
- [\[AAAI 2026\] Arbitrary-Scale 3D Gaussian Super-Resolution](arbitrary-scale_3d_gaussian_super-resolution.md)

</div>

<!-- RELATED:END -->
