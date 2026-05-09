---
title: >-
  [Paper Note] Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras
description: >-
  [AAAI 2026][Particle Image Velocimetry] This paper proposes Spike Imaging Velocimetry (SIV), the first systematic application of **spike cameras** (20,000 Hz ultra-high temporal resolution) to fluid velocimetry. Three fluid-aware modules are designed: Detail-Preserving Hierarchical Transform (DPHT), Graph Encoder (GE), and Multi-Scale Velocity Refinement (MSVR). A new PSSD dataset is constructed, and SIV comprehensively outperforms existing baselines on steady-state turbulence, high-speed flow, and HDR scenarios.
tags:
  - AAAI 2026
  - Particle Image Velocimetry
  - Spike Camera
  - Fluid Motion Estimation
  - Graph Neural Network
  - Multi-Scale Optimization
date: 2026-05-08
content_hash: 41a2dbe1334a0265
---

# Spike Imaging Velocimetry: Dense Motion Estimation of Fluids Using Spike Cameras

**Conference**: AAAI 2026
**arXiv**: [2504.18864](https://arxiv.org/abs/2504.18864)
**Code**: N/A
**Area**: Computer Vision / Fluid Mechanics
**Keywords**: Particle Image Velocimetry, Spike Camera, Fluid Motion Estimation, Graph Neural Network, Multi-Scale Optimization

## TL;DR

This paper proposes Spike Imaging Velocimetry (SIV), the first systematic application of **spike cameras** (20,000 Hz ultra-high temporal resolution) to fluid velocimetry. Three fluid-aware modules are designed: Detail-Preserving Hierarchical Transform (DPHT), Graph Encoder (GE), and Multi-Scale Velocity Refinement (MSVR). A new PSSD dataset is constructed, and SIV comprehensively outperforms existing baselines on steady-state turbulence, high-speed flow, and HDR scenarios.

## Background & Motivation

1. **Background**: Particle Image Velocimetry (PIV) is a widely adopted non-intrusive imaging technique in fluid mechanics that captures velocity distributions by tracking the displacement of tracer particles. Learning-based PIV methods built on optical flow networks (e.g., RAFT-PIV) have achieved significant progress.
2. **Limitations of Prior Work**: (a) Conventional cameras have limited temporal resolution, causing accuracy degradation due to large inter-frame displacements in high-speed fluid scenarios; (b) sparse regions without tracer particles yield discrete signals, yet a continuous flow field must still be estimated; (c) small-scale vortices and unstructured patterns in turbulence further complicate estimation; (d) performance gains in existing PIV networks primarily stem from advances in optical flow architectures, lacking designs tailored to fluid characteristics.
3. **Key Challenge**: High-speed turbulence simultaneously demands high temporal resolution (to reduce inter-frame displacement) and high dynamic range (to handle uneven illumination), requirements that conventional cameras cannot satisfy concurrently.
4. **Goal**: Exploit the ultra-high temporal resolution (20,000 Hz) and high dynamic range of spike cameras to resolve the hardware bottleneck, while designing a dedicated network tailored to fluid properties.
5. **Key Insight**: Spike cameras asynchronously accumulate photons and output binary spike streams, making them naturally suited for high-speed motion scenarios. Graph neural networks are employed to aggregate context over fluid topology, and multi-scale velocity refinement addresses small-scale vortices.
6. **Core Idea**: Spike cameras resolve the hardware bottleneck; fluid-aware graph encoders and multi-scale refinement resolve the algorithmic bottleneck.

## Method

### Overall Architecture

The input is a spike stream $\mathbf{S} \in \mathbb{B}^{H \times W \times T}$. DPHT extracts a multi-scale feature pyramid; GE maps features to a graph structure for adaptive context aggregation; a RAFT-based Multi-Scale Iterative Optimizer (MSIO) estimates residual flow fields; and MSVR refines the multi-scale velocity field to recover small-scale vortices.

### Key Designs

1. **Detail-Preserving Hierarchical Transform (DPHT)**

    - **Function**: Preserves fine-grained particle signals during multi-scale downsampling.
    - **Mechanism**: A multi-level pyramid in which each level applies 3D detail-preserving downsampling: $\mathbf{x}_{out}^{(l)} = \mathscr{F}_{2C}(\mathscr{F}_{3C}(\mathbf{x}^{(l)}) + \mathscr{F}_{3MP}(\mathbf{x}^{(l)}))$, where $\mathscr{F}_{3C}$ is a 5×3×3 3D convolution, $\mathscr{F}_{3MP}$ combines 3D max pooling with a 5×1×1 3D convolution, and $\mathscr{F}_{2C}$ is a stack of 2D convolutions. A global temporal aggregation module concatenates and fuses the outputs of three levels.
    - **Design Motivation**: Standard downsampling discards bright-spot signals from particles. 3D max pooling preserves peak responses while 3D convolution retains spatiotemporal continuity; their summation captures both.

2. **Graph Encoder (GE)**

    - **Function**: Maps features to a graph structure and employs GAT for adaptive context aggregation, capturing flow field information in particle-sparse regions.
    - **Mechanism**: (a) *Graph projection*: DPHT features are projected onto $K=128$ node spaces $\mathbf{V}_c$, where each node is a weighted aggregation of regional features; (b) *Graph convolution*: a dynamic adjacency matrix $\widetilde{\mathbf{A}} = \mathbf{V}_c^T \mathbf{V}_c$ is constructed, and two-layer GAT updates node features; (c) *Graph reprojection*: node features are mapped back to the spatial domain to produce $\widetilde{\mathbf{R}}_c$, which is residually connected with the original features and passed through a local refinement network. A learnable parameter $\alpha$ (initialized to 0) controls the injection strength of graph features.
    - **Design Motivation**: In high-Reynolds-number turbulence, large eddies decompose into small vortices exhibiting topological structure. GNNs can model such non-Euclidean relational structure, and the attention mechanism of GAT reduces interference from particle-sparse regions.

3. **Multi-Scale Velocity Refinement (MSVR)**

    - **Function**: Recovers a complete high-resolution velocity field containing small-scale vortices.
    - **Mechanism**: The multi-scale velocity fields output by MSIO are individually processed by 2D convolutions, and cross-scale information exchange is achieved through coarse-to-fine cross convolutions. The fused features are passed through two independent networks that respectively generate a residual velocity field $\mathbf{u}_{res}$ and a quality map $\mathbf{Q}$. The final refined output is $\mathbf{u}_{ref} = \mathbf{u}_N + \mathbf{u}_{res} \odot \mathbf{Q}$ (element-wise product weighted residual).
    - **Design Motivation**: The convex upsampling in RAFT is insufficient for accurately reconstructing small vortex structures in fluids. The quality map $\mathbf{Q}$ allows the network to adaptively determine which regions require refinement.

### Loss & Training

$L = L_{flow} + 0.3 \cdot L_{grad}$, where $L_{flow}$ is a multi-iteration exponentially weighted L1 flow loss and $L_{grad}$ is an L1 loss on the velocity field gradient (to preserve small vortex structures). An exponential decay weight of $\gamma=0.8$ is used. Adam optimizer, initial learning rate $10^{-4}$, 100 epochs, single RTX 2080Ti.

## Key Experimental Results

### Main Results

Average EPE↓ on PSSD Dataset Problem 1 (Steady-State Turbulence):

| Method | Δt=21 | Δt=11 |
|--------|-------|-------|
| RAFT-PIV-Image | 1.442 | 0.843 |
| RAFT-PIV-Spike | 0.846 | 0.525 |
| HiST-SFlow | 0.908 | 0.567 |
| Flowformer-Spike | 0.722 | 0.425 |
| **SIV (Ours)** | **0.607** | **0.402** |

### Ablation Study

| Configuration | Avg. EPE | Note |
|---------------|----------|------|
| Baseline (DIP) | Higher | No fluid-specific modules |
| + DPHT | Reduced | Detail preservation effective |
| + GE | Further reduced | Graph aggregation improves context |
| + MSVR | Best | Small-vortex refinement critical |
| + Gradient loss | Further reduced | Preserves velocity field edges/vortices |

### Key Findings

- Spike camera vs. conventional camera: using spike inputs reduces EPE by ~40% compared to image inputs under the same method (RAFT-PIV: 1.442→0.846), validating the hardware advantage of spike cameras for PIV.
- SIV achieves the best performance across all three scenarios (steady-state turbulence, high-speed flow, HDR).
- The graph encoder provides the most notable improvement in sparse-particle regions, where traditional methods perform worst.
- The gradient loss is critical for preserving small-vortex structures.

## Highlights & Insights

- **First systematic exploration of spike cameras in PIV**: The 20,000 Hz temporal resolution dramatically reduces inter-frame displacement, fundamentally lowering the difficulty of high-speed flow velocimetry. This opens a new application direction for neuromorphic cameras in fluid mechanics.
- **Graph encoder for sparse signal handling**: Projecting features onto a graph structure for aggregation is an elegant solution to the problem of particle-free regions in PIV. The attention mechanism of GAT naturally adapts to non-uniform particle distributions.
- **PSSD Dataset**: A synthetic dataset based on high-fidelity DNS simulations from JHTDB, covering three challenging scenarios and providing a foundation for spike-PIV research.

## Limitations & Future Work

- Validation is performed on synthetic data only; the method has not been tested in real spike camera PIV experiments.
- Particle concentration and noise models in the PSSD dataset may differ from those in actual experiments.
- The number of nodes $K=128$ in the graph encoder is fixed; an adaptive graph structure may yield better results.
- The application of spike cameras to 3D PIV (e.g., stereo PIV, tomographic PIV) remains unexplored.

## Related Work & Insights

- **vs. RAFT-PIV**: General-purpose optical flow architectures applied directly to PIV, lacking designs tailored to fluid characteristics. All three modules in SIV are fluid-specific.
- **vs. Event Camera PIV**: Event cameras detect brightness changes but lack absolute intensity; spike cameras retain absolute intensity, making them better suited for PIV.
- **vs. HiST-SFlow**: Multi-level spike stream representations without graph structure or multi-scale refinement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First exploration of spike cameras for PIV + three fluid-specific modules
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of three scenarios but lacking real-world experiments
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and detailed module descriptions
- Value: ⭐⭐⭐⭐ Opens a new direction for neuromorphic cameras in fluid measurement

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [\[CVPR 2026\] Shoe Style-Invariant and Ground-Aware Learning for Dense Foot Contact Estimation](../../CVPR2026/others/shoe_style-invariant_and_ground-aware_learning_for_dense_foot_contact_estimation.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/others/learning_dense_hand_contact_estimation_from_imbalanced_data.md)
- [\[AAAI 2026\] Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)

</div>

<!-- RELATED:END -->
