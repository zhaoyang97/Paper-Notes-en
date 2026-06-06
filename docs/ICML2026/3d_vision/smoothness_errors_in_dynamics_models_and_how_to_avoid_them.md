---
title: >-
  [Paper Note] Smoothness Errors in Dynamics Models and How to Avoid Them
description: >-
  [ICML 2026][3D Vision][GNN] The authors theoretically demonstrate that "unitary GNNs" by Kiani et al. over-constrain physical systems that naturally increase in smoothness (such as heat diffusion) by forcibly preserving…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "GNN"
  - "Mesh Learning"
  - "Over-smoothing/Under-smoothing"
  - "Unitary Convolution"
  - "Rayleigh Quotient"
  - "Weather Forecasting"
date: 2026-05-08
content_hash: e4f28f11644399f6
---

# Smoothness Errors in Dynamics Models and How to Avoid Them

**Conference**: ICML 2026  
**arXiv**: [2602.05352](https://arxiv.org/abs/2602.05352)  
**Code**: Available (provided at the end of the paper)  
**Area**: 3D Vision / Geometric Deep Learning / Neural PDE Solvers  
**Keywords**: GNN, Mesh Learning, Over-smoothing/Under-smoothing, Unitary Convolution, Rayleigh Quotient, Weather Forecasting

## TL;DR
The authors theoretically demonstrate that "unitary GNNs" by Kiani et al. over-constrain physical systems that naturally increase in smoothness (such as heat diffusion) by forcibly preserving the Rayleigh quotient. Consequently, they propose "relaxed unitary convolutions" (R-UniGraph / R-UniMesh) and extend the Rayleigh quotient-unitary convolution framework from graphs to triangular meshes, outperforming several strong baselines on MeshPDE and WeatherBench22.

## Core Problem: Core Proposition
GNNs should be neither over-smooth nor non-smooth—the smoothing tendency of the architecture must precisely match the smoothing tendency of the ground-truth physical process.

## Background & Motivation

**Background**: Solving PDEs defined on meshes or manifolds (e.g., heat diffusion, wave equations, Cahn–Hilliard, global atmosphere) using neural networks is a highly active research area. The standard approach involves discretizing the manifold into a mesh and employing mesh-GNNs (GCN, MPNN, EGNN, Gauge-Equivariant CNN, Hermes, etc.) that support high-order connectivity for message passing. However, GNNs generally suffer from over-smoothing: as layers increase, features of adjacent nodes tend towards uniformity. Kiani et al. recently introduced "unitary graph convolution," which constrains the weight matrix to be a unitary matrix to preserve the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$, thereby strictly preventing over-smoothing.

**Limitations of Prior Work**: Unitary convolutions are to GCNs what "zero smoothing" is to "over-smoothing." Yet, most real-world physical systems possess an "intrinsic degree of smoothness"—heat diffusion makes features increasingly smooth, while the wave equation requires the preservation of high-frequency structures. Forcing unitary convolutions onto these systems leads to "under-smoothing," where the network cannot learn the intermediate smoothing processes required by the physics.

**Key Challenge**: GCNs and unitary convolutions represent two extremes regarding the Rayleigh quotient: GCNs strictly decrease the Rayleigh quotient (continuous smoothing), while unitary convolutions strictly preserve it (no smoothing). Real physical dynamics require a "tunable" smoothing rate rather than a choice between two extremes.

**Goal**: (i) Theoretically derive the lower bound of approximation errors for unitary functions to prove they are over-constrained on targets with strong angular dependence; (ii) design controllable "relaxed unitary convolutions" that allow the network to switch freely between the two extremes; (iii) extend the Rayleigh quotient and unitary convolutions from graphs to meshes for real physical tasks like PDE solving and weather forecasting.

**Key Insight**: The authors observe that the unitary property of Lie unitary convolutions $f(X) = \exp(AXW)$ (where $W = -W^\dagger$) stems from the Taylor expansion "extending to infinity." Truncating the expansion to the $T_\max$-th order results in layers that are no longer strictly unitary but approximate unitarity in a tunable manner—serving as a natural "continuous relaxation knob."

**Core Idea**: Relaxing the strict preservation of the Rayleigh quotient through either Taylor-truncated Lie convolutions or a "zero-pad + unitary encoder + arbitrary decoder" approach. This allows the network to adaptively match the true smoothness of the physical process during training. The framework is extended to meshes using a Robust Laplacian and tangent weights.

## Method

### Overall Architecture
Starting from the unified smoothness measure of the Rayleigh quotient $R_\mathcal{G}(X)$, the approach follows four steps: (1) theoretically prove the non-vanishing approximation error lower bound of unitary functions for targets where magnitude depends on angle; (2) propose two relaxation strategies: "Taylor truncation" and "Encoder-Decoder"; (3) generalize the Rayleigh quotient and unitary convolutions from graphs to triangular meshes (Mesh Rayleigh quotient + UniMesh convolution) via a Robust Laplacian; (4) stack these components into the final R-UniGraph (for graphs) and R-UniMesh (for meshes) models, using GroupSort for activation and MLP/GCN as decoders to break the unitary constraint.

### Key Designs

1.  **Taylor-Truncated Relaxed Lie Convolution (R-UniGraph)**:
    *   **Function**: Precision matching of the target system's true smoothing rate by smoothly interpolating between "GCN-style over-smoothing" and "unitary-style zero-smoothing" via the Taylor expansion order $T_\max$.
    *   **Mechanism**: The matrix exponential in the Lie unitary convolution $\exp(AXW)$ is truncated to the $T_\max$ order using a Taylor series: $f_{\text{Relaxed}}(X; A, T_\max) = \sum_{i=0}^{T_\max} \frac{1}{i!} L^i(X)$, where $L(X) = AXW$ and $W = -W^\dagger$. If $T_\max = 1$, it approximates GCN behavior; as $T_\max \to \infty$, it recovers the strict Lie unitary convolution. Intermediate values ($T_\max = 3$ for heat diffusion, $T_\max = 10$ for others) allow the network to perform small smoothing corrections while maintaining most of the Rayleigh quotient.
    *   **Design Motivation**: While Kiani et al. proposed a "relaxation" for separable unitary convolutions, their method mixed two sources—truncating Taylor series and allowing $U$ to be non-unitary—making it impossible to quantitatively analyze the origin of the relaxation. R-UniGraph retains the antisymmetric Lie form $W$, making $T_\max$ the sole knob for Rayleigh quotient preservation.

2.  **Encoder-Decoder Relaxation (R-UniMesh)**:
    *   **Function**: Addresses the limitation that Taylor truncation cannot change channel dimensions, providing a high-capacity, smoothness-aware model for meshes.
    *   **Mechanism**: First, a zero-padding operation $f_{\text{pad}}: \mathbb{R}^{n\times d_{in}} \to \mathbb{R}^{n\times d_{out}}$ pads node features to a hidden dimension (zero-padding naturally preserves the Rayleigh quotient by preserving the norm). Then, $k$ layers of Lie unitary mesh convolutions $f_{\text{UniMeshConv}}^{\text{Lie}}(X; A, \mathcal{W}) = \exp(\tilde A X W)$ are used as an encoder $E$, where $\tilde A = D^{-1/2}(\mathcal{W}\odot A)D^{-1/2}$ introduces cotangent weights $\mathcal{W}$. Finally, an MLP or GCN decoder $D$ maps to the target channel count and breaks the unitary constraint, providing the flexibility to express the target smoothing rate.
    *   **Design Motivation**: Lie unitary convolutions cannot change channels; increasing capacity requires deeper stacks, which can be unstable (e.g., "shattered gradients"). This "wide + shallow" approach concentrates parameter degrees of freedom in the decoder, letting the unitary encoder preserve geometric/smoothness structures while the decoder fits the target label smoothness.

3.  **Mesh Rayleigh Quotient and Unitary Convolutions on Robust Laplacian**:
    *   **Function**: Extends the smoothness analysis framework to real manifold PDE tasks by generalizing the Rayleigh quotient and unitary convolutions to triangular meshes.
    *   **Mechanism**: Traditional symmetric cotangent Laplacians $\tilde L$ can have negative weights on non-Delaunay triangulations, causing the Rayleigh quotient to lose its positive-definite meaning. The authors adopt the Robust Laplacian from Sharp & Crane, which ensures all cotangent weights $\mathcal{W}_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ satisfy the Delaunay criterion ($\alpha_{ij}+\beta_{ij}\le\pi$) through minimal edge intrinsic flips, ensuring non-negative off-diagonal elements. The mesh Rayleigh quotient is defined as $R_\mathcal{M}(X) = \mathrm{Tr}(X^\dagger \tilde L X)/\|X\|_F^2$, and the normalized adjacency matrix in unitary convolutions is updated with cotangent weights.
    *   **Design Motivation**: Previous mesh-GNNs used cotangent weights primarily for numerical precision. The authors link these weights to "strict smoothness preservation" via the Robust Laplacian, allowing the mathematical conclusions of the unitary framework to transfer automatically to meshes.

### Loss & Training
All tasks directly minimize regression losses like MSE/NRMSE. No additional Rayleigh loss terms are introduced; the core argument is that smoothness preservation should be determined by the architecture's inductive bias rather than soft constraints. R-UniMesh uses GroupSort (Anil et al. 2019) to ensure activations do not destroy the norm.

## Key Experimental Results

### Main Results
Evaluation was conducted on two task categories: (1) MeshPDE (autoregressive solving of heat, wave, and Cahn–Hilliard equations on complex PyVista meshes); (2) WeatherBench22 global weather forecasting (T850 temperature, Z500 geopotential).

| Dataset | Task | Metric | R-UniMesh | Prev. SOTA | Remarks |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MeshPDE / Heat | 196-step Autoregressive | NRMSE ↓ | **51.9 ± 3.6** | 73.0 ± 4.7 (Hermes) | Nearly halved |
| MeshPDE / Heat | As above | RE ↓ | **9.1 ± 7.4** | 14.2 ± 1.4 (EMAN) | Best smoothness match |
| MeshPDE / Wave | 196-step Autoregressive | NRMSE ↓ | **236.5 ± 6.4** | 281.3 ± 15.5 (EMAN) | Significant lead |
| MeshPDE / C-H | As above | NRMSE ↓ | 123.9 ± 2.6 | **121.2 ± 1.8** (GemCNN) | Close to SOTA |
| WB22 / T850 | RMSE @ 1-10 d | RMSE / ACC | Comparable | Pangu / GraphCast | Competitive under data limits |

### Ablation Study
A motivating experiment compared GCN, Lie unitary, and R-UniGraph on 2D mesh heat diffusion:

| Configuration | MSE ($\times 10^{-2}$) ↓ | MRE ($\times 10^{-2}$) ↓ | Interpretation |
| :--- | :--- | :--- | :--- |
| GCN | 1.08 | 5.99 | Over-smooth, high error |
| Lie Uni | 0.14 | 8.86 | Zero smoothing, under-smooth |
| **R-UniGraph (Ours, $T_\max=3$)** | **0.11** | **2.07** | Optimal MSE & Rayleigh error |

### Key Findings
- R-UniGraph outperforms both GCN and strict unitary models in both MSE and Rayleigh error, demonstrating that "just enough smoothing" is closer to the physical truth than "none" or "too much."
- In mesh heat diffusion, the Rayleigh Error (RE) of R-UniMesh aligns almost perfectly with the ground truth at every timestep; visualizations show rollouts that are neither over-smooth (like EMAN) nor under-smooth (like Hermes).
- On simple geometries (e.g., toroid mesh for Cahn-Hilliard), most equivariant/unitary/MPNN models perform similarly. Performance gaps emerge primarily in complex geometry generalization (different PyVista meshes), proving that geometric inductive bias is critical for cross-mesh generalization.
- GCNs and EGNNs performed worst across tasks, indicating that message passing or Euclidean equivariance alone is insufficient for manifold PDEs without explicit consideration of mesh smoothing structures.

## Highlights & Insights
- The design logic of "approximation lower bound + knob-based relaxation" is elegant: Theorem 1 proves the "cost" of strict unitarity, followed by Taylor truncation to turn "strict" into "tunable."
- The Rayleigh Error (RE) metric is a significant contribution, providing a physically meaningful smoothness alignment measure for PDE neural surrogates beyond standard RMSE.
- Integrating the Robust Laplacian, tangent weights, and unitary convolutions into a complete mesh framework provides a ready-to-use "scaffold" for researchers in manifold PDE solvers.

## Limitations & Future Work
- $T_\max$ and zero-pad dimensions still require tuning based on task priors. Future work could consider learnable $T_\max$ or adaptive attention to dynamically determine truncation orders.
- Performance gains are less pronounced on equations like Cahn–Hilliard, which are neither strictly smoothing nor strictly norm-preserving, suggesting the current binary view of relaxation might be too coarse for some systems.
- The WB22 experiments were limited by compute to $1.5°$ resolution and small-scale training; scalability to larger models compared to ECMWF SOTA remains for future verification.

## Related Work & Insights
- **vs Kiani et al. 2024 (Unitary GNN)**: Ours is a direct extension and "correction"—proving strict Rayleigh preservation is a flaw in dynamics tasks, providing controllable relaxation, and extending the theory to meshes.
- **vs Hermes / EMAN / GemCNN (Gauge equivariant mesh GNN)**: These methods use gauge equivariance for mesh orientation invariance; R-UniMesh provides an orthogonal inductive bias through Rayleigh quotient preservation, proving superior in strongly smoothing tasks like heat diffusion.
- **vs Subich 2025 / Bonev 2025 (Spectral domain training)**: These works improve effective resolution via soft spectral losses. R-UniMesh achieves similar goals through architectural constraints, avoiding loss-weight tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies "over- vs under-smoothing" via the Rayleigh quotient for the first time; provides controllable relaxation and mesh extensions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across motivating experiments, MeshPDE (multiple PDEs/meshes), and WB22 real-world data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Seamless transition between theory and method; clear citations for theorems and propositions.
- **Value**: ⭐⭐⭐⭐ Serves as both a SOTA PDE neural surrogate and a theoretically significant study of mesh-GNN inductive biases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[ICLR 2026\] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](../../ICLR2026/3d_vision/omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)
- [\[ICML 2026\] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)
- [\[CVPR 2026\] OpenVO: Open-World Visual Odometry with Temporal Dynamics Awareness](../../CVPR2026/3d_vision/openvo_open-world_visual_odometry_with_temporal_dynamics_awareness.md)

</div>

<!-- RELATED:END -->
