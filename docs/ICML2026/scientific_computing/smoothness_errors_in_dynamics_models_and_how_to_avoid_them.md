---
title: >-
  [Paper Note] Smoothness Errors in Dynamics Models and How to Avoid Them
description: >-
  [ICML 2026][Scientific Computing][GNN] The authors theoretically identify that Kiani et al.'s "unitary GNN" overly constrains physical systems like heat diffusion, which naturally smooth over time…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "GNN"
  - "Mesh Learning"
  - "Over-smoothing/Under-smoothing"
  - "Unitary Convolution"
  - "Rayleigh Quotient"
  - "Weather Forecasting"
date: 2026-05-08
content_hash: 67ca4a82542c2c20
---

# Smoothness Errors in Dynamics Models and How to Avoid Them

**Conference**: ICML 2026  
**arXiv**: [2602.05352](https://arxiv.org/abs/2602.05352)  
**Code**: Available (provided at the end of the paper)  
**Area**: 3D Vision / Geometric Deep Learning / PDE Neural Solvers  
**Keywords**: GNN, Mesh Learning, Over-smoothing/Under-smoothing, Unitary Convolution, Rayleigh Quotient, Weather Forecasting

## TL;DR
The authors theoretically identify that Kiani et al.'s "unitary GNN" overly constrains physical systems like heat diffusion, which naturally smooth over time, due to its strict preservation of the Rayleigh quotient. They propose "relaxed unitary convolution" (R-UniGraph / R-UniMesh), extending the Rayleigh quotient-unitary convolution framework from graphs to triangular meshes, achieving superior performance over strong baselines on MeshPDE and WeatherBench22.

## Core Proposition
GNNs should neither over-smooth nor under-smooth—architectures must align their smoothing tendencies with the true physical process.

## Background & Motivation

**Background**: Using neural networks to solve PDEs defined on meshes/manifolds (e.g., heat diffusion, wave equations, Cahn–Hilliard, Earth's atmosphere) has been a highly active area in scientific computing over the past two years. The mainstream approach discretizes manifolds into meshes and applies mesh-GNNs (e.g., GCN, MPNN, EGNN, Gauge-Equivariant CNN, Hermes) for message passing. However, GNNs often suffer from over-smoothing: as layers deepen, neighboring node features converge. Recently, Kiani et al. proposed "unitary graph convolution," constraining weight matrices to unitary matrices to preserve the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$, thereby strictly avoiding over-smoothing.

**Limitations of Prior Work**: Unitary convolution represents "no smoothing" as opposed to GCN's "over-smoothing." However, most physical systems exhibit "just the right amount of smoothness"—e.g., heat diffusion smooths features over time, while wave equations preserve high-frequency structures. Applying unitary convolution to such systems results in "under-smoothing," preventing the network from learning the moderate smoothing processes required by physics.

**Key Challenge**: GCN and unitary convolution represent two extremes in Rayleigh quotient behavior: GCN strictly decreases the Rayleigh quotient (continuous smoothing), while unitary convolution strictly preserves it (no smoothing). Real physical dynamics require "adjustable" smoothing rates, not a binary choice between extremes.

**Goal**: (i) Theoretically derive a lower bound on the approximation error of unitary functions, proving their over-constraint on angle-dependent objectives; (ii) Design controllable "relaxed unitary convolution" to allow networks to interpolate between extremes; (iii) Extend the Rayleigh quotient and unitary convolution framework from graphs to meshes, making it applicable to real physical tasks like PDE/weather forecasting.

**Key Insight**: The authors observe that the unitary property of Lie unitary convolution $f(X) = \exp(AXW)$, $W = -W^\dagger$, arises from a Taylor expansion "to infinite order." Truncating the expansion at order $T_\max$ yields layers that are no longer strictly unitary but approximate unitary in a tunable manner—providing a natural "continuous relaxation knob."

**Core Idea**: Relax strict Rayleigh quotient preservation using either Taylor-truncated Lie convolution or "zero-pad + unitary encoder + arbitrary decoder," enabling the network to adaptively match the true smoothness of physical processes. The framework is extended to meshes using Robust Laplacian and tangent weights.

## Method

### Overall Architecture
The authors start from the unified smoothness metric of the Rayleigh quotient $R_\mathcal{G}(X)$ and proceed in four steps: (1) Theoretically prove a lower bound on the approximation error of unitary functions for angle-dependent objectives; (2) Propose two relaxation strategies: "Taylor truncation" and "encoder-decoder"; (3) Generalize the graph-based Rayleigh quotient and unitary convolution to triangular meshes using Robust Laplacian (i.e., mesh Rayleigh quotient + UniMesh convolution); (4) Stack these components into the final R-UniGraph (graph-based) and R-UniMesh (mesh-based) models, using GroupSort for activation and MLP/GCN for decoding to break unitary constraints.

### Key Designs

1. **Taylor-Truncated Relaxed Lie Convolution (R-UniGraph)**:

    - **Function**: Smoothly interpolates between "GCN-like over-smoothing" and "unitary-like no smoothing" by controlling the Taylor expansion order $T_\max$, precisely matching the target system's true smoothing rate.
    - **Mechanism**: Truncate the matrix exponential in Lie unitary convolution $\exp(AXW)$ to $T_\max$ terms, yielding $f_{\text{Relaxed}}(X; A, T_\max) = \sum_{i=0}^{T_\max} \frac{1}{i!} L^i(X)$, where $L(X) = AXW$, $W = -W^\dagger$. $T_\max = 1$ approximates GCN behavior; $T_\max \to \infty$ recovers strict Lie unitary convolution; intermediate values (e.g., $T_\max = 3$ for heat diffusion, $T_\max = 10$ for other tasks) allow small smoothing adjustments while preserving most of the Rayleigh quotient.
    - **Design Motivation**: While Kiani et al. proposed "relaxed" separable unitary convolution, their relaxation mixed two sources—both Taylor truncation and non-unitary $U$—making it impossible to quantify the relaxation's origin. R-UniGraph isolates the Lie form's antisymmetric $W$, making $T_\max$ the sole knob for Rayleigh quotient preservation. If the target process's smoothness is known, $T_\max$ can even be selected from a lookup table.

2. **Encoder-Decoder Relaxed R-UniMesh (Scalable Mesh Version)**:

    - **Function**: Addresses the Taylor truncation method's inability to change channel dimensions, providing a high-capacity smoothing-aware model for meshes.
    - **Mechanism**: Use zero-padding $f_{\text{pad}}: \mathbb{R}^{n\times d_{in}} \to \mathbb{R}^{n\times d_{out}}$ to pad node features to hidden dimensions (zero-padding naturally preserves the Rayleigh quotient as it maintains norm length). Then stack $k$ layers of Lie unitary mesh convolution $f_{\text{UniMeshConv}}^{\text{Lie}}(X; A, \mathcal{W}) = \exp(\tilde A X W)$ as the encoder $E$, where $\tilde A = D^{-1/2}(\mathcal{W}\odot A)D^{-1/2}$ introduces cotangent weights $\mathcal{W}$. Finally, apply an MLP or GCN decoder $D$ to map to the target channel dimensions and break unitary constraints, balancing strong unitary backbone constraints with the flexibility to express target smoothness rates.
    - **Design Motivation**: Lie unitary convolution cannot change channel dimensions, so increasing capacity requires deeper networks. However, deep unitary stacks are unstable to train (Balduzzi et al.'s "shattered gradients"). The authors achieve "wide + shallow" via zero-padding and concentrate "parameter freedom" in the decoder, effectively letting the unitary encoder preserve geometric/smoothness structure while the decoder fits arbitrary label smoothness, ensuring clear division of labor and stable training.

3. **Mesh Rayleigh Quotient and Unitary Convolution on Robust Laplacian**:

    - **Function**: Extends the Rayleigh quotient definition and unitary convolution from graphs to triangular meshes, enabling the smoothness analysis framework to be applied to real manifold PDE tasks for the first time.
    - **Mechanism**: Traditional symmetric cotangent Laplacian $\tilde L$ can produce negative weights under non-Delaunay triangulation, invalidating the Rayleigh quotient's positive-definiteness. The authors adopt Sharp & Crane's Robust Laplacian, which ensures all cotangent weights $\mathcal{W}_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ satisfy the Delaunay criterion ($\alpha_{ij}+\beta_{ij}\le\pi$) by minimally reconnecting edges, ensuring all off-diagonal entries are non-negative. They then define the mesh Rayleigh quotient $R_\mathcal{M}(X) = \mathrm{Tr}(X^\dagger \tilde L X)/\|X\|_F^2$ and replace $\tilde A$ in separable/Lie unitary convolution with the normalized adjacency matrix weighted by cotangents. Corollary 1 proves that these mesh versions of unitary convolution also preserve the mesh Rayleigh quotient.
    - **Design Motivation**: Previous mesh-GNNs used cotangent weights to improve numerical precision but did not connect them to "strict smoothness preservation." By enforcing the Delaunay assumption and Robust Laplacian, the authors ensure the key condition of "positive weights," allowing all mathematical conclusions of the unitary framework to transfer to meshes without redoing algebraic proofs.

### Loss & Training
All tasks directly minimize regression losses like MSE/NRMSE without introducing additional Rayleigh loss terms—the authors argue that "whether to preserve smoothness" should be determined by the architecture's inductive bias, not soft constraints. R-UniMesh uses GroupSort (Anil et al., 2019) for activation to preserve norm length and orthogonal weights (sufficient for real-valued tasks), with GCN decoders for end-to-end backpropagation.

## Key Experimental Results

### Main Results
The authors evaluate on two tasks: (1) MeshPDE (autoregressive solving of heat, wave, and Cahn–Hilliard equations on PyVista complex meshes); (2) WeatherBench22 global weather forecasting (T850 temperature, Z500 geopotential).

| Dataset | Task | Metric | R-UniMesh | Strongest Baseline | Notes |
|---------|------|--------|-----------|---------------------|-------|
| MeshPDE / Heat | Autoregressive 196 steps | NRMSE ↓ | **51.9 ± 3.6** | 73.0 ± 4.7 (Hermes) | Nearly halved |
| MeshPDE / Heat | Same | RE ↓ | **9.1 ± 7.4** | 14.2 ± 1.4 (EMAN) | Best smoothness match |
| MeshPDE / Wave | Autoregressive 196 steps | NRMSE ↓ | **236.5 ± 6.4** | 281.3 ± 15.5 (EMAN) | Still leading |
| MeshPDE / Cahn–Hilliard | Same | NRMSE ↓ | 123.9 ± 2.6 | **121.2 ± 1.8** (GemCNN) | Close to SOTA |
| WB22 / T850 | RMSE @ 1-10 d | RMSE / ACC | Early comparable to SOTA | Pangu/GraphCast | Limited training data |

### Ablation Study
The authors compare GCN, Lie unitary, and R-UniGraph on heat diffusion over 2D meshes:

| Configuration | MSE ($\times 10^{-2}$) ↓ | MRE ($\times 10^{-2}$) ↓ | Interpretation |
|---------------|--------------------------|--------------------------|----------------|
| GCN | 1.08 | 5.99 | Over-smoothing, large error |
| Lie Uni | 0.14 | 8.86 | No smoothing, under-smoothing |
| **R-UniGraph (Ours, $T_\max=3$)** | **0.11** | **2.07** | Optimal MSE and Rayleigh error |

### Key Findings
- R-UniGraph outperforms GCN and strict unitary in both MSE and Rayleigh error, demonstrating that "just the right smoothing rate" is closer to physical truth than "always smooth" or "never smooth."
- In mesh-based heat diffusion prediction, R-UniMesh's Rayleigh error aligns with ground truth at nearly every timestep. Visualizations show its rollout avoids over-smoothing (like EMAN) or under-smoothing (like Hermes).
- On simple geometries (e.g., toroidal mesh for Cahn–Hilliard), most equivalent/unitary/MPNN methods perform similarly. Differences emerge in generalization to complex geometries (e.g., different PyVista meshes), proving that "geometric inductive bias" is critical for cross-mesh generalization.
- GCN and EGNN consistently underperform across all tasks, indicating that message passing or Euclidean equivariance alone is insufficient for simulating PDEs on manifolds—explicit consideration of mesh smoothness structure is necessary.

## Highlights & Insights
- The "lower bound on approximation error + knob-based relaxation" design logic is elegant: Theorem 1 diagnoses the "cost" of strict unitary (integrating norm variance over the fundamental domain), while Taylor truncation provides a tunable solution. This "diagnose-and-address" writing template is valuable for other inductive bias studies.
- The Rayleigh error (RE) metric itself is a paper-level contribution: It offers a physically meaningful smoothness alignment measure for PDE neural surrogates. Future mesh-GNN papers should adopt it as a standard metric.
- Packaging Robust Laplacian, tangent weights, and unitary convolution into a complete mesh framework provides a "ready-made scaffold" for future manifold PDE solver research.

## Limitations & Future Work
- $T_\max$ and zero-pad dimensions still require task-specific tuning. The authors suggest lookup tables for known target smoothness but lack automatic scheduling strategies for unknown PDEs. Future work could explore learnable $T_\max$ or adaptive attention to dynamically determine truncation order.
- The method shows limited advantage on equations like Cahn–Hilliard, which are neither strictly smoothing nor strictly preserving. This suggests the current binary "relax both ends" perspective may be too coarse for truly intermediate systems, requiring finer Rayleigh spectrum analysis.
- WB22 experiments were limited to $1.5°$ resolution and small-scale training due to computational constraints. Future work should validate scalability on larger datasets.

## Related Work & Insights
- **vs Kiani et al. 2024 (Unitary GNN)**: This work directly extends and "corrects" it—proving strict Rayleigh preservation is a flaw for dynamics tasks, introducing Taylor truncation for controllable relaxation, and generalizing the framework to meshes.
- **vs Hermes / EMAN / GemCNN (Gauge-equivariant mesh GNN)**: These methods handle mesh direction invariance via gauge equivariance. R-UniMesh introduces Rayleigh preservation as an orthogonal and complementary inductive bias, excelling in strongly smoothing tasks like heat diffusion.
- **vs Subich 2025 / Bonev 2025 (Frequency-domain training objectives)**: These works improve weather model effective resolution via soft constraints (spectral loss). R-UniMesh achieves similar goals via architectural constraints, avoiding loss weighting hyperparameters and providing clearer physical meaning for PDE tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify "over-smoothing vs under-smoothing" via Rayleigh quotient and propose controllable relaxation with mesh extension—innovative in both theory and method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on motivating experiments, multiple PDEs, multiple meshes in MeshPDE, and real-world WB22 data; WB22 limited by computational resources.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth transitions between theory, method, motivation, and experiments; clear citation of theorems and propositions.
- Value: ⭐⭐⭐⭐ Both a usable PDE neural surrogate (SOTA on heat diffusion) and a theoretically insightful study of mesh-GNN inductive biases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [\[ICML 2026\] (Sparse) Attention to the Details: Preserving Spectral Fidelity in ML-based Weather Forecasting Models](sparse_attention_to_the_details_preserving_spectral_fidelity_in_ml-based_weather.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[ICLR 2026\] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](../../ICLR2026/3d_vision/omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)

</div>

<!-- RELATED:END -->
