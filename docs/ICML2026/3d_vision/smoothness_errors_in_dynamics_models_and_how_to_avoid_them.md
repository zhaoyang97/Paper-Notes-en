---
title: >-
  [Paper Note] Smoothness Errors in Dynamics Models and How to Avoid Them
description: >-
  [ICML 2026][3D Vision][GNN] The authors theoretically demonstrate that the "unitary GNN" by Kiani et al. over-constrains physical systems that are "naturally smoothing" (such as heat diffusion) by strictly maintaining the Rayleigh quotient. They propose "relaxed unitary convolutions" (R-UniGraph / R-UniMesh) and extend the entire Rayleigh quotien
tags:
  - ICML 2026
  - 3D Vision
  - GNN
date: 2026-05-08
content_hash: a1467ac5896171bf
---
# Smoothness Errors in Dynamics Models and How to Avoid Them

**Conference**: ICML 2026  
**arXiv**: [2602.05352](https://arxiv.org/abs/2602.05352)  
**Code**: Yes (provided at the end of the paper)  
**Area**: 3D Vision / Geometric Deep Learning / PDE Neural Solvers  
**Keywords**: GNN, Mesh Learning, Over-smoothing/Under-smoothing, Unitary Convolution, Rayleigh Quotient, Weather Forecasting

## TL;DR
The authors theoretically demonstrate that the "unitary GNN" by Kiani et al. over-constrains physical systems that are "naturally smoothing" (such as heat diffusion) by strictly maintaining the Rayleigh quotient. They propose "relaxed unitary convolutions" (R-UniGraph / R-UniMesh) and extend the entire Rayleigh quotient-unitary convolution framework from graphs to triangular meshes, outperforming several strong baselines on MeshPDE and WeatherBench22 simultaneously.

## Core Problem
GNNs should neither be over-smoothed nor non-smoothed—the architectural tendency for smoothness must precisely match the smoothing tendency of the ground truth physical process.

## Background & Motivation

**Background**: Solving PDEs defined on meshes/manifolds (heat diffusion, wave equations, Cahn–Hilliard, Earth's atmosphere) using neural networks is currently one of the most active directions in scientific computing. The dominant approach involves discretizing manifolds into meshes and utilizing mesh-GNNs that support high-order connectivity (GCN, MPNN, EGNN, Gauge-Equivariant CNN, Hermes, etc.) for message passing. However, GNNs generally suffer from over-smoothing: as the number of layers increases, the features of adjacent nodes tend to converge. Kiani et al. recently proposed "unitary graph convolution," which constrains weight matrices to be unitary, thereby strictly preserving the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$ and preventing over-smoothing.

**Limitations of Prior Work**: Compared to GCNs, unitary convolutions represent the extreme of "zero smoothness" versus "excessive smoothness." However, most real-world physical systems possess an "inherent smoothness"—heat diffusion causes features to become increasingly smooth, while the wave equation requires the preservation of high-frequency structures. Forcing unitary convolutions onto these systems results in "under-smoothing," where the network cannot learn the intermediate smoothing processes dictated by physics.

**Key Challenge**: GCNs and unitary convolutions represent two extremes in terms of the Rayleigh quotient: GCNs strictly decrease it (continuous smoothing), while unitary convolutions strictly preserve it (no smoothing). Any realistic physical dynamics requires a "tunable" smoothing rate rather than a binary choice between two extremes.

**Goal**: (i) Theoretically establish a lower bound for the approximation error of unitary functions, proving they are over-constrained for targets with strong angular dependence; (ii) design controllable "relaxed unitary convolutions" that allow the network to switch freely between the two extremes; (iii) extend the Rayleigh quotient and unitary convolutions from graphs to meshes to make them applicable to real physical tasks like PDE solving and weather forecasting.

**Key Insight**: The authors observe that the unitary property of the Lie unitary convolution $f(X) = \exp(AXW)$ with $W = -W^\dagger$ arises from the Taylor expansion "extending to infinity." By truncating the expansion at a designated $T_\max$ degree, the resulting layer is no longer strictly unitary but remains close to it in a tunable manner—providing a natural "continuous relaxation knob."

**Core Idea**: Relax strict Rayleigh quotient preservation using either Taylor-truncated Lie convolutions or a "zero-pad + unitary encoder + arbitrary decoder" approach, allowing the network to adaptively match the true smoothness of physical processes during training. The theory is transitioned to meshes using a Robust Laplacian and tangent weights.

## Method

### Overall Architecture
This method addresses the misalignment between the smoothing tendency of GNNs and actual physical processes—it aims for a tunable smoothing rate rather than being fixed at either the GCN extreme or the strictly unitary extreme. The framework centers on the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$ as a unified metric. First, it theoretically proves that strictly unitary functions have an irreducible approximation error lower bound for targets where "magnitude varies with angle," explaining "why relaxation is necessary." Second, it provides two relaxation paths to transform "strict Rayleigh preservation" into a "tunable smoothing rate": Taylor truncation (resulting in R-UniGraph) and zero-padded encoding-decoding (resulting in R-UniMesh). Finally, it adapts the Rayleigh quotient and unitary convolutions to triangular meshes using the Robust Laplacian, enabling the framework to be applied to real manifold PDE tasks. The final model utilizes GroupSort for activation (to preserve magnitude) and MLP/GCN decoders to actively break unitary constraints for flexibility.

### Key Designs

**1. Taylor-Truncated Relaxed Lie Convolution (R-UniGraph): A continuous knob between over-smoothing and under-smoothing**

The pain point is that GCNs strictly reduce the Rayleigh quotient (constant smoothing) while unitary convolutions strictly preserve it (zero smoothing). Physical dynamics need a tunable intermediate rate. The authors note that the unitary nature of the Lie unitary convolution $\exp(AXW)$ ($W=-W^\dagger$) comes from the infinite Taylor expansion of the matrix exponential. By truncating at the $T_\max$ degree, they obtain $f_{\text{Relaxed}}(X; A, T_\max) = \sum_{i=0}^{T_\max} \frac{1}{i!} L^i(X)$ where $L(X)=AXW$. Thus, $T_\max$ becomes a continuous knob: $T_\max=1$ approximates GCN behavior, while $T_\max\to\infty$ recovers strict Lie unitary convolution. Intermediate values (e.g., $T_\max=3$ for heat diffusion, $T_\max=10$ elsewhere) allow small-scale smoothing corrections. Unlike previous "separable" unitary relaxations that relax both the Taylor expansion and the matrix property, R-UniGraph keeps the anti-symmetric $W$ to make $T_\max$ the sole knob for Rayleigh preservation.

**2. Zero-Padded Encoding-Decoding Relaxation (R-UniMesh): Concentrating capacity in the decoder to bypass instability in deep unitary stacks**

Taylor truncation cannot change channel dimensions; increasing parameters requires deeper layers, which leads to "shattered gradients" and training instability in deep unitary stacks. R-UniMesh opts for a "wide and shallow" approach: it first uses zero-padding $f_{\text{pad}}: \mathbb{R}^{n\times d_{in}}\to\mathbb{R}^{n\times d_{out}}$ to reach hidden dimensions (which preserves magnitude and thus the Rayleigh quotient), followed by $k$ layers of Lie unitary mesh convolutions $f_{\text{UniMeshConv}}^{\text{Lie}}(X; A, \mathcal{W}) = \exp(\tilde A X W)$ as an encoder $E$, introducing cotangent weights $\mathcal{W}$. Finally, an MLP or GCN decoder $D$ is attached. The decoder serves two roles: mapping to target channels and actively breaking the unitary constraint. This concentrates parameter freedom at the decoder to fit arbitrary label smoothness while keeping the strong inductive bias in the backbone.

**3. Mesh Rayleigh Quotient and Unitary Convolutions on Robust Laplacian: Extending smoothness analysis to triangular meshes**

To apply these concepts to manifolds, the Rayleigh quotient must be defined on meshes. Standard symmetric cotangent Laplacians $\tilde L$ can produce negative weights on non-Delaunay triangulations, causing the Rayleigh quotient to lose its positive-definite meaning. The authors use Sharp & Crane’s Robust Laplacian, which ensures all cotangent weights $\mathcal{W}_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ satisfy the Delaunay criterion ($\alpha_{ij}+\beta_{ij}\le\pi$) via minimal edge flips, ensuring all off-diagonal elements are non-negative.

$$R_\mathcal{M}(X) = \frac{\mathrm{Tr}(X^\dagger \tilde L X)}{\|X\|_F^2}$$

On this mesh Rayleigh quotient, replacing $\tilde A$ in unitary convolutions with a normalized adjacency matrix using cotangent weights allows Corollary 1 to prove that mesh unitary convolutions similarly preserve the mesh Rayleigh quotient. This step connects cotangent weights to "strict smoothness preservation" theoretically, rather than just numerical precision.

### Loss & Training
All tasks minimize regression losses like MSE or NRMSE directly without additional Rayleigh loss terms—the core argument is that "whether to preserve smoothness" should be determined by the architectural inductive bias rather than soft constraints. R-UniMesh uses GroupSort (Anil et al. 2019) for activation to ensure activations do not destroy magnitude, employs orthogonal weights, and uses end-to-end backpropagation with GCN decoders.

## Key Experimental Results

### Main Results
Evaluation was conducted on two task categories: (1) MeshPDE (autoregressive solvers for heat, wave, and Cahn–Hilliard equations on complex PyVista meshes); (2) WeatherBench22 global weather forecasting (T850 temperature, Z500 geopotential).

| Dataset | Task | Metric | R-UniMesh | Prev. SOTA | Remarks |
|--------|------|------|-----------|----------|------|
| MeshPDE / Heat | Autoregressive 196 steps | NRMSE ↓ | **51.9 ± 3.6** | 73.0 ± 4.7 (Hermes) | Nearly halved |
| MeshPDE / Heat | As above | RE ↓ | **9.1 ± 7.4** | 14.2 ± 1.4 (EMAN) | Best smoothness match |
| MeshPDE / Wave | Autoregressive 196 steps | NRMSE ↓ | **236.5 ± 6.4** | 281.3 ± 15.5 (EMAN) | Still leading |
| MeshPDE / Cahn–Hilliard | As above | NRMSE ↓ | 123.9 ± 2.6 | **121.2 ± 1.8** (GemCNN) | Close to SOTA |
| WB22 / T850 | RMSE @ 1-10 d | RMSE / ACC | Comparable to SOTA early on | Pangu/GraphCast | Comparable under limited data |

### Ablation Study
Comparison of GCN, Lie unitary, and R-UniGraph on 2D mesh heat diffusion:

| Configuration | MSE ($\times 10^{-2}$) ↓ | RE ($\times 10^{-2}$) ↓ | Insight |
|------|--------------------------|---------------------------|------|
| GCN | 1.08 | 5.99 | Over-smoothed, high error |
| Lie Uni | 0.14 | 8.86 | Zero smoothing, under-smoothed |
| **R-UniGraph (Ours, $T_\max=3$)** | **0.11** | **2.07** | Optimal MSE and Rayleigh Error |

### Key Findings
- R-UniGraph outperforms both GCN and strictly unitary models in MSE and Rayleigh Error, indicating "just the right smoothness" is closer to physical truth than "always changing" or "never changing."
- In mesh heat diffusion, R-UniMesh's Rayleigh error matches the ground truth at almost every time step; visualizations show it is neither over-smoothed (like EMAN) nor under-smoothed (like Hermes).
- On simple geometries (e.g., Cahn-Hilliard on toroid meshes), most models perform similarly; performance gaps widen on complex geometries, proving geometric inductive bias is crucial for cross-mesh generalization.
- GCNs and EGNNs perform worst across all tasks, suggesting message passing or Euclidean equivariance alone is insufficient; explicit consideration of mesh smoothing structures is mandatory.

## Highlights & Insights
- The "Approximation error bound + knob-style relaxation" logic is elegant: Theorem 1 provides a lower bound for the variance of strictly unitary functions, diagnosing the "cost" of being strict, while Taylor truncation provides the "cure." This serves as a great template for studying other inductive biases.
- The Rayleigh Error (RE) metric is a significant contribution, providing a physically meaningful smoothness alignment measure for PDE neural surrogates compared to simple RMSE.
- Integrating Robust Laplacian, tangent weights, and unitary convolutions provides a "ready-to-use" scaffold for future manifold PDE research.

## Limitations & Future Work
- $T_\max$ and zero-pad dimensions still requires hyperparameter tuning; the authors suggest look-up tables if target smoothness is known, but an automated or learnable $T_\max$ or adaptive attention mechanism would be beneficial for unknown PDEs.
- Advantage is less pronounced on equations like Cahn–Hilliard that are neither strictly smoothing nor strictly preserving; a finer spectral analysis of the Rayleigh quotient might be needed.
- WB22 experiments were limited to $1.5°$ resolution due to compute constraints; scalability for large-scale training compared to ECMWF SOTA remains to be verified.

## Related Work & Insights
- **vs Kiani et al. 2024 (Unitary GNN)**: This work is a direct extension and "correction"—proving that strict Rayleigh preservation can be a flaw in dynamics tasks and offering controllable relaxation and mesh extension.
- **vs Hermes / EMAN / GemCNN (Gauge equivariant mesh GNN)**: While these handle directional invariance, R-UniMesh provides an orthogonal and complementary bias through Rayleigh quotient preservation, proving significantly better in tasks like heat diffusion.
- **vs Subich 2025 / Bonev 2025 (Spectral training targets)**: These works use soft constraints (spectral loss) to improve effective resolution; R-UniMesh achieves this through architectural constraints, avoiding loss-weight tuning and providing clearer physical meaning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify "over-smoothing vs under-smoothing" via the Rayleigh quotient with controllable relaxation and mesh extension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across motivating experiments, MeshPDE (multiple PDEs/meshes), and WB22; limited somewhat by compute in WB22.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from theory to method, with well-notated links between theorems and propositions.
- Value: ⭐⭐⭐⭐ Both a high-performance PDE surrogate and a theoretically significant study on mesh-GNN inductive biases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [\[CVPR 2026\] Charge: A Comprehensive Novel View Synthesis Benchmark and Dataset to Bind Them All](../../CVPR2026/3d_vision/charge_a_comprehensive_novel_view_synthesis_benchmark_and_dataset_to_bind_them_a.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[ICML 2026\] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)

</div>

<!-- RELATED:END -->
