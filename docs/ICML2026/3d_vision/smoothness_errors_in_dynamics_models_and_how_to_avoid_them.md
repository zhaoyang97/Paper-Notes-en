---
title: >-
  [Paper Note] Smoothness Errors in Dynamics Models and How to Avoid Them
description: >-
  [ICML 2026][3D Vision][GNN] The authors theoretically point out that the "unitary GNN" by Kiani et al. over-constrains physical systems that "naturally become smooth," such as heat diffusion, by forcibly maintaining the Rayleigh quotient. They further propose "relaxed unitary convolution" (R-UniGraph / R-UniMesh) and extend the entire Rayleigh qu
tags:
  - ICML 2026
  - 3D Vision
  - GNN
date: 2026-05-08
content_hash: 198bd7a60463cb38
---
# Smoothness Errors in Dynamics Models and How to Avoid Them

**Conference**: ICML 2026  
**arXiv**: [2602.05352](https://arxiv.org/abs/2602.05352)  
**Code**: Available (provided at the end of the paper)  
**Area**: 3D Vision / Geometric Deep Learning / PDE Neural Solvers  
**Keywords**: GNN, mesh learning, oversmoothing/undersmoothing, Unitary convolution, Rayleigh quotient, weather forecasting

## TL;DR
The authors theoretically point out that the "unitary GNN" by Kiani et al. over-constrains physical systems that "naturally become smooth," such as heat diffusion, by forcibly maintaining the Rayleigh quotient. They further propose "relaxed unitary convolution" (R-UniGraph / R-UniMesh) and extend the entire Rayleigh quotient-unitary convolution framework from graphs to triangular meshes, outperforming several strong baselines in MeshPDE and WeatherBench22.

## One-sentence Supplement: Core Proposition
GNNs should neither be oversmoothed nor undersmoothed—the smoothing tendency of the architecture must precisely match the smoothing tendency of the true physical process.

## Background & Motivation

**Background**: Solving PDEs defined on meshes/manifolds (heat diffusion, wave equations, Cahn–Hilliard, global atmosphere) using neural networks has been one of the most active directions in scientific computing over the past two years. The mainstream approach involves discretizing the manifold into a mesh and using mesh-GNNs that support high-order connectivity (GCN, MPNN, EGNN, Gauge-Equivariant CNN, Hermes, etc.) for message passing. However, GNNs generally suffer from oversmoothing: as the number of layers increases, neighbor node features tend to converge. Kiani et al. recently proposed "unitary graph convolution," which constrains the weight matrix to be a unitary matrix, effectively maintaining the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$ to strictly prevent oversmoothing.

**Limitations of Prior Work**: Compared to GCN, unitary convolution represents the extreme of "zero smoothing" versus "excessive smoothing." However, most real-world physical systems possess a "just-right" degree of smoothness—heat diffusion makes features increasingly smooth, while the wave equation requires maintaining high-frequency structures. Forcing unitary convolution onto these systems leads to "undersmoothing," where the network cannot learn the intermediate smoothing process required by the physics.

**Key Challenge**: GCN and unitary convolution represent two extremes regarding the Rayleigh quotient: GCN strictly reduces the Rayleigh quotient (continuous smoothing), while unitary convolution strictly maintains it (no smoothing). Any realistic physical dynamics requires a "tunable" smoothing rate rather than a choice between two extremes.

**Goal**: (i) Theoretically provide a lower bound for the approximation error of unitary functions, proving they are over-constrained on targets with strong angular dependence; (ii) design controllable "relaxed unitary convolutions" that allow the network to switch freely between the two extremes; (iii) extend the Rayleigh quotient and unitary convolution from graphs to meshes, making them applicable to real physical tasks such as PDE solving and weather forecasting.

**Key Insight**: The authors found that the unitarity of Lie unitary convolution $f(X) = \exp(AXW)$, where $W = -W^\dagger$, stems from expanding the Taylor series to "infinite orders." If truncated at a specific order $T_\max$, the resulting layer is no longer strictly unitary but approaches unitarity in a tunable manner—this serves as a natural "continuous relaxation knob."

**Core Idea**: Relaxing strict Rayleigh quotient preservation is achieved through Taylor-truncated Lie convolutions or a "zero-pad + unitary encoder + arbitrary decoder" approach. This allows the network to adaptively match the true smoothness of the physical process during training. The entire theory is migrated to meshes using a Robust Laplacian and cotangent weights.

## Method

### Overall Architecture
The methodology aims to address the mismatch between the smoothing tendency of GNNs and the actual physical processes—it should neither become progressively smoother like GCNs nor remain entirely unsmoothed like strict unitary convolutions. Instead, the smoothing rate should be tunable. The general idea revolves around the Rayleigh quotient $R_\mathcal{G}(X) = \mathrm{Tr}(X^\dagger L X)/\|X\|_F^2$ as a unified metric for smoothness. First, it is theoretically proven that strict unitary functions have a non-vanishing lower bound on approximation error for targets where "magnitude varies with angle," explaining "why relaxation is necessary." Second, two relaxation paths are provided to transform "strict Rayleigh quotient preservation" into a "tunable smoothing rate": Taylor truncation (resulting in R-UniGraph on graphs) and zero-padded encoding-decoding (resulting in R-UniMesh on meshes). Finally, the Rayleigh quotient and unitary convolution are bridged from graphs to triangular meshes using the Robust Laplacian, allowing the framework to be applied to manifold PDE tasks. The final model uses GroupSort as an activation (preserving magnitude) and employs an MLP/GCN decoder to actively break unitary constraints for flexibility.

### Key Designs

**1. Taylor-Truncated Relaxed Lie Convolution (R-UniGraph): A knob for seamless switching between oversmoothing and undersmoothing**

The pain point is that GCN strictly reduces the Rayleigh quotient (continuous smoothing) and unitary convolution strictly maintains it (zero smoothing); both are hard-coded extremes. Realistic physical dynamics require an intermediate, tunable smoothing rate. The authors noticed that the unitarity of Lie unitary convolution $\exp(AXW)$ ($W=-W^\dagger$) comes from the infinite-order Taylor expansion of the matrix exponential. By truncating it at the $T_\max$-th order, they obtain $f_{\text{Relaxed}}(X; A, T_\max) = \sum_{i=0}^{T_\max} \frac{1}{i!} L^i(X)$, where $L(X)=AXW$. Thus, $T_\max$ becomes a continuous knob for the smoothing rate: at $T_\max=1$, it approximates GCN behavior; as $T_\max\to\infty$, it recovers strict Lie unitary convolution. Intermediate values (e.g., $T_\max=3$ for heat diffusion, $T_\max=10$ for others) allow the network to perform small smoothing corrections while largely maintaining the Rayleigh quotient.

The key difference compared to the "separable unitary convolution" relaxation previously proposed by Kiani et al. is the isolation of the source of relaxation. Their relaxation modified two aspects simultaneously—truncating the Taylor series and making $U$ non-unitary—making it impossible to quantify the source of relaxation. R-UniGraph keeps the anti-symmetric $W$ in Lie form, making $T_\max$ the sole knob for Rayleigh quotient preservation. If the smoothness of the target process is known physically, $T_\max$ can even be selected via a lookup table.

**2. Zero-Padded Encoder-Decoder Relaxation (R-UniMesh): Concentrating capacity in the decoder to bypass training instability of deep unitary stacks**

The Taylor truncation path cannot change channel dimensions; increasing parameters requires deepening the network, but deep unitary stacks face "shattered gradients" training instability as noted by Balduzzi et al. R-UniMesh adopts a "wide + shallow" approach: it first uses zero-padding $f_{\text{pad}}: \mathbb{R}^{n\times d_{in}}\to\mathbb{R}^{n\times d_{out}}$ to lift node features to the hidden dimension (zero-padding preserves magnitude and thus naturally the Rayleigh quotient), then stacks $k$ layers of Lie unitary mesh convolution $f_{\text{UniMeshConv}}^{\text{Lie}}(X; A, \mathcal{W}) = \exp(\tilde A X W)$ as an encoder $E$, where $\tilde A = D^{-1/2}(\mathcal{W}\odot A)D^{-1/2}$ introduces cotangent weights $\mathcal{W}$. Finally, an MLP or GCN decoder $D$ is attached.

The decoder serves a dual purpose: mapping features to the target channel count and actively breaking unitary constraints. This is equivalent to a clear division of labor—the unitary encoder maintains geometric and smoothness structures, while the decoder fits the output to arbitrary label smoothness. Parameter freedom is concentrated at the decoder side, which is not restricted by unitarity, thus retaining the strong inductive bias of the backbone while insulating training instability from the shallow unitary core.

**3. Mesh Rayleigh Quotient and Unitary Convolution on Robust Laplacian: Moving smoothness analysis to triangular meshes**

To apply the above paths to manifold PDE tasks, the Rayleigh quotient and unitary convolution must be extended from graphs to meshes. The obstacle is that the traditional symmetric cotangent Laplacian $\tilde L$ can have negative weights under non-Delaunay triangulations, causing the Rayleigh quotient to lose its positive-definite meaning. The authors utilize the Robust Laplacian of Sharp & Crane, which ensures all cotangent weights $\mathcal{W}_{ij} = \frac{1}{2}(\cot\alpha_{ij} + \cot\beta_{ij})$ satisfy the Delaunay criterion ($\alpha_{ij}+\beta_{ij}\le\pi$) through minimal edge intrinsic flips, ensuring all off-diagonal elements are non-negative.

$$R_\mathcal{M}(X) = \frac{\mathrm{Tr}(X^\dagger \tilde L X)}{\|X\|_F^2}$$

On top of this mesh Rayleigh quotient, one only needs to replace the original $\tilde A$ in separable/Lie unitary convolutions with the normalized adjacency matrix using cotangent weights. Corollary 1 proves that both mesh versions of unitary convolution similarly preserve the mesh Rayleigh quotient. The ingenuity lies in the fact that while mesh-GNNs previously used cotangent weights for numerical precision, no one linked them to "strict smoothness preservation." By using the Delaunay assumption and the Robust Laplacian to secure the "positive weights" condition, all mathematical conclusions of the unitary framework on graphs migrate automatically to meshes, bypassing the need for redundant algebraic proofs.

### Loss & Training
All tasks directly minimize regression losses such as MSE/NRMSE; no additional Rayleigh loss term is introduced. The authors' key argument is that "smoothness preservation" should be determined by the architecture's inductive bias rather than soft constraints. R-UniMesh uses GroupSort (Anil et al., 2019) as the activation to ensure it does not destroy magnitude. It utilizes orthogonal weights (sufficient for real-valued tasks) and is trained end-to-end with a GCN decoder via backpropagation.

## Key Experimental Results

### Main Results
The authors evaluate on two types of tasks: (1) MeshPDE (autoregressive solving of heat, wave, and Cahn–Hilliard equations on complex PyVista meshes); (2) WeatherBench22 global weather forecasting (T850 temperature and Z500 geopotential).

| Dataset | Task | Metric | R-UniMesh | Prev. SOTA | Remarks |
|--------|------|------|-----------|----------|------|
| MeshPDE / Heat | Autoregressive 196 steps | NRMSE ↓ | **51.9 ± 3.6** | 73.0 ± 4.7 (Hermes) | Nearly halved |
| MeshPDE / Heat | Same as above | RE ↓ | **9.1 ± 7.4** | 14.2 ± 1.4 (EMAN) | Best smoothness match |
| MeshPDE / Wave | Autoregressive 196 steps | NRMSE ↓ | **236.5 ± 6.4** | 281.3 ± 15.5 (EMAN) | Still leading |
| MeshPDE / Cahn–Hilliard | Same as above | NRMSE ↓ | 123.9 ± 2.6 | **121.2 ± 1.8** (GemCNN) | Close to SOTA |
| WB22 / T850 | RMSE @ 1-10 d | RMSE / ACC | Comparable to early SOTA | Pangu/GraphCast | Competitive despite limited data |

### Ablation Study
The authors compare GCN, Lie unitary, and R-UniGraph for heat diffusion on a 2D grid in a motivating experiment:

| Configuration | MSE ($\times 10^{-2}$) ↓ | MRE ($\times 10^{-2}$) ↓ | Interpretation |
|------|--------------------------|---------------------------|------|
| GCN | 1.08 | 5.99 | Oversmoothed, high error |
| Lie Uni | 0.14 | 8.86 | Zero smoothing, undersmoothed |
| **R-UniGraph (Ours, $T_\max=3$)** | **0.11** | **2.07** | Optimal MSE and Rayleigh error |

### Key Findings
- R-UniGraph outperforms both GCN and strict unitary in MSE and Rayleigh error, indicating that "just-right smoothness" is closer to physical ground truth than "never changing" or "always changing."
- In heat diffusion prediction on meshes, R-UniMesh's Rayleigh error perfectly aligns with the ground truth at almost every timestep. Visualizations show its rollouts are neither oversmoothed like EMAN nor undersmoothed like Hermes.
- On simple geometries (e.g., toroid mesh for Cahn–Hilliard), almost all equivariant/unitary/MPNN models perform similarly. Differences emerge mainly in complex geometry generalization (different PyVista meshes), proving that "geometric inductive bias" is crucial for cross-mesh generalization.
- GCN and EGNN rank last in all tasks, suggesting that message passing or Euclidean equivariance alone is insufficient to simulate PDEs on manifolds; smoothing structures on the mesh must be explicitly considered.

## Highlights & Insights
- The design logic of "approximation error lower bound + knob-style relaxation" is elegant: starting with Theorem 1 (providing a lower bound on $\int p(\|te\|)\mathbb{V}_{Gz}[\|f\|]dz$ through integration of magnitude variance over a fundamental domain) to explain the "cost" of strict unitarity, then using Taylor truncation to turn "strict" into "tunable." This "diagnosis-treatment" writing template is instructive for other inductive bias studies.
- The Rayleigh Quotient Error (RE) metric is itself a significant contribution: it provides a smoothness alignment metric for PDE neural surrogates that is more physically meaningful than RMSE. Future mesh-GNN papers should adopt it as a standard metric.
- Packaging the Robust Laplacian, cotangent weights, and unitary convolution into a complete mesh framework provides a "ready-to-use scaffold" for researchers working on manifold PDE solvers.

## Limitations & Future Work
- $T_\max$ and zero-pad dimensions still require tuning based on task priors. The authors suggest using a lookup table for $T_\max$ when target smoothness is known, but there is no automatic scheduling strategy for unknown PDEs. Future work could consider learnable $T_\max$ or adaptive attention to dynamically determine truncation orders.
- The advantage is less pronounced on equations like Cahn–Hilliard, which are neither strictly smoothing nor strictly preserving; the current binary perspective of "loosening both ends" may be too coarse for truly intermediate systems. Finer Rayleigh quotient spectral analysis might be required.
- The WB22 experiments were limited by compute to $1.5°$ resolution and small-scale training, leaving a gap compared to ECMWF SOTA. Scalability on large scales needs further verification.

## Related Work & Insights
- **vs Kiani et al. 2024 (Unitary GNN)**: This work is a direct extension and "reverse correction"—proving that strict Rayleigh quotient preservation is a defect in dynamics tasks while providing controllable relaxation through Taylor truncation and extending the theory to meshes.
- **vs Hermes / EMAN / GemCNN (Gauge equivariant mesh GNN)**: These methods handle directional invariance on meshes via gauge equivariance. R-UniMesh provides an orthogonal and complementary inductive bias through Rayleigh quotient preservation, performing significantly better on strongly smoothing tasks like heat diffusion.
- **vs Subich 2025 / Bonev 2025 (Spectral training targets)**: These works improve the effective resolution of weather models via soft constraints (spectral loss). R-UniMesh achieves similar goals through architectural constraints, avoiding loss-weight tuning and providing clearer physical meaning for PDE tasks.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified characterization of "oversmoothing vs undersmoothing" via Rayleigh quotient, with controllable relaxation and mesh extension. Both theory and method are new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across motivating experiments, MeshPDE (multi-PDE, multi-mesh), and real-world WB22 data. WB22 is slightly limited by compute.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamless transition between theory and method, motivation and experiments; citations for theorems and propositions are clearly marked.
- Value: ⭐⭐⭐⭐ Serves as both a usable PDE neural surrogate (SOTA on tasks like heat diffusion) and a theoretically significant study of mesh-GNN inductive biases.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One Diffusion to Generate Them All](../../CVPR2025/3d_vision/one_diffusion_to_generate_them_all.md)
- [\[NeurIPS 2025\] Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting](../../NeurIPS2025/3d_vision/temporal_smoothness-aware_rate-distortion_optimized_4d_gaussian_splatting.md)
- [\[CVPR 2026\] LumiMotion: Improving Gaussian Relighting with Scene Dynamics](../../CVPR2026/3d_vision/lumimotion_gaussian_relighting_dynamics.md)
- [\[ICML 2026\] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation](foundobj_self-supervised_foundation_models_as_rewards_for_label-free_3d_object_s.md)
- [\[ICLR 2026\] Omni-View: Unlocking How Generation Facilitates Understanding in Unified 3D Model based on Multiview images](../../ICLR2026/3d_vision/omni-view_unlocking_how_generation_facilitates_understanding_in_unified_3d_model.md)

</div>

<!-- RELATED:END -->
