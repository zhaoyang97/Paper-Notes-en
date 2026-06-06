---
title: >-
  [Paper Note] Physics-Informed Coarsening for Multigrid Graph Neural Surrogates
description: >-
  [ICML 2026][Graph Learning][Multigrid GNN] This paper trains an Encoder-Processor-Decoder multigrid GNN surrogate model for finite element simulation in solid mechanics. The core innovation lies in replacing the "node se…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Multigrid GNN"
  - "Physical Residuals"
  - "Solid Mechanics Surrogates"
  - "Mesh Coarsening"
  - "Long-term Rollout"
date: 2026-05-08
content_hash: 55c0a7dfb5208ca3
---

# Physics-Informed Coarsening for Multigrid Graph Neural Surrogates

**Conference**: ICML 2026  
**arXiv**: [2605.31013](https://arxiv.org/abs/2605.31013)  
**Code**: [Project Page](https://sites.google.com/view/physics-informed-coarsening)  
**Area**: Scientific Computing / Graph Neural PDE Surrogates / Solid Mechanics  
**Keywords**: Multigrid GNN, Physical Residuals, Solid Mechanics Surrogates, Mesh Coarsening, Long-term Rollout

## TL;DR
This paper trains an Encoder-Processor-Decoder multigrid GNN surrogate model for finite element simulation in solid mechanics. The core innovation lies in replacing the "node selection during coarsening (downsampling)" mechanism—typically based on geometric heuristics (FPS) or learned attention—with a "TopK scoring based on the discrete residuals of momentum conservation equations." This concentrates coarse-layer computational power on key dynamic regions such as stress concentrations, contact interfaces, and large deformations. On the DeformingPlate dataset, it reduces the rollout RMSE from the SOTA $11.46\times 10^{-3}$ to $6.5\times 10^{-3}$ (an improvement of approximately 43%).

## Background & Motivation

**Background**: Replacing Finite Element Methods (FEM) with neural networks for PDE simulation has achieved orders-of-magnitude acceleration in fluid mechanics (Navier-Stokes, turbulence, airfoils). The mainstream architecture consists of Encode-Process-Decode graph neural networks (MeshGraphNet series), often combined with multigrid or U-Net-style hierarchical message passing (MultiScale MeshGraphNets, BSMS-GNN, Multi-Scale GNN, HCMT, UNISOMA) to mitigate over-smoothing in deep GNNs and enhance long-range information propagation.

**Limitations of Prior Work**: (i) Solid mechanics is significantly underestimated; unlike fluids, it involves strong non-linear local phenomena such as large deformations, plasticity, contact, and stress concentrations, which mainstream benchmarks (predominantly fluid-based) fail to capture. (ii) The core design of "choosing which nodes to retain during coarsening" in multigrid architectures generally relies on pure geometric heuristics like Farthest Point Sampling (FPS) or learned attention scores. The former ignores physics and distributes nodes uniformly across the domain, wasting computation on physically inactive regions, while the latter is prone to instability during training.

**Key Challenge**: Coarse-layer nodes are limited (fixed at 50% in this study). If nodes are distributed uniformly via geometry, regions that are "dynamically critical but spatially local," such as stress concentrations and contact interfaces, receive insufficient coarse-layer resolution. During long-term rollout, errors in these regions are the first to diverge, subsequently polluting the entire solution.

**Goal**: (i) Design a coarsening criterion that prioritizes "physically important" regions for coarse-layer nodes; (ii) Ensure this criterion is applicable across quasi-static hyperelasticity, transient non-linear elasticity, and elastoplasticity with contact; (iii) Release a solid mechanics benchmark to fill the current gap in the field.

**Key Insight**: The authors borrow the classical FEM concept of "residual-based adaptive mesh refinement," where meshes are refined in areas with high PDE residuals. They transfer this strategy to GNN multigrid: nodes are scored during coarsening based on the discrete residuals of the momentum conservation equations.

**Core Idea**: Use the "norm of the discrete residual of the momentum conservation equation" as the node importance score and select the TopK nodes with the highest residuals to construct the coarse graph. This naturally steers the multigrid hierarchy towards stress concentrations, contact interfaces, and large deformation zones.

## Method

### Overall Architecture
Input: A 3D unstructured mesh $\mathcal{G}=(\mathcal{V},\mathcal{E})$ and node physical fields $\bm{u}^t$ (displacement, velocity, or position) at time $t$.  
Output: The next-step increment $\bm{u}^{t+1}=\bm{u}^t+\Phi_\theta(\bm{u}^t,\mathcal{G})$, utilizing residual-style time-stepping to facilitate stable long-term rollout.

The backbone is an Encoder–Processor–Decoder. The Encoder uses point-wise MLPs to lift node features to a hidden dimension $h$; the Decoder maps the final hidden features back to $\mathbb{R}^3$. The Processor alternates between three operators in the latent space:
(i) GraphNet blocks $\mathrm{GN}$ for fine-mesh message passing (following standard MeshGraphNet update rules);
(ii) Downsampling blocks $\mathrm{DN}$ that compress the fine graph into a coarse graph with $n_s=0.5n$ nodes;
(iii) Upsampling blocks $\mathrm{UP}$ that use KNN to interpolate coarse-graph features back to the fine graph, fusing them with fine-layer features before further GraphNet processing. The system follows a U-Net-style schedule $\mathcal{G}\to\tilde{\mathcal{G}}\to\mathcal{G}_c\to\tilde{\mathcal{G}}$, where the coarse layer is responsible for expanding the effective receptive field and propagating global information.

The key innovation is inside the $\mathrm{DN}$ block: instead of geometric selection, it temporarily decodes the current hidden features into physical quantities using the main Decoder $\phi_{\mathrm{dec}}$, calculates the "momentum conservation residual" $\bm{r}_i^t$ for each node, and ranks them by $s_i^t=\|\bm{r}_i^t\|_2$ for TopK selection.

### Key Designs

1.  **Residual-based Physical Scoring**:
    -   **Function**: Computes a scalar score $s_i^t$ for each node, characterizing the extent to which the predicted physical field violates momentum conservation, serving as an a posteriori indicator of node importance.
    -   **Mechanism**: The main Decoder $\hat{\bm{u}}^t=\phi_{\mathrm{dec}}(\tilde{\mathcal{G}})$ is "borrowed" to map the current hidden graph to physical space (this step does not participate in the final prediction). The predicted field $\hat{\bm{u}}^t$ is used to compute stress $\hat{\bm{\sigma}}^t$. For transient cases, the residual is $\bm{r}_i^t=\rho_i\ddot{\hat{\bm{u}}}_i^t-(\nabla_h\cdot\hat{\bm{\sigma}}^t)_i-\rho_i\mathbf{b}_i^t$; for quasi-static cases, the inertial term is dropped to form the equilibrium residual $\bm{r}_i^t=-(\nabla_h\cdot\hat{\bm{\sigma}}^t)_i-\rho_i\mathbf{b}_i^t$. Divergence $\nabla_h\cdot$ is reconstructed using fixed mesh-based discrete operators, and the score is $s_i^t=\|\bm{r}_i^t\|_2$.
    -   **Design Motivation**: In FEM, regions with high residuals are physically "hard-to-predict" or involve intense dynamics. Translating this to GNN coarsening allows the multigrid hierarchy to automatically capture stress concentrations and contact interfaces. Notably, using the **shared main Decoder** for scoring (rather than an independent one) forces the main branch to learn physically consistent representations.

2.  **TopK Physics-Guided Node Selection + KNN Remeshing**:
    -   **Function**: Converts the score vector $\bm{s}\in\mathbb{R}^n$ into a set of $n_s$ coarse node indices $\mathcal{V}_c$ and reconstructs the coarse edge set $\mathcal{E}_c$.
    -   **Mechanism**: Node selection employs either deterministic TopK $\mathcal{I}=\mathrm{TopK}(\bm{s}^t,n_s)$ or categorical sampling based on $p_i=s_i/\sum_j s_j$. For edge construction, the model either inherits the fine-mesh subgraph or uses Euclidean KNN to "remesh" the selected nodes. The optimal combination found is **TopK + remeshing**, which significantly outperformed categorical sampling and inherited connectivity in stability.
    -   **Design Motivation**: TopK is more aggressive than random sampling under physical guidance, dedicating all coarse-layer capacity to the most critical regions. KNN remeshing avoids topological fragmentation, which is crucial for long-range information propagation.

3.  **Encoder-Processor-Decoder + KNN Upsampling Fusion**:
    -   **Function**: Returns processed global information from the coarse graph to the fine graph while maintaining fine-level local precision.
    -   **Mechanism**: After processing in coarse GraphNet blocks to obtain $\tilde{\mathcal{G}}_c^{n_s\times h}$, $k$-NN (in physical Euclidean space) is used to interpolate features for fine nodes as a weighted sum of their $k$ nearest coarse neighbors. These are fused with original fine features before final GraphNet refinement.
    -   **Design Motivation**: Traditional single-scale GNNs are limited by message-passing radii and cannot capture global coupling. Multigrid coarse layers bridge long distances in a single hop, while the Encoder-Processor-Decoder structure preserves local stress gradients.

### Loss & Training
Direct supervision of next-state prediction is performed using node-wise MSE loss. The AdamW optimizer is used, and all baselines are trained for 30 epochs (~$10^6$ steps) under the same protocol. All multigrid models use a fixed coarsening ratio of 50% to control for capacity, isolating "coarsening strategy" as the primary variable. Experiments were conducted on NVIDIA A100 GPUs.

## Key Experimental Results

### Main Results: Comparison with 7 SOTA Methods on DeformingPlate

| Method | Rollout RMSE ($\times 10^{-3}$) ↓ | 1-step RMSE ($\times 10^{-3}$) ↓ | #Params ↓ |
| :--- | :--- | :--- | :--- |
| MeshGraphNets | 12.75 | 0.10 | 2.8M |
| BSMS-GNN | 16.60 | 0.15 | 2.1M |
| Transolver++ | 29.80 | 1.00 | 722K |
| Transformer GNN | 24.97 | 1.20 | 3.5M |
| Multi-Scale GNN | 15.7 | 0.10 | 3.1M |
| HCMT | 12.97 | 0.14 | 2.53M |
| UNISOMA | 11.46 | 0.16 | 2.85M |
| **Ours (Physics-informed Multigrid)** | **6.50** | **0.095** | 2.9M |

Rollout error is halved compared to UNISOMA (11.46 → 6.50, ~43% gain), and the 1-step error is the lowest among all methods, achieved without increasing parameter counts.

### Ablation Study: Sampling Strategies in Multigrid Architecture

| Sampling Strategy | Rollout RMSE ($\times 10^{-3}$) ↓ | 1-step RMSE ($\times 10^{-5}$) ↓ |
| :--- | :--- | :--- |
| BSMS (Topological bi-stride) | 16.60 | 15 |
| FPS (No remeshing) | 15.0 | 10.31 |
| Attention-based | 8.1 | 17.10 |
| FPS (With remeshing) | 8.0 | 9.74 |
| Physics-informed + Categorical | 13.1 | 11.32 |
| **Physics-informed + TopK (Ours)** | **6.5** | **9.57** |

### Key Findings
-   **Coarsening Strategy > Architecture Capacity**: Maintaining the same backbone but replacing FPS with physics-informed TopK reduces rollout error from 8.0 to 6.5.
-   **TopK Significantly Outperforms Probabilistic Sampling** (13.1 → 6.5): At a high coarsening rate of 50%, the variance introduced by randomness outweighs the benefits of exploration.
-   **Decoder Reuse is Essential**: Using an independent scoring decoder degrades performance. Sharing the decoder forces the representation to support both prediction and residual calculation, promoting physical consistency.
-   **Include Boundary/Contact Nodes in Scoring**: Results show that scoring all nodes outperforms scoring only "normal" nodes, as reaction forces and constraints carry strong physical signals.
-   **Increased Coarse-Layer Width is Ineffective**: Increasing $h_c$ from 128 to 256 degraded performance, suggesting that the bottleneck lies in selection rather than capacity.

## Highlights & Insights
-   **Translating Residual-based Adaptive Refinement to GNN Multigrid**: This analogy provides explainability ("high residual = intense physical activity") and leverages decades of numerical analysis intuition.
-   **Zero Additional Parameters for Scoring**: The physical residual is computed entirely using the main Decoder and fixed discrete operators, introducing a powerful inductive bias without new learnable modules.
-   **Decoder Sharing as a Technical Trick**: Having one decoder perform both final output and intermediate scoring naturally injects physical consistency as a regularizer.
-   **TopK + KNN Remeshing**: Prioritizing "semantic relevance" via remeshing over "topological fidelity" (preserving original connectivity) proves superior for multigrid GNNs.

## Limitations & Future Work
-   **Complexity of Residual Scoring**: Implementing residual scoring for complex materials or distorted meshes requires discrete divergence reconstruction, moving away from pure "black-box" ML.
-   **Robustness in Extreme Regimes**: On the SpindleUpsetting dataset (heavy plasticity and contact), the method slightly underperformed pure FPS, suggesting residual signals may be noisy in extreme non-linear regimes.
-   **Fixed Coarsening Rate**: The study only evaluated a 50% coarsening rate and did not investigate multi-level nesting (e.g., coarse-coarser-coarsest).
-   **Future Directions**: Integrating residual scoring with gradient vector fields, implementing adaptive coarsening rates, and combining the mechanism with PINN losses for joint regularization.

## Related Work & Insights
-   **vs MeshGraphNets**: Ours inherits the backbone and residual time-stepping from MGN but adds the multigrid hierarchy and physics-guided coarsening, improving rollout performance from 12.75 to 6.50.
-   **vs BSMS-GNN**: BSMS uses topological bi-stride pooling. Ours uses physical TopK and KNN remeshing, demonstrating that semantic focus is more effective than preserving original topology.
-   **vs FPS-based Multigrid**: Geometric uniform coverage vs. physics-guided focus. In solid mechanics, where local concentrations dictate global dynamics, physics guidance is clearly superior.
-   **vs UNISOMA**: UNISOMA uses fixed slice tokens for attention-based compression, which may lose details at sharp contact interfaces; Ours explicitly selects nodes based on residuals to mitigate this.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)
- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
