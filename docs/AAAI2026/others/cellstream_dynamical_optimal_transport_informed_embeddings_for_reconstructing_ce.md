---
title: >-
  [Paper Note] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data
description: >-
  [AAAI 2026][single-cell RNA sequencing] This paper proposes CellStream, a deep learning framework that jointly learns an autoencoder and unbalanced dynamical optimal transport (OT) to simultaneously obtain low-dimensiona…
tags:
  - "AAAI 2026"
  - "single-cell RNA sequencing"
  - "optimal transport"
  - "cellular trajectory inference"
  - "dimensionality reduction embedding"
  - "autoencoder"
date: 2026-05-08
content_hash: ff3184f1c015b55e
---

# CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data

**Conference**: AAAI 2026
**arXiv**: [2511.13786](https://arxiv.org/abs/2511.13786)
**Code**: [github.com/PQ-Zhang/CellStream](https://github.com/PQ-Zhang/CellStream)
**Area**: Other
**Keywords**: single-cell RNA sequencing, optimal transport, cellular trajectory inference, dimensionality reduction embedding, autoencoder

## TL;DR

This paper proposes CellStream, a deep learning framework that jointly learns an autoencoder and unbalanced dynamical optimal transport (OT) to simultaneously obtain low-dimensional embeddings and continuous cellular dynamics from discrete-time single-cell snapshot data, achieving significant improvements over existing methods in temporal consistency and velocity consistency.

## Background & Motivation

### Problem Definition

Time-resolved single-cell RNA sequencing (scRNA-seq) data can capture gene expression profiles at the single-cell level across discrete time points. However, since sequencing destroys cells, the resulting data consists only of **sparse, static snapshots** rather than continuous trajectories. The core challenge is: how can one reconstruct continuous cell differentiation dynamics from such noisy snapshots?

### Limitations of Prior Work

**Geometric embedding methods** (PCA, t-SNE, UMAP, Diffusion Maps): focus on preserving topological/geometric relationships but **ignore temporal structure**, causing cell populations from different time points to overlap in the embedding space.

**RNA velocity methods** (CellPath, VeloViz, Ocelli): rely on RNA velocity estimation, which is unreliable on datasets with low unspliced counts, and do not explicitly account for cross-time-point analysis.

**Deep generative models** (scVI, GeneFormer, TarDis): based on VAE/Transformer architectures, these face interpretability challenges in dynamic modeling.

**Dynamical OT methods** (TIGON, CytoBridge): require **pre-computed embeddings** (e.g., PCA/UMAP) as input, **decoupling** embedding construction from trajectory inference and thus failing to leverage temporal structure.

### Root Cause

**Embedding construction and cellular dynamics inference should be learned jointly** — the embedding space should explicitly encode temporal information, while dynamics learning can in turn guide better embedding construction.

## Method

### Overall Architecture

CellStream comprises three learnable components:
1. **Autoencoder** $f_\theta^{enc}, f_\theta^{dec}$: maps high-dimensional gene expression to a low-dimensional embedding space and reconstructs it.
2. **Velocity field network** $\mathbf{v}_\phi$: models the direction of cellular movement in the embedding space.
3. **Growth term network** $g_\psi$: models mass changes due to cell proliferation and apoptosis.

A Block Coordinate Descent (BCD) strategy is employed to alternately optimize the autoencoder and the dynamical components.

### Key Designs

#### 1. **Unbalanced Dynamical Optimal Transport**

Standard dynamical OT assumes mass conservation, which is unsuitable for biological processes involving cell proliferation and apoptosis. CellStream introduces a growth term $g(t, \mathbf{x})$ to modify the continuity equation:

$$\partial_t \rho + \nabla_\mathbf{x} \cdot (\mathbf{v} \rho) = g \rho$$

The Wasserstein-Fisher-Rao (WFR) distance is used to jointly penalize the cost of transport and growth:

$$\mathcal{L}_{\text{WFR}} = \int_0^T \int_{\mathbb{R}^d} (\|\mathbf{v}(t,\mathbf{x})\|^2 + \alpha g^2(t,\mathbf{x})) \rho(t,\mathbf{x}) \, d\mathbf{x} \, dt$$

where $\alpha=1$ controls the relative weight between transport and growth costs.

#### 2. **Joint Optimization in Embedding Space**

The core optimization objective is:

$$\min_{f^{enc}, f^{dec}, \mathbf{v}, g} \mathcal{L}_{AE}(f^{enc}, f^{dec}) + \lambda \mathcal{L}_{\text{WFR}}^{emb}(f^{enc}, \mathbf{v}, g)$$

By leveraging the Lagrangian-Eulerian equivalence, the WFR loss is converted from PDE form to particle form, avoiding the need to solve high-dimensional PDEs:

$$\mathcal{L}_{\text{WFR}}^{emb, par} = \sum_{j=1}^{N_0} \int_0^T (\|\mathbf{v}_\phi(t, \hat{\mathbf{z}}^j(t))\|^2 + \alpha g_\psi^2(t, \hat{\mathbf{z}}^j(t))) \hat{w}^j(t) \, dt$$

Particle trajectories are efficiently computed via a Neural ODE solver.

#### 3. **Data Matching Loss**

Under the unbalanced setting, the data matching loss consists of a mass loss and an OT loss:

$$\mathcal{L}_{Match} = \lambda_{Mass} \mathcal{L}_{Mass} + \lambda_{OT} \mathcal{L}_{OT}$$

- **Mass loss**: $\mathcal{L}_{Mass} = \sum_i |\sum_j \hat{w}_i^j - N_i/N_0|$, ensuring predicted cell population sizes are consistent with observations.
- **OT loss**: $\mathcal{L}_{OT} = \sum_i \mathcal{W}_2(\hat{\mathbf{w}}_i / \|\hat{\mathbf{w}}_i\|_1, \mathbf{w}_i / \|\mathbf{w}_i\|_1)$, measuring distributional discrepancy via normalized Wasserstein-2 distance.

### Loss & Training

**Total loss**: $\mathcal{L} = \lambda_{AE} \mathcal{L}_{AE} + \lambda_{WFR} \mathcal{L}_{WFR}^{emb} + \lambda_{Match} \mathcal{L}_{Match}$

Hyperparameter settings: $\lambda_{AE}=10, \lambda_{WFR}=1, \lambda_{Match}=5, \lambda_{OT}=1, \lambda_{Mass}=1$

**Training strategy**:
1. **PCA initialization** of the autoencoder.
2. **BCD alternating optimization**: the autoencoder is first optimized with fixed dynamical components (considering both $\mathcal{L}_{AE}$ and the dynamical loss), then $\mathbf{v}_\phi$ and $g_\psi$ are optimized with a fixed autoencoder.
3. Adam optimizer is used throughout.

**Network architecture**:
- Autoencoder: 3 hidden layers, width 10, ReLU activation, output dimension 2.
- Velocity network: 4 hidden layers, width 10, Tanh activation.
- Growth network: 3 hidden layers, width 10, Tanh activation.

## Key Experimental Results

### Main Results

| Dataset | Metric | CellStream | VeloViz | MIOFlow | TIGON+PCA | TIGON+UMAP | TIGON+DiffMap |
|---------|--------|------------|---------|---------|-----------|------------|---------------|
| EMT | VC | **0.97** | 0.88 | 0.96 | 0.66 | 0.70 | 0.68 |
| EMT | TC | **0.99** | 0.70 | 0.77 | 0.59 | 0.94 | 0.57 |
| iPSC | VC | 0.97 | 0.41 | **0.98** | 0.32 | 0.74 | 0.78 |
| iPSC | TC | 0.91 | 0.92 | 0.92 | 0.94 | **0.95** | 0.84 |
| MOSTA | VC | **0.98** | 0.83 | 0.42 | 0.85 | 0.43 | 0.98 |
| MOSTA | TC | **0.99** | 0.89 | 0.92 | 0.99 | 0.99 | 0.99 |

(VC = Velocity Consistency, TC = Temporal Consistency; higher is better)

### Ablation Study

| Configuration | Key Impact | Notes |
|---------------|-----------|-------|
| Decoupled AE and dynamics | VC/TC decrease | Joint learning of embeddings and dynamics is necessary |
| Without growth term | Mass mismatch | Unbalanced OT is critical for modeling proliferation/apoptosis |
| Varying noise levels | CellStream is robust | Maintains high VC and TC under high noise |
| Hyperparameter $\alpha$ sensitivity | Moderately sensitive | Results remain relatively stable across $\alpha$ values |

### Key Findings

1. **Noise robustness on synthetic data**: Across 5 simulated datasets with progressively increasing noise, CellStream maintains consistently high VC and TC, while Diffusion Maps shows a sharp TC decline under high noise.
2. **EMT dataset**: CellStream embeddings clearly reveal the temporal structure of epithelial-mesenchymal transition, with cell populations from different time points arranged in order; VeloViz and MIOFlow exhibit severe overlap across time points.
3. **iPSC bifurcation events**: CellStream successfully uncovers the bifurcated differentiation of stem cells into mesoderm and endoderm lineages, which MIOFlow and TIGON+PCA fail to fully resolve.
4. **Spatial transcriptomics (MOSTA)**: CellStream correctly captures the dynamic growth and drift of cell populations during mouse organogenesis.

## Highlights & Insights

1. **Joint learning paradigm**: This work is the first to unify embedding learning and dynamical OT inference in an end-to-end framework, eliminating information loss from pre-computed embeddings.
2. **Dynamics-informed embeddings**: During autoencoder training, the WFR loss provides real-time trajectory feedback, causing the embedding space to actively encode temporal structure rather than being analyzed post hoc.
3. **Particle-form WFR computation**: The Euler-Lagrange equivalence is cleverly exploited to avoid directly solving high-dimensional PDEs, making the method scalable.
4. Two new evaluation metrics (VC and TC) are proposed, addressing the lack of tools for assessing embedding quality in the absence of ground truth.

## Limitations & Future Work

1. **Limited decoder reconstruction fidelity**: The authors acknowledge that the current decoder architecture is insufficient to project dynamics back to the original gene expression space with high accuracy.
2. **Fixed embedding dimension of 2**: While convenient for visualization, this limits expressiveness; more complex biological processes may require higher-dimensional embeddings.
3. **Computational efficiency**: The computational cost of the Neural ODE solver may limit applicability to large-scale datasets.
4. Integration with gene regulatory networks remains unexplored.
5. Support for multi-omics data and cell-cell communication is lacking.

## Related Work & Insights

- **TIGON** (Sha et al., 2024) is the most closely related work, but decouples embedding and dynamical inference.
- **MIOFlow** (Huguet et al., 2022) learns a low-dimensional manifold via a geodesic autoencoder but does not handle the unbalanced setting.
- **CytoBridge** (Zhang et al., 2025) combines RUOT with Mean-Field Schrödinger Bridge but similarly relies on pre-computed embeddings.
- The joint learning paradigm of CellStream may inspire other domains requiring simultaneous representation and dynamics learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (Joint embedding + dynamical OT learning is a novel problem formulation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers synthetic and multiple real datasets, with ablation and noise robustness analysis)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous mathematical derivations and clear framework description)
- Value: ⭐⭐⭐⭐ (Practically valuable for trajectory inference in single-cell biology)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LaCoOT: Layer Collapse through Optimal Transport](../../ICCV2025/others/lacoot_layer_collapse_through_optimal_transport.md)
- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](../../NeurIPS2025/others/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
