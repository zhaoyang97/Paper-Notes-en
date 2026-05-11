---
title: >-
  [Paper Note] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization
description: >-
  [AAAI 2026][Graph Learning][Sheaf Neural Networks] This paper proposes SGPC (Sheaf GNNs with PAC-Bayes Calibration), which integrates Wasserstein optimal transport for learning sheaf restriction maps…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Sheaf Neural Networks"
  - "PAC-Bayes"
  - "Spectral Optimization"
  - "Heterophilic Graphs"
  - "Optimal Transport"
  - "Semi-supervised Node Classification"
date: 2026-05-08
content_hash: 54736f2764f26490
---

# Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization

**Conference**: AAAI 2026
**arXiv**: [2508.00357](https://arxiv.org/abs/2508.00357)
**Code**: [GitHub](https://github.com/ChoiYoonHyuk/SGPC)
**Area**: Graph Learning / Graph Neural Networks
**Keywords**: Sheaf Neural Networks, PAC-Bayes, Spectral Optimization, Heterophilic Graphs, Optimal Transport, Semi-supervised Node Classification

## TL;DR

This paper proposes SGPC (Sheaf GNNs with PAC-Bayes Calibration), which integrates Wasserstein optimal transport for learning sheaf restriction maps, variance-reduced diffusion with an adaptive frequency mixing layer, and PAC-Bayes spectral regularization. SGPC consistently outperforms existing GNN and sheaf methods on both homophilic and heterophilic graph node classification benchmarks while providing theoretical generalization guarantees.

## Background & Motivation

### Limitations of Prior Work

**Background**: Classic GNNs (GCN, GAT) are essentially low-pass filters that perform well on homophilic graphs (where neighboring nodes share the same label) but suffer severe performance degradation on heterophilic graphs—a manifestation of the over-smoothing problem. Cellular sheaf theory reinterprets edges as linear restriction maps between local feature spaces; the spectrum of the sheaf Laplacian captures edge directionality and class dispersion.

Existing sheaf GNNs exhibit three key limitations: (1) restriction maps are typically fixed or controlled by simple gating; (2) they lack heterophily-aware uncertainty calibration; and (3) they offer no generalization guarantees beyond empirical test accuracy. PAC-Bayes analysis can provide principled generalization bounds, yet existing PAC-Bayes bounds for GNNs remain loose because they neglect the spectrum of the underlying operator.

### Root Cause

**Goal**: How can one simultaneously learn sheaf restriction maps and tighten PAC-Bayes generalization bounds through spectral gap optimization, thereby achieving provably strong performance on heterophilic graphs?

## Method

### Overall Architecture

SGPC consists of three main modules: (1) an OT + WE Lift for learning sheaf restriction maps; (2) SVR-AFM layers for diffusion and frequency mixing; and (3) PAC-Bayes spectral optimization for uncertainty calibration and tightening of generalization bounds.

### Key Designs

1. **Wasserstein-Entropic Lift**: A Sinkhorn OT plan $P_0$ is used for initialization, which is then refined via JKO gradient flow steps into a globally KL-stable configuration $P_\star$. Restriction maps are subsequently generated as $R_{ij} = P_{\star,ij} W_\theta$. These maps constitute the sheaf Laplacian: $L_\mathcal{F} = (B \otimes I_{d_0})^\top R^\top R (B \otimes I_{d_0})$.

2. **SVR-AFM Layer**:

   - Stochastic variance-reduced (SVR) diffusion: $H^{svr} \approx (I + \Delta t L_\mathcal{F})^{-1} H$
   - Adaptive frequency mixing (AFM) via Chebyshev polynomials: $H^{afm} = \sum_{q=0}^Q \alpha_q T_q(\tilde{L}) H$, where learnable coefficients $\alpha_q$ automatically emphasize high frequencies on heterophilic graphs
   - Branch fusion: $H' = F_{mix}([H^{svr} \| H^{afm}])$

3. **PAC-Bayes Spectral Optimization**:

   - A $\beta$-Dirichlet prior models the message consistency rate $\kappa_{ij}$ for each edge
   - A fixed-point solver iteratively updates the posterior
   - Spectral gap regularization: $\mathcal{L}_{spec} = c_{het} / \lambda_2(L_\mathcal{F})$
   - Total loss: $\mathcal{L} = \mathcal{L}(y, \hat{y}) + \lambda_{KL}\mathcal{L}_{KL} + \lambda_{spec}\mathcal{L}_{spec}$
   - A theorem guarantees that $\lambda_2$ increases monotonically (by at least $c_w/4$ per iteration), causing the PAC-Bayes bound to shrink geometrically

## Key Experimental Results

### Node Classification (9 Benchmarks, Accuracy %)

### Main Results

| Method | Cora | Citeseer | Actor | Chameleon | Cornell | Texas | Wisconsin |
|--------|------|----------|-------|-----------|---------|-------|-----------|
| GCN | 81.3 | 71.1 | 20.4 | 49.8 | 60.2 | 68.3 | 57.7 |
| GAT | 82.5 | 72.0 | 21.7 | 49.3 | 63.4 | 70.2 | 59.4 |
| GCNII | 82.1 | 71.4 | 25.1 | 49.1 | 79.7 | 82.6 | 75.3 |
| NSD (sheaf) | — | — | — | — | — | — | — |
| **SGPC** | **top-3** | **top-3** | **top-3** | **top-3** | **top-3** | **top-3** | **top-3** |

The advantage is particularly pronounced on heterophilic graphs: SGPC substantially outperforms classic GNNs on Cornell, Texas, and Wisconsin (homophily ratios of only 0.06–0.16).

## Highlights & Insights

- **Theoretical Completeness**: The paper provides four levels of theoretical guarantees—CG convergence, monotonic growth of $\lambda_2$, risk-variance contraction, and PAC-Bayes generalization bounds.
- **Elegant Connection Between Spectral Gap and Generalization**: The $c_{het}/\lambda_2$ regularization term ties heterophily penalization directly to diffusion capacity.
- **Clean Modular Design**: The four-stage pipeline OT → Sheaf → SVR-AFM → PAC-Bayes proceeds in a logically progressive manner.
- **Joint Handling of Homophilic and Heterophilic Graphs**: The learnable frequency coefficients in the AFM branch adapt automatically to the graph's homophily level.

## Limitations & Future Work

- The computational overhead of OT and JKO steps may become a bottleneck for large-scale graphs.
- The convergence speed of the fixed-point solver for the $\beta$-Dirichlet posterior depends on the choice of prior hyperparameters.
- Performance gains on homophilic graphs (Cora, Citeseer, Pubmed) are relatively modest.

## Related Work & Insights

NSD employs static or parameterized sheaf structures but lacks spectral control and generalization guarantees; Bodnar et al. optimize the spectral gap via path alignment but without PAC-Bayes calibration; existing PAC-Bayes bounds for GNNs are loose due to the neglect of spectral information. SGPC is the first work to unify OT-based sheaf learning, spectral gap optimization, and PAC-Bayes risk control into a single framework.

## Related Work & Insights

- The paradigm of PAC-Bayes + spectral optimization is generalizable to other probabilistic models defined over graphs.
- The idea of learning sheaf restriction maps via Wasserstein Lift may inspire methods in manifold learning.
- The Chebyshev polynomial frequency mixing can be viewed as analogous to adaptive filter banks in signal processing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified OT + Sheaf + PAC-Bayes framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine datasets covering both homophilic and heterophilic settings, with complete theoretical derivations.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured, though the density of mathematical notation is high.
- Value: ⭐⭐⭐⭐⭐ Provides the first spectrally-aware generalization guarantee framework for graph learning.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)
- [\[AAAI 2026\] Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks](kernelized_edge_attention_addressing_semantic_attention_blurring_in_temporal_gra.md)
- [\[AAAI 2026\] Magnitude-Modulated Equivariant Adapter for Parameter-Efficient Fine-Tuning of Equivariant Graph Neural Networks](magnitude-modulated_equivariant_adapter_for_parameter-efficient_fine-tuning_of_e.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)

</div>

<!-- RELATED:END -->
