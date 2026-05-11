---
title: >-
  [Paper Note] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation
description: >-
  [ICLR 2026][Image Generation][Graph generation] This paper proposes HOG-Diff, a graph diffusion framework that leverages higher-order topological structures (e.g., rings, triangles…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Graph generation"
  - "diffusion models"
  - "higher-order topology"
  - "cell complexes"
  - "diffusion bridges"
date: 2026-05-08
content_hash: 7416217c7e778c44
---

# HOG-Diff: Higher-Order Guided Diffusion for Graph Generation

**Conference**: ICLR 2026
**arXiv**: [2502.04308](https://arxiv.org/abs/2502.04308)
**Code**: Unavailable
**Area**: Graph Generation
**Keywords**: Graph generation, diffusion models, higher-order topology, cell complexes, diffusion bridges

## TL;DR

This paper proposes HOG-Diff, a graph diffusion framework that leverages higher-order topological structures (e.g., rings, triangles, motifs) as generative guidance. By extracting higher-order skeletons via Cell Complex Filtration (CCF) and combining them with a generalized OU diffusion bridge, the framework realizes coarse-to-fine progressive graph generation, achieving state-of-the-art performance on 8 benchmarks for both molecular and general graph generation.

## Background & Motivation

1. **Existing graph diffusion models neglect higher-order topology**: Mainstream graph generation methods (e.g., GDSS, DiGress, DeFoG) operate directly on adjacency matrices or at the edge level, treating graphs as collections of pairwise edges and entirely ignoring higher-order structures such as triangles, rings, and cliques. Yet these structures are critical in chemical molecules (e.g., benzene rings, heterocycles) and biological networks.

2. **Intermediate states degrade into unstructured noise**: The forward process of classical diffusion models progressively corrupts data into a Gaussian distribution, leaving intermediate states as meaningless noisy adjacency matrices that neither preserve topological properties nor provide useful structural guidance.

3. **Importance of ring systems in molecular generation**: Approved drug molecules contain only a few hundred distinct ring systems, far fewer than the astronomically large chemical space ($10^{23}$–$10^{60}$), indicating that higher-order topological structures are core constraints of real molecular distributions.

4. **Inspiration from topological deep learning**: TDL research has demonstrated that explicitly modeling complex topological structures (simplicial complexes, cell complexes) enhances the expressiveness and stability of graph representation learning, yet this insight has not been introduced into generative models.

5. **Limitations of adjacency-domain diffusion**: Injecting Gaussian noise directly into adjacency matrices suffers from permutation ambiguity, signal degradation due to sparsity, and poor scalability. Spectral-domain diffusion offers a more robust alternative.

6. **Lack of topology-aware quality evaluation**: Existing metrics (e.g., Validity, FCD) primarily assess chemical validity and distributional distance, rarely measuring the topological preservation ability of generated graphs. Evaluation methods grounded in TDA, such as curvature filtration, are needed.

## Method

### Overall Architecture: Coarse-to-Fine Topology-Guided Generation

The core idea of HOG-Diff is to decompose graph generation into hierarchical sub-tasks:

1. **Cell complex lifting**: The original graph $\bm{G}$ is lifted to a cell complex $\mathcal{S}$ by attaching 2-dimensional closed disks along simple cycles to construct 2-cells.
2. **Cell Complex Filtration (CCF)**: A $p$-cell filtration operation is defined to extract nodes and edges belonging exclusively to higher-order cells, yielding a "higher-order skeleton" $\bm{G}_{[p]}$.
3. **Hierarchical diffusion**: The diffusion process is partitioned into $K$ hierarchical time windows $\{[\tau_{k-1}, \tau_k]\}$, with the higher-order skeleton serving as the intermediate target state.

The joint distribution of the generation process is factorized as:

$$p(\bm{G}_0) = p(\bm{G}_0|\bm{G}_{\tau_1}) \cdot p(\bm{G}_{\tau_1}|\bm{G}_{\tau_2}) \cdots p(\bm{G}_{\tau_{K-1}}|\bm{G}_T)$$

### Generalized OU Diffusion Bridge

Within each time window, a generalized Ornstein-Uhlenbeck (GOU) bridge process connects adjacent intermediate states. The GOU process is governed by the following SDE:

$$\mathrm{d}\bm{G}_t = \theta_t(\bm{\mu} - \bm{G}_t)\mathrm{d}t + g_t\mathrm{d}\bm{W}_t$$

A terminal constraint $\bm{\mu} = \bm{G}_{\tau_k}$ is imposed via Doob's $h$-transform, yielding a closed-form transition probability for the bridge process and enabling simulation-free training.

Key advantages:
- The Brownian bridge is a special case of the GOU bridge ($\theta_t \to 0$).
- The closed-form conditional probability $p(\bm{G}_t|\bm{G}_{\tau_{k-1}}, \bm{G}_{\tau_k})$ supports one-step forward sampling.
- Terminal-state variance is zero, ensuring smooth transitions to predefined intermediate structures.

### Spectral-Domain Diffusion

Diffusion is performed in the spectral domain of the graph Laplacian $\bm{L} = \bm{D} - \bm{A}$ rather than directly on the adjacency matrix. Following the decomposition $\bm{L} = \bm{U}\bm{\Lambda}\bm{U}^\top$, separate diffusion processes are established for the eigenvalues $\bm{\Lambda}$ and node features $\bm{X}$:

- Permutation ambiguity is resolved by exploiting the permutation invariance of the Laplacian spectrum.
- Eigenvalue diffusion captures global topological structure.
- Node feature diffusion handles local attributes.

### Score Network

A dual-module architecture is adopted: a GCN for local feature aggregation combined with a Graph Transformer for global information extraction. Time information is fused via FiLM layers, and separate MLPs predict the score functions for nodes and the spectrum. The entire network is permutation equivariant.

## Key Experimental Results

### Molecular Generation (QM9, ZINC250k, MOSES, GuacaMol)

| Method | QM9 FCD↓ | QM9 NSPDK↓ | ZINC FCD↓ | ZINC NSPDK↓ | MOSES Val.↑ | GuacaMol FCD↑ |
|--------|----------|------------|-----------|-------------|-------------|--------------|
| GDSS | 2.900 | 0.003 | 14.656 | 0.019 | — | — |
| DiGress | 0.360 | 0.0005 | 23.060 | 0.082 | 85.7 | 68.0 |
| Cometh | 0.248 | 0.0005 | — | — | 90.5 | 72.7 |
| DeFoG | 0.268 | 0.0005 | 2.030 | 0.002 | 92.8 | 73.8 |
| **HOG-Diff** | **0.172** | **0.0003** | **1.633** | **0.001** | **99.7** | **78.5** |

HOG-Diff substantially outperforms competing methods on FCD and NSPDK, indicating that generated molecules are closer to the real distribution in both chemical and graph spaces. On the large-scale MOSES dataset, Validity reaches 99.7%, far surpassing all other methods.

### General Graph Generation + Topological Preservation Analysis

| Method | Community-small Avg.↓ | Enzymes Avg.↓ | QM9 $\kappa_{FR}$↓ | ZINC $\kappa_{FR}$↓ |
|--------|-----------------------|---------------|---------------------|---------------------|
| GDSS | 0.046 | 0.032 | 0.925 | 1.781 |
| DiGress | 0.038 | 0.030 | 0.251 | — |
| DeFoG | — | — | 0.177 | 0.728 |
| **HOG-Diff** | **0.010** | **0.027** | **0.077** | **0.190** |

Under topology-aware evaluation based on Curvature Filtration, HOG-Diff achieves the lowest distance scores across all datasets, with particularly pronounced advantages on complex molecular datasets (the $\kappa_{FR}$ score on QM9 is 56% lower than the second-best method).

### Ablation Study: Importance of Topological Guidance

| Guidance Type | QM9 Val.↑ | QM9 FCD↓ | QM9 NSPDK↓ |
|---------------|-----------|----------|------------|
| Noise (classical diffusion) | 91.52 | 0.829 | 0.0015 |
| Peripheral (peripheral structure) | 97.58 | 0.305 | 0.0009 |
| **Cell (higher-order skeleton)** | **98.74** | **0.172** | **0.0003** |

Higher-order skeleton guidance substantially outperforms both noise-based guidance (classical diffusion) and peripheral structure guidance, validating the effectiveness of higher-order topology as a generative signal. Training curves further confirm that HOG-Diff converges faster than classical methods, consistent with Theorem 3.

## Highlights & Insights

- **First explicit use of higher-order topology as a guidance signal in graph generation**: Unlike prior work that treats topology as a post-hoc evaluation metric, HOG-Diff embeds it at the core of the generative process.
- **Elegant design of Cell Complex Filtration**: The CCF operation avoids costly exhaustive lifting enumeration and efficiently extracts higher-order skeletons as intermediate targets.
- **Unification of theory and practice**: The paper proves that HOG-Diff strictly outperforms classical diffusion models in terms of score matching convergence rate and reconstruction error bounds (Theorems 3 & 4), corroborated by experiments.
- **Comprehensive evaluation framework**: Covers 8 benchmarks, 4 molecular datasets, and introduces Curvature Filtration as a topological preservation metric.

## Limitations & Future Work

- CCF depends on the quality of the lifting operation; different lifting strategies (2-cell vs. simplicial complex) may perform differently across datasets, requiring manual selection.
- The advantages are less pronounced on datasets with sparse higher-order structures (e.g., Ego-small), suggesting that the method's gains are strongly correlated with the topological richness of the data.
- Spectral-domain diffusion requires selecting a fixed eigenvector basis $\hat{\bm{U}}_0$ sampled from the training set, introducing additional sampling bias.
- The two-stage approach ($K=2$) introduces additional hyperparameters (time window partition $\tau_k$, coefficients $c_1, c_2$), imposing a greater tuning burden than classical methods.
- The effects of higher-dimensional (3-cell and above) topological guidance remain unexplored.

## Related Work & Insights

| Dimension | HOG-Diff | DiGress (Vignac et al. 2023) | DeFoG (Qin et al. 2025) |
|-----------|----------|------------------------------|------------------------|
| Diffusion domain | Spectral (Laplacian eigenvalues) | Discrete space (categorical) | Flow matching |
| Higher-order topology | Explicit guidance (CCF) | None | None |
| Intermediate states | Meaningful topological skeleton | Noise | Optimal transport path |
| Bridge process | GOU bridge (closed-form) | None | None |
| Theoretical guarantees | Convergence rate + error bounds | None | None |
| MOSES Val. | 99.7% | 85.7% | 92.8% |

| Dimension | HOG-Diff | GDSS (Jo et al. 2022) | MiCaM (Geng et al. 2023) |
|-----------|----------|----------------------|--------------------------|
| Generation paradigm | One-shot (spectral diffusion) | One-shot (adjacency matrix diffusion) | Autoregressive (motif merging) |
| Higher-order information | Generative guidance | None | Implicit (motif vocabulary) |
| Topological preservation | Excellent (lowest $\kappa_{FR}$) | Poor (high $\kappa_{FR}$) | Moderate |
| Scalability | Large-scale (MOSES/GuacaMol) | Moderate | Limited by motif vocabulary |
| Interpretability | Analyzable impact of different guidance types | None | Limited |

## Rating

- ⭐⭐⭐⭐⭐ **Novelty**: First systematic integration of higher-order topology into a graph diffusion framework; the CCF + GOU bridge combination is elegant and theoretically grounded.
- ⭐⭐⭐⭐⭐ **Experimental Thoroughness**: 8 benchmarks, comprehensive baseline comparisons, topological evaluation, ablation studies, and theoretical validation provide thorough coverage.
- ⭐⭐⭐⭐ **Writing Quality**: Overall structure is clear and mathematical derivations are rigorous, though notation is dense and the preliminaries section is lengthy.
- ⭐⭐⭐⭐ **Value**: Directly applicable to molecular generation and drug discovery; the topology-guided generation paradigm is generalizable to broader scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] GGBall: Graph Generative Model on Poincaré Ball](ggball_graph_generative_model_on_poincaré_ball.md)
- [\[ICLR 2026\] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training](generate_any_scene_scene_graph_driven_data_synthesis_for_visual_generation_train.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/image_generation/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)

</div>

<!-- RELATED:END -->
