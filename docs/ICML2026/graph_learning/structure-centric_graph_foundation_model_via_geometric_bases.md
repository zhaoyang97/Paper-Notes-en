---
title: >-
  [Paper Note] Structure-Centric Graph Foundation Model via Geometric Bases
description: >-
  [ICML 2026][Graph Learning][Structure-Centric GFM] SCGFM reformulates the cross-domain Graph Foundation Model (GFM) as a "triangulation" problem in metric measure spaces. It learns a set of $K$ trainable geometric bases…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Structure-Centric GFM"
  - "Geometric Bases"
  - "Sliced GW"
  - "Structural Coordinates"
  - "Feature Re-encoding"
date: 2026-05-08
content_hash: fb02e57fe7d43987
---

# Structure-Centric Graph Foundation Model via Geometric Bases

**Conference**: ICML 2026  
**arXiv**: [2605.08689](https://arxiv.org/abs/2605.08689)  
**Code**: https://github.com/Xd-He/SCGFM  
**Area**: Graph Foundation Models / Cross-domain Transfer / Gromov–Wasserstein / Metric Geometry  
**Keywords**: Structure-Centric GFM, Geometric Bases, Sliced GW, Structural Coordinates, Feature Re-encoding

## TL;DR
SCGFM reformulates the cross-domain Graph Foundation Model (GFM) as a "triangulation" problem in metric measure spaces. It learns a set of $K$ trainable geometric bases $\{B_k\}$, where each graph is represented by structural coordinates $\mathbf{w}$ (obtained via softmax of Gromov–Wasserstein distances $\delta_k$ to each base). By utilizing the OT plan from the bases to aggregate node features into a unified dimension, SCGFM bypasses the traditional GFM constraint of "mandatory node feature space alignment," outperforming baselines in both in-domain and OOD few-shot graph/node classification.

## Background & Motivation

**Background**: Two mainstream paths for GFMs are (a) injecting linguistic priors using LLMs/prompts and (b) pre-training GNNs on large graph corpora with contrastive/generative objectives. Both assume that node feature spaces can be aligned across datasets—typically achieved through padding, dimensionality reduction, or dataset-specific adapters. This "feature alignment" works when source and target distributions are similar but often fails during cross-domain transfer.

**Limitations of Prior Work**: (i) Existing GFMs force a fixed node feature dimension (e.g., OFA, BRIDGE), projecting features from different datasets into the same space and losing intrinsic structural differences; (ii) Graph tokenization schemes (treating graphs as sequences of tokens) violate graph permutation invariance by imposing an artificial node order; (iii) There is no "shared geometric reference frame"—graphs are non-Euclidean, permutation-invariant relational objects that cannot be aligned pixel-by-pixel like images.

**Key Challenge**: Transferable knowledge in graphs resides in the **structure (topology)** rather than the **features**. However, current GFMs focus alignment on features, causing structural information to be compressed or distorted cross-domain. Additionally, explicit Gromov–Wasserstein (GW) barycenter computation involves nested OT optimization, which is theoretically elegant but computationally intractable.

**Goal**: Establish a **structure-centric** unified representation space that (1) encodes arbitrary graphs without relying on fixed feature dimensions; (2) maps graphs with heterogeneous topologies to a shared continuous coordinate system; and (3) enables robust transfer in few-shot in-domain and OOD cross-domain settings.

**Key Insight**: View graphs through the lens of metric measure (mm-) spaces, where each graph is represented as $(\mathcal{V},d_G,\mu_G)$, independent of node identity. By invoking the Gromov Compactness Theorem, it is assumed that real-world graphs lie within a bounded subset $\mathcal{K}\subset\mathcal{X}$ of the mm-space, implying the existence of a finite $\epsilon$-cover. Learning this cover yields a "dictionary of geometric bases."

**Core Idea**: Rewrite graph representation learning as "triangulation under the GW distance using $K$ trainable prototypes." Each graph is represented by structural coordinates $\mathbf{w}$ (the softmax of its GW distances to all bases) combined with node features re-projected via the OT plan into a unified embedding.

## Method

### Overall Architecture
The method consists of two stages. **Pre-training**: $K$ geometric bases $B_k=([M],d_k,\mu_k)$ (mm-spaces with $M$ nodes, represented by symmetric matrices $\mathbf{B}_k\in[0,1]^{M\times M}$ without diagonals and uniform $\mu_k$) are jointly optimized across multi-source domain graphs. Sliced Gromov–Wasserstein (SGW) maps each source graph to structural coordinates $\mathbf{w}$, minimizing a joint loss of structural reconstruction, statistical reconstruction, and diversity regularization. **Downstream Projection**: With frozen pre-trained bases, structural coordinates $\mathbf{w}$ are calculated for target graphs. The OT plan $\mathbf{T}_{ik}\in\mathbb{R}^{N\times M}$ from the GW computation projects node features onto the $M$ base nodes to obtain $\mathbf{H}(G_i)\in\mathbb{R}^{M\times F}$. The final embedding $\mathbf{z}(G_i)=[\mathbf{w}\|f(\mathbf{w})\|\mathrm{vec}(\mathbf{H}(G_i))]\in\mathbb{R}^{K+r+MF}$ is fed to the downstream classifier.

### Key Designs

1. **Geometric Bases & Structural Coordinates**:

    - **Function**: Constructs a shared "graph coordinate reference frame," allowing any graph to be represented as a vector of GW distances to these bases.
    - **Mechanism**: Each base $B_k$ is parameterized by a symmetric matrix $\mathbf{B}_k \in [0,1]^{M\times M}$ (pseudo-metrics suffice as GW kernels). The measure $\mu_k$ is fixed as uniform. For an input graph $G_i$, $\delta_k=d_{GW}(\mathbf{A}_i,\mathbf{B}_k)$ is computed using SGW (reducing complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(N\log N)$). Structural coordinates are derived as $w_k=\exp(-\delta_k/\tau)/\sum_j\exp(-\delta_j/\tau)$. Theorem 3.2 proves $\|\mathbf{w}-\mathbf{w}'\|_2\le L_w\eta$, ensuring Lipschitz continuity of coordinates relative to GW distance.
    - **Design Motivation**: Direct GW barycenter learning is intractable. Using "distance vectors to a set of bases" as a surrogate for an explicit barycenter is an elegant application of dictionary learning in metric geometry. SGW's 1D projection makes the $\mathcal{O}(N^3)$ complexity quasi-linear.

2. **Linear Proxy GW Barycenter + Multi-objective Reconstruction**:

    - **Function**: Avoids nested optimization of true GW barycenters while ensuring $\mathbf{w}$ "reconstructs" the original graph's structure and statistics.
    - **Mechanism**: Use the linear combination $\widetilde{\mathbf{B}}(G)=\sum_k w_k \mathbf{B}_k$ as a finite basis expansion of the barycenter to compute structural reconstruction loss $\mathcal{L}_{gw}=\mathbb{E}_G[d_{GW}(\mathbf{A},\widetilde{\mathbf{B}}(G))]$. An MLP decoder $f(\mathbf{w})$ predicts degree histograms, clustering coefficient histograms, and log-scaled motif counts (triangles, short cycles) via $\mathcal{L}_{rec}=\mathrm{MSE}(\mathrm{FE}(G),f(\mathbf{w}))$. Corollary 3.3 ensures statistical reconstruction is also Lipschitz stable.
    - **Design Motivation**: While the linear proxy is not a strict mm-space barycenter, it allows gradient flow and is compatible with softmax coordinates. Multi-objective reconstruction mitigates the non-uniqueness inherent in OT.

3. **Diversity Regularization + Structure-Aware Feature Re-encoding**:

    - **Function**: Prevents the $K$ geometric bases from collapsing and projects heterogeneous node features $\mathbf{X}_i\in\mathbb{R}^{N\times F}$ into a shared $\mathbb{R}^{M\times F}$ space.
    - **Mechanism**: (a) Diversity loss $\mathcal{L}_{div}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)}\max(0,m-\|\mathbf{B}_i-\mathbf{B}_j\|_F)$ enforces a minimum Frobenius distance $m$ between bases. (b) Feature re-encoding uses the OT plan $\mathbf{T}_{ik}$ to aggregate node features onto bases: $\mathbf{H}_k=N\cdot\mathbf{T}_{ik}^\top\mathbf{X}_i$ (preserving multiset injectivity), followed by weighted summation $\mathbf{H}(G_i)=\sum_k w_k \mathbf{H}_k$. Total loss: $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$.
    - **Design Motivation**: Diversity regularization prevents prototype collapse. Using the OT plan instead of padding/MLPs for alignment ensures "structural similarity leads to similar feature aggregation."

### Loss & Training
Pre-training optimizes $\mathcal{L}_{total}$ on source domain graphs using SGW approximations. For downstream tasks, bases and $f(\cdot)$ are frozen, and only the classification head is trained. Few-shot (5-shot) evaluation is averaged over 50 runs.

## Key Experimental Results

### Main Results
5-shot graph classification (Selected from Paper Table 1):

| Training | Testing | GCN | GIN | GraphCL | SCGFM (Ours) |
|---|---|---|---|---|---|
| in-domain | COX2 | 49.84 | 54.31 | 54.68 | **Best** |
| in-domain | NCI1 | 51.85 | 52.95 | 57.22 | **Best** |
| in-domain | BZR | 54.41 | 51.29 | 60.28 | **Best** |
| S1→COL-3 (OOD) | COL-3 | 9.53 | 9.25 | — | **Large Gain** |
| S2→COX2 (OOD) | COX2 | 50.33 | 55.16 | — | **Superior** |
| Avg. | — | 43.23 | 44.85 | — | **Highest** |

### Ablation Study

| Configuration | Critical Change | Conclusion |
|---|---|---|
| Full SCGFM | Highest mean | Validates complete model |
| w/o Geometric Bases | Significant OOD degradation | Structural coordinates core for transfer |
| w/o $\mathcal{L}_{rec}$ | OOD degradation | Statistical reconstruction provides inductive bias |
| w/o $\mathcal{L}_{div}$ | Base collapse | Diversity regularization is essential |
| Exact GW instead of SGW | Same accuracy, high memory | SGW provides scalability gains |
| Varying $K$ | Moderate $K$ optimal | Trade-off between expressiveness and redundancy |

### Key Findings
- Learned geometric bases exhibit interpretable topological patterns (chains, stars, dense clusters), confirming the $\epsilon$-cover hypothesis.
- While feature distributions shift significantly cross-domain, structural coordinates $\mathbf{w}$ remains stable, enabling direct transfer.
- SGW reduces training time and memory to quasi-linear, allowing scalability to graphs with millions of nodes.

## Highlights & Insights
- **Unified Graph Representation via mm-space**: Interprets the GFM alignment problem as an $\epsilon$-covering in metric geometry, a framework extensible to point clouds and 3D shapes.
- **Structural Coordinates + Lipschitz Generalization**: Provides rare "geometric consistency" guarantees—structurally similar graphs must have similar coordinates, making it more reliable than heuristic contrastive GFMs.
- **OT-based Feature Re-encoding**: Transforms "feature alignment" from rigid padding into "transmission along structural neighborhoods," maintaining semantics while handling heterogeneous spaces.
- **SGW Scalability**: Proves that sliced OT is a highly practical tool in the graph domain, reducing $\mathcal{O}(N^3)$ complexity to $\mathcal{O}(N\log N)$.

## Limitations & Future Work
- The linear proxy barycenter is an engineering compromise; future work could explore "nonlinear GW barycenter approximations."
- Hyperparameters $K$ (number of bases) and $M$ (nodes per base) are fixed; data-driven selection based on covering-number bounds could be explored.
- Evaluation is predominantly on few-shot classification; node-level and link-level tasks require further coverage.
- Real-world graphs often include edge features/timestamps; the current mm-space model only considers node measures and adjacency.

## Related Work & Insights
- **vs OFA / BRIDGE / SAMGPT**: These models impose fixed feature dimensions and are limited by feature alignment cross-domain; ours aligns via structural coordinates with flexible feature dimensions.
- **vs Graph Tokenization (e.g., GIT)**: While tokenization destroys permutation invariance, the mm-space approach preserves it.
- **vs FGW / GW-coarsening**: Prior works use GW for pairwise comparison or coarsening; ours uses learned bases to establish a "unified coordinate system."
- **vs Prototype/Dictionary Learning**: While others select prototypes in feature space, we select structural prototypes in mm-space, aligning better with the intrinsic nature of graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformulates GFM with mm-spaces and geometric bases; solid theoretical grounding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong in-domain and OOD settings/ablations; could expand to larger-scale node-level transfer.
- Writing Quality: ⭐⭐⭐⭐ Clear geometric motivation and readable algorithmic steps.
- Value: ⭐⭐⭐⭐ Proposes a new "structure-centric, feature-flexible" paradigm for cross-domain GFM transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
