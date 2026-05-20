---
title: >-
  [Paper Note] Structure-Centric Graph Foundation Model via Geometric Bases
description: >-
  [ICML 2026][Graph Learning][Structure-centric GFM] SCGFM reframes cross-domain graph foundation modeling as a "triangulation" problem in metric measure spaces: it learns a set of $K$ trainable geometric bases $\{B_k\}$…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Structure-centric GFM"
  - "geometric bases"
  - "Sliced GW"
  - "structural coordinates"
  - "feature re-encoding"
date: 2026-05-08
content_hash: d9f4fa4ce74d8fc3
---

# Structure-Centric Graph Foundation Model via Geometric Bases

**Conference**: ICML 2026  
**arXiv**: [2605.08689](https://arxiv.org/abs/2605.08689)  
**Code**: https://github.com/Xd-He/SCGFM  
**Area**: Graph Foundation Models / Cross-domain Transfer / Gromov–Wasserstein / Metric Geometry  
**Keywords**: Structure-centric GFM, geometric bases, Sliced GW, structural coordinates, feature re-encoding

## TL;DR
SCGFM reframes cross-domain graph foundation modeling as a "triangulation" problem in metric measure spaces: it learns a set of $K$ trainable geometric bases $\{B_k\}$, represents each graph by the softmax of its Gromov–Wasserstein distances $\delta_k$ to these bases to obtain a set of structural coordinates $\mathbf{w}$, and uses the OT plan on the bases to aggregate node features into a unified dimension. This approach eliminates the traditional GFM constraint of "must align node feature spaces," and outperforms baselines in both in-domain and OOD few-shot graph/node classification.

## Background & Motivation

**Background**: The two mainstream GFM approaches are (a) injecting language priors via LLM/prompt, and (b) pretraining GNNs on large graph corpora with contrastive/generative objectives. Both assume node feature spaces can be aligned across datasets—typically via padding, dimensionality reduction, or dataset-specific adapters. Such "feature alignment" works when source/target distributions are similar, but almost always fails in cross-domain scenarios.

**Limitations of Prior Work**: (i) Existing GFMs enforce fixed node feature dimensions (e.g., OFA, BRIDGE), projecting features from different datasets into the same space and losing essential structural differences; (ii) Graph tokenization (discretizing graphs into tokens as words) violates permutation invariance and imposes artificial node order; (iii) There is no "shared geometric reference frame"—graphs are non-Euclidean, permutation-invariant relational objects, unlike images that can be pixel-aligned.

**Key Challenge**: Transferable knowledge in graphs lies in **structure (topology)** rather than **features**, but current GFMs focus alignment on features, causing structural information to be compressed/distorted in cross-domain settings; explicit GW barycenter computation requires nested OT optimization, which is theoretically elegant but practically infeasible.

**Goal**: To establish a **structure-centric** unified representation space such that (1) arbitrary graphs can be encoded without relying on fixed feature dimensions; (2) graphs with heterogeneous topologies can be mapped to the same continuous coordinate system; (3) strong transfer is achieved in both few-shot in-domain and OOD cross-domain settings.

**Key Insight**: View graphs as metric measure (mm-) spaces—each graph is $(\mathcal{V},d_G,\mu_G)$, independent of node identity; leveraging Gromov's compactness theorem, assume real graphs lie in a bounded subset $\mathcal{K}\subset\mathcal{X}$ of mm-space, so a finite $\epsilon$-cover exists. Learning this cover yields a "geometric basis dictionary."

**Core Idea**: Reformulate graph representation learning as "triangulation with $K$ trainable prototypes under GW distance," where each graph is represented by the softmax of its GW distance vector to all bases, plus node features reprojected via the OT plan, forming a unified embedding.

## Method

### Overall Architecture
Two stages. **Pre-training**: Jointly optimize $K$ geometric bases $B_k=([M],d_k,\mu_k)$ (mm-spaces with $M$ nodes, $\mathbf{B}_k\in[0,1]^{M\times M}$ symmetric, zero diagonal, $\mu_k$ set as uniform) on multi-source graphs. Use Sliced Gromov–Wasserstein (SGW) to map each source graph to structural coordinates $\mathbf{w}$, minimizing a joint loss of structural reconstruction, statistical reconstruction, and diversity regularization. **Downstream Projection**: Freeze the pretrained bases, compute $\mathbf{w}$ for target graphs, and use the OT plan $\mathbf{T}_{ik}\in\mathbb{R}^{N\times M}$ from GW computation to aggregate node features onto the $M$ basis nodes, yielding $\mathbf{H}(G_i)\in\mathbb{R}^{M\times F}$. The final embedding $\mathbf{z}(G_i)=[\mathbf{w}\|f(\mathbf{w})\|\mathrm{vec}(\mathbf{H}(G_i))]\in\mathbb{R}^{K+r+MF}$ is fed to the downstream classifier.

### Key Designs

1. **Learnable Geometric Bases & Structural Coordinates**:

    - **Function**: Construct a shared "graph coordinate reference frame" so any graph can be represented as its GW distance vector to the bases.
    - **Mechanism**: Each basis $B_k$ is parameterized by a symmetric, zero-diagonal $[0,1]^{M\times M}$ matrix $\mathbf{B}_k$ (not requiring triangle inequality; pseudo-metric suffices as GW kernel), with measure $\mu_k$ fixed as uniform to reduce degrees of freedom. For input graph $G_i$, compute $\delta_k=d_{GW}(\mathbf{A}_i,\mathbf{B}_k)$ (SGW reduces complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(N\log N)$), then softmax to obtain structural coordinates $w_k=\exp(-\delta_k/\tau)/\sum_j\exp(-\delta_j/\tau)$. Theorem 3.2 proves $\|\mathbf{w}-\mathbf{w}'\|_2\le L_w\eta$, i.e., coordinates are Lipschitz continuous with respect to GW distance, ensuring structurally similar graphs have close coordinates.
    - **Design Motivation**: Directly learning GW barycenters requires nested OT, which is infeasible; using "distance vectors to a set of bases" as a substitute is an elegant instance of dictionary learning in metric geometry. SGW's 1D projection reduces infeasible $\mathcal{O}(N^3)$ to quasi-linear, enabling scalability.

2. **Linear Proxy GW Barycenter & Multi-objective Reconstruction**:

    - **Function**: Avoid nested optimization of true GW barycenter, while ensuring $\mathbf{w}$ reconstructs the original graph at both structural and statistical levels.
    - **Mechanism**: Use the linear combination $\widetilde{\mathbf{B}}(G)=\sum_k w_k \mathbf{B}_k$ as a finite basis expansion proxy for the barycenter, yielding structural reconstruction loss $\mathcal{L}_{gw}=\mathbb{E}_G[d_{GW}(\mathbf{A},\widetilde{\mathbf{B}}(G))]$; an MLP decoder $f(\mathbf{w})$ predicts degree histogram, clustering coefficient histogram, and log-scaled motif counts (triangles, short cycles), supervised by MSE $\mathcal{L}_{rec}=\mathrm{MSE}(\mathrm{FE}(G),f(\mathbf{w}))$; Corollary 3.3 guarantees $\|f(\mathbf{w})-f(\mathbf{w}')\|_2\le L_f L_w \eta$, so statistical reconstruction is also Lipschitz with respect to GW distance.
    - **Design Motivation**: The linear proxy is not a strict mm-space barycenter, but as a dictionary expansion it is differentiable and naturally compatible with softmax coordinates; multi-objective reconstruction ensures "coordinates can recover both adjacency and coarse statistics," mitigating OT's inherent non-uniqueness.

3. **Diversity Regularization & Structure-aware Feature Re-encoding**:

    - **Function**: Prevent the $K$ geometric bases from collapsing to a single prototype; unify node features $\mathbf{X}_i\in\mathbb{R}^{N\times F}$ of heterogeneous dimensions into the shared coordinate system $\mathbb{R}^{M\times F}$.
    - **Mechanism**: (a) Diversity loss $\mathcal{L}_{div}=\frac{1}{|\mathcal{P}|}\sum_{(i,j)}\max(0,m-\|\mathbf{B}_i-\mathbf{B}_j\|_F)$ enforces a minimum Frobenius distance $m$ between bases, ensuring the bases span the structural space; (b) Feature re-encoding uses the OT plan $\mathbf{T}_{ik}$ from GW to sum-aggregate node features onto basis nodes $\mathbf{H}_k=N\cdot\mathbf{T}_{ik}^\top\mathbf{X}_i$ (multiplying by $N$ offsets the averaging effect from $\mathbf{T}$ as a normalized measure, preserving multiset injectivity), then weights by $\mathbf{w}$ to obtain $\mathbf{H}(G_i)=\sum_k w_k \mathbf{H}_k$. The total objective is $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$.
    - **Design Motivation**: Basis collapse is a classic failure mode in prototype/dictionary models; hinge-style diversity regularization is simple and effective. Using the OT plan instead of padding/MLP for feature alignment ensures "structural similarity → features are aggregated along structural neighborhoods," maintaining feature flexibility without losing semantic correspondence.

### Loss & Training
Pre-training is performed only on source graphs with $\mathcal{L}_{total}=\mathcal{L}_{gw}+\alpha\mathcal{L}_{rec}+\beta\mathcal{L}_{div}$, all GW computations approximated by SGW; downstream, the bases and $f(\cdot)$ are frozen, and only the classifier head is trained. Few-shot (5-shot) evaluation is repeated 50 times, reporting mean ± standard deviation.

## Key Experimental Results

### Main Results
5-shot graph classification (excerpted from Table 1 in the paper, showing representative in-domain and OOD values):

| Training | Test | GCN | GIN | GraphCL | SCGFM (Best in Paper) |
|---|---|---|---|---|---|
| in-domain | COX2 | 49.84 | 54.31 | 54.68 | Best, bolded in Table 1 |
| in-domain | NCI1 | 51.85 | 52.95 | 57.22 | Best |
| in-domain | BZR | 54.41 | 51.29 | 60.28 | Best |
| S1→COL-3 (OOD) | COL-3 | 9.53 | 9.25 | — | Significant improvement |
| S2→COX2 (OOD) | COX2 | 50.33 | 55.16 | — | Comparable or higher |
| Avg. | — | 43.23 | 44.85 | — | Highest |

(Note: For complete raw values, see Table 1 in the original paper. This note lists representative slices to highlight dual in-domain/OOD advantages.)

### Ablation Study

| Configuration | Key Change | Conclusion |
|---|---|---|
| Full SCGFM | mean highest | Complete model |
| w/o geometric bases (directly use structural features) | large cross-domain degradation | Structural coordinates are core to transfer |
| w/o $\mathcal{L}_{rec}$ | in-domain still works, OOD degrades | Statistical reconstruction provides inductive bias for cross-domain stability |
| w/o $\mathcal{L}_{div}$ | basis collapse, effective dimension drops | Diversity regularization is essential |
| Replace SGW with exact GW | same accuracy but memory explosion | Confirms scalability benefit of SGW |
| Change basis number $K$ | moderate $K$ optimal | Too few underfits, too many redundant |

### Key Findings
- Learned geometric bases exhibit interpretable topological patterns (chains, stars, dense clusters, etc.), supporting the theoretical assumption that "geometric bases approximate an $\epsilon$-cover."
- Feature distributions drift significantly in cross-domain settings, but structural coordinates $\mathbf{w}$ remain nearly unchanged and can be directly transferred; the OT plan naturally adapts features to new dimensions.
- SGW reduces training time and memory to quasi-linear, enabling scalability to million-node graphs.

## Highlights & Insights
- **Unifying graph representation from the mm-space perspective**: Reinterprets the long-standing "how to align graph foundation models" problem as $\epsilon$-covering in metric geometry, a reconstruction that can be immediately applied to point clouds, 3D shapes, and other non-Euclidean objects.
- **Structural coordinates + Lipschitz stability theorem**: Provides a rare "geometric consistency" guarantee in representation learning—structurally similar graphs must have similar coordinates, making it more trustworthy than contrastive GFM without theoretical backing.
- **OT projection for feature re-encoding**: Transforms "feature alignment" from rigid padding to "transport along structural neighborhoods," preserving multiset injectivity and serving as an elegant template for GFMs handling heterogeneous feature spaces.
- **SGW enables scalability**: Reducing $\mathcal{O}(N^3)$ GW to $\mathcal{O}(N\log N)$ is key to scaling, demonstrating that sliced OT is also a practical tool in the graph domain.

## Limitations & Future Work
- The linear proxy barycenter is an engineering compromise and cannot guarantee optimality for graph families with extremely non-Gaussian distributions; future work may explore "nonlinear GW barycenter approximations."
- The number of bases $K$ and nodes $M$ are fixed hyperparameters, not adaptive; data-driven selection based on covering-number bounds could be explored.
- Only few-shot classification tasks are validated; coverage of node-level and link-level tasks is limited.
- Real-world graphs often have edge features/timestamps; this work's mm-space considers only node measures and adjacency, so extending to richer graph structures requires redesigning basis parameterization.

## Related Work & Insights
- **vs OFA / BRIDGE / SAMGPT**: These enforce fixed feature dimensions and are limited by feature alignment in cross-domain settings; this work aligns via structural coordinates, allowing variable feature dimensions.
- **vs Graph tokenization (GIT, etc.)**: These break permutation invariance, while this work leverages mm-space's natural permutation invariance.
- **vs FGW / GW-coarsening (Vayer 2019, Chen 2023)**: These use GW for "pairwise comparison or coarsening," while this work uses a learned set of bases to create a "unified coordinate system."
- **vs prototype/dictionary learning methods**: These select prototypes in feature space, while this work selects structural prototypes in mm-space, better matching the essence of graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses mm-space + learnable geometric bases to reconstruct GFM, independent approach with theoretical support
- Experimental Thoroughness: ⭐⭐⭐⭐ in-domain + two OOD settings + multiple ablations, lacks larger-scale/node-level transfer evaluation
- Writing Quality: ⭐⭐⭐⭐ Geometric motivation is clearly explained, theorems and algorithm steps are highly readable
- Value: ⭐⭐⭐⭐ Provides a new "structure-centric, feature-flexible" paradigm for cross-domain transfer in GFM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)
- [\[AAAI 2026\] Feature-Centric Unsupervised Node Representation Learning Without Homophily Assumption](../../AAAI2026/graph_learning/feature-centric_unsupervised_node_representation_learning_without_homophily_assu.md)

</div>

<!-- RELATED:END -->
