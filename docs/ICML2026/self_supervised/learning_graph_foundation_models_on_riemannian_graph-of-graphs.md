---
title: >-
  [Paper Note] Learning Graph Foundation Models on Riemannian Graph-of-Graphs
description: >-
  [ICML 2026][Self-Supervised Learning][Graph Foundation Model] R-GFM treats subgraphs of "different hop counts" as nodes in a higher-level Graph-of-Graphs (GoG). A dynamic MoE router distributes each GoG to Riemannian man…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Graph Foundation Model"
  - "Graph-of-Graphs"
  - "Riemannian MoE"
  - "adaptive-hop"
  - "Domain Generalization"
date: 2026-05-08
content_hash: a770c54eea9e43f9
---

# Learning Graph Foundation Models on Riemannian Graph-of-Graphs

**Conference**: ICML 2026  
**arXiv**: [2605.09993](https://arxiv.org/abs/2605.09993)  
**Code**: <https://github.com/USTC-DataDarknessLab/R-GFM>  
**Area**: Graph Foundation Models / Self-supervised Representation / Riemannian Geometry  
**Keywords**: Graph Foundation Model, Graph-of-Graphs, Riemannian MoE, adaptive-hop, Domain Generalization

## TL;DR
R-GFM treats subgraphs of "different hop counts" as nodes in a higher-level Graph-of-Graphs (GoG). A dynamic MoE router distributes each GoG to Riemannian manifolds (Hyperbolic / Euclidean / Spherical) that best match its curvature. This simultaneously addresses two inherent limitations of existing graph foundation models—fixed receptive fields and single Euclidean embeddings—yielding relative improvements of up to 49% in downstream tasks.

## Background & Motivation
**Background**: Graph Foundation Models (GFMs, such as OFA, Prodigy, and MDGFM) represent a breakthrough in pre-training on massive graphs to achieve cross-task and cross-domain transfer, marking the "foundation model era" of graph ML.

**Limitations of Prior Work**: (1) Existing GFMs use **fixed-hop subgraph sampling** (e.g., 1-hop or 2-hop neighbors) as the receptive field. However, downstream requirements vary significantly: homophilous citation networks require 1-2 hops, whereas e-commerce fraud detection often requires $\geq 4$ hops to uncover long-chain collusion. Fixed-hop schemes inevitably lead to underfitting or noise saturation. (2) Existing methods embed all subgraphs into a single Euclidean space, despite the vast structural differences across hops (local density vs. global hierarchical sparsity), which distorts representations.

**Key Challenge**: The conflict between fixed-structure receptive fields and heterogeneous hop requirements of downstream tasks, and the conflict between single geometry and multi-scale structural heterogeneity.

**Goal**: (1) Design a pre-training paradigm capable of adaptively capturing multi-hop structures; (2) Enable the model to dynamically switch between Riemannian manifolds within the representation space.

**Key Insight**: Elevate "subgraphs of different hops" to nodes within a Graph-of-Graphs (GoG), allowing the model to reason explicitly over scales; then use Mixture-of-Experts (MoE) to route each GoG to experts that match the geometric curvature.

**Core Idea**: **"Structural scale as a first-class citizen"**—adaptive-hop GoG handles scale mismatch, while confidence-aware dynamic Riemannian MoE addresses geometric mismatch.

## Method

### Overall Architecture
R-GFM is composed of four stages: (A) Deciding the Riemannian expert candidate set via the coefficient of variation (CV) of node degree distributions and sampling subgraphs $\{G_v^{(i)}\}_{i=1}^K$ for hops $1, 2, \ldots, K$ for each node $v$; (B) Using a contrastive pre-trained subgraph encoder to encode each subgraph into embedding $\mathbf{X}_{\text{sub}}$; (C) Constructing sparse GoGs $\mathcal{G}$ based on subgraph similarity and encoding them via dynamic MoE-based Riemannian routing; (D) Aggregating expert outputs into a fused embedding for downstream node classification or link prediction.

### Key Designs

1. **Adaptive-hop GoG Construction (Adaptive Hops + Similarity Sparsification)**:

    - **Function**: Avoids fixed receptive fields while controlling GPU memory consumption.
    - **Mechanism**: For each training node $v$, the hop count $K$ is incrementally increased using an "online greedy + memory test" approach, retreating to the last viable $K$ if an out-of-memory (OOM) error occurs, ensuring $K \leq \mathcal{B}_{\text{GPU}}$. Subgraph embeddings are pre-trained using NT-Xent contrastive learning. GoG edges are constructed by sampling $\mathcal{B}_{\text{edge}} = 0.6 \cdot K(K-1)/2$ edges without replacement from a distribution $\text{Prob}(i,j) = e^{\mathbf{S}[i,j]} / \sum_{u,v} e^{\mathbf{S}[u,v]}$ based on cosine similarity $\mathbf{S}$. Theoretically, Thm 3.2 proves that the noise in multi-hop embeddings $\|\boldsymbol\sigma_V\|_2 \leq \|\boldsymbol\sigma_F\|_2$ is strictly lower than fixed hops, and Thm 3.3 shows similarity-sparse GoGs yield smaller errors than "edgeless" or "fully-connected" GoGs.
    - **Design Motivation**: Fixed hops lose long-range information or drown local details. Dense GoGs introduce noise, while random sparse GoGs lack structural priors; similarity-based sparsification balances these factors.

2. **Dynamic MoE-based Riemannian Routing (Dual Dynamics: Candidates + Top-m)**:

    - **Function**: Adaptively determines the number of Riemannian experts and the activation frequency based on structural heterogeneity.
    - **Mechanism**: Structural heterogeneity of each training set is quantified using the coefficient of variation $\text{CV}(\mathcal{D}_i) = \text{std}(\deg)/\text{mean}(\deg)$. The candidate expert set size is determined by $\lceil \mathcal{S}_i \cdot \zeta \rceil$ with curvatures alternating through $0, -1, +1, -2, +2, \ldots$ (including hyperbolic, Euclidean, and spherical). Routing scores $\boldsymbol\alpha_{\mathcal{G}} = \text{softmax}(g(\mathcal{G})/\tau)$ are generated by a GCN encoder. As the confidence $\text{conf} = (1/\psi) \sum_i \max \alpha^{(i)}$ increases during training, the number of activated experts $m$ is dynamically reduced via $m \leftarrow \max(1, m - \text{conf})$.
    - **Design Motivation**: Fixed expert counts lead to either insufficient capacity or overfitting. Leveraging the pattern where a router becomes more confident over time allows for a natural reduction in active experts, improving generalization (Thm 3.5 provides an excess risk bound $\mathcal{R}(\psi_D) \leq \mathcal{R}(\psi_F)$).

3. **Theoretical Support for Domain Generalization**:

    - **Function**: Proves that the error bound for R-GFM on unseen domains is strictly lower than that of the SOTA MDGFM.
    - **Mechanism**: The encoder classes $\Phi_R$ and $\Phi_M$ are introduced into the domain generalization error bound. Since R-GFM expands the expressiveness of the encoder via multi-hop GoGs and Riemannian MoE, while controlling capacity through sparsification and dynamic top-$m$, Thm 3.5 demonstrates $\epsilon_{\text{R-GFM}} < \epsilon_{\text{MDGFM}}$.
    - **Design Motivation**: The core evaluation for GFMs is "zero-shot" or "few-shot" transfer to unseen graphs, necessitating formal guarantees beyond empirical gains.

### Loss & Training
The pre-training phase uses NT-Xent contrastive loss for the subgraph encoder. The GoG encoding phase employs downstream task losses (CE for node classification, BCE for link prediction) combined with standard MoE load-balancing losses. A leave-one-dataset-out strategy is used: pre-training on other graphs and fine-tuning on the target (1-shot for node classification, 5-shot for link prediction).

## Key Experimental Results

### Main Results

| Method | Wisconsin | Cornell | Citeseer | Cora | Pubmed | Computers | Photos | Texas |
|---|---|---|---|---|---|---|---|---|
| GCN | 17.46 | 19.53 | 26.89 | 31.98 | 44.29 | 39.43 | 50.39 | 18.48 |
| GAT | 16.86 | 16.51 | 25.27 | 26.81 | 45.11 | 38.05 | 56.51 | 18.36 |
| GFM (e.g., MDGFM) | (Lower) | — | — | — | — | — | — | — |
| **R-GFM** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

R-GFM consistently achieves SOTA performance across 18 real-world graphs; relative improvements on some datasets reach 49%.

### Ablation Study

| Configuration | Impact |
|---|---|
| Fixed 1-hop subgraphs only | Performance drops, proving the necessity of adaptive-hop |
| Fully-connected / Edgeless GoG | Both underperform similarity-sparse GoG, aligning with Thm 3.3 |
| Fixed top-$m$ routing | Underperforms confidence-aware dynamic top-$m$ |
| Single Euclidean expert | Significant performance loss on high-heterogeneity datasets |
| Edge budget $\mathcal{B}_{\text{edge}} = 0.6$ | Performance degrades if made sparser or denser |

### Key Findings
- The maximum gain of 49% occurs on datasets with high structural heterogeneity, supporting the expectation that dynamic geometric selection primarily benefits heterogeneous graphs.
- In robustness tests against graph perturbations, R-GFM shows the smallest decline under 30% random edge perturbations compared to baselines, attributed to the redundant information in multi-hop GoGs.
- Cross-scale generalization: After pre-training on ArXiv_2023, ogbn-ArXiv, Reddit, and PubMed, the model remains robust on Cora, Ele-Computers, Books-History, and Instagram test sets.

## Highlights & Insights
- While "Graph of Graphs" is not a new concept, combining it with adaptive hops and Riemannian MoE for GFMs is a first, addressing the long-standing issues of hop count and geometry.
- The "Router confidence increase $\rightarrow$ automatic $m$ shrinkage" mechanism is an elegant utilization of training dynamics, allowing MoE capacity to self-regulate for better generalization—a concept applicable to LLM MoE training.
- Using node degree CV to pre-determine expert counts avoids the pain of trial-and-error; this "prediction of capacity via statistics" can be extended to other MoE scenarios.

## Limitations & Future Work
- GoG construction requires traversing multi-hop subgraphs, leading to higher time and memory complexity than fixed-hop GFMs; adaptation for million-node graphs requires further acceleration.
- Currently, only three classes of constant-curvature Riemannian manifolds are considered; mixed-curvature or learnable curvature spaces remain unexplored.
- The similarity thresholds and edge budget of 0.6 are empirical values lacking a task-adaptive mechanism.
- Transfer effectiveness on data with stronger domain priors (e.g., molecular graphs, knowledge graphs) has not been fully verified.

## Related Work & Insights
- **vs. MDGFM**: MDGFM also provides theoretical analysis for GFMs but relies on a single receptive field and single geometric space; R-GFM introduces dynamism in both dimensions.
- **vs. Graph MoE (e.g., GMoE)**: Existing Graph MoEs use fixed top-$m$ and lack geometric priors; R-GFM introduces curvature as an inductive bias.
- **vs. Hyperbolic GNNs (HGNN/HGCN)**: Hyperbolic methods use only a single negative curvature space; R-GFM adaptively mixes curvatures to cover more diverse structures.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of adaptive-hop GoG and Riemannian MoE is fresh, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ 18 datasets, cross-domain transfer, perturbation robustness, and theoretical guarantees provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, theory and methodology are tightly aligned, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Provides actionable solutions to two major pain points in GFM research; open-source code helps lower the barrier to reproduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[AAAI 2026\] Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](../../AAAI2026/self_supervised/explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)
- [\[AAAI 2026\] HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](../../AAAI2026/self_supervised/hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)
- [\[ICML 2026\] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models](numleak_public_numeric_benchmarks_as_latent_labels_in_foundation_models.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)

</div>

<!-- RELATED:END -->
