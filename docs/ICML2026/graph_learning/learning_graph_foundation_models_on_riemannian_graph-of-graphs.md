---
title: >-
  [Paper Note] Learning Graph Foundation Models on Riemannian Graph-of-Graphs
description: >-
  [ICML 2026][Graph Learning][Graph Foundation Model] R-GFM treats "subgraphs with different hop counts" as nodes in a higher-level Graph-of-Graphs…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Graph Foundation Model"
  - "Graph-of-Graphs"
  - "Riemannian MoE"
  - "adaptive-hop"
  - "domain generalization"
date: 2026-05-08
content_hash: 0fc3034b39f3a3e9
---

# Learning Graph Foundation Models on Riemannian Graph-of-Graphs

**Conference**: ICML 2026  
**arXiv**: [2605.09993](https://arxiv.org/abs/2605.09993)  
**Code**: <https://github.com/USTC-DataDarknessLab/R-GFM>  
**Area**: Graph Foundation Models / Self-supervised Representation / Riemannian Geometry  
**Keywords**: Graph Foundation Model, Graph-of-Graphs, Riemannian MoE, adaptive-hop, domain generalization

## TL;DR
R-GFM treats "subgraphs with different hop counts" as nodes in a higher-level Graph-of-Graphs, and uses a dynamic MoE routing mechanism to assign each GoG to the Riemannian manifold (hyperbolic / Euclidean / spherical) with the best-matched curvature. This simultaneously addresses two inherent limitations of existing graph foundation models: fixed receptive fields and single Euclidean embedding. It achieves up to 49% relative improvement on downstream tasks.

## Background & Motivation
**Background**: Graph foundation models (GFMs, e.g., OFA, Prodigy, MDGFM) achieve cross-task and cross-domain transfer via pretraining on massive graphs, representing the "foundation model era" in graph ML.

**Limitations of Prior Work**: (1) Existing GFMs use **fixed-hop subgraph sampling**, such as 1-hop or 2-hop neighbors as the receptive field. However, downstream tasks have vastly different hop requirements—homogeneous citation networks only need 1-2 hops, while e-commerce fraud detection requires ≥4 hops to uncover long-chain collusion. Fixed hops inevitably lead to underfitting or noise in some tasks. (2) Existing methods embed all subgraphs into a single Euclidean space, but subgraphs with different hops have drastically different structures (locally dense vs. globally sparse and hierarchical), and a single geometry distorts representations.

**Key Challenge**: The conflict between fixed-structure receptive fields and heterogeneous hop requirements of downstream tasks; the conflict between single geometry and multi-scale structural heterogeneity.

**Goal**: (1) Design a pretraining paradigm that can adaptively capture multi-hop structures; (2) Enable the model to dynamically switch Riemannian manifolds in the representation space.

**Key Insight**: Elevate "subgraphs of different hops" to nodes in a Graph-of-Graphs (GoG), allowing the model to explicitly reason over scales; then use MoE to route each GoG to the expert with the best-matched geometry curvature.

**Core Idea**: **"Structural scale as a first-class citizen"**—adaptive-hop GoG addresses scale mismatch, confidence-aware dynamic Riemannian MoE addresses geometry mismatch.

## Method

### Overall Architecture
R-GFM consists of four stages: (A) Compute the coefficient of variation (CV) of node degree distribution to determine the candidate set of Riemannian experts, and for each training node $v$, sample a set of $1, 2, \ldots, K$-hop subgraphs $\{G_v^{(i)}\}_{i=1}^K$; (B) Pretrain a subgraph encoder with contrastive learning to embed each subgraph as $\mathbf{X}_{\text{sub}}$; (C) Construct a sparse GoG $\mathcal{G}$ based on subgraph similarity, and encode the GoG using dynamic MoE-based Riemannian routing; (D) Aggregate outputs from all experts to obtain a fused embedding for downstream node classification / link prediction tasks.

### Key Designs

1. **Adaptive-hop GoG Construction (adaptive hop count + similarity sparsification)**:

    - Function: Avoids fixed receptive fields while controlling GPU memory.
    - Mechanism: For each training node $v$, incrementally increase hop count $K$ using "online greedy + memory test", reverting to the last feasible $K$ upon OOM, ensuring $K \leq \mathcal{B}_{\text{GPU}}$. Subgraph embeddings are pretrained with NT-Xent contrastive loss. GoG edges are constructed by sampling from a distribution based on subgraph cosine similarity: $\text{Prob}(i,j) = e^{\mathbf{S}[i,j]} / \sum_{u,v} e^{\mathbf{S}[u,v]}$, then sample $\mathcal{B}_{\text{edge}} = 0.6 \cdot K(K-1)/2$ edges without replacement and symmetrize. Theoretically (Thm 3.2), the embedding noise of multi-hop sampling $\|\boldsymbol\sigma_V\|_2 \leq \|\boldsymbol\sigma_F\|_2$ is strictly less than fixed-hop, and similarity-sparse GoG yields lower error than "edgeless GoG" or "fully connected GoG" (Thm 3.3).
    - Design Motivation: Fixed hops miss long-range information or drown out local details; dense GoG introduces noise, sparse but random GoG lacks structural priors—similarity sparsification balances all three.

2. **Dynamic MoE-based Riemannian Routing (dual dynamics: candidate set + Top-m)**:

    - Function: Adaptively determines the number of Riemannian experts and how many to activate per instance, based on dataset structural heterogeneity.
    - Mechanism: Quantify each training set's structural heterogeneity using the coefficient of variation $\text{CV}(\mathcal{D}_i) = \text{std}(\deg)/\text{mean}(\deg)$, then use sliding statistics $\mathcal{S}_i = \text{normalize}(\mu_i + \sigma_i)$ to accumulate heterogeneity; candidate expert set size is $\lceil \mathcal{S}_i \cdot \zeta \rceil$, with curvatures alternately assigned as $0, -1, +1, -2, +2, \ldots$ (covering hyperbolic, Euclidean, spherical). Routing scores $\boldsymbol\alpha_{\mathcal{G}} = \text{softmax}(g(\mathcal{G})/\tau)$ are computed with a GCN encoder; confidence $\text{conf} = (1/\psi) \sum_i \max \alpha^{(i)}$ increases during training, dynamically shrinking the number of activated experts $m \leftarrow \max(1, m - \text{conf})$.
    - Design Motivation: Fixed expert count is either insufficient or overfits; fixed top-$m$ is also inflexible. Leveraging the observation that "router confidence increases during training" naturally reduces activated experts, improving generalization (theoretically, excess risk bound $\mathcal{R}(\psi_D) \leq \mathcal{R}(\psi_F)$ is provided).

3. **Theoretical Support for Domain Generalization**:

    - Function: Proves that R-GFM's error upper bound on unseen target domains is strictly lower than SOTA MDGFM.
    - Mechanism: Introduce encoder classes $\Phi_R$ and $\Phi_M$ into the domain generalization error bound. Since R-GFM expands the encoder class expressiveness via GoG multi-hop and Riemannian MoE, and controls capacity via similarity sparsification and dynamic top-$m$, Thm 3.5 shows $\epsilon_{\text{R-GFM}} < \epsilon_{\text{MDGFM}}$.
    - Design Motivation: The core evaluation for GFM is "can it still work on unseen graphs"—formal guarantees are required, not just empirical gains.

### Loss & Training
In pretraining, the subgraph encoder uses NT-Xent contrastive loss; in the GoG encoding stage, downstream task losses are used (CE for node classification, BCE for link prediction) plus standard MoE load balancing loss. Leave-one-dataset-out transfer: pretrain on other graphs, fine-tune on the target graph (1-shot node classification / 5-shot link prediction).

## Key Experimental Results

### Main Results

| Method | Wisconsin | Cornell | Citeseer | Cora | Pubmed | Computers | Photos | Texas |
|---|---|---|---|---|---|---|---|---|
| GCN | 17.46 | 19.53 | 26.89 | 31.98 | 44.29 | 39.43 | 50.39 | 18.48 |
| GAT | 16.86 | 16.51 | 25.27 | 26.81 | 45.11 | 38.05 | 56.51 | 18.36 |
| GFM (MDGFM, etc. baselines) | (slightly lower than R-GFM) | — | — | — | — | — | — | — |
| **R-GFM** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** | **Best** |

Consistently SOTA on 18 real-world graphs (10 main settings + 4 large-scale training sets + 4 test sets); up to 49% relative improvement on some datasets.

### Ablation Study

| Configuration | Impact |
|---|---|
| Fixed 1-hop subgraph only | Performance drops, proving adaptive-hop is necessary |
| Fully connected / edgeless GoG | Both worse than similarity-sparse GoG, consistent with Thm 3.3 |
| Fixed top-$m$ routing | Worse than confidence-aware dynamic top-$m$ |
| Single Euclidean expert | Significant drop on highly heterogeneous datasets |
| Edge budget $\mathcal{B}_{\text{edge}} = 0.6 \cdot K(K-1)/2$ | Both sparser and denser settings degrade performance |

### Key Findings
- The maximum 49% improvement occurs on datasets with the highest structural heterogeneity, confirming that "dynamic geometry selection mainly benefits heterogeneous graphs."
- In graph perturbation robustness tests, R-GFM shows the smallest relative drop under 30% random edge perturbation, attributed to the redundancy of GoG multi-hop information.
- Cross-scale generalization: After pretraining on ArXiv_2023 + ogbn-Arxiv + Reddit + PubMed, the model remains robust on test sets like Cora / Ele-Computers / Books-History / Instagram.

## Highlights & Insights
- "Graph of Graphs" is not a new concept, but integrating it with adaptive-hop + Riemannian MoE as the core of a GFM is novel; this addresses two long-standing GFM challenges: hop and geometry.
- "Router confidence increases → automatic shrinkage of top-$m$" is an elegant use of training dynamics: MoE capacity self-shrinks in late training to improve generalization, a strategy transferable to LLM MoE training.
- Using node degree CV to pre-determine expert count avoids trial-and-error; this "capacity prediction based on data statistics" can be generalized to other MoE scenarios.

## Limitations & Future Work
- GoG construction requires traversing multi-hop subgraphs, resulting in higher time and memory complexity than fixed-hop GFM; further acceleration is needed for ultra-large graphs (million-node scale).
- Riemannian manifolds currently only consider three constant curvatures; more complex mixed-curvature or learnable curvature settings remain unexplored.
- The subgraph similarity threshold and edge budget 0.6 are empirical values, lacking a task-adaptive mechanism.
- Transfer performance on domain-prior-rich data such as molecular graphs and knowledge graphs is not fully validated.

## Related Work & Insights
- **vs MDGFM**: MDGFM is also GFM + theoretical analysis, but uses a single receptive field and single geometry; R-GFM is dynamic in both dimensions.
- **vs Graph MoE (GMoE, etc.)**: Their top-$m$ is fixed and experts lack geometric priors; R-GFM introduces curvature as an inductive bias.
- **vs Hyperbolic GNNs (HGNN / HGCN)**: Hyperbolic methods use only a single negative curvature space; R-GFM uses MoE to adaptively mix curvatures, covering more structures.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of adaptive-hop GoG + Riemannian MoE is fresh, though each component has precedents
- Experimental Thoroughness: ⭐⭐⭐⭐ 18 datasets + cross-domain generalization + perturbation robustness + theoretical guarantees, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Clear structure, tight alignment of theory and method, intuitive illustrations
- Value: ⭐⭐⭐⭐ Provides actionable solutions to two major GFM pain points, open-source code lowers reproduction barriers

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](view_space_learning_representation_across_arbitrary_graphs.md)
- [\[AAAI 2026\] Adaptive Riemannian Graph Neural Networks](../../AAAI2026/graph_learning/adaptive_riemannian_graph_neural_networks.md)

</div>

<!-- RELATED:END -->
