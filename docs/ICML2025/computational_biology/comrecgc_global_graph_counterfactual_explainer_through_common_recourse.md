---
title: >-
  [Paper Note] ComRecGC: Global Graph Counterfactual Explainer through Common Recourse
description: >-
  [ICML 2025][Computational Biology][Counterfactual Explanation] This paper formally defines the **Common Recourse** global counterfactual explanation problem for Graph Neural Networks (GNNs) for the first time, proves its NP-hardness, and proposes the ComRecGC algorithm. By searching for counterfactual graphs using Multi-head Vertex-Reinforced Random Walk (VRRW) and extracting common recourses via DBScan clustering, ComRecGC consistently outperforms existing baselines by 10%–3…
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Counterfactual Explanation"
  - "Graph Neural Networks"
  - "Common Recourse"
  - "Global Explanation"
  - "Submodular Optimization"
date: 2026-05-08
content_hash: 4759c9b4bc1d5c3a
---

# ComRecGC: Global Graph Counterfactual Explainer through Common Recourse

**Conference**: ICML 2025  
**arXiv**: [2505.07081](https://arxiv.org/abs/2505.07081)  
**Code**: [GitHub](https://github.com/ssggreg/COMRECGC)  
**Area**: Computational Biology  
**Keywords**: Counterfactual Explanation, Graph Neural Networks, Common Recourse, Global Explanation, Submodular Optimization

## TL;DR

This paper formally defines the **Common Recourse** global counterfactual explanation problem for Graph Neural Networks (GNNs) for the first time, proves its NP-hardness, and proposes the ComRecGC algorithm. By searching for counterfactual graphs using Multi-head Vertex-Reinforced Random Walk (VRRW) and extracting common recourses via DBScan clustering, ComRecGC consistently outperforms existing baselines by 10%–30% in coverage across four real-world datasets: NCI1, Mutagenicity, AIDS, and Proteins.

## Background & Motivation

Graph Neural Networks (GNNs) have been widely applied in fields such as drug discovery, molecular property prediction, and protein function classification, but their predictions remain a black box. **Counterfactual Explanation (CE)** is an important explainability approach: given an input graph $G$ classified as "rejected" by a GNN, it finds a structurally similar graph $H$ classified as "accepted", thereby revealing the decision boundary.

The main limitations of prior work include:

**Local counterfactual explanations** (e.g., CF-GNNExplainer, CFF, RCExplainer): These generate an independent counterfactual for each input graph, resulting in too many explanations that are difficult to comprehend globally at the model level.

**Global counterfactual explanations** (e.g., GCFExplainer): These generate a small set of counterfactual graphs to cover all input graphs. However, the exact same counterfactual might imply completely different transformations (recourses) for different input graphs, failing to provide actionable guidance.

**GLOBE-CE**: Generates global explanations by learning translation directions, but has not yet been applied to graph data.

**Key Insight**: If a single transformation (recourse) can simultaneously turn multiple "rejected" graphs into "accepted" ones, this **Common Recourse** can provide both global explainability and instance-level actionability. For example, in toxicity prediction, "removing the NO₂ group" could be a common recourse applicable to various mutagenic molecules, holding high practical value.

## Method

### Overall Architecture

ComRecGC consists of three stages:

1. **Graph Embedding**: Maps graphs into $\mathbb{R}^l$ ($l=64$) using the GREED algorithm to approximate the normalized Graph Edit Distance (GED).
2. **Multi-head Vertex-Reinforced Random Walk (VRRW)**: Performs random walks in the graph edit space to search for high-quality counterfactual graphs.
3. **Clustering & Greedy Selection**: Clusters all recourse vectors using DBScan, and greedily selects $R$ common recourses to maximize coverage.

The overall pseudo-code (Algorithm 2) is highly concise:
```
ComRecGC(Φ, G, k, M, τ, R, n):
  S ← Multi-head VRRW(Φ, G, k, M, τ)       // Phase 1: Search counterfactuals
  S ← top n frequently visited in S          // Filter highly visited counterfactuals
  return CR_clustering(G, S, R)               // Phase 2: Clustering + Greedy
```

### Key Designs

#### 1. Problem Formulation

Given a set of input graphs $\mathbb{G}$ classified as 'rejected' by a GNN $\Phi$, a graph $H$ is a **counterfactual** for $G$ if and only if $\Phi(H)=1$ and $d(G,H) \leq \theta$ (where the normalized GED distance does not exceed a threshold $\theta$).

A **Recourse** is defined as a vector in the embedding space $\vec{r} = \overrightarrow{z(G)z(H)}$, representing the displacement from the input graph embedding to the counterfactual graph embedding. Two recourses $r_1, r_2$ form a **common recourse** if there exists a center vector $\vec{v}$ such that $\|\vec{v} - \vec{r_i}\|_2 \leq \Delta$.

The **Finding Common Recourse (FCR) Problem**: Given a budget $R$, find a set of common recourses $\mathbb{F}$ of size at most $R$ that maximizes coverage:

$$\max_{\mathbb{F}} \text{coverage}(\mathbb{F}) \quad \text{s.t.} \quad |\mathbb{F}| \leq R$$

where coverage is defined as the proportion of input graphs that can be transitioned into a counterfactual by at least one common recourse in $\mathbb{F}$.

**Theoretical Results**:
- **Theorem 1**: The FCR problem is NP-hard (proved via reduction from the Maximum Coverage problem).
- The coverage function $f$ is **monotonic and submodular** $\rightarrow$ the greedy algorithm offers a $(1 - 1/e)$ approximation guarantee.

#### 2. Graph Embedding (GREED Algorithm)

The GREED algorithm is employed to learn graph embeddings into $\mathbb{R}^{64}$ such that the distance in the embedding space approximates the normalized Graph Edit Distance (GED):

$$\widehat{\text{GED}}(G_1, G_2) = \frac{\text{GED}(G_1, G_2)}{|V_1| + |V_2| + |E_1| + |E_2|}$$

This allows a unified comparison of distances between graphs of different sizes.

#### 3. Multi-head Vertex-Reinforced Random Walk (VRRW)

This is the core innovation of ComRecGC, running random walks in the graph edit space to search for counterfactuals:

- **$k$ heads** start simultaneously from different input graphs (where $k=5$ in experiments).
- At each step, a **lead** is randomly chosen, which moves according to the transition probability:

$$p(u_\ell, v) \propto p_\phi(v) \cdot C(v)$$

where $p_\phi(v)$ is the probability predicted by the GNN that $v$ belongs to the 'accepted' class, and $C(v)$ is the visit count of $v$ plus one (the vertex-reinforced mechanism, which encourages the exploration of frequently visited counterfactual regions).

- **Non-lead heads** follow the lead, choosing the neighbor that makes their recourse vector closest to the lead's recourse vector:

$$\arg\min_{v \in \mathcal{N}(u_i) \cup \{u_i\}} \|\overrightarrow{z(G_\ell)z(u_\ell)} - \overrightarrow{z(G_i)z(v)}\|_2$$

A key design aspect is forcing multiple heads to **execute similar recourses synchronously**, which naturally promotes the emergence of common recourses.

- **Teleportation**: With probability $\tau=0.05$, all heads are reset to the input graphs, where the teleportation probability biases towards input graphs with fewer visits:

$$p_\tau(G) = \frac{\exp(-t(G))}{\sum_{G' \in \mathbb{G}} \exp(-t(G'))}$$

where $t(G)$ is the number of walks initiated from $G$, ensuring a balanced exploration.

- The random walk terminates after $M=50000$ steps, selecting highly frequented counterfactuals as candidates.

#### 4. Clustering and Greedy Selection

The recourse vectors $\{\overrightarrow{z(G)z(v)}\}$ generated by all candidate counterfactuals are clustered using DBScan with a radius $\Delta$, where each cluster represents a common recourse. Then, $R=100$ common recourses are greedily selected, choosing the one with the maximum marginal coverage gain at each step:

```
CR_clustering(G, S, R):
  C ← Δ-clusterize {z(G)z(v)→ : v∈S, G∈G}
  F ← ∅
  for i = 1 to R:
    r ← argmax_{r∈C} gain(r; F)    // Greedy choice for max marginal gain
    F ← F ∪ {r}
  return F
```

### Loss & Training

ComRecGC itself **requires no training**; it is a post-hoc explainer. The training configuration of the underlying GNN (GCN) is as follows:

- Architecture: 3-layer graph convolution + max pooling + fully-connected layer
- Optimizer: Adam, learning rate 0.001
- Training: 1000 epochs
- Data split: 80%/10%/10% (train/validation/test)

**Key Hyperparameters**:
- $\theta = 0.1$ (counterfactual distance threshold, 0.15 for the Proteins dataset)
- $\Delta = 0.02$ (similarity threshold for common recourses)
- $k = 5$ (number of random walk heads)
- $\tau = 0.05$ (teleportation probability)
- $M = 50000$ (number of random walk steps)
- $R = 100$ (number of common recourses)

**Complexity Analysis**: $O(Mhk + n^2|\mathbb{G}|^2)$, where $h$ is the maximum degree in the graph edit space and $n$ is the number of filtered highly frequented counterfactuals.

## Key Experimental Results

### Main Results

**Dataset Statistics**:

| Dataset | Graph Count | Node Count | Edge Count | Node Label Count | Classification Task |
|--------|--------|--------|------|-----------|---------|
| NCI1 | 3978 | 118714 | 128663 | 10 | Anti-cancer activity |
| Mutagenicity | 4308 | 130719 | 132707 | 10 | Mutagenicity |
| AIDS | 1837 | 28905 | 29985 | 9 | Anti-HIV activity |
| Proteins | 1113 | 43471 | 81044 | 3 | Enzymes / Non-enzymes |

**Main Results for the FCR Problem** (Coverage ↑ / Cost ↓):

| Method | NCI1 Cov. | NCI1 Cost | Mutag. Cov. | Mutag. Cost | AIDS Cov. | AIDS Cost | Prot. Cov. | Prot. Cost |
|------|-----------|-----------|-------------|-------------|-----------|-----------|------------|------------|
| GCFExplainer | 21.4% | 5.75 | 20.6% | 6.91 | 14.2% | 6.97 | 32.8% | 10.65 |
| ComRecGC 2-head | 40.5% | 5.12 | 45.9% | 5.74 | 32.8% | 6.71 | 45.9% | 11.44 |
| ComRecGC 3-head | 44.5% | 5.07 | **52.6%** | 5.61 | 33.6% | **6.62** | 45.9% | 11.51 |
| **ComRecGC 5-head** | **42.9%** | **4.95** | 51.8% | **5.63** | **34.7%** | 6.74 | **46.4%** | 11.59 |

ComRecGC outperforms GCFExplainer in coverage by **+21.5%, +31.2%, +20.5%, and +13.6%** across the four datasets, respectively.

**FC Problem Results** (limiting the number of counterfactuals to $T=|\mathbb{G}|$):

| Method | NCI1 | Mutagenicity | AIDS | Proteins |
|------|------|-------------|------|----------|
| Dataset CF | 8.52% | 10.4% | 0.41% | 29.0% |
| LocalRWExplainer | 19.0% | 18.2% | 12.9% | 22.1% |
| GCFExplainer | 14.7% | 11.9% | 14.2% | 29.8% |
| **ComRecGC** | **33.4%** | **46.7%** | **24.3%** | **39.6%** |

### Ablation Study

| Configuration | NCI1 Cov. | Mutag. Cov. | AIDS Cov. | Prot. Cov. | Description |
|------|-----------|-------------|-----------|------------|------|
| ComRecGC (Full) | 42.9% | 51.8% | 34.7% | 46.4% | All components |
| w/o clustering | 10.1% | 8.2% | 13.6% | 33.9% | Without DBScan clustering, coverage drops significantly |
| w/o multi-head | 21.4% | 20.6% | 14.2% | 32.8% | Degenerates to single-head, performance drops to GCFExplainer level |

Clustering is crucial for large-scale datasets (NCI1, Mutagenicity); the multi-head mechanism is the core of common recourse formation.

### Key Findings

1. **Common Recourse vs. Global Counterfactual** (coverage with 10 explanations): ComRecGC's common recourse explanations reach 18.0% coverage on Proteins, significantly outperforming GCFExplainer's 10.9%, and showing comparable or superior performance on other datasets.
2. **Explainability Case Study**: On the Mutagenicity dataset, ComRecGC finds that "removing the NO₂ group" is a common recourse that transforms mutagenic molecules into non-mutagenic ones, which aligns with established chemical domain knowledge.
3. **GNN Architecture Generalization**: Consistently outperforms GCFExplainer across GAT, GraphSAGE, and GIN.
4. **Running Time**: The total execution time on Mutagenicity is approximately 199 minutes (172 minutes for VRRW + 27 minutes for clustering).

## Highlights & Insights

1. **Elegance of Problem Definition**: Formalizing "common recourse" as a vector clustering problem in the embedding space bypasses the difficulty of defining "identical transformations" in the graph structure space, while preserving actionability.
2. **Synergistic Design of Multi-head VRRW**: The mechanism where non-leads follow leads to perform similar recourses is a key design that naturally promotes the emergence of common recourses. This is far more efficient than generating counterfactuals independently followed by post-hoc clustering.
3. **Combination of Theory and Practice**: Solid theoretical guarantees with NP-hardness proof, submodularity analysis, and a $(1-1/e)$ approximation limit, combined with significant empirical improvements.
4. **Practical Value**: In drug discovery, a common recourse acts like a "universal prescription", informing researchers that "this type of transformation typically makes molecules non-toxic," which directly guides molecular design.

## Limitations & Future Work

1. **Physical Feasibility Neglected**: Generated counterfactual graphs may not correspond to real-world molecules or proteins, lacking domain-specific constraints.
2. **Parameter Sensitivity**: The selection of $\theta$ and $\Delta$ is highly empirical. For instance, the paper specifically adjusts $\theta=0.15$ on Proteins; an automated tuning mechanism is absent.
3. **Scalability Bottleneck**: As $\theta$ increases, the number of recourses grows exponentially (increasing 100-fold on Mutagenicity), rendering DBScan intractable.
4. **Binary Classification Only**: The current framework is limited to "reject/accept" binary classification, requiring extension for multi-class scenarios.
5. **Dependency on Embedding Quality**: The approach relies heavily on the assumption that the embedding space accurately approximates GED, making it sensitive to the choice of embedding algorithms.
6. **High Computational Overhead**: Running 50,000 random walk steps takes over 2 hours; acceleration is needed for real-world deployment.

## Related Work & Insights

- **GCFExplainer** (Huang et al., WSDM 2023): Global counterfactual explanation baseline, but does not consider the consistency of recourses.
- **GLOBE-CE** (Ley et al., ICML 2023): Learns global explanations in translation directions, but has not yet been applied to graph data.
- **CF-GNNExplainer** (Lucic et al., AISTATS 2022): Local counterfactual explanations based on subgraph generation.
- **GREED** (Ranjan et al., 2023): Neural approximation of graph edit distance, providing the embedding foundation.
- **VRRW** (Pemantle, 2004): Theoretical foundation for vertex-reinforced random walks.

**Insights**: The idea of common recourse can be transferred to other domains. For instance, in medical imaging, finding "common edit operations" can explain why a set of scans is classified as abnormal; in recommender systems, finding "common preference shifts" can explain recommendation results for a cohort of users.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ - Formulates the common recourse problem for the first time; highly original multi-head VRRW design.
- Experimental Thoroughness: ⭐⭐⭐⭐ - Evaluated on 4 datasets and multiple GNN architectures with a comprehensive ablation study; lacks large-scale graph data.
- Writing Quality: ⭐⭐⭐⭐ - Well-structured with a strong integration of theory and experiments; heavy notation requires some effort.
- Value: ⭐⭐⭐⭐ - Holds direct application value for fields like molecular design, though physical feasibility remains a hurdle for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[ICML 2025\] PolyConf: Unlocking Polymer Conformation Generation through Hierarchical Generative Models](polyconf_unlocking_polymer_conformation_generation_through_hierarchical_generati.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](supercharging_graph_transformers_with_advective_diffusion.md)
- [\[ICML 2026\] scCBGM: Interpretable Single-Cell Counterfactual Editing](../../ICML2026/computational_biology/sccbgm_interpretable_single-cell_counterfactual_editing.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
