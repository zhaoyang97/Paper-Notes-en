---
title: >-
  [Paper Note] Random Search Neural Networks for Efficient and Expressive Graph Learning
description: >-
  [NeurIPS 2025][Computational Biology][Graph Neural Networks] This paper proposes Random Search Neural Networks (RSNN), which replace random walks with randomized depth-first search (DFS) for graph structure sampling. On…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "Graph Neural Networks"
  - "Random Walks"
  - "Depth-First Search"
  - "Graph Representation Learning"
  - "Universal Approximation"
  - "Isomorphism Invariance"
date: 2026-05-08
content_hash: 8aac53c872573730
---

# Random Search Neural Networks for Efficient and Expressive Graph Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.22520](https://arxiv.org/abs/2510.22520)  
**Authors**: Michael Ito, Danai Koutra, Jenna Wiens (University of Michigan)
**Code**: [MLD3/RandomSearchNNs](https://github.com/MLD3/RandomSearchNNs)  
**Area**: Medical Imaging
**Keywords**: Graph Neural Networks, Random Walks, Depth-First Search, Graph Representation Learning, Universal Approximation, Isomorphism Invariance

## TL;DR

This paper proposes Random Search Neural Networks (RSNN), which replace random walks with randomized depth-first search (DFS) for graph structure sampling. On sparse graphs, RSNN achieves complete edge coverage with only $O(\log|V|)$ searches. Paired with a universal sequence model, RSNN attains universal approximation capability, and consistently outperforms RWNN on molecular and protein benchmarks using up to 16× fewer samples.

## Background & Motivation

### State of the Field
Random Walk Neural Networks (RWNN) have emerged as a promising graph learning paradigm that represents graphs as sequences of random walks processed by sequence models, overcoming limitations of message-passing neural networks (MPNNs) and graph Transformers. However, under practical sampling constraints, RWNNs face a severe expressiveness bottleneck — the node cover time of random walks can reach $O(|V||E|)$, and remains $O(|V|^2)$ even with non-backtracking walks or minimum-degree local rules, making complete coverage infeasible even on medium-sized graphs.

### Limitations of Prior Work
- **Incomplete coverage**: Random walks with limited steps frequently miss critical structures (e.g., side chains on rings), resulting in incomplete graph reconstruction.
- **Limited expressiveness**: This paper provides the first proof that RWNNs under partial coverage are strictly weaker than MPNNs — information loss fundamentally constrains model capacity regardless of sequence model strength.
- **High sampling cost**: RWNNs require $O(|V|)$ walks of length $O(|V|)$ to approach full coverage, incurring substantial computational overhead.
- Existing improvements such as non-backtracking walks (CRaWl) and MDLR walks reduce cover time but remain quadratic in complexity.

### Root Cause
The paper aims to design a novel graph sampling strategy that efficiently guarantees complete node and edge coverage, thereby achieving maximum theoretical expressiveness under practical sampling budgets. The key insight is that DFS-induced subgraphs are spanning trees, which naturally guarantee full node coverage, reducing the problem to edge coverage across multiple trees.

## Method

### Theoretical Analysis of RWNN Expressiveness

**Theorem 3.1 (RWNN–MPNN Equivalence)**: When an RWNN has access to a complete multiset of walks of length reaching the edge cover time $C_E(G)$, it is equivalent in expressiveness to an MPNN under injective sequence models and aggregation functions: $f_{\text{RWNN}}^{FC} \simeq f_{\text{MPNN}}$.

**Corollary 3.2 (RWNN under Partial Coverage)**: When an RWNN achieves only partial node/edge coverage, its expressiveness is strictly weaker than that of an MPNN: $f_{\text{RWNN}}^{PC} \prec f_{\text{MPNN}}$. This result bridges RWNNs and the WL hierarchy via the introduction of the Walk Weisfeiler-Lehman (WWL) color refinement algorithm.

### RSNN Architecture

RSNN consists of four core components:
1. **Sampling strategy**: Randomized depth-first search (DFS) instead of random walks. A DFS sequence is sampled uniformly: $S \sim \mathbb{U}(\mathcal{S}_{\text{DFS}}(G))$.
2. **Recording function**: $f_{\text{emb}}[i] = h_V(w_i) + \text{proj}(h_{\text{PE}}[i])$, incorporating positional encodings to mark discontinuities (backtracking points) in the sequence.
3. **Sequence model**: $f_{\text{seq}}: \mathbb{R}^{\ell \times d} \to \mathbb{R}^{\ell \times d}$, supporting GRU, LSTM, or Transformer backbones.
4. **Node aggregation**: Representations of the same node across multiple search sequences are averaged.

### Logarithmic Sampling Coverage Theorem

**Lemma 4.1**: For sparse connected graphs satisfying $|E| \leq C|V|$ with bounded maximum degree, the union of spanning trees from $m$ independent random searches contains all edges with probability at least $1 - \delta$, requiring only:

$$m \geq \frac{\ln(C|V|/\delta)}{\ln(d_{\max}/(d_{\max}-1))}$$

This is $O(\log|V|)$, representing an exponential improvement over the $O(|V|)$ walks required by RWNNs.

### Universal Approximation and Isomorphism Invariance

**Theorem 4.2 (Universal Approximation)**: On the space of sparse bounded-degree graphs, when the number of searches $m$ satisfies the condition in Lemma 4.1, RSNN with a universal sequence model can approximate any continuous graph function to arbitrary precision $\epsilon$.

**Theorem 4.3 (Probabilistic Isomorphism Invariance)**: The randomized DFS process in RSNN satisfies probabilistic invariance: for all isomorphisms $G \cong H$, $f_{\text{RSNN}}(G) \stackrel{d}{=} f_{\text{RSNN}}(H)$, and the expected predictor $\Phi(G) = \mathbb{E}[f_{\text{RSNN}}(G)]$ is a graph isomorphism-invariant function.

**Corollary 4.4**: SGD training converges to the optimal solution of an invariant objective, even when only $m=1$ search is sampled per forward pass.

### Runtime Complexity
- **RSNN**: A single DFS on sparse graphs costs $O(|V|)$; $m$ searches cost $O(m|V|)$.
- **RWNN**: $m$ walks of length $\ell$ cost $O(m\ell)$.
- When $\ell \ll |V|$, RWNN sampling is faster, but short walks fail to capture global structure. RSNN trades slightly higher sampling cost for complete coverage and stronger expressiveness.

## Key Experimental Results

### Experiment 1: Molecular and Protein Benchmarks

Evaluated on 4 small molecular datasets (MoleculeNet) and 4 protein datasets (ProteinShake), with all methods using the same number of samples $m$ and walk length $\ell = |V|$.

| Model | m | CLINTOX (AUC) | BBBP (AUC) | TOX21 (AUC) | SC Family (ACC) | EC Subclass (ACC) |
|------|---|--------------|------------|-------------|-----------------|-------------------|
| GCN | — | 62.4 | 73.9 | 67.5 | 3.9 | 31.2 |
| GIN | — | 59.7 | 75.3 | 66.9 | 10.4 | 37.2 |
| Fingerprint | — | 66.5 | 86.2 | 79.1 | — | — |
| CRAWL | 1 | 70.0 | 77.6 | 71.7 | 5.2 | 28.7 |
| **RSNN** | **1** | **88.1** | **87.5** | **79.8** | **13.9** | **36.8** |
| CRAWL | 16 | 89.1 | 87.0 | 80.9 | 15.5 | 48.7 |
| **RSNN** | **16** | **88.5** | **89.4** | **82.2** | **19.0** | **50.0** |

Key findings: RSNN at $m=1$ surpasses all RWNN variants at $m=16$ on molecular benchmarks; on protein graphs, RSNN maintains a significant advantage across all values of $m$, achieving 19.0% SC Family accuracy at $m=16$ compared to CRAWL's 15.5%.

### Experiment 2: Large-Scale Molecular Benchmarks and Dense Graphs

Evaluated on large-scale OGB datasets (160K–240K graphs) and a dense brain graph dataset:

| Dataset | Scale | RWNN-mdlr (m=1) | CRAWL (m=1) | **RSNN (m=1)** |
|--------|------|-----------------|-------------|----------------|
| PCBA-1030 | 160K graphs | 63.5 | 64.2 | **78.8** |
| PCBA-1458 | 195K graphs | 76.2 | 77.0 | **87.0** |
| PCBA-4467 | 240K graphs | 75.4 | 75.6 | **85.2** |

| Dataset | Avg.V | Avg.E | CRAWL m=4 | **RSNN m=4** | CRAWL m=16 | **RSNN m=16** |
|--------|-------|-------|-----------|--------------|------------|---------------|
| NeuroGraph | 1000 | 7029 | 77.5 | **80.4** | 68.3 | **86.5** |

RSNN achieves AUC improvements exceeding 10 percentage points at $m=1$ on large-scale molecular benchmarks. On the dense brain graph dataset at $m=16$, RSNN (86.5%) substantially outperforms CRAWL (68.3%), demonstrating that RSNN is not limited to sparse graphs.

## Highlights & Insights

- **Deep theoretical contributions**: The paper provides the first rigorous proof that RWNNs under partial coverage are strictly weaker than MPNNs, establishes a formal bridge between Walk WL and classical WL, and unifies the expressiveness analysis of two seemingly distinct graph model families.
- **$O(\log|V|)$ coverage efficiency**: By exploiting the property that DFS spanning trees naturally guarantee full node coverage, the number of samples required for edge coverage is reduced from $O(|V|)$ to $O(\log|V|)$ — an exponential improvement.
- **Three-in-one theoretical guarantees**: The paper simultaneously establishes efficient coverage, universal approximation capability, and probabilistic isomorphism invariance, forming a complete theoretical framework.
- **16× sampling efficiency**: Empirically, RSNN at $m=1$ matches or exceeds the performance of RWNN at $m=16$, demonstrating substantial practical sampling efficiency gains.

## Limitations & Future Work

- **Sparse graph assumption**: The theoretical analysis relies on $|E| = O(|V|)$ and bounded maximum degree; coverage efficiency degrades for dense graphs such as social networks and knowledge graphs.
- **DFS computational cost**: A full DFS traversal costs $O(|V|)$, making it infeasible for very large graphs (millions of nodes); theoretical analysis of truncated search is absent.
- **Sequence length constraints**: DFS sequences have length $O(|V|)$ (including backtracking steps), which can impair sequence model efficiency on large graphs.
- **Domain coverage**: Validation is primarily conducted on molecular and protein graphs; performance on other important graph learning scenarios such as social networks and recommendation systems remains untested.
- **Training stability**: Only median results are reported; the min–max spread on some datasets (e.g., CRAWL on NeuroGraph) is substantial, suggesting potential training instability.

## Related Work & Insights

- **CRaWl (Tönshoff et al. 2023)**: Uses non-backtracking walks with node-level aggregation and is the strongest RWNN baseline, but its coverage efficiency remains $O(|V|^2)$, making it significantly weaker than RSNN under low sampling budgets.
- **RWNN-mdlr (Kim et al. 2025)**: Minimum-degree local rule walks achieve optimal first-order walk coverage at $O(|V|^2)$, but quadratic complexity remains insufficiently efficient.
- **CRW (Chen et al. 2025)**: A random walk method learning long-range dependencies via non-backtracking and learnable walk policies.
- **Wang & Cho (2024)**: A non-convolutional GNN using uniform random walks with node anonymization; expressiveness is limited by partial coverage.
- **GIN (Xu et al. 2019)**: A WL-equivalent MPNN; RSNN theoretically matches and can exceed its expressiveness.
- **Graph Transformer (Dwivedi & Bresson 2020)**: Full-graph attention mechanism; exhibits unstable performance on small datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to introduce DFS-based search into graph neural networks, effecting a paradigm shift from walks to searches.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers molecular, protein, large-scale, and dense graph settings, but lacks experiments on social and knowledge graphs.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory, method, and experiments are tightly integrated with a progressive analytical structure and clear illustrations.
- **Value**: ⭐⭐⭐⭐ — Introduces a theoretically grounded and efficient new paradigm for graph representation learning, though the sparse graph assumption limits applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Autoencoding Random Forests](autoencoding_random_forests.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](../../ICML2026/computational_biology/learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)

</div>

<!-- RELATED:END -->
