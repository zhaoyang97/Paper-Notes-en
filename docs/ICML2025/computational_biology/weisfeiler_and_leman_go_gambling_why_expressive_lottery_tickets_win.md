---
title: >-
  [Paper Note] Weisfeiler and Leman Go Gambling: Why Expressive Lottery Tickets Win
description: >-
  [ICML2025][Computational Biology][Graph Lottery Ticket Hypothesis] This work theoretically connects GNN expressiveness (Weisfeiler-Leman test) with the lottery ticket hypothesis (LTH) for the first time. It proposes and proves the Strong Expressive Lottery Ticket Hypothesis (SELTH), demonstrating that there exist trainable subnetworks in sparsely initialized GNNs that preserve 1-WL expressiveness, and that sparsely initialized networks with higher expressiveness are more like…
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Graph Lottery Ticket Hypothesis"
  - "GNN Expressiveness"
  - "Weisfeiler-Leman Test"
  - "Sparse Initialization"
  - "Critical Path"
  - "Graph Pruning"
date: 2026-05-08
content_hash: ff3754923f0cda46
---

# Weisfeiler and Leman Go Gambling: Why Expressive Lottery Tickets Win

**Conference**: ICML2025  
**arXiv**: [2506.03919](https://arxiv.org/abs/2506.03919)  
**Authors**: Lorenz Kummer, Samir Moustafa, Anatol Ehrlich, Franka Bause, Nikolaus Suess, Wilfried N. Gansterer, Nils M. Kriege (University of Vienna)
**Code**: [lorenz0890/wl2025lottery](https://github.com/lorenz0890/wl2025lottery)  
**Area**: Computational Biology  
**Keywords**: Graph Lottery Ticket Hypothesis, GNN Expressiveness, Weisfeiler-Leman Test, Sparse Initialization, Critical Path, Graph Pruning

## TL;DR

This work theoretically connects GNN expressiveness (Weisfeiler-Leman test) with the lottery ticket hypothesis (LTH) for the first time. It proposes and proves the Strong Expressive Lottery Ticket Hypothesis (SELTH), demonstrating that there exist trainable subnetworks in sparsely initialized GNNs that preserve 1-WL expressiveness, and that sparsely initialized networks with higher expressiveness are more likely to become "winning tickets." Additionally, it highlights the severe consequences of unrecoverable expressiveness loss caused by improper pruning in scenarios such as drug discovery.

## Background & Motivation

The Lottery Ticket Hypothesis (LTH) suggests that a large, randomly initialized neural network contains smaller, trainable subnetworks ("winning tickets") that can match the performance of the full network. While LTH has been thoroughly studied theoretically in the CNN domain (e.g., SLTH proves the existence of subnetworks without training), its application in GNNs remains largely limited to empirical validation, leaving a severe lack of theoretical foundations.

Limitations of Prior Work in GNN Lottery Tickets:
- **UGS**: Jointly prunes the adjacency matrix and model weights to discover Graph Lottery Tickets (GLTs), while completely ignoring expressiveness preservation.
- **wang2022searching**: Searches for GLTs through layer-wise sparsification and regularization routing pruning, likewise neglecting expressiveness.
- **yan2024multicoated**: Proposes Multicoated Supermasks based on SLTH, focusing on memory efficiency rather than expressiveness.

**Key Challenge**: The link between LTH and GNN expressiveness remains completely unexplored. Existing works focus on reducing parameters and computation; however, the impact of pruning before training on the GNN's ability to distinguish non-isomorphic graphs has been neglected.

**Design Motivation in Drug Discovery**: In molecular property prediction, structurally similar molecules can exhibit drastically different activity against the same protein target (known as activity cliffs). Stereoisomers may differ only in three-dimensional spatial arrangements but vary vastly in biological activity. If pruning renders GNNs incapable of distinguishing these critical molecular structures, it could lead to toxic drugs being misclassified as safe.

## Method

### Overall Architecture

This paper establishes a theoretical framework to understand GNN pruning from the perspective of expressiveness. The analysis target is moment-based GNN architectures (e.g., GIN, GCN), whose update rule is given by:

$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)}\left(\mathbf{h}_v^{(k-1)} + \sum_{u \in N(v)} \mathbf{h}_u^{(k-1)}\right)$$

Graph-level embeddings are obtained by concatenating readout vectors across layers: $\mathbf{h}_G = \|_{k=0}^{n} \sum_{v \in V(G)} \mathbf{h}_v^{(k)}$

### Design 1: Expressiveness Preservation Criterion and Critical Paths

**Criterion 1**: For $\Phi^{(k+1)}$ (the GNN with a classification head) to correctly classify all graphs in a dataset $D$, $\Phi^{(k)}$ (the message passing layers) must distinguish all non-isomorphic graph pairs belonging to different classes. Any pruning mask that loses this distinguishing ability cannot be a winning ticket.

**Critical Path (Definition 3.1)**: Let $\mathcal{W}_{\Phi^{(k)}} = \bigcup_i^k \mathcal{W}_i$ be the union of all MLP weights across all MP layers of the GNN. If there exists no weight configuration that allows the pruned network to match the original performance after removing a set of paths, then that set of paths is critical. Critical paths form a subset of the maximally expressive subnetwork—pruning that keeps all critical paths can theoretically constitute a winning ticket.

### Design 2: Strong Expressive Lottery Ticket Hypothesis (SELTH)

**Theorem 3.2 (Core Theorem)**: In moment-based GNNs, there exist sparsely initialized, trainable subnetworks whose expressiveness matches the 1-WL test. For MLPs using injective activation functions (where output dimensions of each layer are no smaller than input dimensions), there exists a sparse weight configuration that maintains the injectivity of each MLP layer over the input domain, thereby guaranteeing 1-WL equivalent expressiveness.

This is the first formal generalization of the classical SLTH to GNNs, introducing the "expressiveness" constraint—the subnetwork must not only be trainable but must also preserve maximum graph discrimination capability.

### Design 3: Expressiveness versus Convergence/Generalization

**Theorem 3.3**: If two graphs obtain partially identical embeddings upon initialization, the gradient contributions of these shared embeddings are codirectional, causing the embeddings of the two graphs to remain entangled throughout training. A more expressive initialization (with more unique embeddings) provides more diverse gradient signals, potentially accelerating convergence and improving generalization.

### Design 4: Unrecoverability of Expressiveness Loss

**Lemma 3.6**: Given a dataset with $I$ isomorphism types, if pruning renders $U$ types indistinguishable, there exists a strict upper bound on the classification accuracy. The training process cannot recover from this loss—which represents an unacceptable risk in safety-critical scenarios like molecular property prediction.

## Key Experimental Results

### Experimental Setup
- **Architectures**: GIN and GCN
- **Datasets**: 10 standard graph classification benchmarks (including molecular datasets like MoleculeNet)
- **Total Runs**: Over **13,500 runs**
- **Expressiveness Metric $\tau$**: The ratio of non-isomorphic graph pairs that remain distinguishable ($\tau_{pre}$ before training, $\tau_{post}$ after training)
- **Winning Criterion**: Sparse models with an accuracy drop of no more than 5% are considered winning tickets
- **Total Computation Time**: Approximately 8 weeks

### Table 1: Transition Probabilities of Expressiveness

| Pruning Ratio | $P(\tau_{post} > \tau_{pre})$ | Meaning |
|---|---|---|
| Low (<30%) | Very low (single-digit %) | Training barely improves expressiveness |
| Medium (30-60%) | Even lower | The downward trend in expressiveness is more pronounced |
| High (>60%) | Near 0 | Large and irreversible degradation of expressiveness |

**Core Conclusion**: The probability of a GNN recovering from a low $\tau_{pre}$ to a high $\tau_{post}$ is extremely low. Training cannot compensate for the expressiveness loss caused by pruning.

### Figure 2+3: Relationship between Winning Probability and Expressiveness

| Analytical Dimension | Results |
|---|---|
| Winning Probability vs. Expressiveness Threshold $\vartheta$ | Monotonically increasing: Higher expressiveness yields a larger winning probability |
| Pearson Correlation $\tau_{pre}$ vs. Post-training Accuracy (GIN) | Positively correlated, statistically significant |
| Pearson Correlation $\tau_{pre}$ vs. Post-training Accuracy (GCN) | Positively correlated, statistically significant |
| $\tau_{pre}$ vs. $\tau_{post}$ Scatter plot (Figure 4) | All points lie below the diagonal: $\tau_{post} \leq \tau_{pre}$ |

Across all 10 datasets and different pruning ratios, the higher the initialization expressiveness, the greater the probability of becoming a winning ticket.

## Key Findings

1. **Expressiveness is Key to Winning**: Over 13,500+ experiments consistently show that sparse subnetworks that preserve 1-WL expressiveness at initialization are significantly more likely to become winning tickets.
2. **Training Does Not Compensate for Pruning Loss**: $\tau_{post} \leq \tau_{pre}$ holds across all experiments, meaning the expressiveness loss from improper pruning is irreversible.
3. **Positive Correlation between Expressiveness and Performance**: Pearson correlation analysis confirms that pre-training expressiveness is positively correlated with post-training accuracy.
4. **Warn for Drug Discovery**: Improper pruning in molecular property prediction leads to indistinguishable activity cliffs and stereoisomers, potentially posing safety hazards.

## Highlights & Insights

- **Theoretical Pioneering**: Formally connects GNN expressiveness (at the WL-test level) with LTH for the first time, proposing the SELTH hypothesis as a GNN-specific theoretical extension and filling a theoretical gap of LTH in GNNs.
- **Concept of Critical Paths**: Shifts the discussion of pruning from "which weights can be removed" to "which computational paths must be preserved to maintain graph distinction capabilities."
- **Practical Guidance**: Provides theoretical direction for GNN pruning—pruning strategies should check if they break the injectivity of MLPs, which is more principled than random or magnitude-based pruning.
- **Scale of Experiments**: Validated extensively with over 13,500 experimental runs across 10 datasets, multiple sparsity levels, and two architectures.

## Limitations & Future Work

- **Limited Architecture Coverage**: Only validated GIN and GCN, without covering attention-based GNNs like GAT or max/min aggregation architectures (as consistently noted by reviewers).
- **Strong Theoretical Assumptions**: Assumes injective, continuously differentiable, zero-fixed activation functions; although experiments indicate it holds for ReLU, formal guarantees are still lacking.
- **Insufficient Practicality of Lemma 3.6**: Requires full knowledge of all isomorphism types in the dataset and assumes a uniform class distribution, which is hard to satisfy in practice.
- **Lack of Practical Pruning Algorithms**: The paper focuses on theoretical analysis and does not propose a complete expressiveness-aware pruning method (only a simple idea was described in the rebuttal).
- **Graph Classification Only**: Application to node classification and other scenarios requires additional adaptations.
- **Hurried Experimental Descriptions**: Reviewers consistently pointed out that the readability and clarity of the experimental section could be improved.

## Related Work & Insights

| Dimension | Ours (SELTH) | Classic LTH/SLTH | Existing GNN LTH |
|---|---|---|---|
| Theoretical Level | Expressiveness preservation condition + WL equivalence | Subnetwork existence proof | Empirical validation only |
| Focus of Attention | Graph discrimination capability (expressiveness) | Function approximation capability | Parameter / computational efficiency |
| Pruning Target | MLP parameter weights | Model parameters | Graph structure + Model parameters |
| Application Example | Drug discovery (activity cliffs) | CV/NLP | Molecular graphs / Social networks |

**Insights**: The concept of expressiveness preservation can be extended to design better GNN pruning algorithms—checking aggregation function injectivity after pruning each layer to choose configurations that maintain the most graph-distinguishing pairs. "WL Go Indifferent" from the same author group (KDD 2024) analyzed the impact of bit-flip attacks on GNN expressiveness, forming a complementary attack/defense perspective with this work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Formally connects WL expressiveness with LTH for the first time; the SELTH hypothesis is a brand new theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Large scale with 13,500+ experiments, but limited to two architectures and lacks comparison with practical algorithms.
- Writing Quality: ⭐⭐⭐ — Theoretically rigorous but notation-heavy; the experimental section is described in a rushed manner (consistent feedback from reviewers).
- Value: ⭐⭐⭐⭐ — Provides an important theoretical foundation for GNN pruning, with practical guidance value in safety-critical drug discovery fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic and Expressive Variation in Image Captions Across Languages](../../CVPR2025/computational_biology/semantic_and_expressive_variations_in_image_captions_across_languages.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](../../NeurIPS2025/computational_biology/why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[CVPR 2026\] FEAST: Fully Connected Expressive Attention for Spatial Transcriptomics](../../CVPR2026/computational_biology/feast_fully_connected_expressive_attention_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
