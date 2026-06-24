---
title: >-
  [Paper Note] Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity
description: >-
  [AAAI 2026][LLM Efficiency][Higher-order GNN] Co-Sparsify proposes a connectivity-aware sparsification framework that restricts 3-node interactions to biconnected components and 2-node interactions to connected components, eliminating provably redundant computation. It preserves full 2-FWL expressivity while substantially improving efficiency, achieving state-of-the-art results on synthetic substructure counting tasks and benchmarks including ZINC and QM9.
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "Higher-order GNN"
  - "2-FWL"
  - "graph sparsification"
  - "biconnected components"
  - "expressivity preservation"
date: 2026-05-08
content_hash: 595fcbe979347191
---

# Connectivity-Guided Sparsification of 2-FWL GNNs Preserving Full Expressivity

**Conference**: AAAI 2026
**arXiv**: [2511.12838](https://arxiv.org/abs/2511.12838)  
**Code**: [Available](https://github.com/RongqinChen/HOGNN-Sparsify)  
**Area**: LLM Efficiency
**Keywords**: Higher-order GNN, 2-FWL, graph sparsification, biconnected components, expressivity preservation

## TL;DR

Co-Sparsify proposes a connectivity-aware sparsification framework that restricts 3-node interactions to biconnected components and 2-node interactions to connected components, eliminating provably redundant computation. It preserves full 2-FWL expressivity while substantially improving efficiency, achieving state-of-the-art results on synthetic substructure counting tasks and benchmarks including ZINC and QM9.

## Background & Motivation

**Background**: Standard message-passing GNNs (GCN, GIN, GAT, etc.) are bounded in expressivity by the 1-WL test. Higher-order GNNs (HOGNNs) improve expressivity through $k$-tuple message passing under the $k$-WL/$k$-FWL hierarchy, but incur $O(n^k)$ computational complexity. 2-FWL GNNs strike a balance between practicality and expressivity—matching 3-WL power—yet still require $O(n^3)$ computation and $O(n^2)$ memory.

**Limitations of Prior Work**: Existing efficiency-oriented methods (subgraph sampling ESAN, set reduction KCSetGNN, local aggregation 1-2-3-GNN) all trade expressivity for efficiency. A natural question arises: can efficiency be improved by eliminating only computations that are redundant with respect to expressivity?

**Key Insight**: 3-node interactions $(u, t, v)$ are expressively necessary only within biconnected components. By Menger's theorem, any two nodes within a biconnected component are connected by at least two internally disjoint paths, and 3-node interactions are required to detect such path diversity. Outside biconnected components, 3-node interactions are redundant—2-node interactions suffice when cut vertices separate nodes, and structural information for disconnected node pairs is captured by the readout.

## Method

### Overall Architecture

Co-Sparsify is a connectivity-aware sparsification scheme consisting of three steps:

1. **Preprocessing**: Connected components are identified via BFS, and biconnected components via block-cut tree decomposition, in $O(n+m)$ time.
2. **Sparse Neighborhood Construction**: For each node pair $(u, v)$, a Co-Sparsified neighborhood set is defined.
3. **Sparse Message Passing**: 2-FWL updates are performed only over necessary interactions.

### Key Designs

**1. Connectivity-Guided Sparsification Principles**

Three core lemmas underpin the correctness of sparsification:

- **Lemma 1 (Necessity of 3-node interactions within biconnected components)**: If $u$ and $v$ belong to the same biconnected component, Menger's theorem guarantees at least two internally disjoint paths between them. Detecting this path diversity requires 3-node interactions; 2-node interactions are insufficient.
- **Lemma 2 (Redundancy of 3-node interactions across cut vertices)**: If all paths from $u$ to $v$ through $t$ pass through a cut vertex $c$ that separates $t$ from $\{u, v\}$, then the 3-node interaction $(u, t, v)$ is fully expressible via 2-node interactions $(u, t)$ and $(t, v)$, and its removal does not affect 2-FWL expressivity.
- **Lemma 3 (Irrelevance of interactions across disconnected components)**: Node pairs from different connected components share no paths; their structural independence is implicitly encoded by the readout function and requires no explicit modeling.

**2. Co-Sparsified Neighborhood Construction**

The sparsified neighborhood set for node pair $(u, v)$ is defined as follows:

- If $u$ and $v$ belong to different connected components: the neighborhood is empty.
- If $u$ and $v$ belong to the same connected component $C$:
    - **3-node interactions**: $((u,t),(t,v))$ is included only when $u$, $t$, and $v$ all reside in the same biconnected block $B$.
    - **2-node interactions**: $((u,u),(u,v))$ and $((u,v),(v,v))$ are included for node-to-pair information propagation.
    - For $u \neq v$, $((v,u),(u,v))$ is included in the neighborhood of the self-pair $(v,v)$ to propagate pair-to-node information.
    - Self-pairs $(u,u)$ always include $((u,u),(u,u))$ to ensure non-empty neighborhoods.

**3. Co-Sparsified Message Passing**

Initial representations follow the standard 2-FWL formulation, combining node features, edge features, and structural encodings (RRWP). At layer $l$, aggregation for $(u, v)$ within the same connected component iterates only over intermediate nodes $t$ in the sparsified neighborhood, rather than all of $V$. Disconnected pairs are not updated.

**4. Multi-Level Readout**

- **Node-level**: Aggregates all incident pair representations within the connected component $C$ containing $v$.
- **Graph-level**: Self-pair and off-diagonal pair representations are aggregated per connected component $C$, then aggregated across all components to obtain the graph representation.

**5. Expressivity Equivalence (Theorem 4)**

Under injective aggregation and consistent initialization, Co-Sparsified 2-FWL GNNs detect substructures with the same capability as standard 2-FWL GNNs. The proof relies on the fact that all 3-node interactions are preserved when the query subgraph $Q$ lies within a biconnected component; when $Q$ spans multiple blocks, its structure decomposes into independent 2-node interactions; and disconnected queries are handled by the readout.

**6. Computational Efficiency**

- Standard 2-FWL: $O(n^2)$ pairs and $O(n^3)$ triple interactions.
- Co-Sparsify: reduced to $O(\sum n_i^2)$ over connected components and $O(\sum n_{B_j}^3)$ over biconnected components.
- On graphs with sparse higher-order connectivity (e.g., molecular graphs), complexity is reduced from cubic to sub-cubic.

### Loss & Training

Co-Sparsify is applied to the PPGN architecture using element-wise multiplication as the pair interaction function. An $\ell_1$ loss is used for molecular property regression on QM9, and MAE loss for graph-level regression on ZINC. Preprocessing (block-cut tree decomposition) takes approximately 1 ms per graph.

## Key Experimental Results

### QM9 Molecular Property Prediction

| Model | mu | alpha | e_HOMO | e_LUMO | delta_e | ZPVE |
|---|---|---|---|---|---|---|
| PPGN+RRWP | 0.332 | 0.200 | 0.00236 | 0.00231 | 0.00327 | 0.00012 |
| **CoSp-PPGN+RRWP** | **0.321** | **0.183** | **0.00214** | **0.00210** | **0.00299** | **0.00012** |
| N2-GNN | 0.333 | 0.193 | 0.00217 | 0.00210 | 0.00304 | 0.00013 |

CoSp-PPGN+RRWP achieves top-2 performance on 10 out of 12 targets.

### ZINC Graph Regression

| Model | Params | ZINC-Subset MAE | ZINC-Full MAE |
|---|---|---|---|
| GRIT | ~500k | 0.059±0.002 | 0.023±0.001 |
| N2-GNN | 316k/414k | 0.059±0.002 | 0.022±0.002 |
| PPGN+RRWP | 478K | 0.055±0.002 | 0.020±0.002 |
| **CoSp-PPGN+RRWP** | **478K** | **0.050±0.001** | **0.018±0.002** |

### Ablation Study (Substructure Counting, Normalized MAE)

| Substructure | PPGN | CoSp-PPGN |
|---|---|---|
| Tailed Triangle | 0.0025 | **0.0016** |
| 4-Cycles | 0.0024 | **0.0007** |
| 5-Cycles | 0.0039 | **0.0032** |
| 6-Cycles | 0.0075 | **0.0056** |

CoSp-PPGN matches or surpasses PPGN on all substructure counting tasks.

### Key Findings

- Runtime reduction of 13–60% (ZINC-subset: 9.3s → 7.9s/epoch; QM9: 97.1s → 60.7s/epoch).
- Memory reduction of 12–52% (ZINC-subset: 3.7 → 3.0 GB; QM9: 6.4 → 4.2 GB).
- Preprocessing overhead is negligible (~1 ms/graph).
- State-of-the-art performance on TUD classification benchmarks (FRANKENSTEIN 77.65%, NCI1 82.87%).

## Highlights & Insights

- This is the first sparsification method for HOGNNs that provides a provable expressivity-preservation guarantee—without relying on approximation, sampling, or heuristic pruning.
- The core insight is elegant: it bridges classical graph-theoretic connectivity theory (Menger's theorem, block-cut trees) with GNN expressivity analysis.
- The implementation is clean: only $O(n+m)$ preprocessing is required, after which message passing follows the standard 2-FWL format.
- Expressivity and efficiency need not be in opposition—structural insight enables both simultaneously.

## Limitations & Future Work

- On tasks requiring long-range dependencies (e.g., Peptides-struct), performance degrades due to over-squashing in HOGNNs.
- The framework is restricted to the 2-FWL level and has not been extended to higher orders ($k > 2$).
- For highly biconnected graphs (e.g., complete graphs), sparsification yields limited gains.
- A localized variant (restricting shortest path distance to $\leq 4$) performs better on certain tasks, but a principled adaptive pruning mechanism is lacking.

## Related Work & Insights

- **PPGN**: A batched implementation of 2-FWL GNNs and the backbone architecture for Co-Sparsify.
- **ESAN / KCSetGNN / 1-2-3-GNN**: Prior efficiency methods all sacrifice expressivity.
- **N2-GNN**: Redesigns channels in Folklore WL; Co-Sparsify achieves better performance with stronger theoretical guarantees.
- **Block-cut tree**: A classical graph-theoretic tool (Hopcroft–Tarjan algorithm) that Co-Sparsify creatively repurposes for GNN sparsification.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 5 |
| Theoretical Depth | 5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 5 |
| Practicality | 4 |
| Overall | 4.6 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head](../../ICLR2026/llm_efficiency/mhla_restoring_expressivity_of_linear_attention_via_token-level_multi-head.md)
- [\[ACL 2026\] StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference](../../ACL2026/llm_efficiency/structkv_preserving_the_structural_skeleton_for_scalable_long-context_inference.md)
- [\[ICLR 2026\] Unlocking Full Efficiency of Token Filtering in Large Language Model Training](../../ICLR2026/llm_efficiency/unlocking_full_efficiency_of_token_filtering_in_large_language_model_training.md)
- [\[ICML 2026\] TuneAhead: Predicting Fine-tuning Performance Before Full Training Begins](../../ICML2026/llm_efficiency/tuneahead_predicting_fine-tuning_performance_before_full_training_begins.md)
- [\[ICLR 2026\] Influence-Preserving Proxies for Gradient-Based Data Selection in LLM Fine-Tuning](../../ICLR2026/llm_efficiency/influence-preserving_proxies_for_gradient-based_data_selection_in_llm_finetuning.md)

</div>

<!-- RELATED:END -->
