---
title: >-
  [Paper Note] Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis
description: >-
  [ICML2025][Graph Learning][Graph Prompt] Provides the first comprehensive theoretical framework for Graph Prompts from the perspective of "data operation": proves that prompts can map the original graph to a "bridge graph" by simulating graph data transformations to adapt the frozen model to downstream tasks, and derives the error upper bounds and distributions in both single-graph and multi-graph scenarios.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Graph Prompt"
  - "Bridge Set"
  - "Data Operation"
  - "Theoretical Analysis"
  - "Error Upper Bound"
date: 2026-05-08
content_hash: 6117929b186ffe51
---

# Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis

**Conference**: ICML2025  
**arXiv**: [2410.01635](https://arxiv.org/abs/2410.01635)  
**Code**: [GitHub - dgpw](https://github.com/qunzhongwang/dgpw)  
**Area**: Graph Learning  
**Keywords**: Graph Prompt, Bridge Set, Data Operation, Theoretical Analysis, Error Upper Bound

## TL;DR
Provides the first comprehensive theoretical framework for Graph Prompts from the perspective of "data operation": proves that prompts can map the original graph to a "bridge graph" by simulating graph data transformations to adapt the frozen model to downstream tasks, and derives the error upper bounds and distributions in both single-graph and multi-graph scenarios.

## Background & Motivation

### Key Challenge

**Key Challenge**: Graph Prompt shifts "pre-train + fine-tune" to "pre-train + prompt": freezing the model and learning prompts to attach to the input graph. While empirically effective, the core question remains unanswered: **Why does Graph Prompt work?**

### Limitations of Prior Work

- GPF (2022) proved that prompts can precisely simulate graph operations under linear GNNs.
- However, real-world GNNs possess non-linear layers, making the linearity assumption invalid.
- The relationship between error and non-linearity/prompt design has only been empirically observed without theoretical proof.

### Proposed Solution

**Goal**: ### Core Concept: Bridge Graph

**Theorem 1**: For any optimal downstream task function $C$ and original graph $G_{ori}$, there always exists a bridge graph $G_{bri}$ such that $F_{\theta^*}(G_{bri}) = C(G_{ori})$.


## Method

### Core Concept: Bridge Graph

**Theorem 1**: For any optimal downstream task function $C$ and original graph $G_{ori}$, there always exists a bridge graph $G_{bri}$ such that $F_{\theta^*}(G_{bri}) = C(G_{ori})$.

### Bridge Set and $\epsilon$-extended Bridge Set

- **Bridge Set** $B_G = \{G_p \mid F_{\theta^*}(G_p) = C(G)\}$
- **$\epsilon$-extended Bridge Set**: A relaxed version allowing errors of at most $\epsilon^*$.

### Error Upper Bound for Single Graph

**Theorem 3/4**: In non-linear GCNs, if the weight matrix is row full-rank, both GPF and All-in-One can be mapped precisely to $B_G$.

**Theorem 5 (Non-full-rank)**: $\epsilon \leq \zeta(\theta) \cdot \kappa(G)$, where the error is co-determined by the model rank deficiency and graph complexity.

### Error Upper Bound for Multi-Graph

**Theorem 7**: The RMSE upper bound is $\epsilon^* = \sqrt{\sum_{i=k+1}^{M} \lambda_i / M}$, which is determined by the eigenvalues of the correlation matrix in the downstream solution space. Key conclusion: Since the eigenvalues decay exponentially, a small number of prompt tokens is sufficient.

### Error Distribution

**Theorem 8**: Under specific conditions, $\epsilon$ follows a Chi-distribution $\chi_r$.

### Extension to Non-linear Models

**Theorem 9**: All conclusions apply equally to attention-based models such as GAT.

## Key Experimental Results

### Full-rank Single Graph Convergence


### Main Results

| Model | GPF Loss | All-in-One Loss |
|------|---------|----------------|
| GCN | → 0 | → 0 |
| GAT | → 0 | → 0 |

### Impact of Non-full-rank Factors


### Ablation Study

| Factor | Impact on Error Upper Bound |
|------|--------------|
| Weight matrix rank ↓ | Error ↑ |
| Feature dimension ↑ | Error ↑ |
| Graph size ↑ | Error ↑ |
| Model layers ↑ | Error ↓ |
| Prompt complexity ↑ | Error ↓ |

### Error Distribution Verification
Chi-distribution p-value = 0.65, which is much higher than Gamma(0.23), Chi²(0.04), and Exponential(0.01).

### Multi-graph Prompt Tokens Experiment
The error plateaus when the number of tokens exceeds 10, which aligns with the eigenvalue decay predicted in Theorem 7.

### Key Findings
- The main components/modules contribute to the most critical performance improvements.


## Highlights & Insights

1. Provides the first comprehensive theoretical framework for Graph Prompt, addressing "why it works".
2. The concept of "bridge graph" is elegant: unifying data transformation and task adaptation under a set theory framework.
3. Eigenvalue decay explains the empirical experience that "a small number of tokens is sufficient".
4. Extension from linear to non-linear models demonstrates theoretical robustness.
5. The discovery of Chi-distribution for errors provides a tool for estimating performance in practice.

## Limitations & Future Work

1. The full-rank condition requires a special initialization to be guaranteed and is not always naturally satisfied.
2. Mainly covers GCN/GAT; more complex architectures require separate analysis.
3. The error upper bounds might be loose; practical errors are usually much lower.
4. The existence proof of "bridge graph" is non-constructive and does not directly guide prompt design.
5. The eigenvalue decay assumption may not hold for heterogeneous datasets.

## Related Work & Insights

- Provides theoretical proof and error analysis for GPF/All-in-One.
- Analogy with NLP prompt: "bridge graph" is akin to finding an appropriate soft prompt to align model outputs.
- Insight: Use eigenvalue analysis to guide prompt complexity selection and design full-rank pre-training strategies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (5.0/5) — First complete theoretical framework
- Experimental Thoroughness: ⭐⭐⭐⭐☆ (4.0/5) — Primarily synthetic data
- Writing Quality: ⭐⭐⭐⭐⭐ (5.0/5) — Step-by-step theorem organization
- Value: ⭐⭐⭐⭐⭐ (5.0/5) — Lays the foundation for the Graph Prompt field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICML 2025\] Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via Contextual Stochastic Block Models](graph_attention_is_not_always_beneficial_a_theoretical_analysis_of_graph_attenti.md)
- [\[ICML 2026\] Message Tuning Outshines Graph Prompt Tuning: A Prismatic Space Perspective](../../ICML2026/graph_learning/message_tuning_outshines_graph_prompt_tuning_a_prismatic_space_perspective.md)
- [\[ICML 2025\] Diss-l-ECT: Dissecting Graph Data with Local Euler Characteristic Transforms](diss-l-ect_dissecting_graph_data_with_local_euler_characteristic_transforms.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)

</div>

<!-- RELATED:END -->
