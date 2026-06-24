---
title: >-
  [Paper Note] Bayesian Network Structural Consensus via Greedy Min-Cut Analysis
description: >-
  [AAAI 2026][Bayesian network] This paper proposes the MCBNC algorithm, which quantifies the structural support of each edge via min-cut analysis and embeds this scoring into the backward phase of Greedy Equivalence Search (GES) to iteratively prune redundant edges from a fused Bayesian network. The method produces sparser and more accurate consensus structures without accessing any data, making it well-suited for federated learning scenarios.
tags:
  - "AAAI 2026"
  - "Bayesian network"
  - "structural fusion"
  - "consensus"
  - "min-cut"
  - "max-flow"
  - "greedy equivalence search"
  - "federated learning"
  - "treewidth"
date: 2026-05-08
content_hash: 586374f06349878c
---

# Bayesian Network Structural Consensus via Greedy Min-Cut Analysis

**Conference**: AAAI 2026
**arXiv**: [2504.00467](https://arxiv.org/abs/2504.00467)  
**Code**: [https://github.com/ptorrijos99/BayesFL](https://github.com/ptorrijos99/BayesFL)  
**Area**: Others (Bayesian Networks / Federated Learning)
**Keywords**: Bayesian network, structural fusion, consensus, min-cut, max-flow, greedy equivalence search, federated learning, treewidth

## TL;DR

This paper proposes the MCBNC algorithm, which quantifies the structural support of each edge via min-cut analysis and embeds this scoring into the backward phase of Greedy Equivalence Search (GES) to iteratively prune redundant edges from a fused Bayesian network. The method produces sparser and more accurate consensus structures without accessing any data, making it well-suited for federated learning scenarios.

## Background & Motivation

- **Need for Bayesian network fusion**: In federated learning and expert knowledge aggregation, multiple parties each learn a local Bayesian network (BN) and must aggregate these structures into a single consensus DAG. Structural fusion is central to this process and must be performed without sharing raw data.
- **Fatal flaw of naive fusion**: Classical unconstrained fusion methods—taking the union of edges across all input DAGs under a heuristic node ordering—guarantee the I-map property but yield extremely dense graphs. Since exact inference complexity is $O(n \cdot k^{tw+1})$, treewidth ($tw$) inflation renders inference completely infeasible; for example, the treewidth of the Win95pts network explodes from approximately 10 to approximately 20.
- **Limitations of existing pruning approaches**: (1) Genetic-algorithm-based methods (Torrijos 2024, 2025) can enforce treewidth constraints or optimize objective functions, but are computationally expensive and require a pre-specified treewidth upper bound—setting it too low discards important edges, while setting it too high leaves inference infeasible. (2) Existing greedy heuristics rely only on edge frequency, ignoring richer structural properties, and cannot be used in isolation.
- **Key insight**: If removing an edge $u \to v$ still leaves a large number of alternative paths between $u$ and $v$ in the moralized ancestral subgraphs of most input DAGs (i.e., the min-cut is large), then the edge is redundant. Conversely, a small (or zero) min-cut indicates that the edge is critical for preserving the dependency structure. This observation naturally connects edge importance quantification with classical max-flow/min-cut theory.
- **Suitability for federated learning**: The method requires only structural information and no data, making it inherently compatible with aggregating locally learned DAGs across federated clients. The sole parameter $\theta$ can be determined post hoc using only structural information.

## Method

### Overall Architecture

MCBNC proceeds in three stages:

1. **Construct the unconstrained fusion graph $G^+$**: Given $r$ input DAGs $\{G_i\}_{i=1}^r$ over a shared variable set $V$, each DAG is first converted to $G_i^\sigma$ under a heuristic node ordering $\sigma$, and the edge union $E^+ = \bigcup_i E_i^\sigma$ yields the fused DAG $G^+$.
2. **Convert to CPDAG and iteratively prune**: $G^+$ is converted to its completed partially directed acyclic graph (CPDAG) $\mathcal{G}^+$, and edges are iteratively removed in the Markov equivalence class space by eliminating the least critical edge, as scored by the min-cut-based scoring function $\Psi$.
3. **Threshold control and output**: The process terminates when the criticality score of all remaining edges exceeds the threshold $\theta$; the final CPDAG is then converted back to a DAG to produce the consensus structure $G^*$.

The overall procedure is equivalent to replacing the likelihood-based scoring function in the backward elimination step (BES) of GES with a max-flow-based structural score, enabling structure pruning without any data access.

### Key Design 1: Min-Cut-Based Edge Criticality Score

- **Function**: Computes a structural criticality score $\Psi_{(u \to v)}^H$ for each edge $e = (u \to v)$ in the fused graph.
- **Core computation**: For each input DAG $G_i$: (a) extract the ancestral subgraph of $\{u, v\} \cup H$; (b) moralize this subgraph; (c) remove nodes in the conditioning set $H$ to obtain the conditioned graph $\tilde{G}_i^H$; (d) compute the min-cut size $|S_i^H|$ between $u$ and $v$ in $\tilde{G}_i^H$ using the Ford-Fulkerson algorithm. The final score is the average min-cut across all input graphs: $\Psi_{(u \to v)}^H = \frac{1}{r} \sum_{i=1}^r |S_i^H|$.
- **Interpretation**: A lower score indicates weaker structural support for the edge across the input networks—most input networks lack alternative paths between $u$ and $v$, so removing this edge does not disrupt the consensus dependency structure. A higher score indicates abundant alternative paths, suggesting redundancy.
- **Design motivation**: By applying the max-flow/min-cut theorem, the structural contribution of each edge is quantified as a continuous score rather than a simple frequency count, capturing richer topological information.

### Key Design 2: Conditioning Set Enumeration and BES Delete Operator

- **Function**: For each edge, enumerate all valid conditioning sets $H \subseteq \mathcal{N}_{uv}$ (of size at most $k_{\max}$) and select the pair $(e^*, H^*)$ that minimizes the criticality score.
- **Mechanism**: The valid conditioning set $\mathcal{N}_{uv}$ is defined as nodes that are simultaneously parents of $v$ and connected to $u$ by an undirected edge (following Chickering's BES delete condition). For each candidate subset $H$, $\Psi$ is computed, and the globally lowest-scoring pair is selected. The BES Delete operator is applied to remove the edge, ensuring that all operations remain within the Markov equivalence class.
- **Design motivation**: Directly inheriting the GES/BES theoretical framework guarantees correctness—after each deletion, the CPDAG is automatically updated to maintain equivalence class consistency. Although the enumeration of conditioning sets is exponential in theory, $|\mathcal{N}_{uv}|$ is typically small in practice and shrinks further as pruning proceeds.

### Key Design 3: Post Hoc Adaptive Threshold $\theta$ Selection

- **Function**: Proposes a strategy for selecting $\theta$ without requiring data or access to the ground-truth network.
- **Mechanism**: A single run of MCBNC produces the complete pruning trajectory $\{G^*(\theta)\}$. The value of $\theta$ is chosen to minimize the average SMHD (Structural Moral Hamming Distance) between the consensus DAG and the set of input DAGs.
- **Theoretical support**: Experiments demonstrate that the $\theta$ minimizing SMHD relative to input graphs also approximately minimizes SMHD relative to the ground-truth network and yields strong BDeu scores. A Friedman test with Holm post hoc analysis at $\alpha = 0.01$ confirms the statistical significance of this strategy.

### Theoretical Properties

- **Monotonicity (Lemma 1)**: After each edge deletion, the criticality scores of remaining edges are monotonically non-decreasing, ensuring that the greedy process does not defer edges that should have been removed earlier.
- **Complexity (Lemma 3)**: The total time complexity is $O(r \cdot m^3 \cdot 2^{k_{\max}})$ and space complexity is $O(r \cdot m)$, where $m$ is the number of edges in the fused graph. In practice, the $2^{k_{\max}}$ term has limited impact because large conditioning sets rarely arise.

## Key Experimental Results

### Experimental Setup

- **Benchmark networks**: 15 standard Bayesian networks from the bnlearn repository, ranging from 8 nodes (Asia) to 441 nodes (Pigs), with edge counts from 8 to 602.
- **Federated scenario**: For each benchmark network, $r \in \{5, 10, 20, 30, 50, 100\}$ clients are simulated; each client samples 5,000 i.i.d. records from the ground-truth network, learns a local DAG with GES, and then contributes only the structure for fusion.
- **Evaluation metrics**: SMHD (relative to ground-truth / relative to input networks), BDeu score, treewidth.
- **Each configuration is repeated 10 times**, with Friedman + Holm statistical testing.

### Table 1: Statistical Rank Comparison of MCBNC vs. Baselines across 90 Configurations (15 networks × 6 client counts)

| Metric | Method | Mean Rank | p-value (Holm) | Win/Tie/Loss |
|--------|--------|-----------|----------------|--------------|
| SMHD | MCBNC ($G^*$) | **1.40** | — | — |
| SMHD | GES average | 1.91 | $6.95 \times 10^{-4}$ | 61/13/16 |
| SMHD | Unconstrained fusion ($G^+$) | 2.69 | $7.68 \times 10^{-18}$ | 68/17/5 |
| BDeu | GES average | **1.58** | — | — |
| BDeu | MCBNC ($G^*$) | 1.70 | $4.12 \times 10^{-1}$ (n.s.) | 48/9/33 |
| BDeu | Unconstrained fusion ($G^+$) | 2.72 | $3.25 \times 10^{-14}$ | 71/9/10 |

MCBNC ranks first in SMHD and significantly outperforms both baselines; in BDeu it is statistically indistinguishable from GES ($p \approx 0.41$), which directly optimizes data likelihood, demonstrating that MCBNC achieves data-driven learning quality without any data access.

### Table 2: Representative Improvements of MCBNC on Selected Networks (extracted from Fig. 1 and Fig. 3)

| Network | Nodes/Edges | Fusion $G^+$ SMHD | MCBNC Best SMHD | SMHD Gain | $G^+$ Treewidth | MCBNC Treewidth |
|---------|------------|-------------------|----------------|-----------|-----------------|-----------------|
| Win95pts | 76/112 | Far above empty graph | Significantly below input DAG average | ~58.7% | ~20 | ~10 |
| Andes | 223/338 | Extremely high | ~2 orders of magnitude below $G^+$ | >90% | Extremely high | Near ground-truth |
| Diabetes | 413/602 | Extremely high | ~2 orders of magnitude below $G^+$ | >90% | Extremely high | Near ground-truth |
| Alarm | 37/46 | Above input average | Below input average | Significant | High | Near ground-truth |

Gains are most pronounced on large networks (Andes, Diabetes), where SMHD reduction reaches two orders of magnitude. The treewidth reduction restores the feasibility of exact inference.

### Runtime Efficiency

MCBNC is several orders of magnitude faster than genetic-algorithm-based methods. Most computation is concentrated in early iterations (low $\theta$), with per-iteration cost decreasing as pruning progresses. Sensitivity analysis on $k_{\max}$ (tested on the largest Diabetes network with $k_{\max} \in \{0, 1, \ldots, 20\}$) shows that pruning quality is nearly unaffected, confirming that $k_{\max}$ does not require careful tuning.

## Highlights & Insights

- **Elegant scoring function design**: Embedding min-cut analysis into BN structural fusion to quantify edge support is an elegant application of classical graph-theoretic tools to probabilistic graphical models. Each edge is modeled as a capacity unit in a flow network, and its removability is determined by the number of alternative paths in the moralized ancestral subgraph.
- **No data ≈ With data**: The most striking experimental finding is that MCBNC, without accessing any data, achieves significantly better structural accuracy than the input GES networks (which directly optimize data likelihood) and yields statistically indistinguishable BDeu scores. This demonstrates that the consensus information embedded in multiple noisy structures is sufficient to recover dependency structures close to the ground truth.
- **Self-referential threshold selection**: $\theta$ is chosen by minimizing the distance between the consensus graph and the set of input graphs, forming a self-consistent selection mechanism that requires no external reference—a practically valuable property in federated learning settings.
- **Organic treewidth reduction**: Without imposing hard treewidth constraints, the pruning process naturally reduces treewidth to near or below that of the ground-truth network, circumventing the difficult question of choosing an appropriate upper bound.

## Limitations & Future Work

- **Assumes identical variable sets**: All input DAGs must be defined over the same variable set $V$; the method cannot handle heterogeneous settings where different participants observe different subsets of variables.
- **Non-i.i.d. distributions not considered**: Federated learning commonly involves heterogeneous client data distributions; all clients in this paper's experiments sample from the same distribution, and robustness to distribution shift is unexplored.
- **Greedy search limitations**: The greedy nature of BES means the algorithm may converge to local optima; different deletion orders may yield different final structures. Although variance across repeated runs is low, no global optimality guarantee exists.
- **Synthetic-to-real gap**: All benchmarks are generated by sampling from known ground-truth structures; validation in realistic settings where the true DAG is unknown has not been performed.
- **Scalability bottleneck**: The theoretical complexity of $O(r \cdot m^3 \cdot 2^{k_{\max}})$ may still be limiting for very large networks, particularly when the fused graph is extremely dense.

## Related Work & Insights

- **Bayesian network fusion**: Peña (2011) and Puerta et al. (2021) represent classical structural fusion approaches; the unconstrained fusion $G^+$ in this paper is based on the heuristic ordering of the latter. Torrijos et al. (2024, 2025) employ genetic algorithms to optimize fused structures under fixed treewidth constraints; MCBNC provides an alternative that is orders of magnitude more efficient.
- **GES and BES**: Chickering's (2002) greedy equivalence search is the canonical algorithm for BN structure learning. MCBNC reuses only its backward phase (BES) and the Delete operator, replacing the data-driven BDeu score with a purely structural scoring function.
- **Max-flow applications in other ML domains**: Min-cut/max-flow has been widely applied in image segmentation (graph cuts) and community detection; applying it to structural aggregation in probabilistic graphical models is novel.
- **Implications for federated BN learning**: This work validates the "learn locally, fuse structurally" paradigm for federated BN learning, demonstrating that transmitting only structures—not data—can yield consensus quality superior to individual learning, providing both theoretical and empirical support for federated causal discovery.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing min-cut analysis as a scoring function for BN structural fusion is an entirely new approach; the reuse of the GES/BES framework is natural and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 15 standard networks × 6 client counts × 10 repetitions, with rigorous statistical testing and comprehensive ablation and sensitivity analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured, with complete theoretical analysis, illustrative visualizations, and detailed technical appendices.
- **Value**: ⭐⭐⭐⭐ — Addresses a critical practical problem in BN fusion (treewidth inflation), extends naturally to federated learning, and is accompanied by publicly available code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](../../ICML2025/others/continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)
- [\[ICLR 2026\] Federated ADMM from Bayesian Duality](../../ICLR2026/others/federated_admm_from_bayesian_duality.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)

</div>

<!-- RELATED:END -->
