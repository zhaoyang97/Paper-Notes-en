---
title: >-
  [Paper Note] EvA: Evolutionary Attacks on Graphs
description: >-
  [ICLR 2026][Graph Learning][Paper Note] Ours utilizes a carefully designed genetic algorithm to search for adversarial perturbations directly in the discrete edge-flip space, bypassing gradient relaxation and differentiable proxy losses. It achieves an average accuracy drop of ~11% more than the SOTA PRBCD in node classification attacks and implements the fi
tags:
  - ICLR 2026
  - Graph Learning
date: 2026-05-08
content_hash: 4151453b009c00c4
---
# EvA: Evolutionary Attacks on Graphs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=EzXzGRngYb](https://openreview.net/forum?id=EzXzGRngYb)  
**Code**: [github.com/UoC-tail/EvA](https://github.com/UoC-tail/EvA)  
**Area**: Graph Learning / Graph Adversarial Attack / Evolutionary Algorithms  
**Keywords**: GNN, Adversarial Attack, Evolutionary Algorithms, Combinatorial Optimization, Black-box Attack, Conformal Prediction, Robustness Certificates  

## TL;DR
Ours utilizes a carefully designed genetic algorithm to search for adversarial perturbations directly in the discrete edge-flip space, bypassing gradient relaxation and differentiable proxy losses. It achieves an average accuracy drop of ~11% more than the SOTA PRBCD in node classification attacks and implements the first graph structural attacks on conformal prediction and robustness certificates.

## Background & Motivation
**Background**: Graph Neural Networks (GNNs) are extremely vulnerable to structural perturbations; adding or deleting just a few edges can drop accuracy below that of an MLP, which ignores graph structure. The vast majority of graph adversarial attacks are gradient-based—represented by PRBCD / LRBCD—which calculate gradients of a tanh-margin loss with respect to the adjacency matrix and use Block Coordinate Descent (BCD) to sample which edges to flip.

**Limitations of Prior Work**: This paper lists seven "deadly sins" of gradient attacks. (i) The original problem is a combinatorial optimization over $\{0,1\}$, but gradient methods must relax this to $[0,1]$, leading to solutions far from optimal; (ii) Gradients only provide local information and fail to reflect the true loss landscape after edge flips (Fig.1 Left: gradient suggests a decrease, but loss actually increases); (iii) Gradients only characterize the effect of "flipping a single edge," whereas simultaneous flips can have the opposite effect (Fig.1 Middle); (iv) Accuracy itself is non-differentiable, requiring sub-optimal differentiable proxy losses like cross-entropy; (v) White-box access is required, or a surrogate model must be trained; (vi) Defenders can create a "false sense of security" by obfuscating gradients; (vii) While the adjacency matrix is sparse, its gradient is not, resulting in memory growth quadratic to the number of nodes.

**Key Challenge**: The problem to be solved is fundamentally discrete combinatorial optimization. Gradient methods, to be differentiable, are forced into relaxation, proxy losses, and white-box requirements—compromises that sacrifice both optimality and applicability.

**Goal**: To solve the original combinatorial optimization problem directly in the binary matrix space without relying on any gradient information, achieving a black-box, arbitrary-target approach scalable to large graphs.

**Core Idea**: **Search rather than derivation**—revitalizing the forgotten Evolutionary/Genetic Algorithms (GA). Early GA attacks (e.g., Dai et al. 2018) were outperformed by gradient methods due to poor fitness functions and mutation strategies. Ours proves that by carefully redesigning each component of the meta-heuristic pipeline (fitness function, encoding, mutation, initialization, divide-and-conquer), pure search can significantly outperform SOTA gradient attacks.

## Method

### Overall Architecture
EvA models "which edges to flip" as a discrete search via a genetic algorithm: each individual in the population is a set of edge indices to be flipped, evolving through fitness evaluation, tournament selection, crossover, and mutation. Five key modifications are made to this framework: sparse encoding to save memory, using accuracy directly as fitness, restricting search to the receptive field of attack targets, adaptive targeted mutation, and a divide-and-conquer strategy for large graphs. The black-box nature also allows substituting targets with conformal prediction or certificate attacks.

```mermaid
flowchart TD
    A[Initial Population: Each individual = δ edge indices<br/>At least one end in target Vatt] --> B[Fitness Evaluation<br/>Global: perturbed accuracy<br/>Targeted: tanh-margin]
    B --> C[Tournament Selection: ntour takes the best two]
    C --> D[Crossover: Splicing parental segments]
    D --> E[Adaptive Targeted Mutation ATM<br/>New edge restricted to unconquered Vatt nodes]
    E --> F{Max iterations?}
    F -- No --> B
    F -- Yes --> G[Output Optimal Perturbation P]
    H[Large Graphs: D&C splits Vatt<br/>Cumulative attack per subset] -.Enhance.-> A
    I[Sparse Encoding: Store O(δ) indices<br/>instead of O(N²) boolean vectors] -.Throughout.-> A
```

### Key Designs

**1. Sparse Encoding + $O(1)$ Index Mapping: Reducing memory from quadratic to linear.** A naive approach uses an $N^2$-dimensional boolean vector to mark flipped edges, with memory $O(|S|N^2)$, ignoring the sparsity of the adjacency matrix. EvA represents each candidate individual as a "list of indices of edges to be flipped." The individual dimension is exactly the budget $\delta=\lfloor\epsilon\cdot|E[V_{att}:V]|\rfloor$. Each component $z[i]\in\{1,\dots,n(n-1)/2\}$ is a diagonal enumeration index of the upper triangular matrix, converted between 1D indices and 2D undirected edges at $O(1)$ cost via mapping $\Pi$. Memory complexity is reduced to $O(\epsilon\cdot E\cdot P)$ (where $P$ is population size), which respects sparsity and **naturally satisfies global constraints** because the individual length is capped by the budget.

**2. Accuracy as Fitness instead of Differentiable Proxies: The greatest dividend of search.** Gradient methods are forced to use differentiable proxies like cross-entropy or tanh-margin. However, ablation (Fig.2 Left) shows cross-entropy wastes budget on nodes already misclassified; tanh-margin (proposed by Geisler et al. 2023) has better correlation. In EvA, **directly using non-differentiable accuracy as fitness is optimal**. Since search requires no derivatives, this target is "free." Crucially, even when PRBCD uses tanh-margin, it is significantly outperformed by EvA, suggesting the advantage comes not just from loss quality but from GA's ability to escape local optima where PRBCD gets stuck.

**3. Receptive Field Constraints + Adaptive Targeted Mutation (ATM): Focusing the budget.** The core insight is shrinking the search space from $\frac{n(n-1)}{2}$ edges to the receptive field of the target nodes $V_{att}$. Perturbations within the training subgraph are easily recovered by memory, and flips outside the receptive field do not affect $V_{att}$'s prediction. During initialization, every edge is forced to have at least one endpoint in $V_{att}$. Mutation evolves from Uniform Mutation (UM, random global indices) to Targeted Mutation (TM, restricting one end to $V_{att}$) and finally **Adaptive Targeted Mutation (ATM)**. ATM removes nodes from the restricted endpoint list once they are successfully conquered (as further perturbation yields no gain), while still allowing them to connect to other $V_{att}$ nodes to contribute to others' misclassification.

**4. Divide & Conquer (D&C): Shrinking exponential search spaces for large graphs.** On large graphs, the search space expands quadratically. D&C splits $V_{att}=V_1\cup V_2$ and allocates budgets $\delta=\delta_1+\delta_2$ proportionally to their connections. It attacks $V_1$ with $\delta_1$, uses the modified graph as a starting point for $V_2$ with $\delta_2$, and finally merges the results. Standard attacks search in $\binom{n/2}{\delta}$, while D&C only searches $\binom{n/2}{\delta_1}+\binom{n/2}{\delta_2}$—an exponential reduction. While this relaxation might deviate from the global optimum (beneficial for Ogbn-Arxiv with ~8% gain, but negligible for small graphs like CoraML), D&C also enhances PRBCD, serving as a general scalability technique. Additionally, EvA sets **non-differentiable objectives** like conformal prediction coverage or certified ratios as fitness, utilizing "stacked perturbations" and resampling to implement the first structural attacks on conformal sets and robustness certificates.

## Key Experimental Results

### Main Results: Perturbed Accuracy on CoraML under various budgets (Lower is better)

| Attack | 0.01 | 0.02 | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|---|
| DICE | 80.93 | 80.93 | 80.78 | 80.57 | 80.07 |
| PGA | 79.58 | 76.92 | 70.94 | 64.62 | 60.46 |
| PGD | 78.22 | 75.37 | 67.18 | 59.14 | 53.09 |
| GRPCD | 78.07 | 75.08 | 66.76 | 58.29 | 54.80 |
| PRBCD (Prev. SOTA) | 76.44 | 73.17 | 66.48 | 58.51 | 52.67 |
| **EvA** | **74.80** | **68.97** | **52.95** | **41.99** | **37.65** |

At $\epsilon=0.10$, EvA drops accuracy to 41.99%, approximately 17 percentage points lower than the second-best PGD (59.14%), and ~11% lower than the best previous work on average.

### Attacking Robust Defense Models (CoraML, Accuracy, Lower is better)

| Defense | Attack | 0.01 | 0.02 | 0.05 | 0.10 |
|---|---|---|---|---|---|
| GCN-SVD | EvA / PRBCD | **0.70**/0.76 | **0.64**/0.75 | **0.54**/0.73 | **0.41**/0.70 |
| GNNGuard | EvA / PRBCD | **0.71**/0.74 | **0.67**/0.72 | **0.55**/0.70 | **0.45**/0.66 |
| Robust-GCN | EvA / PRBCD | **0.75**/0.77 | **0.70**/0.73 | **0.59**/0.68 | **0.52**/0.63 |
| Soft-Median | EvA / PRBCD | **0.75**/0.77 | **0.72**/0.76 | **0.69**/0.73 | **0.62**/0.68 |

EvA is more penetrative than PRBCD against all robust defenses. On multiple models, $\epsilon\sim0.05$ is sufficient to drop accuracy below that of an MLP, implying that "using the graph structure becomes worse than ignoring it."

### Ablation Study
- **Fitness Function (Fig.2 Left)**: Accuracy > tanh-margin > Cross-Entropy. Cross-entropy wastes budget on already conquered nodes.
- **Mutation Strategy (Fig.2 Middle)**: ATM > TM > UM. Adaptive removal of conquered nodes consistently provides gains.
- **Scalability (Fig.3)**: EvA improves continuously as population size or iterations increase (open-ended search), whereas PRBCD shows little improvement when increasing block size or steps. EvA is more frequently Pareto optimal across time-memory-performance tradeoffs.
- **Local Constraints (Fig.2 Right)**: Even without explicit local constraints, EvA has fewer degree violations than PRBCD, with more scattered perturbations. Explicit local projection using "edge frequency in population" $s(e)=\sum_{s\in S}\mathbb{I}[e\in s]/|S|+u$ can strictly satisfy degree constraints.

## Highlights & Insights
- **Proof of Paradigm Resurgence**: Forgotten search/evolutionary attacks are not ineffective but were previously poorly designed. Fine-tuning components allows them to surpass gradient SOTA, warning the community not to over-rely on gradients for robustness evaluation.
- **Black-box + Arbitrary Targets as Killer Features**: Since gradients are unnecessary, swapping targets to non-differentiable composite objectives like "certified ratio," "conformal coverage," or "set size" (involving quantiles, majority voting, or Monte Carlo) costs almost nothing.
- **Scalability of Open-ended Search**: Performance rises monotonically with computational resources (time or memory), making EvA more formidable for upper-bound evaluation than "fixed-convergence" gradient methods.
- **Solid Engineering**: $O(1)$ index mapping, sparse encoding, population stacking for single-forward passes, and selective resampling for certificate attacks transform GA from "slow and memory-heavy" into a scalable solution.

## Limitations & Future Work
- **Targeted Single-node Attacks**: 0-1 accuracy only has two values for a single node, causing GA to degrade to random search. Tanh-margin proxies must be used, showing pure accuracy fitness is not universal.
- **D&C Relaxation as a Double-edged Sword**: It can be harmful on small graphs; divergence from the global solution necessitates careful activation based on graph scale and budget.
- **Hyperparameter Overhead**: Population size, tournament size, mutation probability, $k_{dc}$ blocks, and warmup iterations $t_{warm}$ all require tuning.
- **Evaluation Scope**: Results are centered on citation networks (CoraML/Citeseer/Pubmed/Ogbn-Arxiv). Performance on other tasks (graph classification, link prediction) and larger industrial graphs remains to be verified.

## Related Work & Insights
- **Gradient Attack Lineage**: PRBCD (Geisler et al. 2021) and LRBCD (Gosch et al. 2024) use tanh-margin gradients + BCD. These are the direct competitors surpassed by this work.
- **Search Attack Ancestry**: Dai et al. (2018) GA attack was the forgotten starting point; Ours acts as a complete component-level upgrade.
- **Evaluation Methodology**: Follows the inductive setting of Gosch et al. (2024) to avoid "clean graph memorization" artifacts and uses exchangeable sampling for data splits (Lingam et al. 2023).
- **Inspiration**: For any scenario where "the original problem is discrete but forced into relaxation/proxies," returning to combinatorial search + carefully designed meta-heuristics may be more effective than forcing differentiability. Plug-and-play black-box targets are a structural advantage of search over gradients.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Not for inventing GA, but for systematically proving "fine-tuned search can beat gradient SOTA" and opening new attack surfaces like conformal/certificates.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple datasets, defenses, budgets, and comprehensive ablations on fitness/mutation/scalability. Inductive settings are rigorous.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear breakdown of limitations, intuitive diagrams, and logical flow. Some technical details moved to the appendix might slightly increase reading effort.
- **Value**: ⭐⭐⭐⭐ — Substantially raises the attack upper bound (~11% average gain) and highlights blind spots in current robustness evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temporal Graph Thumbnail: Robust Representation Learning with Global Evolutionary Skeleton](temporal_graph_thumbnail_robust_representation_learning_with_global_evolutionary.md)
- [\[NeurIPS 2025\] Practical Bayes-Optimal Membership Inference Attacks](../../NeurIPS2025/graph_learning/practical_bayes-optimal_membership_inference_attacks.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] LEAP: Local ECT-Based Learnable Positional Encodings for Graphs](leap_local_ect-based_learnable_positional_encodings_for_graphs.md)
- [\[ICLR 2026\] Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)

</div>

<!-- RELATED:END -->
