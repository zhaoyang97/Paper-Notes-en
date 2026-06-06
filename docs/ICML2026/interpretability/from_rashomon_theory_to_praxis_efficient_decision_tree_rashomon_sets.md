---
title: >-
  [Paper Note] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets
description: >-
  [ICML 2026][Interpretability][Rashomon Sets] PRAXIS utilizes a "fast but approximate" proxy algorithm (an improved version of LicketySPLIT) to estimate the optimal objective value of each subproblem…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Rashomon Sets"
  - "Sparse Decision Trees"
  - "Approximate Algorithms"
  - "Proxy Optimization"
  - "Branch-and-bound"
date: 2026-05-08
content_hash: 51ad309f461b664b
---

# From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets

**Conference**: ICML 2026  
**arXiv**: [2606.00202](https://arxiv.org/abs/2606.00202)  
**Code**: https://github.com/zakk-h/PRAXIS  
**Area**: Interpretable Machine Learning / Rashomon Sets / Sparse Decision Trees  
**Keywords**: Rashomon Sets, Sparse Decision Trees, Approximate Algorithms, Proxy Optimization, Branch-and-bound

## TL;DR
PRAXIS utilizes a "fast but approximate" proxy algorithm (an improved version of LicketySPLIT) to estimate the optimal objective value of each subproblem, enabling "on-demand expansion" pruning search for the Rashomon set of sparse decision trees. This approach compresses the runtime and memory, originally exponential relative to the tree space, into "polynomial time per output tree," allowing execution on scales of 11M samples / 472 features with a recall $\ge 0.98$.

## Background & Motivation

**Background**: The Rashomon set (the collection of all "approximately good" models under a given dataset and loss) is an important paradigm in current interpretable ML. Once all near-optimal sparse decision trees are enumerated, users can overlay secondary objectives such as fairness, causal constraints, or specific feature requirements by traversing the set once. Representative works include TreeFARMS (Xin 2022) and SORTeD (Arslan 2026), both of which exactly enumerate the Rashomon set of decision trees using branch-and-bound and dynamic programming.

**Limitations of Prior Work**: Exact algorithms suffer from exponential expansion in both runtime and memory relative to depth and the number of features. The paper provides a striking comparison: the search space for a decision tree of depth 4 and 20 features is approximately $8.4 \times 10^{18}$. On real-world data, TreeFARMS frequently encounters OOM errors with 100+ features, and SORTeD takes nearly 35 hours on the Churn dataset (472 features). RESPLIT (Babbar 2025) is the SOTA approximate method but remains "exponential in the worst case per tree" and has a very low recall for the true Rashomon set (Figure 3 shows 0 trees found on Churn/Electricity actually fall within the target Rashomon set).

**Key Challenge**: The Rashomon set itself may occupy only a tiny fraction of the total hypothesis space (estimated at the $10^{-37}$ scale by Semenova 2022), but the overhead of existing algorithms is tied to the total search space rather than the size of the output set. This results in "exponential waste to output a single tree."

**Goal**: Construct a Rashomon set approximation algorithm such that the **marginal cost of outputting each tree is polynomial**, while maintaining a recall close to 1.

**Key Insight**: The authors draw inspiration from the "pilot method" approach. LicketySPLIT is a single-tree version that uses a fast greedy completion at each node to estimate the true cost of each split. This paper generalizes this idea to Rashomon set enumeration: using a proxy to estimate the optimal objective $\text{Obj}^*$ of each subproblem, any split satisfying $\text{Proxy}(D_L) + \text{Proxy}(D_R) > \varepsilon_{\text{abs}}$ is pruned immediately.

**Core Idea**: Replace expensive lower-bound estimates with a PROXY algorithm satisfying "recursive refinement" (Definition 3.1) and design a "sibling subproblem iterative budget refinement" process to make pruning aggressive without losing true Rashomon set members.

## Method

### Overall Architecture

PRAXIS maintains an AND/OR search graph to compactly represent all candidate Rashomon trees. Starting from the root, for each subproblem (data subset $D$, remaining depth $d$, global budget $\varepsilon_{\text{abs}}$), three actions are performed:

1.  **Add Leaf Nodes**: Check if the objective values of "predict 0" and "predict 1" leaves, $C_b = \gamma + |\{y_i \neq b\}|$, are within the budget; if so, add them to the OrNode.
2.  **Traverse All Feature Splits $j$**: Partition the data into $(D_L, D_R)$ and use PROXY to calculate estimated costs for subproblems $P_L = \text{Proxy}(D_L, d-1, \gamma)$, $P_R = \text{Proxy}(D_R, d-1, \gamma)$. If $P_L + P_R > \varepsilon_{\text{abs}}$, the split is directly pruned.
3.  **Recursive Solving**: For remaining splits, call `Solve_Siblings` to allocate the budget between left and right subproblems, recursively construct subgraphs $G_L, G_R$, and then use `AddSplit` to attach the split nodes back to the current OR-graph.

The output is a search graph from which all approximate Rashomon trees can be enumerated in ascending order of objective value. The budget $\varepsilon_{\text{abs}}$ can be given explicitly or as a multiplicative form $(1+\varepsilon_{\text{mult}}) \cdot \text{Proxy}(D, d, \gamma)$. The objective function is $\text{Obj}(t, D, \gamma) = \gamma |t| + \sum_i \mathbb{1}\{t(x_i) \neq y_i\}$, where $|t|$ is the number of leaves and $\gamma = \lambda |D|$.

### Key Designs

1.  **Proxy Optimizer Framework**:
    *   **Function**: Uses a "fast but approximate" subroutine to estimate the objective of a decision tree on any subproblem $(D, d, \gamma)$ for search pruning.
    *   **Mechanism**: The proxy algorithm is defined to satisfy refinement: $\text{Proxy}(D, d, \gamma) = \text{Obj}(t)$ and for its children $\text{Proxy}(D_{t_{\text{left}}}, d-1, \gamma) \le \text{Obj}(t_{\text{left}})$ (similarly for the right subtree, see Def 3.1). If the proxy is optimal, PRAXIS degrades to exact enumeration (equivalent to TreeFARMS); if approximate, it still prunes via $P_L + P_R > \varepsilon_{\text{abs}}$. The default proxy is an improved LicketySPLIT: it first finds the leaf objective $\text{leaf\_obj} = \gamma + \min(p, n'-p)$ and returns if no further splitting is possible; otherwise, it greedily finds the best split and returns the minimum of the leaf objective and the recursive result. Improvements include moving "prepruning" after recursion, directly maximizing accuracy at $d=1$, and using 64-bit fingerprints for caching.
    *   **Design Motivation**: Exact algorithms use the "optimal subtree objective" as a lower bound, which itself requires exponential search. The proxy algorithm completes this with $O(n k^2 d^2)$ greedy approach + rollout, pushing overall complexity to $O(|R| n k^3 d^3)$—**outputting each Rashomon tree takes only polynomial time** (Theorem 3.2), achieving super-polynomial relative speedup over optimal DP (Corollary 3.3).

2.  **Iterative Budget Refinement for Sibling Subproblems (Solve_Siblings, Algorithm 3)**:
    *   **Function**: When a split is retained, determines how to split the root budget $\varepsilon_{\text{abs}}$ between the left and right subproblems.
    *   **Mechanism**: Initially sets the left subproblem budget to $\varepsilon_L^{\text{new}} = \varepsilon_{\text{abs}} - P_R$ (assuming the right side achieves the proxy level), runs PRAXIS to get $G_L$ and its minimum objective $G_L.\text{min\_obj}$. Then, uses the tighter actual value to update the right budget $\varepsilon_R^{\text{new}} = \varepsilon_{\text{abs}} - G_L.\text{min\_obj}$, runs the right side to get $G_R$, and then relaxes the left budget using $G_R.\text{min\_obj}$. This repeats until budgets stabilize.
    *   **Design Motivation**: Proxies are pessimistic estimates. If both subproblems perform better than the proxy, the initial allocation might be too tight, losing Rashomon set members. This "ping-pong" refinement allows PRAXIS to incrementally add budget to the side that produces good trees without restarting the whole search, satisfying recovery conditions (Theorem 3.5 "frontier cut") as much as possible.

3.  **Bitvector-Fingerprint Caching**:
    *   **Function**: Eliminates redundant subproblem computations arising from different split sequences corresponding to the same data subset.
    *   **Mechanism**: TreeFARMS uses bitvectors of length $n$ as keys (memory intensive); SORTeD uses split sets as keys (memory efficient but misses equivalent subsets). PRAXIS hashes each bitvector (plus depth budget) to a 64-bit fingerprint as the cache key. Every subproblem encountered by LicketySPLIT and the greedy subroutine is cached.
    *   **Design Motivation**: Combines the benefits of both—memory stores only fingerprints and solutions, while time is saved by reusing equivalent subproblems. The collision probability is minimal with a good hash, while the runtime and memory gains are order-of-magnitude (see Appendix B.7). This is key to achieving a memory complexity of $O(nk + \sum_{t \in R} |t|)$ (Theorem 3.4).

### Loss & Training

No training loss (discrete decision tree structure + 0/1 loss). The optimization objective is $\text{Obj}(t, D, \gamma) = \gamma |t| + \sum_i \mathbb{1}\{t(x_i) \neq y_i\}$, with $\gamma$ constrained to integers to avoid floating point issues. Default experiments use $\lambda \in \{0.005, 0.01, 0.02\}$, $\varepsilon_{\text{mult}} = 0.03$, and depth $d=5$ (some experiments $d=7$).

## Key Experimental Results

### Main Results

Comparison of PRAXIS / TreeFARMS / SORTeD / RESPLIT on 50 dataset-binarization combinations with $\lambda=0.02, \varepsilon=0.03, d=5$ (selected results, "–" indicates failure within 90 hours or 200 GB RAM):

| Dataset ($n / k$) | PRAXIS Time (s) | SORTeD Time (s) | RESPLIT Time (s) | PRAXIS Peak MB | SORTeD Peak MB |
|------|------|------|------|------|------|
| Churn (5K / 472) | **34.84** | 123776 (~34h) | 2564 | 279 | 22013 |
| Christine (5.4K / 231) | **944** | 38971 (~10.8h) | 12625 | 10439 | 12710 |
| Covertype (581K / 96) | **358** | 64673 (~18h) | 10102 | 1301 | 16107 |
| Higgs (11M / 84) | **2375** (~40 min) | – | – | 21537 | – |
| Compas (5K / 44) | **0.09** | 7.23 | 11.90 | 130 | 163 |

PRAXIS achieves up to 5 orders of magnitude speedup over TreeFARMS, 3 over SORTeD, and 2 over RESPLIT. Memory efficiency is approximately 5x relative to RESPLIT and up to 4 orders of magnitude better than TreeFARMS.

### Ablation Study (Approximate Quality & Single-Tree Solver)

| Dataset | recall ($\lambda=0.005$) | recall ($\lambda=0.02$) |
|------|------|------|
| Churn-472 | 0.997±0.005 | 1.000±0.000 |
| Electricity-264 | 0.994±0.003 | 1.000±0.000 |
| Christine-231 | 1.000±0.000 | 0.994±0.011 |
| Bike-164 | 0.986±0.010 | 1.000±0.000 |
| Other 18 datasets | 0.984–1.000 | 0.998–1.000 |

Average recall $\ge 0.98$ on 22 datasets with ground truth, mostly 1.0 (Table 2). On large datasets like Churn/Electricity where exact methods fail (Figure 3), PRAXIS finds over 1M more trees superior to RESPLIT's best tree, while **none of RESPLIT's trees fall within the target Rashomon set**.

As a single-tree solver: recovered global optima for all 50 datasets at $\lambda = 0.02$ (vs STreeD); only 2 failed at $\lambda=0.01$; 3 failed at $\lambda=0.005$, with objectives exceeding optimal by only ~1.00069x. Speed is up to 3 orders of magnitude faster than STreeD/GOSDT (e.g., News dataset: PRAXIS <3 min vs STreeD >20h).

### Key Findings

- **Proxy quality is decisive**: Using a weaker proxy (like pure greedy CART) results in significantly lower recall and speed (Appendix D.5). Small adjustments like "comparing leaves after recursion" and "maximizing accuracy at depth=1" heavily impact recall.
- **Marginal benefits of fingerprint caching**: Removing the cache leads to OOM on large datasets; the hit rate correlates positively with feature count and depth.
- **Depth scalability**: At $d = 7$, SORTeD fails within 150 hours, while PRAXIS finishes in 11 seconds, a 4-order-of-magnitude speedup over RESPLIT.

## Highlights & Insights

- **Theoretical contribution of "output-sensitive" complexity**: Rewrites the complexity of decision tree Rashomon set enumeration from "search-space dependent" to "output-set dependent" $O(|R| n k^3 d^3)$, with formal proof of super-polynomial speedup (Cor 3.3).
- **Elegant generalization of the Pilot Method**: Generalizes the rollout idea from single best-split selection to set output—using proxies for split evaluation, pruning, and sibling budget allocation while reusing the same cache.
- **Synergy between frontier cut and iterative refinement**: Theorem 3.5 provides a sufficient condition for recall. The iterative refinement in Algorithm 3 ensures that cases where the proxy overestimates, but good trees exist, are recovered—bridging theory and engineering.
- **Transferability**: The proxy + budget refinement framework is potentially applicable to other discrete structures like rule lists, decision forests, or subset selection.

## Limitations & Future Work

- **Supports only binary classification + binary features**: Continuous features require binarization (McTavish 2022); multi-class and regression are not addressed.
- **Trade-off between proxy strength and pruning precision**: Stronger proxies lead to higher recall but slower individual calls.
- **Theoretical guarantees are sufficient conditions**: The frontier cut condition is difficult to verify in practice, and the $\sigma$ factor in Cor 3.6 is a worst-case relaxation; perfect recall is not guaranteed.
- **Cache hash collisions**: Although "vanishingly small," 64-bit fingerprints may collide on extremely large $n$ or subproblem counts.
- **Lack of comparison with forward tree search**: As a Rashomon set solver, PRAXIS was not compared against sampling-based approximations like RL or CMA-ES for single-tree tasks.

## Related Work & Insights

- **vs TreeFARMS (Xin et al., 2022)**: Both target Rashomon sets; TreeFARMS uses bitvector subproblems + exact DP, causing memory explosion; PRAXIS uses proxy + fingerprint caching, reducing memory to be linear in output.
- **vs SORTeD (Arslan et al., 2026)**: SORTeD uses split-based subproblems to save memory but misses equivalent subsets; PRAXIS captures both advantages and reduces the per-tree cost to polynomial.
- **vs RESPLIT (Babbar et al., 2025)**: Both are approximate, but RESPLIT uses "subproblem exact solutions + global approximate stitching," while PRAXIS uses "global approximate search + proxy pruning." RESPLIT's per-tree cost remains exponential in the worst case.
- **vs LicketySPLIT (Babbar et al., 2025)**: LicketySPLIT is the single-tree pilot algorithm; PRAXIS is its Rashomon set counterpart.
- **vs GOSDT / STreeD**: Optimal single-tree solvers; PRAXIS can output approximate optimal trees up to 3 orders of magnitude faster.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically migrates pilot/rollout methods to Rashomon set enumeration with output-sensitive complexity proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 50 datasets / 11M samples / 472 features, covering runtime, memory, recall, and optimality across 3 SOTA baselines.
- Writing Quality: ⭐⭐⭐⭐ Algorithms are clear, theory and engineering are well-separated, despite minor LaTeX rendering issues in the cache section.
- Value: ⭐⭐⭐⭐⭐ Pushes the paradigm from "hundreds of features" to "millions of samples / hundreds of features," making it practical for fairness auditing and variable importance estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](diagnosing_the_reliability_of_llm-as-a-judge_via_item_response_theory.md)
- [\[ICML 2026\] PINE: Pruning Boosted Tree Ensembles with Conformal In-Distribution Prediction Equivalence](pine_pruning_boosted_tree_ensembles_with_conformal_in-distribution_prediction_eq.md)
- [\[AAAI 2026\] ToC: Tree-of-Claims Search with Multi-Agent Language Models](../../AAAI2026/interpretability/toc_tree-of-claims_search_with_multi-agent_language_models.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)

</div>

<!-- RELATED:END -->
