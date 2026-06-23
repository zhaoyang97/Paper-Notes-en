---
title: >-
  [Paper Note] Welfarist Formulations for Diverse Similarity Search
description: >-
  [ICLR 2026][Information Retrieval & RAG][ANN] This paper models "attribute diversity in retrieval results" as a **welfare function maximization** problem from mathematical economics—treating each attribute as an agent and replacing the similarity sum of standard nearest neighbor search with **Nash Social Welfare (geometric mean)**. This achieves a **query-adaptive
tags:
  - ICLR 2026
  - Information Retrieval & RAG
  - ANN
  - RAG
date: 2026-05-08
content_hash: 9bbc6e97740e0df3
---
# Welfarist Formulations for Diverse Similarity Search

**Conference**: ICLR 2026  
**Paper**: [OpenReview](https://openreview.net/forum?id=welfarist_formulations_for_diverse_similarity_search) (⚠️ Link subject to original source)  
**Code**: None  
**Area**: Information Retrieval / Nearest Neighbor Search  
**Keywords**: Nearest Neighbor Search, Diversity, Nash Social Welfare, ANN, RAG

## TL;DR
This paper models "attribute diversity in retrieval results" as a **welfare function maximization** problem from mathematical economics—treating each attribute as an agent and replacing the similarity sum of standard nearest neighbor search with **Nash Social Welfare (geometric mean)**. This achieves a **query-adaptive** trade-off between "relevance" and "diversity" and provides efficient algorithms with provable approximation guarantees that can be applied on top of any ANN.

## Background & Motivation
**Background**: Nearest Neighbor Search (NNS) is the underlying primitive for web search, recommendation systems, and recently RAG: given a query vector $q$ and a vector database $P$, find $k$ vectors most similar to $q$, aiming for $\arg\max_{S\subseteq P,|S|=k}\sum_{v\in S}\sigma(q,v)$. Since exact solutions are too expensive for high-dimensional large-scale data, Approximate Nearest Neighbor (ANN) techniques (LSH, k-d trees, clustering, graph indices like DiskANN/HNSW, etc.) are used.

**Limitations of Prior Work**: Pure relevance often leads to high redundancy—ads from the same merchant, results from the same domain, or shirts of the same color filling the screen. Thus, "diversity" is as important as "relevance" (e.g., Google limits results per domain; Microsoft adds diversity constraints in ads). To quantify diversity, a natural approach is to assign **attributes** (color, seller, category...) to each vector and seek dispersion across these attributes in the results.

**Key Challenge**: The only prior diversity scheme (Anand et al. 2025) uses **hard constraints**—specifying that each attribute $\ell$ contributes at most $k'$ results, then maximizing relevance under this constraint. However, $k'$ is a **fixed heuristic quota**: it cannot adapt to query intent. When searching for "blue shirt," the "blue" color constraint might block valid blue shirts and sacrifice relevance; when searching for "shirts," one expects maximum color dispersion. A single $k'$ cannot satisfy both types of queries. Worse, generalizing hard constraints to **multi-attribute** cases (one vector having multiple attributes) is NP-hard even to determine if a set of $k$ vectors exists that satisfies the constraints.

**Goal**: To create a unified framework that preserves relevance, promotes diversity, **adjusts automatically per query**, and allows practitioners to **parameterize trade-off control**, supported by deployable algorithms (on existing ANN) with theoretical guarantees.

**Key Insight**: The authors borrow the **collective welfare** theory from mathematical economics. By treating $c$ attributes as $c$ agents, the "utility" of an agent is the sum of similarities of the selected vectors belonging to that attribute. These utilities are aggregated using a **welfare function** rather than a simple sum.

**Core Idea**: Use **Nash Social Welfare (geometric mean) instead of arithmetic summation** as the retrieval objective—geometric mean naturally rewards "non-zero utility for all attributes," thus automatically promoting diversity without hard quotas. It also allows the selection of many items from one attribute when that attribute dominates the query intent, achieving query adaptability.

## Method

### Overall Architecture
This work presents an **objective function + algorithm**; it is not a neural network pipeline. The core lies in changing the retrieval objective from "similarity summation" to a "welfare function" and designing greedy algorithms with proofs for this new objective. The structure consists of three layers:

1.  **Modeling Layer (NaNNS)**: Each attribute $\ell\in[c]$ is an agent. A subset $S$ provides utility $u_\ell(S)=\sum_{v\in S\cap D_\ell}\sigma(q,v)$ ($D_\ell$ is the set of vectors with attribute $\ell$). The retrieval goal is to maximize Nash Social Welfare $\mathrm{NSW}(S)=\big(\prod_{\ell=1}^c(u_\ell(S)+\eta)\big)^{1/c}$.
2.  **Generalization Layer (p-NNS)**: A $p$-mean $M_p$ unifies a family of objectives—$p=1$ reverts to standard NNS (pure relevance), $p\to 0$ is NSW, and $p\to-\infty$ is egalitarianism (maximum diversity). A "knob" continuously adjusts the trade-off.
3.  **Algorithm Layer**: For single-attribute settings, an **exactly optimal** greedy algorithm, Nash-ANN, is provided. it can call any ANN as a subroutine and inherits its approximation ratio. For multi-attribute settings, the problem is proven to be NP-hard, and a $(1-1/e)$-approximation algorithm for $\log\mathrm{NSW}$ is provided.

Since this is a pure algorithm/theory improvement, no Mermaid framework diagram is included.

### Key Designs

**1. NaNNS: Treating attributes as agents and replacing sum with geometric mean**

This is the modeling foundation, directly addressing the "rigid quota" issue. By defining the utility of an attribute agent as the cumulative similarity $u_\ell(S)=\sum_{v\in S\cap D_\ell}\sigma(q,v)$, the objective becomes the geometric mean of these $c$ utilities (plus a smoothing constant $\eta>0$):

$$\mathrm{NaNNS}:\quad \arg\max_{S\subseteq P,\,|S|=k}\ \log\mathrm{NSW}(S)=\frac{1}{c}\sum_{\ell\in[c]}\log\big(u_\ell(S)+\eta\big).$$

Why it works: The geometric mean satisfies several fairness axioms in welfare economics (Symmetry, Independence of Irrelevant Agents, Scale Invariance, Pigou–Dalton Principle) and is bounded between egalitarianism ($\min_\ell u_\ell$) and utilitarianism (arithmetic mean) by the AM–GM inequality. Crucially, its marginal utility is **diminishing**: when an attribute $u_\ell$ is already large, adding another vector results in a small $\log$ increment. The model automatically allocates slots to attributes with lower utility. **Diversity is a byproduct of the optimization objective rather than an external hard constraint.** If a query is naturally dominated by one attribute (e.g., "blue shirt"), and other attributes have near-zero utility, the geometric mean will not force irrelevant selections, thus preserving relevance—an adaptive behavior hard quotas cannot achieve. Note: If the geometric mean is replaced by the arithmetic mean, it **exactly degrades** to standard NNS in single-attribute settings.

**2. p-mean Generalization: A knob for the "relevance ↔ diversity" trade-off**

NaNNS provides a fixed balance point. However, different tasks require different diversity intensities. The authors parameterize the objective using generalized $p$-means: for $p\in(-\infty,1]$,

$$M_p(u_1,\dots,u_c)=\Big(\tfrac{1}{c}\textstyle\sum_{\ell=1}^c u_\ell^{\,p}\Big)^{1/p},\qquad \mathrm{p\text{-}NNS}:\ \max_{|S|=k} M_p\big(u_1(S),\dots,u_c(S)\big).$$

$p=1$ is utilitarianism (pure relevance), $p\to0$ is NSW, and $p\to-\infty$ is egalitarianism (max diversity). This provides a **monotonic knob**: smaller $p$ favors diversity, while larger $p$ favors relevance.

**3. Nash-ANN: An exact greedy algorithm on top of any ANN**

In single-attribute settings (vectors belong to exactly one attribute, $D_\ell$ are disjoint), **Nash-ANN** works as follows: in **pre-processing**, run NNS for each attribute $\ell$ to get $k$ candidates $\widehat D_\ell$ (using any standard ANN); in **greedy selection**, maintain cumulative similarity $w_\ell$ for each attribute and select the next vector maximizing the marginal $\log\mathrm{NSW}$ gain:

$$a=\arg\max_{\ell\in[c]}\Big[\log\big(w_\ell+\eta+\sigma(q,v^\ell_{(k_\ell+1)})\big)-\log(w_\ell+\eta)\Big].$$

After $k$ iterations, the result is **exactly optimal** for NaNNS (Theorem 1). **Corollary 2** proves that if the pre-processing uses an $\alpha$-approximate ANN, Nash-ANN outputs an $\alpha$-approximation of the optimal NSW. It **inherits the approximation ratio and sub-linear time of the underlying ANN**, allowing it to be layered on DiskANN/HNSW without rebuilding indices.

**4. Multi-attribute NP-hardness and (1−1/e) Approximation**

When a vector can have multiple attributes, NaNNS becomes **NP-hard** (Theorem 3, reduced from Maximum Independent Set). Fortunately, $\log\mathrm{NSW}(S)$ is **monotone submodular**, so a standard marginal gain greedy algorithm (Algorithm 2, Multi Nash-ANN) yields a $(1-\tfrac1e)$-approximation (Theorem 4).

## Loss & Training
Ours involves no model training. The "objective functions" are $\log\mathrm{NSW}$ or $M_p$. The only hyperparameters are $\eta>0$ and the trade-off knob $p$.

## Key Experimental Results

### Main Results
Comparison on 4 datasets against ANN (pure relevance), Div-ANN (hard constraints, Anand et al. 2025), and Nash-ANN (Ours). Metrics: **Approximation ratio** for relevance and **Entropy** for diversity.

| Dataset | Scale / Dim | Attributes | Key Findings ($k=50$, single-attribute) |
|--------|------------|------|------|
| Amazon | 92K / 768 | Product Color | Nash-ANN ratio ≈ 1, Entropy ≈ Div-ANN ($k'=1$), and **Pareto dominates** Div-ANN ($k'=2$). |
| Deep1b-(Clus) | 9.99M / 96 | Synthetic | Nash-ANN **Pareto dominates** Div-ANN ($k'=5$). |
| Sift1m-(Clus) | 1M / 128 | Synthetic (Multi-attr) | Multi Nash-ANN yields a superior Relevance-Diversity frontier compared to Multi Div-ANN. |
| ArXiv | 200K / 1536 | Year, Category | Trends are consistent (see Appendix F.3). |

Qualitative example: For "shirts," Nash-ANN selects various colors; for "blue shirt," it converges to blue. **The same objective adapts to the query without parameter changes.**

### Ablation Study
Sweeping $p$ validates the knob effect:

| Configuration | Relevance (Ratio) | Diversity (Entropy) | Description |
|------|------|------|------|
| $p=1$ | Highest (≈1) | Lowest | Reverts to NNS (Utilitarian) |
| $p=0$ (Nash) | High | High | Adaptive balance point |
| $p=-1, -0.5$ | Medium | Higher | Diversity-oriented |
| $p=-10$ | Lowest | Highest | Egalitarian (Min-utility focus) |

As $p$ decreases, the results move smoothly along the Relevance-Diversity Pareto frontier.

### Key Findings
- **NSW inherently provides diversity**: Without hard quotas, the diminishing marginal utility of the geometric mean allocates slots to under-represented attributes.
- **Inheritance of ANN properties**: Nash-ANN treats ANN as a black box. An $\alpha$-approximate input leads to an $\alpha$-approximate output, enabling deployment on industrial indices.
- **Multi-attribute robustness**: While hard constraints face feasibility issues in multi-attribute cases, NSW leverages submodularity for a $(1-1/e)$ approximation guarantee.

## Highlights & Insights
- **Interdisciplinary Leverage**: Bringing Nash Social Welfare from economics into retrieval transforms diversity from "adding constraints" to "changing the objective," aligning relevance with fairness axioms.
- **Diminishing Marginal Utility = Free Diversity**: This is the most significant "Aha!" moment—the concave $\log$ function naturally implements adaptive quotas without extra parameters.
- **Decoupling Algorithm from Index**: Since the method is a wrapper, it provides a "plug-and-play" paradigm for diversity in NNS.

## Limitations & Future Work
- **Discrete Attribute Dependency**: Relies on pre-labeled discrete attributes; not directly applicable to continuous or latent attributes.
- **Multi-attribute Complexity**: The NP-hardness limits multi-attribute settings to an approximation, requiring a large candidate pool from the ANN.
- **Selection of $\eta$ and $p$**: While $p$ is a continuous knob, determining the optimal $p$ for a specific production environment still requires tuning.
- Future work: Extending to continuous/hierarchical attributes or joint learning of $p$ with re-ranking models.

## Related Work & Insights
- **vs Div-ANN (Anand et al. 2025)**: They maximize relevance under hard quotas ("at most $k'$ per attribute"). Ours uses soft objectives. Advantages of Ours include query adaptability and better Pareto performance.
- **vs Standard ANN**: Standard ANN ignores diversity. Ours acts as a wrapper to add diversity without re-indexing.
- **vs MMR (Maximal Marginal Relevance)**: MMR uses "Similarity - Redundancy" heuristics. Ours is grounded in welfare economics axioms and provides approximation guarantees.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](../../CVPR2025/information_retrieval/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)
- [\[ICLR 2026\] ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains](elvis_efficient_visual_similarity_from_local_descriptors_that_generalizes_across.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICLR 2026\] Lean Finder: Semantic Search for Mathlib That Understands User Intents](lean_finder_semantic_search_for_mathlib_that_understands_user_intents.md)
- [\[ICLR 2026\] Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics](demystifying_deep_search_a_holistic_evaluation_with_hint-free_multi-hop_question.md)

</div>

<!-- RELATED:END -->
