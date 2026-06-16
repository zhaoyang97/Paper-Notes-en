---
title: >-
  [Paper Note] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval
description: >-
  [ICML 2026][Physics & Scientific Computing][Paper Note] This paper proves that for three scoring functions—inner product, Euclidean distance, and cosine—the Minimum Embeddable Dimension (MED) required to precisely recall all subsets of size $\le k$ out of $m$ objects using score-thresholding is $\Theta(k)$, independent of $m$. After incorporating unit normalization and a po
tags:
  - ICML 2026
  - Physics & Scientific Computing
date: 2026-05-08
content_hash: 2d4195c0b21be7e9
---
# $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval

**Conference**: ICML 2026  
**arXiv**: [2601.20844](https://arxiv.org/abs/2601.20844)  
**Code**: https://github.com/zihao-wang/med  
**Area**: Information Retrieval / Embedding Dimension Theory / Learning Theory  
**Keywords**: Minimum Embeddable Dimension, top-k retrieval, cyclic polytope, VC dimension, robust margin

## TL;DR
This paper proves that for three scoring functions—inner product, Euclidean distance, and cosine—the Minimum Embeddable Dimension (MED) required to precisely recall all subsets of size $\le k$ out of $m$ objects using score-thresholding is $\Theta(k)$, independent of $m$. After incorporating unit normalization and a positive score margin $\epsilon$, the feasible margin for robust MED is locked by an upper bound $\epsilon_\star(m,k)=m/\sqrt{k(m-1)(m-k)}\sim 1/\sqrt{k}$, while a Gaussian centroid construction provides a feasible upper bound of $O(k^2\log m)$ dimensions.

## Background & Motivation
**Background**: Dense retrieval is the core of open-domain QA, recommendation systems, and RAG. All objects are embedded as $\bm{x}_i\in\mathbb{R}^d$, queries as $\bm{w}_q\in\mathbb{R}^d$, and results are given by the top-$k$ ranking of $s(\bm{x}_i,\bm{w}_q)$. A long-standing but confusing question is: how large must $d$ be to ensure any subset of size $\le k$ can be precisely recalled by some query?

**Limitations of Prior Work**: A prior work by Weller et al. at ICLR'26 (WBNL) provided a pessimistic conclusion. By using free embedding optimization to fit all top-2 subsets, they fitted a curve where $d$ grows polynomially with $m$, claiming that "for web-scale retrieval, even the largest embedding dimensions are insufficient to cover all combinations." This was interpreted as a geometric capacity ceiling for single-vector embeddings.

**Key Challenge**: WBNL conflates "whether an optimization algorithm can find a set of vectors" with "whether such a set of vectors exists." The former depends on the learning algorithm, loss surface, tokenizer, and numerical precision, while the latter is the true question of geometric expressivity. This paper aims to answer the latter: geometrically, how large does $d$ actually need to be?

**Goal**: To formalize the above problem as the Minimum Embeddable Dimension MED$(m,k;\mathcal{F})$ and its $\epsilon$-robust version RMED$(m,k,\epsilon;\mathcal{F})$, provide tight upper and lower bounds, and disprove the "hard" benchmark of WBNL using synthetic and real-world experiments.

**Key Insight**: The authors noted a natural correspondence between the $k$-shattering problem and $k$-neighborly polytopes in combinatorial geometry. A cyclic polytope in $\mathbb{R}^{2k}$ is $k$-neighborly, meaning any $\le k$ vertices can be separated from the remaining vertices by an affine hyperplane. This implies that $2k$ dimensions are already "geometrically sufficient," leaving only the problem of constructing the corresponding query vectors.

**Core Idea**: Use a cyclic polytope (moment curve $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$) for object embeddings and the coefficients of a squared polynomial $P_S^2(t)=\prod_{i\in S}(t-t_i)^2$ as the query vector to provide an exact $2k$-dimensional construction. Simultaneously, define RMED to incorporate the margin dimension, proving that in robust cases, $m$ returns to the upper bound in the form of $\log m$.

## Method
The paper consists of pure theory and numerical verification. The "Method" corresponds to a set of definitions, constructions, and proofs of bounds.

### Overall Architecture
**Input**: Universe size $m$, target top-$k$, scoring family $\mathcal{F}\in\{\mathcal{F}_{\rm linear},\mathcal{F}_{\cos},\mathcal{F}_{\ell_2}\}$.  
**Output**: The minimum dimension $d^*$ such that "any subset of size $\le k$ can be accurately separated by some query."  
**Mechanism**: First, provide a $2k$ upper bound using cyclic polytopes $\rightarrow$ establish a $k-1$ lower bound via VC dimension $\rightarrow$ transfer bounds to Euclidean/Cosine via geometric reduction $\rightarrow$ introduce normalization and margin $\epsilon$ to provide the feasible ceiling $\epsilon_\star(m,k)$ for RMED and an $O(k^2\log m)$ dimensional Gaussian centroid construction $\rightarrow$ validate on synthetic top-2 and LIMIT datasets.

### Key Designs

**1. Cyclic Polytope + Squared Polynomial Query: Translating subset selection into polynomial construction for an inner product $2k$ upper bound**

The reason WBNL reached the pessimistic conclusion that "dimension grows polynomially with $m$" is the conflation of optimization feasibility and geometric existence. To address pure geometric expressivity, the authors place objects on the moment curve $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$. For any $S\subseteq[m],|S|\le k$, they explicitly construct a query: take a univariate polynomial $P_S(t)=\prod_{i\in S}(t-t_i)$, expand $P_S^2(t)=\sum_{j=0}^{2|S|}c_j t^j$, and set $\bm{q}_S=(-c_1,-c_2,\dots,-c_{2k})$. Then:

$$\langle\bm{v}_i,\bm{q}_S\rangle=c_0-P_S^2(t_i),$$

When $i\in S$, $P_S^2(t_i)=0$ and achieves the upper bound $c_0$; when $i\notin S$, $P_S^2(t_i)>0$ and is strictly smaller. This is the algebraic evidence that a cyclic polytope is $\lfloor d/2\rfloor$-neighborly—"picking any $\le k$ objects with a single query" is equivalent to "finding a non-negative polynomial that is zero only on $S$."

**2. VC Dimension Lower Bound + Radon Sharpening: Tightening MED within $[k-1,2k]$**

An upper bound alone is insufficient; $\Theta(k)$ must be proven as the true lower limit. The authors define a binary threshold class $\mathcal{C}_{\mathcal{F},n}$ induced by $k$-shattering and prove $\textsc{MED}(m,k;\mathcal{F})\ge\textsc{VCD}^{-1}(k;\mathcal{F})$. Since the VC dimensions for inner product, cosine, and Euclidean scoring are all $n+1$, MED $\ge k-1$. Using Radon’s theorem, they prove that if $d<\min\{2k,m-1\}$, there must exist a pair of subsets $A,B$ that serve as the "selected/unselected" sets for some query, making shattering impossible. This sharpens the inner product case to $\mathrm{MED}(m,k;\mathcal{F}_{\rm linear})=\min\{2k,m-1\}$.

**3. Gaussian Centroid Construction + Margin Feasibility Ceiling: Two-sided bounds for robust RMED**

The authors define robust RMED under the stronger requirements of unit sphere normalization and a margin $\epsilon$ between selected and unselected objects. They use variance identities to give a feasibility ceiling: if all $k$-subset queries reach margin $\epsilon$, then $\|\bar{\bm{v}}_S-\bar{\bm{v}}\|_2\ge\frac{m-k}{m}\epsilon$ holds for all $S$. By calculating expectations over random subsets and using the unit norm property $\frac1m\sum\|\bm{v}_i-\bar{\bm{v}}\|^2\le1$, they derive:

$$\epsilon\le\epsilon_\star(m,k)=\frac{m}{\sqrt{k(m-1)(m-k)}}\sim\frac{1}{\sqrt{k}}\ (\text{for large }m).$$

The upper bound is sampled by normalizing $m$ isotropic Gaussian vectors and taking the query for each $S$ as the normalized centroid $\bm{u}_S\propto\sum_{i\in S}\bm{v}_i$. In $n=Ck^2\log m$ dimensions, all pairwise inner products are $O(1/k)$, making the selected object autocorrelation $\Theta(1)$ and external noise only $O(|S|/k)$. The real bottleneck is optimization/numerical conditions rather than geometric capacity.

## Key Experimental Results

### Main Results

Synthetic top-2 witness: Minimal dimensions for "successful witness" under universe size $m$ compared across constructions.

| Setting (top-2, universe $m$) | Cyclic Polytope | Centroid GD Optimization | WBNL Fitted Curve |
|---|---|---|---|
| Any $m$ | Dim $=4$ (independent of $m$) | $d\sim\log_2 m$ slow growth | Polynomial growth with $m$ |

LIMIT / LIMIT-small Recall@2 (Single-vector retrieval vs strong baseline Promptriever Llama3-8B @ 4096 dim):

| Dataset | Tokenizer | $d$ | Recall@2 | Promptriever 8B @ 4096 |
|---|---|---|---|---|
| LIMIT | handmade | 256 | Exceeds baseline | 0.030 |
| LIMIT | vanilla | 512 | Exceeds baseline | 0.030 |
| LIMIT @ 4096 | qwen | 4096 | 0.2675 | 0.030 |
| LIMIT-small @ 4096 | qwen | 4096 | 0.8010 | 0.543 |

### Ablation Study

"Ablation" between theorems consists of comparisons between three regimes:

| Regime | Dimension Upper Bound | Depends on $m$ | Key Tool |
|---|---|---|---|
| Exact MED (no margin) | $\min\{2k,m-1\}$, $\Theta(k)$ | No (independent of $m$) | Cyclic polytope |
| Robust RMED, margin $\epsilon=c/\sqrt{k}$ | $O(k^2\log m)$ | Yes (via $\log m$) | Gaussian centroid |
| Robust RMED, margin $\epsilon>\epsilon_\star(m,k)$ | $\infty$ | Yes (margin locked by $m,k$) | Variance identity ceiling |

### Key Findings
- Cyclic polytopes can exactly overfit any size LIMIT-small top-2 dataset at $d=4$, disproving the "high dimensions are insufficient" narrative.
- Using a vanilla tokenizer with random additive vectors at only 512 dimensions outperforms the 4096-dimensional Promptriever 8B on Recall@2. This suggests that the failure of models on LIMIT is due to tokenizers/objectives/optimization rather than geometric capacity.
- The robust margin $\epsilon_\star(m,k)\sim 1/\sqrt{k}$ is an absolute hard cap; exceeding this margin cannot be fixed by any number of dimensions.

## Highlights & Insights
- Decoupling "geometric expressivity" from "optimization findability" clarifies pessimistic conclusions in the dense retrieval community.
- The Cyclic polytope + squared polynomial query construction is elegant, transforming subset selection into a polynomial root problem.
- Gaussian centroid $O(k^2\log m)$ provides a natural geometric explanation for mean-pooled representations like contrastive learning/DPR—centroid queries are not engineering hacks but feasible witnesses with quantitative guarantees.
- The robust margin feasibility ceiling $\epsilon_\star(m,k)$ provides an upper bound for selecting temperatures/margins in contrastive loss.

## Limitations & Future Work
- The lower bound for exact MED and RMED is still $\Omega(k)$, while the Gaussian centroid upper bound is $O(k^2\log m)$; this gap of $k$ might not be tight.
- While $2k$ dimensions suffice for cyclic polytopes, the margin is extremely small and numerical conditions are poor, making it nearly undeployable in engineering.
- There is still a representational gap between choosing an arbitrary query for $S$ and mapping $S$ to a query via a neural encoder.
- Experiments focused on LIMIT; complete comparisons on larger real-world benchmarks like MS MARCO/BEIR were not included.

## Related Work & Insights
- **vs Weller et al. 2026 (WBNL)**: WBNL used free optimization to claim polynomial growth; this paper proves it is an optimization failure and provides a $d=4$ counterexample.
- **vs Guo et al. 2019**: They provided structured bounds for multi-class embedding; this paper maps the unstructured top-$k$ subset class, where the number of classes is significantly larger.
- **vs Reimers & Gurevych 2021**: This paper provides a theoretical explanation for the observed dimensionality curse, suggesting it stems from robust margins and packing bounds rather than exact embeddability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems](../../AAAI2026/physics/just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)
- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](../../NeurIPS2025/physics/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[ICML 2025\] Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](../../ICML2025/physics/mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)

</div>

<!-- RELATED:END -->
