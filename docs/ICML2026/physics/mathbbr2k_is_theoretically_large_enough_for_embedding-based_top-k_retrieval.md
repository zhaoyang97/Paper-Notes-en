---
title: >-
  [Paper Note] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval
description: >-
  [ICML 2026][Physics & Scientific Computing][Minimum Embeddable Dimension] This paper proves that for three scoring functions—inner product, Euclidean distance, and cosine similarity—the Minimum Embeddable Dimension (MED) required to precisely retrieve all subsets of $m$ objects with size $\le k$ via score-thresholding is $\Theta(k)$, independent of $m$. With unit normalization and a positive score margin $\epsilon$, the feasible margin for robust MED is locked by an upper bou…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Minimum Embeddable Dimension"
  - "top-k retrieval"
  - "cyclic polytope"
  - "VC dimension"
  - "robust bounds"
date: 2026-05-08
content_hash: c4ac9bffd202a159
---

# $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval

**Conference**: ICML 2026  
**arXiv**: [2601.20844](https://arxiv.org/abs/2601.20844)  
**Code**: https://github.com/zihao-wang/med  
**Area**: Information Retrieval / Embedding Dimension Theory / Learning Theory  
**Keywords**: Minimum Embeddable Dimension, top-k retrieval, cyclic polytope, VC dimension, robust bounds

## TL;DR
This paper proves that for three scoring functions—inner product, Euclidean distance, and cosine similarity—the Minimum Embeddable Dimension (MED) required to precisely retrieve all subsets of $m$ objects with size $\le k$ via score-thresholding is $\Theta(k)$, independent of $m$. With unit normalization and a positive score margin $\epsilon$, the feasible margin for robust MED is locked by an upper bound $\epsilon_\star(m,k)=m/\sqrt{k(m-1)(m-k)}\sim 1/\sqrt{k}$, while a Gaussian centroid construction provides a feasible upper bound of $O(k^2\log m)$ dimensions.

## Background & Motivation
**Background**: Dense vector retrieval is the core of open-domain QA, recommendation systems, and RAG. All objects are embedded as $\bm{x}_i\in\mathbb{R}^d$, queries as $\bm{w}_q\in\mathbb{R}^d$, and results are given by the top-$k$ ranking of $s(\bm{x}_i,\bm{w}_q)$. A long-debated but confusing question is: to ensure any subset of size $\le k$ can be accurately retrieved by some query, how large must $d$ be?

**Limitations of Prior Work**: A study by Weller et al. at ICLR'26 (WBNL) presented a pessimistic conclusion—using free embedding optimization to fit all top-2 subsets, they fitted a curve where $d$ grows polynomially with $m$, claiming that "for web-scale retrieval, even the largest embedding dimensions are insufficient to cover all combinations." This was interpreted as a geometric capacity ceiling for single-vector embeddings.

**Key Challenge**: WBNL conflates whether an "optimization can find a set of vectors" with whether "a set of vectors exists." The former depends on the learning algorithm, loss surface, tokenizer, and numerical precision, while the latter is a true problem of geometric expressivity. This paper aims to answer the latter: geometrically, exactly how large does $d$ need to be?

**Goal**: Formalize the problem as the Minimum Embeddable Dimension MED$(m,k;\mathcal{F})$ and its $\epsilon$-robust version RMED$(m,k,\epsilon;\mathcal{F})$, providing tight upper and lower bounds, and refuting WBNL's "hard" benchmark through synthetic and real experiments.

**Key Insight**: The authors noted a natural correspondence between the $k$-shattering problem and $k$-neighborly polytopes in combinatorial geometry. A cyclic polytope in $\mathbb{R}^{2k}$ is $k$-neighborly, meaning any $\le k$ vertices can be separated from the others by an affine hyperplane. This implies $2k$ dimensions are "geometrically sufficient"; the remaining task is to construct the corresponding query vectors.

**Core Idea**: Use a cyclic polytope (moment curve $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$) for object embeddings and use the coefficients of a squared polynomial $P_S^2(t)=\prod_{i\in S}(t-t_i)^2$ as the query vector to provide an exact $2k$-dimensional construction. Simultaneously, define robust RMED to incorporate the margin dimension, proving that in the robust case, $m$ reappears in the upper bound in the form of $\log m$.

## Method
The paper consists of pure theory and numerical validation. The "method" corresponds to a set of definitions, constructions, and proofs of bounds.

### Overall Architecture
**Input**: Universe size $m$, target top-$k$, scoring family $\mathcal{F}\in\{\mathcal{F}_{\rm linear},\mathcal{F}_{\cos},\mathcal{F}_{\ell_2}\}$.  
**Output**: The minimum dimension $d^*$ such that "any subset of size $\le k$ can be precisely separated by some query."  
**Mechanism**: First, provide a $2k$ upper bound using cyclic polytopes $\rightarrow$ generate a $k-1$ lower bound via VC dimension $\rightarrow$ transfer these bounds to Euclidean/Cosine distances via geometric reduction $\rightarrow$ introduce normalization and margin $\epsilon$ to provide a feasible ceiling $\epsilon_\star(m,k)$ for RMED and an $O(k^2\log m)$ dimension Gaussian centroid construction $\rightarrow$ validate on synthetic top-2 and LIMIT datasets.

### Key Designs

**1. Cyclic Polytope + Squared Polynomial Query: Translating subset selection into polynomial construction for an inner product $2k$ upper bound**

The pessimistic conclusion of WBNL arises from conflating expressivity with optimization. The authors answer the purely geometric expressivity question by placing objects on a moment curve $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$. For any $S\subseteq[m],|S|\le k$, a query is explicitly constructed: take a univariate polynomial $P_S(t)=\prod_{i\in S}(t-t_i)$, expand $P_S^2(t)=\sum_{j=0}^{2|S|}c_j t^j$, and let $\bm{q}_S=(-c_1,-c_2,\dots,-c_{2k})$, then:

$$\langle\bm{v}_i,\bm{q}_S\rangle=c_0-P_S^2(t_i),$$

For $i\in S$, $P_S^2(t_i)=0$, reaching the upper bound $c_0$; for $i\notin S$, $P_S^2(t_i)>0$, making the score strictly smaller. This is algebraic evidence that the cyclic polytope is $\lfloor d/2\rfloor$-neighborly—"picking any $\le k$ objects with one query" is equivalent to "finding a non-negative polynomial that is zero only on $S$." This reduces subset selection to polynomial construction and serves as the geometric engine of the paper.

**2. VC Dimension Lower Bound + Radon Sharpening: Tightening MED into $[k-1, 2k]$**

An upper bound is insufficient; $\Theta(k)$ must be proven as the true lower limit. The authors define a binary threshold class $\mathcal{C}_{\mathcal{F},n}$ induced by $k$-shattering, proving $\textsc{MED}(m,k;\mathcal{F})\ge\textsc{VCD}^{-1}(k;\mathcal{F})$. Since the VC dimension for inner product, cosine, and Euclidean scoring is $n+1$, MED $\ge k-1$. Radon's theorem is then used (any $d+2$ points can be partitioned into two sets whose convex hulls intersect) to prove that if $d<\min\{2k,m-1\}$, there must exist a subset pair $A,B$ that serve as "selected/unselected" sets for the same query, making shattering impossible. This sharpens the inner product case to $\mathrm{MED}(m,k;\mathcal{F}_{\rm linear})=\min\{2k,m-1\}$. The VC dimension provides a general lower bound, and Radon refines it to a constant level, establishing that $\Theta(k)$ is sufficient and independent of $m$.

**3. Gaussian Centroid Construction + Feasible Margin Ceiling: Bilateral bounds for robust RMED**

Exact geometric expressivity differs from practical engineering difficulty. The authors define robust RMED with unit sphere normalization and a requirement that selected objects score at least $\epsilon$ higher than unselected ones. A variance identity provides the feasible ceiling: if all $k$-subset queries reach margin $\epsilon$, then $\|\bar{\bm{v}}_S-\bar{\bm{v}}\|_2\ge\frac{m-k}{m}\epsilon$ holds for all $S$. Taking the expectation over random subsets and using the unit norm property $\frac1m\sum\|\bm{v}_i-\bar{\bm{v}}\|^2\le1$ yields:

$$\epsilon\le\epsilon_\star(m,k)=\frac{m}{\sqrt{k(m-1)(m-k)}}\sim\frac{1}{\sqrt{k}}\ (\text{large }m).$$

For the upper bound, $m$ isotropic Gaussian vectors are sampled and normalized, and for each $S$, the query is the normalized centroid $\bm{u}_S\propto\sum_{i\in S}\bm{v}_i$. In $n=Ck^2\log m$ dimensions, all pairwise inner products are $O(1/k)$, where the selected object autocorrelation is $\Theta(1)$ and external objects contribute only $O(|S|/k)$ noise, ensuring a consistent margin of $\Omega(1/\sqrt{k})$ after normalization. This locates the core thesis: exact MED is independent of $m$, but once a positive margin is introduced, $m$ returns to the dimension formula via $\log m$.

### Loss & Training
The theoretical paper does not have a learning objective per se. Section 5's synthetic experiments use hinge loss on all positive/negative pairs with Adam optimization to find centroid GD witnesses, using the deterministic cyclic polytope as a control. The baseline on LIMIT/LIMIT-small is a label-unaware "random additive" approach—each token is assigned a fixed unit Gaussian vector, and documents/queries are sums of token vectors $\phi(x)=\sum_{t\in\tau(x)}G_t$, without supervised training.

## Key Experimental Results

### Main Results

Synthetic top-2 witness: Minimal dimensions for a "successful witness" at universe size $m$ are plotted, comparing cyclic polytope construction, centroid GD optimization, and WBNL's fitted curve.

| Setting (top-2, universe $m$) | Cyclic Polytope | Centroid GD | WBNL Fitted Curve |
|---|---|---|---|
| Any $m$ | Dimension $=4$ (Indep. of $m$) | $d\sim\log_2 m$ slow growth | Polynomial growth with $m$ |

LIMIT / LIMIT-small Recall@2 (Single-vector retrieval vs. WBNL's strongest baseline Promptriever Llama3-8B @ 4096 dims):

| Dataset | Tokenizer | $d$ | Recall@2 | Promptriever 8B @ 4096 |
|---|---|---|---|---|
| LIMIT | handmade | 256 | Exceeds baseline | 0.030 |
| LIMIT | vanilla (space/punct) | 512 | Exceeds baseline | 0.030 |
| LIMIT | qwen | 512 | Exceeds baseline | 0.030 |
| LIMIT @ 4096 | handmade/vanilla/qwen | 4096 | 0.9980 / 0.7060 / 0.2675 | 0.030 |
| LIMIT-small @ 4096 | handmade/vanilla/qwen | 4096 | 1.0000 / 0.9545 / 0.8010 | 0.543 |
| LIMIT-small | Cyclic Polytope (overfit) | 4 | Complete overfit | — |

### Ablation Study

"Ablations" between theorems consist of comparisons across three regimes:

| Regime | Dimension Upper Bound | Depends on $m$ | Key Tool |
|---|---|---|---|
| Exact MED (No margin) | $\min\{2k,m-1\}$, constant $\Theta(k)$ | No | Cyclic Polytope + Squared Polynomial |
| Robust RMED, margin $\epsilon=c/\sqrt{k}$ | $O(k^2\log m)$ | Yes (via $\log m$) | Gaussian centroid witness |
| Robust RMED, margin $\epsilon>\epsilon_\star(m,k)$ | $\infty$ | Yes (locked by $m,k$) | Variance identity ceiling |

### Key Findings
- Cyclic polytopes can exactly overfit any size LIMIT-small top-2 dataset at $d=4$, refuting the geometric narrative that "high dimensions are insufficient."
- Using a vanilla tokenizer with random additive vectors and no learning, 512 dimensions outperform Promptriever 8B (4096 dims) on Recall@2, indicating that failure on LIMIT is due to tokenizer/objective/optimization, not geometric capacity.
- In robust cases, higher margins become impossible: $\epsilon_\star(m,k)\sim 1/\sqrt{k}$ is a hard limit. Exceeding this margin cannot be fixed by increasing dimensions, providing guidance for margin loss design in retrieval systems.

## Highlights & Insights
- Decouples "geometric expressivity" from "searchability by learning algorithms," clarifying pessimistic conclusions in the dense retrieval community.
- Elegant construction: Turning subset selection into "finding roots of a polynomial" is a universal trick transferable to multi-label retrieval or ranking with set constraints.
- Gaussian centroid $O(k^2\log m)$ provides a natural geometric explanation for mean-pooled representations like contrastive learning/DPR—centroid queries are not hacks but feasible witnesses with quantitative guarantees.
- The robust margin ceiling $\epsilon_\star(m,k)$ suggests that setting a margin exceeding $1/\sqrt{k}$ is likely infeasible for large corpora, regardless of dimension.

## Limitations & Future Work
- The gap between the $\Omega(k)$ lower bound and the $O(k^2\log m)$ Gaussian centroid upper bound persists. Closing this may require tools beyond independent score comparison (e.g., one-bit recovery).
- Cyclic polytopes, while sufficient at $2k$, have extremely small margins and poor numerical stability, making them geometric existence proofs rather than deployable architectures.
- The representational gap between an arbitrary query choice and a neural encoder mapping $S$ to a query is not fully explored.
- Experiments focus on refuting LIMIT/LIMIT-small; verification on long-tail real-world benchmarks or cross-encoders is still needed.

## Related Work & Insights
- **vs Weller et al. 2026 (WBNL, ICLR'26)**: WBNL suggested $d$ must grow polynomially with $m$. This paper proves this is an optimization failure and provides a $d=4$ counterexample.
- **vs Guo et al. 2019**: They provided structured bounds for multi-class embedding; this paper addresses unstructured top-$k$ subsets where the number of classes is far larger.
- **vs Reimers & Gurevych 2021**: Provides a theoretical complement to their empirical study—observed dimension correlation likely stems from robust margins and packing limits, not exact embeddability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[AAAI 2026\] Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems](../../AAAI2026/physics/just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](../../NeurIPS2025/physics/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[ICML 2025\] Mixture-of-Expert Variational Autoencoders for Cross-Modality Embedding of Type Ia Supernova Data](../../ICML2025/physics/mixture-of-expert_variational_autoencoders_for_cross-modality_embedding_of_type_.md)

</div>

<!-- RELATED:END -->
