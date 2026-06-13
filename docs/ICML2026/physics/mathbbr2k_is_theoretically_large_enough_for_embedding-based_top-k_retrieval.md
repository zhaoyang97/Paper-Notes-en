---
title: >-
  [Paper Note] $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval
description: >-
  [ICML 2026][Physics & Scientific Computing][Minimum Embeddable Dimension] This paper demonstrates that for inner product, Euclidean distance, and cosine scoring functions…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Minimum Embeddable Dimension"
  - "top-k retrieval"
  - "cyclic polytope"
  - "VC dimension"
  - "robust margin"
date: 2026-05-08
content_hash: b25f942c6a2626c0
---

# $\mathbb{R}^{2k}$ is Theoretically Large Enough for Embedding-based Top-$k$ Retrieval

**Conference**: ICML 2026  
**arXiv**: [2601.20844](https://arxiv.org/abs/2601.20844)  
**Code**: https://github.com/zihao-wang/med  
**Area**: Information Retrieval / Embedding Dimension Theory / Learning Theory  
**Keywords**: Minimum Embeddable Dimension, top-k retrieval, cyclic polytope, VC dimension, robust margin

## TL;DR
This paper demonstrates that for inner product, Euclidean distance, and cosine scoring functions, the Minimum Embeddable Dimension (MED) required to precisely recall all subsets of size $\le k$ from $m$ objects via score-thresholding is $\Theta(k)$, independent of $m$. Upon introducing unit normalization and a positive score margin $\epsilon$, the feasible margin of Robust RMED is capped by $\epsilon_\star(m,k)=m/\sqrt{k(m-1)(m-k)}\sim 1/\sqrt{k}$, while a Gaussian centroid construction provides a feasible upper bound of $O(k^2\log m)$ dimensions.

## Background & Motivation
**Background**: Dense vector retrieval is core to open-domain QA, recommendation, and RAG. Objects are embedded as $\bm{x}_i\in\mathbb{R}^d$, queries as $\bm{w}_q\in\mathbb{R}^d$, and results are provided via top-$k$ ranking of $s(\bm{x}_i,\bm{w}_q)$. A long-debated question with conflicting answers is: how large must $d$ be to ensure any subset of size $\le k$ can be precisely recalled by some query?

**Limitations of Prior Work**: Weller et al. (ICLR'26, WBNL) presented a pessimistic conclusion. By using free embedding optimization to fit all top-2 subsets, they produced a curve where $d$ grows polynomially with $m$, claiming that for web-scale retrieval, even the largest embedding dimensions are insufficient to cover all combinations. This was interpreted as a geometric capacity ceiling for single-vector embeddings.

**Key Challenge**: WBNL conflates whether an optimization algorithm can find a set of vectors with whether such a set of vectors geometrically exists. The former depends on learning algorithms, loss surfaces, tokenizers, and numerical precision, while the latter is the true question of geometric expressivity. This paper aims to answer the latter: how large does $d$ truly need to be from a purely geometric perspective?

**Goal**: Formalize the problem as Minimum Embeddable Dimension $\text{MED}(m,k;\mathcal{F})$ and its $\epsilon$-robust version $\text{RMED}(m,k,\epsilon;\mathcal{F})$, provide tight upper and lower bounds, and debunk the "hard" benchmarks of WBNL using synthetic and real-world experiments.

**Key Insight**: The authors observe a natural correspondence between the $k$-shattering problem and $k$-neighborly polytopes in combinatorial geometry. A cyclic polytope is $k$-neighborly in $\mathbb{R}^{2k}$, meaning any $\le k$ vertices can be separated from the others by an affine hyperplane. This implies $2k$ dimensions are geometrically sufficient; the remaining task is constructing the corresponding query vectors.

**Core Idea**: Use cyclic polytopes (moment curve $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$) as object embeddings and coefficients of squared polynomials $P_S^2(t)=\prod_{i\in S}(t-t_i)^2$ as query vectors to provide an exact $2k$-dimensional construction. Simultaneously, define Robust RMED to incorporate the margin dimension, proving that in robust scenarios, $m$ returns to the upper bound in a $\log m$ form.

## Method
The paper consists of pure theory and numerical validation. The "method" corresponds to a set of definitions, constructions, and proofs of bounds.

### Overall Architecture
**Input**: Universe size $m$, target top-$k$, scoring family $\mathcal{F}\in\{\mathcal{F}_{\rm linear},\mathcal{F}_{\cos},\mathcal{F}_{\ell_2}\}$.  
**Output**: The minimum dimension $d^*$ such that "any subset of size $\le k$ can be precisely separated by some query."  
**Mechanism**: Use cyclic polytopes to provide a $2k$ upper bound → Use VC dimension to provide a $k-1$ lower bound → Transfer bounds to Euclidean/Cosine via geometric reduction → Introduce normalization and margin $\epsilon$ to derive the feasible ceiling $\epsilon_\star(m,k)$ for RMED and an $O(k^2\log m)$ Gaussian centroid construction → Validate on synthetic top-2 and LIMIT datasets.

### Key Designs

1.  **Cyclic Polytope + Squared Polynomial Query: $2k$ Upper Bound for Inner Product MED**:
    *   **Function**: Places $m$ objects on a moment curve and explicitly writes a $2k$-dimensional query vector for any $S\subseteq[m],|S|\le k$, such that $\langle\bm{v}_i,\bm{q}_S\rangle$ reaches a maximum value $c_0$ for $i\in S$ and is strictly less than $c_0$ for $i\notin S$.
    *   **Mechanism**: Take $\bm{v}_i=(t_i,t_i^2,\dots,t_i^{2k})$ with distinct $t_i$. Construct a univariate polynomial $P_S(t)=\prod_{i\in S}(t-t_i)$ and expand $P_S^2(t)=\sum_{j=0}^{2|S|}c_j t^j$. Setting the query $\bm{q}_S=(-c_1,-c_2,\dots,-c_{2k})$ yields $\langle\bm{v}_i,\bm{q}_S\rangle=c_0-P_S^2(t_i)$. For $i\in S$, $P_S^2(t_i)=0$, reaching the upper bound $c_0$; for $i\notin S$, $P_S^2(t_i)>0$, making the score strictly smaller.
    *   **Design Motivation**: This is the algebraic evidence that a cyclic polytope is $\lfloor d/2\rfloor$-neighborly. Selecting any $\le k$ objects with one query is translated into "finding a non-negative polynomial that is zero only on $S$," reducing combinatorial subset selection to polynomial construction.

2.  **VC Dimension Lower Bound + Radon Sharpening: Tightness of MED**:
    *   **Function**: Proves that for all three scoring functions, MED is at least $k-1$; further provides the exact formula $\mathrm{MED}(m,k;\mathcal{F}_{\rm linear})=\min\{2k,m-1\}$ for the inner product case.
    *   **Mechanism**: Defines a binary threshold class $\mathcal{C}_{\mathcal{F},n}$ induced by $k$-shattering and proves $\text{MED}(m,k;\mathcal{F})\ge \text{VCD}^{-1}(k;\mathcal{F})$. Since the VC dimension of $\mathcal{F}_{\rm linear},\mathcal{F}_{\cos},\mathcal{F}_{\ell_2}$ is $n+1$, MED $\ge k-1$. Radon’s theorem is then used to show that if $d<\min\{2k,m-1\}$, shattering is impossible as some subset pairs will inevitably have intersecting convex hulls.
    *   **Design Motivation**: VC dimension provides a general lower bound applicable to any scoring class, while Radon sharpening refines the inner product case to a constant level. Together, they bound the theory within $[k-1, 2k]$.

3.  **Gaussian Centroid Construction + Margin Ceiling: Bounding Robust RMED**:
    *   **Function**: Under unit sphere normalization and a requirement that selected objects score at least $\epsilon$ higher than unselected ones, it proves the feasible margin is capped by $\epsilon_\star(m,k)=m/\sqrt{k(m-1)(m-k)}$ (approximately $1/\sqrt{k}$ for large $m$). An $O(k^2\log m)$ existence construction is provided for margin $\epsilon=c/\sqrt{k}$.
    *   **Mechanism**: The ceiling is derived via variance identities. If all $k$-subset queries reach margin $\epsilon$, then $\|\bar{\bm{v}}_S-\bar{\bm{v}}\|_2\ge\frac{m-k}{m}\epsilon$ must hold for all $S$. Taking the expectation over random subsets and using unit norm properties $\frac{1}{m}\sum\|\bm{v}_i-\bar{\bm{v}}\|^2\le 1$ leads to $\epsilon\le\epsilon_\star$. For the upper bound, $m$ isotropic Gaussian vectors are sampled and normalized. The query $\bm{u}_S\propto\sum_{i\in S}\bm{v}_i$ is the normalized centroid. In $n=Ck^2\log m$ dimensions, pair-wise inner products are $O(1/k)$, ensuring selected objects have $\Theta(1)$ "auto-correlation" while external noise is $O(|S|/k)$, yielding a consistent margin of $\Omega(1/\sqrt{k})$.
    *   **Design Motivation**: This distinguishes between "exact geometric expressivity" and "practical engineering difficulty." Since exact MED is independent of $m$, assertions that "large $m$ requires higher dimensions" are formally incorrect in an exact sense. However, the introduction of a positive margin (implicit in robust retrieval) reintroduces $m$ into the dimension formula via packing lower bounds.

### Loss & Training
The theoretical work does not have a learning objective. Synthetic experiments in Section 5 utilize hinge loss with Adam optimization to find centroid GD witnesses. Deterministic cyclic polytope and LIMIT constructions serve as baselines. The "random additive" baseline on LIMIT datasets is label-unaware: each token is assigned a fixed unit Gaussian vector, and document/query embeddings are sums of token vectors $\phi(x)=\sum_{t\in\tau(x)}G_t$ without supervised training.

## Key Experimental Results

### Main Results

Synthetic top-2 witness: The minimum dimension for a "successful witness" under universe size $m$ is plotted, comparing cyclic polytope construction, centroid GD optimization, and the WBNL fitting curve.

| Setting (top-2, universe $m$) | Cyclic Polytope Construction | Centroid GD Fitting | WBNL Fitting Curve |
|---|---|---|---|
| Any $m$ | Dimension $=4$ (Independent of $m$) | $d\sim\log_2 m$ (Slow growth) | $d$ grows polynomially with $m$ |

LIMIT / LIMIT-small Recall@2 (Single-vector retrieval):

| Dataset | Tokenizer | $d$ | Recall@2 | Promptriever 8B @ 4096 |
|---|---|---|---|---|
| LIMIT | handmade | 256 | Exceeds baseline | 0.030 |
| LIMIT | vanilla | 512 | Exceeds baseline | 0.030 |
| LIMIT @ 4096 | qwen | 4096 | 0.2675 | 0.030 |
| LIMIT-small @ 4096 | handmade | 4096 | 1.0000 | 0.543 |

### Ablation Study

"Ablations" between theorems consist of comparisons across three regimes:

| Regime | Dimension Upper Bound | Depends on $m$ | Key Tool |
|---|---|---|---|
| Exact MED (No margin) | $\min\{2k,m-1\}$, $\Theta(k)$ | No | Cyclic Polytope |
| Robust RMED ($\epsilon \sim 1/\sqrt{k}$) | $O(k^2\log m)$ | Yes ($\log m$) | Gaussian Centroid |
| Robust RMED ($\epsilon > \epsilon_\star$) | $\infty$ | Yes | Variance Identity |

### Key Findings
*   Cyclic polytopes can exactly overfit any LIMIT-small top-2 dataset at $d=4$, debunking the "high dimensions are insufficient" geometric narrative.
*   A vanilla tokenizer with random additive vectors at 512 dimensions outperforms a 4096-dimensional Promptriever 8B on LIMIT. This suggests the failure of learned models on LIMIT stems from tokenizers, objectives, or optimization rather than geometric capacity.
*   Robustness is constrained by margin: $\epsilon_\star(m,k)\sim 1/\sqrt{k}$ is an absolute hard ceiling. Exceeding this margin cannot be solved by increasing dimensionality.

## Highlights & Insights
*   Separating "geometric expressivity" from "optimization searchability" clarifies pessimistic conclusions in the dense retrieval community.
*   The Cyclic Polytope + Squared Polynomial construction is elegant, converting subset selection into polynomial roots.
*   Gaussian centroid $O(k^2\log m)$ provides a natural geometric explanation for mean-pooled representations like DPR/Contrastive learning.
*   The feasible margin ceiling $\epsilon_\star(m,k)$ provides a theoretical guide for temperature/margin selection in contrastive loss.

## Limitations & Future Work
*   There remains a gap between the $\Omega(k)$ lower bound and the $O(k^2\log m)$ upper bound for RMED.
*   Cyclic polytope constructions have poor numerical conditions and tiny margins, making them unsuitable for direct engineering deployment.
*   The theory assumes queries can be chosen arbitrarily for each $S$, whereas real systems use neural encoders.
*   Experiments are restricted to LIMIT/LIMIT-small; full comparisons on MS MARCO/BEIR are not included.

## Related Work & Insights
*   **vs Weller et al. 2026 (WBNL)**: This paper proves WBNL's polynomial growth curve is an optimization failure, providing a $d=4$ counter-example.
*   **vs Guo et al. 2019**: Guo investigated structured multi-class embedding. This paper explores unstructured top-$k$ subsets where the number of classes $\sum \binom{m}{i}$ is significantly larger.
*   **vs Reimers & Gurevych 2021**: Provides a theoretical complement to empirical studies on the curse of dimensionality in dense retrieval, attributing observed dimension dependencies to robust margins rather than exact embeddability.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ 
*   Experimental Thoroughness: ⭐⭐⭐⭐ 
*   Writing Quality: ⭐⭐⭐⭐⭐ 
*   Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Quiver: Quantum-Informed Views for Enhanced Representations in Large ML Models](quiver_quantum-informed_views_for_enhanced_representations_in_large_ml_models.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[AAAI 2026\] Just Few States are Enough: Randomized Sparse Feedback for Stability of Dynamical Systems](../../AAAI2026/physics/just_few_states_are_enough_randomized_sparse_feedback_for_stability_of_dynamical.md)
- [\[NeurIPS 2025\] A Variational Manifold Embedding Framework for Nonlinear Dimensionality Reduction](../../NeurIPS2025/physics/a_variational_manifold_embedding_framework_for_nonlinear_dimensionality_reductio.md)
- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/physics/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)

</div>

<!-- RELATED:END -->
