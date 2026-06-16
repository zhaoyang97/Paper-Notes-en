---
title: >-
  [Paper Note] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch
description: >-
  [ICML 2026][Self-Supervised Learning][triplet embedding] The authors prove that in typical triplet tasks within contrastive learning, if the embedding dimension $d$ is less than a certain constant multiple of the true dimension $D$, the accuracy "collapses" to the 50% baseline (equivalent to a 1D random embedding) regardless of the optimizer. Furthermore, this phenomenon is
tags:
  - ICML 2026
  - Self-Supervised Learning
  - triplet embedding
date: 2026-05-08
content_hash: d463edb1323307e5
---
# Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.03346](https://arxiv.org/abs/2605.03346)  
**Code**: None  
**Area**: Representation Learning Theory / Contrastive Learning / Embedding Dimension  
**Keywords**: triplet embedding, dimensionality collapse, VC dimension, Unique Games Conjecture, inapproximability

## TL;DR
The authors prove that in typical triplet tasks within contrastive learning, if the embedding dimension $d$ is less than a certain constant multiple of the true dimension $D$, the accuracy "collapses" to the 50% baseline (equivalent to a 1D random embedding) regardless of the optimizer. Furthermore, this phenomenon is shown to be hard to approximate in polynomial time under the Unique Games Conjecture.

## Background & Motivation

**Background**: From Word2Vec and SimCLR to modern foundation models, mapping data points into $\mathbb R^d$ via contrastive or triplet embeddings is a standard practice in representation learning. Choices for $d$ range from hundreds to thousands; large models often use a 3072-dimensional latent space but may truncate to 128 dimensions to save storage and retrieval costs (e.g., Matryoshka embeddings).

**Limitations of Prior Work**: Recent empirical studies (Takeshita 2025, Tsukagoshi 2025) across 6 SOTA text encoders and 26 downstream tasks observed a universal phenomenon: truncating 50% of the dimensions results in $<10\%$ performance loss, but truncating $\sim 90\%$ leads to a precipitous drop in accuracy. This "dimensionality threshold" phenomenon lacks a theoretical explanation.

**Key Challenge**: While the classic Johnson-Lindenstrauss lemma suggests distance *values* can be preserved in $O(\log n)$ dimensions, ordinal embeddings must preserve distance *rankings*. Any $(1\pm\varepsilon)$-distortion can flip a massive number of triplets (Alon 2008), rendering JL-style tools inapplicable.

**Goal**: To formalize two questions: (1) Given a triplet instance perfectly realizable in dimension $D$, at what point does accuracy collapse as $d$ decreases? (2) For non-realizable instances, does a polynomial-time algorithm exist that can consistently exceed the 50% baseline?

**Key Insight**: Triplet embedding is treated as a hypothesis class, allowing for a tight VC-dimension analysis (leveraging the $\Theta(nd)$ upper bound from Alon 2024). Simultaneously, triplet embedding is linked to the Maximum Acyclic Subgraph (MAS) problem via a gap-preserving reduction, inheriting Khot’s UGC hardness-of-approximation results.

**Core Idea**: A set of $m=\Theta(Dn)$ triplets can simultaneously possess two properties: (i) high probability of being perfectly realizable in $D$ dimensions; (ii) high probability that any embedding in $d=c\varepsilon^2 D$ dimensions satisfies no more than $1/2+\varepsilon$ of the triplets. This establishes a sharp dimension-accuracy cliff.

## Method

### Overall Architecture
The core of the paper consists of two theoretical results and a set of synthetic experiments: (i) From an information-theoretic perspective, the probabilistic method is used to construct $m=c_1 Dn$ random triplets that are *simultaneously* realizable in $D$ dimensions but restricted to accuracy $\leq 1/2+\varepsilon$ in $c_2\varepsilon^2 D$ dimensions; (ii) On the computational complexity side, a gap reduction maps MAS to triplet embedding, yielding NP-hardness under UGC; (iii) Experimentally, AdamW with hinge triplet loss is used on synthetic data to verify the accuracy cliff relative to $d/D$.

### Key Designs

The "methodology" components represent three independent proofs, where the first two form the information-theoretic lower bound (Theorem 1.3) and the third provides the computational lower bound (Theorem 1.4).

**1. Graph-Theoretic Characterization of Realizability: Proving "Random Dense Instances Are Realizable in $D$ Dimensions"**

To attribute accuracy collapse to *insufficient dimensionality*, one must first rule out the trivial explanation that the instance itself is contradictory. The authors use the Bilu-Linial equivalence to translate this into graph theory: on a directed multigraph $\mathcal G_{\text{MAS}}(n,\lambda)$ where vertices represent $\binom{V}{2}$ distance pairs, each triplet $(x,y^+,z^-)$ corresponds to a directed edge $\{x,y\}\to\{x,z\}$. Precision in $n$ dimensions is equivalent to the graph being an acyclic directed graph (DAG). Using the first moment method, they prove the graph is acyclic with high probability when $\lambda=o(n^{-3/2})$ (corresponding to $D=o(\sqrt n)$). By further proving the graph's arboricity is $O(D)$ and employing the algorithm from Avdiukhin 2024, they compress the required dimension from $n$ back to $\Theta(D)$, confirming the instance is perfectly satisfiable in $D$ dimensions.

**2. VC-Dimension Uniform Convergence: Proving "No Embedding Can Succeed"**

The collapse theorem requires a *universal* conclusion—independent of optimizer, loss, or architecture—that *all* $d$-dimensional embeddings fail to beat the random baseline. The authors use statistical learning theory: each embedding $f:V\to\mathbb R^d$ is viewed as a hypothesis $h_f(x,y,z)\in\{0,1\}$. Given the VC dimension is $\Theta(nd)$, they construct a distribution $\mathcal D$ over $V^3\times\{0,1\}$ where triplets are uniform and labels are *purely random*. Here, empirical risk represents the triplet accuracy. Applying the uniform convergence bound $m\geq C\,\text{VC}/\varepsilon^2$, for $m=\Theta(Dn)$ and $d=\Theta(\varepsilon^2 D)$, the bound $|\text{acc}(f)-1/2|\leq\varepsilon$ holds for *every* $f$ simultaneously.

**3. MAS → Triplet Embedding Gap-Preserving Reduction: Importing UGC Hardness**

To address whether poly-time algorithms can beat the $50\%$ baseline on noisy instances, the authors perform a reduction from Maximum Acyclic Subgraph (MAS), which is known to be approximation-resistant. Given a MAS instance $G(V,E)$, an anchor $S$ is introduced. Each directed edge $u\to v$ is converted to a triplet $(S, u, v)$, implying "$u$ should be closer to $S$ than $v$". For any $d$-dimensional embedding $f$, sorting by $r_f(v)=\|f(v)-f(S)\|_2$ yields a total ordering $\pi_f$. A triplet is satisfied if and only if $\pi_f(u)<\pi_f(v)$. Conversely, any total ordering can be realized by a 1D embedding. The optimal values match exactly, transferring the $1-\varepsilon$ vs. $1/2+\varepsilon$ inapproximability gap from MAS to triplet embedding. This reduction is *independent of the dimension $d$ available to the algorithm*.

### Loss & Training
Synthetic experiments use the hinge triplet loss $\mathcal L=\max(0,\|f(i)-f(j)\|_2^2-\|f(i)-f(k)\|_2^2+\gamma)$ with $\gamma=1$ and AdamW optimization. Two data types: (1) $n=1000$ points sampled uniformly on a $D$-dimensional unit sphere ($\in\{128, 256, 512, 1024\}$) with $10^6$ triplets; (2) random triplets on $n=4000$ points. Embeddings are either unconstrained or projected onto the unit sphere.

## Key Experimental Results

### Main Results

Summary of the accuracy cliff observed in synthetic experiments (generalized from Figures 1/2):

| Ground-truth $D$ | $d/D \approx 5\%$ | $d/D \approx 50\%$ | $d \geq D$ |
|---|---|---|---|
| 128 / 256 / 512 / 1024 | $\approx 1/2+\varepsilon$, $\varepsilon\approx 22\%$ | Near Perfect | 1.0 |

(Both unconstrained and spherical embeddings show the same cliff position, consistent with the theoretical $d=c\varepsilon^2 D$.)

### Ablation Study

| Setup | Phenomenon | Implication |
|---|---|---|
| Spherical vs. Unconstrained | Identical cliff position | Dimension, not norm, is the bottleneck |
| Ground-truth Geometry vs. Random | Both collapse | Independent of specific geometry |
| Different AdamW Initializations | Persistent collapse | Independent of optimizer; rules out local minima issues |

### Key Findings
- Experimental cliffs align with theoretical predictions: at $d/D\approx 5\%$, $\varepsilon^2\approx 5\%$ implies $\varepsilon\approx 22\%$, resulting in $\text{acc} \approx 72\%$, which is very close to $1/2+\varepsilon$.
- The marginal utility of increasing dimension is highly non-linear: performance plateaus after $d \geq D$ and collapses almost instantly once $d < cD$, contradicting the "more dimensions are always better" heuristic.
- The algorithmic hardness result implies that even with polynomial time and arbitrarily high dimensions, it may be impossible to consistently beat a 1D random embedding, highlighting the necessity of *input structure assumptions* (marginal/separability).

## Highlights & Insights
- Provides a *sharp* constant-factor lower bound for the "performance cliff" in dimension truncation, a phenomenon long acknowledged by engineers but previously lacking theoretical grounding.
- Cleanly integrates statistical learning theory (VC dimension + uniform convergence) and approximation theory (UGC + MAS) to provide simultaneous information-theoretic and computational complexity lower bounds.
- The gap reduction design is elegant—using a single anchor to translate any MAS instance into triplets proves that "increasing dimensions" cannot bypass fundamental algorithmic limits.

## Limitations & Future Work
- The lower bound is a mix of worst-case and average-case results: it does not preclude surpassing the limit by adding margin, separability, or manifold structures. Whether real-world data resembles these "hard" instances remains unknown.
- Experiments are restricted to synthetic data; how the optimal dimension $d^*$ varies with distribution parameters in real text or image data is a subject for future study.
- Provides no computable advice on the exact dimension to use—only the lower bound "collapse occurs below $cD$," where $c$ remains a theoretical constant.

## Related Work & Insights
- **vs JL lemma**: While JL preserves distances in $O(\log n/\varepsilon^2)$, this work proves rank preservation cannot achieve such compression, highlighting the fundamental difference between ordinal and metric embedding.
- **vs Bilu-Linial / Avdiukhin**: While they proved $O(\min(n-1,\sqrt m))$ dimensions are sufficient for realizability, this work provides the converse: dimensions below a constant factor of $D$ lead to total failure.
- **vs Matryoshka representation learning (Kusupati 2022)**: This work provides a theoretical backdrop for why nested embeddings must reach a certain dimensionality threshold to become effective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide sharp information-theoretic and computational lower bounds for triplet embedding thresholds.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments strongly support the theory, but lacks real-world dataset follow-up.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation; reduction and probabilistic arguments are elegantly separated.
- Value: ⭐⭐⭐⭐ Provides fundamental theoretical guidance for embedding dimension selection and opens new avenues for exploring how input structures can break these lower bounds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Geometry of Projection Heads: Conditioning, Invariance and Collapse](the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2025\] Contextures: Representations from Contexts](../../ICML2025/self_supervised/contextures_representations_from_contexts.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)

</div>

<!-- RELATED:END -->
