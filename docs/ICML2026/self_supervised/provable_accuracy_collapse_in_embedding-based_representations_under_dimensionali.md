---
title: >-
  [Paper Note] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch
description: >-
  [ICML 2026][Self-Supervised Learning][triplet embedding] The authors prove: For typical triplet tasks in contrastive learning, as long as the embedding dimension $d$ is less than a constant multiple of the true dimension…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "triplet embedding"
  - "dimensional collapse"
  - "VC dimension"
  - "Unique Games Conjecture"
  - "inapproximability"
date: 2026-05-08
content_hash: 36889cbb63f52cc3
---

# Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch

**Conference**: ICML 2026  
**arXiv**: [2605.03346](https://arxiv.org/abs/2605.03346)  
**Code**: None  
**Area**: Representation Learning Theory / Contrastive Learning / Embedding Dimension  
**Keywords**: triplet embedding, dimensional collapse, VC dimension, Unique Games Conjecture, inapproximability

## TL;DR
The authors prove: For typical triplet tasks in contrastive learning, as long as the embedding dimension $d$ is less than a constant multiple of the true dimension $D$, the accuracy will "collapse" to the 50% baseline of a 1D random embedding, regardless of the optimizer used. Moreover, algorithmically, under the Unique Games Conjecture, this cannot be approximated in polynomial time.

## Background & Motivation

**Background**: From Word2Vec and SimCLR to modern foundation models, mapping data points to $\mathbb R^d$ via contrastive/triplet embedding is standard in representation learning. The choice of $d$ ranges from hundreds to thousands; large models often use a 3072-dimensional latent space, but downstream tasks truncate to 128 dimensions to save storage/retrieval costs (e.g., Matryoshka embeddings).

**Limitations of Prior Work**: Recent empirical studies (Takeshita 2025, Tsukagoshi 2025) observed a universal phenomenon across 6 SOTA text encoders and 26 downstream tasks—truncating 50% of the dimensions results in <10% performance drop, but truncating ~90% leads to a cliff-like accuracy drop. This "dimension threshold" phenomenon lacks any theoretical explanation.

**Key Challenge**: The classic Johnson-Lindenstrauss lemma tells us that distance *values* can be preserved in $O(\log n)$ dimensions, but ordinal embedding aims to preserve distance *orderings*. Any $(1\pm\varepsilon)$-distortion will flip a large number of triplets (Alon 2008), so JL-type tools are inapplicable.

**Goal**: To formalize two questions—(1) Given a triplet instance perfectly realizable in dimension $D$, how small can $d$ be before collapse occurs? (2) For non-realizable instances, is there any polynomial-time algorithm that can consistently exceed the 50% baseline?

**Key Insight**: Treat triplet embedding as a hypothesis class and conduct a tight VC-dimension analysis (using Alon 2024's $\Theta(nd)$ upper bound); simultaneously, perform a gap-preserving reduction from triplet embedding to Maximum Acyclic Subgraph (MAS), directly leveraging Khot's UGC hardness of approximation result.

**Core Idea**: $m=\Theta(Dn)$ triplets simultaneously have two properties—(i) with high probability, they are perfectly realizable in $D$ dimensions; (ii) with high probability, in $d=c\varepsilon^2 D$ dimensions, any embedding achieves at most $1/2+\varepsilon$ accuracy, yielding a sharp dimension-accuracy cliff.

## Method

### Overall Architecture
The paper centers on two theoretical results and a set of synthetic experiments: (i) On the information-theoretic side, use probabilistic methods to construct $m=c_1 Dn$ random triplets, proving such instances are *simultaneously* realizable in $D$ dimensions and have accuracy $\leq 1/2+\varepsilon$ in $c_2\varepsilon^2 D$ dimensions; (ii) On the computational complexity side, use gap reduction to embed MAS into triplet embedding, yielding NP-hardness under UGC; (iii) On the experimental side, use AdamW + hinge triplet loss on synthetic data to verify the accuracy cliff as $d/D$ varies.

### Key Designs

1. **Graph-Theoretic Characterization of Realizability**:

    - **Function**: Prove that randomly generated dense triplet instances *can* be perfectly embedded in $D$ dimensions.
    - **Mechanism**: Construct a directed multigraph $\mathcal G_{\text{MAS}}(n,\lambda)$ over $V\times V$, where vertices are $\binom{V}{2}$ distance pairs, and each triplet $(x,y^+,z^-)$ corresponds to a directed edge $\{x,y\}\to\{x,z\}$. Bilu-Linial's equivalence states: an instance is realizable in $n$ dimensions ⇔ the graph is acyclic. Using the first moment method, show that for $\lambda=o(n^{-3/2})$, acyclicity holds with high probability. Then, using Avdiukhin 2024's arboricity bound, reduce the dimension from $n$ to $4\rho(G_c)$, where $\rho$ is the arboricity of the constraint graph; it can be shown that $\rho(G_c)\leq 30 c_1' D$, so $120 c_1' D=D$ dimensions suffice.
    - **Design Motivation**: It is necessary to rule out the trivial explanation that "the instance itself is contradictory" before attributing subsequent collapse to *insufficient dimension* rather than realizability.

2. **VC-Dimension-Based Collapse Theorem**:

    - **Function**: When the dimension is insufficient, *any* embedding (regardless of optimizer/loss/architecture) cannot exceed the random baseline.
    - **Mechanism**: Treat each embedding $f:V\to\mathbb R^d$ as a hypothesis $h_f(x,y,z)\in\{0,1\}$; Alon 2024 has shown $\text{VC}(\mathcal H)\leq cnd$. Construct a uniform and *randomly labeled* distribution $\mathcal D$ over $V^3\times\{0,1\}$, so that $m$ samples from it match the distribution of $\mathcal I(n,m)$. By Shalev-Shwartz–Ben-David's uniform convergence: for $m\geq C\text{VC}/\varepsilon^2$, $|acc(f)-1/2|\leq\varepsilon$ holds uniformly for all $f$. Substituting $m=c_1 Dn$, $d=c_2\varepsilon^2 D$ yields the result.
    - **Design Motivation**: Use uniform convergence of VC dimension to turn an "existential" problem into a "universal" one—proving for one is equivalent to proving for all. This approach is naturally independent of the specific optimization method.

3. **Gap-Preserving Reduction from MAS to Triplet Embedding**:

    - **Function**: Transfer the approximation resistance of MAS under UGC directly to triplet embedding.
    - **Mechanism**: Given a MAS instance $G(V,E)$, add an anchor $S$, and translate each directed edge $u\to v$ into a triplet $(S, u, v)$, meaning "$u$ should be closer to $S$ than $v$." For any $d$-dimensional embedding $f$, sorting by $r_f(v)=\|f(v)-f(S)\|_2$ yields a total order $\pi_f$, and each triplet is satisfied ⇔ $\pi_f(u)<\pi_f(v)$. Conversely, any total order $\pi$ corresponds to a 1D embedding $f_\pi(v)=\pi(v)$. Thus, the optimal values on both sides are exactly equal, and the $1-\varepsilon$ vs $1/2+\varepsilon$ gap is precisely preserved under UGC.
    - **Design Motivation**: Mapping a new geometric CSP to a classical CSP already proven approximation-resistant is a standard technique for "cheaply" transferring Khot's hardness results to new problems; here, the reduction is completely independent of the available dimension $d$, meaning increasing the dimension does not help.

### Loss & Training
In synthetic experiments, hinge triplet loss is used: $\mathcal L=\max(0,\|f(i)-f(j)\|_2^2-\|f(i)-f(k)\|_2^2+\gamma)$, with $\gamma=1$, optimized by AdamW. Two types of data: (1) $D\in\{128,256,512,1024\}$, $n=1000$ points uniformly sampled on the unit sphere + $10^6$ triplets labeled by true distances; (2) $n=4000$ random triplets (empirically verifiable for realizability). Embeddings are of two types: unconstrained and projected onto the unit sphere.

## Key Experimental Results

### Main Results

The accuracy cliff observed in synthetic experiments (summarized from Figures 1/2):

| Ground-truth $D$ | $d/D \approx 5\%$ | $d/D \approx 50\%$ | $d \geq D$ |
|---|---|---|---|
| 128 / 256 / 512 / 1024 | $\approx 1/2+\varepsilon$, $\varepsilon\approx 22\%$ | Near perfect | 1.0 |

(Both unconstrained and spherical embeddings show the same cliff location, consistent with the theoretical $d=c\varepsilon^2 D$.)

### Ablation Study

| Setting | Phenomenon | Meaning |
|---|---|---|
| Spherical projection vs. unconstrained | Same cliff location | Dimension, not norm, is the bottleneck |
| Ground-truth geometry vs. random triplets | Both collapse | Independent of specific geometry |
| AdamW with different initializations | Still collapses | Optimizer-independent, rules out non-convexity as cause |

### Key Findings
- The experimental cliff matches theoretical predictions: at $d/D\approx 5\%$, $\varepsilon^2\approx 5\%$ so $\varepsilon\approx 22\%$, actual accuracy ≈ 72%, almost equal to $1/2+\varepsilon$.
- The marginal benefit curve for "increasing dimension" is highly nonlinear: after $d\geq D$, there is almost no further improvement; below $d<cD$, accuracy almost instantly collapses to 50%, contrary to the common engineering intuition that "more dimensions are always better."
- The algorithmic hardness result implies: even with polynomial time and arbitrarily high dimension, it is impossible to consistently exceed the 1D random embedding baseline, highlighting the necessity of *input structure assumptions* (margin/separability).

## Highlights & Insights
- Provides a *sharp* constant-factor lower bound for the long-assumed but theoretically unexplained "dimension truncation causes cliff-like drops" phenomenon, with textbook-level clarity.
- Combines statistical learning theory (VC dimension + uniform convergence) and approximation algorithms (UGC + MAS approximation resistance), delivering both information-theoretic and computational lower bounds in a single paper, with clean arguments.
- The gap reduction design is extremely simple—one anchor translates any MAS instance into triplets, with obvious equivalence; this "translation trick" is worth reusing in new geometric problems.

## Limitations & Future Work
- The lower bound is a mix of worst-case and average-case: does not rule out that "adding margin, separability, or manifold structure" can break the bound; whether real data lies near hard instances is unknown.
- Experiments are only on synthetic data; how the "$d^*$" for real text/image/retrieval data varies with distribution parameters is a topic for future work.
- No computable recommendation for "how many dimensions to use"—only a lower bound that "less than $cD$ will collapse," with $c$ still at the theoretical constant level.

## Related Work & Insights
- **vs JL lemma**: JL preserves distances in $O(\log n/\varepsilon^2)$ dimensions, but this work proves that preserving orderings is fundamentally impossible with such compression, highlighting the essential difference in dimension requirements between ordinal and metric embedding.
- **vs Bilu-Linial / Avdiukhin (realizable triplet embedding)**: They showed $O(\min(n-1,\sqrt m))$ dimensions always suffice; this work complements with the reverse—"below a certain constant factor of $D$, performance completely fails," pinning down both ends of the dimension-accuracy curve.
- **vs Matryoshka representation learning (Kusupati 2022)**: Matryoshka empirically trains nested, truncatable embeddings; this work provides a theoretical explanation for "why a certain minimum dimension is necessary" for utility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide sharp information-theoretic + computational lower bounds for the contrastive embedding dimension threshold.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments strongly support the theory, but lack real dataset follow-up.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations, elegantly separated reduction and probabilistic arguments.
- Value: ⭐⭐⭐⭐ Provides fundamental theoretical guidance for embedding dimension selection, and opens new directions for exploring "how input structure breaks the lower bound."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2025\] Contextures: Representations from Contexts](../../ICML2025/self_supervised/contextures_representations_from_contexts.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)
- [\[NeurIPS 2025\] Angular Constraint Embedding via SpherePair Loss for Constrained Clustering](../../NeurIPS2025/self_supervised/angular_constraint_embedding_via_spherepair_loss_for_constrained_clustering.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)

</div>

<!-- RELATED:END -->
