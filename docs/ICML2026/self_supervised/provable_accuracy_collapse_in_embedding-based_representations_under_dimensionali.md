---
title: >-
  [Paper Note] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch
description: >-
  [ICML 2026][Self-Supervised Learning][triplet embedding] The authors prove that in typical triplet tasks within contrastive learning, as long as the embedding dimension $d$ is less than a certain constant multiple of the…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "triplet embedding"
  - "dimensionality collapse"
  - "VC dimension"
  - "Unique Games Conjecture"
  - "inapproximability"
date: 2026-05-08
content_hash: d9505f96dfc98ae9
---

# Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch

**Conference**: ICML 2026  
**arXiv**: [2605.03346](https://arxiv.org/abs/2605.03346)  
**Code**: None  
**Area**: Representation Learning Theory / Contrastive Learning / Embedding Dimensions  
**Keywords**: triplet embedding, dimensionality collapse, VC dimension, Unique Games Conjecture, inapproximability

## TL;DR
The authors prove that in typical triplet tasks within contrastive learning, as long as the embedding dimension $d$ is less than a certain constant multiple of the true dimension $D$, the accuracy "collapses" to the 50% baseline of a 1D random embedding regardless of the optimizer used. Furthermore, from an algorithmic perspective, this phenomenon is inapproximable in polynomial time under the Unique Games Conjecture.

## Background & Motivation

**Background**: From Word2Vec and SimCLR to modern foundation models, mapping data points to $\mathbb R^d$ via contrastive/triplet embeddings is a standard configuration in representation learning. The choice of $d$ varies from hundreds to thousands; large models often use 3072-dimensional latent spaces but truncate them to 128 dimensions for downstream tasks to reduce storage and retrieval costs (e.g., Matryoshka embeddings).

**Limitations of Prior Work**: Recent empirical studies (Takeshita 2025, Tsukagoshi 2025) across 6 SOTA text encoders and 26 downstream tasks observed a universal phenomenon: truncating 50% of the dimensions results in a performance drop of $<10\%$, but truncating to ~90% causes accuracy to fall off a cliff. This "dimensionality threshold" phenomenon lacks any theoretical explanation.

**Key Challenge**: The classic Johnson-Lindenstrauss lemma suggests that distance *values* can be preserved in $O(\log n)$ dimensions, but ordinal embeddings must preserve the *ranking* of distances. Since any $(1±\varepsilon)$-distortion can flip a large number of triplets (Alon 2008), JL-type tools are entirely inapplicable.

**Goal**: To formalize two questions: (1) Given any triplet instance that is perfectly satisfiable in dimension $D$, how small can $d$ be before collapse occurs? (2) For non-realizable instances, is there a polynomial-time algorithm that can consistently exceed the 50% baseline?

**Key Insight**: Triplet embedding is treated as a hypothesis class to perform a tight VC-dimension analysis (leveraging the $\Theta(nd)$ upper bound from Alon 2024). Simultaneously, a gap-preserving reduction is established between triplet embedding and the Maximum Acyclic Subgraph (MAS) problem, directly linking the work to Khot’s UGC hardness-of-approximation results.

**Core Idea**: $m=\Theta(Dn)$ triplets simultaneously possess two properties: (i) they are perfectly realizable in $D$ dimensions with high probability; (ii) in $d=c\varepsilon^2 D$ dimensions, any embedding satisfies no more than $1/2+\varepsilon$ of the triplets with high probability. This provides a sharp dimension-accuracy cliff.

## Method

### Overall Architecture
The core of the paper consists of two theoretical results and a set of synthetic experiments: (i) On the information-theoretic side, a probabilistic method is used to construct $m=c_1 Dn$ random triplets, proving that such instances are *simultaneously* realizable in $D$ dimensions and yield an accuracy of $\leq 1/2+\varepsilon$ in $c_2\varepsilon^2 D$ dimensions; (ii) On the computational complexity side, a gap reduction maps MAS into triplet embedding to derive NP-hardness under UGC; (iii) On the experimental side, AdamW + hinge triplet loss is used on synthetic data to verify the collapse of accuracy relative to $d/D$.

### Key Designs

1.  **Realizability Characterization via Graph Theory**:
    - **Function**: To prove that densely sampled random triplet instances *can* be perfectly embedded in $D$ dimensions.
    - **Mechanism**: A directed multigraph $\mathcal G_{\text{MAS}}(n,\lambda)$ is constructed over $V\times V$, where vertices are $\binom{V}{2}$ distance pairs, and each triplet $(x,y^+,z^-)$ corresponds to a directed edge $\{x,y\}\to\{x,z\}$. The Bilu-Linial characterization states that an instance is realizable in $n$ dimensions ⇔ this graph is acyclic. First-moment methods prove that acyclicity holds with high probability when $\lambda=o(n^{-3/2})$. Avdiukhin 2024’s arboricity bounds are then used to compress the dimension from $n$ to $4\rho(G_c)$, where $\rho$ is the arboricity of the constraint graph, proving that $D$ dimensions are sufficient.
    - **Design Motivation**: It is necessary to first rule out the trivial explanation that the "instance itself is contradictory" to attribute the subsequent collapse to *insufficient dimensionality* rather than non-realizability.

2.  **Collapse Theorem via VC-dimension**:
    - **Function**: To show that when dimensions are insufficient, *any* embedding (regardless of optimizer, loss, or architecture) cannot exceed the random baseline.
    - **Mechanism**: Each embedding $f:V\to\mathbb R^d$ is viewed as a hypothesis $h_f(x,y,z)\in\{0,1\}$. Alon 2024 established $\text{VC}(\mathcal H)\leq cnd$. A distribution $\mathcal D$ on $V^3\times\{0,1\}$ is constructed with uniform and *random labels*, such that $m$ samples follow the same distribution as $\mathcal I(n,m)$. By the uniform convergence principle of Shalev-Shwartz–Ben-David, $|acc(f)-1/2|\leq\varepsilon$ holds consistently for all $f$ when $m\geq C\text{VC}/\varepsilon^2$. Substituting $m=c_1 Dn$ and $d=c_2\varepsilon^2 D$ completes the proof.
    - **Design Motivation**: Using uniform convergence turns an "existence" problem into a "universal" one—proving for one is equivalent to proving for all. This approach is naturally independent of specific optimization methods.

3.  **MAS → triplet embedding gap-preserving reduction**:
    - **Function**: To transfer the approximation resistance of MAS under UGC directly to triplet embedding.
    - **Mechanism**: Given an MAS instance $G(V,E)$, an anchor $S$ is added, and each directed edge $u\to v$ is translated into a triplet $(S, u, v)$, implying "$u$ should be closer to $S$ than $v$". For any $d$-dimensional embedding $f$, a total order $\pi_f$ is obtained by sorting $r_f(v)=\|f(v)-f(S)\|_2$, and a triplet is satisfied ⇔ $\pi_f(u)<\pi_f(v)$. Conversely, any total order $\pi$ corresponds to a 1-dimensional embedding $f_\pi(v)=\pi(v)$. Thus, the optimal values are equal, and the gap between $1-\varepsilon$ and $1/2+\varepsilon$ under UGC is precisely maintained.
    - **Design Motivation**: Establishing a one-to-one correspondence between a new geometric CSP and a classic CSP proven to be approximation-resistant is a standard technique to "cheaply" migrate Khot's hardness results; here, the reduction is independent of the available dimension $d$, meaning dimensionality expansion does not help.

### Loss & Training
In synthetic experiments, hinge triplet loss $\mathcal L=\max(0,\|f(i)-f(j)\|_2^2-\|f(i)-f(k)\|_2^2+\gamma)$ is used with $\gamma=1$ and AdamW optimization. Two types of data are used: first, $n=1000$ points sampled uniformly on a unit sphere with $D\in\{128,256,512,1024\}$ and $10^6$ triplets labeled by true distance; second, random triplets on $n=4000$ points. Embeddings are either unconstrained or forced via projection onto the unit sphere.

## Key Experimental Results

### Main Results

Accuracy cliffs observed in synthetic experiments (summarized from Figures 1/2):

| Ground-truth $D$ | $d/D \approx 5\%$ | $d/D \approx 50\%$ | $d \geq D$ |
|---|---|---|---|
| 128 / 256 / 512 / 1024 | $\approx 1/2+\varepsilon$, $\varepsilon\approx 22\%$ | Near perfect | 1.0 |

(Both unconstrained and spherical embedding settings exhibit the same cliff location, consistent with the theoretical $d=c\varepsilon^2 D$.)

### Ablation Study

| Setting | Phenomenon | Meaning |
|---|---|---|
| Spherical vs. Unconstrained | Identical cliff location | Dimension, not norm, is the bottleneck |
| Ground-truth geometry vs. Random triplets | Both collapse | Independent of specific geometry |
| Different AdamW initializations | Still collapses | Independent of optimizer; rules out non-convex stalling |

### Key Findings
- Experimental cliffs match theoretical predictions: at $d/D\approx 5\%$, $\varepsilon^2\approx 5\%$ implies $\varepsilon\approx 22\%$, and actual accuracy is $\approx 72\%$, nearly equal to $1/2+\varepsilon$.
- The marginal gain curve of "increasing dimensions" is highly non-linear: there is basically no improvement once $d\geq D$, but it collapses to 50% almost instantly once $d<cD$. This contradicts the industrial intuition that "more dimensions are always better."
- Algorithm-level hardness results imply that even if polynomial time and arbitrarily high dimensions are allowed, it is impossible to consistently exceed a 1D random embedding. This pessimistic conclusion underscores the necessity of *input structure assumptions* (margin / separability).

## Highlights & Insights
- Provides a *sharp* constant-factor lower bound for the phenomenon where dimensionality truncation causes a cliff-like drop in performance—a fact long accepted by engineers but lacking theoretical explanation.
- Combines statistical learning theory (VC dimension + uniform convergence) with approximation algorithm theory (UGC + MAS approximation resistance) to provide both information-theoretic and computational complexity lower bounds within a single paper.
- The gap reduction design is minimalist—a single anchor translates any MAS instance into triplets with obvious equivalence. This "translation trick" is highly reusable for new geometric problems.

## Limitations & Future Work
- The lower bound conclusions are a mix of worst-case and average-case: it does not deny that "adding margin, separability, or manifold structures" could break the bounds. Whether real-world data falls near hard instances remains unknown.
- Experiments are restricted to synthetic data; how $d^*$ varies with distribution parameters in real text, image, or retrieval data warrants follow-up.
- No computable suggestion is provided for "how many dimensions one should use"—only a lower bound stating it "will collapse if less than $cD$," where the specific value of $c$ remains a theoretical constant.

## Related Work & Insights
- **vs JL lemma**: JL uses $O(\log n/\varepsilon^2)$ dimensions to preserve distances, but this paper proves that preserving rankings cannot achieve such compression, highlighting the fundamental difference in dimensionality requirements between ordinal and metric embeddings.
- **vs Bilu-Linial / Avdiukhin (realizable triplet embedding)**: While they proved $O(\min(n-1,\sqrt m))$ dimensions are always sufficient, this paper provides the inverse conclusion that "it fails completely below a certain constant factor $D$," pinning down both ends of the dimension-accuracy curve.
- **vs Matryoshka representation learning (Kusupati 2022)**: Matryoshka empirically trains nested, truncatable embeddings; this paper provides the theoretical background for why they only become useful starting from a certain dimensionality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide sharp information-theoretic and computational complexity lower bounds for the embedding dimension threshold.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments fully support the theory, but lack follow-up on real datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations; reduction and probabilistic proofs are elegantly separated.
- Value: ⭐⭐⭐⭐ Provides a fundamental theoretical guide for embedding dimension selection and opens a new avenue for exploring how input structures might break lower bounds.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] The Geometry of Projection Heads: Conditioning, Invariance and Collapse](the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)
- [\[ICML 2026\] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models](limix-2m_mitigating_low-rank_collapse_and_attention_bottlenecks_in_tabular_found.md)
- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)
- [\[CVPR 2026\] Suppressing Non-Semantic Noise in Masked Image Modeling Representations](../../CVPR2026/self_supervised/suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)

</div>

<!-- RELATED:END -->
