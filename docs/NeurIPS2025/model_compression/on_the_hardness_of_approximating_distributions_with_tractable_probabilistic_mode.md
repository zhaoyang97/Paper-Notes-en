---
title: >-
  [Paper Note] On the Hardness of Approximating Distributions with Tractable Probabilistic Models
description: >-
  [NeurIPS 2025][Model Compression][probabilistic circuits] This paper proves that approximating arbitrary distributions with tractable probabilistic models (e.g.…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "probabilistic circuits"
  - "tractable probabilistic models"
  - "approximate modeling"
  - "NP-hardness"
  - "expressive efficiency"
date: 2026-05-08
content_hash: d362067bd531d1e4
---

# On the Hardness of Approximating Distributions with Tractable Probabilistic Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.01281](https://arxiv.org/abs/2506.01281)  
**Code**: None  
**Area**: Model Compression / Probabilistic Models
**Keywords**: probabilistic circuits, tractable probabilistic models, approximate modeling, NP-hardness, expressive efficiency

## TL;DR
This paper proves that approximating arbitrary distributions with tractable probabilistic models (e.g., decomposable probabilistic circuits) under bounded $f$-divergence is NP-hard, and establishes an exponential size separation between decomposable PCs and (deterministic + decomposable) PCs under approximate modeling, demonstrating that approximation relaxations do not alleviate the complexity bottlenecks inherent in exact modeling.

## Background & Motivation
Probabilistic models face a fundamental trade-off between expressiveness and inference efficiency. Tractable probabilistic models (TPMs), particularly probabilistic circuits (PCs), impose structural constraints to guarantee efficient computation of certain queries (e.g., marginal probabilities, MAP inference). However, these structural constraints may limit the compactness of model representations — a large body of work has studied the "exact" expressive efficiency of different circuit families.

**Limitations of Prior Work**:
- Nearly all work on PC expressive efficiency focuses on exact representation, whereas the goal in practical learning is to approximate, not exactly represent, data distributions.
- Exact representation often requires exponential size blowup (e.g., from decomposable PCs to deterministic + decomposable PCs). A natural question arises: can allowing a small approximation error avoid such exponential blowup?

**Key Challenge**: Approximate modeling is practically relevant in two ways: (1) as a constraint on the hypothesis space of learning algorithms — if a PC family cannot approximate a target distribution within bounded error, any learning algorithm will fail; and (2) for approximate compilation of inference — compiling a Bayesian network approximately into a PC and then performing exact inference. The central question is whether approximate modeling is fundamentally "easier" than exact modeling.

**Key Insight**: The paper adopts a computational complexity perspective, proving NP-hardness via SAT reductions and establishing exponential separations via the Sauerhoff function construction.

## Method

### Overall Architecture
Rather than proposing an algorithm, this paper establishes a series of theoretical results: (1) proving that approximating arbitrary distributions within bounded $f$-divergence using models that support tractable marginals is NP-hard; (2) proving an exponential size separation between decomposable PCs and deterministic + decomposable PCs under approximate modeling; and (3) analyzing the relationship between approximate modeling and approximate inference.

### Key Designs

1. **NP-hardness Proof (Theorems 3.4/3.5)**:

    - **Reduction**: From SAT. Given a Boolean formula $\hat{P}$, construct a new formula $\hat{P}' = (Y \wedge \hat{P}) \vee (\neg Y \wedge X_1 \wedge \cdots \wedge X_n)$.
    - Key insight: If $\hat{P}$ is satisfiable, the probability of $Y=1$ under the normalized distribution $P$ is $\geq 1/2$; otherwise it is $0$. Any approximation $Q$ with TV distance $< 1/4$ must preserve this distinction.
    - Therefore, deciding whether $Q(Y=1) \geq 1/4$ is equivalent to deciding the satisfiability of $\hat{P}$.
    - Corollary: For any $k$-convex $f$-divergence $D_f$, approximate modeling with bounded $k\epsilon^2$-$D_f$ is NP-hard.
    - Significance: This applies not only to decomposable PCs but to all model classes supporting tractable marginals, including sum-product circuits, probabilistic sentential decision diagrams, determinantal point processes, and others.

2. **Exponential Size Separation (Theorem 4.1)**:

    - **Construction**: Uses the Sauerhoff function $S_n$ — a Boolean function defined over $n \times n$ matrices, equal to 1 when the XOR modulo 3 of row sums or column sums satisfies a specific condition.
    - $S_n$ admits an exact DNNF (decomposable negation normal form) representation of size $O(n^2)$, and the corresponding decomposable PC $P_n$ is also of size $O(n^2)$.
    - **Proof chain**: (a) A deterministic + decomposable PC approximator with bounded TV distance induces a weak approximation of the corresponding Boolean function as a d-DNNF (via probabilistic threshold pruning); (b) Sauerhoff proved that weak approximating d-DNNFs for $S_n$ require size $2^{\Omega(n)}$.
    - **Conclusion**: For $\epsilon$-TV approximations of $P_n$ (with $\epsilon \leq 1/16 - \Omega(1/\mathrm{Poly}(n^2))$), any deterministic + decomposable PC requires size $2^{\Omega(n)}$.

3. **Relationship Between Approximate Modeling and Approximate Inference (Section 5)**:

    - **Positive results**: Bounded TV distance implies absolute approximation of marginals (Theorems 5.1/5.2); bounded TV distance implies absolute approximation of MAP queries (Theorems 5.3/5.4).
    - **Negative results**: Bounded TV distance does not imply absolute approximation of conditional probabilities or conditional MAP (Theorem 5.5) — demonstrated via counterexamples where TV distance $< \epsilon$ but conditional query error can be arbitrarily large.
    - It also does not imply relative approximation of marginals (Proposition 3.2).
    - **Practical implication**: Even if approximate compilation succeeds (which is itself NP-hard), it guarantees only absolute approximation of marginals and MAP, not of conditional queries.

### Loss & Training
This is a purely theoretical work with no training procedure. The core measure is the family of $f$-divergences (including KL divergence, TV distance, $\chi^2$ divergence, etc.), unified through a framework of $k$-convexity.

## Key Experimental Results

### Main Results

| Result | Condition | Scope |
|--------|-----------|-------|
| Approximate modeling is NP-hard | $\epsilon < 1/4$, any $k$-convex $f$-divergence | All models supporting tractable marginals |
| Exponential separation | $\epsilon \leq 1/16 - \Omega(1/\mathrm{Poly}(n^2))$ | Decomposable PC vs. deterministic + decomposable PC |
| Bounded TV $\Rightarrow$ absolute marginal approximation | $D_{\mathrm{TV}}(P \| Q) < \epsilon$ | Arbitrary distributions |
| Bounded TV $\Rightarrow$ absolute MAP approximation | $D_{\mathrm{TV}}(P \| Q) < \epsilon$ | Arbitrary distributions |
| Bounded TV $\not\Rightarrow$ conditional probability approximation | Any $\epsilon > 0$ | Counterexample distributions exist |

### Key Findings

| Divergence | NP-hard Threshold | Notes |
|------------|-------------------|-------|
| TV distance | $\epsilon < 1/4$ | Directly from Theorem 3.5 |
| KL divergence | $D_{\mathrm{KL}} < 1/8$ | Derived via Pinsker's inequality |
| $\chi^2$ divergence | Corresponding threshold | Derived via $k$-convexity |

### Key Findings
- Approximate modeling is not fundamentally "easier" than exact modeling — NP-hardness persists in the approximate setting.
- The exponential size separation between decomposable PCs and deterministic + decomposable PCs holds under both exact and approximate modeling.
- Approximate compilation schemes (first approximately compiling to a PC, then performing exact inference) face fundamental computational barriers.

## Highlights & Insights
- **First proof of NP-hardness for approximate modeling of probability distributions**: Prior work addressed only exact modeling and approximate hardness for logic circuits; this paper is the first to extend such results to probability distributions.
- **Unified framework**: The $k$-convex $f$-divergence family provides a unified hardness result across multiple commonly used divergence measures.
- **Seamless extension from exact to approximate**: The Sauerhoff function elegantly generalizes from exact separations for logic circuits to approximate separations for probabilistic circuits.
- **Precise characterization of inference guarantees**: The paper clearly delineates which inference queries can and cannot be guaranteed to be approximated under approximate modeling.

## Limitations & Future Work
- Only Boolean variables are considered; continuous or mixed-variable settings are not addressed.
- The TV distance threshold ($1/16$) for the exponential separation is small; results for larger thresholds remain unknown.
- No constructive approximation algorithms or practically feasible solutions are provided.
- Direct implications for models widely used in deep learning (e.g., diffusion models, Transformers) are limited.
- The negative results (conditional queries cannot be guaranteed) lack positive counterparts: under what additional conditions can such guarantees be obtained?

## Related Work & Insights
- **vs. Martens & Medabalimi**: They proved that approximating certain functions requires PC sequences that converge but grow exponentially in size; this paper allows nonzero error and applies to a broader class of models.
- **vs. De Colnet & Mengel**: They studied the hardness of approximate compilation for logic circuits; this paper extends the results to probabilistic circuits and distribution approximation.
- **vs. Chubarian & Turán**: They studied interpretability and OBDD approximation for Bayesian classifiers; this paper focuses on more general probabilistic modeling.
- **Implication for practitioners**: For those working on PC/TPM learning and knowledge compilation — even when approximation is permitted, one cannot arbitrarily expand the inference capabilities of a model family while maintaining compactness.

## Rating
- Novelty: ⭐⭐⭐⭐ — First extension of approximation hardness to probability distributions and tractable models, though the core techniques (SAT reductions, Sauerhoff function) build on existing work.
- Experimental Thoroughness: ⭐⭐⭐ — Purely theoretical; no empirical validation, but the theoretical results are internally consistent and complete.
- Writing Quality: ⭐⭐⭐⭐⭐ — Definitions are precise, proofs are rigorous, conclusions are clearly stated, and the paper is well-organized.
- Value: ⭐⭐⭐⭐ — Provides important theoretical foundations for the probabilistic circuits and tractable probabilistic models community, though its direct relevance to mainstream deep learning practice is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Exploiting Weight-Space Symmetries for Approximating Curvature](../../ICML2026/model_compression/exploiting_weight-space_symmetries_for_approximating_curvature.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](reordering_patches_improves_vision_models.md)
- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](geometry_of_decision_making_in_language_models.md)
- [\[NeurIPS 2025\] Elastic ViTs from Pretrained Models without Retraining](elastic_vits_from_pretrained_models_without_retraining.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)

</div>

<!-- RELATED:END -->
