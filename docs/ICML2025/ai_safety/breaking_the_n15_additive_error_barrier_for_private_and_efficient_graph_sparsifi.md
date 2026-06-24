---
title: >-
  [Paper Note] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification
description: >-
  [ICML 2025][AI Safety][differential privacy] This paper breaks the $n^{1.5}$ additive error barrier for differentially private graph cut sparsification by proposing a polynomial-time $(\varepsilon,\delta)$-DP algorithm that reduces the additive error to $n^{1.25+o(1)}$. The core technology is the first privacy-preserving expander decomposition algorithm.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "differential privacy"
  - "graph sparsification"
  - "expander decomposition"
  - "cut approximation"
  - "synthetic graph"
date: 2026-05-08
content_hash: 9f46c663362f7338
---

# Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification

**Conference**: ICML 2025  
**arXiv**: [2507.01873](https://arxiv.org/abs/2507.01873)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: differential privacy, graph sparsification, expander decomposition, cut approximation, synthetic graph

## TL;DR
This paper breaks the $n^{1.5}$ additive error barrier for differentially private graph cut sparsification by proposing a polynomial-time $(\varepsilon,\delta)$-DP algorithm that reduces the additive error to $n^{1.25+o(1)}$. The core technology is the first privacy-preserving expander decomposition algorithm.

## Background & Motivation

**Background**: Cut sparsification is a fundamental problem in graph algorithms—given a graph, output a smaller sparse graph such that the values of all cuts are approximately preserved. Under privacy-preserving settings, the goal is to output a synthetic graph that satisfies differential privacy.

**Limitations of Prior Work**: Under differential privacy constraints, the best known result for efficient (polynomial-time) graph cut sparsification algorithms has an additive error of $\tilde{O}(n^{1.5})$ (Gupta, Roth, Ullman, TCC'12). In contrast, "non-efficient" (exponential-time) algorithms can achieve an additive error of $\tilde{O}(n)$. Whether the gap between $n^{1.5}$ and $n$ can be closed in polynomial time remains an open question.

**Key Challenge**: Privacy preservation requires adding noise to the graph, but excessive noise destroys the approximation accuracy of the cuts. Existing efficient algorithms directly add Laplace/Gaussian noise to the edges, resulting in a noise level of $O(n^{1.5}/\varepsilon)$. To break this barrier, a more fine-grained noise injection strategy is required.

**Goal**: Break the $n^{1.5}$ additive error barrier in polynomial time.

**Key Insight**: Utilize expander decomposition—a technique that decomposes a graph into several expander subgraphs and a small number of inter-group edges—and independently add private noise to each decomposed subgraph.

**Core Idea**: The cut structure within each expander is more regular (approximately uniform), thus requiring less private noise. By recursively applying expander decomposition, the total amount of noise can be significantly reduced.

## Method

### Overall Architecture
Input: Weighted graph $G$ with $n$ nodes, privacy parameters $(\varepsilon, \delta)$, approximation parameter $\gamma$.  
Output: A private synthetic graph with $(1+\gamma)$-multiplicative + $n^{1.25+o(1)}$-additive approximation.  

Pipeline:
1. Perform private expander decomposition on the graph.
2. Perform private sparsification on each expander subgraph.
3. Merge the results.

### Key Designs

1. **Private Expander Decomposition**:

    - Function: Under the premise of satisfying differential privacy, decompose the graph into several $\phi$-expanders and a small number of inter-group edges (edges accounting for $\leq \phi$ of the volume are cut).
    - Mechanism: Classical expander decomposition is based on Cheeger's inequality and spectral methods, but spectral computations involve global graph information, which directly violates differential privacy. This paper designs a private decomposition algorithm based on local random walks.
    - Design Motivation: This is the core contribution of the paper. Non-private expander decomposition is already a powerful tool in graph algorithms, and extending it to the private setting opens up broad application possibilities.

2. **Private Sparsification on Expanders**:

    - Function: Construct a private sparsification on each expander subgraph.
    - Mechanism: The cut structure of an expander is highly regular—the value of any cut is at least $\phi$ times the volume of the smaller side. This implies a smaller relative error (additive error / cut value) and allows using less noise.
    - Key Formula: On a $\phi$-expander, the additive error can be reduced to $\tilde{O}(n/\phi)$.
    - Design Motivation: Decompose the global problem into several "well-structured" subproblems, making each subproblem easier to solve.

3. **Recursive Decomposition Strategy**:

    - Function: Recursively apply expander decomposition to inter-group edges.
    - Mechanism: Each layer of decomposition reduces the additive error by a factor of $n^{O(1/L)}$, reaching $n^{1+1/4+o(1)}$ after $L$ layers of recursion.
    - Design Motivation: A single-layer decomposition is insufficient to break the $n^{1.5}$ barrier; recursion is key to further compressing the error.

### Loss & Training
No training is involved. The core metric is the cut approximation error: for all cuts $S \subset V$, $|w(S, \bar{S})_{\text{output}} - w(S, \bar{S})_{\text{input}}|$.

## Key Experimental Results

### Main Results

| Algorithm | Time Complexity | Multiplicative Error | Additive Error | Privacy Guarantee |
|---|---|---|---|---|
| GRU'12 | Poly(n) | $1+\gamma$ | $\tilde{O}(n^{1.5})$ | $(\varepsilon,\delta)$-DP |
| EKKL'20 | Exp(n) | $1+\gamma$ | $\tilde{O}(n)$ | $(\varepsilon,\delta)$-DP |
| **Ours** | **Poly(n)** | $1+\gamma$ | $\tilde{O}(n^{1.25})$ | $(\varepsilon,\delta)$-DP |

### Ablation Study (Impact of Recursion Layers)

| Recursion Layers $L$ | Additive Error | Computational Cost | Description |
|---|---|---|---|
| 1 | $\tilde{O}(n^{1.5})$ | Low | Degenerates to the classical method |
| 2 | $\tilde{O}(n^{1.33})$ | Medium | Begins to break the barrier |
| 4 | $\tilde{O}(n^{1.25})$ | Relatively High | Main result of this paper |
| $O(\log n)$ | Close to $\tilde{O}(n)$ | High | Theoretically optimal, but high practical computational cost |

### Key Findings
- For the first time, breaks the $n^{1.5}$ additive error barrier in polynomial time.
- Private expander decomposition is an independent technical contribution with potentially broader applications.
- Each additional layer of recursive decomposition reduces the exponent of the additive error by approximately $1/4$.
- A gap of $n^{0.25}$ still remains with respect to the $n^1$ lower bound in the non-private setting; whether this can be further reduced is an open question.

## Highlights & Insights
- **Significant Theoretical Breakthrough**: Breaks the $n^{1.5}$ barrier that stood for 13 years.
- **Technical Depth**: Private expander decomposition is a highly challenging technical innovation.
- **Methodological Inspiration**: The "decompose $\rightarrow$ handle locally $\rightarrow$ merge" strategy can be generalized to other private graph problems.

## Limitations & Future Work
- This work is purely theoretical, without practical experimental verification.
- A gap still exists between $n^{1.25}$ and the information-theoretic lower bound $n$.
- The constants in the polynomial overhead of the algorithm might be large, and the practical scalability remains to be evaluated.
- Only cut sparsification is considered; the private version of spectral sparsification is far more challenging.

## Related Work & Insights
- Gupta, Roth, Ullman (TCC'12): Prior best efficient private cut sparsification.
- Eliáš, Kapralov, Kulkarni, Lee (SODA'20): Prior optimal non-efficient result.
- Private expander decomposition may inspire research on problems like private shortest paths and private network flow.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Breaks the long-standing theoretical barrier.
- Experimental Thoroughness: ⭐⭐ Purely theoretical work, no experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear theory, but highly technical.
- Value: ⭐⭐⭐⭐⭐ Highly significant for the field of private graph algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Skirting Additive Error Barriers for Private Turnstile Streams](../../ICLR2026/ai_safety/skirting_additive_error_barriers_for_private_turnstile_streams.md)
- [\[ICML 2025\] An Efficient Private GPT Never Autoregressively Decodes](an_efficient_private_gpt_never_autoregressively_decodes.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Faster Rates for Private Adversarial Bandits](faster_rates_for_private_adversarial_bandits.md)
- [\[ICML 2025\] Improving the Variance of Differentially Private Randomized Experiments through Clustering](improving_the_variance_of_differentially_private_randomized_experiments_through_.md)

</div>

<!-- RELATED:END -->
