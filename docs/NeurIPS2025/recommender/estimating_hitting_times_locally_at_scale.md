---
title: >-
  [Paper Note] Estimating Hitting Times Locally At Scale
description: >-
  [NeurIPS 2025][Recommender Systems][hitting time] Two local (sublinear) algorithms are proposed for estimating hitting times on graphs — Algorithm 1 based on meeting times and Algorithm 3 based on spectral truncation. Both require only short random walks centered at $u$ and $v$ without full graph access, achieving relative error <1.4% on synthetic and real-world graphs. An optimal sample complexity lower bound for walk-based estimation is also established.
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "hitting time"
  - "local algorithms"
  - "random walks"
  - "sublinear complexity"
  - "spectral methods"
date: 2026-05-08
content_hash: 224dd11aaa25c4df
---

# Estimating Hitting Times Locally At Scale

**Conference**: NeurIPS 2025
**arXiv**: [2511.04343](https://arxiv.org/abs/2511.04343)  
**Code**: [https://github.com/tkhar/hitting-time-at-scale](https://github.com/tkhar/hitting-time-at-scale)  
**Area**: Graph Algorithms / Random Walks
**Keywords**: hitting time, local algorithms, random walks, sublinear complexity, spectral methods

## TL;DR
Two local (sublinear) algorithms are proposed for estimating hitting times on graphs — Algorithm 1 based on meeting times and Algorithm 3 based on spectral truncation. Both require only short random walks centered at $u$ and $v$ without full graph access, achieving relative error <1.4% on synthetic and real-world graphs. An optimal sample complexity lower bound for walk-based estimation is also established.

## Background & Motivation

**Background**: The hitting time $H_G(u,v)$ denotes the expected number of steps for a random walk to travel from $u$ to $v$, and is widely used in graph clustering, link prediction, and recommender systems. Exact computation requires $O(n^3)$ time (via linear system solving or matrix inversion), which is infeasible for large-scale graphs.

**Limitations of Prior Work**: Naïve Monte Carlo methods (simulating random walks until arrival) suffer from enormous variance and lack theoretical guarantees; global spectral methods require eigenvalue decomposition at $O(n^3)$ cost; no existing method can efficiently estimate hitting times from a local subgraph.

**Key Challenge**: Hitting time is inherently a global quantity (depending on the global graph structure), yet large-scale settings permit only local computation. The central question is whether provably accurate estimates can be obtained using only short walks centered at $u$ and $v$.

**Goal**: Design sublinear-time local algorithms for estimating hitting times, with theoretical error guarantees and optimality proofs.

**Key Insight**: The key insight is that hitting times can be truncated and estimated via the meeting time of two independent random walks launched from $u$ and $v$ respectively.

**Core Idea**: Launch $\ell$ random walks each from $u$ and $v$, use meeting times to truncate the hitting time estimate, and avoid full graph traversal. The runtime depends only on the mixing time and stationary distribution.

## Method

### Overall Architecture
Two algorithms are presented. **Algorithm 1** (primary): launches $\ell$ random walks from each of $u$ and $v$, advances them step by step, and upon meeting, annihilates the pair while accumulating the hitting time estimate. **Algorithm 3** (spectral truncation): expands the hitting time as a matrix power series, truncates to a number of terms determined by the spectral gap, and estimates each term using short random walks.

### Key Designs

1. **Meeting-Time Truncation (Algorithm 1)**:

    - Function: Uses meeting events between two groups of walks to truncate the infinite expectation.
    - Mechanism: Lemma 3.4 establishes $H_G(u,v) = E[\sum_{t<T}(\frac{1}{2}[Y_t=v] - \frac{1}{2}[X_t=v])]$, where $T$ is the meeting time of walks started from $u$ and $v$. Initialize $\ell$ pairs of walks, advance them jointly, and upon meeting annihilate the pair while accumulating $\tilde{H}_{uv} \leftarrow \tilde{H}_{uv} + (y_v - x_v)/(\ell \cdot \pi(v))$.
    - Design Motivation: Concentration is guaranteed via a Markov chain Chernoff bound on the Kronecker product graph $G \times G$; concentration of meeting times (Lemma 3.1) controls truncation error.
    - Complexity: $O(t_{mix}^3 / (\|\pi\|_2^6 \cdot \pi^2(v) \cdot \varepsilon^2) \cdot \log^3(n/(\pi(u)\pi(v))))$

2. **Spectral Truncation Algorithm (Algorithm 3)**:

    - Function: Represents hitting time as a truncated power series via spectral decomposition.
    - Mechanism: $H_G(u,v) = \sum_{i=0}^{\infty} (\mathbb{1} - \frac{1}{\pi(v)}\mathbb{1}_v)^T W^i \chi_{uv}$; a truncation point $\ell$ is chosen such that the tail sum is $\leq \varepsilon/2$ (determined by spectral gap $\lambda$), and each term is estimated using $r$ walks of length $i$.
    - Design Motivation: Theoretically cleaner but less practical than Algorithm 1.

3. **Optimality Proof**:

    - Function: Establishes a sample complexity lower bound for walk-based estimation.
    - Mechanism: Theorems 5.1–5.2 prove that any local walk-based algorithm requires $\Omega(t_{mix}^2 / (\pi^2(v) \varepsilon^2))$ walks, and Algorithm 1 nearly matches this bound.
    - Design Motivation: Confirms that the algorithm is (near-)optimal.

### Loss & Training
- No training — purely algorithmic.
- Parameters: $t_{max} = 100 t_{mix} \ln(n/(\pi(u)\pi(v)))/\|\pi\|_2^2$, $\ell = 2t_{max}^2 \ln(n) / (\pi^2(v) \varepsilon^2)$.

## Key Experimental Results

### Main Results

| Graph | Algorithm 1 Error | Sampling Method Error |
|----|-----------------|------------|
| Barabási-Albert | <0.014 | — |
| Erdős-Rényi | <0.014 | — |
| Football (real) | 0.012±0.009 | 0.025±0.019 |
| Facebook (real) | — | — |
| Twitter (real) | 0.002±0.001 | 0.022±0.016 |

### Algorithm Comparison

| Algorithm | Error | Variance | Applicable Scenario |
|------|------|------|---------|
| Algorithm 1 (meeting) | Best | Lowest | General |
| Algorithm 3 (spectral truncation) | Higher | Higher | Theoretical analysis |
| Naïve sampling | Highest | Highest | Baseline only |

### Key Findings
- Algorithm 1 consistently outperforms Algorithm 3 and naïve sampling across all graph types.
- Error on large-scale graphs such as Twitter is extremely low (0.002), suggesting that the mixing time characteristics of large graphs are favorable for local algorithms.
- Different node-pair sampling strategies (uniform / degree-proportional / PageRank-proportional) have negligible impact on Algorithm 1 (<2%).
- Knowledge of the global mixing time is not required — Appendix F proposes local estimation via binary search.

## Highlights & Insights
- **Local algorithms for global quantities**: Although hitting time inherently depends on global graph structure, meeting-time truncation enables high-quality estimates from short local walks.
- **Theoretical completeness**: A near-optimal sample complexity lower bound is established, proving that the algorithm cannot be substantially improved.
- **Practical effectiveness**: Error below 1.5% on real-world social networks makes the method directly applicable as a relevance measure in recommender systems.

## Limitations & Future Work
- Efficiency degrades for graphs with large mixing times (a fundamental limitation of any local method).
- Requires a connected and aperiodic Markov chain.
- Smaller accuracy parameter $\varepsilon$ requires more samples.

## Related Work & Insights
- **vs. Effective Resistance**: Effective resistance admits local estimation and is symmetric, whereas hitting time is asymmetric ($H(u,v) \neq H(v,u)$), making it strictly harder to estimate locally.
- **vs. Personalized PageRank**: PPR is also estimated via local walks, but hitting time requires truncation rather than convergence.

## Rating
- Novelty: ⭐⭐⭐⭐ First local hitting-time estimation algorithm with theoretical guarantees.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and real-world graphs with optimality lower bounds.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and clearly presented theoretical derivations.
- Value: ⭐⭐⭐⭐ Establishes a new theoretical framework for local computation on graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ASAP: An Agentic Solution to Auto-Optimize Performance of Large-Scale LLM Training](asap_an_agentic_solution_to_auto-optimize_performance_of_large-scale_llm_trainin.md)
- [\[ICML 2025\] How to Set AdamW's Weight Decay as You Scale Model and Dataset Size](../../ICML2025/recommender/how_to_set_adamws_weight_decay_as_you_scale_model_and_dataset_size.md)
- [\[ICLR 2026\] Adaptive Regularization for Large-Scale Sparse Feature Embedding Models](../../ICLR2026/recommender/adaptive_regularization_for_large-scale_sparse_feature_embedding_models.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[NeurIPS 2025\] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](overcoming_sparsity_artifacts_in_crosscoders_to_interpret_chat-tuning.md)

</div>

<!-- RELATED:END -->
