---
title: >-
  [Paper Note] Opinion Maximization in Social Networks by Modifying Internal Opinions
description: >-
  [NeurIPS 2025][LLM/NLP][opinion dynamics] This paper studies the optimization problem of maximizing the overall opinion in a social network by modifying the internal opinions of $k$ key nodes. Two sampling-based approximation algorithms (random walk and forest sampling) and one exact asynchronous algorithm MIS are proposed. MIS provides theoretical convergence guarantees to the optimal solution and demonstrates superior efficiency and accuracy on real-world networks with tens of millions of nodes.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - opinion dynamics
  - social networks
  - opinion maximization
  - random walk
  - asynchronous algorithm
  - structural centrality
date: 2026-05-08
content_hash: d61532a605e7a4b5
---

# Opinion Maximization in Social Networks by Modifying Internal Opinions

**Conference**: NeurIPS 2025
**arXiv**: [2510.17226](https://arxiv.org/abs/2510.17226)
**Code**: To be confirmed
**Area**: LLM/NLP
**Keywords**: opinion dynamics, social networks, opinion maximization, random walk, asynchronous algorithm, structural centrality

## TL;DR

This paper studies the optimization problem of maximizing the overall opinion in a social network by modifying the internal opinions of $k$ key nodes. Two sampling-based approximation algorithms (random walk and forest sampling) and one exact asynchronous algorithm MIS are proposed. MIS provides theoretical convergence guarantees to the optimal solution and demonstrates superior efficiency and accuracy on real-world networks with tens of millions of nodes.

## Background & Motivation

Opinion governance in social networks is critical for public health campaigns, political elections, and commercial marketing.

**Opinion Dynamics Model**: The Friedkin-Johnson (FJ) model is a classical framework for opinion evolution. Each agent $i$ has an internal opinion $s_i \in [0,1]$ (innate position) and a resistance coefficient $\alpha_i \in (0,1]$ (degree of resistance to neighbors' opinions). Opinions propagate iteratively through the network until reaching equilibrium: $\bm{z} = (\bm{I} - (\bm{I} - \bm{R})\bm{P})^{-1} \bm{R} \bm{s}$

**Overall Opinion Optimization**: The aggregate expressed opinion $f(\bm{R}, \bm{s}) = \bm{1}^\top \bm{M} \bm{s} = \sum_v \rho_v s_v$ is the sum of all agents' equilibrium opinions, where $\rho_v$ is the **structural centrality** of node $v$—measuring the influence of its internal opinion on the global opinion.

**Bottleneck of Traditional Methods**: Exact computation of structural centrality requires matrix inversion with complexity $O(n^3)$, which is infeasible for networks at the million-node scale.

**Limitations of Prior Work**: Previous opinion optimization has been achieved primarily by modifying resistance coefficients, adjusting expressed opinions, or altering network structure. The problem of **maximizing overall opinion by modifying internal opinions** has not yet been efficiently addressed in algorithm design.

**Core Problem (OpinionMax)**: Given a social network and an integer $k$, how should one select $k$ nodes and set their internal opinions to 1 in order to maximize the overall equilibrium opinion?

## Method

### Overall Architecture

Three progressively refined algorithms:

1. **RWB** (Random Walk-Based sampling): Approximates structural centrality via absorbing random walks.
2. **Forest** (Forest sampling): Estimates structural centrality based on a combinatorial interpretation using spanning converging forests.
3. **MIS** (asynchronous exact search): A two-phase deterministic algorithm that performs global coarse filtering followed by local refinement to exactly identify the optimal set of $k$ nodes.

### Key Designs

**Structural Centrality and Potential Influence**:

The overall opinion can be decomposed as $f(\bm{R}, \bm{s}) = \sum_v \rho_v s_v$, where structural centrality $\rho_v = \sum_u \bm{M}(u,v)$ measures the contribution of node $v$ to the equilibrium opinions of all other nodes. The marginal gain from modifying the internal opinion of node $v$ from $s_v$ to 1 is $\Delta(v) = \rho_v (1 - s_v)$, termed **potential influence**. Thus OpinionMax is equivalent to identifying the $k$ nodes with the largest potential influence.

**Algorithm 1: RWB (Random Walk-Based Sampling)**

- Core Idea: Structural centrality equals the sum of absorption probabilities at that node over absorbing random walks initiated from all nodes (Lemma 1).
- Implementation: Simulate $N$ absorbing random walks with uniformly random starting points, and estimate $\hat{\rho}_v$ by multiplying each node's absorption frequency by $n$.
- Theoretical Guarantee: $N = O(\frac{n}{\epsilon^2} \log n)$ walks suffice to ensure $|\hat{\rho}_i - \rho_i| \geq \epsilon$ with probability at most $1/n$.
- Time Complexity: $O(\frac{\alpha_{\max}(1-\alpha_{\min})}{\epsilon^2 \alpha_{\min}^2} \cdot n \log n)$
- Near-Optimality Guarantee (Corollary 1): $\sum_{i \in \hat{T}} \rho_i(1-s_i) \geq \sum_{i \in T^*} \rho_i(1-s_i) - 2k\epsilon$

**Algorithm 2: Forest (Forest Sampling)**

- Core Idea: Via the Neumann series and the combinatorial interpretation of spanning converging forests (Lemma 3), a connection is established between structural centrality and the root distribution of sampled forests.
- Implementation: Based on an extension of Wilson's algorithm, using loop-erased random walks to sample spanning converging forests.
- Time Complexity: $O(\frac{1}{\alpha_{\min}} l n)$, where $l$ is the number of samples.

**Algorithm 3: MIS (Exact Asynchronous Search)**—Core Contribution

**Phase 1: GlobalInfApprox (Global Influence Approximation)**

Maintain a residual vector $\bm{r}_a$ (initialized to $\bm{1}$) and an estimate vector $\hat{\bm{\Delta}}$ (initialized to $\bm{0}$). For each node $v$ whose residual exceeds threshold $\epsilon$, perform three asynchronous operations:
- Distribute $(1-\alpha_v)/d_v^+ \cdot r_a(v)$ to each out-neighbor.
- Accumulate $(1-s_v)\alpha_v \cdot r_a(v)$ into the node's own potential influence estimate.
- Reset $r_a(v) = 0$.

Key Guarantee (Lemma 5): $(1-\epsilon)\Delta(v) \leq \hat{\Delta}(v) \leq \Delta(v)$

**Phase 2: TargetedNodeRefine (Candidate Node Refinement)**

Using the exact decomposition from Lemma 4, $\Delta(v) - \hat{\Delta}(v) = (1-s(v)) \cdot \bm{e}_v^\top \bm{M}^\top \bm{r}_a$, backward propagation refinement is performed individually for each node in candidate set $C$, progressively tightening the error bounds.

**Merge Criterion** (Lemma 8): Based on estimated values and error bounds, nodes are classified into three categories:
- Nodes $T$ definitively belonging to the optimal set.
- Candidate nodes $C$ potentially belonging to the optimal set.
- Nodes definitively excluded.

By progressively reducing $\epsilon'$ (from 1 to $1/2n$ ...), the candidate set $C$ shrinks until $|T| = k$.

### Loss & Training

Time complexity upper bound of MIS (Theorem 3):

$$O\left(\frac{d_{\max}^+ n}{\alpha_{\min}} \log \frac{1}{\epsilon} + \frac{m}{k_{\text{gap}} \cdot \alpha_{\min}}\right)$$

where $k_{\text{gap}}$ is the gap between the $k$-th and $(k+1)$-th largest values in the true potential influence vector.

## Key Experimental Results

### Main Results: Runtime Comparison ($k=64$, unit: seconds)

| Dataset (# nodes) | RWB (Unif.) | Forest (Unif.) | MIS (Unif.) |
|-------------------|-------------|----------------|-------------|
| DBLP (317K) | 120.61 | 66.02 | **0.28** |
| Google (876K) | 344.02 | 162.87 | **0.39** |
| YoutubeSnap (1.1M) | 896.66 | 196.53 | **1.03** |
| Pokec (1.6M) | 1037.87 | 467.81 | **3.72** |
| Flixster (2.5M) | 2342.12 | 326.52 | **2.59** |
| LiveJournal (4.8M) | 3275.20 | 1454.37 | **8.61** |
| Twitter (41.7M) | Timeout | 12115.80 | **279.73** |
| SinaWeibo (58.7M) | Timeout | 10538.27 | **156.83** |

MIS is 2–3 orders of magnitude faster than RWB and 1–2 orders of magnitude faster than Forest. On Twitter (41.7M nodes), MIS requires approximately 280 seconds.

### Main Results: Accuracy Comparison

| Method | Opinion Value | Precision | NDCG |
|--------|--------------|-----------|------|
| topRandom | Low | Low | Low |
| topDegree | Medium | Medium | Medium |
| topPageRank | Medium | Medium | Medium |
| RWB | High | Fluctuating | High |
| Forest | High | Fluctuating | High |
| **MIS** | **Best** | **1.000** | **≈1.000** |

MIS achieves perfect precision (= 1) and near-perfect NDCG across all datasets and all metrics.

### Ablation Study

| Ablation Item | Impact |
|--------------|--------|
| Resistance coefficient distribution (uniform/normal/exponential) | Exponential distribution (low $\alpha_{\min}$) significantly increases runtime for RWB and MIS |
| Varying $k$ (1→1024) | Performance gaps narrow at larger $k$, but MIS remains consistently optimal |
| RWB sample count | A large number of samples ($O(n \log n / \epsilon^2)$) is required to guarantee accuracy |
| Forest sample count $l=4000$ | Performs well on most networks but is sensitive to extreme distributions |

### Key Findings

1. **MIS dominates in both efficiency and accuracy**: Across all 8 real-world networks, MIS runs 40–60× faster than the fastest baseline (Forest) while guaranteeing an exact optimal solution.
2. **Sampling methods face a precision-efficiency dilemma**: RWB and Forest require trade-offs between sample size and accuracy; MIS entirely avoids this issue through deterministic iteration.
3. **Resistance coefficient distribution has significant impact**: Exponential distributions (small $\alpha_{\min}$) increase random walk lengths, causing RWB runtime to rise sharply (up to 10×), while MIS's relative advantage becomes even more pronounced.
4. **Asynchronous updates combined with progressive refinement are critical**: The two-phase strategy of global coarse filtering to quickly identify candidates, followed by local refinement to progressively eliminate boundary nodes, substantially reduces unnecessary computation.
5. **Excellent scalability**: MIS requires approximately 157 seconds on SinaWeibo (58.7M nodes), demonstrating feasibility on truly large-scale real-world networks.

## Highlights & Insights

- **Elegance of the random walk interpretation**: Establishing the equivalence between structural centrality and absorption probabilities of absorbing random walks provides an intuitive probabilistic foundation for algorithm design.
- **Engineering value of asynchronous updates**: The asynchronous update paradigm—where each node converges independently and supports local termination—is better suited to large-scale networks than globally synchronous methods.
- **Generality of the two-phase strategy**: The coarse-to-fine candidate set reduction approach is not limited to opinion optimization but can be generalized to other top-$k$ selection problems.
- **Complete theoretical guarantees**: Every algorithmic component is supported by rigorous mathematical proofs, from the approximation bound (Lemma 5) to the exact convergence condition (Lemma 8).
- **Equivalence of OpinionMax and OpinionMin**: The paper establishes that maximizing and minimizing the overall opinion are equivalent problems, allowing a single algorithm to address both directions in practice.

## Limitations & Future Work

1. **Sensitivity to low resistance coefficients**: When $\alpha_{\min}$ is small, MIS runtime increases significantly due to the $1/\alpha_{\min}$ factor in Theorem 3.
2. **Termination issue with tied boundary nodes**: When the $k$-th and $(k+1)$-th true potential influence values are exactly equal ($k_{\text{gap}} = 0$), the algorithm may fail to terminate.
3. **Static network assumption**: Only fixed network structures are considered; dynamically evolving social networks are not addressed.
4. **Single opinion dimension**: Only one-dimensional opinions over $[0,1]$ are handled; multi-dimensional or multi-topic opinions are not explored.
5. **Uniform modification strategy**: Selected nodes' opinions are set directly to 1 (or 0), without considering intermediate modifications or optimal allocation under budget constraints.
6. **Absence of learning components**: The approach is purely algorithmic and does not compare against end-to-end learning methods such as graph neural networks.

## Related Work & Insights

- **Gionis et al. (2013)**: First proposed the opinion maximization problem via setting expressed opinions to 1; this paper provides a more practical approach by modifying internal opinions.
- **Abebe et al. (2018, 2021)**: Studied opinion optimization through modifying resistance coefficients, complementary to this paper's strategy of modifying internal opinions.
- **Zhu et al. (2021)**: Studied polarization minimization via link recommendation, focusing on network structure modification rather than node attribute modification.
- **Extension of Wilson's algorithm**: The Forest algorithm, based on loop-erased random walks and spanning converging forests, demonstrates the applicability of combinatorial tools to network algorithms.
- **Inspiration**: The algorithmic paradigm of asynchronous local updates with global consistency guarantees offers valuable insights for other optimization problems on large graphs, such as influence maximization and community detection.

## Rating

- Novelty: ⭐⭐⭐⭐ The asynchronous exact algorithm MIS is the primary contribution; the random walk and forest interpretations provide novel theoretical perspectives.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight real large-scale networks, three resistance coefficient distributions, five baselines, and three metrics yield an extremely comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and algorithm pseudocode is clear, though equation density is high.
- Value: ⭐⭐⭐⭐ Provides a practical exact algorithm for opinion optimization in social networks; the scalability to 58.7M nodes is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML](msf-cnn_patch-based_multi-stage_fusion_with_convolutional_neural_networks_for_ti.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[NeurIPS 2025\] Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity](polar_sparsity_high_throughput_batched_llm_inferencing_with_scalable_contextual_.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](detecting_high-stakes_interactions_with_activation_probes.md)
- [\[NeurIPS 2025\] Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs](triplets_better_than_pairs_towards_stable_and_effective_self-play_fine-tuning_fo.md)

</div>

<!-- RELATED:END -->
