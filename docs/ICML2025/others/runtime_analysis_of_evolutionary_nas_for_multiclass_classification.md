---
title: >-
  [Paper Note] Runtime Analysis of Evolutionary NAS for Multiclass Classification
description: >-
  [ICML2025][Evolutionary Neural Architecture Search] This work presents the first theoretical runtime analysis of evolutionary neural architecture search (ENAS) on multiclass classification. It proves that the (1+1)-ENAS algorithms with both one-bit and bit-wise mutations scale with an expected runtime of $O(rM\ln rM)$ to find the optimal architecture, demonstrating that simple one-bit mutation can perform on par with complex bit-wise mutation.
tags:
  - "ICML2025"
  - "Evolutionary Neural Architecture Search"
  - "Runtime Analysis"
  - "Multiclass Classification"
  - "(1+1)-EA"
  - "Fitness Function"
date: 2026-05-08
content_hash: 9ac27f48dd60f958
---

# Runtime Analysis of Evolutionary NAS for Multiclass Classification

**Conference**: ICML2025  
**arXiv**: [2506.06019](https://arxiv.org/abs/2506.06019)  
**Code**: None  
**Area**: NAS Theory / Theoretical Evolutionary Computation  
**Keywords**: Evolutionary Neural Architecture Search, Runtime Analysis, Multiclass Classification, (1+1)-EA, Fitness Function

## TL;DR

This work presents the first theoretical runtime analysis of evolutionary neural architecture search (ENAS) on multiclass classification. It proves that the (1+1)-ENAS algorithms with both one-bit and bit-wise mutations scale with an expected runtime of $O(rM\ln rM)$ to find the optimal architecture, demonstrating that simple one-bit mutation can perform on par with complex bit-wise mutation.

## Background & Motivation

Evolutionary Neural Architecture Search (ENAS) utilizes evolutionary algorithms to automatically design deep neural network architectures, displaying outstanding performance in practical applications. However, the **theoretical research** of ENAS lags significantly behind its practice:

- **Runtime analysis** is a central topic in the theory of evolutionary algorithms, yet runtime analysis in ENAS is almost non-existent.
- The core challenge is that the fitness of ENAS must be obtained by training the neural network, making it hard to formulate mathematically.
- Previously, only Fischer et al. (2023/2024) formulated fitness functions for classification tasks with a fixed architecture, while Lv et al. (2024b) established the first fitness function for ENAS, which was nevertheless restricted to **binary classification**.
- Practical applications (e.g., image recognition, speech recognition, medical diagnoses) are predominantly **multiclass classification** tasks, creating an urgent need for theoretical analysis in multiclass scenarios.
- Existing designs of search spaces are unable to support the analysis of multiclass classification and bit-wise mutations.

This paper extends the runtime analysis of ENAS from binary classification to multiclass classification for the first time, filling this theoretical gap.

## Method

### 1. Neural Architecture Modeling

The $M$-class classification task is decomposed into $M-1$ binary classifiers (cells):

- Each cell contains $l$ blocks and an OR neuron.
- Three types of blocks: **A-type** (segment decision region), **B-type** (sector decision region), and **C-type** (triangle decision region).
- The outputs of the $M-1$ cells are aggregated by $M$ neurons in a hidden layer, followed by Softmax to output probabilities for each class.

The architecture is encoded as $M-1$ triplets:

$$\boldsymbol{x} = \{(n_A^1, n_B^1, n_C^1), \ldots, (n_A^{M-1}, n_B^{M-1}, n_C^{M-1})\}$$

### 2. Multiclass Classification Benchmark Problem Mcc

An $M$-class classification problem is defined on the unit circle $S$:

- The unit circle is divided equally into $n = 2rM$ sectors ($r \geq 2$).
- Each class contains $r$ segment regions, $r$ sector regions, and $r$ triangle regions.
- These three categories of regions correspond to representative structures of halfspaces, unbounded polyhedra, and bounded polyhedra, respectively.

### 3. Fitness Function

A closed-form fitness function is mathematically derived:

$$\mathcal{F}(\boldsymbol{x}) = \frac{Ar_{tri} \cdot (\mathbb{I}_x + 2r) + Ar_{seg} \cdot (\mathbb{J}_x + 2r - \epsilon_x)}{\pi}$$

where:

| Notation | Meaning |
|------|------|
| $\mathbb{I}_x$ | Number of correctly classified triangles for classes 1 to $M-1$ |
| $\mathbb{J}_x$ | Number of correctly classified segments (including segments within sectors) for classes 1 to $M-1$ |
| $\epsilon_x$ | Number of segments in class $M$ misclassified by the $(M-1)$-th cell |
| $Ar_{tri}$, $Ar_{seg}$ | Area of a single triangle and segment |

Fitness is dominated by the $\mathbb{I}_x$ term (since $Ar_{tri} > (n-2r) \cdot Ar_{seg}$), resembling the classic OneMax function in evolutionary computation.

### 4. (1+1)-ENAS Algorithm

A **two-level mutation** strategy is adopted:

- **Outer Mutation**: Choice of which cells to mutate
    - **One-bit**: Select one cell at random.
    - **Bit-wise**: Each cell is independently selected with probability $1/(M-1)$.
- **Inner Mutation**: Action taken on the selected cells
    - **Local**: Mutate each selected cell once.
    - **Global**: Mutate $K$ times, where $K \sim \text{Pois}(1)$.
- Mutation operation (Definition 2.1): Select a block type at random and perform Addition / Deletion / Modification.

## Theoretical Results

### Runtime Upper and Lower Bounds

| Algorithm | Upper Bound | Lower Bound |
|------|------|------|
| (1+1)-ENAS$_{\text{onebit}}$ | $O(rM\ln(rM))$ | $\Omega(rM\ln M)$ |
| (1+1)-ENAS$_{\text{bitwise}}$ | $O(rM\ln(rM))$ | $\Omega(rM\ln M)$ |

**Core Finding**: The upper and lower runtime bounds for both mutation strategies are **nearly identical**, differing only by a factor of $\ln r$.

### Proof Framework

The optimization process is divided into two phases:

| Phase | Goal | Distance Function | Lower Bound of Drift |
|------|------|----------|----------|
| Phase 1 | $\mathbb{I}_x = N = 2r(M-1)$ | $V_1(x) = N - \mathbb{I}_x$ | $\geq (V_1/N) \cdot 2/9$ |
| Phase 2 | $\mathbb{J}_x - \epsilon_x = N$ | $V_2(x) = N - (\mathbb{J}_x - \epsilon_x)$ | $\geq (V_2/N) \cdot 1/9$ |

The **Multiplicative Drift Theorem** is applied to both phases to yield $O(N\ln N)$ each, totaling $O(rM\ln(rM))$.

### Search Space Partitioning

Based on the values of $\mathbb{I}_x$ and $\mathbb{J}_x - \epsilon_x$, the search space $\mathcal{S}$ is partitioned hierarchically:

- First-level partition: $\mathcal{S}_0 <_{\mathcal{F}} \mathcal{S}_1 <_{\mathcal{F}} \cdots <_{\mathcal{F}} \mathcal{S}_{n-2r}$
- Second-level partition: $\mathcal{S}_i^0 <_{\mathcal{F}} \mathcal{S}_i^1 <_{\mathcal{F}} \cdots <_{\mathcal{F}} \mathcal{S}_i^{n-2r}$

This hierarchical partition guarantees the monotonicity of the fitness function, allowing drift analysis to be performed.

## Highlights & Insights

1. **First Runtime Analysis of ENAS for Multiclass Classification**: Extending from binary to $M$-class classification is of pioneering significance.
2. **Simple Mutation Suffices**: It is theoretically proven that one-bit mutation and bit-wise mutation achieve equivalent runtime. This challenges the intuition that "complex mutations are superior," directly guiding practical ENAS algorithm design.
3. **Mathematically Formulated ENAS Fitness Function**: Bypassing the black-box training process, the classification accuracy is characterized directly via geometric properties, serving as a benchmarking tool for future theoretical analysis of ENAS.
4. **Two-Level Search Space Design**: The encoding scheme of the cell level + block level aligns with prevailing constructs in the ENAS community, enhancing the practicality of the theoretical results.
5. **Extended Application of Drift Analysis**: The multiplicative drift analysis is successfully extended from standard EAs to the two-level search space in ENAS.

## Limitations & Future Work

1. **Overly Idealized Problem**: The data regions in the Mcc benchmark are uniformly distributed on the unit circle, which deviates significantly from actual multiclass data distributions.
2. **Analysis Limited to the (1+1) Setting**: Practical ENAS often employs population-based strategies and crossover operators. The (1+1) framework fails to fully capture these complex algorithmic behaviors.
3. **Assuming Optimal Parameters are Reachable**: The analysis assumes that parameters for each architecture can always reach optimality, bypassing the complexity of the weight training process.
4. **Two-Dimensional Input Limitation**: The input is restricted to two dimensions, which does not characterize architecture search in high-dimensional feature spaces.
5. **Limited Practical Significance of the Optimal Architecture**: The optimal architecture for Mcc is defined by a set of block-counting constraints, which differs substantially from architecture quality in real-world NAS.
6. **Missing Empirical Comparison against Practical ENAS**: The paper only includes theoretical verification experiments, without evaluations on standard NAS benchmarks.

## Related Work & Insights

- **Classic EA Runtime Analysis**: (1+1)-EA analysis on OneMax and LeadingOnes (Droste et al., 2002; Witt, 2013).
- **NAS Fitness Function for Classification**: Hyperplane/curved-hyperplane systems proposed by Fischer et al. (2023/2024).
- **Binary ENAS Analysis**: Polytope-based decision boundary methods by Lv et al. (2024b), which this work directly extends.
- **Insights**: Future studies can attempt to extend this framework to larger populations, crossover operators, or dynamic search space settings.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first runtime analysis of ENAS for multiclass classification, presenting clear theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Includes verification experiments to support the theories, but lacks comparison in real NAS scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Clear proof framework with consistent notation.
- Value: ⭐⭐⭐ — Highly significant in theory but limited in practical impact, with applicability constrained by idealized assumptions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Theoretical and Empirical Analysis of Lehmer Codes to Search Permutation Spaces with Evolutionary Algorithms](../../AAAI2026/others/theoretical_and_empirical_analysis_of_lehmer_codes_to_search_permutation_spaces_.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](../../NeurIPS2025/others/evolutionary_prediction_games.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)
- [\[ICML 2025\] Continuous-Time Analysis of Heavy Ball Momentum in Min-Max Games](continuous-time_analysis_of_heavy_ball_momentum_in_min-max_games.md)
- [\[AAAI 2026\] Improved Runtime Guarantees for the SPEA2 Multi-Objective Optimizer](../../AAAI2026/others/improved_runtime_guarantees_for_the_spea2_multi-objective_optimizer.md)

</div>

<!-- RELATED:END -->
