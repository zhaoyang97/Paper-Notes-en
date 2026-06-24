---
title: >-
  [Paper Note] From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm
description: >-
  [AAAI 2026][SHAP] This paper proposes Woodelf, an algorithm that converts decision tree ensemble models into pseudo-Boolean functions in Weighted Disjunctive Normal Form (WDNF), enabling linear-time computation of both Background SHAP and Path-Dependent SHAP within a unified framework, achieving 16–31× CPU speedup and 24–333× GPU speedup on large-scale datasets.
tags:
  - "AAAI 2026"
  - "SHAP"
  - "Shapley Values"
  - "Decision Tree Ensembles"
  - "Boolean Logic"
  - "GPU Acceleration"
date: 2026-05-08
content_hash: f957233dbb6db46c
---

# From Decision Trees to Boolean Logic: A Fast and Unified SHAP Algorithm

**Conference**: AAAI 2026
**arXiv**: [2511.09376](https://arxiv.org/abs/2511.09376)  
**Code**: [GitHub](https://github.com/ron-wettenstein/woodelf)  
**Area**: Explainable AI / Feature Attribution / Game Theory
**Keywords**: SHAP, Shapley Values, Decision Tree Ensembles, Boolean Logic, GPU Acceleration

## TL;DR

This paper proposes Woodelf, an algorithm that converts decision tree ensemble models into pseudo-Boolean functions in Weighted Disjunctive Normal Form (WDNF), enabling linear-time computation of both Background SHAP and Path-Dependent SHAP within a unified framework, achieving 16–31× CPU speedup and 24–333× GPU speedup on large-scale datasets.

## Background & Motivation

SHAP (SHapley Additive exPlanations) is one of the most widely adopted methods for explaining decision tree ensemble models (e.g., XGBoost, Random Forest, CatBoost), assigning feature contributions via Shapley values across domains such as finance, advertising, and healthcare. Two mainstream SHAP computation paradigms have historically required fundamentally different algorithmic treatments:

- **Path-Dependent SHAP**: Estimates missing feature distributions using tree structure and training-time cover attributes — efficient but less accurate.
- **Background SHAP**: Uses a background dataset to estimate feature distributions — most accurate but computationally expensive, potentially requiring years on large datasets.

Existing state-of-the-art methods each have limitations: FastTreeShap v2 accelerates PD-SHAP; PLTreeShap achieves $O(m+n)$ linear complexity for BG-SHAP; GPUTreeSHAP exploits GPU parallelism. However, all rely on custom C++/CUDA code that is difficult to integrate and extend, and none support alternative game-theoretic metrics such as Banzhaf values.

The core insight of this paper is that **decision tree structure can be naturally encoded as WDNF formulas in Boolean logic, and Shapley values over WDNF can be computed in linear time**. Building on this, Woodelf establishes a unified framework implemented in pure Python (NumPy/SciPy/CuPy), supporting multiple SHAP variants and interaction values.

## Method

### Overall Architecture

Woodelf's pipeline consists of four steps: (1) compute the frequency vector $f$ (from a background dataset or path cover attributes); (2) convert each leaf of each tree into a weighted cube in WDNF via decision patterns, constructing contribution matrix $M$; (3) combine $M$ and $f$ into a precomputed vector $s$ via matrix multiplication; (4) index $s$ using the consumer's decision pattern to obtain the final SHAP values. The overall complexity is reduced from $O(nm)$ to $O(n+m)$.

### Key Designs

1. **Linear-Time Shapley Value Formula over WDNF**:

    - Function: Provides a closed-form computation of Shapley values for pseudo-Boolean functions in WDNF.
    - Mechanism: For a WDNF formula $F(x_1,\dots,x_h)=\sum_{k=1}^m w_k \cdot c_k$, the Shapley value of variable $i$ is $$\phi_i(F) = \sum_{k=1}^m w_k \times \begin{cases} \frac{1}{|S_k^+| \binom{|S_k|}{|S_k^+|}} & i \in S_k^+ \\ \frac{-1}{|S_k^-| \binom{|S_k|}{|S_k^-|}} & i \in S_k^- \end{cases}$$
    - Design Motivation: Exploits the linearity and null player (NPO) properties of Shapley values to reduce exponential enumeration to a constant-time contribution computation per cube.
    - The Banzhaf value formula $\beta_i(F) = \sum_k \frac{w_k}{2^{|S_k|-1}}$, as well as Shapley/Banzhaf interaction values, are derived analogously.

2. **Decision Pattern**:

    - Function: Encodes the behavior of a consumer or baseline instance on a decision tree as a binary sequence.
    - Mechanism: For the root-to-leaf path $(n_1, \dots, n_D)$ of leaf $l$, the decision pattern $p[i]=1$ if and only if the consumer follows the path direction at node $n_i$.
    - Design Motivation: Decision patterns fully characterize consumer behavior on a tree and can be computed efficiently via vectorized BFS (CalcDecisionPatterns algorithm, complexity $O(nL)$).

3. **Mapping Decision Patterns to WDNF Cubes**:

    - Function: Maps consumer–baseline decision pattern pairs to weighted cubes in WDNF.
    - Mechanism: Four rules — consumer follows/baseline does not → add positive literal; consumer does not follow/baseline does → add negative literal; both follow → cube unchanged; neither follows → cube is unsatisfiable.
    - The MapPatternsToCube algorithm precomputes mappings for all possible pattern pairs, avoiding redundant computation.

4. **$O(n+m)$ Background SHAP Derivation**:

    - Function: Reduces the original $O(nm)$ Background SHAP computation to $O(n+m)$.
    - Mechanism: The key observation is that a baseline's contribution depends only on its decision pattern. Patterns are aggregated via value_counts ($O(mL)$), then a matrix multiplication of the precomputed Shapley matrix $M_{l,i}$ and frequency vector $f_l$ yields the precomputed vector $s_{l,i}$; final results are obtained by table lookup per consumer.
    - Design Motivation: Matrix multiplication is natively GPU-friendly and requires no custom CUDA code when using NumPy/CuPy.

5. **GPU-Friendly Pure Python Implementation**:

    - Function: All core steps are implemented using standardized vectorized operations.
    - Mechanism: Sparse matrices reduce $O(4^D)$ to $O(3^D)$; leaves sharing the same feature pattern are cached; adjacent leaves whose decision patterns differ only in the last bit are exploited for acceleration; minimal integer types (uint8/16/32) are selected based on tree depth to optimize SIMD utilization.
    - Design Motivation: Eliminates C++/CUDA maintenance overhead while enabling seamless CPU-to-GPU switching via CuPy.

### Loss & Training

This paper does not involve model training. Woodelf is an inference-time explanation computation tool that operates directly on pre-trained decision tree ensemble models.

## Key Experimental Results

### Main Results

Performance comparison on two large-scale industrial datasets (XGBoost, 100 trees, depth 6):

**IEEE-CIS Fraud Detection Dataset** ($|B|=118K, |C|=472K, F=397$)

| Task | shap CPU | SOTA GPU | Woodelf CPU | Woodelf GPU | Speedup (vs. SOTA, any platform) |
|------|----------|----------|-------------|-------------|----------------------------------|
| PD-SHAP | 151s | 16s | 6s | 3.3s | — |
| BG-SHAP | ~10 days | ~14 hours | 12s | 10s | 24× (GPU) |
| PD-SHAP IV | ~33 hours | 105s | 11s | 8s | 13× (GPU) |
| BG-SHAP IV | ✗ | ✗ | 19s | 12s | First computable |

**KDD Cup 1999 Dataset** ($|B|=4.9M, |C|=3.0M, F=127$)

| Task | shap CPU | SOTA GPU | Woodelf CPU | Woodelf GPU | Speedup (vs. SOTA, any platform) |
|------|----------|----------|-------------|-------------|----------------------------------|
| PD-SHAP | 51min | 7.9s | 96s | 3.3s | — |
| BG-SHAP | ~8 years | ~3 months | 162s | 16s | 165× (GPU) |
| PD-SHAP IV | ~8 days | 229s | 193s | 6s | 38× (GPU) |
| BG-SHAP IV | ✗ | ✗ | 262s | 19s | First computable |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Sparse matrix optimization | $O(4^D) \to O(3^D)$ | Exploits sparsity produced by MapPatternsToCube to reduce core step complexity |
| Caching mechanism | Significant reduction in matrix computations | Leaves with identical depth and feature pattern reuse computed results |
| Integer type selection (uint8/16/32) | Improved computation speed | Selects minimal integer type based on tree depth to optimize SIMD utilization |
| NumPy vectorized indexing (line 29) | Avoids Python loops | Performs result lookup for all consumers via vectorized operations |

### Key Findings

- Background SHAP computation time on the KDD dataset dropped from "estimated 8 years" to "16 seconds" over five years, demonstrating the remarkable pace of algorithmic progress.
- Woodelf is the first practical implementation capable of computing Background SHAP interaction values (no prior implementation existed).
- A pure Python implementation (without C++/CUDA) still outperforms competing methods built on custom native code.
- Woodelf's derivation provably relies only on the linearity property of Shapley values, making it naturally applicable to any metric satisfying linearity.

## Highlights & Insights

- The core contribution is not merely a faster algorithm, but the revelation of deep connections among "decision trees → Boolean logic → game theory," providing an elegant theoretical framework.
- The linear-time Shapley value formula over WDNF (Formula 2) is remarkably concise: it requires only a single pass over each cube, computing contributions directly from the sizes of positive and negative literal sets.
- The unified framework simultaneously covers both Background and Path-Dependent SHAP variants, as well as Shapley/Banzhaf values and interaction values — previously believed to require separate algorithms.
- The key step in the $O(n+m)$ derivation — moving baseline aggregation outside the inner loop (value_counts) followed by matrix multiplication for precomputation — represents a broadly applicable algorithm design principle.

## Limitations & Future Work

- When trees are very deep (large $D$) or datasets are small, the $O(3^D)$ precomputation step may become a bottleneck, in which case PLTreeShap or the shap package may be preferable.
- The method currently requires univariate feature splits in decision tree ensembles (standard XGBoost format); extensions are needed for oblique splits or other variants.
- Experiments are conducted only on regression tasks; results for classification tasks are not presented.
- The choice of background dataset significantly affects SHAP values, but the paper does not discuss how to select an optimal background set.
- For interaction values, the matrix dimension is $O(4^D \times 4^D)$, which may become a memory bottleneck for deep trees.

## Related Work & Insights

- Lundberg et al. (2020) first proposed a polynomial-time SHAP algorithm, laying the foundation; Woodelf can be viewed as its ultimate optimization.
- PLTreeShap was the first to achieve $O(n+m)$ Background SHAP but relies on custom C++ code; Woodelf achieves superior performance in pure Python.
- The framework mapping pseudo-Boolean functions to game-theoretic characteristic functions may inspire efficient Shapley value computation for other model types (e.g., neural network substructures).
- The use of WDNF/WCNF representations suggests potential deeper connections between satisfiability solving and model interpretability.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices](dfdt_dynamic_fast_decision_tree_for_iot_data_stream_mining_on_edge_devices.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[AAAI 2026\] Model Change for Description Logic Concepts](model_change_for_description_logic_concepts.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] How Hard is it to Explain Preferences Using Few Boolean Attributes?](how_hard_is_it_to_explain_preferences_using_few_boolean_attributes.md)

</div>

<!-- RELATED:END -->
