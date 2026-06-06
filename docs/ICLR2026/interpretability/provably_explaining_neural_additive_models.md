---
title: >-
  [Paper Note] Provably Explaining Neural Additive Models
description: >-
  [ICLR 2026][Interpretability][Neural Additive Models] This paper proposes a dedicated efficient explanation algorithm for Neural Additive Models (NAMs) that generates provably cardinally-minimal explanations using only a…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Neural Additive Models"
  - "provable explanations"
  - "cardinally-minimal explanations"
  - "formal verification"
  - "explainable AI"
date: 2026-05-08
content_hash: a0e4af6b00c5e245
---

# Provably Explaining Neural Additive Models

**Conference**: ICLR 2026
**arXiv**: [2602.17530](https://arxiv.org/abs/2602.17530)  
**Code**: None  
**Area**: Interpretability / Formal Verification
**Keywords**: Neural Additive Models, provable explanations, cardinally-minimal explanations, formal verification, explainable AI

## TL;DR

This paper proposes a dedicated efficient explanation algorithm for Neural Additive Models (NAMs) that generates provably cardinally-minimal explanations using only a logarithmic number of verification queries, outperforming existing general-purpose subset-minimal explanation algorithms in both speed and explanation quality.

## Background & Motivation

The interpretability of neural networks is a central concern for AI safety and trustworthy deployment. Existing post-hoc explanation methods face the following challenges:

**Lack of provable guarantees**: Most explanation methods (e.g., SHAP, LIME, Grad-CAM) are inherently heuristic and cannot guarantee the correctness of their explanations. For instance, feature importance rankings produced by SHAP may not faithfully reflect the model's actual decision basis.

**Computational bottleneck of provable explanations**: The key approach to obtaining explanations with provable guarantees is to find a "cardinally-minimal subset"—the smallest number of input features that alone sufficiently determine the model's prediction. For standard neural networks, this requires:
   - An exponential number of verification queries in the number of input features
   - Each query being NP-hard in itself
   - Making the approach computationally infeasible in practice

**Opportunity with NAMs**: Neural Additive Models constitute a more interpretable family of neural networks. The core structure of a NAM is $f(\mathbf{x}) = h_1(x_1) + h_2(x_2) + \cdots + h_n(x_n)$, where each $h_i$ is an independent univariate neural network. This additive structure should facilitate explanation—yet existing work has not fully exploited this structural property.

**Subset-minimal vs. cardinally-minimal**: Most existing algorithms can only find subset-minimal explanations (i.e., no feature can be further removed), without guaranteeing cardinally-minimal explanations (i.e., the subset with the fewest features). Cardinally-minimal explanations are more informative but harder to compute.

The core problem addressed in this paper: **Can the additive structure of NAMs be exploited to efficiently generate provably cardinally-minimal explanations?**

## Method

### Overall Architecture

The proposed algorithm consists of two phases:
1. **Preprocessing phase**: Each univariate NAM component $h_i$ is analyzed to compute its output range and critical intervals.
2. **Explanation generation phase**: The preprocessing results are leveraged to find cardinally-minimal explanations using a logarithmic number of verification queries.

### Key Designs

1. **Exploiting NAM Structure**:

    - Function: Leverages the additive decomposability of NAMs to reduce the global verification problem to independent univariate analyses.
    - Mechanism: Since $f(\mathbf{x}) = \sum_i h_i(x_i)$, the contribution $h_i(x_i)$ of each feature $x_i$ to the output is independent. Determining whether a feature subset is "sufficient" can therefore be accomplished by analyzing the output ranges of the individual $h_i$ functions.
    - Specifically, for a fixed feature subset $S$, the range of variation in $f$'s output depends solely on the sum of the possible output ranges of $h_i$ for features outside $S$.
    - **Design Motivation**: Additive decomposability is a structural advantage of NAMs over general neural networks and should be fully exploited.

2. **Parallelizable Preprocessing**:

    - Function: Performs interval analysis on each small univariate NAM component $h_i$.
    - Mechanism: Computes the output range $[\underline{h}_i, \overline{h}_i]$ of each $h_i$ over its domain, along with finer interval partitions. Formal verification techniques (e.g., interval propagation, linear relaxation) are used to obtain tight upper and lower bounds. Preprocessing runtime is logarithmic in the required precision. The preprocessing of each $h_i$ is fully independent and can be parallelized.
    - **Design Motivation**: The one-time preprocessing cost is traded for efficiency in subsequent explanation generation; verifying univariate networks is far easier than verifying multivariate ones.

3. **Logarithmic Verification Query Algorithm**:

    - Function: After preprocessing, generates cardinally-minimal explanations using $O(\log n)$ verification queries, where $n$ is the number of features.
    - Mechanism: Uses the precomputed "influence" information of each feature and a binary search strategy to identify the minimal sufficient subset. The algorithm proceeds as follows:
      a. Compute the "uncertainty contribution" of each feature $x_i$—i.e., the output variation range of $h_i$ when $x_i$ is not fixed.
      b. Sort features by their uncertainty contribution.
      c. Apply a greedy-plus-binary-search strategy to determine the minimal subset: iteratively remove features with the smallest contributions and verify whether the remaining subset remains sufficient.
      d. Each verification step is completed efficiently via interval arithmetic.
    - **Design Motivation**: Sorting combined with binary search reduces an exponential search problem to a logarithmic one.

4. **Formalization of Provable Guarantees**:

    - Function: Ensures that generated explanations are mathematically correct—both cardinally-minimal and sufficient.
    - Mechanism: "Sufficiency" is defined as follows: for a given input $\mathbf{x}$, fixing the features in the explanation set $S$ ensures that the model's predicted class remains unchanged regardless of the values taken by the remaining features. Under the additive NAM structure, sufficiency can be verified by checking whether $\sum_{i \notin S} (\overline{h}_i - \underline{h}_i)$ is smaller than the decision boundary margin.
    - Provable guarantees mean that the returned explanation set is correct in the worst case—no adversarial example can invalidate the explanation.
    - **Design Motivation**: Unlike the probabilistic guarantees of sampling-based methods, formal guarantees are necessary for safety-critical applications.

### Loss & Training

- This paper proposes an explanation method rather than a training method—no loss functions or training strategies are involved.
- The algorithm operates on a pre-trained NAM model and constitutes post-hoc processing at inference time.
- Preprocessing complexity: $O(n \cdot \text{poly}(\log(1/\epsilon)))$, where $\epsilon$ is the precision parameter.
- Explanation generation complexity: $O(\log n)$ verification queries.

## Key Experimental Results

### Main Results

Comparison with existing subset-minimal explanation algorithms:

| Method | Explanation Type | Verification Queries | Explanation Size | Computation Time |
|--------|-----------------|---------------------|-----------------|-----------------|
| Existing general-purpose algorithms | Subset-minimal | Exponential | Larger | Slower |
| Proposed NAM-specific algorithm | Cardinally-minimal | Logarithmic | **Smallest** | **Fastest** |

Key observation: The proposed algorithm solves a harder problem (cardinally-minimal vs. subset-minimal) while achieving superior performance in both speed and explanation quality.

### Ablation Study

| Configuration | Key Metric | Remarks |
|--------------|-----------|---------|
| Direct search without preprocessing | Significantly more queries | Preprocessing contributes substantially |
| Varying precision $\epsilon$ | Higher precision yields slower preprocessing but better explanation quality | Precision–efficiency trade-off exists |
| Varying feature count $n$ | Query count grows logarithmically | Confirms theoretical logarithmic complexity |
| Different NAM architectures | Consistent performance | Generality of the algorithm |

### Key Findings

1. **Cardinally-minimal ≠ subset-minimal**: Cardinally-minimal explanations can be significantly smaller than subset-minimal ones, providing more refined information.
2. **Formal vs. sampling-based explanations**: Sampling-based methods (e.g., permutation sampling in SHAP) yield substantially different—and incorrect—conclusions in certain cases.
3. **NAM interpretability advantages extend beyond visualization**: Prior work on NAMs primarily relied on plotting the shape functions $h_i$; this paper demonstrates that NAMs also support efficient, formally provable explanations.
4. **Practical implications**: In safety-critical domains (healthcare, finance), unreliable explanations can be more dangerous than no explanation at all.

## Highlights & Insights

1. **Qualitative change in computational complexity**: The reduction from exponential to logarithmic complexity represents not an incremental optimization but a fundamental breakthrough—enabled by deep exploitation of NAM structure.
2. **"Solving a harder problem can be faster"**: Cardinally-minimal explanations are harder to find than subset-minimal ones, yet leveraging problem structure makes the former more efficient—an embodiment of the "structure implies efficiency" principle in algorithm design.
3. **Productive intersection of theory and application**: Tools from the formal verification community (interval propagation, SMT solving, etc.) are productively integrated with machine learning explanation methods.
4. **New understanding of NAM value**: NAMs are not only visually interpretable (via shape function plots) but also computationally interpretable in a formal sense.
5. **Relevance to safety-critical deployment**: Provable explanations have important implications for AI regulatory compliance (e.g., the explainability requirements of the EU AI Act).

## Limitations & Future Work

1. **Restricted to NAMs**: The algorithm relies critically on the additive structure and cannot be directly extended to general neural networks or models with feature interactions (e.g., Neural Additive Models with Interactions, NAM-I).
2. **Expressiveness limitations of NAMs**: NAMs cannot model feature interactions, which limits model performance on certain tasks. Whether adopting a NAM is worthwhile depends on the trade-off between interpretability requirements and performance needs.
3. **Selection of preprocessing precision**: The choice of $\epsilon$ affects both explanation quality and computational cost, yet the paper does not provide a method for automatically determining $\epsilon$.
4. **High-dimensional settings**: When feature dimensionality is extremely large (e.g., image pixels), even logarithmic queries may be insufficient—though NAMs themselves are not suited for such high-dimensional inputs.
5. **Extension to GA2M**: Extending the algorithm to GA2M models that incorporate pairwise feature interactions is a natural direction, but interaction terms would substantially increase complexity.

## Related Work & Insights

- **Explainable AI**: SHAP, LIME → reliability issues of post-hoc explanations → demand for provable explanations.
- **Formal verification**: Neural network verification tools such as NNV, Marabou, and α-β-CROWN provide the technical foundation for this work.
- **NAMs**: NAMs proposed by Agarwal et al. (2021), along with subsequent interpretable model families including NODE-GAM and EBM.
- **Insights**: The idea of reducing explanation complexity by exploiting model structure is generalizable—for example, can dedicated provable explanation methods be designed for attention mechanisms?

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First algorithm achieving logarithmic complexity for cardinally-minimal explanations of NAMs; significant theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparison with multiple baselines and demonstration of sampling method limitations, though dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical sections are rigorous, though notation density is high.
- **Value**: ⭐⭐⭐⭐ — Important contributions to explainable AI and safety-critical applications, though scope is constrained by NAM applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Provably Learning Attention with Queries](../../ICML2026/interpretability/provably_learning_attention_with_queries.md)
- [\[NeurIPS 2025\] FaCT: Faithful Concept Traces for Explaining Neural Network Decisions](../../NeurIPS2025/interpretability/fact_faithful_concept_traces_for_explaining_neural_network_decisions.md)
- [\[NeurIPS 2025\] Additive Models Explained: A Computational Complexity Approach](../../NeurIPS2025/interpretability/additive_models_explained_a_computational_complexity_approach.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](../../ICML2026/interpretability/query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
