---
title: >-
  [Paper Note] Towards Attributions of Input Variables in a Coalition
description: >-
  [ICML 2025][Interpretability][Shapley value] This paper re-derives the computation mechanism of the Shapley value from the perspective of AND-OR interactions, proving that attribution conflicts under different variable partitions inherently stem from interaction effects that only cover a subset of coalition variables. Based on this, the authors define a coalition attribution metric and three fidelity metrics, with experiments validating their consistency with human intuition.
tags:
  - "ICML 2025"
  - "Interpretability"
  - "Shapley value"
  - "AND-OR interaction"
  - "coalition attribution"
  - "attribution conflict"
  - "explainable AI"
date: 2026-05-08
content_hash: 0c1ccd4e14959e29
---

# Towards Attributions of Input Variables in a Coalition

**Conference**: ICML 2025  
**arXiv**: [2309.13411](https://arxiv.org/abs/2309.13411)  
**Code**: None  
**Area**: Explainability / XAI  
**Keywords**: Shapley value, AND-OR interaction, coalition attribution, attribution conflict, explainable AI

## TL;DR

This paper re-derives the computation mechanism of the Shapley value from the perspective of AND-OR interactions, proving that attribution conflicts under different variable partitions inherently stem from interaction effects that only cover a subset of coalition variables. Based on this, the authors define a coalition attribution metric and three fidelity metrics, with experiments validating their consistency with human intuition.

## Background & Motivation

**Background**: The Shapley value is the most classic variable attribution method in explainable AI, widely used because it satisfies five fundamental axioms: anonymity, symmetry, dummy, additivity, and efficiency. Methods based on the Shapley value, such as SHAP and Integrated Gradients, have become the de facto standards for attribution analysis.

**Limitations of Prior Work**: However, the computation of the Shapley value heavily depends on the partition of input variables. For example, in image classification, using pixels versus local regions as input variables, or in NLP, using tokens versus words, can lead to completely different attribution results. Specifically, when treating a group of variables $S = \{x_1, x_2\}$ as a coalition, the coalition attribution $\varphi(S)$ is often not equal to $\phi(x_1) + \phi(x_2)$, which is known as **attribution conflict**.

**Key Challenge**: Existing methods (such as Faith-Shap) attempt to "eliminate" such conflicts through engineering means (e.g., minimizing squared loss) but lack a theoretical explanation of the underlying causes. Neither the Shapley value nor the Banzhaf value can guarantee attribution consistency under arbitrary variable groupings.

**Goal**: (1) To reveal the mathematical nature of attribution conflicts: what factors exactly cause the inconsistency of attribution values under different partitions? (2) To provide a theoretically grounded definition of attribution for coalitions. (3) To propose metrics for evaluating whether a group of variables can form a "faithful" coalition.

**Key Insight**: The authors find that AND-OR interactions can completely describe the non-linear relationships among all input variables in a DNN, and the Shapley value is essentially an even redistribution of these interaction effects. This perspective provides an exact mathematical tool for analyzing conflicts.

**Core Idea**: By reformulating the Shapley value as an allocation problem of AND-OR interaction effects, this work proves that the root cause of attribution conflicts lies in interactions that cover only a part (rather than all) of the variables in a coalition.

## Method

### Overall Architecture

Given an AI model $v(\cdot)$ and an instance $x$ with $n$ input variables, the objective of the method is to: (1) reformulate the Shapley value and the Banzhaf value using AND-OR interactions; (2) define coalition attribution $\varphi(S)$ based on these interactions; (3) decompose the attribution conflict into a shared component and a conflict component; and (4) propose three fidelity metrics to evaluate the quality of the coalition. The entire framework forms a closed loop from theoretical derivation to experimental validation.

### Key Designs

1. **AND-OR Interaction Re-deriving the Shapley Value**:

    - **Function**: Decomposes the classic Shapley value $\phi(i)$ into a weighted sum of AND-OR interaction effects.
    - **Mechanism**: An AND interaction $I_{\text{and}}(S)$ indicates that all variables in $S$ must appear simultaneously to produce the effect (e.g., the four words "raining cats and dogs" must co-occur to express "heavy rain"), while an OR interaction $I_{\text{or}}(S)$ indicates that the presence of any single variable triggers the effect. Theorem 3.2 proves that $\phi(i) = \sum_{S \ni i} \frac{1}{|S|} [I_{\text{and}}(S) + I_{\text{or}}(S)]$, meaning the Shapley value is the result of uniformly allocating each interaction effect containing variable $i$ to all participating variables.
    - **Design Motivation**: This reformulation directly links the "black-box" Shapley value computation with the internal interaction patterns of the model, establishing a mathematical foundation for subsequent conflict analysis.

2. **Definition of the Coalition Attribution Metric $\varphi(S)$**:

    - **Function**: Extends the Shapley value from individual variables to an arbitrary coalition of variables.
    - **Mechanism**: For a coalition $S \subseteq N$, define $\varphi(S) = \sum_{T \supseteq S} \frac{|S|}{|T|} [I_{\text{and}}(T) + I_{\text{or}}(T)]$. That is, the attribution of coalition $S$ only comes from interactions $T$ that completely contain $S$ ($T \supseteq S$), with an allocation weight of $|S|/|T|$.
    - **Design Motivation**: Unlike previous methods that forcefully eliminate conflicts using engineering loss, this work accepts the objective existence of conflicts and explains their sources through a clear definition of coalition attribution.

3. **Decomposition and Explanation of Attribution Conflicts**:

    - **Function**: Decomposes the sum of individual variable attributions $\sum_{i \in S} \phi(i)$ into a shared component and a conflict component.
    - **Mechanism**: Theorem 3.4 proves that $\sum_{i \in S} \phi(i) = \varphi(S) + \phi_{\text{conflict}}(S)$, where the conflict component is $\phi_{\text{conflict}}(S) = \sum_{T: \emptyset \neq T \cap S \neq S} \frac{|T \cap S|}{|T|} [I_{\text{and}}(T) + I_{\text{or}}(T)]$. That is, those interactions $T$ that only partially cover variables in $S$ ($T \cap S \neq \emptyset$ and $T \cap S \neq S$) are the direct causes of conflict. Corollary 3.5 further proves that if the model never encodes interactions that only cover part of $S$, the conflict is zero.
    - **Design Motivation**: This decomposition theoretically clarifies for the first time that "attribution conflict is an inevitable, objective existence" because DNNs can rarely guarantee encoding interactions solely as complete coalitions.

### Fidelity Metrics

Based on the theoretical decomposition above, three metrics are proposed to evaluate the fidelity of a coalition:

- **$R(i)$**: Evaluates how much of the attribution of a specific variable $i$ in coalition $S$ comes from coalition-related interactions, where $R(i) = |U_{i,S}| / (|U_{i,S}| + |U_{i,\bar{S}}|)$. A value closer to 1 indicates a more faithful coalition.
- **$R'(i)$**: A more fine-grained evaluation, using the absolute values of interaction effects to measure the significance of variable $i$'s participation in the coalition.
- **$Q(S)$**: Evaluates the fidelity of the entire coalition $S$, measuring the ratio of the interaction strength allocated to the complete coalition to the interaction strength of all interactions involving variables in $S$.

## Key Experimental Results

### Main Results: Theoretical Verification — Approximation Error

| Verification Content | Coalition Order $m$ | Approximation Error |
|----------|-------------|---------|
| Approximating Shapley value with coalition attribution | $m=1$ | $3.6 \times 10^{-8}$ |
| Approximating Shapley value with coalition attribution | $m=5$ | $7.6 \times 10^{-7}$ |
| Approximating Shapley value with coalition attribution | $m=10$ | $2.8 \times 10^{-7}$ |
| Approximating model output with coalition attribution | $m=1$ | $2.3 \times 10^{-7}$ |
| Approximating model output with coalition attribution | $m=5$ | $6.3 \times 10^{-7}$ |
| Approximating model output with coalition attribution | $m=10$ | $4.7 \times 10^{-7}$ |

All errors are on the order of $10^{-7}$, demonstrating the correctness of the theoretical derivation.

### Ablation Study: Validation of Coalition Fidelity Metrics on Toy Functions

| Coalition Type | $\mathbb{E}[R(i)]$ | $\mathbb{E}[R'(i)]$ | $\mathbb{E}[Q(S)]$ |
|---------|--------------------|--------------------|-------------------|
| Purely faithful coalition | 0.944 | 0.936 | 0.948 |
| Partially faithful coalition | 0.471 | 0.608 | 0.590 |
| Purely unfaithful coalition | 0.031 | 0.016 | 0.013 |

All three metrics can clearly distinguish between faithful/unfaithful coalitions: purely faithful coalitions approach 1, while purely unfaithful coalitions approach 0.

### NLP Experiment: Coalition Fidelity on SST-2 Sentiment Classification

| Sentence | Coalition | Model | $Q(S)$ | Description |
|------|------|------|--------|------|
| "the mesmerizing performances..." | {mesmerizing, performances} | BERT-large | 0.743 | Natural phrase, faithful coalition |
| "the mesmerizing performances..." | {mesmerizing, performances} | LLaMA | 0.746 | Consistent across models |
| "...easily rivaling blair witch..." | {rivaling, blair} | BERT-large | 0.425 | Fragmented phrase, unfaithful |
| "...easily rivaling blair witch..." | {rivaling, blair} | LLaMA | 0.312 | Consistently deemed unfaithful across models |

### Key Findings

- Attribution conflict is prevalent across all tested DNNs and is a natural consequence of how models encode interactions rather than a computational error.
- The three metrics perfectly distinguish the three types of coalitions on toy functions and highly align with human semantic intuition in real NLP models.
- In Go experiments, coalition patterns corresponding to high-intensity interactions encoded by KataGo align closely with classic joseki (shape patterns) labeled by professional players.

## Highlights & Insights

- **Unified Understanding of Attribution Methods from an Interaction Perspective**: Formulating both the Shapley value and the Banzhaf value as allocation problems of AND-OR interactions provides an elegant unified framework, offering a common mathematical language to understand various attribution methods.
- **The Philosophy of "Conflicts Cannot Be Eliminated, Only Explained"**: Unlike prior methods that attempt to algorithmically eliminate conflicts, this paper proves that conflicts are objective reflections of DNN interaction structures. This paradigm shift paves the way for future research.
- **Coalition Attribution Satisfies Five Axioms**: The newly defined coalition attribution retains the core theoretical advantages of the Shapley value and can serve as a standard attribution method for arbitrary variable groupings.
- **Application to Go Demonstrates Practical Value**: Applying the theory to the attribution analysis of shape patterns in KataGo helps players understand AI's move strategies, serving as an excellent demonstration from theory to application.

## Limitations & Future Work

- **Extremely High Computational Complexity**: Precise calculation of AND-OR interactions involves traversing $2^n$ subsets; hence, only $n=10$ stones could be analyzed in Go experiments. Efficient approximate computation remains the primary bottleneck for practical applications.
- **Validation Restricted to Small-Scale Scenarios**: NLP experiments selected 10 words, and Go experiments selected 10 stones, leaving the feasibility on large-scale inputs (such as thousands of pixels in full images) unverified.
- **Subjectivity of Human-Annotated Coalitions**: "Natural coalitions" in NLP and Go experiments rely on human annotations, and there is a lack of algorithms to automatically discover optimal coalition partitions.
- **Lack of Quantitative Comparison with Baselines**: Quantitative comparison under identical settings against baseline methods like Faith-Shap is missing, with only a qualitative comparison provided in Table 1.

## Related Work & Insights

- **vs Faith-Shap (Tsai et al., 2023)**: Faith-Shap uses a least-squares loss $\|v(S) - \sum_{i \in S} \phi(i)\|^2$ to force coalition attributions toward the sum of individual attributions, representing an engineering solution to eliminate conflict. This paper proves conflicts are inevitable and provides a theoretical explanation, offering a more fundamental approach.
- **vs Joint Shapley Value (Harris et al., 2021)**: Joint Shapley proposes an attribution metric for joint feature sets, but it essentially measures the attribution of feature sets/interactions (similar to Shapley Taylor) rather than explaining conflict mechanisms. This paper provides a clearer theoretical framework from the perspective of interaction decomposition.
- **vs Banzhaf Value**: The Banzhaf value satisfies 2-efficiency but not efficiency for general sets. The coalition attribution in this work is applicable to both Shapley and Banzhaf frameworks.

## Rating

- Novelty: ⭐⭐⭐⭐ Using AND-OR interactions to unify the explanation of attribution conflicts is highly novel and theoretically profound.
- Experimental Thoroughness: ⭐⭐⭐ Experiments cover synthetic/NLP/image/Go scenarios, but the scale is small and quantitative comparison with baselines is lacking.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivation is rigorous, with clean theorem-corollary chains, though the heavy notation presents a learning curve.
- Value: ⭐⭐⭐⭐ Provides an important theoretical base for attribution methods in XAI, though computational complexity limits practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Minimizing False-Positive Attributions in Explanations of Non-Linear Models](../../NeurIPS2025/interpretability/minimizing_false-positive_attributions_in_explanations_of_non-linear_models.md)
- [\[ICLR 2026\] The Tutor-Pupil Augmentation: Enhancing Learning and Interpretability via Input Corrections](../../ICLR2026/interpretability/the_tutor-pupil_augmentation_enhancing_learning_and_interpretability_via_input_c.md)
- [\[ACL 2026\] Jacobian Scopes: Token-Level Causal Attributions in LLMs](../../ACL2026/interpretability/jacobian_scopes_token-level_causal_attributions_in_llms.md)
- [\[AAAI 2026\] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees](../../AAAI2026/interpretability/shapbpt_image_feature_attributions_using_data-aware_binary_partition_trees.md)
- [\[ICML 2025\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)

</div>

<!-- RELATED:END -->
