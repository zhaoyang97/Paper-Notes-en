---
title: >-
  [Paper Note] Compact Example-Based Explanations for Language Models
description: >-
  [ACL 2026][LLM Pretraining][Training data influence] This paper proposes the Selection Relevance Score, a retraining-free metric for evaluating the quality of training sample subsets as example-based explanations. It demonstrates that the commonly used "select top-k by influence" strategy frequently underperforms random selection, and introduces a new strategy that balances influence and representativeness.
tags:
  - ACL 2026
  - LLM Pretraining
  - Training data influence
  - example-based explanations
  - selection relevance
  - gradient reconstruction
  - redundancy elimination
date: 2026-05-08
content_hash: bcb126d69dff3b95
---

# Compact Example-Based Explanations for Language Models

**Conference**: ACL 2026
**arXiv**: [2601.03786](https://arxiv.org/abs/2601.03786)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: Training data influence, example-based explanations, selection relevance, gradient reconstruction, redundancy elimination

## TL;DR

This paper proposes the Selection Relevance Score, a retraining-free metric for evaluating the quality of training sample subsets as example-based explanations. It demonstrates that the commonly used "select top-k by influence" strategy frequently underperforms random selection, and introduces a new strategy that balances influence and representativeness.

## Background & Motivation

**Background**: Training data influence estimation methods (e.g., influence functions) can quantify each training document's contribution to model outputs, making them a promising source for example-based explanations. However, humans cannot process thousands of documents, so only a small number of training samples can be selected as explanations in practice.

**Limitations of Prior Work**: (1) Selecting the top-k highest-influence samples is the current default strategy, but high-influence samples tend to be global outliers (e.g., mislabeled data) and are not necessarily the most relevant to the test instance at hand; (2) top-k samples are often highly redundant, leading to diminishing returns; (3) existing evaluation methods either operate in embedding space (whereas ranking is performed in gradient space), rely on class labels (inapplicable to generative tasks), or require retraining (infeasible for LLMs).

**Key Challenge**: Influence estimation methods assign independent scores to individual training samples, but effective explanations require consideration of inter-sample complementarity and redundancy — a good explanation set should collectively cover the key aspects of the model's decision.

**Goal**: (1) Propose a retraining-free metric for evaluating selection quality; (2) expose the shortcomings of common selection strategies; (3) design improved selection strategies.

**Key Insight**: Framing example-based explanation as a gradient reconstruction task — good explanation samples should enable reconstruction of the test instance's gradient via a linear combination of their own gradients.

**Core Idea**: Selection relevance = the ability of selected samples' gradients to reconstruct the test instance's gradient; a high-quality explanation set should maximize reconstruction fidelity.

## Method

### Overall Architecture

The selection quality evaluation is formalized as a gradient reconstruction problem. Given the loss gradient $\nabla\mathcal{L}'$ of a test instance and the gradient matrix $A$ of $k$ selected training samples, the method computes the optimal linear combination $\hat{\nabla\mathcal{L}}' = At$ and measures the reconstruction error. The Selection Relevance Score $\xi^{SR}$ is the ratio of the original gradient norm to the reconstruction error, expressed in dB.

### Key Designs

1. **Selection Relevance Score**:

    - **Function**: Quantifies the overall quality of a selected training sample set as an explanation.
    - **Mechanism**: $\xi^{SR} = \frac{\mathbb{E}[\|G(\omega)\|^2]}{\mathbb{E}[\|G(\omega) - At_\omega\|^2]}$, i.e., the ratio of the expected squared gradient norm to the expected squared reconstruction error. Values $> 0$ dB indicate that the selected samples provide useful information; values $< 0$ dB indicate performance worse than a zero-vector baseline.
    - **Design Motivation**: Reconstruction capability in gradient space directly reflects the explanatory power of training samples over model decisions, and assesses samples as a group rather than scoring them independently.

2. **Constrained Projection**:

    - **Function**: Ensures that the linear combination coefficients satisfy explanation semantics.
    - **Mechanism**: A non-negativity constraint is imposed on coefficients $t$ (preventing irrelevant samples from gaining weight through cancellation) along with a normalization constraint ($\sum t = 1$, rendering $t$ interpretable as relative importance). The unconstrained least-squares solution is first computed, then projected onto the unit simplex.
    - **Design Motivation**: Unconstrained least squares may yield negative coefficients, implying that certain "explanation" samples are in fact contradicting the prediction.

3. **Influence-Representativeness Balanced Selection Strategy**:

    - **Function**: Replaces the naive "select top-k by influence" strategy.
    - **Mechanism**: The selection process jointly considers influence scores and inter-sample diversity/representativeness to avoid redundant selections and dominance by global outliers.
    - **Design Motivation**: Experiments demonstrate that naive top-k selection frequently underperforms random selection, as global outliers and redundant information degrade explanation quality.

### Loss & Training

This paper involves no model training. The Selection Relevance Score is computed analytically (least squares + simplex projection) without any gradient updates. Validation experiments use fine-tuning comparisons to confirm the metric's effectiveness.

## Key Experimental Results

### Main Results

**Selection Relevance Score by Selection Strategy (dB, higher is better)**

| Selection Strategy | k=1 | k=5 | k=10 | k=25 |
|---|---|---|---|---|
| Random Selection | Baseline | Baseline | Baseline | Baseline |
| Top-k (Highest Influence) | < Random | < Random | ≈ Random | > Random |
| Balanced Strategy (Ours) | > Random | > Random | > Random | > Random |

### Ablation Study

| Influence Estimation Method | Combined with Top-k | Combined with Balanced Strategy |
|---|---|---|
| Influence Functions | Poor (many global outliers) | Significant improvement |
| TracIn | Moderate | Improvement |
| TRAK | Relatively good | Further improvement |

### Key Findings

- The top-k selection strategy frequently underperforms random selection at small budgets ($k \leq 10$), primarily due to global outliers and redundancy.
- The Selection Relevance Score is highly correlated with fine-tuning validation metrics, confirming its validity as a proxy evaluation measure.
- Different influence estimation methods significantly affect selection quality: TRAK is better suited for the selection task than influence functions.
- The balanced strategy consistently outperforms both top-k and random selection across all budget sizes and estimation method combinations.

## Highlights & Insights

- The paper reveals an overlooked yet important issue: the quality of example-based explanations depends not only on the accuracy of influence estimation but critically on the selection strategy.
- The finding that "top-k underperforms random" challenges a default assumption in the field.
- The Selection Relevance Score provides the first retraining-free, task-agnostic tool for evaluating selection quality.

## Limitations & Future Work

- Gradient reconstruction as a proxy for explanation quality may not fully capture users' actual needs.
- The constrained projection (non-negativity + normalization) may exclude certain valid reconstruction solutions.
- Gradient computation on large-scale LLMs remains computationally expensive.
- Validation is conducted only on classification tasks; effectiveness on generative tasks remains to be confirmed.

## Related Work & Insights

- **vs. Bhatt et al. (2021)**: Their approach reduces redundancy via an additive diversity-plus-influence objective, but may favor outliers; this paper proposes representativeness as an alternative.
- **vs. Bae et al. (2022)**: They introduce the concept of prediction-constrained influence; the proposed score is highly compatible with their framework.
- **vs. Influence Functions**: The global outlier problem of influence functions is particularly pronounced in the selection task; this paper quantitatively corroborates this observation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The gradient reconstruction perspective and Selection Relevance Score constitute novel evaluation tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic evaluation across multiple influence methods × selection strategies × budget sizes.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous formalization, clear motivation, and in-depth analysis.
- **Value**: ⭐⭐⭐⭐ Provides important evaluation tools and practical guidance for the example-based explanation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[ICLR 2026\] Lossless Vocabulary Reduction for Auto-Regressive Language Models](../../ICLR2026/llm_pretraining/lossless_vocabulary_reduction_for_auto-regressive_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)

</div>

<!-- RELATED:END -->
