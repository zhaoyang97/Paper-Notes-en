---
title: >-
  [Paper Note] Compact Example-Based Explanations for Language Models
description: >-
  [ACL 2026][LLM Pretraining][Training Data Influence] This paper proposes Selection Relevance Score, a retraining-free metric to evaluate the quality of training sample subsets as example-based explanations. It demonstrat…
tags:
  - "ACL 2026"
  - "LLM Pretraining"
  - "Training Data Influence"
  - "Example-based Explanations"
  - "Selection Relevance"
  - "Gradient Reconstruction"
  - "Redundancy Elimination"
date: 2026-05-08
content_hash: dd4c9b33801ef0d8
---

# Compact Example-Based Explanations for Language Models

**Conference**: ACL 2026  
**arXiv**: [2601.03786](https://arxiv.org/abs/2601.03786)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Training Data Influence, Example-based Explanations, Selection Relevance, Gradient Reconstruction, Redundancy Elimination

## TL;DR

This paper proposes Selection Relevance Score, a retraining-free metric to evaluate the quality of training sample subsets as example-based explanations. It demonstrates that common "top-influence" selection strategies are often inferior to random selection and introduces a new strategy that balances influence with representativeness.

## Background & Motivation

**Background**: Training data influence estimation methods (e.g., influence functions) quantify the contribution of each training document to the model output, serving as a promising source for example-based explanations. However, since humans cannot process thousands of documents, only a small number of training samples can be selected as explanations in practice.

**Limitations of Prior Work**: (1) Selecting the top-k highest influence samples is the current default strategy, but high-influence samples are often global outliers (e.g., mislabeled data) that are not necessarily the most relevant to the current test instance; (2) Redundancy among top-influence samples leads to diminishing returns; (3) Existing evaluations either operate in the embedding space (whereas ranking occurs in the gradient space), rely on class labels (inapplicable to generative tasks), or require expensive retraining (infeasible for LLMs).

**Key Challenge**: Influence estimation methods generate independent influence scores for each training sample. However, as explanations, the complementarity and redundancy between samples must be considered—a good explanatory set should collectively cover key aspects of the model's decision.

**Goal**: (1) Propose a retraining-free metric to evaluate selection quality; (2) Reveal deficiencies in common selection strategies; (3) Design better selection strategies.

**Key Insight**: Example-based explanation is treated as a gradient reconstruction task—good explanatory samples should enable the reconstruction of the test instance's gradient through a linear combination of their own gradients.

**Core Idea**: Selection Relevance = The ability of selected samples' gradients to reconstruct the test instance's gradient. A high-quality set of explanations should maximize reconstruction accuracy.

## Method

### Overall Architecture

The evaluation of selection quality is formalized as a gradient reconstruction problem: Given the loss gradient of a test instance $\nabla\mathcal{L}'$ and a gradient matrix $A$ of k selected training samples, the reconstruction error of the optimal linear combination $\hat{\nabla\mathcal{L}}' = At$ is calculated. The Selection Relevance Score $\xi^{SR}$ is defined as the ratio between the original gradient norm and the reconstruction error (expressed in dB).

### Key Designs

1.  **Selection Relevance Score**:
    - **Function**: Quantifies the comprehensive quality of the selected training sample set as an explanation.
    - **Mechanism**: $\xi^{SR} = \frac{\mathbb{E}[\|G(\omega)\|^2]}{\mathbb{E}[\|G(\omega) - At_\omega\|^2]}$, representing the ratio of expected squared gradient norm to expected squared reconstruction error. A value > 0 dB indicates that the selected samples provide useful information, while < 0 dB suggests they perform worse than a zero-vector baseline.
    - **Design Motivation**: Reconstruction capability in the gradient space directly reflects the explanatory power of training samples for model decisions; it considers sample combinations rather than independent scores.

2.  **Constrained Projection**:
    - **Function**: Ensures that the linear combination coefficients satisfy explanatory semantics.
    - **Mechanism**: Constraints of non-negativity (preventing irrelevant samples from gaining weight through cancellation) and normalization ($\sum t = 1$, allowing $t$ to be interpreted as relative importance) are applied to the coefficients $t$. An unconstrained least squares solution is computed first, followed by projection onto the unit simplex.
    - **Design Motivation**: Unconstrained least squares may produce negative coefficients, implying that some "explanatory" samples actually contradict the prediction.

3.  **Balanced Influence and Representativeness Strategy**:
    - **Function**: Replaces the naive "top-k influence" strategy.
    - **Mechanism**: Considers both the influence score and the diversity/representativeness among samples during selection to avoid redundant selections and dominance by global outliers.
    - **Design Motivation**: Experiments prove that naive top-k selection is often worse than random selection because global outliers and redundant information degrade explanation quality.

### Loss & Training

This paper does not involve model training. The Selection Relevance Score is computed analytically (least squares + simplex projection) without gradient updates. Validation experiments confirm the score's effectiveness through fine-tuning comparisons.

## Key Experimental Results

### Main Results

**Selection Relevance Scores for different selection strategies (dB, higher is better)**

| Selection Strategy | k=1 | k=5 | k=10 | k=25 |
| :--- | :--- | :--- | :--- | :--- |
| Random Selection | Baseline | Baseline | Baseline | Baseline |
| Top-k (Highest Influence) | < Random | < Random | ≈ Random | > Random |
| Balanced Strategy (Ours) | > Random | > Random | > Random | > Random |

### Ablation Study

| Influence Method | Coupled with Top-k | Coupled with Balanced Strategy |
| :--- | :--- | :--- |
| Influence Function | Poor (due to global outliers) | Significant Gain |
| TracIn | Moderate | Gain |
| TRAK | Better | Further Gain |

### Key Findings

- The Top-k selection strategy is often inferior to random selection at small budgets (k≤10)—primarily due to global outliers and redundancy.
- The Selection Relevance Score correlates highly with fine-tuning validation metrics, proving its effectiveness as a proxy evaluation metric.
- Different influence estimation methods significantly affect selection quality: TRAK is more suitable for selection tasks than traditional Influence Functions.
- The balanced strategy outperforms both Top-k and random selection across all budget sizes and estimation method combinations.

## Highlights & Insights

- Reveals an overlooked issue: the quality of example-based explanations depends not only on the accuracy of influence estimation but also heavily on the selection strategy.
- The finding that "Top-k is worse than random" challenges the default assumptions in the field.
- The Selection Relevance Score provides the first retraining-free, task-agnostic tool for evaluating selection quality.

## Limitations & Future Work

- Gradient reconstruction as a proxy for explanation quality may not fully capture actual user requirements.
- Constrained projection (non-negative + normalized) might exclude certain valid reconstruction solutions.
- Gradient computation remains expensive on large-scale LLMs.
- Validation was limited to classification tasks; effectiveness on generative tasks remains to be confirmed.

## Related Work & Insights

- **vs. Bhatt et al. (2021)**: They reduce redundancy via an additive objective of diversity and influence but may favor outliers; this paper proposes representativeness as an alternative.
- **vs. Bae et al. (2022)**: Proposed the concept of predictive constrained influence, which is highly compatible with the score in this paper.
- **vs. Influence Functions**: The global outlier issue of influence functions is particularly prominent in selection tasks, which this paper confirms quantitatively.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The gradient reconstruction perspective and Selection Relevance Score are novel evaluation tools.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic evaluation across multiple influence methods, selection strategies, and budgets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous formalization, clear motivation, and in-depth analysis.
- **Value**: ⭐⭐⭐⭐ Provides important evaluation tools and practical suggestions for the field of example-based explanations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Fine-tuning vs. In-context Learning in Large Language Models: A Formal Language Learning Perspective](fine-tuning_vs_in-context_learning_in_large_language_models_a_formal_language_le.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](../../ICML2026/llm_pretraining/consistent_diffusion_language_models.md)
- [\[ACL 2026\] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models](script_a_subcharacter_compositional_representation_injection_module_for_korean_p.md)
- [\[NeurIPS 2025\] Learning in Compact Spaces with Approximately Normalized Transformer](../../NeurIPS2025/llm_pretraining/learning_in_compact_spaces_with_approximately_normalized_transformer.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)

</div>

<!-- RELATED:END -->
