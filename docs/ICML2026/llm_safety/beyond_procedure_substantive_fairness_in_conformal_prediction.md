---
title: >-
  [Paper Note] Beyond Procedure: Substantive Fairness in Conformal Prediction
description: >-
  [ICML2026][LLM Safety][Conformal Prediction] This paper moves beyond the procedural fairness perspective of Conformal Prediction (CP) to focus on the substantive fairness of downstream decision-making. It theoretically p…
tags:
  - "ICML2026"
  - "LLM Safety"
  - "Conformal Prediction"
  - "Substantive Fairness"
  - "Differences in Prediction Set Sizes"
  - "Label Clustering"
  - "LLM Evaluators"
date: 2026-05-08
content_hash: 7195208ab3b074c4
---

# Beyond Procedure: Substantive Fairness in Conformal Prediction

**Conference**: ICML2026  
**arXiv**: [2602.16794](https://arxiv.org/abs/2602.16794)  
**Code**: https://github.com/layer6ai-labs/llm-in-the-loop-conformal-fairness  
**Area**: AI Safety/Fairness  
**Keywords**: Conformal Prediction, Substantive Fairness, Differences in Prediction Set Sizes, Label Clustering, LLM Evaluators  

## TL;DR
This paper moves beyond the procedural fairness perspective of Conformal Prediction (CP) to focus on the substantive fairness of downstream decision-making. It theoretically proves and experimentally validates that **equalizing prediction set sizes** (rather than equalizing coverage) is the procedural metric strongly correlated with substantive fairness. The authors propose a scalable evaluation framework based on LLM-in-the-loop and a label-clustered CP method to effectively balance utility and fairness.

## Background & Motivation

**Background**: Conformal Prediction (CP) provides distribution-free uncertainty quantification for machine learning models by constructing prediction sets that satisfy $\mathbb{P}[y \in \mathcal{C}(x)] \geq 1-\alpha$, giving statistical guarantees. Regarding fairness, existing research primarily focuses on **procedural fairness**, which aims to ensure equalized coverage across demographic groups (Equalized Coverage). For instance, Mondrian CP calibrates thresholds independently for each sensitive group.

**Limitations of Prior Work**: Equalized coverage does not equate to downstream decision fairness. A CP method can achieve 90% coverage for all groups but output compact and useful prediction sets for one group while producing large and useless sets for another. Cresswell et al. (2025) found through human experiments that although Mondrian CP equalizes coverage, it actually exacerbates group differences in downstream prediction accuracy (disparate impact).

**Key Challenge**: Equalized Coverage and Equalized Set Size are two competing objectives—pursuing the former often comes at the expense of the latter, whereas the latter truly impacts downstream fairness. However, this association previously lacked theoretical explanation and large-scale empirical validation.

**Goal**: (1) Establish a scalable substantive fairness evaluation framework to replace expensive human experiments; (2) Clarify the quantitative relationship between procedural metrics and substantive fairness; (3) Theoretically analyze and verify why label-clustered CP can effectively reduce differences in set sizes.

**Key Insight**: The authors observe that the "accuracy improvement" obtained by a downstream decision-maker from prediction sets is the true measure of fairness, rather than the statistical properties of the sets themselves. By using LLMs to approximate human decision-making behavior, this downstream group difference can be evaluated at scale and low cost.

**Core Idea**: Replace human experiments with an LLM-in-the-loop evaluator to measure substantive fairness (maxROR), and decompose the prediction set size difference into three interpretable components via theoretical bounds to guide the use of label-clustered CP for reducing downstream unfairness.

## Method

### Overall Architecture
The complete pipeline of this paper consists of three layers: (1) A base classifier $f$ outputs prediction probabilities; (2) A CP method constructs prediction sets $\mathcal{C}(x)$ based on a calibration set; (3) An LLM decision-maker makes final predictions assisted by the prediction sets. Fairness is not measured at step (2), but rather at step (3) by comparing the "accuracy improvement" across groups.

### Key Designs

1.  **Theoretical Decomposition of Prediction Set Size Differences (Theorem 4.1)**:
    - **Function**: Explains why label-clustered CP can reduce inter-group prediction set size differences $\Delta_{a,b}$.
    - **Mechanism**: For a label clustering mapping $h: \mathcal{Y} \to [K]$, the upper bound of $\Delta_{a,b}$ is decomposed into three terms: (I) intra-cluster label heterogeneity $\max_k \epsilon_{k,a}$, which measures set size differences for different labels within the same cluster; (II) inter-cluster difference $\max_k \mu_{k,a} - \min_k \mu_{k,a}$, which measures the dispersion of expected set sizes across different clusters; (III) inter-group intra-label difference $|\sum_y \mathbb{P}(Y=y|A=b)(r_{y,a}-r_{y,b})|$, which measures set size differences between groups for the same label. When $K=1$ (Marginal CP), term (I) is large because all labels are mixed; when $K=|\mathcal{Y}|$, term (II) is large because rare label calibration is unstable. An appropriate $K$ can control both simultaneously.
    - **Design Motivation**: Converts the unobservable "substantive fairness" problem into an optimizable procedural metric (set size difference) and provides a basis for selecting the hyperparameter $K$.

2.  **LLM-in-the-loop Substantive Fairness Evaluation Framework**:
    - **Function**: Measures the downstream fairness of CP methods in a low-cost and scalable manner.
    - **Mechanism**: For each test sample $x_j$ and CP method $t$, let the LLM generate $M$ independent predictions assisted by prediction sets to calculate accuracy $R_{jt}$. Using GEE regression $\text{logit}(\mathbb{E}[R_{jt}]) \sim \text{treat}_t \times \text{group}_j + \text{diff}_j + \text{adoption}_{jt}$ for controlling confounders, group-specific improvements $\delta_{t,a}$ are extracted. Finally, $\text{maxROR}_t = \max_{a,b}(\text{OR}_{t,a}/\text{OR}_{t,b}) - 1$ is defined as the substantive unfairness metric. Calculating relative to a control baseline without prediction sets using odds ratios helps eliminate confounders like task difficulty.
    - **Design Motivation**: Human experiments cost approximately £1500 per 30k responses, while LLM evaluators are as low as $1 per 60k predictions and avoid human fatigue and learning effects. Experiments also validated that qualitative rankings align with human assessments.

3.  **Fairness Advantage Mechanism of Label-Clustered CP**:
    - **Function**: Reduces prediction set size differences without explicitly relying on sensitive attributes.
    - **Mechanism**: Unlike Mondrian CP, which calibrates separately by group, label-clustered CP groups $\mathcal{Y}$ into $K$ clusters based on label difficulty similarity, and each cluster independently calibrates a threshold $\hat{q}_k$. Label $y$ is included in the set if and only if $s(x_{\text{test}}, y) \leq \hat{q}_{h(y)}$. Since thresholds are shared across groups (within the same cluster) and calibration data is pooled, it avoids the variance inflation and artificial group differences caused by splitting the calibration set in Mondrian.
    - **Design Motivation**: Mondrian and Group-Clustered CP explicitly condition on sensitive attributes, which amplifies set size differences despite equalizing coverage. Label clustering naturally bypasses group information and achieves a fairer distribution of sets indirectly through label-level adaptivity.

## Key Experimental Results

### Background
Covers four modalities (Image/Text/Audio/Tabular) and four datasets (FACET, BiosBias, RAVDESS, ACSIncome). Compares five CP methods (Marginal, Mondrian, Label-Clustered, Group-Clustered, Backward) with $1-\alpha=0.9$.

### Main Results: Substantive Fairness maxROR (%)

| CP Method | FACET | BiosBias | RAVDESS | ACSIncome | Average Rank |
|-----------|-------|----------|---------|-----------|--------------|
| Marginal | 9.0 | 6.9 | 11 | — | Medium |
| Mondrian | 38 | 8.1 | 79 | — | Worst |
| Label-Clustered | — | — | One of lowest | One of lowest | **Best** |
| Group-Clustered | High | — | High | — | Poor |
| Backward | Lowest | Lowest | Relatively High | Relatively High | Medium |

> Label-Clustered CP has a significantly lower maxROR on RAVDESS and ACSIncome compared to Backward, while providing higher accuracy improvement (better utility). Mondrian and Group-Clustered show the most severe unfairness on FACET and RAVDESS.

### LLM Evaluator Validation: Alignment with Human Experiments

| Evaluation Mode | Dataset | Marginal maxROR% | Mondrian maxROR% | Qualitative Rank Consistency |
|-----------------|---------|-------------------|-------------------|-----------------------------|
| Human-in-the-loop | FACET | 26 | 51 | ✓ |
| Human-in-the-loop | BiosBias | 12 | 33 | ✓ |
| Human-in-the-loop | RAVDESS | 1.0 | 28 | ✓ |
| LLM-in-the-loop | FACET | 9.0 | 38 | ✓ |
| LLM-in-the-loop | BiosBias | 6.9 | 8.1 | ✓ |
| LLM-in-the-loop | RAVDESS | 11 | 79 | ✓ |

> The LLM evaluator successfully replicated the unfairness ranking of Mondrian > Marginal across all three datasets, validating its feasibility as a surrogate for human experiments.

### Key Findings: Relationship between Procedural Metrics and Substantive Fairness
- **Coverage gap is negatively correlated with maxROR**: Equalizing coverage actually increases downstream unfairness (negative regression slopes across all 4 datasets).
- **Set size gap is positively correlated with maxROR**: Reducing the gap in set sizes lowers downstream unfairness (positive regression slopes across all 4 datasets).
- The set size difference for label-clustered CP follows a **V-shaped curve** relative to the number of clusters $K$, with the optimum at $K=2$, validating the prediction of Theorem 4.1.

## Highlights & Insights
- **Subversive Conclusion**: CP fairness research has long focused on Equalized Coverage. This paper powerfully argues that this is the wrong goal—Equalized Set Size is the correct proxy for substantive fairness.
- **Low-cost Evaluation**: LLM-in-the-loop reduces fairness evaluation costs from £1500 to $1, making systematic comparisons across methods and modalities possible for the first time.
- **Theoretical-Empirical Closed Loop**: The three-component decomposition of Theorem 4.1 numerically validates the tightness of the bound on RAVDESS, and the V-shaped curve matches empirical results perfectly.
- **Practical Advice**: Avoid conditioning on demographic attributes (Mondrian); prioritize label-clustered CP and use the set size gap as a diagnostic tool for hyperparameter $K$ selection.

## Limitations & Future Work
- LLM evaluators differ from humans in **absolute values** (only qualitative ranking is consistent) and cannot fully replace human experiments.
- Only correlation was studied; a **causal relationship** from procedural metrics to substantive fairness has not been established (the authors propose controlling the adoption rate for future work).
- The optimal $K$ for minimizing set size gap and maxROR in label-clustered CP does not perfectly coincide; selecting $K$ still requires downstream validation.
- Experiments covered only 4 datasets and a single coverage value for $\alpha=0.1$.

## Related Work & Insights
- **Cresswell et al. (2025)** first revealed the disparate impact of Mondrian CP through human experiments; this paper systematizes and significantly extends that work.
- **Ding et al. (2023)** proposed clustered conformal prediction originally to improve conditional coverage; this paper discovers its unexpected advantage in fairness.
- **Insight**: The LLM-in-the-loop evaluation paradigm can be generalized to other AI fairness scenarios requiring human evaluation (e.g., recommendation systems, information retrieval).

## Rating
- Novelty: ⭐⭐⭐⭐ (Shifting CP fairness from procedural metrics to substantive outcomes; LLM replacement for human evaluation is a novel contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive comparison across 4 modalities × 5 methods + theoretical validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, interlinking theory, empirical results, and practical advice)
- Value: ⭐⭐⭐⭐ (Valuable for correcting the research direction in CP fairness)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICLR 2026\] Moving Beyond Medical Exams: A Clinician-Annotated Fairness Dataset of Real-World Tasks and Ambiguity in Mental Healthcare](../../ICLR2026/llm_safety/moving_beyond_medical_exams_a_clinician-annotated_fairness_dataset_of_real-world.md)
- [\[ACL 2026\] Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data](../../ACL2026/llm_safety/gap-k_measuring_top-1_prediction_gap_for_detecting_pretraining_data.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](../../CVPR2026/llm_safety/interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](../../ACL2026/llm_safety/beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
