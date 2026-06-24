---
title: >-
  [Paper Note] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?
description: >-
  [ACL 2025][LLM (Other)][Epistemic markers] This paper defines the concept of "marker confidence" to measure the actual accuracy when LLMs utilize epistemic markers (e.g., "fairly certain"). Through systematic experiments on 7 models and 7 datasets, it is discovered that epistemic markers show stable performance in in-distribution scenarios but are highly unreliable under out-of-distribution conditions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Epistemic markers"
  - "confidence estimation"
  - "uncertainty"
  - "LLM calibration"
  - "out-of-distribution generalization"
date: 2026-05-08
content_hash: 672f119e64e4d05e
---

# Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?

**Conference**: ACL 2025  
**arXiv**: [2505.24778](https://arxiv.org/abs/2505.24778)  
**Code**: [github.com/HKUST-KnowComp/MarCon](https://github.com/HKUST-KnowComp/MarCon)  
**Area**: LLM/NLP  
**Keywords**: Epistemic markers, confidence estimation, uncertainty, LLM calibration, out-of-distribution generalization

## TL;DR

This paper defines the concept of "marker confidence" to measure the actual accuracy when LLMs utilize epistemic markers (e.g., "fairly certain"). Through systematic experiments on 7 models and 7 datasets, it is discovered that epistemic markers show stable performance in in-distribution scenarios but are highly unreliable under out-of-distribution conditions.

## Background & Motivation

With the increasing application of LLMs in high-risk areas (such as medicine and law), accurately assessing the confidence of model outputs has become crucial. Humans typically express confidence through epistemic markers (e.g., "I am fairly confident" or "it is unlikely that") rather than providing numerical values. Since natural language is the primary interface for human-LLM interactions, whether LLMs can reliably reflect their internal confidence via epistemic markers is a critical question.

Prior studies have primarily focused on the differences in understanding epistemic markers between humans and LLMs, concluding that models consistently fail to accurately verbalize confidence. However, this paper points out that even if the markers do not perfectly align with human understanding, they remain useful as long as the model maintains **internal consistency** (i.e., the same marker corresponds to a similar accuracy rate across different scenarios). Therefore, previous research might be insufficient—failing to test whether LLMs can consistently apply their own confidence frameworks.

## Method

### Overall Architecture

This paper proposes a systematic evaluation framework for epistemic marker confidence: (1) defining marker confidence as the actual accuracy when the model uses a specific epistemic marker; (2) calculating the confidence of all markers across multiple QA datasets; (3) evaluating marker stability and consistency from multiple dimensions through 7 evaluation metrics.

### Key Designs

1. **Definition of Marker Confidence**: Given an epistemic marker $W$, dataset $D$, and model $M$, marker confidence is defined as: $\text{Conf}(W, D, M) = \frac{1}{|Q_W|}\sum_{q \in Q_W} \mathbb{I}(M(q))$, which is the accuracy over the subset of questions where the model's generated answer contains the marker $W$. This definition departs from traditional interpretations of semantic uncertainty, focusing instead on the correspondence between markers and actual accuracy.

2. **Seven-dimensional Evaluation Metric Suite**:

    - **I-AvgECE** (In-distribution Average Expected Calibration Error): Measures the alignment between marker confidence and actual accuracy under the same distribution (the lower, the better).
    - **C-AvgECE** (Cross-distribution Average Expected Calibration Error): Evaluates the robustness of marker confidence in out-of-distribution scenarios (the lower, the better).
    - **NumECE** (Numerical ECE): Measures the overall calibration performance of the model's numerical confidence outputs, serving as a baseline comparison.
    - **MAC** (Marker-Accuracy Correlation): Measures the correlation between marker confidence and model accuracy across different datasets based on the Pearson coefficient. 0 indicates no correlation, and 1 indicates perfect positive correlation.
    - **MRC** (Marker Rank Correlation): Measures the consistency of marker confidence rankings across different datasets based on the Spearman coefficient.
    - **I-AvgCV** (In-distribution Average Coefficient of Variation): Captures the dispersion of marker confidence within a dataset (higher values indicate stronger discriminative capability).
    - **C-AvgCV** (Cross-distribution Average Coefficient of Variation): Measures the consistency of marker confidence across different datasets (the lower, the more stable).

3. **Marker Filtering Strategy**: Only markers appearing at least 10 times in the dataset are analyzed to eliminate the influence of randomness. The choice of filtering threshold represents a trade-off between data integrity and reliability.

### Loss & Training

This paper is an analytical study and does not involve model training. The core methodology employs prompts to guide the model to express uncertainty using epistemic markers in its answers, followed by statistical analysis of the relationship between markers and accuracy. A numerical confidence baseline is also designed for comparison.

## Key Experimental Results

### Main Results

| Model | I-AvgECE↓ | C-AvgECE↓ | NumECE↓ | C-AvgCV↓ | MAC | MRC↑ | I-AvgCV |
|------|-----------|-----------|---------|----------|-----|------|---------|
| Llama-3.1-8B | 10.09 | 15.95 | 22.70 | 20.80 | 60.91 | 11.37 | 20.48 |
| Qwen2.5-7B | 7.85 | 23.60 | 21.84 | 31.29 | 68.06 | 11.85 | 22.39 |
| Qwen2.5-32B | 4.78 | 10.40 | 8.86 | 19.24 | 78.20 | 36.97 | 16.26 |
| Mistral-7B | 10.58 | 24.81 | 24.46 | 28.52 | 84.57 | 10.54 | 21.01 |
| GPT-4o | 8.55 | 11.84 | 7.56 | 15.72 | 76.44 | 27.54 | 14.30 |
| GPT-4o-mini | 7.65 | 17.15 | 12.79 | 21.98 | 87.68 | 16.48 | 20.61 |
| Average | 8.17 | 17.73 | 16.60 | 23.43 | 75.69 | 21.34 | 19.84 |

### Ablation Study

| Filtering Threshold | C-AvgCV↓ | MAC | MRC↑ | I-AvgCV |
|---------|----------|-----|------|---------|
| 10 (Main) | 23.43 | 75.69 | 21.34 | 19.84 |
| 50 | 23.84 | 86.62 | 23.02 | 13.75 |
| 100 | 23.90 | 82.71 | 20.91 | 12.24 |

Correlation analysis between model capability and marker consistency:

| Metric | Correlation with Model Accuracy |
|------|-------------------|
| C-AvgCV | -0.88 (Strong negative correlation; stronger capability corresponds to more stable markers) |
| MRC | 0.75 (Strong positive correlation; stronger capability corresponds to more consistent ranking) |

### Key Findings

1. **In-Distribution Stability but Out-of-Distribution Unreliability**: I-AvgECE is consistently lower than C-AvgECE (6/7 models), indicating that markers are well-calibrated under the same distribution but generalize poorly across distributions. The average C-AvgCV reaches 23.43%, showing that marker confidence is highly sensitive to distribution shifts.

2. **Inconsistent Marker Rankings**: MRC is overall low (average 21.34%), meaning models fail to maintain stable rankings of epistemic markers across different datasets. For example, "fairly certain" might have lower confidence than "very likely" in one dataset, but reverse in another.

3. **Clustered Marker Confidence Distribution**: I-AvgCV ranges only between 14% and 24%, showing limited distinctiveness among markers. Only 4 out of 49 (dataset, model) settings contain markers with confidence below 10%, indicating that models are severely inadequate at expressing uncertainty.

4. **Stronger Models Comprehend Better**: C-AvgCV is strongly negatively correlated with model accuracy (-0.88). More capable models (GPT-4o, Qwen2.5-32B) exhibit more stable marker usage.

5. **Marker Confidence Shifts with Model Accuracy**: The MAC of 5/7 models exceeds 0.7. Marker confidence is strongly positively correlated with model accuracy across different datasets, essentially reflecting dataset difficulty rather than true confidence calibration.

## Highlights & Insights

- Proposed the novel definition of "marker confidence," shifting the focus from "whether markers align with human understanding" to "whether the model can use markers self-consistently," providing a more practical and profound perspective.
- Designed a comprehensive seven-dimensional evaluation metric system that offers systematic insights into calibration, ranking, discriminative power, and cross-domain stability.
- Found that stronger models exhibit a better understanding of epistemic markers, implying that marker calibration might naturally improve as model capability increases.
- Unveiled a deeper issue: although LLMs perform well on QA tasks, they do not truly "understand" the meaning of epistemic markers.

## Limitations & Future Work

- Evaluated only on English closed-source QA tasks, without considering other languages or open-ended generation tasks.
- The definition of epistemic markers is relatively simplified, without considering the impact of syntactic structures and context on confidence expression.
- Evaluated only short-answer scenarios; confidence expression in long-form text is much more complex.
- Did not propose improvement solutions, focusing solely on diagnostic analysis. Future work can explore enhancing marker consistency through training or prompting strategies.

## Related Work & Insights

- **Kadavath et al. (2022)** first investigated the capability of LLMs to express uncertainty in answers.
- **Yona et al. (2024)** found that models cannot accurately verbalize confidence, but this paper suggests that their criteria (aligning with human understanding) might be overly strict.
- **Xiong et al. (2024)** explored consistency-based black-box confidence estimation methods, which served as an important reference for the numerical baseline in this paper.
- The findings of this paper offer key insights for trustworthiness evaluation when deploying LLMs in high-risk scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes a new definition of marker confidence and a comprehensive evaluation framework, though the core finding (markers are unreliable) is not very surprising.
- Experimental Thoroughness: ⭐⭐⭐⭐ A comprehensive matrix of experiments across 7 models × 7 datasets, supplemented by a robustness validation of the filtering threshold.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-defined metrics, though the mathematical notation is somewhat dense.
- Value: ⭐⭐⭐⭐ Makes important methodological contributions to the field of LLM confidence estimation, but the lack of improvement schemes reduces its practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Harmonized Uncertainty Estimation for Large Language Models](towards_harmonized_uncertainty_estimation_for_large_language_models.md)
- [\[ACL 2025\] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models](direct_confidence_alignment_aligning_verbalized_confidence_with_internal_confide.md)
- [\[ACL 2025\] Reconsidering LLM Uncertainty Estimation Methods in the Wild](reconsidering_llm_uncertainty_estimation_methods_in_the_wild.md)
- [\[ACL 2025\] Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models?](uncertainty_unveiled_can_exposure_to_more_in-context_examples_mitigate_uncertain.md)
- [\[ACL 2025\] Can Large Language Models Accurately Generate Answer Keys for Health-related Questions?](can_large_language_models_accurately_generate_answer_keys_for_health-related_que.md)

</div>

<!-- RELATED:END -->
