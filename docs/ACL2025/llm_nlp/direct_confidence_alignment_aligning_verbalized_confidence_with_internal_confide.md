---
title: >-
  [Paper Note] Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Confidence calibration] The paper proposes Direct Confidence Alignment (DCA), which utilizes DPO to align the verbalized confidence of LLMs with their internal token probability confidence, thereby improving the consistency and transparency of the model's confidence expressions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Confidence calibration"
  - "DPO"
  - "Internal confidence"
  - "Verbalized confidence"
  - "LLM reliability"
date: 2026-05-08
content_hash: 5e3bcbbd7c6ffe55
---

# Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence In Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2512.11998](https://arxiv.org/abs/2512.11998)  
**Code**: Not publicly available  
**Area**: LLM/NLP  
**Keywords**: Confidence calibration, DPO, Internal confidence, Verbalized confidence, LLM reliability  

## TL;DR

The paper proposes Direct Confidence Alignment (DCA), which utilizes DPO to align the verbalized confidence of LLMs with their internal token probability confidence, thereby improving the consistency and transparency of the model's confidence expressions.

## Background & Motivation

**Core Problem:** There is a severe discrepancy between the internal confidence of the LLM ($C_i$, based on the softmax of token probabilities) and its verbalized confidence ($C_v$, the confidence percentage outputted in the model's answer). For example, an answer with an internal probability of only 30% might claim a 95% verbalized confidence.

**Limitations of Prior Work:**
- Calibration methods such as temperature scaling and self-consistency focus on aligning confidence with accuracy, ignoring the intrinsic gap between $C_v$ and $C_i$.
- RLHF can compromise the calibration of internal logits, making $C_i$ itself unreliable.
- Black-box models cannot access logits, which restricts calibration methods based on $C_i$.

**Core Motivation:** Even if $C_i$ is not perfectly calibrated, ensuring consistency between $C_v$ and $C_i$ is still highly valuable—it makes the model's expression of uncertainty more transparent and consistent. DPO's preference pair format is naturally suited for this type of alignment task.

## Method

### Overall Architecture

The DCA pipeline consists of four steps: (1) Use the base model to generate answers with $C_v$ for given questions $\rightarrow$ (2) Extract $C_i$ from the softmax probability of the answer tokens $\rightarrow$ (3) Construct preference data pairs: the original answer (containing the original $C_v$) is labelled as rejected, while the answer where $C_v$ is replaced with $C_i$ is labelled as chosen $\rightarrow$ (4) Train with DPO to complete alignment.

### Key Designs

1. **Verbalized Confidence Extraction:** A specific prompt template is used to require the model to output "Probability: X%" at the end of the response. The numerical value is parsed to obtain $C_v$, with an extraction error rate of <5%.

2. **Internal Confidence Extraction:** The softmax probability when the model outputs the answer token (e.g., A/B/C/D) is taken as $C_i$, directly reflecting the model's internal level of certainty regarding the answer.

3. **Preference Data Construction:** A pair of data is generated for each sample: the original response is designated as rejected, while the response where the $C_v$ value is replaced with the $C_i$ value serves as chosen, with all other text remaining completely identical.

### Evaluation Metrics

Three new metrics based on calibration error $\epsilon = C_v - C_i$ are proposed:
- **$\sigma_\epsilon$ (calibration error standard deviation)**: measures the variability of $\epsilon$.
- **$\overline{|\epsilon|}$ (mean absolute calibration error)**: measures the average discrepancy between $C_v$ and $C_i$.
- **$\sigma_M$ (calibration error standard error)**: estimates the sampling uncertainty of the mean alignment.

## Experiments

### Main Results: DCA Alignment Effect (Average over Four Datasets)

| Model | Method | ρ↑ | σ_ε↓ | |ε|↓ | σ_M↓ |
|------|------|------|------|------|------|
| Gemma-2-9B-Instruct | Vanilla | 0.34 | 16.97 | 9.91 | 0.57 |
| | **DCA** | **0.42** | **13.79** | **5.03** | **0.46** |
| Llama-3.2-3B-Instruct | Vanilla | 0.28 | 41.19 | 38.67 | 1.40 |
| | DCA | 0.23↓ | **22.88** | 44.03↑ | **0.75** |
| Mistral-7B-Instruct | Vanilla | 0.19 | 25.63 | 22.96 | 0.85 |
| | DCA | 0.13↓ | **22.93** | 48.93↑ | **0.74** |

### Impact of DCA on Accuracy

| Model | OpenBookQA | TruthfulQA | CosmosQA | MMLU |
|------|------|------|------|------|
| Gemma-2-9B Vanilla→DCA | 86.06→**86.21** | 59.68→**60.85** | 79.63→**80.01** | 72.41→72.05 |
| Llama-3.2-3B Vanilla→DCA | 47.14→**64.00** | 29.71→**37.75** | 66.43→**73.55** | 39.92→**49.77** |
| Mistral-7B Vanilla→DCA | 59.00→58.23↓ | 32.84→20.98↓ | 60.48→54.02↓ | 55.91→48.85↓ |

### Key Findings

1. **DCA performance is highly model-dependent**: Gemma-2-9B consistently improved across all metrics ($\rho$ +0.08, $|\epsilon|$ -4.88), whereas Mistral-7B deteriorated on multiple metrics.
2. Gemma's success may be partly because the initial distributions of its $C_v$ and $C_i$ were already heavily biased toward the 90-100% interval, and DCA reinforced this concentration trend.
3. $\sigma_\epsilon$ and $\sigma_M$ generally improved across all models, indicating that DCA at least reduces the variance of calibration errors.
4. The effect of DCA on accuracy is inconsistent: stable for Gemma, significantly improved for Llama (+16.86% on OpenBookQA), and significantly decreased for Mistral (-11.86% on TruthfulQA).
5. In-domain and out-of-domain datasets show similar performance patterns, implying that effectiveness depends more on the model architecture than on the task type.

## Highlights & Insights

- **Novel Calibration Perspective**: Instead of striving for alignment with ground-truth accuracy, it aligns the model's own two forms of confidence expression, focusing on transparency rather than accuracy.
- **Methodological Simplicity**: Cleverly leverages the preference pair format of DPO, requiring only the replacement of $C_v$ with $C_i$ to construct the training data.
- **Three New Metrics**: $\sigma_\epsilon$, $\overline{|\epsilon|}$, and $\sigma_M$ measure the quality of confidence alignment from different perspectives, providing a more comprehensive evaluation than a single Spearman correlation.

## Limitations & Future Work

- Requires access to model logits, making it inapplicable to closed-source models such as GPT-4.
- The method assumes $C_i$ is more reliable than $C_v$ as a reference signal, but $C_i$ itself might be poorly calibrated after RLHF.
- The preference data contains incorrect answer options, leading to a severe drop in Mistral's accuracy.
- Validated on only three models, and the performance on two of them was suboptimal, raising doubts about generalizability.
- Gemma's "success" might be an illusion of confidence distribution collapsing into high-value intervals.

## Related Work & Insights

- **Confidence Calibration**: Temperature scaling (Guo et al.), self-consistency methods (Wang et al.), CQO alignment (Tao et al.).
- **Verbalized Confidence**: Multi-sample averaging (Tian et al.), multi-temperature multi-prompt strategy (Xiong et al.).
- **Confidence-Probability Alignment**: Kumar et al. first defined Confidence-Probability Alignment.
- **DPO**: Rafailov et al., Direct Preference Optimization as an alternative to RLHF.

## Rating

| Dimension | Score |
|------|------|
| Novelty | 7/10 |
| Effectiveness | 5/10 |
| Experimental Thoroughness | 6/10 |
| Writing Quality | 7/10 |
| Overall Score | 6/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs](do_language_models_mirror_human_confidence_exploring_psychological_insights_to_a.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)
- [\[ACL 2025\] Revisiting Epistemic Markers in Confidence Estimation: Can Markers Accurately Reflect Large Language Models' Uncertainty?](revisiting_epistemic_markers_in_confidence_estimation_can_markers_accurately_ref.md)
- [\[ACL 2025\] Aligning Large Language Models with Implicit Preferences from User-Generated Content](pugc_align_implicit_pref_ugc.md)
- [\[ACL 2025\] Binary Classifier Optimization for Large Language Model Alignment](bco_binary_classifier_alignment.md)

</div>

<!-- RELATED:END -->
