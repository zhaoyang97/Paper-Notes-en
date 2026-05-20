---
title: >-
  [Paper Note] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning
description: >-
  [NeurIPS 2025][Time Series][Martingale Score] This paper proposes the Martingale Score as an unsupervised metric that quantifies belief entrenchment in LLM reasoning processes based on the martingale property from Bayesi…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Martingale Score"
  - "belief entrenchment"
  - "Bayesian rationality"
  - "LLM reasoning"
  - "unsupervised evaluation"
date: 2026-05-08
content_hash: a62dd158b33ac371
---

# Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2512.02914](https://arxiv.org/abs/2512.02914)  
**Code**: None  
**Area**: LLM Reasoning Evaluation / Time Series
**Keywords**: Martingale Score, belief entrenchment, Bayesian rationality, LLM reasoning, unsupervised evaluation

## TL;DR

This paper proposes the Martingale Score as an unsupervised metric that quantifies belief entrenchment in LLM reasoning processes based on the martingale property from Bayesian statistics. The study finds that belief entrenchment is pervasive across models and domains, and is significantly correlated with degraded accuracy.

## Background & Motivation

**Background**: LLM reasoning techniques (CoT, reinforced reasoning, etc.) are advancing rapidly, yet whether reasoning processes genuinely "seek truth" remains unclear.

**Limitations of Prior Work**: Existing evaluation methods are primarily outcome-based and cannot assess the quality of the reasoning process itself; they are also inapplicable in open-ended domains lacking ground truth, such as value judgments and academic peer review.

**Key Challenge**: LLM reasoning may exhibit "belief entrenchment"—systematically biasing belief updates toward prior views rather than new evidence—yet it is difficult to distinguish rational updating from bias in individual cases.

**Goal**: To propose a ground-truth-free, domain-agnostic metric for evaluating the quality of reasoning processes.

**Key Insight**: The martingale property in Bayesian statistics states that the direction of rational belief updates should not be predictable from the prior.

**Core Idea**: If a model's belief updates can be reliably predicted from its prior beliefs, the martingale property is violated, indicating the presence of belief entrenchment.

## Method

### Overall Architecture

The LLM reasoning process is modeled as a belief-updating procedure: the output at the start of reasoning is treated as the "prior belief" $b_{\text{prior}}$, and the output after reasoning as the "posterior belief" $b_{\text{posterior}}$. Regression analysis is then used to test whether the belief update $\Delta b$ is predictable from $b_{\text{prior}}$.

### Key Designs

1. **Definition of Martingale Score**:

    - **Function**: Quantifies the strength of the linear relationship between belief updates and prior beliefs.
    - **Design Motivation**: The martingale property requires $E[\Delta b | b_{\text{prior}} = p] = 0$, meaning the direction of belief updates must not be predictable from the prior.
    - **Mechanism**: The regression $\Delta b = \beta_1 \cdot b_{\text{prior}} + \beta_0 + \epsilon$ is estimated, and the Martingale Score is defined as $M = \hat{\beta}_1$:
    $$M = \hat{\beta}_1 = \frac{\sum_{i=1}^{n}(\Delta b_i - \overline{\Delta b})(b_{\text{prior},i} - \overline{b_{\text{prior}}})}{\sum_{i=1}^{n}(b_{\text{prior},i} - \overline{b_{\text{prior}}})^2}$$
    - **Novelty**: The linear coefficient is preferred over $R^2$ (to avoid confounding factors) or logistic regression (which ignores update magnitude), yielding a simple yet empirically reliable metric.

2. **Theoretical Grounding (Proposition 1)**:

    - **Function**: Establishes that the Martingale Score is a statistically principled measure of martingale property violations.
    - **Design Motivation**: Ensures the statistical rigor of the proposed metric.
    - **Mechanism**: It is proven that if the martingale property holds, the population coefficient $\beta_1 = 0$, and the OLS estimator $\hat{\beta}_1$ is an unbiased and consistent estimator of $\beta_1$.
    - **Key Implication**: $E(M) = 0$ and $M \xrightarrow{p} 0$ as sample size grows.

3. **LLM-as-Judge Belief Extraction**:

    - **Function**: Extracts "expressed belief" scores $b \in [0,1]$ from LLM outputs.
    - **Design Motivation**: LLM internal confidence calibration is poor; expressed beliefs better reflect what users actually perceive.
    - **Mechanism**: An independent judge model (e.g., GPT-4o) evaluates reasoning steps and assigns belief scores.
    - **Validation**: High agreement is observed across multiple judge models and between human and LLM judges (Pearson $r$ up to 0.88).

### Experimental Domain Design

Three domains are selected to satisfy: (1) non-memorizable, (2) containing new evidence capable of shifting beliefs, and (3) possessing post-hoc verifiable ground truth:
- **Forecasting**: Prediction questions from Metaculus/Polymarket.
- **Value Judgment (r/ChangeMyView)**: Opinion debates on Reddit.
- **Academic Peer Review (OpenReview)**: Acceptance decisions for ICLR papers.

## Key Experimental Results

### Main Results

Martingale Scores under CoT reasoning across models and domains (positive values indicate belief entrenchment):

| Model | Forecasting (CoT) | ChangeMyView (CoT) | OpenReview (CoT) |
|------|-------------------|---------------------|-------------------|
| GPT-4o (No Prompt) | +0.0018 | +0.0671* | +0.0734* |
| DeepSeek R1 (No Prompt) | +0.0207* | +0.0502* | +0.0676* |
| DeepSeek V3 (No Prompt) | +0.0335* | +0.1155* | +0.1028* |
| Gemini 2.0 Flash (No Prompt) | +0.0764* | +0.1209* | +0.1012* |
| Llama 4 Scout (No Prompt) | +0.0350* | +0.1420* | +0.0890* |
| Llama 4 Maverick (No Prompt) | +0.0178* | +0.1038* | +0.0823* |

*Denotes $p < 0.05$ significance. Under CoT, 51 out of 54 experimental conditions yield positive scores.

### Relationship Between Belief Entrenchment and Accuracy

- The absolute value of the Martingale Score is positively correlated with the Brier Score: stronger belief entrenchment corresponds to lower forecasting accuracy.
- This relationship remains significant after controlling for confounding factors (domain, reasoning method, model, prompt).
- At a Martingale Score of 0.04, predictive performance already falls below random chance.

### Ablation Study

| Condition | Mean Martingale Score (95% CI) |
|---------|-------------------------------|
| Prior-conforming prompt | 0.082 ± 0.018 |
| No system prompt | 0.075 ± 0.014 |
| Critical thinking prompt | 0.072 ± 0.018 |

### Key Findings

- Belief entrenchment is pervasive across **all models, all domains, and all prompt types**.
- The value judgment domain (r/ChangeMyView) exhibits the most severe entrenchment, while the forecasting domain shows the least.
- Even with "critical thinking" prompts, belief entrenchment remains significant, ruling out prompt-induced artifacts.
- Debate-based reasoning partially mitigates entrenchment in some settings, but results are inconsistent.

## Highlights & Insights

- **Highly innovative theoretical framework**: Introducing the Bayesian martingale property into LLM reasoning evaluation establishes a mathematically grounded metric for reasoning process quality.
- **Unsupervised and domain-agnostic**: The Martingale Score requires no ground truth and is applicable to open-ended problems.
- **Bridging process and outcome**: It fills the gap left by outcome-based evaluation and reveals how reasoning processes affect final accuracy.
- **Thorough validation of judge consistency**: Cross-LLM and human–LLM agreement are rigorously verified.

## Limitations & Future Work

- The correlation between the Martingale Score and the Brier Score could not be established in the OpenReview domain, possibly due to ground truth quality issues.
- The degree of entrenchment in reinforced reasoning (e.g., DeepSeek R1's deep reasoning mode) has not been systematically studied.
- The analysis is restricted to belief updates within the internal reasoning process and does not address external evidence retrieval scenarios.
- The metric relies on LLM judges for belief extraction, introducing an indirect measurement step.

## Related Work & Insights

- The work is deeply connected to confirmation bias theory in **cognitive bias research**; belief entrenchment can be viewed as its operational definition.
- The framework can be extended into a unified paradigm for evaluating other LLM behavioral issues such as **sycophancy** and **herd conformity**.
- A promising future direction is using the Martingale Score as a training objective for debiasing.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First application of the martingale property to LLM reasoning evaluation; the theoretical framework is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 6 models × 3 domains × 3 prompt types × 2 reasoning methods comprehensively; reinforced reasoning is notably absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logic is clear, theoretical derivations are rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Introduces a broadly applicable tool for evaluating reasoning processes, with significant implications for AI safety and trustworthy AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](../../ACL2026/time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)
- [\[NeurIPS 2025\] MASFIN: A Multi-Agent System for Decomposed Financial Reasoning and Forecasting](masfin_a_multi-agent_system_for_decomposed_financial_reasoning_and_forecasting.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[ICLR 2026\] Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment](../../ICLR2026/time_series/rating_quality_of_diverse_time_series_data_by_meta-learning_from_llm_judgment.md)

</div>

<!-- RELATED:END -->
