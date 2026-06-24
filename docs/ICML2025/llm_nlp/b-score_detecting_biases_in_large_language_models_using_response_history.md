---
title: >-
  [Paper Note] B-score: Detecting biases in large language models using response history
description: >-
  [ICML 2025][LLM (Other)][LLM bias detection] The paper proposes B-score, a metric that detects bias by comparing the difference in probability of LLM responses between single-turn and multi-turn dialogues. It discovers that LLMs can "self-debias" in multi-turn dialogues and leverages B-score to improve answer verification accuracy.
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "LLM bias detection"
  - "multi-turn dialogue"
  - "self-debiasing"
  - "confidence calibration"
  - "answer verification"
date: 2026-05-08
content_hash: e8dd771f669dcf77
---

# B-score: Detecting biases in large language models using response history

**Conference**: ICML 2025  
**arXiv**: [2505.18545](https://arxiv.org/abs/2505.18545)  
**Code**: [b-score.github.io](https://b-score.github.io/)  
**Area**: LLM/NLP  
**Keywords**: LLM bias detection, multi-turn dialogue, self-debiasing, confidence calibration, answer verification

## TL;DR

The paper proposes B-score, a metric that detects bias by comparing the difference in probability of LLM responses between single-turn and multi-turn dialogues. It discovers that LLMs can "self-debias" in multi-turn dialogues and leverages B-score to improve answer verification accuracy.

## Background & Motivation

### Key Challenge

**Background**: LLMs often exhibit strong biases. For instance, when GPT-4o is asked to generate a random number from 0-9, it chooses 7 with a 70% probability. Existing research on bias primarily focuses on biases resulting from imbalanced training data, but in reality, biases can originate from multiple sources:

**Biases from actual preferences**: The model truly possesses a certain "preference" (e.g., political orientation).

**Biases from insufficient capability**: The question is too difficult, leading the model to consistently choose incorrect answers.

**Training data bias**: Coming from imbalanced training data.

Existing methods usually detect bias by repeatedly asking the same question in a single-turn setup, but this evaluation only captures a "snapshot" of the model's response and cannot utilize historical information. The core research question is: **If LLMs are allowed to see their own prior responses, can they produce less biased answers?**

The answer is affirmative. For example, in multi-turn dialogues, GPT-4o can transition from choosing 7 with 70% probability to a near-uniform distribution of approximately 10% for each digit. This finding inspires the design of the B-score metric.

## Method

### Overall Architecture

The proposed method consists of three core components: (1) a single-turn vs. multi-turn evaluation protocol, (2) the B-score bias metric, and (3) an answer verification framework based on B-score.

### Key Designs

**Single-turn Evaluation**: The model is queried independently 30 times for the same question, with the context reset each time, preventing the model from remembering prior responses.

**Multi-turn Evaluation**: The model is queried 30 times for the exact same question sequentially within a continuous dialogue, allowing it to see its previous responses. The option order is randomly shuffled during each query to eliminate position bias.

**Definition of B-score**: For a given option $a$ in a multiple-choice question:

$$\text{B-score}(a) = P_{\text{single}}(a) - P_{\text{multi}}(a)$$

Where $P_{\text{single}}(a)$ is the empirical probability of selecting $a$ in single-turn evaluation, and $P_{\text{multi}}(a)$ is the empirical probability of selecting $a$ in multi-turn evaluation.

- **B-score > 0**: The model is biased toward $a$ in single-turn evaluation but self-corrects in multi-turn evaluation -> Bias detected.
- **B-score ≈ 0**: Similar frequency in both single-turn and multi-turn evaluations -> Likely represents a true preference or no bias.
- **B-score < 0**: The model selects $a$ more frequently in multi-turn evaluation -> Suggests an "inverse bias" toward this option.

**Four-Category Question Evaluation Framework**: Spanning 9 topics (numbers, gender, politics, mathematics, race, names, countries, sports, professions), 4 categories of questions are designed for each topic:

1. 🗣️ **Subjective**: Inquires about preferences or subjective opinions.
2. 🎲 **Random**: Demands a random selection.
3. ✅ **Easy**: Simple questions with clear correct answers.
4. ❓ **Hard**: Difficult questions requiring reasoning.

**2-Step Cascade Verification**: A primary metric (single-turn probability / multi-turn probability / confidence) is first used for initial screening, followed by B-score as a secondary check. If the B-score exceeds a threshold, the answer is rejected.

### Loss & Training

B-score is an unsupervised, post-hoc metric that does not require training. The optimal threshold combination is identified on the validation set via grid search, covering both the primary metric and B-score dimensions.

## Key Experimental Results

### Main Results

**8 LLMs Tested**: GPT-4o, GPT-4o-mini, Gemini-1.5-Pro, Gemini-1.5-Flash, Llama-3.1-70B, Llama-3.1-405B, Command R, Command R+.

**Self-debiasing Effect** (Table 2, Average B-score):

| Question Type | Average B-score |
|----------|-----------|
| Subjective | +0.27 |
| Random | **+0.41** |
| Easy | +0.06 |
| Hard | +0.15 |
| Total Average | +0.23 |

The debiasing effect is most pronounced in Random questions: the maximum single-turn selection probability drops from 0.77 to 0.29.

**Answer Verification Accuracy Improvement** (Table 3):

- Custom evaluation framework: 2-step verification using B-score yields an average improvement of **+9.3%**.
- Standard benchmarks (MMLU+HLE+CSQA): Average improvement of **+4.8%** (up to +2.9% for some models).
- Maximum improvement: Llama-3.1-405B shows a +27.3% gain on the custom framework.

**BBQ Benchmark Verification** (Table T4): Used alone, B-score achieves 89.6% verification accuracy, significantly outperforming single-turn probability (20.9%), multi-turn probability (33.9%), and confidence (77.6%). Integrating B-score leads to an overall gain of +45.7%.

### Ablation Study

**Sampling Temperature Experiment**: Even at temperature=1.5, GPT-4o still selects 7 with a 40% probability (compared to the ideal 10%). Multi-turn feedback reduces bias much more effectively than high temperature.

**Sensitivity to Sample size**: B-score remains highly stable (0.22-0.23) across $k$=10, 20, and 30, suggesting a recommended $k$ of 2 to 3 times the number of options.

**LLM Distribution Replication Capability**: GPT-4o and GPT-4o-mini successfully replicate uniform and Gaussian distributions, indicating their intrinsic capability to monitor and adjust output patterns.

### Key Findings

1. All 8 tested LLMs exhibit reduced bias in multi-turn dialogues, with the most significant effect seen in the Random category.
2. For Subjective questions, B-score ≈ 0 reflects "true preference" rather than bias (e.g., GPT-4o's political preference consistently leaning towards Biden).
3. Confidence scores remain nearly constant across different options and only reflect question difficulty, failing to detect bias.
4. B-score is ≈ 0 for Easy questions (no bias to detect) but successfully reveals the model's erroneous tendencies in Hard questions.

## Highlights & Insights

1. **Extremely Elegant and Concise Core Insight**: Bias can be detected solely via the probability difference between single-turn and multi-turn setups, requiring no labels or external calibration.
2. **Distinguishing "Preference" from "Bias"**: For Subjective questions, a B-score ≈ 0 indicates a genuine preference rather than an artifact; conversely, a B-score > 0 indicates a correctable bias.
3. **High Practical Utility**: B-score can be computed at runtime, serving as a live "bias alert" in practical applications.
4. **Challenging Existing Bias Evaluation Paradigms**: Traditional single-turn evaluation might overestimate the systematic bias of LLMs.
5. **Uncovering LLMs' Self-Debiasing Capability**: Models can intrinsically track output distributions and actively adjust them without external intervention.

## Limitations & Future Work

1. **Computational Overhead**: Computing B-score requires an extra 30 single-turn and 30 multi-turn queries, posing high deployment costs for practical applications.
2. **Limited Evaluation Scenarios**: Evaluated only in QA scenarios, without verification on broader benchmarks such as hallucination detection or open-ended generation.
3. **Threshold Dependence**: The optimal thresholds are obtained via grid search, which may require readjustment for different scenarios.
4. **Unexplored Training-Stage Debiasing**: Bias is only detected during inference without leveraging B-score insights to improve training.
5. Only 8 LLMs were evaluated, and mostly on older model versions.

## Related Work & Insights

- **Difference from Self-Consistency (Wang et al., 2023)**: Self-Consistency computes confidence based on option distribution, assigning identical scores to all options, whereas B-score provides distinct scores for different options.
- **Difference from MultiAgent Debate**: The prompt is kept identical in multi-turn dialogues, introducing no new context or personas.
- **Complementarity with BBQ (Parrish et al., 2022)**: The same conclusions are validated on the BBQ benchmark.
- **Insights**: The core idea of B-score can be applied to detect LLM hallucinations (which might also count as a form of "bias"); multi-turn self-reflection could serve as a cost-effective reasoning enhancement method.

## Rating

- Novelty: ⭐⭐⭐⭐ (4/5) — Innovative angle with an impressively elegant metric design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5/5) — Extensive ablations across 8 LLMs, 9 topics, 4 question classes, and multiple benchmarks.
- Writing Quality: ⭐⭐⭐⭐ (4/5) — Clear writing, aided by abundant visualizations.
- Value: ⭐⭐⭐⭐ (4/5) — Highly practical, though computational overhead remains an obstacle for actual deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Binary Hypothesis Testing for Softmax Models and Leverage Score Models](binary_hypothesis_testing_for_softmax_models_and_leverage_score_models.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](../../ACL2025/llm_nlp/detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Revisiting Uncertainty Quantification Evaluation in Language Models: Spurious Interactions with Response Length Bias Results](../../ACL2025/llm_nlp/revisiting_uncertainty_quantification_evaluation_in_language_models_spurious_int.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [\[ACL 2025\] Assessing the Vulnerability of LLMs to Cognitive Biases in Scientific Research](../../ACL2025/llm_nlp/assessing_the_vulnerability_of_llms_to_cognitive_biases_in_scientific_research.md)

</div>

<!-- RELATED:END -->
