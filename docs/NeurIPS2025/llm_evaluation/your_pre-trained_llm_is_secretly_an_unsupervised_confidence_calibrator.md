---
title: >-
  [Paper Note] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator
description: >-
  [NeurIPS 2025][LLM Evaluation][confidence calibration] This paper identifies that post-training (SFT/RLHF/DPO) degrades the confidence calibration of pre-trained language models, and proposes DACA…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "confidence calibration"
  - "temperature scaling"
  - "pre-trained LM"
  - "post-trained LM"
  - "unsupervised calibration"
  - "DACA"
date: 2026-05-08
content_hash: ec64a0e29c495bac
---

# Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator

**Conference**: NeurIPS 2025
**arXiv**: [2505.16690](https://arxiv.org/abs/2505.16690)  
**Code**: [GitHub](https://github.com/ml-stat-Sustech/Disagreement-Aware-Calibration)  
**Area**: LLM Evaluation
**Keywords**: confidence calibration, temperature scaling, pre-trained LM, post-trained LM, unsupervised calibration, DACA

## TL;DR

This paper identifies that post-training (SFT/RLHF/DPO) degrades the confidence calibration of pre-trained language models, and proposes DACA, a method that exploits the well-calibrated nature of pre-trained models by aligning confidence distributions exclusively on prediction-consistent samples, achieving label-free calibration of post-trained models with up to 15.08% ECE improvement.

## Background & Motivation

### Problem Definition

The standard LLM training paradigm follows a "pre-training → post-training" pipeline. Pre-trained language models (PLMs) generally exhibit well-calibrated confidence (i.e., the model's output confidence faithfully reflects its accuracy), but after post-training procedures such as SFT, RLHF, and DPO, models become **overconfident**—assigning high confidence scores to both correct and incorrect outputs.

### Limitations of Prior Work

1. **Temperature Scaling** is the most practical post-hoc calibration method, but **requires labeled data**.
2. Obtaining labels is extremely costly and time-consuming in domains such as mathematical reasoning and medical diagnosis.
3. Large quantities of **unlabeled data remain unused** in real-world deployments.
4. Prompt-based calibration methods (verbalization) yield limited effectiveness.

### Core Insight

Pre-trained models are inherently well-calibrated—can the confidence of PLMs be leveraged to calibrate overconfident post-trained language models (PoLMs)? A key finding emerges: **direct alignment leads to underconfidence**, rooted in the interference of prediction-disagreement examples.

## Method

### Overall Architecture

DACA (Disagreement-Aware Confidence Alignment) operates via the following mechanism:

1. Detect whether the PLM and PoLM agree in their predictions.
2. Align the confidence distributions of the two models exclusively on agreement samples.
3. Solve for the optimal temperature parameter by minimizing KL divergence.

### Why Naive Alignment Fails

**Naive confidence alignment** minimizes the KL divergence between PLM and PoLM over all unlabeled data:

$$\tau^* = \arg\min_{\tau > 0} \mathbb{E}_{\boldsymbol{x} \in \mathcal{D}} \left[ \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{\sigma_i(g(\boldsymbol{x})/\tau)} \right]$$

**Problem**: When the two models disagree, the PLM's confidence reflects **its own prediction accuracy** rather than the accuracy of the PoLM's prediction. Since post-training typically improves the PoLM's accuracy, the PLM's confidence **underestimates** the PoLM's true correctness rate, pushing the temperature parameter $\tau$ toward excessively large values.

### Theoretical Analysis

**Proposition 3.2**: Even when the PLM is perfectly calibrated ($ECE_f = 0$), the ECE of the perfectly aligned PoLM is:

$$ECE_g = \pi \cdot \left| \mathbb{E}_{\boldsymbol{x}} \left[ \mathbf{1}\{\arg\max_i f_i(\boldsymbol{x}) = \tilde{y}\} - \mathbf{1}\{\arg\max_i g_i(\boldsymbol{x}) = \tilde{y}\} \right] \right|$$

where $\pi$ denotes the proportion of disagreement samples. **Conclusion**: Due to prediction disagreement, zero ECE cannot be achieved even under ideal alignment.

**Proposition 3.3**: For disagreement samples where the PoLM predicts class $c$ but the PLM assigns probability $< 1/k$ to $c$, the optimal temperature is $\tau^* = \infty$. **Implication**: The gradient of KL divergence with respect to $\tau$ on disagreement samples is always positive, continuously pushing the temperature upward.

### Key Design: DACA Loss Function

$$\mathcal{L}(\tau; \boldsymbol{x}) = \mathbf{1}\{\hat{y} = \hat{y}'\} \cdot \left[ \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{\sigma_i(g(\boldsymbol{x})/\tau)} \right]$$

where $\hat{y} = \arg\max_i f_i(\boldsymbol{x})$ (PLM prediction) and $\hat{y}' = \arg\max_i g_i(\boldsymbol{x})$ (PoLM prediction).

**Core Operation**: The indicator function $\mathbf{1}\{\hat{y} = \hat{y}'\}$ filters out disagreement samples, computing KL divergence exclusively on agreement samples.

### General Extension

The method generalizes to arbitrary post-hoc calibration approaches (vector scaling, matrix scaling):

$$\boldsymbol{\theta}^* = \arg\min_{\tau > 0} \mathbb{E}_{\boldsymbol{x} \in \mathcal{D}} \left[ \mathbf{1}\{\hat{y} = \hat{y}'\} \cdot \sum_{i=1}^{k} p_i(\boldsymbol{x}) \log \frac{p_i(\boldsymbol{x})}{q_i(\boldsymbol{x}; \boldsymbol{\theta})} \right]$$

### Open-Domain QA Extension

For open-ended question answering, the P(True) method is adopted to obtain confidence scores: the model is prompted to judge whether its own generated answer is correct, and $p(\text{Yes}|x, f)$ is used as the confidence score.

## Key Experimental Results

### Main Results: Average Calibration Performance on MMLU (57 Subjects)

| Model | Method | ECE(%) ↓ | MCE(%) ↓ | AECE(%) ↓ | Brier ↓ |
|------|------|----------|----------|-----------|---------|
| Qwen3-8B | Vanilla | 16.38 | 38.19 | 24.99 | 0.179 |
| Qwen3-8B | CAPE | 11.52 | 31.74 | 17.61 | 0.157 |
| Qwen3-8B | **DACA** | **8.39** | **23.70** | **12.60** | **0.144** |
| Qwen3-8B | TS (labeled) | 8.66 | 28.11 | 14.55 | 0.146 |
| Gemma-3-12B-IT | Vanilla | 23.68 | 48.51 | 35.89 | 0.235 |
| Gemma-3-12B-IT | **DACA** | **8.60** | **27.02** | **13.55** | **0.154** |
| Gemma-3-12B-IT | TS (labeled) | 9.75 | 29.80 | 15.60 | 0.159 |
| Yi-1.5-34B-Chat | Vanilla | 16.20 | 33.82 | 20.35 | 0.199 |
| Yi-1.5-34B-Chat | **DACA** | **9.47** | **19.90** | **11.70** | **0.174** |
| Llama-3-70B-IT | Vanilla | 12.87 | 36.87 | 23.84 | 0.143 |
| Llama-3-70B-IT | **DACA** | **7.84** | **24.28** | **13.16** | **0.120** |

**Highlight**: Under the label-free setting, DACA **matches or surpasses** labeled temperature scaling (e.g., Gemma-3-12B: 8.60 vs. 9.75).

### API Model Calibration (GPT-4o + Different PLMs)

| Calibration PLM | PLM ECE(%) | GPT-4o ECE(%) ↓ |
|-----------|----------------|-----------------|
| None (Vanilla) | — | 21.23 |
| Llama-3-8B | 9.45 | 7.98 |
| Qwen2.5-7B | 6.99 | 7.82 |
| **Gemma-3-12B** | **4.42** | **6.99** |

**Finding**: Better-calibrated PLMs yield more effective DACA alignment. Smaller PLMs can be used to calibrate larger or closed-source models.

### Ablation Study: Different Post-Training Strategies

| Post-Training | Vanilla ECE(%) | DACA ECE(%) |
|-----------|---------------|-------------|
| SFT | 14.85 | 4.57 |
| SFT + DPO | 25.12 | 5.42 |
| SFT + DPO + RLVR | 25.19 | 5.99 |

**Finding**: More aggressive post-training (with DPO/RLVR) leads to more severe overconfidence, yet DACA effectively calibrates all variants.

### Open-Domain QA and Selective Classification

- On TruthfulQA, Qwen2.5-32B-Instruct ECE decreases from 30.96% to 5.24%.
- In selective classification, accuracy of high-confidence predictions improves significantly across all thresholds (0.5–0.95) after calibration.
- The advantage is more pronounced at higher thresholds, where overconfidence is most detrimental.

### Key Findings

1. Verbalization-based methods (Elicitation series) perform substantially worse than logit-based methods.
2. Larger models exhibit lower Vanilla ECE, consistent with prior findings.
3. DACA is robust across model scales, architectures, and post-training strategies.
4. Temperature values continue to grow to extreme magnitudes when trained on disagreement samples, empirically validating the theoretical analysis.

## Highlights & Insights

1. **Profound insight**: The identification of prediction disagreement as the root cause of naive alignment failure is a simple yet highly explanatory observation.
2. **Theory and experiments reinforce each other**: Propositions 3.2 and 3.3 clearly explain why direct alignment leads to underconfidence, and the experiments perfectly validate the theory.
3. **Highly practical**:
    - Requires no labeled data.
    - Applicable across architectures (small PLM calibrating large PoLM).
    - Applicable to closed-source API models (GPT-4o, DeepSeek-V3).
    - Minimal computational overhead (single inference pass + temperature optimization).
4. **Minimal yet effective method**: The core contribution is essentially an indicator function filtering out disagreement samples, yet the empirical gains are remarkable.
5. **DACA without labels outperforms labeled TS**, demonstrating that PLM calibration information is remarkably informative.

## Limitations & Future Work

1. **Additional inference cost**: An extra PLM inference pass is required to determine prediction agreement.
2. **Disagreement samples are discarded**: Information from filtered disagreement samples is entirely unused.
3. **Logit accessibility required**: The method is not applicable to fully black-box models that do not return logits.
4. **Calibration target is MCQA**: Although extended to open-ended QA, the main experiments are limited to multiple-choice scenarios.
5. **Future directions**:
    - Design methods that leverage disagreement samples (e.g., weighting rather than complete filtering).
    - Explore applicability to generative tasks.
    - Integrate calibration into the RLHF training process as an in-training objective.

## Related Work & Insights

- **Relation to Temperature Scaling (Guo et al. 2017)**: TS is the foundational method but requires labels; DACA substitutes PLM confidence for labels, representing the first label-free post-hoc calibration approach.
- **Comparison with CAPE (Jiang et al. 2023)**: CAPE calibrates by permuting option order as a form of prompt engineering; DACA is a principled method that achieves superior performance.
- **Comparison with Shen et al. 2024 (Thermometer)**: Thermometer trains an auxiliary model to predict temperature, requiring labels and training; DACA requires neither.
- **Insight**: The calibration properties of PLMs constitute an underutilized asset that can be more effectively exploited within the post-training pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First label-free post-hoc calibration method for LLMs; the core insight is deep and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage of multiple model families, scales, post-training strategies, datasets, and both open- and closed-source models.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear, experimental presentation is well-structured, and the transition from motivation to method is natural.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a core pain point in LLM deployment; the method is simple, practical, and immediately deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[ICML 2026\] CapBencher: Give Your LLM Benchmark a Built-in Alarm for Test-Set Overfitting](../../ICML2026/llm_evaluation/capbencher_give_your_llm_benchmark_a_built-in_alarm_for_test-set_overfitting.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](../../ACL2026/llm_evaluation/how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](../../AAAI2026/llm_evaluation/dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
