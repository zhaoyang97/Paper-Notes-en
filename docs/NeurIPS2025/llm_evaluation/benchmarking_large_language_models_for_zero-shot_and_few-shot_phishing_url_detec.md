---
title: >-
  [Paper Note] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection
description: >-
  [NeurIPS 2025][LLM Evaluation][phishing URL detection] This paper systematically evaluates three commercial LLMs — GPT-4o, Claude-3.7, and Grok-3-Beta — on phishing URL detection under a unified zero-shot and few-shot pr…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "phishing URL detection"
  - "LLM"
  - "zero-shot learning"
  - "few-shot learning"
  - "prompt engineering"
  - "cybersecurity benchmark"
date: 2026-05-08
content_hash: 9d65f2d37348418f
---

# Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection

**Conference**: NeurIPS 2025
**arXiv**: [2602.02641](https://arxiv.org/abs/2602.02641)  
**Code**: Available  
**Area**: Cybersecurity / LLM Evaluation
**Keywords**: phishing URL detection, LLM, zero-shot learning, few-shot learning, prompt engineering, cybersecurity benchmark

## TL;DR

This paper systematically evaluates three commercial LLMs — GPT-4o, Claude-3.7, and Grok-3-Beta — on phishing URL detection under a unified zero-shot and few-shot prompt framework. Results show that few-shot prompting consistently improves performance across all models, with Grok-3-Beta achieving the best F1 (0.9399) on the balanced dataset, while different models exhibit distinct precision–recall trade-off behaviors.

## Background & Motivation

**Escalating Threat**: Phishing attacks have surged by over 4,000% since 2022, with generative AI being widely exploited by cybercriminals to craft context-aware phishing sites and URLs, rendering traditional detection methods increasingly inadequate.

**Limitations of Prior Work**: Blacklist-based approaches and hand-crafted feature ML classifiers (decision trees, SVMs, ensemble models) generalize poorly to novel phishing URLs, relying heavily on known patterns and labeled data.

**Insufficient Progress in Deep Learning**: While CNN/RNN-based deep models can automatically learn URL sequence features, they still require large-scale supervised training and lack zero-shot inference capability.

**Unsystematic Evaluation of LLM Potential**: Although BERT fine-tuning (DomURLs_BERT) and LoRA adaptation (PhishURLDetect) have demonstrated the viability of language models in security contexts, no systematic comparison of commercial LLMs under prompt-only inference has been conducted.

**Flaws in Existing Evaluations**: Prior work (Nasution et al.) evaluated only 21 open-source LLMs with English-only prompts, excluding commercial models; inconsistent evaluation conditions further undermine the comparability of findings.

**Lag in Labeled Data**: Phishing tactics evolve far faster than labeled data can be produced, making zero-shot/few-shot learning a pragmatic approach for addressing rapidly evolving threats.

## Method

### Evaluation Framework Design

The paper constructs a unified prompt-based classification framework comprising three components:

1. **Task Instruction $\mathcal{I}$**: Assigns the model the role of a cybersecurity expert and requires binary output of 0 (phishing) or 1 (legitimate).
2. **Query $\mathcal{Q}(u)$**: Embeds the target URL into a standardized query template.
3. **Example Set $\mathcal{E}$** (few-shot only): Provides labeled URL examples; the few-shot prompt takes the form $\mathcal{P}_{\text{FS}}(u) = \mathcal{I} \oplus \mathcal{E} \oplus \mathcal{Q}(u)$.

### Model and Data Configuration

- **Models**: GPT-4o (OpenAI), Claude-3.7-sonnet-20250219 (Anthropic), and Grok-3-Beta (xAI), all accessed via official APIs with temperature=0 and max tokens=10.
- **Balanced Dataset**: 10,000 URLs randomly sampled from the PhiUSIIL dataset (5,000 phishing / 5,000 legitimate); zero-shot evaluation uses the full set, while few-shot evaluation reserves 6 examples (3+3) and evaluates on the remaining 9,994 URLs.
- **Imbalanced Dataset**: A 1,000-URL test set is constructed with phishing ratios of 1% and 10%, validated with dual random seeds (S123/S456) to assess stability.
- **Few-Shot Variants**: Effects of $|\mathcal{E}| \in \{1, 3, 9\}$ are tested under the 10% imbalanced condition.

### Evaluation Metrics

Six metrics are employed for comprehensive evaluation: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, AUROC, and AUPRC, all computed as macro-averages across classes.

## Key Experimental Results

### Table 1: Performance on the Balanced Dataset (10,000 URLs)

| Model | Setting | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|-------|---------|----------|-----------|--------|------|-------|-------|
| GPT-4o | Zero-shot | 0.8752 | 0.8421 | 0.9232 | 0.8808 | 0.8752 | 0.9018 |
| GPT-4o | Few-shot | 0.9050 | 0.8880 | 0.9270 | 0.9071 | 0.9050 | 0.9258 |
| Claude-3.7 | Zero-shot | 0.8759 | 0.8778 | 0.8734 | 0.8756 | 0.8759 | 0.9072 |
| Claude-3.7 | Few-shot | 0.9250 | 0.9027 | **0.9526** | 0.9270 | 0.9250 | 0.9395 |
| Grok-3-Beta | Zero-shot | 0.8914 | 0.8361 | 0.9735 | 0.8996 | 0.8914 | 0.9114 |
| Grok-3-Beta | Few-shot | **0.9405** | **0.9492** | 0.9307 | **0.9399** | **0.9405** | **0.9573** |

### Table 2: Few-Shot F1 on Imbalanced Dataset (10% Phishing, S123)

| Model | Zero-shot | $\mathcal{E}$=1 | $\mathcal{E}$=3 | $\mathcal{E}$=9 |
|-------|-----------|-----------------|-----------------|-----------------|
| GPT-4o | 0.785 | 0.709 | 0.801 | 0.861 |
| Claude-3.7 | 0.761 | 0.857 | 0.842 | 0.876 |
| Grok-3-Beta | **0.854** | **0.906** | 0.821 | 0.831 |

### Key Findings

- **Consistent Few-Shot Gains**: Six examples improve F1 by 3–6 percentage points across all models on the balanced set, with corresponding improvements in AUROC and AUPRC.
- **Divergent Precision–Recall Trade-offs**: Grok-3-Beta exhibits a decrease in Recall (0.9735→0.9307) but a substantial gain in Precision (0.8361→0.9492) from zero-shot to few-shot, suggesting that few-shot examples induce a more conservative decision threshold.
- **Differentiated Behavior Under Imbalance**: Grok-3-Beta achieves F1=0.906 with only one example but degrades with more; GPT-4o improves monotonically with increasing examples; Claude-3.7 shows a non-monotonic pattern of initial decline followed by recovery.
- **Cross-Seed Stability**: F1 differences across dual random seeds remain below 0.05, confirming the robustness of the evaluation conclusions.

## Highlights & Insights

- **Unified Evaluation Framework**: This is the first study to compare the phishing detection capabilities of three major commercial LLMs under standardized conditions, ensuring fair comparability across models.
- **Fine-Grained Behavioral Analysis**: The work reveals markedly different few-shot learning behaviors across LLMs (monotonic / non-monotonic / overfitting), providing direct guidance for model selection in practical deployments.
- **Low Annotation Cost**: Significant performance gains are achievable with only 1–6 labeled examples, demonstrating the practical value of LLMs in security scenarios where labeled data is scarce.
- **Reproducibility**: Open-sourced code and standardized datasets ensure result reproducibility.

## Limitations & Future Work

- **Limited Model Coverage**: Only three commercial models are evaluated; open-source LLMs (e.g., Llama-3, Mistral) are excluded, precluding assessment of cost-effectiveness of open-source alternatives.
- **Single Dataset**: Evaluation is confined to the PhiUSIIL dataset without cross-dataset validation; URL distribution differences across data sources may affect the generalizability of conclusions.
- **No Adversarial Analysis**: The robustness of LLMs against adversarial phishing URLs (e.g., Unicode obfuscation, homoglyph attacks) is not examined.
- **Timeliness**: Phishing URLs evolve rapidly, limiting the longevity of the current benchmark and necessitating continuous update mechanisms.
- **Missing Cost Analysis**: API call costs and latency across models are not compared, despite being critical factors in real-world deployment.

## Related Work & Insights

| Dimension | Ours | Nasution et al. (2025) | PhishURLDetect (Ali & Subba, 2025) |
|-----------|------|------------------------|-------------------------------------|
| Model Type | Commercial LLMs (GPT-4o/Claude/Grok) | 21 open-source LLMs | Single fine-tuned LLM (LoRA) |
| Evaluation Paradigm | Prompt-only inference | Prompt engineering | Parameter-efficient fine-tuning |
| Data Conditions | Balanced + imbalanced + multi-seed | English prompts only | Single dataset |
| Few-Shot Analysis | Systematic 1/3/6/9-shot comparison | Not addressed | N/A (fine-tuning paradigm) |
| Behavioral Analysis Depth | Precision–recall trade-off + non-monotonic patterns | Basic metric comparison | Limited analysis |

| Dimension | Ours | DomURLs_BERT (Mahdaouy et al., 2024) |
|-----------|------|--------------------------------------|
| Training Requirement | No training required | Requires fine-tuning |
| Best F1 | 0.9399 (Grok few-shot) | Higher (fully supervised) |
| Privacy / Deployment | API calls; data remains local | Requires local training data |

## Rating

- **Novelty**: ⭐⭐⭐ The systematic evaluation framework is valuable, but there is no methodological innovation; the core contribution lies in experimental design and behavioral analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers balanced/imbalanced settings, zero-shot/few-shot conditions, multi-seed validation, and a comprehensive metric suite.
- **Writing Quality**: ⭐⭐⭐⭐ Experimental design is clearly described, prompt construction is transparent, and results are presented in a well-organized manner.
- **Value**: ⭐⭐⭐ Provides useful reference for LLM applications in the security domain; the behavioral difference analysis offers meaningful practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Few-Shot Pattern Detection via Template Matching and Regression](../../ICCV2025/llm_evaluation/few-shot_pattern_detection_via_template_matching_and_regression.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](../../ICCV2025/llm_evaluation/rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
