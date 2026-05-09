---
title: >-
  [Paper Note] Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives
description: >-
  [AAAI 2026][LLM/NLP][Syllogistic Reasoning] This work systematically evaluates 14 LLMs on 160 syllogisms using a dual-dimensional ground truth framework (syntactic validity + NLU believability), revealing that top models approach near-perfect performance on formal logic (99.6%) while performing at chance level on natural language believability (~52%)—the inverse of human reasoning patterns. 12 out of 14 models exhibit significant belief bias, and few-shot prompting degrades formal reasoning performance.
tags:
  - AAAI 2026
  - LLM/NLP
  - Syllogistic Reasoning
  - Belief Bias
  - Formal Logic
  - Natural Language Understanding
  - LLM Evaluation
  - Dual-Dimensional Evaluation
date: 2026-05-08
content_hash: 01127058ca97e7d7
---

# Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives

**Conference**: AAAI 2026
**arXiv**: [2512.12620](https://arxiv.org/abs/2512.12620)
**Authors**: Aheli Poddar, Saptarshi Sahoo, Sujata Ghosh
**Code**: [GitHub](https://github.com/XAheli/Logic-in-LLMs)
**Area**: NLP Understanding
**Keywords**: Syllogistic Reasoning, Belief Bias, Formal Logic, Natural Language Understanding, LLM Evaluation, Dual-Dimensional Evaluation

## TL;DR

This work systematically evaluates 14 LLMs on 160 syllogisms using a dual-dimensional ground truth framework (syntactic validity + NLU believability), revealing that top models approach near-perfect performance on formal logic (99.6%) while performing at chance level on natural language believability (~52%)—the inverse of human reasoning patterns. 12 out of 14 models exhibit significant belief bias, and few-shot prompting degrades formal reasoning performance.

## Background & Motivation

### State of the Field
Syllogistic reasoning is a classical logical form originating with Aristotle, consisting of two premises and a conclusion, where the task is to determine whether the conclusion follows logically from the premises. Humans exhibit the well-known *belief-bias effect* in syllogistic reasoning: conclusions consistent with everyday beliefs are more readily accepted (even when logically invalid), while inconsistent ones are more readily rejected (even when logically valid). As LLM reasoning capabilities advance rapidly, a critical question emerges: do LLMs reason more like formal logic engines or human-like reasoners?

### Limitations of Prior Work
- Prior studies largely evaluate LLM reasoning along a single dimension—either logical correctness or semantic understanding—lacking frameworks that simultaneously measure both.
- Systematic quantification of belief bias in LLMs is insufficient, and existing methodologies are inconsistent with classical paradigms from cognitive psychology.
- The effects of confounding factors such as premise order and nonsense terms on LLM reasoning have not been thoroughly investigated.
- A small number of related works (e.g., Eisape et al.) employ substantially different methodologies and do not cover as broad a range of models and prompting strategies.

### Root Cause
By constructing a dual-dimensional evaluation framework (syntactic validity × NLU believability), the paper simultaneously measures LLMs' formal logical reasoning ability and natural language believability judgment, quantifies belief bias, and addresses the fundamental question: are LLMs evolving into formal reasoning engines or human-like reasoners?

## Method

### Overall Architecture
This is a systematic evaluation study comprising four core components: (1) construction of a 160-syllogism dataset (40 base × 4 variants); (2) dual-dimensional annotation (syntactic validity + NLU believability); (3) a full-factorial evaluation of 14 LLMs across 4 prompting strategies × 3 temperatures (168 configurations, 26,880 total evaluations); and (4) statistical analyses of belief bias, consistency, and cross-dimensional correlations.

### Dataset Construction
Forty base syllogisms are constructed from cognitive science and psychology literature, each generating four variants:
- **Normal variant (N)**: Uses meaningful natural language predicates (e.g., "footballers," "swimmers").
- **Nonsense variant (X)**: Predicates are replaced with abstract terms (e.g., "blargs," "zimons," "glorps") to test pure logical reasoning ability.
- **Order-swapped variant (O)**: The two premises are presented in reversed order to test order sensitivity.
- **Combined variant (OX)**: Both nonsense substitution and premise swapping are applied simultaneously.

Data distribution: 76 valid (47.5%) / 84 invalid (52.5%); 82 congruent (51.2%) / 78 incongruent (48.8%), approximately balanced.

### Dual-Dimensional Ground Truth Framework
Each syllogism carries two independent labels:
- **Syntactic validity label**: Whether the conclusion follows logically from the premises (independent of content truth).
- **NLU believability label**: Whether the conclusion is intuitively plausible under real-world knowledge (independent of logical structure).

This yields four instance types, for example:
- valid-believable (congruent): "All birds have feathers; robins are birds; therefore robins have feathers."
- invalid-believable (incongruent): "All flowers need water; roses need water; therefore roses are flowers."—the conclusion is factually correct but logically invalid (affirming the consequent fallacy).

Belief bias is quantified as: $\Delta_{\text{bias}} = \text{Acc}_{\text{congruent}} - \text{Acc}_{\text{incongruent}}$

### Temperature-Adaptive Reasoning Algorithm
- $\tau=0$: Single deterministic decoding, confidence = 1.0.
- $\tau>0$: Self-consistency majority voting with up to $K_{\max}=10$ samples; early stopping if the first $\eta=5$ samples agree; ties default conservatively to "invalid."
- **Design Motivation**: Eliminates randomness-induced evaluation noise while maintaining efficiency (early stopping reduces API calls when consensus is reached quickly).

### Consistency Metrics
$C_{\text{all}}$ measures the proportion of syllogisms on which a model is fully consistent across all four variants; $C_{N \leftrightarrow X}$ tests robustness to semantic content (normal vs. nonsense); $C_{O \leftrightarrow OX}$ tests robustness to premise order.

## Key Experimental Results

### Table 1: Overall Model Performance (14 models, aggregated across 4 strategies × 3 temperatures)

| Model | Syntactic Acc% | Prec% | Rec% | $C_{\text{all}}$% | NLU Acc% | Syntactic–NLU Gap |
|-------|---------------|-------|------|----------|---------|------------------|
| Gemini 2.5 Flash | **99.6** | 100.0 | 99.1 | 99.0 | 51.7 | +47.9 |
| GPT-OSS-20B | 99.5 | 100.0 | 99.0 | 96.5 | 51.6 | +47.9 |
| Gemini 2.5 Pro | 99.3 | 100.0 | 98.6 | 98.3 | 51.9 | +47.4 |
| GLM-4.6 | 99.0 | 100.0 | 97.8 | 95.8 | 52.2 | +46.8 |
| Kimi-K2-Instruct | 96.0 | 97.0 | 94.5 | 88.3 | 54.9 | +41.1 |
| DeepSeek V3.1 | 95.8 | 99.6 | 91.6 | 89.0 | 55.1 | +40.7 |
| Gemini 2.5 Flash Lite | 88.9 | 89.8 | 86.5 | 71.9 | 57.2 | +31.7 |
| Qwen3-Next 80B Instruct | 79.4 | 73.3 | 88.9 | 69.2 | 46.8 | +32.6 |
| Qwen3-Next 80B Thinking | 72.7 | 99.2 | 42.8 | 76.7 | 64.5 | +8.2 |
| Llama 3.3 70B Instruct | 69.8 | 82.1 | 46.7 | 66.2 | 66.3 | +3.5 |
| Gemma 3 27B IT | 68.4 | 61.0 | 93.1 | 69.0 | 43.6 | +24.8 |
| Llama 3.1 8B Instruct | 64.3 | 66.3 | 50.7 | 51.9 | 56.8 | +7.5 |
| Llama 3.2 3B Instruct | 59.2 | 88.1 | 16.2 | 75.0 | 73.7 | **−14.5** |
| Llama 3.2 1B Instruct | 51.9 | 49.2 | 41.9 | 57.9 | 60.4 | −8.5 |
| **Average** | **81.7** | — | — | — | **56.2** | **+25.5** |

### Table 2: Belief Bias Analysis (sorted by bias magnitude)

| Model | Congruent Acc% | Incongruent Acc% | $\Delta_{\text{bias}}$ |
|-------|---------------|-----------------|----------------------|
| Llama 3.2 3B Instruct | 82.0 | 35.2 | **+46.9** |
| Llama 3.3 70B Instruct | 85.3 | 53.6 | +31.6 |
| Qwen3-Next 80B Thinking | 86.3 | 58.3 | +28.0 |
| Llama 3.2 1B Instruct | 62.0 | 41.2 | +20.8 |
| Llama 3.1 8B Instruct | 70.6 | 57.7 | +12.9 |
| Gemini 2.5 Flash Lite | 95.0 | 82.5 | +12.5 |
| DeepSeek V3.1 | 99.7 | 91.8 | +7.9 |
| Kimi-K2-Instruct | 99.6 | 92.1 | +7.5 |
| GLM-4.6 | 99.4 | 97.5 | +1.9 |
| Gemini 2.5 Pro | 100.0 | 98.6 | +1.4 |
| Gemini 2.5 Flash | 100.0 | 99.2 | **+0.9** |
| GPT-OSS-20B | 99.2 | 98.4 | +0.8 |
| Qwen3-Next 80B Instruct | 75.5 | 83.4 | −7.9 |
| Gemma 3 27B IT | 61.7 | 75.4 | −13.7 |

12/14 models exhibit positive bias, with a mean of $\Delta_{\text{bias}} = +10.81$ pp ($t_{13}=2.47, p=0.028, d=0.66$). Stronger reasoning models show smaller bias ($\rho=-0.565, p=0.035$).

### Prompting Strategy Effects
- ZS **82.7%** > ZS-CoT 82.6% > OS 82.2% > FS **79.1%**
- FS is significantly lower than ZS: $\Delta=-3.57$ pp, paired $t$-test $p=0.0165$, Holm-corrected $p_{\text{adj}}=0.0495$.
- McNemar instance-level test: ZS resolves 786 instances where FS fails, while FS resolves only 546 instances where ZS fails ($\chi^2=42.88, p<0.0001$).
- Temperature effects are negligible: Friedman $\chi^2=3.77, p=0.152$.

## Highlights & Insights

- **Core finding is profound and counterintuitive**: Top LLMs achieve near-perfect formal logic performance (99.6%) yet perform at chance on NLU believability (~52%), suggesting LLMs are evolving into "formal logic engines" rather than "human-like reasoners"—the inverse of the human pattern where belief bias dominates over logical analysis.
- **Dual-dimensional framework is elegantly designed**: Independent annotation of syntactic validity and NLU believability enables quantification of belief bias; the framework is transferable to other logical reasoning evaluations.
- **Few-shot prompting degrades formal reasoning**: The introduction of examples with semantic content may interfere with the model's pure logical processing, implying that formal reasoning and semantic processing compete internally within LLMs.
- **Statistical analysis is exceptionally rigorous**: 26,880 evaluations coupled with paired $t$-tests, Friedman tests, McNemar tests, and Holm–Bonferroni corrections provide multi-level statistical validation.
- **Architecture matters more than parameter count**: Llama 3.2 3B exhibits +46.9 pp bias vs. Gemini 2.5 Flash at only +0.9 pp, indicating that training methodology far outweighs model scale.
- **LMArena ranking strongly predicts reasoning ability**: $\rho=-0.825, p=0.001$, showing that instruction-following quality is highly correlated with formal reasoning performance.

## Limitations & Future Work

- Only 160 syllogisms are used; the dataset scale is limited and cannot cover all logical structures.
- Syllogisms represent the most basic form of logic; more complex forms such as conditional reasoning, modal logic, and nested quantifiers are not addressed.
- NLU believability annotation is inherently subjective and may differ across cultural contexts.
- The causal relationship between logical training and bias reduction remains unresolved, as the effects of fine-tuning and RLHF on belief bias are not explored.
- Only English is tested; multilingual logical reasoning may exhibit different patterns.
- The prompting space is limited to four strategies; finer-grained combinations are not explored.
- Adversarial robustness testing and mechanistic explanations for the origin of bias are absent.

## Related Work & Insights

- **vs. Evans et al. (1983, classical psychology)**: The seminal work on human belief bias. This paper finds that LLMs exhibit bias in the same direction as humans but with different magnitudes, and that top models' formal logical capabilities far exceed those of humans—the inverse of human reasoning preferences.
- **vs. Eisape et al. (2024)**: Also studies belief bias in LLMs but employs a different methodology. This paper's dual-dimensional annotation and full-factorial design provide broader coverage.
- **vs. Kim et al. (2025)**: Offers mechanistic explanations for syllogistic reasoning in LLMs. This paper complements that work from a macro-evaluation perspective, focusing on systematic cross-model patterns.
- **vs. Zong et al. (2024)**: Conducts a detailed investigation of LLMs on categorical syllogisms but does not introduce a dual-dimensional framework or belief bias quantification.
- **vs. NLI benchmarks (SNLI/MNLI)**: Evaluate natural language inference but do not address formal logical validity. The dual-label design in this paper fills that gap.
- **vs. LogiQA/FOLIO**: Focus on accuracy in complex logical reasoning; this paper uniquely highlights the dissociation between formal and natural language reasoning in LLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The dual-dimensional evaluation framework and the central finding that "LLMs are logic engines rather than human-like reasoners" are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The full-factorial design across 14 models × 4 strategies × 3 temperatures is rigorous with thorough statistical analysis, though the dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically structured, findings are articulated with precision and clarity, and statistical reporting is standardized.
- Value: ⭐⭐⭐⭐ — Provides deep insight into the nature of LLM reasoning, though the scope is limited to syllogisms and generalizability remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment](temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[AAAI 2026\] Collaborative LLM Numerical Reasoning with Local Data Protection](collaborative_llm_numerical_reasoning_with_local_data_protection.md)

</div>

<!-- RELATED:END -->
