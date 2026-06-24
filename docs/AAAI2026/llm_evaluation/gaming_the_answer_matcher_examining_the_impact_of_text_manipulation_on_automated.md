---
title: >-
  [Paper Note] Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment
description: >-
  [AAAI 2026][LLM Evaluation][Answer Matching] This paper systematically evaluates three text manipulation strategies—verbosity, strategic multi-answer embedding, and correct-answer-first with contradictory suffix—against LLM-based answer-matching judges. The results show that these manipulations **do not improve scores and often reduce them**. Binary scoring proves more robust than continuous scoring, demonstrating that answer matching is resistant to low-cost text manipulatio…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "Answer Matching"
  - "Adversarial Attack"
  - "Robustness"
  - "Automated Judgment"
date: 2026-05-08
content_hash: 33725cfc1f6ac036
---

# Gaming the Answer Matcher: Examining the Impact of Text Manipulation on Automated Judgment

**Conference**: AAAI 2026
**arXiv**: [2601.08849](https://arxiv.org/abs/2601.08849)  
**Code**: [GitHub](https://github.com/KevorkSulahian/Gaming-the-Answer-Matcher)  
**Area**: Robotics (NLP / Evaluation)
**Keywords**: Answer Matching, LLM Evaluation, Adversarial Attack, Robustness, Automated Judgment

## TL;DR

This paper systematically evaluates three text manipulation strategies—verbosity, strategic multi-answer embedding, and correct-answer-first with contradictory suffix—against LLM-based answer-matching judges. The results show that these manipulations **do not improve scores and often reduce them**. Binary scoring proves more robust than continuous scoring, demonstrating that answer matching is resistant to low-cost text manipulation as an evaluation method.

## Background & Motivation

### Evolution of Evaluation Paradigms

Model evaluation is a bottleneck in LLM development:
- **Human evaluation**: Reliable but expensive and slow.
- **LLM-as-Judge**: Flexible but prone to reliability issues, bias, and hallucination.
- **Answer Matching**: Compares free-text responses against reference answers; objective and scalable.

Answer matching has demonstrated strong alignment in MCQ evaluation, making it particularly suitable for pre-deployment benchmark validation—since many benchmark datasets include reference answers.

### Core Concern

**Problem**: Can answer matchers be fooled by surface-level cues? Prior work has shown that LLM judges may be influenced by superficial factors such as chain-of-thought prompting, verbose responses, and punctuation. If answer matchers share this vulnerability, their value as a reliable evaluation method would be substantially undermined.

### Three Research Questions

1. Can responses containing ambiguous multi-answer content deceive the matcher?
2. Do verbose responses receive higher scores due to verbosity bias?
3. Which is more robust—binary judgment (correct/incorrect) or continuous judgment (scores in $[0,1]$)?

## Method

### Overall Architecture

The experimental pipeline proceeds as follows: manually designed prompts → examinee model generates baseline and manipulated free-text responses → matcher model scores responses against reference answers → metrics are computed to test hypotheses.

### Key Designs

#### 1. **Three Attack Strategies**

- **Verbose Attack**: Appends substantial redundant text to the generated response without altering its substantive content.
- **Strategic Attack**: When the model is uncertain, it generates an ambiguous response embedding multiple candidate answers. Few-shot prompting improves this strategy. For example, given a question about the emission spectrum of Li$^{++}$, the baseline response directly states "wavelength decreases by a factor of 1/9," while the strategic response vaguely states "wavelength generally decreases, with the reduction factor related to the square of the atomic number."
- **Forward Attack**: Places the correct answer at the beginning of the response and embeds a contradictory incorrect answer at the end.

#### 2. **Dataset Preparation**

Two challenging benchmarks are used:
- **MMLU-Pro**: Quantitative subset (1,962 questions) and qualitative subset (1,405 questions).
- **GPQA Diamond**: Quantitative subset (92 questions) and qualitative subset (106 questions).

Preprocessing: GPT-4.1 mini classifies questions as quantitative or qualitative; questions containing phrases such as "which of the following" that reference answer options are filtered out, since free-text respondents do not have access to those options.

#### 3. **Model Configuration**

Examinee models: GPT-4.1 mini, Qwen2.5-7B-Instruct.
Matcher models: GPT-4.1 mini, Qwen2.5-7B-IT, Qwen3-4B, Gemma-2-2B-IT.

This selection covers different model families and scales and enables observation of self-preference bias, as GPT-4.1 mini and Qwen2.5-7B-IT serve as both examinees and matchers.

#### 4. **Evaluation Metrics**

- **Average Alignment ($\bar{A}_c$)**: Mean score across all questions under each condition, reflecting overall accuracy.
- **Attack Success Rate (ASR)**: Proportion of questions where the post-attack score exceeds the baseline score (in binary settings, the rate of $0 \to 1$ flips).
- **Cohen's $d$**: Effect size of the attack; negative values indicate that the attack reduces scores.

### Statistical Testing

Two-proportion $z$-tests compare average alignment between attack conditions and baseline conditions, with a $p$-value threshold of $0.05$.

## Key Experimental Results

### Main Results: Strategic Attack on GPQA

**Binary Judgment**:

| Matcher | Examinee | ASR (Qual) | ASR (Quant) | Cohen's $d$ (Qual) | Cohen's $d$ (Quant) |
|---------|----------|-----------|------------|-----------------|-------------------|
| GPT-4.1 mini | Qwen3-4B | 0.094 | 0.043 | 0.093 | -0.535 |
| GPT-4.1 mini | GPT-4.1 mini | 0.038 | 0.011 | -0.130 | -0.578 |
| Qwen 2.5 7B | GPT-4.1 mini | 0.028 | 0.011 | -0.111 | -0.212 |
| Qwen 2.5 7B | Gemma-2-2B | 0.075 | 0.076 | 0.300 | 0.127 |

**Continuous Judgment**:

| Matcher | Examinee | ASR (Qual) | ASR (Quant) | Cohen's $d$ (Qual) | Cohen's $d$ (Quant) |
|---------|----------|-----------|------------|-----------------|-------------------|
| GPT-4.1 mini | GPT-4.1 mini | 0.349 | 0.326 | -0.044 | -0.478 |
| Qwen 2.5 7B | GPT-4.1 mini | 0.415 | 0.489 | 0.056 | 0.057 |
| Qwen 2.5 7B | Qwen3-4B | 0.475 | 0.360 | 0.224 | 0.111 |

**Core Findings**:
- Cohen's $d$ is negative in the vast majority of experiments, indicating that scores **decrease** after attack.
- All experiments yield $p < 0.05$ under the $z$-test, with baseline prompts achieving statistically significantly higher average alignment than attack prompts.
- ASR remains consistently low under binary judgment (mostly $< 0.1$); ASR is higher under continuous judgment but Cohen's $d$ remains negative or negligible.

### Ablation Study: Binary vs. Continuous Judgment Robustness

| Judgment Mode | Characteristics | ASR Range | Notes |
|---------|------|---------|------|
| Binary | More strict | 0.00–0.094 | Attacks are nearly ineffective; negative Cohen's $d$ indicates attacks reduce accuracy |
| Continuous | More lenient | 0.13–0.489 | Notably higher ASR; continuous scale tolerates partial correctness |
| Answer Matcher vs. LLM-as-Judge | Matcher is stricter | — | Matcher scores are systematically lower than judge scores |

**Gemma-2-2B Anomaly**: As a binary matcher on GPQA, Gemma-2-2B assigns anomalously high scores (sometimes reaching perfect accuracy of 1.0), suggesting that smaller models may be unreliable in certain settings.

### Key Findings

1. **All three attacks fail**: Verbose, strategic, and forward attacks do not improve answer-matching scores and typically reduce them.
2. **Binary judgment is more robust**: ASR under continuous judgment is significantly higher than under binary judgment, as the continuous scale permits more "partially correct" judgments.
3. **Answer Matcher is stricter than LLM-as-Judge**: This corroborates findings from prior work.
4. **Model size may affect robustness more than attack strategy**: The anomalous behavior of Gemma-2-2B suggests this direction warrants further investigation.

## Highlights & Insights

- **Rigorous experimental design**: A comprehensive combination of 2 examinees × 4 matchers × 4 dataset subsets × 4 prompt types yields 32 experimental configurations.
- **Practically significant conclusions**: When using answer matching for pre-deployment benchmark evaluation, practitioners need not be concerned about simple text manipulation interfering with results.
- **Null results carry value**: The original hypothesis anticipated successful attacks, but the data clearly demonstrates their ineffectiveness—this type of negative result is important for building community confidence.
- **Caching ensures reproducibility**: All model calls are cached to guarantee deterministic results.

## Limitations & Future Work

- The attack strategies are low-cost, non-adaptive attacks; optimized and adaptive adversarial attacks may prove more effective.
- Only English-language settings are evaluated; cross-lingual robustness remains unknown.
- The anomalous behavior of Gemma-2-2B is not analyzed in depth; the effect of model scale on robustness merits systematic investigation.
- More sophisticated attacks, such as optimized prompts that exploit model-specific weaknesses, are not considered.

## Related Work & Insights

- Chandak et al. first demonstrated that small Qwen models can serve as effective answer matchers; this paper validates their robustness.
- Compared to LLM-as-Judge approaches (e.g., MT-Bench), answer matching is better suited for settings where reference answers are available.
- Insight: The robustness of evaluation methods should be a central consideration in their design—measuring not only accuracy but also resistance to manipulation.

## Rating

- Novelty: ⭐⭐⭐ — The research question is valuable, but the attack strategies are relatively simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 32 comprehensive experimental configurations spanning multiple models, datasets, and judgment modes.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous hypothesis–experiment–conclusion logic.
- Value: ⭐⭐⭐⭐ — Provides confidence in answer matching as a reliable evaluation method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](../../ACL2026/llm_evaluation/sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](../../ACL2026/llm_evaluation/question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)
- [\[ICLR 2026\] ProfBench: Multi-Domain Rubrics requiring Professional Knowledge to Answer and Judge](../../ICLR2026/llm_evaluation/profbench_multi-domain_rubrics_requiring_professional_knowledge_to_answer_and_ju.md)
- [\[ACL 2026\] Automated Creativity Evaluation of Language Models Across Open-Ended Tasks](../../ACL2026/llm_evaluation/automated_creativity_evaluation_of_language_models_across_open-ended_tasks.md)

</div>

<!-- RELATED:END -->
