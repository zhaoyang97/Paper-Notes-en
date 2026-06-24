---
title: >-
  [Paper Note] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations
description: >-
  [ACL 2025][LLM Evaluation][psychometrics] Proposes the PATCH framework, which introduces Item Response Theory (IRT 3PL/2PL models) from psychometrics into LLM benchmarking. By comparing GPT-4V, Gemini-Pro-Vision, and Qwen-VL with human populations on the TIMSS 2011 eighth-grade mathematics test (88 questions, 56 countries/regions), it finds that IRT ability estimations differ significantly from simple accuracy rankings, with GPT-4V ranking in the same bracket as students from…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "psychometrics"
  - "IRT"
  - "TIMSS"
  - "human-LLM comparison"
  - "benchmarking"
date: 2026-05-08
content_hash: 642e515d5f5071c4
---

# PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations

**Conference**: ACL 2025  
**arXiv**: [2404.01799](https://arxiv.org/abs/2404.01799)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: psychometrics, IRT, TIMSS, human-LLM comparison, benchmarking

## TL;DR
Proposes the PATCH framework, which introduces Item Response Theory (IRT 3PL/2PL models) from psychometrics into LLM benchmarking. By comparing GPT-4V, Gemini-Pro-Vision, and Qwen-VL with human populations on the TIMSS 2011 eighth-grade mathematics test (88 questions, 56 countries/regions), it finds that IRT ability estimations differ significantly from simple accuracy rankings, with GPT-4V ranking in the same bracket as students from South Korea, Singapore, and Chinese Taipei. Additionally, four high-quality datasets are released (TIMSS 2011 & 2008 Mathematics/Science/Physics).

## Background & Motivation

**Background**: LLM academic capability benchmarks such as MMLU and GSM8K are widely used. Researchers often employ simple accuracy to compare LLMs with "human-level" performance, serving as a core reference for model selection and development directions.

**Measurement Quality Issues**: The item quality of existing benchmarks has not been validated through psychometrics—the difficulty and discrimination of items remain entirely unknown, and some items may even exhibit zero or negative discrimination.

**Coarse Evaluation Metrics**: Simple accuracy treats all questions with equal weight—scoring 10 simple questions correctly yields the same accuracy as 10 difficult ones, yet they reflect drastically different capabilities. This is a classic problem that psychometrics has solved for over 50 years.

**Unclear Human Reference**: Human performance in existing benchmarks often relies on convenience samples (such as MTurk workers), which do not represent any distinct human population. Consequently, the conclusion that "LLMs surpass humans" lacks rigorous meaning.

**Key Insight**: IRT is the gold standard of educational measurement (with a history of over 50 years), and TIMSS is one of the world's largest standardized international mathematics assessments. Combining the two simultaneously addresses the dual core problems of "measurement quality" and "human reference."

**Core Idea**: Use IRT models to estimate the ability parameter $\theta$ of LLMs, enabling a fair and precise comparison on the exact same scale as human students from 56 countries/regions.

## Method

### Overall Architecture
Select high-quality standardized tests (TIMSS 2011 Grade 8 Mathematics, 88 released questions) $\rightarrow$ Fit IRT models using response data from approximately 300,000 students across 56 countries/regions (estimating the difficulty $b$, discrimination $a$, and guessing rate $c$ of each item) $\rightarrow$ Have LLMs complete the exact same 88 questions $\rightarrow$ Estimate LLM ability values $\theta$ using the calibrated IRT models $\rightarrow$ Direct comparison with 56 human populations on the same scale.

### Key Designs

1. **Three-Parameter Logistic Item Response Theory Model (3PL-IRT)**

    - **Function**: Estimates three parameters for each multiple-choice item $j$—discrimination $a_j$, difficulty $b_j$, and guessing rate $c_j$.
    - Core Formula: $P(\theta) = c_j + \frac{1-c_j}{1+\exp(-a_j(\theta - b_j))}$
    - Uses the 2PL model ($c_j = 0$) for open-ended questions, as guessing does not apply to open response formats.
    - **Design Motivation**: 3PL is the standard model for handling multiple-choice questions in educational measurement, and the guessing parameter $c$ is critical for estimating the capabilities of lower-performing models.

2. **TIMSS 2011 as a Evaluation Benchmark**

    - **Function**: Utilizes 88 released eighth-grade mathematics questions from TIMSS 2011 (Trends in International Mathematics and Science Study).
    - Problems span four math domains: algebra, geometry, data and chance, and number theory.
    - Response data from approximately 300,000 students across 56 countries/regions is leveraged, with IRT parameters pre-calibrated by IEA experts.
    - **Design Motivation**: TIMSS undergoes rigorous cross-cultural validation and quality control procedures, yielding a measurement quality that far exceeds existing LLM benchmarks.

3. **Multimodal LLM Evaluation**

    - **Function**: Benchmarks multimodal models, including GPT-4V, Gemini-Pro-Vision, and Qwen-VL.
    - Core Process: Input questions (including diagrams/geometric figures) into LLMs $\rightarrow$ Extract answers $\rightarrow$ Embed response patterns into the IRT model to estimate $\theta$.
    - **Design Motivation**: Some TIMSS items contain diagrams/graphics, necessitating multimodal capabilities for fair evaluation.

4. **Four High-Quality Dataset Releases**

    - TIMSS 2011 Math + TIMSS 2008 Math + TIMSS 2011 Science + TIMSS 2011 Physics
    - Each dataset includes the original question text/images, gold standard answers, grading rubrics, and calibrated IRT parameters.

## Key Experimental Results

### Main Results -- LLM vs 56 Human Populations

| Model | IRT Ability Value | Equivalent Human Level | Simple Accuracy Ranking |
|------|---------|------------|----------|
| GPT-4V | High | Top 5 countries' level | May differ |
| GPT-3.5 | Medium | Median countries' level | May differ |
| Llama-3 | Medium-Low | Below average | May differ |

### IRT vs. Simple Accuracy Ranking Differences

| Comparison | Findings |
|------|------|
| Model Ranking | IRT and raw accuracy rankings can differ significantly |
| Human Comparison | IRT provides more precise positioning |

### Key Findings
- IRT ability estimation yields significantly different model rankings compared to simple accuracy, showing that simple accuracy can be misleading.
- Under IRT estimation, GPT-4V reaches the performance level of average 8th-grade students from the top 5 countries.
- Questions' difficulty and discrimination parameters heavily impact evaluation outcomes.
- The item quality of TIMSS far exceeds that of existing LLM benchmarks.

## Highlights & Insights
- **Introducing over 50 years of mature psychometric theory to LLM evaluation** represents a major methodological contribution.
- **Clear human referents across 56 countries/regions** solve the "compared to whom" pain point of existing benchmarks.
- **The ranking discrepancies between IRT and raw accuracy** demonstrate that simplistic metrics can be highly misleading.

## Limitations & Future Work
- Testing is restricted to 8th-grade mathematics and does not cover higher educational levels or other disciplines.
- IRT assumes that LLMs and humans follow the same underlying measurement model, which might not entirely hold true.
- Future directions: Expansion to multiple disciplines and the implementation of computerized adaptive testing (CAT) based on IRT.

## Related Work & Insights
- **vs. MMLU/GSM8K**: These benchmarks utilize simple accuracy, whereas PATCH leverages IRT.
- **vs. Elo rating**: Elo is typically used for peer-to-peer model comparison, whereas PATCH specializes in model-to-human population comparison.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing psychometrics into LLM evaluation is a key breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model evaluation against 56 human populations.
- Writing Quality: ⭐⭐⭐⭐⭐ Standardized with a highly robust theoretical foundation.
- Value: ⭐⭐⭐⭐⭐ Highly advances the methodology of LLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration](grace_a_granular_benchmark_for_evaluating_model_calibration_against_human_calibr.md)
- [\[ACL 2025\] WebWalker: Benchmarking LLMs in Web Traversal](webwalker_benchmarking_llms_in_web_traversal.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories](benchmarking_llms_and_llm-based_agents_in_practical_vulnerability_detection_for_.md)
- [\[ACL 2025\] ChatBench: From Static Benchmarks to Human-AI Evaluation](chatbench_from_static_benchmarks_to_human-ai_evaluation.md)

</div>

<!-- RELATED:END -->
