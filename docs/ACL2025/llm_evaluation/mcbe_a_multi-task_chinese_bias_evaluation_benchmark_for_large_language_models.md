---
title: >-
  [Paper Note] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models
description: >-
  [ACL 2025][LLM Evaluation][Chinese bias] This paper proposes McBE, the first multi-task Chinese bias evaluation benchmark, containing 4,077 Bias Evaluation Instances (BEIs) across 12 bias categories and 82 subcategories. By utilizing 5 evaluation tasks (Preference Computation, Subcategory Classification, Scenario Selection, Bias Analysis, and Bias Scoring), McBE multi-dimensionally quantifies Chinese bias in LLMs, revealing that the conventional conclusion of "larger paramete…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Chinese bias"
  - "benchmark"
  - "fairness"
  - "stereotypes"
  - "multi-task evaluation"
date: 2026-05-08
content_hash: f11ecd6105c1fbea
---

# McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2507.02088](https://arxiv.org/abs/2507.02088)  
**Code**: [GitHub](https://github.com/) (dataset and code accompanied with the paper)  
**Area**: LLM Evaluation  
**Keywords**: Chinese bias, benchmark, fairness, stereotypes, multi-task evaluation

## TL;DR

This paper proposes McBE, the first multi-task Chinese bias evaluation benchmark, containing 4,077 Bias Evaluation Instances (BEIs) across 12 bias categories and 82 subcategories. By utilizing 5 evaluation tasks (Preference Computation, Subcategory Classification, Scenario Selection, Bias Analysis, and Bias Scoring), McBE multi-dimensionally quantifies Chinese bias in LLMs, revealing that the conventional conclusion of "larger parameters yield stronger bias" may stem from the limitations of single-task evaluations.

## Background & Motivation

**Background**: LLMs are widely applied in various NLP tasks, but biases in training data (such as gender-occupation associations and regional stereotypes) are inevitably inherited by models, posing risks to social fairness.

**Limitations of Prior Work**: (1) The vast majority of bias evaluation datasets are based on English and North American culture (e.g., WinoBias, StereoSet, CrowS-Pairs, BBQ), which fail to cover biases specific to Chinese society; (2) Existing Chinese bias datasets (e.g., CHBias, CBBQ) have limited categories and only support a single evaluation task (such as QA), failing to measure bias from multiple angles.

**Key Challenge**: There are numerous unique types of bias within Chinese culture (e.g., regional bias, worldview bias, subculture bias), but targeted multi-dimensional evaluation tools are sorely lacking.

**Goal**: To construct a comprehensive Chinese bias evaluation benchmark with diverse tasks, systematically quantifying Chinese bias in LLMs from multiple perspectives.

**Key Insight**: This work proposes the concept of Bias Evaluation Instances (BEIs) as the elementary evaluation unit. Each BEI contains six attributes: context, sentence template, substitution list, subcategories, explanation, and bias score, natively supporting 5 evaluation tasks.

**Core Idea**: Multiple bias categories × multiple evaluation tasks = more comprehensive and accurate Chinese bias evaluation.

## Method

### Overall Architecture

The core design of McBE is the **Bias Evaluation Instance (BEI)**. Each BEI consists of 6 attributes:
- **Context**: Provides linguistic context to help models understand sentences
- **Sentence Template**: A sentence template containing a placeholder `[PLH]`
- **Substitution List**: A list of replacement words used to generate complete sentences for different demographic groups
- **Bias Subcategories**: Manually annotated bias subcategories
- **Explanation**: Detailed explanation of bias (initially written independently by 3 annotators, merged by ChatGLM, and finally audited by 2 annotators)
- **Bias Score**: Bias severity rating (0-10 scale, the mean of 6 annotators' scores)

All 4,077 BEIs cover 12 bias categories and 82 subcategories, sourced from social media platforms (51.85%), personal experiences (35.79%), and translations/adaptations of other datasets (12.36%).

### Key Designs

1. **Comprehensive Coverage of 12 Bias Categories + 82 Subcategories**

    - Universal Categories: Gender (7 subcategories), Race (11 subcategories), Religion (9 subcategories), Age (8 subcategories)
    - Chinese-Specific Categories: Regional bias (provinces/urban-rural/locals-outlanders), Worldviews (marriage & childbearing/lifestyle/consumption views), Subcultures (appearance/hobbies/personal attributes)
    - Classification Criteria: Partly based on protected groups in Chinese labor law and protection laws for people with disabilities, and partly based on social diversity needs
    - Design Motivation: Distinguishing cultural differences (neutral facts) from bias (discriminatory expressions) to ensure the evaluation target represents genuine bias

2. **Multi-dimensional Measurement via 5 Evaluation Tasks**

    - **Task 1: Preference Computation (PC)**: Computes the negative log-likelihood (NLL) of each sentence in the substitution list under the model, and measures preference differences via NLL variance (variance = 0 indicates no bias), mapped to a scale of 0-100 using an exponential decay function
    - **Task 2: Subcategory Classification (SC)**: Prompts the model to classify which bias subcategory a sentence belongs to; the accuracy × 100 serves as the score
    - **Task 3: Scenario Selection (SS)**: Generates sentence pairs from BEIs, prompts the model to choose which sentence is more likely to occur, and computes the variance of selection frequencies
    - **Task 4: Bias Analysis (BA)**: The model analyzes the biases in a sentence. GLM4-AIR serves as a judge to score the analysis across four weighted dimensions: accuracy, underlying implications, cultural differences, and highlights
    - **Task 5: Bias Scoring (BS)**: The model scores the severity of bias in a sentence, calculating the mean absolute error (MAE) against human-annotated scores; smaller differences yield higher scores
    - Design Motivation: PC and SS measure the model's intrinsic preferences, whereas SC, BA, and BS measure the model's comprehension of bias and alignment of values

### Loss & Training

- The PC/SS tasks utilize an exponential decay function: $\text{Score} = 100 \cdot e^{-r \cdot V}$, where $r = \frac{2e}{3}$, and $V$ is the NLL variance
- The BA task utilizes a weighted scoring formula: $\text{Final Score} = \frac{\sum_{i=1}^{4} s_i \cdot w_i}{\sum_{i=1}^{4} w_i}$ (weight is 3.5 for accuracy, 1.5 for underlying implications, 2.5 for cultural differences, and 0.5 for highlights)
- The BS task utilizes the mean absolute deviation: $\text{Final Score} = 100 - 10 \cdot \frac{1}{n}\sum_{i=1}^{n}|d_i|$

## Key Experimental Results

### Main Results -- Model Performance Across Bias Categories

| Model | Religion | Region | Nationality | Race | Overall Trend |
|------|------|------|------|------|---------|
| InternLM2.5-7B | Higher | Higher | Medium | Lower | Optimal among 7B models |
| Qwen2.5-32B | Higher | Higher | Medium | Lower | Large parameters yield significant improvement |
| GLM4-AIR/0520 | Medium | Medium | Lower | Lower | Inferior to some 7B models |
| Llama2-7B-hf | High in PC/SS | — | — | — | Extremely low in SC/BA/BS |
| Mistral-7B | — | — | — | — | Outperforms Llama2 but shares a similar trend |

- All models score highest in **religion and region** categories (least bias) and lowest in **nationality and race** categories (most bias).

### Parameter Scale Experiment -- Qwen2.5 Series (0.5B → 1.5B → 7B → 32B)

| Parameter Size | SS Score | SC/BA/BS Trend | Interpretation |
|--------|---------|-------------|------|
| 0.5B | 87.69 | Lowest | High SS score stems from random selections |
| 1.5B | 80.49 | Low-medium | 0.5B→1.5B shows the most significant improvement |
| 7B | 77.82 | Medium-high | Diminishing marginal returns |
| 32B | 77.11 | Highest | Low SS but possesses the strongest comprehension |

### Key Findings

1. **Reversal of Traditional Conclusions**: Benchmarks like CBBQ and Rubia concluded that "larger parameter sizes lead to stronger bias" using only SS-like tasks. In contrast, McBE's SC/BA/BS tasks reveal that the high SS scores of small models stem from random selection rather than genuine bias-freeness, and larger models actually exhibit superior performance in bias comprehension and value alignment.
2. **Cultural Specificity**: Multilingual models (e.g., Llama2, Mistral) perform decently on PC/SS but fall severely short on SC/BA/BS, demonstrating that English-centric models struggle to understand Chinese-specific cultural biases.
3. **Anomalies in the GLM4 Series**: Despite its larger size, GLM4-AIR/0520 scores lower than some 7B models, implying a higher proportion of biased content in its training data.

## Highlights & Insights

- **Pioneering the BEI Concept**: The Bias Evaluation Instance is an atomic evaluation unit that naturally supports multi-task settings and can be extended to other languages.
- **Challenging Traditional Cognitive Assumptions**: Multi-task evaluation reveals that the previous conclusion "larger models have stronger bias" might be an artifact of single-task evaluation paradigms.
- **Rigorous Annotation**: Involving 30 annotators from diverse backgrounds, a 1:1 gender ratio, cross-province representation, and sociology experts, which is significantly more representative than CHBias (3 annotators) and IndiBias (5 annotators).
- **Unmatched Scope**: The coverage of 12 categories and 82 subcategories is unparalleled among Chinese bias benchmarks.

## Limitations & Future Work

- **PC Task Unsuitable for Black-Box Models**: Preference computation relies on NLL probability distributions and cannot be applied to API-only models (such as GPT-4).
- **Subjectivity in Bias Definition**: Despite expert review, the boundary of bias (especially bias vs. cultural differences) still retains room for subjective judgment.
- **Static Dataset**: Social biases evolve over time, demanding dynamic update mechanisms for the dataset.
- **BA Task Reliance on LLM Judges**: Using GLM4-AIR as a judge may introduce its own biases.

## Related Work & Insights

- **vs CrowS-Pairs**: CrowS-Pairs covers 9 categories of English bias and only supports counterfactual input evaluation; McBE covers 12 categories of Chinese bias and 5 evaluation tasks, tailoring categories like regional, worldview, and subculture biases specifically to Chinese culture.
- **vs CBBQ**: CBBQ is a Chinese translation of BBQ and only supports QA evaluation tasks; McBE introduces the concept of BEI and 5 tasks, measuring bias from multiple dimensions and exposing the limitations of single-task evaluation.
- **vs CEB**: CEB proposes a compositional bias taxonomy but relies on Perspective API scoring, which is ineffective for biases not covered by the API; McBE utilizes human annotation and multi-task evaluation without relying on external APIs.

## Rating

- Novelty: ⭐⭐⭐⭐ Pioneered the BEI + 5-task paradigm, offering a powerful counter-evidence to the traditional assumption that "larger models exhibit stronger bias".
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation matrix spanning multiple model series across different parameter sizes, testing 12 categories × 5 tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear taxonomy, and profound definitions of the boundaries between bias and cultural differences.
- Value: ⭐⭐⭐⭐ Highly significant baseline contribution to Chinese LLM fairness research; the BEI concept can be easily generalized to other languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)

</div>

<!-- RELATED:END -->
