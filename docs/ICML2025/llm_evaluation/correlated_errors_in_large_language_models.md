---
title: >-
  [Paper Note] Correlated Errors in Large Language Models
description: >-
  [ICML 2025][LLM Evaluation][LLM Correlation] Through a large-scale empirical analysis of over 350 LLMs, this paper reveals highly correlated error patterns across different LLMs. When both models make mistakes, they choose the same incorrect answer in approximately 60% of cases, and more accurate models exhibit higher correlation. Furthermore, the paper investigates the downstream impacts of this correlation on LLM-as-Judge evaluation and the labor market.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "LLM Correlation"
  - "Algorithmic Monoculture"
  - "Error Consistency"
  - "LLM-as-Judge"
  - "Labor Market"
date: 2026-05-08
content_hash: 9542e242c68e8f6d
---

# Correlated Errors in Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2506.07962](https://arxiv.org/abs/2506.07962)  
**Code**: [https://github.com/nikhgarg/llm_correlated_errors_public/](https://github.com/nikhgarg/llm_correlated_errors_public/)  
**Area**: LLM Evaluation  
**Keywords**: LLM Correlation, Algorithmic Monoculture, Error Consistency, LLM-as-Judge, Labor Market

## TL;DR
Through a large-scale empirical analysis of over 350 LLMs, this paper reveals highly correlated error patterns across different LLMs. When both models make mistakes, they choose the same incorrect answer in approximately 60% of cases, and more accurate models exhibit higher correlation. Furthermore, the paper investigates the downstream impacts of this correlation on LLM-as-Judge evaluation and the labor market.

## Background & Motivation
**Background**: LLMs are increasingly deployed in multi-agent or multi-model high-risk scenarios (e.g., evaluation, recruitment), operating under the implicit assumption that "using different models introduces diversity and robustness."  

**Limitations of Prior Work**: There is a lack of large-scale empirical data to verify whether different LLMs truly display meaningful differences in behavior. Whether multiple enterprises using seemingly different models can truly avoid systemic exclusion remains unvalidated.  

**Key Challenge**: Intuitively, models with different architectures or from different vendors should exhibit different error distributions; however, if training data and optimization objectives converge, models may align in their error patterns.  

**Goal**: To quantify the correlation of errors across LLMs, identify factors driving this correlation, and evaluate its impact on practical downstream scenarios.  

**Key Insight**: Utilize multiple-choice question responses from two major leaderboards (HuggingFace Open LLM Leaderboard, Stanford HELM) along with a proprietary resume screening dataset to systematically analyze the consistency of incorrect answers across different models.  

**Core Idea**: LLM errors are not random and independent; more accurate models tend to converge on the same errors, posing risks to application scenarios that rely on model diversity.

## Method

### Overall Architecture
The research framework of this paper consists of three major parts: (1) quantifying the extent of LLM error correlation; (2) using regression analysis to explain the sources of correlation; (3) analyzing the practical impacts of correlation in two downstream tasks. Three datasets are utilized: HuggingFace (349 LLMs, 12,032 multiple-choice questions), HELM (71 LLMs, 14,042 multiple-choice questions), and resume screening (20 LLMs, 1,800 resume-job pairs).

### Key Designs

1. **Agreement Rate When Both Wrong**: 

    - Function: Measures the probability of two models choosing the exact same incorrect option conditional on both answering incorrectly.
    - Mechanism: Controls for accuracy by conditioning on "both models being wrong" to eliminate the confounding factor of model accuracy—highly accurate models naturally agree on correct answers, but this does not represent true algorithmic monoculture.
    - Design Motivation: The baseline agreement rate with random guessing is 1/3 on HELM (3 incorrect options) and 0.127 on HuggingFace (3-10 options).
    - Novelty: Compared to Goel et al. (2025), this approach does not rely on model output probability distributions, making it more applicable to black-box scenarios where only the final answer is accessible.

2. **Regression Analysis of Error Agreement**: 

    - Function: Employs linear regression to analyze which factors drive error correlation between models.
    - Mechanism: Uses the agreement rate of each model pair as the dependent variable, with indicators for whether they are from the same vendor, whether they share the same architecture, their respective accuracies, and their interaction terms as independent variables.
    - Design Motivation: Distinguishes "surface-level similarity" (same company/architecture) from "deep convergence" (intrinsic similarity among highly accurate models).
    - Key Findings: Even when controlling for vendor and architecture, more accurate model pairs still exhibit significantly higher error correlation.

3. **Resume-Job Evaluation**: 

    - Function: Constructs 1,800 pairs from 30 job descriptions $\times$ 60 resumes, scored by 20 LLMs, with 450 pairs annotated by humans as ground truth.
    - Mechanism: Measures correlation through residual correlation, where the residual is defined as the model score minus the human score.
    - Design Motivation: Extends evaluations from multiple-choice questions to open-ended assessments closer to real-world deployment scenarios.

### Loss & Training
This work is an empirical analysis and does not involve model training. Regression analysis is performed using standard OLS regression, with all continuous numerical variables standardized.

## Key Experimental Results

### Main Results

| Dataset | Metric | Average Agreement Rate | Random Baseline | Multiplier |
|--------|------|-------------------|----------|------|
| HuggingFace | Agreement when both wrong | 0.423 | 0.127 | 3.3× |
| HELM | Agreement when both wrong | 0.600 | 0.333 | 1.8× |
| Resumes | Residual correlation | Highly correlated | 0 | - |

| Regression Factor | HuggingFace Coeff. | HELM Coeff. | Resumes Coeff. |
|----------|----------------|----------|------------|
| Same Company | 0.066** | 0.022** | 0.021 |
| Same Architecture | 0.076** | - | - |
| Acc.1 | 0.014** | 0.055** | 0.015** |
| Acc.2 | 0.013** | 0.054** | 0.028** |
| Acc.1 × Acc.2 | 0.023** | 0.026** | 0.043** |
| R² | 0.340 | 0.613 | 0.415 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Remove "Same Company" variable | R² decreases slightly | Vendor is an important but not the sole factor |
| Keep only accuracy variables | R² remains significant | Accuracy itself is an independent driver of error convergence |
| Extreme case | 0.9987 agreement | google/text-unicorn vs writer/palmyra-x-v3 are almost identical |

### Key Findings
- Almost all model pairs exhibit agreement rates significantly above the random baseline (100% of pairs in HuggingFace, 97.5% in HELM).
- More accurate models display higher error correlation, even across different architectures and vendors—indicating that as models "improve," they also "converge."
- In LLM-as-Judge settings, evaluator models systematically overestimate the accuracy of weaker models with which they share incorrect answers; this bias is more pronounced for models from the same vendor.
- In labor market simulations, using 20 different LLMs to filter resumes still results in approximately 20% systemic exclusion.
- The impact of algorithmic monoculture on applicant welfare is dual-sided: it increases systemic exclusion, but yields more choice for successful candidates.

## Highlights & Insights
- Unveils a counter-intuitive phenomenon: employing better/newer models might actually homogenize the ecosystem.
- Unprecedented experimental scale: systematic analysis spanning 350+ models and 26,000+ questions.
- Empirically validates the concept of "algorithmic monoculture" using real LLMs and genuine resume data.
- Issues an important warning regarding the LLM-as-Judge paradigm: evaluators bias toward models that share similar error profiles.

## Limitations & Future Work
- Evaluates only multiple-choice questions and numerical scoring scenarios, lacking analysis of open-ended generation tasks.
- The metric for error correlation treats all incorrect answers equally without distinguishing "near-correct" mistakes.
- The ground truth in resume evaluations is derived from a limited set of human annotations (450 pairs), bearing higher subjectivity.
- Active mitigation strategies to reduce model correlation are not explored.
- The labor market simulation is highly simplified and does not incorporate dynamic feedback loops present in real markets.

## Related Work & Insights
- Goel et al. (2025) contemporaneously and independently proposed a similar measure of model similarity, but utilized probability distribution information.
- Wu et al. (2024) investigated generative diversity and identified monoculture tendencies within LLMs.
- Peng & Garg (2024a) theoretically analyzed the implications of algorithmic monoculture on the labor market; this work provides empirical validation.
- Insight: Future leaderboards should not only track individual model accuracy but also continuously monitor inter-model correlation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ICML 2025\] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)
- [\[ACL 2026\] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](../../ACL2026/llm_evaluation/identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)
- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)

</div>

<!-- RELATED:END -->
