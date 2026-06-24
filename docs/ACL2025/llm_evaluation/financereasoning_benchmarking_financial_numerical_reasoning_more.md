---
title: >-
  [Paper Note] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging
description: >-
  [ACL 2025][LLM Evaluation][Financial Numerical Reasoning] The FinanceReasoning benchmark is proposed to improve financial numerical reasoning evaluation across three dimensions: credibility, comprehensiveness, and level of challenge. This is achieved by re-annotating public datasets, constructing a library of 3,133 Python financial functions, and introducing 908 expert-annotated hard questions.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Financial Numerical Reasoning"
  - "benchmark"
  - "Large Reasoning Models"
  - "knowledge enhancement"
  - "function library"
date: 2026-05-08
content_hash: 7f962331a3b51b9c
---

# FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging

**Conference**: ACL 2025  
**arXiv**: [2506.05828](https://arxiv.org/abs/2506.05828)  
**Code**: [https://github.com/BUPT-Reasoning-Lab/FinanceReasoning](https://github.com/BUPT-Reasoning-Lab/FinanceReasoning)  
**Area**: Financial Reasoning / Benchmark  
**Keywords**: Financial Numerical Reasoning, benchmark, Large Reasoning Models, knowledge enhancement, function library

## TL;DR

The FinanceReasoning benchmark is proposed to improve financial numerical reasoning evaluation across three dimensions: credibility, comprehensiveness, and level of challenge. This is achieved by re-annotating public datasets, constructing a library of 3,133 Python financial functions, and introducing 908 expert-annotated hard questions.

## Background & Motivation

**Background**: Large Reasoning Models (LRMs) such as OpenAI o1 and DeepSeek-R1 have achieved breakthroughs in general reasoning tasks, but they still face challenges in domain-specific numerical reasoning tasks like finance.

**Limitations of Prior Work**: Existing financial reasoning benchmarks (e.g., CodeFinQA, FinanceMath) suffer from three main issues: (1) poor annotation quality (with 9.72%-30% of the questions containing errors or ambiguities); (2) loose evaluation criteria (allowing a 1% error margin, ignoring units and signs); (3) an excess of simple questions where LRMs have saturated (>90% accuracy), making it difficult to objectively evaluate reasoning capabilities.

**Key Challenge**: Existing benchmarks fail to accurately reflect the actual reasoning improvements of LRMs. For instance, DeepSeek-R1 appears only 0.88% lower than V3 on the original CodeFinQA, yet it outperforms V3 by 2.01% on the re-annotated version.

**Goal**: Build a credible, comprehensive, and challenging financial numerical reasoning benchmark.

**Key Insight**: A three-pronged approach is utilized: correcting existing data, constructing a financial function knowledge base, and expertly annotating highly challenging new questions.

**Core Idea**: Build a high-quality financial numerical reasoning evaluation benchmark through a tri-fold strategy of "repairing old questions + building a knowledge base + generating difficult new questions."

## Method

### Overall Architecture

(1) Re-annotation of four public test sets (CodeFinQA/CodeTAT-QA/FinCode/FinanceMath), correcting 15.6% of the questions; (2) extraction and construction of 3,133 Python financial functions from Investopedia; (3) generation of 908 new questions using the function library to guide GPT-4o, followed by expert validation.

### Key Designs

1. **Updates to Public Datasets**: Three types of operations are performed on each question—Disambiguation (fixing unsolvable or ambiguous questions), Elaboration (adding missing calculation steps), and Correction (rectifying wrong answers). A strict 0.2% error limit is enforced, and units, percentages, and decimal places are strictly specified.
2. **Financial Function Library Construction**: Gathering 6,138 articles from Investopedia, and extracting financial calculation functions using GPT-4o. Each function includes a semantic signature, a detailed docstring (functionality, parameters, return values, constraints), and step-by-step implementation code. After review and correction by CFA-certified experts, 3,133 functions covering 1,864 financial concepts are constructed.
3. **Difficulty Grading Algorithm**: A heuristic difficulty evaluation based on the number of operators ($o$), the number of parenthesis pairs ($p$), and the lines of code ($l$) is proposed for the first time: $$rc = \ln(\max(o,1)) + \ln(\max(l+p,1))$$ which categorizes questions into Easy (1,000), Medium (1,000), and Hard (238).

### Loss & Training

Since this work focuses on evaluating a benchmark, there is no training process. Evaluation employs two prompting methods: CoT (Chain-of-Thought) and PoT (Program-of-Thought). Knowledge enhancement strategies and a collaborative Reasoner+Programmer paradigm are also explored.

## Key Experimental Results

### Main Results

Performance of various models on FinanceReasoning (Accuracy %):

| Model | Hard(CoT) | Hard(PoT) | Medium(CoT) | Medium(PoT) | Easy(CoT) | Easy(PoT) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI o1 | 81.1 | **89.1** | 89.7 | — | 88.0 | — |
| DeepSeek-R1 | 83.2 | 85.3 | 91.1 | 89.8 | 89.8 | 89.2 |
| OpenAI o3-mini | 77.3 | 84.0 | 87.8 | 88.6 | 88.8 | 88.1 |
| GPT-4o | 65.6 | 83.6 | 84.6 | 87.9 | 86.8 | 88.1 |
| Claude 3.5 Sonnet | 68.5 | 83.6 | 85.7 | 88.2 | 87.7 | 88.4 |
| Llama 3.3-70B | 50.4 | 71.4 | 79.2 | 85.9 | 83.3 | 84.8 |

### Ablation Study

Comparison before and after re-annotation (revealing true LRM improvements):

| Dataset | Evaluation Metric | DeepSeek-V3 | DeepSeek-R1 | R1 Relative Gain |
|--------|----------|:---:|:---:|:---:|
| CodeFinQA | Silver (Original) | 61.76 | 60.88 | -1.42% |
| CodeFinQA | **Gold (Re-annotated)** | 85.41 | 87.42 | **+2.35%** |
| FinanceMath | Silver (Original) | 58.50 | 71.00 | +21.37% |
| FinanceMath | **Gold (Re-annotated)** | 59.50 | 83.50 | **+40.34%** |

### Key Findings

- PoT significantly outperforms CoT, especially on Hard questions (OpenAI o1: 81.1% $\rightarrow$ 89.1%), demonstrating that programmatic reasoning is crucial for numerical precision.
- Knowledge enhancement is effective: GPT-4o + financial function library improves accuracy from 83.2% to 91.6% (+8.4%).
- The cooperative Reasoner + Programmer strategy is effective: DeepSeek-R1 + Programmer increases accuracy from 83.2% $\rightarrow$ 87.8% (+4.6%).
- LRMs still face challenges with formula misselection and insufficient numerical precision.
- Quality issues present in 9.72%-30% of the original datasets significantly underestimated the actual reasoning improvements of LRMs.

## Highlights & Insights

- The methodology makes a strong contribution by exposing severe annotation quality issues in existing benchmarks through rigorous re-annotation.
- The financial function library (3,133 functions) serves a dual purpose: it is used for both generating questions and as a knowledge enhancement resource to boost model performance.
- The difficulty grading algorithm is simple and effective, providing a solid reference for building future domain-specific benchmarks of varying difficulties.

## Limitations & Future Work

- The Hard subset comprises only 238 questions, which is relatively small and might not be fully representative.
- Difficulty grading is based on structural code features, which does not account for the intrinsic conceptual comprehension difficulty of financial topics.
- It only covers English financial reasoning; other linguistic scenarios (e.g., Chinese finance) are not addressed.
- The function library is sourced from Investopedia, which may introduce coverage bias (skewing towards the Western/US financial system).

## Related Work & Insights

- Relationship with BizBench: FinanceReasoning re-annotates and extends BizBench, establishing more rigorous evaluation criteria.
- Difference from general reasoning benchmarks (GSM8K, MATH): Emphasizes domain-specific knowledge application and numerical precision.
- Inspiration: The collaborative hybrid model of Reasoner + Programmer is worth promoting to other domains (e.g., physics, engineering computation).

## Supplementary Analysis

- The average number of operators in Easy/Medium/Hard is 1.77, 3.79, and 10.12 respectively, and the lines of code are 3.13, 4.27, and 9.49, demonstrating a well-designed difficulty gradient.
- The function library features an average of 2.85 operators and 2.64 parameters per function, covering 1,864 financial concepts.
- The annotation team, consisting of 8 interdisciplinary graduate students and 2 CFA-certified experts, represents a high standard of quality; the entire annotation process took three months.
- The complete dataset and the entire financial function library have been open-sourced, making a significant contribution to the community.
- Difficulty distribution comparisons show that the proportion of medium and hard questions in FinanceReasoning is much higher than in existing datasets.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The threefold strategy of re-annotation + function library + hard questions is novel, though the overall work remains a benchmark engineering project.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 13 models evaluated across multiple strategies; the comparative analysis of original vs. re-annotated data is highly convincing.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with abundant figures and tables.
- **Value**: ⭐⭐⭐⭐ High-quality financial reasoning benchmark + open-sourced function library, providing a significant contribution to domain evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)
- [\[ACL 2025\] PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning](physreason_a_comprehensive_benchmark_towards_physics-based_reasoning.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)

</div>

<!-- RELATED:END -->
