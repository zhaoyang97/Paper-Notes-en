---
title: >-
  [Paper Note] Large Language Models for Predictive Analysis: How Far Are They?
description: >-
  [ACL 2025][LLM (Other)][predictive analysis] Proposes the PredictiQ benchmark—the first comprehensive framework to systematically evaluate the predictive analysis capabilities of LLMs. Integrating 44 real-world datasets across 8 domains and 1,130 expert-designed queries, the benchmark evaluates 12 mainstream LLMs across three dimensions and seven aspects (text analysis, code generation, and text-code alignment). It reveals that even the strongest model…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "predictive analysis"
  - "benchmark"
  - "code generation"
  - "text analysis"
  - "text-code alignment"
  - "LLM evaluation"
date: 2026-05-08
content_hash: af1efd236e70b91e
---

# Large Language Models for Predictive Analysis: How Far Are They?

**Conference**: ACL 2025  
**Authors**: Qin Chen, Yuanyi Ren, Xiaojun Ma, Yuyang Shi (Peking University, Harvard University)  
**arXiv**: [2505.17149](https://arxiv.org/abs/2505.17149)  
**Code**: GitHub  
**Area**: LLM/NLP, Benchmark Evaluation  
**Keywords**: predictive analysis, benchmark, code generation, text analysis, text-code alignment, LLM evaluation  

## TL;DR

Proposes the PredictiQ benchmark—the first comprehensive framework to systematically evaluate the predictive analysis capabilities of LLMs. Integrating 44 real-world datasets across 8 domains and 1,130 expert-designed queries, the benchmark evaluates 12 mainstream LLMs across three dimensions and seven aspects (text analysis, code generation, and text-code alignment). It reveals that even the strongest model, GPT-4o-mini (GPT4O3Mini), still exhibits significant deficiencies in depth of analysis (2.91/4) and data preprocessing (absent in 51% of cases).

## Background & Motivation

- **Core Problem:** Predictive analysis (using statistical modeling and machine learning to forecast future trends from historical data) is the cornerstone of modern decision-making. LLMs have the potential to become out-of-the-box tools for non-expert users to perform predictive analysis, but the academic community lacks systematic benchmarks and methodologies to evaluate the actual capabilities of LLMs in this area.
- **Limitations of Prior Work:** Existing evaluation methods suffer from two fundamental flaws: (1) they only evaluate final model outputs (e.g., total sales from a database query), which is restricted by the LLM's context length, struggles with large-scale datasets, and scales poorly; (2) they only evaluate the execution results of generated code while ignoring textual explanations (such as justifications for algorithm selection or data feature analysis), leading to decreased usability and user trust. Neither approach is sufficient for highly complex predictive analysis tasks.
- **Motivation:** LLM-driven predictive analysis involves multiple phases, including data preprocessing, algorithm selection, and result interpretation. These phases require clear textual explanations to ensure reliability, executable code to implement analytical logic, and consistency between text and code to aid user comprehension. Therefore, a comprehensive evaluation framework that simultaneously assesses text analysis quality, code generation quality, and text-code alignment is urgently needed.
- **Key Insight:** Unlike prior works like Text2Analysis that focus solely on code generation, PredictiQ is the first end-to-end evaluation benchmark covering the entire pipeline of predictive analysis (Input $\rightarrow$ Analysis $\rightarrow$ Code $\rightarrow$ Explanation $\rightarrow$ Alignment).

## Method

### Data Collection and Query Construction

PredictiQ collects 44 real-world datasets across 8 common predictive analysis application domains from public platforms such as Kaggle, TCPD time-series benchmarks, and Econometric Analysis textbooks. Based on the characteristics of each dataset, data science experts formulated predictive queries under strict constraints: (a) use unambiguous language to clearly define the prediction goal; (b) rely solely on internal dataset information without external knowledge dependencies; (c) limit each query to a single question. Query types span classification (285 queries), regression (324 queries), forecasting (220 queries), clustering (137 queries), and anomaly detection (126 queries), totaling 1,130 queries.

| Domain | Datasets | Queries |
|------|---------|-------|
| Economics | 12 | 270 |
| Marketing & Sales | 6 | 200 |
| Industry Analysis | 7 | 180 |
| Transportation | 5 | 130 |
| Healthcare | 4 | 130 |
| Sociology | 4 | 110 |
| Human Resources | 3 | 80 |
| Education | 3 | 70 |
| **Total** | **44** | **1130** |

### Prompt Construction Strategy

Each Prompt consists of three parts: (1) **Query & Instructions**—instructing the LLM to act as a professional data scientist performing predictive analysis; (2) **Data Summary**—listing all column names, data types, maximum/minimum values of numerical columns, and the number of categories in categorical columns; (3) **Data Details**—providing the full dataset in CSV format. The data summary helps the LLM better understand the data structure, while the CSV format provides the concrete information required for analysis.

### Three-Domain, Seven-Aspect Evaluation Protocol

The design of the evaluation protocol is one of the core contributions of this work, covering three domains of the entire predictive analysis workflow:

- **Text Analysis (2 aspects)**: *Relevance*—how well the text analysis matches the data and query; *Depth*—whether the justification for algorithm selection is comprehensive and in-depth.
- **Code Generation (2 aspects)**: *Usefulness*—whether the code effectively solves the given query; *Functional Correctness*—whether the code executes without errors (validated through manual execution and mapping the execution success rate to a 0-4 scale).
- **Text-Code Alignment (3 aspects)**: *Description Accuracy*—whether the text accurately reflects the code functionality; *Coverage*—whether the text covers all relevant aspects of the code; *Clarity*—whether the correspondence between text and code is clear and easy to understand.

Each aspect is scored on a 0-4 scale, totaling 28 points. For the choice of evaluator, GPT-4-Turbo was chosen as the primary evaluator because it achieved a 90.5% alignment rate with 5 human data analysis experts—outperforming GPT-4o and Phi-3-Medium.

## Key Experimental Results

### Overall Performance of 12 LLMs on PredictiQ

| Model | Relevance | Depth | Usefulness | Functional Correctness | Description Accuracy | Coverage | Clarity | Total Score |
|------|-------|------|-------|----------|----------|-------|-------|------|
| GPT4O3Mini | 3.63 | 2.91 | 3.53 | 87% | 3.52 | 3.52 | 3.52 | **24.11** |
| GPT4O1 | 3.61 | 2.80 | 3.45 | 85% | 3.47 | 3.48 | 3.48 | 23.70 |
| GPT4O | 3.60 | 2.39 | 3.12 | 81% | 3.36 | 3.31 | 3.41 | 22.43 |
| GPT4Turbo | 3.39 | 2.18 | 2.78 | 78% | 3.09 | 2.95 | 3.18 | 20.68 |
| Phi4 | 2.94 | 2.55 | 2.87 | 54% | 2.84 | 2.82 | 2.88 | 19.06 |
| GPT3.5Turbo | 3.00 | 1.76 | 2.40 | 53% | 2.66 | 2.47 | 2.80 | 17.21 |
| CohereRPlus | 2.89 | 1.70 | 2.38 | 42% | 2.50 | 2.42 | 2.62 | 16.20 |
| Phi3Medium | 2.90 | 1.74 | 2.33 | 41% | 2.45 | 2.33 | 2.58 | 15.97 |
| ChatLlama2-70B | 2.32 | 1.51 | 1.78 | 21% | 1.25 | 1.27 | 1.60 | 10.57 |
| CodeLlama2-7B | 2.04 | 1.34 | 1.64 | 15% | 0.99 | 1.00 | 1.22 | 8.83 |
| ChatLlama2-13B | 1.97 | 1.24 | 1.53 | 18% | 1.02 | 1.03 | 1.24 | 8.75 |
| ChatLlama2-7B | 2.01 | 1.31 | 1.49 | 18% | 0.83 | 0.85 | 1.14 | 8.34 |

### Deep Analysis of Code Quality

LLM-generated code universally suffers from a lack of data preprocessing, and the types of errors vary in distinct patterns across different model sizes:

| Model | % No Preprocessing | Import Error | Syntax Error | Runtime Error | Library Error | Data Flow Error |
|------|------------|-----------|---------|----------|-------|----------|
| GPT4O3Mini | 51% | 0.3% | 3.1% | 2.3% | 2.9% | 4.4% |
| GPT4O | 66% | 0.4% | 5.1% | 3.9% | 3.8% | 5.8% |
| GPT4Turbo | 66% | 1.3% | 7.8% | 3.8% | 3.4% | 5.7% |
| GPT3.5Turbo | 71% | 3.8% | 15.3% | 7.6% | 7.8% | 12.5% |
| Phi4 | 58% | 3.8% | 12.3% | 10.3% | 9.5% | 10.2% |
| ChatLlama2-70B | 87% | 15.0% | 12.5% | 22.7% | 16.3% | 12.5% |
| CodeLlama2-7B | 89% | 38.2% | 17.4% | 11.3% | 8.8% | 9.3% |

### Key Findings

1. **The Double-Edged Sword of Code Fine-Tuning**: CodeLlama2-7B outperforms the same-sized ChatLlama2-7B and even the 13B Chat model in text analysis and text-code alignment (total score 8.83 vs 8.75), yet its code execution rate is lower (15% vs 18%). This suggests that over-specialized code fine-tuning can harm actual code execution quality.
2. **Non-linear Relationship between Model Scale and Performance**: Increasing the parameter scale of the Llama family ($7\text{B} \rightarrow 13\text{B} \rightarrow 70\text{B}$) improves the total score ($8.34 \rightarrow 8.75 \rightarrow 10.57$), but the improvement in code execution rate is limited ($18\% \rightarrow 18\% \rightarrow 21\%$), and the proportion of logical errors even increases.
3. **LLMs Universally Neglect Data Preprocessing**: Even the best-performing GPT4O3Mini produces code where 51% lacks basic data preprocessing (such as handling missing values or filtering anomalies). This issue is even more severe in smaller models (reaching 92% for ChatLlama2-7B).
4. **Significant Performance Discrepancy Across Domains**: ChatLlama2-70B outscored its average score by 31.4% in the education domain, but its performance dropped significantly in other domains. GPT4O3Mini and GPT4O achieved the most balanced performance, with cross-domain score variations of only 1.67 and 1.82 points, respectively.
5. **Reasoning Models Require Large Contexts**: GPT4O1 and GPT4O3Mini perform far worse than GPT4O under a 4K token limit. They require extending the context to 32K tokens to unleash their reasoning capabilities, though GPT4O3Mini reaches GPT4O1-level performance using fewer tokens.

## Highlights & Insights

- First comprehensive benchmark targeting the predictive analysis capabilities of LLMs, completely covering the end-to-end analysis pipeline from text, code, to alignment, and filling a gap in the field.
- Rigorous evaluation protocol design—investing 300 person-hours to construct queries and 900 person-hours to evaluate responses, with experiments validating a 90.5% alignment rate between GPT4Turbo (as evaluator) and human experts.
- Large-scale experiments and deep insights: A full evaluation of 12 models across 1,130 queries, revealing counter-intuitive findings such as the double-edged sword effect of code fine-tuning and the context dependency of reasoning models.
- Evaluator bias analysis reveals the "round number bias" of LLM-as-a-judge, where Phi3Medium gave almost indistinguishable high scores to all target models, weakening the discriminative power of the evaluation.

## Limitations & Future Work

- Only evaluates predictive analysis tasks, failing to cover other advanced data analysis categories such as prescriptive or diagnostic analysis.
- The datasets are restricted to structured tabular data from 8 common application domains, omitting unstructured forms like images or graph data.
- Queries are limited to single-question formats, without addressing multi-step complex queries or interactive analysis scenarios.
- Only tested the Llama-2 version of the Llama series, leaving out newer models like Llama-3.
- Uses GPT4Turbo for evaluation instead of full human review; despite a 90.5% alignment rate, a score deviation of approximately 10% still exists.

## Related Work & Insights

- **LLM Evaluation Benchmarks:** General LLM capability evaluations (such as MMLU, SuperGLUE) focus on language understanding and reasoning, whereas this work focuses on the specific application scenario of predictive analysis.
- **LLM Data Analysis:** Text2Analysis (He et al., 2023) introduces four types of data analysis queries but only evaluates code generation; Data Interpreter (Hong et al., 2024) focuses on the data science capabilities of LLM Agents but lacks a standardized evaluation protocol.
- **LLM Code Generation Evaluation:** Benchmarks like HumanEval and MBPP focus on general-purpose code generation and do not address the unique textual explanation and alignment requirements of data analysis.
- **Time-series Forecasting and LLMs:** Time-LLM (Jin et al., 2023) and others focus on time-series forecasting in specific domains, but they are limited to a single task type and lack a comprehensive multi-domain, multi-task evaluation.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)
- [\[ACL 2025\] AfroBench: How Good are Large Language Models on African Languages?](afrobench_how_good_are_large_language_models_on_african_languages.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Behavioral Analysis of Information Salience in Large Language Models](behavioral_analysis_of_information_salience_in_large_language_models.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](token_granularity_impact.md)

</div>

<!-- RELATED:END -->
