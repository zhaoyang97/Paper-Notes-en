---
title: >-
  [Paper Note] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks
description: >-
  [ACL 2026][LLM Evaluation][MCQA] This paper adopts established multiple-choice question (MCQ) quality control frameworks from education science to develop BenchMarker. Using an LLM-as-judge…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "MCQA"
  - "benchmark auditing"
  - "LLM-as-judge"
  - "writing errors"
  - "data contamination"
  - "shortcut detection"
date: 2026-05-08
content_hash: 663467817813d686
---

# BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks

**Conference**: ACL 2026  
**arXiv**: [2602.06221](https://arxiv.org/abs/2602.06221)  
**Code**: https://github.com/nbalepur/BenchMarker  
**Area**: LLM Evaluation / Benchmark Auditing  
**Keywords**: MCQA, benchmark auditing, LLM-as-judge, writing errors, data contamination, shortcut detection

## TL;DR
This paper adopts established multiple-choice question (MCQ) quality control frameworks from education science to develop BenchMarker. Using an LLM-as-judge, the tool audits 12 mainstream NLP MCQA benchmarks across three dimensions: "contamination + shortcuts + writing errors." Findings reveal that 47% of TruthfulQA questions are directly searchable online and 100% of HellaSwag questions violate multiple writing rules. Empirical evidence shows these flaws significantly inflate or deflate LLM accuracy and can even alter model rankings.

## Background & Motivation

**Background**: From MMLU to HellaSwag and SGPQA, NLP evaluation increasingly relies on MCQA due to its ease of automated scoring and similarity to human examinations. However, a survey of 39 MCQA datasets by the authors found that 23% report no quality control measures.

**Limitations of Prior Work**:
1. **Data Contamination**: Original questions appear in LLM training data or the web, allowing models to rely on "memorization" rather than "capability."
2. **Shortcuts**: Flaws in question design allow models to guess correctly by looking only at choices (ignoring the question), akin to students using "elimination strategies."
3. **Writing Errors**: Issues in grammar, structure, or distractor quality make the questions inherently misleading.

While the education sector has standardized MCQ quality control for decades (Haladyna & Downing 1989, etc.), these are rarely introduced in NLP.

**Key Challenge**: NLP aims to evaluate "understanding-recall-reasoning" via MCQA, but flawed datasets introduce noise unrelated to the target ability, damaging construct validity.

**Goal**: (a) Transfer education MCQ quality standards to NLP datasets; (b) Automate the process using LLM-as-judge; (c) Quantify the impact of each flaw type on LLM accuracy and ranking to prove the "practical" harm of defective datasets.

**Key Insight**: A cross-disciplinary fusion of education science and LLM-as-judge. Education provides 19 Item-Writing Flaws rubrics (Tarrant 2006), while LLMs provide large-scale automated scoring capabilities.

**Core Idea**: BenchMarker = LLM judge $\times$ three categories of educational metrics (Contamination / Shortcuts / Writing Errors). 8,042 human annotations were used to calibrate judge reliability, followed by a systematic audit of 12 NLP benchmarks to quantify the real-world impact of flaws.

## Method

### Overall Architecture

BenchMarker takes an MCQA dataset as input and outputs three binary labels and judge explanations for each question. The process includes:
1. **Contamination Detection**: Uses web search APIs + LLM judge to determine if a question appears fully on a webpage.
2. **Shortcut Detection**: Strong LLMs (GPT-5 / Gemini Pro / Claude) answer using only choices + reverse-engineer the question. If the inferred question $\neq$ original question $\rightarrow$ marked as a shortcut.
3. **Writing Error Detection**: Individual prompts for each of the 19 education rules (including rule name, definition, and 6 examples) are sent to the LLM judge for yes/no classification.

Outputs are aggregated to the dataset level while per-item explanations are retained for human review. It is integrated into InspectAI with a UI.

### Key Designs

1. **Contamination Detection: Web-search Proxy + Strict Matching**:
    - **Function**: Detects if an MCQ appears verbatim on public webpages (a proxy for pre-training data).
    - **Mechanism**: Queries four search engines (Google/Bing/DuckDuckGo/Brave) using the question stem $q$ + gold answer $a$. Top-K results are fed to an LLM judge to determine "complete or near-complete reproduction." Merely containing knowledge related to the answer (without MCQA format) is not considered contamination to avoid false positives for general knowledge.
    - **Design Motivation**: Accessing private LLM pre-training data is impossible; the web is a reasonable proxy. This is more accurate than simple token-overlap methods.

2. **Shortcut Detection: Partial-input + Question Deduction**:
    - **Function**: Identifies questions solvable by an LLM without the question stem.
    - **Mechanism**: Uses three models with high choices-only accuracy to (1) answer using only choices and (2) deduce the potential original question. An LLM judge evaluates if the inferred question is semantically equivalent to the truth. If all models answer correctly but the deduced question does not match $\rightarrow$ marked as a shortcut.
    - **Design Motivation**: Pure choices-only accuracy might flag "valid inference" (e.g., inferring a question about recycling from distractors). The secondary filter for question deduction isolates "meta-strategy guessing" as defined in education.

3. **Writing Errors: 19 Rubrics + Per-rule LLM Judge**:
    - **Function**: Independently judges each question against 19 rules, outputting binary labels.
    - **Mechanism**: Adapts Tarrant’s (2006) 19 rules (Clarity, Format, Give-away, Misleading categories, e.g., "avoid vague terms like 'mostly'", "distractors must be plausible"). For each rule, a prompt with definitions and 3-shot examples is provided for binary classification.
    - **Design Motivation**: Merging rules increases cognitive load for the LLM. Per-rule prompts with few-shot examples provide the most stable LLM-as-judge performance (Kim et al. 2024).

### Loss & Training
No training; purely an inference-based evaluation tool. Judge models include closed models like GPT-5 / Gemini 2.5 Pro / Claude 4.5 Sonnet using default sampling with structured JSON output.

## Key Experimental Results

### Judge Reliability (vs. 8,042 Human Annotations)

| Judge | Shortcut Acc/F1/$\kappa$ | Writing-NLP Acc/F1/$\kappa$ | Writing-Edu Acc/F1/$\kappa$ |
|---|---|---|---|
| Gemini 2.5 Pro | 0.70 / 0.69 / 0.43 | 0.82 / 0.66 / 0.53 | 0.86 / 0.39 / 0.33 |
| GPT-5 | **0.82 / 0.75 / 0.61** | 0.81 / 0.63 / 0.50 | 0.87 / 0.37 / 0.30 |
| Claude 4.5 Sonnet | 0.81 / 0.75 / 0.59 | 0.79 / 0.63 / 0.48 | 0.83 / 0.36 / 0.28 |

GPT-5 achieved $\kappa=0.61$ (substantial agreement) for shortcut detection and $\kappa=0.50$ (moderate) for NLP-domain writing errors, serving as a reliable judge.

### 12 Dataset Audits (Selected)

| Benchmark | Creation Method | Contamination | Shortcut Rate | $\ge$ 1 Writing Error |
|---|---|---|---|---|
| TruthfulQA | author-written | **47%** | Medium | High |
| HellaSwag | model-generated | Medium | Medium | **100% ($\ge$ 2 errors)** |
| ScholarIQA | – | Medium | **21%** | Medium |
| MMLU | student exams | Low | Low | Lower |
| ARC | student exams | Low | Low | Lower |

Datasets derived from human student exams (MMLU/ARC/AQuA/SAT) exhibit the highest quality. Crowdsourced (CQA/OBQA/PIQA/SIQA) and automatically generated (HellaSwag) datasets are most problematic.

### Key Findings
- **Contamination inflates accuracy**: LLM accuracy is significantly higher on contaminated subsets; models are essentially "retrieving" answers.
- **Writing errors suppress accuracy and alter rankings**: Removing questions with writing errors changes model rankings more than random subsets, meaning deployment decisions could be based on flawed data.
- **Fixes introduce new flaws**: MMLU-Pro's attempt to use LLMs to rewrite distractors introduced implausible distractors and multi-answer questions, highlighting the need for iterative automated detection tools.
- **NLP and Education share core writing issues**: Clarity and distractor quality are top issues in both domains.

## Highlights & Insights
- **Interdisciplinary leverage**: Combining decades of education science experience with modern LLM-as-judge is a powerful paradigm.
- **Three-axis diagnosis**: Unlike works focused solely on contamination, this paper proves shortcuts and writing errors are equally or more impactful on model rankings.
- **Automated repair loop**: BenchMarker transforms quality control into a repeatable pipeline, allowing "benchmark maintenance" to resemble "code maintenance" with CI.

## Limitations & Future Work
- **Limitations**: LLM judge F1 is lower on writing errors (0.39-0.66) specifically for out-of-domain student questions. Only within-item flaws are detected, not global issues like saturation or diversity.
- **Observations**: Some of the 19 rules are unsuitable for long-context questions; future work should incorporate dynamic rule selection.
- **Potential Improvements**: Incorporate distractor difficulty calibration (Item Response Theory) and extend to grading rubrics for open-ended QA.

## Related Work & Insights
- **vs. Li et al. 2024b (Contamination)**: Reuses their web-search template but scales up to 4 search engines and 3-model majority voting for robustness.
- **vs. Balepur et al. 2024b (Shortcuts)**: Adds a "question deduction" filter to distinguish "question shortcuts" from "valid inference."
- **vs. SAQUET (Writing errors)**: BenchMarker outperforms SAQUET on NLP-specific datasets and provides integration with InspectAI.

## Rating
- Novelty: ⭐⭐⭐⭐ Excellent cross-over of Education + LLM-as-judge.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 23 models, 6 search APIs, and 13 datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivations and structured framework.
- Value: ⭐⭐⭐⭐⭐ Open-source tools that expose fundamental flaws in popular benchmarks like TruthfulQA and HellaSwag.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[ACL 2026\] SPENCE: A Syntactic Probe for Detecting Contamination in NL2SQL Benchmarks](spence_a_syntactic_probe_for_detecting_contamination_in_nl2sql_benchmarks.md)
- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ICML 2026\] When AI Benchmarks Plateau: A Systematic Study of Benchmark Saturation](../../ICML2026/llm_evaluation/when_ai_benchmarks_plateau_a_systematic_study_of_benchmark_saturation.md)

</div>

<!-- RELATED:END -->
