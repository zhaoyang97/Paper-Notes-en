---
title: >-
  [Paper Note] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models
description: >-
  [ACL 2025][LLM Safety][Chinese Benchmark] Proposes Chinese SimpleQA, the first comprehensive Chinese factuality evaluation benchmark, containing 3,000 high-quality short Q&A pairs (covering 6 main domains and 99 sub-domains). After evaluating 41 LLMs, only o1-preview (63.8%) and Doubao-pro-32k (61.9%) passed. The study systematically reveals key insights such as "larger models perform better," "RAG narrows the gap," and "alignment lowers factuality."
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Chinese Benchmark"
  - "Factuality Evaluation"
  - "SimpleQA"
  - "Knowledge Boundary"
  - "RAG"
  - "Alignment Tax"
  - "Calibration"
date: 2026-05-08
content_hash: 82b1be0f6e764987
---

# Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2411.07140](https://arxiv.org/abs/2411.07140)  
**Code**: [https://openstellarteam.github.io/ChineseSimpleQA/](https://openstellarteam.github.io/ChineseSimpleQA/)  
**Authors**: Yancheng He, Shilong Li, Jiaheng Liu et al.  
**Institution**: Alibaba Taobao and Tmall Group  
**Area**: LLM Evaluation / Factuality  
**Keywords**: Chinese Benchmark, Factuality Evaluation, SimpleQA, Knowledge Boundary, RAG, Alignment Tax, Calibration

## TL;DR

Proposes Chinese SimpleQA, the first comprehensive Chinese factuality evaluation benchmark, containing 3,000 high-quality short Q&A pairs (covering 6 main domains and 99 sub-domains). After evaluating 41 LLMs, only o1-preview (63.8%) and Doubao-pro-32k (61.9%) passed. The study systematically reveals key insights such as "larger models perform better," "RAG narrows the gap," and "alignment lowers factuality."

## Background & Motivation

**Background**:
   - LLM-generated factually inconsistent content (hallucination) severely hinders the widespread application of Artificial General Intelligence (AGI).
   - OpenAI's SimpleQA benchmark provides a concise and reliable tool for English factuality evaluation, but is primarily designed for English.
   - Existing Chinese LLM benchmarks (C-Eval, CMMLU) mainly test reasoning capabilities and do not specifically evaluate the boundaries of Chinese factual knowledge.

**Core Problem**:
   - Lack of a factuality evaluation benchmark focused on the Chinese language.
   - Significant discrepancy between LLM performance in Chinese knowledge domains (especially Chinese culture-related knowledge) and English.
   - Systematic validation is lacking regarding whether alignment training (RLHF/DPO, etc.) reduces model factuality.

**Goal**: Construct a Chinese, diverse, high-quality, static, and easy-to-evaluate factuality benchmark to comprehensively evaluate the boundaries of existing LLMs on Chinese knowledge.

## Method

### Dataset Design Principles

Chinese SimpleQA follows five core principles:

1. **Chinese**: Focus on Chinese language knowledge evaluation.
2. **Diverse**: 6 main domains + 99 sub-domains.
3. **High-quality**: Strict automatic + manual quality control workflow.
4. **Static**: All answers remain invariant over time (evergreen attribute).
5. **Easy-to-evaluate**: Questions and answers are extremely short, enabling fast scoring via LLM APIs.

### Six Main Domains Covered

| Domain | Sample count | Example |
|------|--------|------|
| Chinese Culture | 323 | Cultural knowledge unique to Chinese |
| Humanities | 623 | History, philosophy, language, etc. |
| Engineering, Technology, and Applied Sciences (ETAS) | 473 | Computer science, engineering, medicine, etc. |
| Life, Arts, and Culture (LAC) | 602 | Daily life, art, sports, etc. |
| Society | 450 | Politics, economics, law, etc. |
| Natural Science | 529 | Physics, chemistry, biology, etc. |

### Data Construction Pipeline

**Automatic Phase**:
1. **Knowledge Content Extraction**: Extract high-quality content from knowledge-rich texts such as Wikipedia.
2. **Q&A Pair Generation**: Use LLMs to auto-generate Q&A pairs based on high-quality knowledge content.
3. **LLM Quality Verification**: Automatically filter samples according to predefined criteria (uniqueness of answer, static nature, etc.).
4. **RAG Verification**: Leverage LlamaIndex + Google/Bing search engines for retrieval-augmented verification.
5. **Difficulty Filtering**: Exclude questions if all four powerful models (GPT-4o, Llama3-70B, Qwen2.5-72B, GLM-4-Plus) answer correctly.

**Manual Verification Phase**:
- Each question is **evaluated independently by 2 annotators**.
- Annotators search for answers using search engines, with each providing at least 2 supporting URLs.
- In case of disagreement, a **3rd annotator arbitrates**.
- Only samples consistent with LLM-verified answers are ultimately kept.

### Q&A Construction Criteria (Four Core Rules)

1. **The answer must be objective and unique**: Exclude subjective or multi-answer questions (e.g., "In which year did Zhu Qizhen ascend the throne?" has two valid answers).
2. **The answer must not change over time**: Exclude current events or temporal questions (e.g., "The current president of country X").
3. **The question must be challenging**: Avoid overly simple questions.
4. **Answerable up to 2023**: Ensure fair evaluation of models trained with different knowledge cutoff dates.

### Dataset Size Transition

| Phase | Sample Count | Retention Rate |
|------|--------|----------|
| Initial Generation | 10,000 | 100% |
| After Difficulty Filtering | 6,310 | 63.1% |
| After Rules & RAG Verification | 3,470 | 34.7% |
| After Manual Verification | **3,000** | **30.0%** |

### Evaluation Metrics

- **Correct (CO)**: The predicted answer completely contains the reference answer and has no contradictions.
- **Not Attempted (NA)**: The model did not provide a reference answer and had no contradictions (e.g., a refusal to answer).
- **Incorrect (IN)**: The predicted answer contradicts the reference answer.
- **Correct Given Attempted (CGA)**: The ratio of correct answers among attempted questions.
- **F-score**: The harmonic mean of CO and CGA.

### Dataset Statistics

| Metric | Value |
|--------|------|
| Average Question Length | 23.6 characters |
| Average Answer Length | 6.1 characters |
| Max Question Length | 81 characters |
| Max Answer Length | 47 characters |

## Key Experimental Results

### Evaluation Scale

Evaluated **41 LLMs**: 17 closed-source + 24 open-source, covering series such as o1, GPT-4o, Qwen2.5, InternLM, Yi, LLaMA3, DeepSeek, Baichuan, and Mistral.

### Main Results (Overall Ranking, Partial Presentation)

| Model | CO↑ | NA | IN↓ | CGA | F-score |
|------|-----|-----|-----|-----|---------|
| o1-preview | **63.8** | 12.2 | 24.0 | 72.7 | **67.9** |
| Doubao-pro-32k | 61.9 | 10.3 | 27.8 | 69.1 | 65.3 |
| GLM-4-Plus | 58.7 | 7.4 | 33.9 | 63.4 | 60.9 |
| GPT-4o | 59.3 | 1.4 | 39.3 | 60.1 | 59.7 |
| Qwen-Max | 54.1 | 11.3 | 34.6 | 61.0 | 57.4 |
| Qwen2.5-72B (Open) | 48.4 | 7.1 | 44.5 | 52.1 | 50.2 |
| DeepSeek-67B | 43.5 | 14.8 | 41.7 | 51.1 | 47.0 |
| LLaMA3.1-70B | 38.3 | 9.4 | 52.3 | 42.3 | 40.2 |
| GPT-3.5 | 29.7 | 2.9 | 67.4 | 30.6 | 30.1 |

**Benchmark Difficulty Verification**: Only o1-preview and Doubao-pro-32k surpassed 60% (passing grade).

### Key Findings 1: Larger Models Perform Better

Scaling effect of the Qwen2.5 series:

| Model | CO |
|------|-----|
| Qwen2.5-72B | 48.4% |
| Qwen2.5-32B | 38.8% |
| Qwen2.5-14B | 35.4% |
| Qwen2.5-7B | 26.6% |
| Qwen2.5-3B | 16.2% |
| Qwen2.5-1.5B | 11.1% |

From 1.5B to 72B, the CO accuracy increases from 11.1% to 48.4%, showing near-linear growth.

### Key Findings 2: Chinese Models Have a Significant Advantage on Chinese Culture Domain

F-score on the "Chinese Culture" subdomain:

| Model | Chinese Culture | Overall |
|------|----------------|------|
| Doubao-pro-32k | **61.8** | 65.3 |
| GLM-4-Plus | 56.5 | 60.9 |
| DeepSeek-V2.5 | 50.4 | 55.7 |
| o1-preview | 45.7 | **67.9** |
| GPT-4o | 39.4 | 59.7 |

Doubao-pro-32k and GLM-4-Plus significantly lead o1-preview on Chinese Culture (+16 / +11 percentage points), despite their lower overall ranking.

### Key Findings 3: RAG Drastically Narrows the Model Performance Gap

Performance changes after introducing RAG:

| Model Comparison | Gap without RAG | Gap with RAG |
|----------|-----------|-----------|
| GPT-4o vs Qwen2.5-3B | 42.4% | **9.3%** |

RAG strategies benefit weaker models more, drastically narrowing the performance gap between models of different sizes.

### Key Findings 4: The Existence of Alignment Tax

Alignment and post-training strategies typically degrade model factuality performance—models may sacrifice knowledge accuracy in exchange for safety and helpfulness during the alignment process.

### Key Findings 5: Larger Models Exhibit Better Calibration

| Model | CO | NA | Interpretation |
|------|-----|-----|------|
| o1-preview | 63.8 | 12.2 | Refuses to answer when uncertain |
| o1-mini | 39.5 | 20.6 | Refuses more but also makes more mistakes |
| GPT-4o | 59.3 | 1.4 | Rarely refuses |
| GPT-4o-mini | 37.6 | 0.9 | Does not refuse either, but makes more mistakes |
| Claude-3.5-Sonnet | 46.2 | **27.4** | The most cautious model |

Claude-3.5-Sonnet has the highest refusal rate (27.4%), which, however, avoids a large number of incorrect answers.

### SimpleQA vs Chinese SimpleQA Ranking Discrepancies

Rankings on the English SimpleQA and the Chinese SimpleQA are inconsistent: models focusing on Chinese (such as Doubao, GLM-4-Plus) see a significant rank boost on the Chinese version, indicating that English and Chinese knowledge evaluations are mutually irreplaceable.

## Highlights & Insights

1. **First Systematic Chinese Factuality Benchmark**: Fills the gap in Chinese LLM factuality evaluation and complements OpenAI's SimpleQA.
2. **Strict Data Quality Assurance**: Only 30% of original data is retained after three rounds of filtering (automatic LLM, RAG verification, and dual-annotator manual audit), ensuring high quality.
3. **Comprehensive Model Evaluation Ecosystem**: Evaluated 41 models (closed-source and open-source, scaled from 0.5B to 671B), delivering the most comprehensive profiling of Chinese factuality capabilities to date.
4. **Systematic Validation of Alignment Tax**: Confirms for the first time on a Chinese factuality benchmark that alignment training can degrade factuality accuracy, offering insights for post-training strategy design.
5. **Discovery of RAG's Equalization Effect**: RAG narrows the inter-model gap from 42.4% to 9.3%, which has significant practical guidance for resource-constrained scenarios (where only small models can be deployed).
6. **Difference in Chinese Cultural Knowledge**: Reveals the shortcomings of internationalized LLMs (GPT, o1) in the domain of Chinese culture, illustrating the critical impact of data sources on knowledge coverage.

## Limitations & Future Work

1. **Low Evaluation Cost but High Construction Cost**: The manual quality control involving dual-annotator labeling and a third-annotator arbitration incurs high costs, making rapid expansion difficult.
2. **Temporal Cutoff Limitation**: All questions must be answerable prior to the end of 2023, making it impossible to evaluate the models' grasp of newer knowledge.
3. **Uneven Domain Coverage**: Chinese Culture (323 questions) is significantly fewer than Life, Arts, and Culture (602 questions), potentially underestimating the evaluation depth of Chinese cultural knowledge.
4. **Evaluates Factuality Only**: Does not cover other dimensions like reasoning or creative writing, thus cannot comprehensively evaluate LLMs.
5. **Scoring Relies on OpenAI API**: Uses LLM-as-a-Judge for automated scoring; the accuracy of the evaluator itself has not been fully verified.
6. **The Double-Edged Sword of Static Design**: Although time-invariant answers guarantee benchmark stability, they also mean that the benchmark cannot capture LLMs' understanding of dynamic world knowledge.

## Related Work & Insights

- **Factuality Benchmarks**: SimpleQA (Wei et al., 2024), TruthfulQA, FreshQA
- **Chinese LLM Benchmarks**: C-Eval (Huang et al., 2023), CMMLU (Li et al., 2023), WebQA (Li et al., 2016)
- **General Evaluation**: MMLU (Hendrycks et al., 2021), GSM8K (Cobbe et al., 2021), AlpacaEval
- **LLM-as-a-Judge**: MT-Bench (Zheng et al., 2023), Arena-Hard (Li et al., 2024)

## Rating

⭐⭐⭐⭐⭐ — An important work filling the gap in Chinese factuality evaluation. Possesses extremely high data quality, comprehensive evaluation coverage, and profound practical value in its discoveries (alignment tax, RAG equalization effect, Chinese cultural differences). It is a must-read benchmark for Chinese LLM developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)
- [\[ACL 2026\] Permutation-Consensus Listwise Judging for Robust Factuality Evaluation](../../ACL2026/llm_safety/permutation-consensus_listwise_judging_for_robust_factuality_evaluation.md)
- [\[ACL 2025\] Ewe: Improving Factuality with Explicit Working Memory](improving_factuality_with_explicit_working_memory.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
