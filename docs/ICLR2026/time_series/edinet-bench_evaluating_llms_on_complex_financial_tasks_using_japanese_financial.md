---
title: >-
  [Paper Note] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements
description: >-
  [ICLR 2026][Time Series][financial benchmark] This paper constructs EDINET-Bench, a financial benchmark derived from ten years of Japanese EDINET annual reports…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "financial benchmark"
  - "LLM evaluation"
  - "fraud detection"
  - "earnings forecasting"
  - "Japanese NLP"
date: 2026-05-08
content_hash: 09b99074dcfa2743
---

# EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements

**Conference**: ICLR 2026
**arXiv**: [2506.08762](https://arxiv.org/abs/2506.08762)  
**Code**: [GitHub](https://github.com/SakanaAI/EDINET-Bench)  
**Area**: Time Series
**Keywords**: financial benchmark, LLM evaluation, fraud detection, earnings forecasting, Japanese NLP

## TL;DR
This paper constructs EDINET-Bench, a financial benchmark derived from ten years of Japanese EDINET annual reports, comprising three expert-level tasks—accounting fraud detection, earnings forecasting, and industry classification—and finds that even state-of-the-art LLMs only marginally outperform logistic regression.

## Background & Motivation
**Background**: LLMs have surpassed human performance in mathematics and programming, with benchmark datasets serving as a key driver of progress. However, financial benchmarks remain relatively scarce, and existing ones (e.g., FinQA, ConvFinQA) are mostly simple QA or data extraction tasks.

**Limitations of Prior Work**: Existing financial benchmarks do not involve expert-level reasoning—such as integrating multiple financial statements and textual passages—and thus fail to assess LLM capability on real-world, high-stakes financial tasks.

**Key Challenge**: Although LLMs excel at general tasks, financial analysis requires simultaneously processing large volumes of tabular data and textual information while performing complex cross-year reasoning.

**Goal**: Provide the first open-source Japanese financial benchmark requiring expert-level reasoning, and in particular the first publicly available accounting fraud detection dataset.

**Key Insight**: Leveraging ten years of real annual report data from Japan's EDINET system (analogous to the U.S. EDGAR), three challenging tasks are constructed.

**Core Idea**: Real annual reports + expert-level financial tasks = revealing the inadequacy of LLMs in financial reasoning.

## Method

### Overall Architecture
Data pipeline: EDINET API → edinet2dataset tool for parsing → EDINET-Corpus (~40,000 annual reports) → three benchmark tasks.

### Key Designs
1. **edinet2dataset Tool**: Downloads annual reports via the EDINET API, parses TSV-format files at high speed using Polars, and extracts six categories of information: Meta, Summary, BS, PL, CF, and Text. Covers approximately 41,691 annual reports spanning 2014–2025.
2. **Accounting Fraud Detection**: Extracts 6,712 amended reports from revision filings; Claude 3.7 Sonnet is used to determine whether the reason for amendment involves fraud (668 confirmed as fraudulent), with manual review yielding a mislabeling rate below 5%. Non-fraudulent samples are randomly drawn from 700 companies and split by company into a training set (865) and test set (224).
3. **Earnings Forecasting**: Randomly selects 1,000 companies and constructs pairs of consecutive annual reports; the direction of change in net income attributable to parent shareholders serves as the label. A temporal split is applied (pre-2020 as training), yielding 549 training and 451 test samples.
4. **Industry Classification**: The TOPIX-33 categories based on SICC are merged into 16 broad classes, with approximately 35 companies per class, totaling 496 test samples.

### Evaluation Setup
- Zero-shot prompting: system prompt set to "You are a financial analyst"; inputs vary across combinations of annual report sections (Summary only / +BS+CF+PL / +Text).
- Models: GPT-4o, o4-mini, GPT-5, Claude 3.5 Haiku/Sonnet, Claude 3.7 Sonnet, Kimi-K2, DeepSeek-V3/R1, Llama 3.3 70B.
- Classical baselines: Logistic Regression, Random Forest, XGBoost.

## Key Experimental Results

### Main Results
Fraud Detection ROC-AUC (selected):

| Model | Summary | +BS/CF/PL | +Text |
|-------|---------|-----------|-------|
| Claude 3.5 Sonnet | 0.64 | 0.63 | **0.73** |
| GPT-5 | 0.56 | 0.62 | 0.67 |
| Logistic Regression† | - | 0.61 | - |

Earnings Forecasting ROC-AUC:

| Model | Summary | +BS/CF/PL | +Text |
|-------|---------|-----------|-------|
| GPT-5 | 0.58 | 0.62 | **0.65** |
| Claude 3.7 Sonnet | 0.55 | 0.58 | 0.61 |
| Logistic Regression† | - | 0.60 | - |

### Ablation Study
Ablation on input information:

| Input Configuration | Fraud Detection (avg) | Earnings Forecasting (avg) |
|--------------------|-----------------------|---------------------------|
| Summary only | ~0.58 | ~0.48 |
| +BS/CF/PL | ~0.59 | ~0.52 |
| +Text | ~0.64 | ~0.52 |

### Key Findings
- **LLMs only marginally outperform logistic regression**: On binary classification tasks, even the strongest LLM achieves MCC values of only 0.1–0.3.
- **Textual information is beneficial**: Adding the Text section improves fraud detection ROC-AUC by ~0.06 on average.
- **Open-source models lag behind**: DeepSeek-V3/R1 performs notably worse than closed-source models on financial tasks.
- **Industry classification is relatively easier**: With complete financial statements, Claude 3.5 Sonnet achieves 41% accuracy (random baseline: 6.25% for 16 classes).
- Each annual report contains approximately 30K tokens; single-inference cost is approximately $0.1 (Claude 3.7 Sonnet).

## Highlights & Insights
- **First open-source accounting fraud detection dataset**: No publicly available fraud detection evaluation benchmark previously existed.
- **Open-source edinet2dataset tool**: Provides a complete pipeline for constructing financial datasets from EDINET, with high-speed TSV parsing based on Polars.
- **Honest conclusions**: The paper explicitly states that providing annual reports for direct LLM inference is insufficient, and that additional scaffolding—such as simulation environments and task-specific reasoning support—is required.
- **Cross-linguistic value**: The Japanese financial benchmark fills a gap in non-English financial NLP.
- **Rigorous experimental design**: Ablation across multiple input configurations, comparison against classical ML baselines, and transparent cost analysis.
- **Label quality control**: Fraud labels are generated by Claude and verified through manual review, with a mislabeling rate below 5%.

## Limitations & Future Work
- Only zero-shot settings are evaluated; few-shot and RAG experiments are absent, and reasoning augmentation strategies such as chain-of-thought are unexplored.
- Fraud labels are generated by Claude 3.7 Sonnet rather than through full human annotation, potentially introducing systematic bias.
- Most evaluated LLMs have limited understanding of Japanese financial terminology, particularly open-source models.
- The data covers only the Japanese market, and cross-national generalizability is not assessed.
- No in-depth analysis of LLM reasoning processes is provided (e.g., which financial statement items are attended to, visualization of reasoning paths).
- Fine-tuned Llama-3.2-1B results are incomplete, and exploration of small-model fine-tuning is insufficient.
- Both fraud detection and earnings forecasting are binary classification tasks; finer-grained regression formulations are not explored.
- Annual reports of approximately 30K tokens approach the context length limits of some models, which may affect results.

## Related Work & Insights
- Compared to FinQA/ConvFinQA: EDINET-Bench requires processing complete annual reports rather than short passages, more closely approximating real-world financial analysis scenarios.
- Compared to FinanceBench: FinanceBench is an open-ended QA benchmark, whereas EDINET-Bench requires integrating multiple financial statements and textual content for expert-level reasoning.
- Compared to FAMMA: FAMMA is based on CFA exams and tutorials, while EDINET-Bench is grounded in real corporate annual reports.
- Compared to kim2024 (GPT-4 for earnings direction prediction): This paper provides open-source data and evaluation code for reproducibility.
- Insight: Financial LLMs need to move beyond simple QA toward agent-based approaches that simulate financial analyst workflows.
- Implications for non-English financial NLP: Countries can adopt similar approaches to construct local financial benchmarks (e.g., China CSRC disclosures, U.S. EDGAR).
- Future direction: Integrating RAG or multi-agent frameworks may substantially improve LLM financial reasoning performance.

## Rating
- Novelty: ⭐⭐⭐⭐ First open-source fraud detection benchmark, though the task designs themselves are relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10+ models and 3 input configurations, but lacks advanced experiments such as few-shot evaluation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed data construction procedures and rich tables.
- Value: ⭐⭐⭐⭐ The open-source tools and datasets make practical contributions to the financial NLP community and expose the limitations of LLMs in financial reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[ACL 2026\] A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting](../../ACL2026/time_series/a_unified_framework_for_modeling_heterogeneous_financial_data_via_dual-granulari.md)
- [\[ICLR 2026\] Benchmarking ECG FMs: A Reality Check Across Clinical Tasks](benchmarking_ecg_fms_a_reality_check_across_clinical_tasks.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](scits_scientific_time_series_llm.md)

</div>

<!-- RELATED:END -->
