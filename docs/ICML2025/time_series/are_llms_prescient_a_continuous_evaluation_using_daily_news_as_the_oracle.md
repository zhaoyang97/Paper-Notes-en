---
title: >-
  [Paper Note] Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle
description: >-
  [ICML 2025][Time Series][LLM evaluation] This paper proposes Daily Oracle, a continuous evaluation benchmark that automatically generates predictive QA pairs from daily news. It systematically reveals a smooth decay in LLM predictive performance as pre-training data becomes outdated, showing an average accuracy drop of 21.55% on True/False (TF) questions and 11.33% on Multiple Choice (MC) questions, which cannot be fully mitigated even with RAG.
tags:
  - "ICML 2025"
  - "Time Series"
  - "LLM evaluation"
  - "continuous benchmark"
  - "news forecasting"
  - "temporal generalization"
  - "knowledge cutoff"
date: 2026-05-08
content_hash: 9eeaa61aa03971a3
---

# Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle

**Conference**: ICML 2025  
**arXiv**: [2411.08324](https://arxiv.org/abs/2411.08324)  
**Code**: [https://agenticlearning.ai/daily-oracle](https://agenticlearning.ai/daily-oracle)  
**Area**: Time Series / LLM Evaluation  
**Keywords**: LLM evaluation, continuous benchmark, news forecasting, temporal generalization, knowledge cutoff

## TL;DR

This paper proposes Daily Oracle, a continuous evaluation benchmark that automatically generates predictive QA pairs from daily news. It systematically reveals a smooth decay in LLM predictive performance as pre-training data becomes outdated, showing an average accuracy drop of 21.55% on True/False (TF) questions and 11.33% on Multiple Choice (MC) questions, which cannot be fully mitigated even with RAG.

## Background & Motivation

**Background**: Existing LLM evaluation benchmarks (such as MMLU, HumanEval) are static, one-off question sets that remain fixed once released. A few dynamic benchmarks (such as RealTimeQA, FreshQA) incorporate update mechanisms, but they are updated infrequently and do not focus on predictive capabilities.

**Limitations of Prior Work**: Static benchmarks face two fundamental problems: (1) as LLMs are continuously updated, benchmark content may leak into the training data, leading to inflated evaluation results; (2) they lack a temporal dimension, making it impossible to track the trajectory of model performance changes over time—we do not know how long the "shelf life" of an LLM is.

**Key Challenge**: The world is constantly changing, but evaluation benchmarks remain static. Existing forecasting datasets (e.g., ForecastQA, AutoCast) are either small in scale, not updated continuously, or rely on human annotation, making them difficult to scale up. No single benchmark concurrently satisfies the requirements of both "continuous updates" and "predictive capability evaluation."

**Goal**: Construct an "always-up-to-date" continuous evaluation framework to answer the core question: How does the predictive capability of LLMs degrade after their knowledge cutoff date? Can RAG rescue outdated knowledge?

**Key Insight**: Leverage the natural temporal characteristics of daily news—today's news is yesterday's "future events." Predictive questions are automatically generated from news articles, with the publication date of the news naturally serving as the verification timestamp for the answer.

**Core Idea**: Utilize automatically generated predictive QA pairs from daily news as continuous evaluation probes to track the degradation curve of LLM predictive capability along the timeline.

## Method

### Overall Architecture

The pipeline of Daily Oracle is as follows: crawl news articles daily $\rightarrow$ automatically generate predictive True/False and Multiple Choice QA pairs using LLMs $\rightarrow$ filter low-quality questions based on 7 rules $\rightarrow$ verify answers with subsequent factual news outcomes $\rightarrow$ continuously evaluate LLMs along the timeline from January 2020 to December 2024. The current dataset contains 16,783 TF questions and 14,727 MC questions, averaging 17.2 questions per day.

### Key Designs

1. **Four-Step QA Generation Pipeline**:

    - **Function**: Automatically generate high-quality predictive QA pairs from daily news articles.
    - **Mechanism**: (1) Article Summary—Extract summaries of new events from news using an LLM, filtering out opinion/editorial articles; (2) QA Generation—Generate 2 TF and 2 MC questions for each article, forcing one true and one false for TF questions to ensure balance; (3) Misleading Choices—Generate 3 highly distracting incorrect options for MC questions; (4) QA Filtering—Score questions with 0/1/2 points against 7 guidelines (answer correctness, non-answerability beforehand, no information leakage, objectivity, inclusion of temporal elements, public interest, and non-obviousness). Only questions with a total score $\ge 13$ are retained.
    - **Design Motivation**: Each step is specifically tailored to the unique requirements of predictive QA. The summarization step ensures the questions concern new events (making them suitable as prediction tasks), and the filtering step ensures that the questions are truly unanswerable before the publication date. Human evaluation indicates an 85% agreement rate between LLM filtering and human consensus.

2. **Three Evaluation Settings (Closed/Open/Gold)**:

    - **Function**: Systematically isolate the impacts of "outdated knowledge" versus "insufficient reasoning capability."
    - **Mechanism**: Closed-Book (purely evaluates the model's internal knowledge); Constrained Open-Book (uses BM25 to retrieve at most top-5 news articles prior to a specific cutoff date as the RAG context, with each article truncated to 512 words), which introduces the concept of RAG Cutoff to limit the latest date of accessible information; Gold Article (directly provides the source article of the generated question), which reduces the task to a reading comprehension test to verify the answerability of the questions.
    - **Design Motivation**: These three settings form a gradient that progressively increases the available information. Closed-Book exposes knowledge obsolescence, Open-Book tests the compensation effect of RAG, and Gold Article verifies the quality of the dataset. The design of the RAG Cutoff is particularly ingenious, preventing the model from directly retrieving the final answers.

3. **Temporal Degradation Analysis Framework**:

    - **Function**: Quantify the temporal patterns of LLM performance degradation.
    - **Mechanism**: Calculate accuracy on a monthly basis and plot 5-month moving average curves; compute average accuracy and Year-over-Year (YoY) change rates annually; specifically distinguish the degradation rates before and after the knowledge cutoff (Pre-Cutoff vs. Post-Cutoff).
    - **Design Motivation**: Distinguishing Pre-Cutoff vs. Post-Cutoff is critical. If degradation occurs only after the Cutoff, it indicates that the issue is purely of outdated knowledge. If degradation also occurs before the Cutoff, it suggests deeper temporal generalization issues.

### Loss & Training

This is a pure evaluation work and does not involve model training. The evaluation metric is accuracy, where the random baseline is 50% for TF questions and 25% for MC (1-out-of-4) questions.

## Key Experimental Results

### Main Results Table

| Model | Knowledge Cutoff | 2020 TF/MC | 2024 TF/MC | Avg YoY (TF) | Avg YoY (MC) |
|------|---------|-----------|-----------|------------|------------|
| Claude-3.5-Sonnet | 2024.4 | 81.2/76.9 | 64.3/61.8 | -5.58% | -5.03% |
| GPT-4 | 2023.4 | 69.7/70.6 | 56.9/51.6 | -4.75% | -7.04% |
| GPT-3.5 | 2021.9 | 62.9/50.3 | 56.1/43.1 | -2.84% | -3.08% |
| Llama-3-8B | 2023.3 | 65.1/52.4 | 57.0/47.0 | -2.97% | -2.30% |
| Mixtral-8x7B | Unknown | 57.8/57.4 | 36.0/46.3 | -10.78% | -4.68% |
| Gemma-2-2B | 2024.7 | 58.7/47.9 | 55.8/43.3 | -1.04% | -1.98% |

Overall average: TF dropped from 64.68% to 50.74% (-21.55%), MC dropped from 58.30% to 51.69% (-11.33%).

### Ablation Study Table (Pre/Post Knowledge Cutoff Degradation Comparison)

| Model | Pre-Cutoff YoY (TF) | Post-Cutoff YoY (TF) | Pre-Cutoff YoY (MC) | Post-Cutoff YoY (MC) |
|------|-------------------|--------------------|--------------------|---------------------|
| Claude-3.5-Sonnet | -4.77% | -12.41% | -6.26% | -11.78% |
| GPT-4 | -5.83% | -1.96% | -4.23% | **-18.54%** |
| GPT-3.5 | -4.33% | -3.43% | +0.14% | -0.31% |
| Llama-3-8B | -1.95% | -6.50% | -2.21% | -1.25% |
| Gemma-2-2B | -1.41% | -3.68% | -4.46% | -4.07% |

### Key Findings

- **Degradation is smooth and continuous**: Rather than falling off a cliff abruptly at the knowledge cutoff date, performance degrades progressively, suggesting that world knowledge gradually "expires" within the model.
- **Post-Cutoff degradation accelerates significantly**: GPT-4's Post-Cutoff YoY on MC questions reaches -18.54%, which is 4.4 times its Pre-Cutoff YoY (-4.23%).
- **RAG improves performance but does not eliminate degradation**: Accuracy recovers somewhat in the open-book setting, but the downward trend persists. Outdated RAG context may even drag down performance (Llama-3-8B performs worse under an older RAG Cutoff than in the closed-book setting).
- **Gemma-2-2B is the most stable**: Benefiting from its late knowledge cutoff (July 2024), its YoY change is the smallest.
- **Mistral/Mixtral fall below the random baseline on TF**: This is mainly due to the model refusing to answer ("I cannot predict the future"), leading to a significant increase in refusal rates.

## Highlights & Insights

- **An "always-up-to-date" evaluation paradigm**: An automated QA generation framework based on daily news structurally guarantees that the benchmark will never become outdated. This evaluation methodology can be generalized to other domains.
- The **smoothness of degradation** is surprising—it suggests that the LLM's "forgetting" of world knowledge is not a binary cutoff switch, but rather a continuous temporal decay.
- **RAG is not a silver bullet**: Even when provided with the latest information, models still need to "comprehend temporal context" to make correct predictions, which yields important insights for the design of RAG systems.

## Limitations & Future Work

- QA generation itself relies on GPT-4/GPT-3.5, which may introduce specific biases (e.g., certain types of questions are easier to generate).
- It only covers the news domain (politics, business, sports, etc.). The temporal generalization characteristics in domains like scientific discoveries and technical breakthroughs might differ.
- It cannot precisely disentangle "outdated knowledge" from "insufficient reasoning capability"—even in the Gold Article setting, some questions might still be answered incorrectly due to reasoning difficulties.
- Although the dataset scale (31.5K questions) is the largest of its kind, the number of samples in certain fine-grained categories (e.g., Healthcare) might be insufficient.

## Related Work & Insights

- **vs ForecastBench**: ForecastBench updates 1,000 questions bi-weekly and relies on human submissions. Daily Oracle automatically generates questions daily, is 30 times larger in scale, and requires no human intervention.
- **vs RealTimeQA**: RealTimeQA evaluates whether models have acquired the latest knowledge (search-engine style), whereas Daily Oracle evaluates whether models can predict the future (forecasting style). Thus, they fundamentally test different capabilities.
- **vs FreshQA**: FreshQA focuses on factual questions whose answers change over time, whereas Daily Oracle focuses on predicting future events—the former tests "whether knowledge is updated", while the latter tests "the capability of temporal extrapolation."

## Rating

- Novelty: ⭐⭐⭐⭐ The continuous evaluation paradigm and the "daily updating" benchmark design possess unique value.
- Experimental Thoroughness: ⭐⭐⭐⭐ A comprehensive analysis spanning a 5-year timeline, 8 models, and 3 evaluation settings.
- Writing Quality: ⭐⭐⭐⭐ Clear figures and charts, with rigorous analytical logic.
- Value: ⭐⭐⭐⭐ Systematically reveals the patterns of temporal generalization degradation in LLMs, providing practical guidance for model update strategies and RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](../../NeurIPS2025/time_series/nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)
- [\[ICML 2026\] Embedding Hybrid Systems into Continuous Latent Vector Fields](../../ICML2026/time_series/embedding_hybrid_systems_into_continuous_latent_vector_fields.md)
- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](../../ICLR2026/time_series/scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](../../ICLR2026/time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ACL 2025\] Revisiting LLMs as Zero-Shot Time-Series Forecasters: Small Noise Can Break Large Models](../../ACL2025/time_series/revisiting_llms_as_zero-shot_time_series_forecasters_small_noise_can_break_large.md)

</div>

<!-- RELATED:END -->
