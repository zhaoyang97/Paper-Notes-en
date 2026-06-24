---
title: >-
  [Paper Note] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset
description: >-
  [ACL 2025][Multilingual & Machine Translation][Financial Meetings] This work introduces M3FinMeeting, the first multilingual (Chinese, English, Japanese), multi-sector, and multi-task evaluation benchmark for financial meetings. Containing 600 real-world financial meetings with three tasks—summarization, Q&A pair extraction, and question answering—it reveals that state-of-the-art LLMs still have significant room for improvement in understanding financial meetings.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Financial Meetings"
  - "Multilingual Benchmark"
  - "Long Context Understanding"
  - "Summarization"
  - "Question Answering"
date: 2026-05-08
content_hash: c4b936e279bfadc0
---

# M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset

**Conference**: ACL 2025  
**arXiv**: [2506.02510](https://arxiv.org/abs/2506.02510)  
**Code**: [Available](https://github.com/aliyun/qwen-dianjin)  
**Area**: Multilingual Translation  
**Keywords**: Financial Meetings, Multilingual Benchmark, Long Context Understanding, Summarization, Question Answering

## TL;DR

This work introduces M3FinMeeting, the first multilingual (Chinese, English, Japanese), multi-sector, and multi-task evaluation benchmark for financial meetings. Containing 600 real-world financial meetings with three tasks—summarization, Q&A pair extraction, and question answering—it reveals that state-of-the-art LLMs still have significant room for improvement in understanding financial meetings.

## Background & Motivation

Existing financial NLP benchmarks (such as FinQA, ConvFinQA, CFLUE, etc.) suffer from three major limitations:

**Single Data Source**: They heavily rely on news articles, financial reports, and announcements, lacking real financial meeting content. Financial meetings possess unique characteristics such as conversational nature, real-time dynamics, and strategic discussions, which existing datasets fail to cover.

**Monolingual Nature**: They are almost exclusively limited to either English or Chinese.

**Lack of Long-Context Challenges**: Financial meetings typically last 1-2 hours, with transcripts often exceeding 10K tokens, presenting a real test for the long-context capabilities of LLMs.

M3FinMeeting aims to fill these gaps and evaluate the comprehensive capabilities of LLMs in understanding real-world financial meetings.

## Method

### Overall Architecture

M3FinMeeting is an evaluation benchmark dataset whose core design revolves around "three multis":

- **Multilingual**: English (100 meetings), Chinese (400 meetings), and Japanese (100 meetings), totaling 600 meetings.
- **Multi-sector**: Covering all 11 sectors under the GICS standard (Communication, IT, Financials, Energy, etc.).
- **Multi-task**: Three tasks consisting of summarization generation, Q&A pair extraction, and question answering.

### Key Designs

#### 1. Data Collection and Annotation

**Function**: Real meeting audio was obtained from partner financial institutions, transcribed via ASR, and then manually corrected.

**Core Pipeline**:
- Collection Criteria: Timeliness (recent meetings), length (prioritizing long audio), sector coverage, and authority.
- Whisper was used for speech-to-text transcription, followed by paragraph-by-paragraph manual correction by annotators.
- Each meeting lasts approximately 1 hour on average, with English averaging 10,086 tokens, Chinese about 11,740 tokens, and Japanese about 13,284 tokens.
- Sensitive and personally identifiable information (PII) was strictly removed.

**Design Motivation**: Direct utilization of real-world financial meetings instead of synthetic data to ensure that the benchmark reflects real-world challenges.

#### 2. Three Evaluation Tasks

**Summarization Generation**: LLMs are required to implicitly identify different topical segments in the document, generate a summary for each segment, and then concatenate them into a complete document summary. Evaluation employs segment-level P/R/F1 (aligned based on cosine similarity $\ge 0.75$) + GPT-4-Judge scoring (covering five dimensions: coverage, redundancy, readability, accuracy, and consistency, from 0 to 100 points).

**Q&A Pair Extraction**: Identifying meaningful questions and their corresponding answers from the full meeting transcript. This requires the LLM to understand the conversational structure, distinguish between meaningful questions and meaningless interruptions, and correctly pair multi-turn Q&A.

**Question Answering**: Given the full meeting transcript and a set of preset questions, LLMs must locate evidence within the long context and generate answers. Merging related questions into a single prompt simulates practical scenarios like writing reports or summaries.

#### 3. Evaluation System

**Function**: Multi-level evaluation that balances both automated metrics and human judgment.

- Segment-level Precision/Recall/F1: Aligning generated and reference summaries based on embedding similarity.
- GPT-4-Judge: 0-100 scoring across five dimensions, cross-validated with Qwen-plus-Judge.
- Human Evaluation + Fleiss' Kappa: Validating the alignment between LLM evaluation and human judgment.

### Loss & Training

This work introduces an evaluation benchmark and does not involve model training. Evaluations are conducted in a zero-shot setting.

## Key Experimental Results

### Main Results

**Comprehensive Evaluation on Three Tasks (GPT-4-Judge Scores)**:

| Model | Summarization | Q&A Pair Extraction | Question Answering | Overall |
|------|------|-----------|------|------|
| GPT-3.5-turbo | 44.56 | 31.13 | 42.78 | 39.55 |
| LLaMA3.1-8B | 52.01 | 44.64 | 40.01 | 45.76 |
| GLM4-9B-Chat | 67.71 | 46.06 | 67.72 | 60.76 |
| Qwen2-7B | 73.59 | 37.33 | 69.99 | 60.71 |
| GPT-4o | 73.61 | 66.85 | 71.79 | 70.66 |
| Qwen2-72B | 74.17 | 60.85 | 73.50 | 69.66 |
| **Qwen2.5-72B** | **74.51** | **68.03** | **74.81** | **72.54** |

**Extremely low F1 scores for Q&A pair extraction**: Even the best model, Qwen2.5-72B, only achieves 38.41% on F1 metric, indicating that automated extraction of high-quality Q&A pairs from long conversations remains extremely challenging.

### Ablation Study

**Impact of RAG on Q&A Performance (Qwen2.5-72B, GPT-4-Judge)**:

| Method | <5K | 5-10K | 10-15K | 15-20K | >20K |
|------|------|------|------|------|------|
| Baseline 1 (All-in-one Answer) | Medium | High | High | Best | Best |
| Baseline 2 (One-by-one Answer) | Medium | Medium | High | Second Best | Second Best |
| RAG (top 5) | Good | Good | Medium | Poor | Poor |
| RAG (top 1) | Poor | Poor | Poor | Worst | Worst |

Key Finding: **On long documents ($>10\text{K}$ tokens), utilizing the full context outperforms RAG**, which is counter-intuitive.

### Key Findings

1. **Qwen2.5-72B achieves the best overall performance**, but still scores only 72.54 points (out of 100), indicating significant room for improvement.
2. **Summarization Task**: Segment-level F1 is below 30%, showing that LLMs perform poorly in implicit document segmentation.
3. **Q&A Extraction is the Most Difficult**: Even the best model has a recall rate of less than 50%, missing more than half of the key questions.
4. **Language Performance**: Most models perform best in Japanese, with no significant difference between Chinese and English; this might be attributed to better instruction following in Japanese.
5. **Sector Variation**: Performance is better in the Communication, Consumer Discretionary, and IT sectors, with sector-wise performance disparities being more pronounced in weaker models.
6. **Impact of Length**: GPT-3.5 degrades sharply beyond 15K tokens (due to the 16K window limit), while Qwen2.5-72B and GPT-4o remain stable on long documents.
7. **Reliability of LLM Evaluation**: GPT-4-Judge exhibits consistent trends with Qwen-plus-Judge, showing a Fleiss' Kappa $= 0.701$ with 5 human annotators.

## Highlights & Insights

- **The First Benchmark Dedicated to Financial Meetings**: Fills the gap in meeting scenarios, showing fundamental HTML/structural differences from news or report data.
- **RAG Underperforms Full-Context Input in Long Contexts**: This holds important reference value for the practical implementation of RAG applications.
- **Comprehensive Multi-Dimensional Evaluation System**: Combines automatic metrics, LLM-Judge, and human evaluation, utilizing cross-validation to eliminate bias.
- **Revealing Q&A Pair Extraction as the Most Challenging Task**: Provides guiding significance for future research directions in financial NLP.

## Limitations & Future Work

1. Extremely high annotation costs (requiring professional analysts to annotate 1-2 hours of audio and more than 10K tokens of text).
2. The Q&A task only uses extracted Q&A pairs and does not evaluate open-ended questions requiring deep reasoning.
3. The number of Chinese meetings (400) is far greater than English and Japanese ones (100 each), presenting data imbalance.
4. Only 7 LLMs were evaluated, without covering broader open-source models (e.g., Mistral, DeepSeek, etc.).
5. ASR errors may still persist despite manual correction, potentially impacting downstream tasks.

## Related Work & Insights

- Complementary to ECTSum (Earnings Call Transfer Summarization), but M3FinMeeting offers more languages, wider sector coverage, and a more comprehensive set of tasks.
- Draws inspiration from long-context evaluation designs like LongBench and RULER, but focuses specifically on the financial vertical domain.
- While GPT-4-Judge evaluation has become mainstream, this work adds Qwen-plus cross-validation and human Kappa computation, enhancing overall credibility.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first multilingual and multi-sector financial meeting understanding benchmark; valuable scenario definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 7 models, 3 tasks, multilingual/multi-sector/multi-length analysis, RAG comparison, and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, detailed statistical tables, and sound evaluation methodology.
- **Value**: ⭐⭐⭐⭐ — Clear contribution to the financial NLP community; the dataset and findings (e.g., RAG's disadvantage in long documents) hold practical reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Multi-perspective Alignment for Increasing Naturalness in Neural Machine Translation](multi-perspective_alignment_for_increasing_naturalness_in_neural_machine_transla.md)
- [\[ACL 2025\] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution](cruxeval-x_a_benchmark_for_multilingual_code_reasoning_understanding_and_executi.md)
- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)

</div>

<!-- RELATED:END -->
