---
title: >-
  [Paper Note] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels
description: >-
  [ACL 2025][LLM Efficiency][long-context summarization] The authors construct CNNSum—a multi-scale long-text summarization benchmark based on Chinese novels (695 samples, 16k-128k tokens)—ensuring quality via human annotation. Through a systematic evaluation of 20+ LLMs, they discover that advanced LLMs tend to generate subjective commentary leading to vague summaries, smaller models offer better cost-effectiveness, fine-tuning Base versions yields superior results over Chat v…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "long-context summarization"
  - "Chinese novels"
  - "benchmark"
  - "LLM evaluation"
  - "RoPE extrapolation"
date: 2026-05-08
content_hash: 76021ef4dd00c3ff
---

# CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels

**Conference**: ACL 2025  
**arXiv**: [2412.02819](https://arxiv.org/abs/2412.02819)  
**Code**: [GitHub](https://github.com/CxsGhost/CNNSum)  
**Area**: LLM Efficiency  
**Keywords**: long-context summarization, Chinese novels, benchmark, LLM evaluation, RoPE extrapolation

## TL;DR

The authors construct CNNSum—a multi-scale long-text summarization benchmark based on Chinese novels (695 samples, 16k-128k tokens)—ensuring quality via human annotation. Through a systematic evaluation of 20+ LLMs, they discover that advanced LLMs tend to generate subjective commentary leading to vague summaries, smaller models offer better cost-effectiveness, fine-tuning Base versions yields superior results over Chat versions, and fine-tuning on short-context data alone can significantly enhance long-text summarization capabilities.

## Background & Motivation

**Background**: Despite the rapid development of long-context LLMs (where 128k contexts are now common), research on long-text summarization has progressed slowly, and existing long-text summarization datasets remain severely insufficient.

**Limitations of Prior Work**:
   - Existing benchmarks are mostly based on older datasets (such as BookSum, CNNDM, etc.), posing high data leakage risks.
   - The sample sizes are small (dozens of samples), and the average/maximum lengths are short (typically <16k).
   - A lack of multi-scale length subsets makes it impossible to evaluate performance across different context lengths.
   - Annotation quality is poor—either collected from the web (high leakage risk) or synthesized by LLMs (introducing various errors).

**Key Challenge**: While 128k context windows have become standard, the performance of LLMs in long-text summarization degrades sharply as length increases. Outputs may become chaotic, meaningless, or fail to follow instructions. The core bottleneck lies in the lack of high-quality datasets and systematic research guidance.

**Goal**: To construct a high-quality multi-scale Chinese long-text summarization benchmark, and to systematically explore the capability boundaries and optimization strategies of LLMs in long-text summarization.

**Key Insight**: Starting from Chinese web novels (which exhibit high originality and low leakage risk), samples are collected across four distinct scales (L, XL, 2XL, 3XL) and annotated through a human-LLM collaborative process.

**Core Idea**: Robust research in long-text summarization requires a rigorous benchmark. CNNSum fills the gap in Chinese long-text summarization benchmarks through strict multi-scale design and human annotation.

## Method

### Overall Architecture

CNNSum construction pipeline: **Corpus Collection** → **Multi-Scale Sampling** → **Summary Annotation** → **Benchmark Evaluation & Exploration**

### Key Designs

**1. Corpus Collection and Filtering**

- Collected 103 Chinese web novels, each characterized by a clear chapter structure.
- Excluded books comprising multiple independent short stories or lacking a main plotline.
- Utilized Qwen2-72B-Instruct to detect potential popular books (with high leakage risk), filtering out 27 novels.
- Applied regular expressions and manual correction to fix non-standard punctuation and remove irrelevant inserted content.

**2. Multi-Scale Sampling Strategy**

Based on the Yi tokenizer, four target lengths and corresponding ranges are defined:

| Subset | Target Length | Sampling Range | Samples | Source Books |
|------|---------|---------|--------|-------|
| L | 16k | [12k, 18k] | 190 | 76 |
| XL | 32k | [26k, 34k] | 195 | 71 |
| 2XL | 64k | [54k, 66k] | 200 | 60 |
| 3XL | 128k | [112k, 130k] | 110 | 45 |

A sliding window approach is employed to sample along chapters, prioritizing samples from rare books to preserve diversity.

**3. Summary Annotation**

- LLMs are first used to generate plot summaries for each chapter.
- 23 human annotators review these summaries, select the key plots, and rewrite them.
- 2XL/3XL samples are annotated by one person and cross-checked by another.
- Requirements: (1) Rewrite in the annotator's own words rather than simply deleting or merging sentences; (2) Avoid subjective comments and focus strictly on objective plot events.
- Word limits: 500 characters for L/XL, 600 characters for 2XL/3XL.

**4. Two Prompt Types**

- **Prompt-IB**: Instructions placed at the beginning of the text.
- **Prompt-IE**: Instructions placed at the end of the text.
- It is observed that different prompt types significantly impact the output quality (showing substantial MSE discrepancies).

### Evaluation Metrics

The evaluation primarily uses ROUGE-L, accompanied by human-conducted fine-grained analysis of anomalous output types.

## Key Experimental Results

### Main Results: ROUGE-L Scores

| Model | L (16k) | XL (32k) | 2XL (64k) | 3XL (128k) |
|------|---------|----------|-----------|------------|
| GPT-4o | 15.5 | 14.2 | 12.5 | - |
| Gemini-1.5-pro | 19.3 | 18.1 | 16.8 | 14.6 |
| Qwen-plus | 20.5 | 18.5 | 16.4 | 14.8 |
| Moonshot-v1-128k | 22.4 | 20.3 | 18.0 | 15.2 |
| Qwen2.5-72B-Inst | 19.6 | 17.6 | 13.6 | 13.4 |
| InternLM2.5-7B-Chat-1M | 18.0 | 17.1 | 14.7 | 13.0 |
| Yi-1.5-34B-32K | 11.6 | 10.5 | 9.6 | 0.1 |
| Yi-6B-200K | 9.9 | 9.4 | 8.8 | 4.0 |
| Llama3.1-8B-Inst | 15.6 | 14.3 | 12.8 | 9.9 |
| LWM-Text-1M | 3.3 | 3.0 | 2.5 | 1.1 |

### Impact of Prompt Types (MSE between P-IB and P-IE)

| Model | MSE |
|------|-----|
| Yi-6B-200K | 4.5 |
| Yi-1.5-34B-32K | **14.5** |
| Yi-1.5-34B-Chat-16K | **7.8** |
| Qwen1.5-7B | **34.4** |
| Qwen2-72B | **16.3** |
| GPT-4o | 0.0 |
| Gemini-1.5-pro | 0.1 |

An MSE $\ge 5.0$ indicates that the prompt type has a profound impact on the model, whereas proprietary models typically exhibit greater stability.

### Key Findings

1. **GPT-4o favors subjective commentary**, resulting in vague summaries. Consequently, its ROUGE-L scores are inferior compared to models like Moonshot and Qwen.
2. **Larger models are not necessarily better**: Their advantages in reasoning and comprehension are difficult to leverage in long-text summarization tasks, making smaller models more cost-effective.
3. **Chat/Instruct versions may compromise the summarization capabilities of Base models**: Experimental fine-tuning demonstrates that Base models achieve superior results.
4. **Models scaled with RoPE ABF exhibit strong extrapolation potential**: Fine-tuning with short-text data alone can significantly improve long-text summarization performance.
5. **Mixed-length samples may lead to misleading evaluation outcomes**: Multi-scale isolated evaluation provides more reliable results.

## Highlights & Insights

1. **"Long-text summarization primarily relies on memory capacity"**—this profound insight explains why the reasoning advantages of larger LLMs fail to manifest in this task.
2. **Highly valuable dataset construction methodology**: The workflow of multi-scale sampling, sliding windows, and human annotation is highly reusable.
3. **Quantification of prompt type impacts**: The MSE metric visually illustrates how prompt placement differently affects various models, offering practical guidance for real-world applications.
4. **Thorough leakage risk control**: A multi-pronged approach combining newly published books, LLM-based detection, and rigorous filtering.
5. **Systematic and comprehensive fine-tuning exploration**: Validation from multiple angles including Base vs. Chat comparison, short-text training for long-text generalization, and RoPE extrapolation.

## Limitations & Future Work

1. **Restricted to the Chinese novel domain**: The summarization style and structure may not generalize well to other domains like academic papers or news.
2. **Limitations of ROUGE-L evaluation**: The paper acknowledges the significant gap between ROUGE-L and human preferences, yet does not propose a superior alternative.
3. **The 3XL subset contains only 110 instances**: The statistical reliability of evaluations at the 128k length remains somewhat limited.
4. **Annotation quality depends heavily on annotator comprehension**: Long-text annotation is inherently demanding for human evaluators, making individual differences difficult to eliminate completely.
5. **Lack of evaluation on the latest models like o1 or Claude**

## Related Work & Insights

- Compared to **BookSum**: CNNSum is more recent, multi-scaled, specifically Chinese, and human-annotated, comprehensively outperforming the older benchmark in design.
- Compared to **CLongEval-LStSum**: While the latter utilizes GPT-4 to synthesize labels across mixed lengths, CNNSum achieves greater reliability in extrapolation evaluations.
- **Insights for fine-tuning strategies**: Training on short texts to generalize to long contexts is particularly effective for models scaled with RoPE ABF, offering an affordable path to boost long-context capabilities.
- The discovery regarding **"subjective commentary vs. objective plot"** provides actionable guidance for prompt design, suggesting that LLMs should be explicitly instructed to avoid evaluative language.

## Rating

- Novelty: ⭐⭐⭐ — The core contribution lies in the benchmark construction; the methodology itself is not highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation over 20+ models, coverage of both proprietary and open-source models, and a systematic exploration of fine-tuning and extrapolation.
- Writing Quality: ⭐⭐⭐⭐ — Highly clear summaries of findings, rich tables, and a well-structured narrative.
- Value: ⭐⭐⭐⭐ — Fills the gap in Chinese long-text summarization benchmarks; the insights offer practical guidance for real-world applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](sliding_windows_full_ranking.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](literary_evidence_retrieval_via_long-context_language_models.md)

</div>

<!-- RELATED:END -->
