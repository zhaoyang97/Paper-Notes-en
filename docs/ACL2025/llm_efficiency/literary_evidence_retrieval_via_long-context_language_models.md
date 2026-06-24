---
title: >-
  [Paper Note] Literary Evidence Retrieval via Long-Context Language Models
description: >-
  [ACL 2025][LLM Efficiency][Long-context understanding] The RELiC dataset is adapted into a long-context literary evidence retrieval benchmark (292 high-quality samples), requiring models to find missing citations for literary analyses within full novel texts (45k-125k tokens). Gemini Pro 2.5 achieves a 62.5% accuracy, surpassing human experts (55%) for the first time, whereas the best open-source model, DeepSeek-R1, reaches only 29.1%, highlighting a huge gap in interpretativ…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long-context understanding"
  - "literary evidence retrieval"
  - "benchmark"
  - "evaluation of reasoning models"
  - "open-source vs. closed-source"
  - "literary analysis"
date: 2026-05-08
content_hash: fe30fb8523c4bd9a
---

# Literary Evidence Retrieval via Long-Context Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.03090](https://arxiv.org/abs/2506.03090)  
**Code**: [katherinethai/long_context_relic](https://github.com/katherinethai/long_context_relic)  
**Area**: LLM Efficiency  
**Keywords**: Long-context understanding, literary evidence retrieval, benchmark, evaluation of reasoning models, open-source vs. closed-source, literary analysis

## TL;DR

The RELiC dataset is adapted into a long-context literary evidence retrieval benchmark (292 high-quality samples), requiring models to find missing citations for literary analyses within full novel texts (45k-125k tokens). Gemini Pro 2.5 achieves a 62.5% accuracy, surpassing human experts (55%) for the first time, whereas the best open-source model, DeepSeek-R1, reaches only 29.1%, highlighting a huge gap in interpretative reasoning between closed-source and open-source models.

## Background & Motivation

**Limitations of Long-Context Evaluation**: Existing long-context benchmarks (such as Needle-in-Haystack) primarily test simple information retrieval, failing to measure the deep understanding and reasoning capabilities of models on long texts.

**Unique Requirements of Literary Analysis**: Literary evidence retrieval requires models to possess both a global narrative reasoning capability and the ability for close textual examination. It is an ideal testbed for evaluating true long-context understanding.

**Limitations of the RELiC Dataset**: The original RELiC dataset (Thai et al., 2022) contains 78k literary analysis excerpts, but the data is noisy, featuring OCR errors, citation leaks, and data contamination, making it unsuitable for direct LLM evaluation.

**Core Motivation**: Exploring the level of understanding of literary fiction achieved by today's long-context LLMs, which are capable of processing millions of tokens.

## Method

### Overall Architecture

Task Definition: Given the full text of a novel and a literary analysis excerpt (where a specific citation is masked), the model is required to generate/retrieve the masked citation passage from the original novel. This simulates the process of literary scholars selecting supporting evidence during their analyses.

### Dataset Construction Pipeline

Starting from the 78k samples in RELiC, 292 high-quality samples are obtained after multi-step filtering and manual review:

1. **CLEAN (GPT-4o-mini)**: Cleans OCR errors and removes in-text citations that expose citation page numbers.
2. **LEAKAGE (Heuristic)**: Excludes samples where the context duplicates the original text using fuzzy matching (threshold 95) to prevent models from cheating via lexical overlap.
3. **LIT ANALYSIS (GPT-4o-mini)**: Categorizes and determines whether a sample is indeed a literary analysis, excluding mislabeled samples.
4. **LOCATION (GPT-4o-mini)**: Detects whether the context leaks positional information of the citation.
5. **FIRST/LAST SENT (Heuristic)**: Flags first/last sentences (which may be famous quotes and easily memorized).
6. **OUTLIER (Heuristic)**: Flags passages that are excessively cited.
7. **EZ2MEM (GPT-4)**: Evaluates whether the model can answer the question purely from memory without the source text, excluding easily memorizable samples.
8. **Manual Review**: English literature degree holders reviewed 400 entries, ultimately identifying 292 entries as high-quality samples.

The F1-score of the filtering scheme is 89.8 (based on 100 manually verified samples, containing 57 TP / 30 TN / 6 FP / 7 FN).

### Dataset Characteristics

| Metric | Full Text (7 Novels) | Samples (292) |
|--------|---------------------|---------------|
| Average token count | 85,526 | 254.9 |
| Maximum token count | 124,544 | 492.0 |
| Minimum token count | 45,038 | 116.0 |
| Average samples per book | — | 36.0 |

Covering 7 classic English novels: *The Great Gatsby*, *Frankenstein*, *The Scarlet Letter*, *Brave New World*, *What Maisie Knew*, *Ethan Frome*, and *The Awakening*.

### Evaluation Subset Design

- **Human Eval Set (40 samples)**: A subset solved by English literature experts who also provided their reasoning processes.
- **Close Reading Set (39 samples)**: Samples labeled as "close reading" types—where the analysis excerpt repeatedly quotes specific words or phrases from the target passage, resulting in lexical overlap.

### Prompting Strategies

- **Simple Prompt**: Directly requests the model to output the citation.
- **Explanation Prompt**: Asks the model to first provide reasons for its choice, and then output the citation (adapted from the Nocha benchmark).

### Embedding Baseline

Using gte-Qwen2-7B-instruct, ranked first on the MTEB leaderboard, as the embedding retrieval baseline to calculate recall@1.

## Key Experimental Results

### Main Results

| Model | Prompt | ALL (n=292) | Human Eval (n=40) | Close Reading (n=39) |
|------|--------|-------------|---------------------|----------------------|
| Gemini Pro 2.5 | Explanation | **64.7%** | **62.5%** | **79.5%** |
| GPT-4.1 | Explanation | 51.0% | 47.5% | 69.2% |
| o3 | Explanation | 50.7% | 50.0% | 66.7% |
| Gemini Pro 1.5 | Explanation | 38.5% | 40.0% | 50.0% |
| Claude Sonnet 3.7 | Explanation | 37.0% | 32.5% | 48.7% |
| DeepSeek-R1 (Best Open-source) | Explanation | 29.1% | 15.0% | 38.5% |
| GPT-4o | Explanation | 24.3% | 22.5% | 31.8% |
| Qwen 3 (32B) | Explanation | 19.2% | 20.0% | 33.3% |
| Qwen 3 (8B) | Explanation | 8.9% | 5.0% | 10.3% |
| o3-mini | Explanation | 8.3% | 10.0% | 13.6% |
| gte-Qwen2-7B (embedding baseline) | — | 4.5% | 2.5% | 6.8% |
| **Human Expert** | — | — | **55.0%** | — |

**Key Findings**: Gemini Pro 2.5 outperforms human experts across all metrics, with an average call time of only 45 seconds (compared to 12 minutes for humans).

### Overgeneration Analysis

| Model | Accuracy | Average Length Ratio |
|------|--------|------------|
| Human Expert | 55.0% | 2.1 |
| Gemini Pro 2.5 | 62.5% | 3.0 |
| o3 | 50.0% | 2.7 |
| GPT-4.1 | 47.5% | 4.8 |
| Claude Sonnet 3.7 | 32.5% | 4.0 |
| DeepSeek-R1 | 15.0% | 3.6 |
| Llama 3.1 (8B) | 5.0% | 5.9 |
| Llama 3.3 (70B) | 0.0% | 5.7 |

Length Ratio = model output length / ground-truth citation length (in characters), where closer to 1.0 is better. All models exceed the human ratio of 2.1. Small models (Llama series > 5.7) suffer more from severe overgeneration, indicating that weaker models tend to produce longer outputs to compensate for uncertainty.

### Key Findings

1. **LLMs significantly outperform embedding baselines**: gte-Qwen2-7B achieves only 4.5%, which is just 1.6% higher than the best recall@1 reported in the original RELiC paper three years ago. This indicates that the task requires full-text reasoning rather than simple semantic matching.
2. **Closed-source models significantly gap open-source ones**: The best open-source model, DeepSeek-R1 (29.1%), achieves less than half the accuracy of the best closed-source model, Gemini Pro 2.5 (64.7%).
3. **Small models struggle to leverage close reading cues**: Models with >8B parameters generally show a 10-20%+ improvement on the Close Reading subset, whereas 7B/8B models show almost no improvement or even a decrease.
4. **Significant generational advancement**: Gemini Pro 2.5 vs. 1.5 (64.7% vs. 38.5%), GPT-4.1 vs. GPT-4o (51.0% vs. 24.3%), o3 vs. o1 (50.7% vs. 32.2%).
5. **Varying effects of Explanation Prompt**: It benefits Gemini Pro 1.5, but causes a performance drop for GPT-4o. Reasoning models (such as o3 and Gemini Pro 2.5), which possess native internal reasoning tokens, are minimally affected by external explanation prompts.

## Case Study

### Case 1: Model Failure / Human Success

In *The Scarlet Letter* (>80k words), the analysis excerpt indirectly refers to the depiction of Roger Chillingworth recognizing Hester Prynne. Human experts considered two candidate passages and ultimately chose correctly by judging which one better illustrated the "melodrama" (exaggerated character portrayal, words suggesting action, emotional tension) mentioned in the analysis. All three top LLMs selected the candidate passage that human experts had ruled out.

### Case 2: Agreement on "Incorrect" Answers between Model and Human

In *Frankenstein*, both humans and Gemini Pro 2.5 selected the same "incorrect" citation—both interpreted the analysis excerpt as an introduction to the jury's reaction, whereas the ground-truth citation actually supports Frankenstein’s own mental state. This indicates that: (1) literary evidence retrieval is inherently polysemous; (2) models can identify plausible alternative evidence, which offers auxiliary value to literary scholars.

## Highlights & Insights

- **High-quality, realistic long-context benchmark**: More convincing than synthetic tasks like Needle-in-Haystack, requiring genuine narrative understanding and interpretative reasoning.
- **First time LLMs outperform human experts in literary tasks**: Gemini Pro 2.5 wins with 62.5% vs. 55.0%, with a 16x speedup.
- **Quantitative evidence of the open- vs. closed-source gap**: The limitation lies not in long-context capability itself, but in interpretative reasoning performance.
- **Rigorous data cleaning pipeline**: The 8-step filtering and manual review provide a methodological reference for benchmark construction.
- **Systematic quantification of overgeneration**: All models tend to generate excessively long outputs, with weaker models being particularly prone.

## Limitations & Future Work

1. **Limited data scale**: Consists of only 292 test samples across 7 novels, which limits statistical power.
2. **Restricted to English/Western literature**: Primarily sourced from public-domain classics, which does not represent the global literary tradition.
3. **Single annotator**: The human baseline was completed by a single annotator (one of the paper's authors), which may introduce individual bias.
4. **Gap between the task and real-world scenarios**: In authentic literary analysis, scholars progressively develop arguments and select citations, rather than filling in blanks for a given critique.
5. **Evaluation depends on fuzzy matching**: Automated evaluation uses partial ratio fuzzy matching, which might miss responses that are semantically correct but phrased slightly differently.

## Related Work & Insights

- **Long-context Benchmarks**: Nocha (Karpinska et al., 2024), FABLES (Kim et al., 2024).
- **Original Dataset**: RELiC (Thai et al., 2022) — 78k literary analysis excerpts and citations.
- **Computational Literary Analysis**: BookWorm (character analysis), STORYSUMM (story summary faithfulness), Reading Subtext (short story comprehension).
- **Narrative Understanding**: HEART-felt (narrative element extraction), Agents' Room (narrative generation).

## Rating

- Novelty: ⭐⭐⭐⭐ Incorporating literary analysis into long-context evaluation offers a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison of 15+ models, human baselines, overgeneration analysis, and case studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with vivid case studies.
- Value: ⭐⭐⭐⭐ Highly significant reference for research in long-context understanding and interpretative reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)

</div>

<!-- RELATED:END -->
