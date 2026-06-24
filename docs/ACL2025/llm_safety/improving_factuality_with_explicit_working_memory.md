---
title: >-
  [Paper Note] Ewe: Improving Factuality with Explicit Working Memory
description: >-
  [ACL 2025][LLM Safety][factuality] Trigers Ewe (Explicit Working mEmory), which introduces explicit working memory consisting of multiple KV cache units during LLM decoding. It dynamically receives feedback from compiled retrieval knowledge and fact-checking. When errors are detected, Ewe deletes the incorrect sentences and regenerates them using the updated memory. It improves VeriScore F1 by 2–6 points across 4 factual long-form generation benchmarks without sacrificing hel…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "factuality"
  - "working memory"
  - "RAG"
  - "fact-checking"
  - "hallucination"
  - "KV cache"
date: 2026-05-08
content_hash: e9d11ed9f581f885
---

# Ewe: Improving Factuality with Explicit Working Memory

**Conference**: ACL 2025  
**arXiv**: [2412.18069](https://arxiv.org/abs/2412.18069)  
**Code**: None (Meta FAIR)  
**Area**: LLM Safety  
**Keywords**: factuality, working memory, RAG, fact-checking, hallucination, KV cache

## TL;DR

Trigers Ewe (Explicit Working mEmory), which introduces explicit working memory consisting of multiple KV cache units during LLM decoding. It dynamically receives feedback from compiled retrieval knowledge and fact-checking. When errors are detected, Ewe deletes the incorrect sentences and regenerates them using the updated memory. It improves VeriScore F1 by 2–6 points across 4 factual long-form generation benchmarks without sacrificing helpfulness.

## Background & Motivation

1. **Severe LLM hallucination issues**: Large Language Models are prone to factual errors (hallucinations) in long-form generation, particularly in knowledge-intensive QA scenarios where misinformation occurs frequently.
2. **Traditional RAG only provides single static knowledge**: Standard RAG retrieves documents once before generation and concatenates them to the prompt, failing to dynamically update knowledge along the generation lifecycle. Consequently, the knowledge becomes outdated once the generated text deviates from the initial retrieval.
3. **Iterative RAG methods are limited by traditional designs**: Although methods like FLARE and Self-RAG introduce sentence-by-sentence retrieval and self-reflection, raw knowledge remains part of the input string, lacking a flexible mechanism for multi-source information fusion.
4. **Lack of real-time fact-checking loop**: Existing methods solely rely on retrieval to acquire new knowledge but lack verification mechanisms to check if the generated content is correct, preventing real-time error correction during generation.
5. **Difficulty in integrating multi-source feedback**: Retrieved knowledge and fact-checking feedback possess distinct attributes (retrieval provides background context, while fact-checking targets specific details). Traditional methods struggle to elegantly integrate these two information streams.
6. **Low computational efficiency**: Iterative approaches require reprocessing embeddings for all context documents at each update, generating significant redundant calculations when only a portion of knowledge needs to be revised.

## Method

### Overall Architecture

On top of standard Transformer decoding, Ewe introduces two core mechanisms: **periodically pausing to obtain feedback** and **memory-augmented generation**.

### 3.1 Initialization

Given an input prompt, Contriever is used to retrieve $k$ relevant text segments from C4 + Wikipedia. These segments are independently encoded by the language model as $k$ KV cache memory units and stored in the working memory. Unlike RAG, which directly concatenates passages to the prompt, Ewe encodes each passage independently and stores them in parallel.

### 3.2 Real-Time Feedback Mechanism

During generation, Ewe pauses every $T_r$ steps to obtain **retrieval feedback** and every $T_v$ steps to obtain **fact-checking feedback** (experimentally $T_r=1, T_v=8$), triggered only when a new complete sentence is generated:

- **Retrieval feedback**: Using the original question + currently generated sentence as the query, Contriever retrieves new passages from C4 + Wikipedia. Passages with a retrieval score above a threshold are considered relevant knowledge and used to update the working memory.
- **Fact-checking feedback**: The claim extraction and verification models from VeriScore are used as the fact-checker. First, atomic claims are extracted from the newly generated sentence, and then Google snippets are used to verify if each claim is supported by evidence. If an incorrect claim is detected, contradictory information (the correct fact) is appended to the working memory as a new memory unit, while the incorrect sentence is deleted to backtrack and regenerate from the previous timestep.

### 3.3 Working Memory Refresh

The working memory adopts a **FIFO (First-In-First-Out)** update strategy. New text passages from the retriever and fact-checker are encoded into KV cache, sharing the same positional IDs and processed in parallel. The memory is stored at the front of the model's context (before the prompt and generated text), precluding the need to recompute existing content during updates.

### 3.4 Attention Aggregation

During self-attention computation at each layer, each memory unit is individually concatenated with the context to execute standard self-attention, and then weighted and aggregated with an attention normalization term:

$$\vec{h}_n = \sum_{i=1}^{k+1} \frac{\alpha_i \vec{h}_{n_i}}{\sum_{j=1}^{k+1} \alpha_j}$$

where $k$ memory units are each concatenated with the context to produce $k$ hidden vectors, while an additional pure context hidden vector (the $(k+1)$-th item) is preserved to improve the fluency of long outputs.

### 3.5 Relationship with Existing Methods

Ewe can degenerate into special cases of several existing methods:
- $k=1$, no pausing $\to$ standard RAG
- $k=1$, sentence-by-sentence pausing + low-probability triggered retrieval $\to$ FLARE
- Multiple memory units + retrieval feedback only $\to$ an enhanced version of iterative RAG

## Experiments

### Evaluation Settings

- **Base Models**: Llama-3.1-70B and Llama-3.1-8B (instruction-tuned)
- **Datasets**: LongFact (250 prompts), Fava (141), AlpacaFact (241), Biography (181)
- **Factuality Metrics**: VeriScore F1 (harmonic mean of precision/recall for claim extraction + verification)
- **Helpfulness Metrics**: AlpacaEval Win Rate (GPT-4o as judge, using Llama-3.1-70B as baseline)
- **Baselines**: Base model, RA (Retrieval-Augmented), Nest (Semi-parametric decoding), DRAGIN (Dynamic retrieval), CoVe (Chain-of-Verification), CoVe w/ Retrieval

### Table 1: Main Results (VeriScore F1 / AlpacaEval Win Rate)

| Model | LongFact F1 | LongFact WR | Fava F1 | Fava WR | AlpacaFact F1 | AlpacaFact WR | Biography F1 | Biography WR |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.1-70B | 64.3 | - | 52.0 | - | 63.8 | - | 37.1 | - |
| + RA | 72.7 | 41.2 | 56.8 | 37.1 | 66.0 | 43.1 | 43.8 | 49.4 |
| + Nest | 63.2 | 9.1 | 50.3 | 24.1 | 58.1 | 30.2 | 41.5 | 22.1 |
| + DRAGIN | 71.5 | 38.2 | 57.2 | 33.9 | 65.3 | 31.5 | 42.8 | 33.5 |
| + CoVe | 63.8 | 39.3 | 49.5 | 33.4 | 61.5 | 33.3 | 37.7 | 31.3 |
| + CoVe w/ Retrieval | 67.4 | 31.8 | 52.6 | 23.1 | 64.0 | 28.8 | 38.2 | 29.4 |
| **+ Ewe** | **75.9** | **50.1** | **61.0** | **50.1** | **66.9** | **49.9** | **49.7** | **50.2** |
| Llama-3.1-8B | 63.1 | 40.6 | 51.0 | 36.5 | 65.3 | 26.7 | 28.9 | 24.2 |
| + RA (8B) | 65.9 | 28.1 | 51.8 | 16.8 | 63.9 | 18.5 | 41.4 | 21.3 |
| + Ewe (8B) | 67.3 | 40.5 | 53.1 | 36.2 | 65.5 | 28.0 | 42.2 | 21.5 |

### Table 2: Impact of Different Retrieval Sources (VeriScore F1, 50 prompts subset)

| Data Source | LongFact | Biography | AlpacaFact | Fava |
|------|:---:|:---:|:---:|:---:|
| Wikipedia | 67.9 | 46.1 | 55.5 | 52.5 |
| C4 | 70.8 | 44.6 | 53.7 | 53.3 |
| C4 + Wikipedia | 74.8 | 48.4 | 53.3 | 52.3 |

### Table 3: Human Evaluation Consistency (Cohen's Kappa)

| Method | Cohen's Kappa |
|------|:---:|
| RA (Baseline) | 0.61 |
| Ewe | 0.65 |

Cohen's Kappa > 0.61 is considered high agreement. Ewe's real-time feedback with VeriScore during inference does not diminish human agreement, verifying that the VeriScore improvement indeed translates into real factuality enhancement.

## Key Findings

1. **Ewe achieves the highest VeriScore F1 across all 4 datasets**, with an absolute improvement of 2–6 points on the 70B model, and maintains an AlpacaEval Win Rate close to 50%, demonstrating that the improvement in factuality does not come at the expense of helpfulness.
2. **The improvement on the 8B model is smaller than on the 70B model**, presumably because smaller models have weaker feedback-utilization capabilities and cannot always correctly regenerate corrected sentences.
3. **The number of memory units needs to be moderate**: Too many memory units cause outdated information to persist under the FIFO strategy for too long, decreasing precision.
4. **There is a trade-off in memory unit length**: Short units improve precision but reduce recall (due to sharper focus on a single document), while long units improve recall but decrease precision (due to diluted attention across multiple documents).
5. **The retrieval threshold needs to be balanced**: A threshold too low introduces irrelevant information, while one too high excludes too many retrieved results; both harm F1.
6. **Different datasets favor different data sources**: LongFact/Fava favor C4, whereas Biography/AlpacaFact favor Wikipedia. The C4 + Wikipedia combination further improves performance on LongFact and Biography.
7. **Model confidence can substitute for fixed intervals**: Using entropy or min-prob as a trigger signal yields better or comparable F1 with fewer verification steps at an optimal threshold.
8. **The format of factuality feedback is crucial**: Directly providing supporting paragraphs outperforms instructions like "Please do not generate the following incorrect claim", as the latter tends to cause the model to misunderstand and repeat the incorrect content in the output.

## Highlights & Insights

- **Unified Framework**: Ewe unifies existing methods such as RAG, FLARE, and iterative retrieval as special cases, providing a more general perspective.
- **Dual-Stream Feedback**: Simultaneously utilizes retrieved knowledge (background context) and fact-checking (error-correction details), naturally fusing them by encoding both into parallel KV cache memory units.
- **Efficient Incremental Updates**: Only updates memory units that need to be refreshed. Outdated knowledge is discarded while valid knowledge is directly reused as KV caches, avoiding redundant encoding.
- **Real-Time Error-Correction Loop**: Detection $\to$ Deletion $\to$ Backtracking $\to$ Regeneration - a complete online error-correction pipeline.
- **Natural Alignment with Streaming Scenarios**: For scenarios like streaming voice input where backtracking edits are impossible, Ewe can receive, retrieve, and generate on the fly.

## Limitations & Future Work

1. **Only English text datasets are validated**, leaving performance on multilingual and non-factual prompts unknown.
2. **Only textual feedback is supported**, with multimodal (image, table, etc.) feedback unexplored.
3. **The scale of human evaluation is relatively small** (only 120 annotations), requiring larger-scale validation in the future.
4. **High inference cost**: Each step potentially involves retrieval, fact-checking, and regeneration, making the computational overhead significantly higher than standard single-turn RAG.
5. **Simplistic FIFO memory update strategy**: Although alternative strategies based on relevance scores were explored, they did not outperform FIFO; more optimal update mechanisms remain for future work.
6. **Fact-checker dependence on external search engines** (Google snippets) imposes search quality and latency bottlenecks.

## Related Work & Insights

- **Iterative/Adaptive Retrieval-Augmentation**: FLARE (sentence-level low-probability triggered retrieval), DRAGIN (confidence metrics based on attention and entropy), Self-RAG (self-reflective critique model), and ITER-RETGEN (using previous outputs as queries). Ewe distinguishes itself by storing memory via KV cache instead of text concatenation and conveying fact-checking feedback through memory rather than raw input strings.
- **Chain-of-Verification Methods**: CoVe relies solely on LLM self-reasoning for verification without introducing external knowledge, which yields limited effectiveness on long texts.
- **Long-Context Memory**: Memory3 encodes 128-token chunks of training corpus as KV cache to serve as explicit memory, retrieving the most relevant ones during inference. Ewe differs in its objective—it dynamically updates memory during iterative decoding rather than pre-encoding the entire corpus. Memorizing Transformers and LongMem employ similar ideas to extend the context window.
- **Factuality Evaluation**: FActScore, SAFE, and VeriScore share the paradigm of "decomposition into atomic claims + external validation". VeriScore uses Google snippets instead of Wikipedia, which is more efficient and provides broader coverage.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The architecture of working memory + dual-stream feedback is elegantly designed, though KV cache memory itself is not pioneering (existing work like Memory3 exists).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 4 datasets + 5 baselines + human evaluation + rich ablation studies (number of memory units, length, threshold, data sources, feedback formats).
- **Writing Quality**: ⭐⭐⭐⭐ Clearly explains the degenerative relationships with existing methods; framework diagrams are intuitive.
- **Value**: ⭐⭐⭐⭐ High inference cost limits practical deployment, but it provides a flexible framework for factual generation in long-form texts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving Model Factuality with Fine-grained Critique-based Evaluator](improving_model_factuality_with_fine-grained_critique-based_evaluator.md)
- [\[ACL 2025\] Unveiling Privacy Risks in LLM Agent Memory](mextra_agent_memory_privacy.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](real-time_factuality_assessment_from_adversarial_feedback.md)

</div>

<!-- RELATED:END -->
