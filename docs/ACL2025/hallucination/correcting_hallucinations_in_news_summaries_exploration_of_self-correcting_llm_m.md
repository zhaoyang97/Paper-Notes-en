---
title: >-
  [Paper Note] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Correction] This paper systematically explores the performance of two self-correction methods (CoVE and RARR) in correcting hallucinations in news summaries. By comparing three search engines, multiple retrieval settings, and prompting strategies, it is found that the combination of Bing search snippets and RARR (few-shot) yields the best performance, with G-Eval aligning closely with human evaluations.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Correction"
  - "Self-Correction"
  - "News Summarization"
  - "Search Engines"
  - "LLM Post-processing"
date: 2026-05-08
content_hash: 2c9edd127022bf14
---

# Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge

**Conference**: ACL 2025  
**arXiv**: [2506.19607](https://arxiv.org/abs/2506.19607)  
**Code**: [GitHub](https://github.com/jvladika/HalluCorrect)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Correction, Self-Correction, News Summarization, Search Engines, LLM Post-processing  

## TL;DR

This paper systematically explores the performance of two self-correction methods (CoVE and RARR) in correcting hallucinations in news summaries. By comparing three search engines, multiple retrieval settings, and prompting strategies, it is found that the combination of Bing search snippets and RARR (few-shot) yields the best performance, with G-Eval aligning closely with human evaluations.

## Background & Motivation

### 1. Background

Although Large Language Models (LLMs) generate fluent text, they frequently suffer from hallucinations—producing factually incorrect or misleading information. Post-hoc correction methods, especially multi-step self-correction approaches, attempt to fix errors in the initial response by generating verification questions, retrieving evidence, and iteratively revising the text.

### 2. Limitations of Prior Work

- Self-correction methods have primarily been validated on encyclopedic generation tasks (biographies, list completions, etc.), but their **application in the news summarization domain remains under-explored**.
- Evidence retrieval is a critical stage, yet there is a lack of systematic comparison regarding **search engine selection, snippet vs. full-text retrieval, and retrieval strategies**.
- The **trade-off between faithfulness and correction aggressiveness** is often overlooked—over-correction may disrupt the original style and information of the summary.

### 3. Key Challenge

News summarization requires accurately correcting factual errors while remaining faithful to the original style, creating tension between these two goals. Every stage of the self-correction pipeline (question generation, evidence retrieval, answer generation, and text rewriting) relies on LLMs, and each step can potentially introduce new errors.

### 4. Goal

How do self-correction methods perform in the context of news summarization hallucination correction? What is the impact of different search engines, retrieval settings, and prompting strategies on correction quality?

### 5. Key Insight

Select two representative self-correction frameworks (CoVE, RARR), enhance their external search capabilities, and conduct comprehensive comparative experiments on news summarization datasets.

### 6. Core Idea

Provide a systematic empirical study of the CoVE and RARR self-correction frameworks, revealing how search engine selection, retrieval granularity, and prompting strategies influence the correction of hallucinations in news summaries.

## Method

### Overall Architecture

Both frameworks share the same four-step workflow:

1. **Retrieve the Initial Response** $b$ (the summary containing hallucinations).
2. **Generate Verification Questions** $q_1, ..., q_k$ (guided by LLM prompt $M_q$).
3. **Answer the Questions**: Retrieve evidence $e$ from evidence source $s$, and generate answers $a_1, ..., a_k$ using $M_a(q, e)$.
4. **Rewrite the Response**: Input the initial response and answers into $M_r(b, a)$ to produce the corrected response $r$.

### Key Designs

#### Differences Between the Two Self-Correction Frameworks

| Feature | CoVE | RARR |
|------|------|------|
| Prompting Method | Zero-shot | Few-shot (6 exemplars) |
| Correction Style | Bold modifications, adding external details | Conservative and faithful, close to original |
| Output Length | Usually longer | Similar to original |

#### Evidence Retrieval Settings

- **Internal Knowledge**: Relying solely on the LLM's own internal knowledge base to answer verification questions.
- **Search Engines**: Google, Bing, DuckDuckGo.
- **Retrieval Granularity**:
    - **Snippet Mode**: Utilizes top-5 summary snippets returned by search engines directly.
    - **Full-Article Mode**: Parses article HTML, chunks the content into embeddings, and selects the top-5 passages using SimCSE cosine similarity.
- **Gold Baseline**: Uses the original news article as the ceiling for evidence.

### Loss & Training

No training is involved—this is a post-processing method purely based on LLM inference. The base model utilized is GPT-4o-mini, with additional testing conducted on LLaMA 3.1 70B and Mixtral 8x7B.

## Key Experimental Results

### Main Results

| Framework | Evidence Source | NED↓ | Semantic Similarity↑ | NLI-Contradiction↓ | G-Eval Overall↑ | G-Eval Factual↑ |
|------|----------|------|------------|----------|-----------------|-----------------|
| CoVE | GPT Internal | 0.51 | 81 | 42 | 50 | 45 |
| RARR | GPT Internal | 0.10 | 94 | 40 | 65 | 62 |
| CoVE | Google Snippet | 0.51 | 84 | 34 | 56 | 50 |
| RARR | **Bing Snippet** | **0.14** | **95** | **35** | **69** | **60** |
| RARR | Google Snippet | 0.24 | 93 | 32 | 67 | 56 |
| RARR | Bing Full-Article | 0.32 | 92 | 32 | 63 | 50 |
| CoVE | Gold Article | 0.49 | 88 | 18 | 70 | 63 |
| RARR | Gold Article | 0.21 | 94 | 19 | **75** | **67** |

### Human Evaluation

| Framework | Human Rating | G-Eval Score | Correlation Coefficient |
|------|---------|------------|----------|
| RARR | 0.68 | 0.65 | Pearson r = 0.87 (p<1%) |
| CoVE | 0.54 | 0.52 | — |

### Key Findings

1. **External Search > Internal Knowledge**: For both CoVE and RARR, using search engines yields better outcomes than relying solely on the LLMs' internal knowledge.
2. **Bing Snippets Perform Best**: RARR + Bing snippets achieves the best performance across 6 metrics.
3. **Snippets > Full-Text**: Querying search snippets is more precise than full-text retrieval, whereas full-text mode introduces substantial irrelevant information (leading to a higher NLI-neutral score).
4. **Few-shot (RARR) > Zero-shot (CoVE)**: Few-shot exemplars help maintain faithfulness, whereas zero-shot prompts encourage aggressive paraphrasing.
5. **G-Eval Aligns Highly with Human Evaluation**: The Pearson correlation reaches 0.87, with an average difference of only 3%.
6. **DuckDuckGo is a Free and Viable Alternative**: Despite slightly lower performance, its API is completely free.
7. **Gold Article as Upper Bound**: The highest G-Eval rating (75) is achieved when feeding original news articles. This demonstrates that precise evidence is paramount.

## Highlights & Insights

- **Comprehensive Empirical Study**: Systematically compares 2 frameworks × 3 search engines × 2 retrieval modes × 3 LLMs with high experimental rigor.
- **Counter-Intuitive Finding of Snippets > Full-Text**: Additional retrieval is not necessarily beneficial; concise snippets returned from search queries are often more on-topic.
- **High Reliability of G-Eval**: A correlation coefficient of 0.87 provides a strong alternative for future research where large-scale human evaluation is unfeasible.
- **Deep Divergence between Zero-shot and Few-shot**: This is not just a performance difference, but a fundamental shift in correction philosophy—aggressive rewriting versus faithful modification.

## Limitations & Future Work

1. All components depend on LLM API calls, introducing high computational cost and latency.
2. The generation quality of verification questions is occasionally unstable, leaving potential errors unnoticed or generating irrelevant queries.
3. Evaluation is restricted to the news domain; generalizability to non-expert or professional fields (e.g., medicine, law) remains unverified.
4. Retrieval optimization tactics, such as structured querying or trusted domain filtering, were not explored.
5. The zero-shot configuration of CoVE results in uncontrollable output styles; more constraints may be necessary for production environments.

## Related Work & Insights

- **CoVE (Dhuliawala et al., 2024)**: Chain-of-Verification methodology using zero-shot corrections.
- **RARR (Gao et al., 2023)**: Retrofit Attribution using Research and Revision, utilizing few-shot corrections.
- **CRITIC**: External tool interaction for self-correction.
- **SummEdits**: A benchmark dataset used for evaluating summary hallucination correction in this paper.
- **Insights**: Retrieve-and-correct remains a highly promising paradigm, yet precise evidence retrieval matters more than complex correction logic.

## Rating

- **Novelty**: ⭐⭐⭐ — The methodology is not fundamentally new (relying on pre-existing frameworks), but the empirical findings are highly valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Features exceptionally comprehensive comparison experiments, multi-dimensional benchmarks, and human validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Structures arguments clearly with intuitive case analysis.
- **Value**: ⭐⭐⭐⭐ — Delivers actionable guidance for choosing self-correction frameworks and retrieval strategies, directly informing future system implementations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[ACL 2025\] ETF: An Entity Tracing Framework for Hallucination Detection in Code Summaries](etf_an_entity_tracing_framework_for_hallucination_detection_in_code_summaries.md)
- [\[ACL 2025\] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning](alleviating_hallucinations_from_knowledge_misalignment_in_large_language_models_.md)
- [\[ACL 2025\] HalluLens: LLM Hallucination Benchmark](hallulens_llm_hallucination_benchmark.md)
- [\[ACL 2025\] HALoGEN: Fantastic LLM Hallucinations and Where to Find Them](halogen_hallucinations.md)

</div>

<!-- RELATED:END -->
