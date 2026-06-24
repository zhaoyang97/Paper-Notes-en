---
title: >-
  [Paper Note] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens
description: >-
  [ACL 2025][LLM (Other)][Training Data Attribution] OLMoTrace is proposed, the first system capable of tracing language model outputs back to their complete multi-trillion-token training data verbatim in real time (average 4.5 seconds). Based on an extended infini-gram engine with suffix array indexing, the system achieves highly efficient and precise matching, supporting application scenarios such as fact-checking, creative attribution, and mathematical capability tracing.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Training Data Attribution"
  - "Verbatim Matching"
  - "Suffix Array"
  - "infini-gram"
  - "Open LM"
date: 2026-05-08
content_hash: d3ec34a6dec0dc1d
---

# OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens

**Conference**: ACL 2025  
**arXiv**: [2504.07096](https://arxiv.org/abs/2504.07096)  
**Code**: [allenai/infinigram-api](https://github.com/allenai/infinigram-api)  
**Area**: LLM/NLP  
**Keywords**: Training Data Attribution, Verbatim Matching, Suffix Array, infini-gram, Open LM  

## TL;DR

OLMoTrace is proposed, the first system capable of tracing language model outputs back to their complete multi-trillion-token training data verbatim in real time (average 4.5 seconds). Based on an extended infini-gram engine with suffix array indexing, the system achieves highly efficient and precise matching, supporting application scenarios such as fact-checking, creative attribution, and mathematical capability tracing.

## Background & Motivation

**Background**: With the increasing adoption of LMs in high-stakes scenarios, understanding why models generate specific responses has become crucial. Fully open LMs (e.g., OLMo) make training data accessible, but existing behavior tracing methods (such as influence functions) cannot scale to trillion-token-level training data due to prohibitive computational overhead.

**Limitations of Prior Work**: (1) Influence functions (Koh & Liang 2017) require gradient information, making computation infeasible at the trillion-token scale; (2) Existing approaches like RAG retrieve before generation to improve generation quality, rather than tracing post-generation to understand the origins of behavior; (3) Search engines retrieve real-time web indexes rather than the actual training data of the LM, failing to establish a direct connection between LM behavior and the data.

**Key Challenge**: The contradiction between the scale of training data (4.6 trillion tokens) and the requirement for real-time attribution—the need to rapidly find text spans in massive datasets that precisely match LM outputs.

**Goal**: How to locate training documents that match the LM output verbatim from trillion-token training data under real-time interactive conditions, helping users understand LM behavior.

**Key Insight**: Leveraging suffix arrays to pre-sort and index training data, reducing the time complexity of exact match lookup to $O(L \log N)$ (with a parallelized latency of $O(\log N)$), which makes real-time attribution feasible.

**Core Idea**: Realizing real-time verbatim matching attribution against trillion-token training data based on the infini-gram suffix array index.

## Method

### Overall Architecture

The inference pipeline of OLMoTrace consists of 5 steps: (1) Find all maximal matching spans in the training data that appear in the LM output; (2) Filter and keep long and unique spans; (3) Retrieve training documents containing these spans; (4) Merge overlapping spans and documents of the same source; (5) Re-rank the documents using BM25 relevance and display them with color coding. The system is deployed on CPU-only nodes, with index files stored on high-IOPS SSD disks.

### Key Designs

1. **Fast Maximal Matching Span Computation**
    - **Function**: Find all maximal matching spans of the LM output in the trillion-token training data
    - **Mechanism**: For each suffix of the LM output, a binary search is conducted on the suffix array via infini-gram's Find query, finding the longest matching prefix with only a single query (utilizing the longest common prefix property of adjacent elements in the suffix array). Queries for all suffixes are completely parallelized, resulting in a total time complexity of $O(L \log N)$.
    - **Design Motivation**: A naive method would require enumerating $O(L^2)$ substrings and searching for them in dataset of size $N > 10^{12}$. By leveraging the sorted nature of the suffix array, only one Find query is required per position (rather than a binary search taking $O(\log L)$ times), achieving an acceleration from $O(L^2 N)$ to $O(L \log N)$.

2. **Span Unigram Probability Filtering**
    - **Function**: Filter the most "interesting" spans from a large number of matching spans
    - **Mechanism**: Compute the unigram probability of each span (the product of the unigram probabilities of individual tokens) and retain the $K$ spans with the lowest probability ($K = \lceil 0.05 \times L \rceil$). A lower unigram probability indicates that the span is longer and contains less frequent words.
    - **Design Motivation**: Compared to simply taking the longest spans, the unigram probability metric performs better in terms of retrieved document relevance as it simultaneously considers length and content uniqueness.

3. **BM25 Relevance Ranking and Color Coding**
    - **Function**: Prioritize the presentation of training documents most relevant to the LM output
    - **Mechanism**: Treating the set of retrieved documents as the "corpus" and the user prompt + LM response as the "query", BM25 scores are calculated for ranking. The relevance scores are categorized into high, medium, and low tiers, and displayed using colors of varying saturation.
    - **Design Motivation**: BM25 aligns closely with human relevance judgments (Spearman correlation of 0.73) and only requires CPU computation, meeting real-time requirements.

### Loss & Training

This paper presents a system-focused work and does not involve model training. The core technical challenges lie in engineering optimization: 12 data shards, with each shard capped at 500B tokens; index files stored on a 40TB SSD (80,000 IOPS); prefetching disabled to optimize throughput; and batch document retrieval adopted to reduce latency.

## Key Experimental Results

### Main Results

System performance and matching statistics (98 dialogue tests):

| Metric | Value |
|------|------|
| Training Data Scale | 3.2 billion documents, 4.6 trillion tokens |
| Avg. Response Length | 458 tokens |
| Avg. Inference Latency | 4.46 seconds |
| Avg. Length of Matching Spans | 10.4 tokens |
| Percentage of High-Relevance Documents | 14% |
| Percentage of High-Relevance Spans | 19% |
| Document Source: Pre-training data | 96.7% |
| Document Source: Intermediate training | 0.9% |
| Document Source: Post-training (SFT+DPO) | 2.4% |

### Ablation Study

Comparison of document relevance evaluation:

| Evaluation Method | Top-1 Document Avg. Score | Top-5 Document Avg. Score |
|---------|--------------------|--------------------|
| Human Evaluation | 1.90 | 1.43 |
| LLM-as-Judge (Initial Setup) | 1.73 | 1.28 |
| LLM-as-Judge (Optimized) | 1.82 | 1.50 |
| Spearman Correlation between Human and LLM Evaluation | 0.73 | - |

(Scoring criteria: 0=irrelevant, 1=broadly relevant, 2=same topic but different context, 3=direct match)

### Key Findings

1. The average length of matching spans is 10.4 tokens, indicating a substantial amount of long-range verbatim replication between the LM output and the training data.
2. 96.7% of the retrieved documents originate from the pre-training data, whereas math-heavy tasks tend to retrieve more SFT/RLVR data.
3. BM25 ranking shows reasonable consistency with human judgment regarding document relevance (Spearman correlation = 0.73).
4. The system is capable of detecting expressions that "appear creative" but are actually pre-existing in the training data (e.g., "I'm going on an adventure" in Tolkien-style stories).
5. Mathematical problem-solving steps can be precisely traced back to verbatim matches within the post-training data.

## Highlights & Insights

- **First Trillion-Scale Real-Time Attribution System**: Constrains inference latency to ~4.5 seconds, reaching interactive levels.
- **Exquisite Algorithmic Core**: Lowers the per-position query complexity from $O(\log L)$ to $O(1)$ by exploiting the properties of adjacent elements in the suffix array.
- **Fully Open Source and Accessible**: The system, model, and training data are completely publicly available, setting a benchmark for LM transparency.
- **Highly Inspiring Applications**: Three case studies—fact-checking, creative attribution, and mathematical reasoning attribution—demonstrate the multi-faceted value of the system.
- **Responsible Design**: Integrates mitigation measures for copyright issues, PII, and toxic content.

## Limitations & Future Work

- **Strictly Verbatim Matching**: Cannot identify the influence of training data that is semantically similar but phrased differently.
- **Non-Causal Interpretation**: Matching training documents should not be interpreted as having a causal effect on the LM output.
- **Supported Only for the OLMo Series**: Requires access to the full training data, preventing its application to closed-source models.
- **Limited Relevance Evaluation**: The Top-1 document achieved an average score of 1.90 (on a 0-3 scale), indicating room for improvement.
- Promising directions: Combining influence functions or gradient information to provide stronger causal attribution.

## Related Work & Insights

- **infini-gram**: The core engine of this system, which scales suffix arrays to the trillion-token level.
- **Fundamental Difference from RAG**: RAG performs retrieval before generation to enhance output, whereas OLMoTrace conducts attribution after generation to understand behavior.
- **ORCA**: Traces pre-training data using influence functions, but at a computational cost far exceeding that of this method.
- This work provides an actionable tool for tackling the fundamental problem of "the relationship between LM behavior and training data."

## Rating

- **Novelty**: 4/5 — The first trillion-scale real-time attribution system.
- **Technical Depth**: 4/5 — Exquisite suffix array algorithmic optimization and comprehensive system engineering.
- **Experimental Thoroughness**: 3/5 — Primarily focused on system demonstration, with relatively preliminary quantitative evaluation.
- **Practicality**: 5/5 — Fully deployed, usable, and entirely open-source.
- **Overall Rating**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm](do_language_models_understand_the_cognitive_tasks_given_to_them_investigations_w.md)
- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Speech Tokens for Neural Codec Language Models](analyzing_and_mitigating_inconsistency_in_discrete_speech_tokens_for_neural_code.md)

</div>

<!-- RELATED:END -->
