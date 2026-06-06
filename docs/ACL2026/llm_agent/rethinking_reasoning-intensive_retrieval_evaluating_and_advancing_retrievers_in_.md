---
title: >-
  [Paper Note] Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems
description: >-
  [ACL2026][LLM Agent][reasoning-intensive retrieval] This paper proposes BRIGHT-PRO, which re-evaluates reasoning-intensive retrievers using multi-aspect evidence annotation and an agentic search protocol. It also introdu…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "reasoning-intensive retrieval"
  - "agentic search"
  - "BRIGHT-PRO"
  - "RTriever"
  - "evidence coverage"
date: 2026-05-08
content_hash: 456436540379af4c
---

# Rethinking Reasoning-Intensive Retrieval: Evaluating and Advancing Retrievers in Agentic Search Systems

**Conference**: ACL2026  
**arXiv**: [2605.04018](https://arxiv.org/abs/2605.04018)  
**Code**: No public code  
**Area**: LLM Agent / Information Retrieval / Agentic Search  
**Keywords**: reasoning-intensive retrieval, agentic search, BRIGHT-PRO, RTriever, evidence coverage

## TL;DR
This paper proposes BRIGHT-PRO, which re-evaluates reasoning-intensive retrievers using multi-aspect evidence annotation and an agentic search protocol. It also introduces RTriever-Synth to train RTriever-4B, demonstrating that retrievers should optimize for "evidence portfolio coverage" rather than single-passage relevance.

## Background & Motivation
**Background**: Traditional information retrieval (IR) systems primarily optimize for keyword matching, semantic similarity, or single-passage relevance, which are suitable for factoid, single-hop questions. With the rise of Deep Research and agentic search systems, LLM agents repeatedly plan, search, read, and synthesize information, making the retriever a critical tool in the agent's reasoning chain.

**Limitations of Prior Work**: Complex queries usually require multiple complementary pieces of evidence to support an answer. However, gold passages in existing benchmarks like BRIGHT are narrow, often originating from only one or two webpages, and evaluation is mostly performed on static ranked lists. On the training side, synthetic retrieval corpora often follow a one-query-one-positive format, which encourages models to "find one relevant passage" rather than covering complete reasoning aspects.

**Key Challenge**: In agentic search, the value of a retriever is not defined by the highest relevance in a single retrieval step, but by its ability to provide the agent with a sufficient, complementary, and citable evidence portfolio in fewer rounds. Static single-passage metrics may fail to predict the final answer quality and search efficiency of the agent.

**Goal**: The authors aim to extend BRIGHT by constructing the BRIGHT-PRO benchmark with multi-aspect evidence, design both static and agentic evaluation protocols, and develop RTriever-Synth to fine-tune RTriever-4B specifically for reasoning-intensive evidence selection.

**Key Insight**: This paper elevates the retrieval task from "passage relevance" to "evidence portfolio construction." This aligns closely with how agents use search, as they require coverage of different sub-aspects of a problem rather than just a single answer fragment.

**Core Idea**: Use human-annotated reasoning aspects as evaluation units and aspect-aware synthetic data as training signals. This allows the retriever to learn to retrieve complementary evidence, with results validated at both static and agent-in-the-loop levels.

## Method

### Overall Architecture
The paper follows two main tracks. The evaluation track is BRIGHT-PRO: starting from the StackExchange subset of BRIGHT, experts annotate reasoning aspects, importance weights, and corresponding positive documents for each query. Retrievers are then evaluated using static $\alpha$-nDCG / A-Recall and an agentic search protocol. The training track is RTriever-Synth: DeepResearch-style analytical queries are generated from MS MARCO seed queries. Reference answers are decomposed into complementary reasoning aspects, for which positive passages and positive-conditioned hard negatives are synthesized. These data are used to fine-tune Qwen3-Embedding-4B via LoRA to obtain RTriever-4B.

### Key Designs
1.  **BRIGHT-PRO Multi-aspect Evidence Annotation**:
    - **Function**: Enables the benchmark to evaluate whether a retriever covers complete reasoning requirements rather than just finding one superficially relevant passage.
    - **Mechanism**: The authors selected the StackExchange subset of BRIGHT for its proximity to open-domain natural language reasoning. Domain experts decomposed each query into several reasoning aspects, described each with 1-2 rationales, and assigned importance using a 1-5 Likert scale (later normalized to weights). Experts then audited original BRIGHT positives, removed weakly relevant passages, merged overlapping segments, and supplemented new evidence via web search (Perplexity or ChatGPT).
    - **Design Motivation**: Answers to complex questions are often composed of multiple sub-questions. If a retriever only covers one high-weight aspect, it may perform well on traditional Recall but cause the agent to miss critical premises in the final synthesis.

2.  **Dual Static and Agentic Evaluation Protocols**:
    - **Function**: Isolates retrieval quality while measuring the systemic value of the retriever in a real agent loop.
    - **Mechanism**: Static evaluation uses $\alpha$-nDCG@k with a novelty penalty $\alpha=0.5$ to penalize redundant coverage of the same aspect; Weighted Aspect Recall, NDCG, and Recall are also reported. Agentic evaluation integrates the retriever with an LLM agent that iteratively issues search queries, reads top-5 passages, and generates cited answers. The fixed-round protocol uses 1/2/3 rounds; the adaptive protocol allows the agent to stop autonomously, using $AER=OQ\times e^{-\gamma(R-1)}$ to reward both quality ($OQ$) and low round count ($R$).
    - **Design Motivation**: In deployment, users care about search efficiency and reliability, not just $\alpha$-nDCG. Dual protocols reveal potential misalignments between static rankings and system performance.

3.  **RTriever-Synth and RTriever-4B Training**:
    - **Function**: Trains the retriever to prioritize complementary evidence selection from the training stage.
    - **Mechanism**: 140K queries were sampled from 1 million MS MARCO queries. Short queries were rewritten into DeepResearch-style queries with personas and context. Reference answers were generated and decomposed into 2-3 non-overlapping reasoning aspects. A positive passage blueprint was generated for each aspect and instantiated. Hard negatives were generated to be lexically similar to the query but specifically missing critical aspects, conditioned on the positive passage titles and abstracts.
    - **Design Motivation**: While standard contrastive retrievers only learn to rank a single relevant passage highly, RTriever-Synth forces the model to recognize complementary relationships and missing aspects within the training data.

### Loss & Training
RTriever-4B is based on Qwen3-Embedding-4B. All linear projection layers are fine-tuned using LoRA (rank 16, scaling factor 32), while the original embedding parameters are frozen. Each training step samples a query, a positive, and a hard negative, utilizing other documents in the batch as in-batch negatives. The model optimizes a query-document contrastive InfoNCE loss with temperature $\tau=0.02$ for 5 epochs, a peak learning rate of $1\times10^{-5}$, and a 5% linear warm-up.

## Key Experimental Results

### Main Results
BRIGHT-PRO covers 7 expert domains, totaling 739 queries and 526,319 documents, with an average of 7.13 positive documents and 3.74 reasoning aspects per query.

| Subset | Queries | Documents | Avg. Positives | Avg. Aspects | Avg. Query Length |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Biology | 103 | 59,513 | 7.81 | 3.94 | 92.6 |
| Earth Science | 115 | 123,575 | 7.44 | 3.83 | 82.2 |
| Economics | 99 | 52,240 | 7.81 | 3.71 | 123.5 |
| Psychology | 100 | 54,741 | 7.07 | 3.84 | 116.2 |
| Robotics | 101 | 63,920 | 6.17 | 3.71 | 218.8 |
| Stack Overflow | 115 | 109,188 | 4.60 | 3.32 | 172.0 |
| Sustainable Living | 106 | 63,142 | 9.25 | 3.86 | 116.9 |
| Overall | 739 | 526,319 | 7.13 | 3.74 | 131.4 |

In static retrieval evaluation, reasoning-trained retrievers significantly outperformed general embedding models, with RTriever-4B reaching the upper-middle tier after fine-tuning from Qwen3-Embedding-4B.

| Model | BRIGHT NDCG@10 | BRIGHT-PRO $\alpha$-nDCG@25 Overall | Position |
| :--- | :--- | :--- | :--- |
| BGE-Reasoner-8B | 33.8 | 68.0 | Strongest reasoning retriever |
| DIVER-4B-1020 | 30.6 | 63.7 | Strong reasoning retriever |
| DIVER-4B | 28.9 | 59.9 | Strong reasoning retriever |
| RTriever-4B | 27.7 | 55.3 | Ours; outperforms most general embeddings |
| INF-Retriever-Pro | 26.3 | 53.8 | Reasoning retriever |
| Qwen3-8B | 23.7 | 49.5 | General embedding base |
| OpenAI-Embed-3L | 17.9 | 45.8 | General embedding |
| BM25 | 14.5 | 40.3 | One of the weakest in static eval |

### Ablation Study
The fixed-round agentic evaluation uses a GPT-5-mini agent, retrieving top-5 results per round. It reports cumulative $\alpha$-nDCG, reasoning completeness, and overall quality. Static strength does not always equate to agent performance.

| Model | Round-3 $\alpha$-nDCG@15 | Round-3 Completeness | Round-3 Overall | Observation |
| :--- | :--- | :--- | :--- | :--- |
| BGE-Reasoner-8B | 63.04 | 4.42 | 4.31 | Leading in both retrieval and answer quality |
| DIVER-4B | 53.08 | 4.38 | 4.29 | Better in agentic than DIVER-4B-1020 |
| RTriever-4B | 50.79 | 4.37 | 4.25 | Answer quality in the top three |
| GTE-7B | 52.68 | 4.33 | 4.23 | Average static but strong agentic performance |
| DIVER-4B-1020 | 51.56 | 4.33 | 4.16 | Strong static but weaker agent fit |
| BM25 | 51.48 | 4.25 | 4.12 | Agent follow-up queries mitigate lexical mismatch |

The adaptive round protocol further highlights efficiency differences.

| Model / Agent | Avg. Rounds | Completeness | Overall | AER | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BGE-Reasoner + GPT-5-mini | 5.10 | 4.63 | 4.43 | 3.65 | High quality and early stopping |
| RTriever-4B + GPT-5-mini | 6.01 | 4.53 | 4.43 | 3.51 | Quality near BGE but more rounds |
| BM25 + GPT-5-mini | 5.73 | 4.50 | 4.42 | 3.53 | Unexpectedly strong in agentic setting |
| GTE-7B + GPT-5-mini | 6.67 | 4.62 | 4.51 | 3.44 | High final quality but high cost |
| RTriever-4B + Qwen3.5 | 4.89 | 4.26 | 4.06 | 3.38 | Consistently top-tier with a second agent |

### Key Findings
- BRIGHT-PRO's aspect-aware metrics clearly differentiate reasoning retrievers from general-purpose embedders, whereas BRIGHT's single NDCG@10 fails to expose this gap.
- Although RTriever-4B is not the strongest model overall, fine-tuning with 140K aspect-decomposed synthetic bundles allowed it to significantly outperform larger general embedding models, suggesting that the training objective is more critical than parameter scale.
- Static retrieval rankings do not fully predict agentic answer quality. For instance, DIVER-4B-1020 is stronger statically but weaker in the agentic loop than DIVER-4B; BM25 is statically weak but highly competitive due to the LLM's keyword-focused follow-up queries.
- AER reveals "failed" modes where an agent provides good answers but takes too long to search. GTE-7B has high overall quality, but its 6.67 average rounds lower its AER.

## Highlights & Insights
- BRIGHT-PRO expands the retrieval evaluation unit from documents to reasoning aspects, a critical step for reasoning-intensive IR. It distinguishes between "finding redundant evidence for one aspect" and "covering multiple necessary aspects."
- The agentic evaluation design is highly practical. Because the retriever is queried repeatedly by the LLM with increasingly specific queries, the value of lexical methods like BM25 is reactivated.
- The hard negatives in RTriever-Synth are not just semantically similar; they are "near neighbors missing critical aspects," mimicking common retrieval failures in complex QA.
- The paper serves as a reminder that agent system optimization should not rely solely on stronger LLMs. A retriever that provides sufficient coverage and matches the agent's query style can directly reduce search rounds and reasoning hallucinations.

## Limitations & Future Work
- BRIGHT-PRO is based only on the StackExchange subset of BRIGHT. While it covers 7 domains, it does not yet include news, legal, full-text medical, or corporate knowledge bases found in real Deep Research scenarios.
- The human cost for 739 queries and the 175-query agentic sample is high, but the scale remains limited; statistical stability across specific domains requires further expansion.
- Agentic evaluation relies on LLM-as-a-Judge for completeness and overall quality, which may be subject to judge bias.
- RTriever-Synth currently samples one positive / one negative triplet, missing the opportunity to leverage the full multi-positive set for each query. Future work could explore multi-positive contrastive learning, aspect-aware sampling, and curriculum negatives.

## Related Work & Insights
- **vs BRIGHT**: BRIGHT first focused on reasoning-intensive retrieval, but its gold evidence is narrow and evaluation is primarily static. BRIGHT-PRO adds aspect labels, weights, and agentic protocols.
- **vs DIVER / ReasonIR**: These methods train reasoning-aware retrievers, but training signals are mostly centered on single passage relevance. RTriever-Synth emphasizes complementary positives and aspect coverage.
- **vs DeepResearch benchmarks**: Many DeepResearch benchmarks evaluate only the final answer, making it difficult to isolate the retriever's contribution. This paper plugs the retriever as the sole variable into the same agent for clearer component analysis.
- **Insight**: When building agentic RAG, retriever evaluation should simultaneously report coverage, round cost, answer quality, and retriever-agent compatibility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Comprehensive integration of multi-aspect coverage and agentic retriever-in-the-loop evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive static, fixed-round, adaptive-round, and qualitative analyses, though benchmark scale is limited by manual annotation.
- Writing Quality: ⭐⭐⭐⭐☆ Well-structured with clear pipelines and diagrams, though some tables are quite dense.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for Deep Research, agentic RAG, and reasoning retriever training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](../../ICML2026/llm_agent/process_reward_agents_for_steering_knowledge-intensive_reasoning.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](../../ICLR2026/llm_agent/livenewsbench_evaluating_llm_web_search_capabilities_with_freshly_curated_news.md)
- [\[ACL 2026\] ToolOmni: Enabling Open-World Tool Use via Agentic Learning with Proactive Retrieval and Grounded Execution](toolomni_enabling_open-world_tool_use_via_agentic_learning_with_proactive_retrie.md)

</div>

<!-- RELATED:END -->
