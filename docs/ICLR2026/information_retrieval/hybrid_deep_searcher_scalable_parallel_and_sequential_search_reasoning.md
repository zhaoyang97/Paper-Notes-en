---
title: >-
  [Paper Note] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning
description: >-
  [ICLR 2026][Information Retrieval & RAG][deep search] This paper proposes HybridDeepSearcher, which constructs the HDS-QA dataset to train a large language reasoning model (LRM) to distinguish parallelizable from sequent…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "deep search"
  - "parallel search"
  - "retrieval-augmented generation"
  - "large language reasoning models"
  - "test-time search scaling"
date: 2026-05-08
content_hash: f6963aa10fc0c5f3
---

# Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning

**Conference**: ICLR 2026
**arXiv**: [2508.19113](https://arxiv.org/abs/2508.19113)  
**Code**: None  
**Area**: Information Retrieval
**Keywords**: deep search, parallel search, retrieval-augmented generation, large language reasoning models, test-time search scaling

## TL;DR

This paper proposes HybridDeepSearcher, which constructs the HDS-QA dataset to train a large language reasoning model (LRM) to distinguish parallelizable from sequentially dependent search queries. The approach achieves F1 gains of +15.9 on FanOutQA and +11.5 on a BrowseComp subset, while substantially reducing inference latency and demonstrating consistent test-time search scaling.

## Background & Motivation

Large language reasoning models (LRMs) such as OpenAI o3 and DeepSeek-R1, when combined with retrieval-augmented generation (RAG), form deep research agents that complete complex multi-step tasks through a reason–query–retrieve loop. However, existing methods exhibit critical limitations:

**High latency**: Purely sequential querying retrieves one result at a time, with each query adding to overall latency.

**Incoherent workflow**: Sequential search causes models to attempt premature answers or issue redundant queries.

**Poor scalability**: For questions requiring exhaustive search across many documents, one-at-a-time querying fails to cover all relevant evidence.

Consider a question about John Carpenter films: each film's runtime must be retrieved individually. Sequential approaches are slow and prone to omission, whereas **simultaneously querying** all film runtimes is far more efficient and accurate.

The core problem is: **how can LRMs exploit both parallel and sequential search strategies in deep research?**

## Method

### Overall Architecture

The method consists of two core components: (1) construction of the HDS-QA dataset, and (2) training and inference of HybridDeepSearcher.

### Key Designs

1. **HDS-QA Dataset Construction**: An automated pipeline constructs a dataset of hybrid-hop questions.

   **Question generation pipeline** (4 steps):
   - **Entity extraction and related question collection**: Starting from single-hop seed questions from Natural Questions, central entities are extracted, related questions are gathered via Google "People Also Ask," and only queries retrieving distinct documents are retained to ensure diversity.
   - **Entity feature summarization**: Retrieved documents are summarized into key features of the corresponding entity.
   - **Parallel-hop question construction**: Features are combined to form parallel-hop questions that implicitly reference the entity without naming it directly.
   - **Hybrid-hop question integration**: Parallel-hop questions are embedded into the original single-hop question by replacing the central entity, introducing an additional sequential hop. Both stages are verified to require multi-step retrieval.

   This pipeline yields 1,987 hybrid-hop questions.

   **Answer trajectory generation**:
   - Qwen3-32B iteratively executes the reason–query–retrieve loop, allowing multiple parallel queries to be issued at each step.
   - Each question is reasoned over 4 times; all correct trajectories are retained to increase diversity in reasoning strategies.
   - 773 questions receive correct answers, yielding 2,111 successful trajectories (a success rate of ~27%, reflecting the genuine difficulty of the task).

2. **HybridDeepSearcher inference procedure**:

    - **Reasoning**: The model reasons within `<think>` and `</think>` tags.
    - **Querying**: Based on the reasoning output, sequential or parallel queries are generated within `<|begin_search_queries|>` and `<|end_search_queries|>` tags, with multiple parallel queries separated by `;\n`.
    - **Retrieval**: Each query is executed via a web search API; retrieved documents are summarized by an external model (Qwen3-32B) before being returned.
    - The model iterates through multiple reason–query–retrieve rounds until sufficient information is gathered to produce a final answer.

3. **Adaptive search strategy**: The model learns to dynamically determine when to issue parallel queries (for independent subquestions) versus sequential queries (for subquestions dependent on prior results), and explicitly represents the current step (in blue) and subsequent plans (in purple) during reasoning.

### Loss & Training

- Full-parameter fine-tuning is performed on Qwen3-8B using the 2,111 question–answer trajectories for 1 epoch.
- Learning rate is $3 \times 10^{-5}$, batch size 4, with gradient accumulation over 32 steps.
- **Gradient updates are not applied to the search result portions** to prevent the model from memorizing retrieved content.
- Training requires only 8 A100 40GB GPUs and takes approximately 30 minutes.

## Key Experimental Results

### Main Results

| Dataset | Metric | HybridDeepSearcher | Prev. SOTA (RAG-R1) | Gain |
|---|---|---|---|---|
| MuSiQue | F1 | 31.2 | 29.7 | +1.5 |
| FanOutQA | F1 | 44.1 | 28.2 | **+15.9** |
| FRAMES | F1 | 39.1 | 35.8 | +3.3 |
| MedBrowseComp | MBE | 30.4 | 28.2 | +2.2 |
| BrowseComp-50 | F1 | 17.2 | 5.7 | **+11.5** |

AUC (efficiency–effectiveness trade-off): HybridDeepSearcher achieves the highest values across all benchmarks, indicating that higher accuracy is attained with fewer search rounds.

### Ablation Study / Search Capability Analysis

| Method | MuSiQue Coverage | FanOutQA Coverage | FRAMES Coverage |
|---|---|---|---|
| Search-o1 | 33.4% | 38.3% | 44.8% |
| DeepResearcher | 38.8% | 49.9% | 49.0% |
| RAG-R1 | 35.9% | 53.2% | 48.0% |
| **HybridDeepSearcher** | **40.7%** | **61.0%** | **55.8%** |

Evidence coverage improvement is largest on FanOutQA (+7.8 pp), the dataset with the most annotated evidence links and the greatest need for extensive parallel retrieval.

### Key Findings

1. **Test-time search scaling** (core advantage):
    - HybridDeepSearcher performance **continues to improve** as the number of search rounds and API calls increases.
    - Baselines such as RAG-R1 **plateau** after 2–3 rounds.
    - This is especially pronounced on BrowseComp-50, where other methods gain almost nothing from additional search budget.

2. **Efficiency advantage**: Higher accuracy is achieved with fewer search rounds.
    - On FanOutQA, approximately 3 rounds suffice to surpass the performance of other methods using 5 or more rounds.

3. **Failure of non-iterative methods**: Direct generation and standard RAG perform extremely poorly (F1 of 0.0/1.8 on BrowseComp-50), confirming that these benchmarks genuinely require external knowledge and multi-step reasoning.

4. **Case study insights**:
    - On the John Carpenter question in FRAMES, HybridDeepSearcher issues parallel queries for the runtimes of 12 films and identifies the correct answer (Starman, 115 minutes), whereas DeepResearcher prematurely assumes The Thing and Search-o1 enters a repetitive query loop.

## Highlights & Insights

1. **Unification of parallel and sequential search**: This work is the first to systematically train LRMs to distinguish parallelizable from sequentially dependent queries, addressing a gap left by prior work.
2. **Elegant dataset construction**: The HDS-QA automated pipeline starts from NQ and introduces parallelism via "People Also Ask," resulting in a design that is both principled and scalable.
3. **SFT outperforms RL**: Supervised fine-tuning on only 2,111 trajectories surpasses RL-based methods using GRPO (e.g., Search-R1, DeepResearcher), demonstrating the critical importance of high-quality hybrid search demonstration data.
4. **Search scalability**: This method is among the few to exhibit consistent test-time search scaling, with performance continuing to improve as the computational budget grows.
5. **Extremely low training cost**: Fine-tuning takes only 30 minutes on 8 A100 GPUs, far less than RL-based training approaches.

## Limitations & Future Work

1. Training relies solely on SFT without preference optimization (DPO/RLHF); successful and failed trajectories in HDS-QA could be leveraged to further improve the model.
2. Search query summarization depends on an external large model (Qwen3-32B), increasing system complexity and API call costs.
3. HDS-QA is constructed exclusively from Natural Questions, potentially limiting domain coverage.
4. Multi-agent collaborative search remains unexplored.
5. BrowseComp-50 selects only 50 questions solvable by o3, and this selection bias may affect evaluation fairness.

## Related Work & Insights

- **Search-o1**: A prompt-based iterative reason–query–retrieve framework with single-query sequential search.
- **Search-R1 / DeepResearcher**: Use GRPO training to enhance search reasoning, but training data lacks parallel search demonstrations.
- **RAG-R1**: A multi-query baseline with competitive performance but lacking search scalability.
- **APR**: Adaptive parallel reasoning, validated only on toy tasks such as Countdown.

This work offers a key insight for RAG system design: explicitly training on "when to parallelize versus when to sequence" is more effective than simply increasing reasoning capacity. Hybrid search strategy is likely a critical capability for large-scale deep research agents.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[NeurIPS 2025\] Deep Research Brings Deeper Harm](../../NeurIPS2025/information_retrieval/deep_research_brings_deeper_harm.md)
- [\[ICLR 2026\] RefTool: Reference-Guided Tool Creation for Knowledge-Intensive Reasoning](reftool_reference-guided_tool_creation_for_knowledge-intensive_reasoning.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)

</div>

<!-- RELATED:END -->
