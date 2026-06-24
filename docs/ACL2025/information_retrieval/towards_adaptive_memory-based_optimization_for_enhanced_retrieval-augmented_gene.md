---
title: >-
  [Paper Note] Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes the Amber framework, which enhances retrieval efficiency and answer quality in open-domain question answering within an iterative RAG paradigm through the collaboration of three components: an Agent-based Memory Updater, an Adaptive Information Collector, and a Multi-granular Content Filter.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Adaptive Retrieval"
  - "Memory Update"
  - "Multi-Agent Collaboration"
  - "Multi-Granular Filtering"
date: 2026-05-08
content_hash: a2e2355eadd05554
---

# Towards Adaptive Memory-Based Optimization for Enhanced Retrieval-Augmented Generation

**Conference**: ACL 2025  
**arXiv**: [2504.05312](https://arxiv.org/abs/2504.05312)  
**Code**: Yes ([https://anonymous.4open.science/r/Amber-B203/](https://anonymous.4open.science/r/Amber-B203/))  
**Area**: NLP / Retrieval-Augmented Generation  
**Keywords**: RAG, Adaptive Retrieval, Memory Update, Multi-Agent Collaboration, Multi-Granular Filtering

## TL;DR

This paper proposes the Amber framework, which enhances retrieval efficiency and answer quality in open-domain question answering within an iterative RAG paradigm through the collaboration of three components: an Agent-based Memory Updater, an Adaptive Information Collector, and a Multi-granular Content Filter.

## Background & Motivation

Retrieval-Augmented Generation (RAG) enhances model response accuracy and mitigates hallucination issues by integrating external knowledge bases. However, existing RAG methods suffer from three core limitations in open-domain QA tasks:

**Lack of memory mechanism**: Each retrieval operates independently, lacking a consolidated memory of previously retrieved information, which causes the generated results to reflect only fragmented knowledge from a single retrieval step.

**Non-adaptive retrieval strategies**: When LLMs perform reasoning using retrieved segments, they cannot actively assess the validity of the information, nor can they decide when to stop retrieval or update retrieval queries based on what is already known.

**Noise interference**: The proportion of valid information in the retrieved text is very low, and the large amount of redundant information introduces noise that obscures key details.

These challenges are particularly prominent in complex multi-hop QA and long-text QA tasks, which require aggregating and synthesizing information across multiple documents.

## Method

### Overall Architecture

Amber is an iterative RAG framework with adaptive memory updates, where three core components collaborate:
- **Agent-based Memory Updater (AMU)**: Consolidates and optimizes the LLM's memory via a multi-agent collaborative approach.
- **Adaptive Information Collector (AIC)**: Serves as the main scheduler controlling the overall RAG workflow, dynamically adjusting retrieval queries and deciding when to stop retrieval.
- **Multi-granular Content Filter (MCF)**: Filters noise during the retrieval process using multi-granular content filtering.

Workflow: Given a query $q$, initialize an empty memory $M_0$ $\rightarrow$ in each iteration, retrieve top-k text chunks $\rightarrow$ filter using MCF $\rightarrow$ update memory using AMU $\rightarrow$ evaluate sufficiency with AIC $\rightarrow$ if insufficient, generate a new query and proceed to the next round $\rightarrow$ finally, generate the answer using memory $M_t$ via ICL.

### Key Designs

1. **Agent-based Memory Updater (AMU)**:

    - Consists of conversational collaboration among three independent agents: **Reviewer**, **Challenger**, and **Refiner**.
    - Reviewer evaluates the correctness and relevance of memory updates.
    - Challenger identifies potential defects and overlooked constraints.
    - Refiner synthesizes feedback from the first two agents to perform specific revisions.
    - **Design Motivation**: A single agent for memory updates is prone to missing information or introducing biases, whereas multi-agent adversarial reviewing ensures memory quality.

2. **Adaptive Information Collector (AIC)**:

    - Consists of three steps per iteration: retrieve top-k chunks $\rightarrow$ update memory with AMU $\rightarrow$ evaluate if the memory is sufficient to answer the query.
    - If insufficient, generate a refined query $q_{t+1} = \text{AIC}(q, q_t, m_{t+1})$.
    - **Design Motivation**: Avoids over-retrieval (wasting computational resources) and under-retrieval (incomplete information).

3. **Multi-granular Content Filter (MCF)**:

    - **Two-stage Filtering**: First determines whether the entire block is relevant to the query at the chunk level, and then filters key sentences from the relevant chunks at the sentence level.
    - Uses STRINC and CXMI metrics along with GPT-4 generated training data to fine-tune the LLM via multi-task learning.
    - **Design Motivation**: The proportion of noise in retrieved texts is extremely high, and direct utilization would interfere with memory updating and final answering.

### Loss & Training

- MCF fine-tunes the LLM through multi-task learning to simultaneously train the filtering capability at both the chunk and sentence levels.
- During the iteration process, zero-shot ICL is used to generate the final answer.
- Base LLMs used include Qwen2-7b, Llama3-8b, and GPT-3.5.

## Key Experimental Results

### Main Results

| Method | SQuAD (acc/f1) | NQ (acc/f1) | TriviaQA (acc/f1) | 2WikiMQA (acc/f1) | HotpotQA (acc/f1) | ASQA (str-em/str-hit) |
|------|--------------|-------------|-------------------|-------------------|-------------------|-----------------------|
| No Retrieval | 12.6/18.4 | 24.0/27.5 | 49.8/52.7 | 28.4/35.6 | 19.8/25.2 | 35.5/8.9 |
| Vanilla RAG (GPT-3.5) | 34.4/37.9 | 35.9/38.4 | 63.8/63.5 | 35.4/38.2 | 38.6/44.4 | 47.8/21.6 |
| Adaptive-RAG | 33.0/38.3 | 44.6/47.3 | 58.2/60.7 | 46.4/49.8 | 44.4/52.6 | 42.1/15.8 |
| **Amber (GPT-3.5)** | **35.8/39.1** | **47.4/52.0** | **66.8/66.1** | **46.7/46.0** | **47.4/53.6** | **51.3/26.3** |

### Ablation Study

| Component | SQuAD acc | NQ acc | 2WikiMQA acc | HotpotQA acc | ASQA str-em |
|------|-----------|--------|-------------|-------------|-------------|
| Amber (Full) | 35.8 | 47.4 | 46.7 | 47.4 | 51.3 |
| - AMU | Decrease | Decrease | Decrease | Decrease | Decrease |
| - AIC | Decrease | Decrease | Decrease | Decrease | Decrease |
| - MCF | Decrease | Decrease | Decrease | Decrease | Decrease |

(The ablation study validates the effectiveness of each component)

### Key Findings

1. **Full Superiority**: Amber achieves state-of-the-art or second-best performance across all six datasets, with the most significant improvements observed on multi-hop QA (2WikiMQA, HotpotQA) and long-text QA (ASQA).
2. **Cross-Model Consistency**: Excellent performance gains are consistently observed across Amber regardless of whether Qwen2-7b, Llama3-8b, or GPT-3.5 is utilized as the base LLM.
3. **Advantage of Multi-Agent Collaboration**: The three-agent conversational collaboration in AMU significantly outperforms memory updates by a single agent.
4. **Value of Adaptive Stopping**: The adaptive stopping mechanism in AIC effectively avoids noise and performance degradation caused by over-retrieval.

## Highlights & Insights

- **Introduction of the Memory Mechanism** is the core innovation: elevating RAG from "stateless retrieval" to "stateful iterative knowledge accumulation."
- The multi-agent collaboration (Reviewer-Challenger-Refiner) design resembles the academic peer review process, helping to guarantee the quality of memory updates.
- The two-stage filtering (chunk $\rightarrow$ sentence) effectively reduces noise while preserving information, presenting a simple and highly effective concept.
- Designing the AIC to dynamically generate new queries based on the current memory addresses the issue of insufficient information from fixed queries in multi-hop reasoning.

## Limitations & Future Work

- The computational overhead of multi-agent dialogue is substantial, requiring multiple LLM calls per iteration.
- Controlling the number of iterations depends on the adaptive judgment of the AIC, introducing risks of premature stopping or excessive iteration.
- Training the MCF depends on labeled data generated by GPT-4, thereby introducing dependency on closed-source models.
- The framework has not been validated on larger-scale LLMs (e.g., 70B).
- The recall rate limitation of the retriever (Contriever) itself could become a bottleneck.

## Related Work & Insights

- Compared to Self-RAG (Asai et al., 2023), Amber does not rely on self-reflection tokens but instead utilizes explicit memory management.
- Unlike the low-confidence triggered retrieval in FLARE (Jiang et al., 2023), Amber dynamically manages retrieval through the structured AIC module.
- **Insight**: The development of RAG systems is transitioning from "one-shot retrieve-and-generate" to "iterative memory accumulation," indicating that memory management is a core mechanism for future RAG.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The multi-agent collaboration paradigm for memory updates and the design of the two-stage content filtering are creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across six datasets covering single-hop/multi-hop/long-text QA, three base LLMs, and compared with multiple baselines.
- **Writing Quality**: ⭐⭐⭐ — The framework is described clearly, but some notations are not entirely consistent.
- **Value**: ⭐⭐⭐⭐ — Possesses solid reference value for iterative retrieval and memory management directions in the RAG domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] Accelerating Adaptive Retrieval Augmented Generation via Instruction-Driven Representation Reduction of Retrieval Overlaps](accelerating_adaptive_retrieval_augmented_generation_via_instruction-driven_repr.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] A Reality Check on Context Utilisation for Retrieval-Augmented Generation](a_reality_check_on_context_utilisation_for_retrieval-augmented_generation.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)

</div>

<!-- RELATED:END -->
