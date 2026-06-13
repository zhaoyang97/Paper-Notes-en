---
title: >-
  [Paper Note] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering
description: >-
  [AAAI 2026][Information Retrieval & RAG][Multi-hop question answering] This paper proposes REAP, a dual-module iterative framework that addresses multi-hop question answering through recursive collaboration between a Sub…
tags:
  - "AAAI 2026"
  - "Information Retrieval & RAG"
  - "Multi-hop question answering"
  - "retrieval-augmented generation"
  - "adaptive planning"
  - "fact extraction"
  - "multi-task fine-tuning"
date: 2026-05-08
content_hash: 6a2cd9ca70114ea2
---

# REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering

**Conference**: AAAI 2026
**arXiv**: [2511.09966](https://arxiv.org/abs/2511.09966)  
**Code**: [https://github.com/Deus-Glen/REAP](https://github.com/Deus-Glen/REAP)  
**Area**: NLP Understanding / RAG
**Keywords**: Multi-hop question answering, retrieval-augmented generation, adaptive planning, fact extraction, multi-task fine-tuning

## TL;DR
This paper proposes REAP, a dual-module iterative framework that addresses multi-hop question answering through recursive collaboration between a Sub-task Planner (SP), which maintains a global perspective to dynamically guide reasoning trajectories, and a Fact Extractor (FE), which extracts structured facts and latent clues from retrieved content. Using Llama-3.1-8B, REAP substantially outperforms all baselines on 4 benchmarks (HotpotQA F1 68.0 vs. runner-up 63.4).

## Background & Motivation

**Background**: RAG mitigates LLM hallucinations by incorporating external knowledge, yet multi-hop question answering (MHQA) requires integrating information across multiple documents. Existing iterative RAG methods (IRCoT, Iter-RetGen, etc.) adopt multi-round retrieval with stepwise reasoning; some approaches further introduce search algorithms such as MCTS to identify optimal reasoning trajectories.

**Limitations of Prior Work**: (1) Lack of global planning — incrementally decomposing complex queries into sub-queries may lead to local reasoning dead ends; (2) Insufficient utilization of retrieved content — models tend to extract only direct answers while overlooking latent clues critical to the final answer; (3) Incorporating scorers or decision modules increases system complexity and reduces interpretability.

**Key Challenge**: Linear pipeline-style reasoning cannot handle reasoning failures or trajectory correction in multi-hop scenarios — when a reasoning step produces erroneous or incomplete information, no mechanism exists to detect and rectify the overall reasoning direction.

**Goal**: To restructure multi-hop reasoning from a linear pipeline into a dynamic, state-driven loop that preserves a global perspective while enabling error detection and trajectory correction.

**Key Insight**: Explicitly maintaining structured sub-task plans and fact lists, decoupling planning (SP) from fact collection (FE) while keeping the two tightly coordinated.

**Core Idea**: SP provides a global planning perspective to guide reasoning direction, while FE supplies high-fidelity structured facts; the two form a self-correcting reasoning loop through recursive feedback.

## Method

### Overall Architecture
Given a complex query $Q$, a Decomposer first generates a structured task plan $\mathcal{P}_0 = \{(id_i, q_i, deps_i)\}$. The system then enters an SP–FE iterative loop: SP analyzes the current state and decides the next action; FE performs retrieval and reasoning to produce structured facts; the facts are fed back to SP to update the plan. The loop continues until the plan is fully resolved, after which a Synthesizer generates the final answer.

### Key Designs

1. **Sub-task Planner (SP)**:

    - **Function**: Maintains the global task plan and dynamically guides reasoning trajectories.
    - **Mechanism**: Dispatches to two sub-modules based on the satisfaction level $l_t$ returned by FE:
        - **Plan Updater** (handles the ideal case, $l_t = \text{DirectAnswer}$): Applies deterministic rule-based updates — (a) *fact substitution*: replaces abstract placeholders in pending sub-tasks with newly obtained concrete entities; (b) *plan branching*: duplicates subsequent dependent sub-tasks into parallel branches when a sub-task yields multiple valid sub-answers.
        - **Re-Planner** (handles non-ideal cases, $l_t = \text{PartialClue/Failed}$): (a) *pragmatic sufficiency assessment*: first evaluates whether partial information is functionally sufficient for subsequent reasoning (if so, treats the sub-task as resolved to avoid perfectionist search loops); (b) *scoped plan repair*: distinguishes local issues (fine-tuning sub-queries) from systemic defects (pruning invalid branches and injecting new sub-task sequences).
    - **Design Motivation**: Routing planning by difficulty — lightweight rules handle simple updates while heavy reasoning handles complex anomalies, balancing efficiency and robustness.

2. **Fact Extractor (FE)**:

    - **Function**: Extracts high-fidelity structured facts and latent clues from retrieved documents.
    - **Mechanism**: For each sub-query $q_t$, after retrieving the top-5 documents, the LLM generates a structured fact tuple $f_t = (s_t, e_t, r_t, l_t)$ — comprising a core statement $s_t$, textual evidence $e_t \subseteq D_t$, reasoning process $r_t$ (CoT explanation), and satisfaction level $l_t$ (DirectAnswer / PartialClue / Failed). Crucially, FE conditions on historical facts $\mathcal{F}_{t-1}$ to enable cross-step coreference resolution and relation identification.
    - **Design Motivation**: (1) Identifies latent clues beyond direct answers to avoid missing critical information; (2) Structured tuples ensure traceability; (3) The satisfaction level provides a decision signal for SP.

3. **Multi-task Fine-tuning**:

    - **Function**: Improves the performance of Re-Planner, which suffers from data scarcity, through joint training.
    - **Mechanism**: The Decomposer, Plan Updater, and Re-Planner share the common capability of generating or modifying structured task plans from existing information. Their datasets are merged to train a single model $M_\phi$: $\min_\phi \sum_{task} \lambda_{task} \mathbb{E}[\mathcal{L}_{task}(M_\phi(x), y)]$. Knowledge is transferred from data-rich tasks (decomposition, updating) to the data-scarce task (re-planning).
    - **Design Motivation**: Re-Planner is invoked infrequently, resulting in scarce training data and poor standalone training performance.

### Loss & Training
- Multi-task fine-tuning: task weights $\lambda = 1$, standard language modeling loss.
- GPT-4 is used to run REAP on HotpotQA and 2WikiMultihopQA to collect 7,000 samples; after filtering, 5,556 training samples are retained.
- Inference uses Llama-3.1-8B-Instruct; retrieval uses e5-large-v2, top-5, with a maximum of 5 iterations.

## Key Experimental Results

### Main Results

| Method | HotpotQA F1 | 2Wiki F1 | MuSiQue‡ F1 | Bamboogle‡ F1 |
|--------|------------|---------|------------|--------------|
| Standard RAG | 48.6 | 38.5 | 13.5 | 30.9 |
| IRCoT | 51.4 | 36.5 | 18.6 | 30.1 |
| R1-Searcher | 63.4 | 69.4 | 33.8 | 58.0 |
| **REAP** | **68.0** | **79.6** | **38.3** | **65.2** |

### Ablation Study

| Configuration | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 | Bamboogle F1 |
|---------------|------------|---------|------------|-------------|
| w/o Replan | 64.9 | 78.6 | 34.2 | 61.6 |
| w/o Verify | 65.1 | 78.0 | 34.8 | 60.8 |
| w/o Clue | 64.6 | 76.5 | 35.2 | 62.7 |
| **REAP (full)** | **68.0** | **79.6** | **38.3** | **65.2** |

### Key Findings
- **Re-Planner contributes the most**: Its removal degrades HotpotQA by 3.1%, MuSiQue by 4.1%, and Bamboogle by 3.6%, with greater impact on more complex multi-hop datasets.
- **Largest gain on 2Wiki**: F1 improves from the runner-up score of 69.4 (R1-Searcher) to 79.6 (+10.2), demonstrating that structured planning is particularly effective for tasks requiring cross-document reasoning.
- **Strong generalization**: Trained only on HotpotQA and 2Wiki, REAP still achieves state-of-the-art performance on the out-of-domain MuSiQue and Bamboogle benchmarks.
- **Multi-task fine-tuning is effective**: Joint training (ft-all) substantially outperforms separate training (ft-separate), especially for the Re-Planner component.
- REAP with an 8B model surpasses the majority of 70B baselines.

## Highlights & Insights
- **"Pragmatic sufficiency assessment" is an elegant design**: Rather than demanding perfect completion of every sub-task, the system evaluates whether partial information is "good enough" to proceed. This prevents infinite loops on difficult sub-tasks while retaining critical information — a principle transferable to any multi-step decision-making system.
- **Design of structured fact tuples**: Packaging the answer, evidence, reasoning process, and satisfaction level into a four-tuple enables SP to route decisions based on satisfaction level, achieving loose coupling with high cohesion between modules.
- **Planning/execution separation with tight coordination**: SP performs no retrieval or reasoning; FE makes no planning decisions; yet the two interact closely through a structured interface. This separation substantially improves interpretability and debuggability.

## Limitations & Future Work
- Training data relies on 5,556 GPT-4-generated samples; data quality is bounded by GPT-4's capabilities.
- The maximum iteration count is fixed at 5, which may be insufficient for extremely complex queries requiring more hops.
- Evaluation is conducted solely on English Wikipedia corpora; performance on multilingual and domain-specific knowledge bases remains unverified.
- The criteria for Re-Planner's scoped repair strategy (local vs. systemic issues) may require more refined design.
- When retriever quality is poor (no relevant documents in top-5), the framework's error-correction capacity is limited.

## Related Work & Insights
- **vs. R1-Searcher**: R1-Searcher uses reinforcement learning to learn retrieval and generation strategies; REAP surpasses it via structured planning and fact extraction (HotpotQA F1 68.0 vs. 63.4) while offering greater interpretability.
- **vs. IRCoT**: IRCoT combines CoT with multi-round retrieval but lacks global planning and error correction. REAP substantially outperforms it across all benchmarks.
- **vs. SearChain**: SearChain also performs question decomposition with multi-round retrieval but does not explicitly maintain a global state or fact list. REAP surpasses it by over 33 F1 points on 2Wiki.

## Rating
- Novelty: ⭐⭐⭐⭐ — The SP–FE dual-module recursive feedback, pragmatic sufficiency assessment, and fact satisfaction-level routing are all valuable designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets (including 2 out-of-domain), 10 baselines, detailed ablations, and multi-task fine-tuning analysis.
- Writing Quality: ⭐⭐⭐⭐ — Formal notation is clear and the architectural diagram is intuitive.
- Value: ⭐⭐⭐⭐⭐ — A practical advancement for multi-hop RAG; an 8B model surpasses 70B baselines with strong generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAVIS: A Benchmark for Multimodal Source Attribution in Long-form Visual Question Answering](mavis_a_benchmark_for_multimodal_source_attribution_in_long-form_visual_question.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[ICML 2026\] Retriever Portfolios: A Principled Approach to Adaptive RAG](../../ICML2026/information_retrieval/retriever_portfolios_a_principled_approach_to_adaptive_rag.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](../../ACL2026/information_retrieval/mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)

</div>

<!-- RELATED:END -->
