---
title: >-
  [Paper Note] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation
description: >-
  [ACL 2026][Multi-Agent RAG] This paper proposes MASS-RAG, a training-free multi-agent synthesis RAG framework that employs three specialized filtering agents—Summarizer, Extractor, and Reasoner—to process retrieved documents from complementary perspectives, followed by a Synthesis Agent that integrates multi-perspective evidence or candidate answers, consistently outperforming strong baselines across four benchmarks.
tags:
  - ACL 2026
  - Multi-Agent RAG
  - Evidence Synthesis
  - Training-Free
  - Multi-Perspective Filtering
  - Heterogeneous Evidence Fusion
date: 2026-05-08
content_hash: 2c32b076e0e10a65
---

# MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation

**Conference**: ACL 2026
**arXiv**: [2604.18509](https://arxiv.org/abs/2604.18509)
**Code**: None
**Area**: Information Retrieval / RAG
**Keywords**: Multi-Agent RAG, Evidence Synthesis, Training-Free, Multi-Perspective Filtering, Heterogeneous Evidence Fusion

## TL;DR

This paper proposes MASS-RAG, a training-free multi-agent synthesis RAG framework that employs three specialized filtering agents—Summarizer, Extractor, and Reasoner—to process retrieved documents from complementary perspectives, followed by a Synthesis Agent that integrates multi-perspective evidence or candidate answers, consistently outperforming strong baselines across four benchmarks.

## Background & Motivation

**State of the Field**: RAG enhances the factuality of LLMs by introducing external knowledge at inference time. However, when retrieved contexts are noisy, incomplete, or heterogeneous, a single generation process struggles to effectively coordinate evidence.

**Limitations of Prior Work**: (1) Existing multi-agent RAG systems (e.g., Chang et al. 2024) rely on a single judge agent to filter context from a single perspective, failing to capture complementary or heterogeneous factual evidence; (2) irrelevant or redundant retrieved information degrades generation quality; (3) for questions requiring cross-document aggregation of complementary evidence, a single perspective is particularly inadequate.

**Root Cause**: Retrieved documents may contain relevant evidence in different forms—some requiring summarization, some requiring precise extraction, and some requiring inferential reasoning—making a single filtering strategy insufficient to handle all cases.

**Paper Goals**: To design a multi-perspective evidence filtering and synthesis mechanism that enables RAG systems to process and integrate retrieved documents from complementary angles.

**Starting Point**: Evidence processing is decomposed into three complementary perspectives—summarization (compressing while preserving semantics), extraction (verbatim extraction of precise evidence), and reasoning (inferring implicit relationships)—realized through a division of labor among multiple agents.

**Core Idea**: Different types of questions require different types of evidence processing. MASS-RAG generates multiple evidence views in parallel through multiple agents and then produces a more robust final answer via explicit comparison and integration.

## Method

### Overall Architecture

MASS-RAG operates in three stages: (1) **Evidence Distillation**—three agents (Summarizer, Extractor, Reasoner) independently extract denoised, query-relevant evidence from retrieved documents; (2) **Candidate Answer Generation (optional)**—an Answer Agent independently generates candidate answers based on each filtered result; (3) **Final Synthesis**—a Synthesis Agent integrates the three evidence views or three candidate answers to produce a unified final prediction.

### Key Designs

1. **Three-Perspective Filtering Agent Design**:

    - **Function**: Extract query-relevant evidence from three complementary perspectives.
    - **Mechanism**: The Summarizer compresses retrieved documents into concise, semantically coherent summaries $R_i^{(s)} = \mathcal{A}_{\text{sum}}(q_i, D)$; the Extractor verbatim extracts precise factual spans $R_i^{(e)} = \mathcal{A}_{\text{ext}}(q_i, D)$; the Reasoner infers implicit cross-document relationships $R_i^{(r)} = \mathcal{A}_{\text{rea}}(q_i, D)$.
    - **Design Motivation**: Different question types are suited to different evidence processing strategies—factoid questions call for precise extraction, synthesis questions call for reasoning, and informational questions call for summarization. Multi-perspective coverage ensures that at least one strategy captures the correct evidence.

2. **Optional Answer Agent + Synthesis**:

    - **Function**: Reconcile competing hypotheses through explicit comparison of intermediate candidate answers.
    - **Mechanism**: When the Answer Agent is enabled, each filtered result independently generates a candidate answer $A_i^{(j)} = \mathcal{A}_{\text{ans}}(q_i, R_i^{(j)})$, and the Synthesis Agent then compares and integrates the three candidates; when disabled, the three evidence representations are integrated directly.
    - **Design Motivation**: For factoid QA, candidate answers carry rich semantic signals and different perspectives may yield complementary or competing hypotheses. For multiple-choice tasks, intermediate candidate answer signals are limited and this step can be skipped.

3. **Training-Free Role Specialization**:

    - **Function**: Achieve agent role differentiation without fine-tuning.
    - **Mechanism**: Each agent is specialized through carefully designed role prompts and output constraints—the Summarizer is constrained to compress, the Extractor to verbatim extraction, and the Reasoner to generate intermediate inferential representations.
    - **Design Motivation**: A training-free design enables plug-and-play deployment on any LLM, lowering the barrier to adoption.

### Loss & Training

MASS-RAG is a training-free framework. All agents share the same LLM backbone (Llama-3-8B and Llama-2-7B/13B in experiments) and are differentiated solely through role prompts.

## Key Experimental Results

### Main Results

**Accuracy on Four Benchmarks (Llama-3-8B + Retrieval)**

| Benchmark | Vanilla RAG | Chang et al. (Single-Agent Filtering) | MASS-RAG |
|-----------|-------------|---------------------------------------|----------|
| TriviaQA | ~70 | ~72 | ~74 |
| PopQA | ~48 | ~52 | ~55 |
| ARC-C | ~60 | ~63 | ~66 |

### Key Findings

- MASS-RAG shows the greatest advantage in scenarios requiring cross-document aggregation of complementary evidence.
- The Answer Agent provides significant gains for factoid QA (TriviaQA/PopQA) but offers limited benefit for multiple-choice tasks (ARC-C).
- Among the three filtering agents, the Reasoner contributes most individually, indicating that cross-document reasoning is the primary bottleneck.

## Highlights & Insights

- The multi-perspective filtering approach is intuitively well-motivated—different questions genuinely require different evidence processing strategies, and a one-size-fits-all filtering approach inevitably discards certain types of evidence.
- The training-free design enables plug-and-play deployment directly on any LLM.
- The optional Answer Agent provides task-adaptive flexibility.

## Limitations & Future Work

- The multi-agent design increases inference cost (3× filtering + synthesis).
- All agents share the same LLM backbone, with role differentiation implemented solely through prompting, limiting the degree of specialization achievable.
- Fair comparison with training-based RAG methods (e.g., Self-RAG) under equivalent conditions is absent.
- Performance on long-document QA scenarios is insufficiently validated.

## Related Work & Insights

- **vs. Self-RAG**: A training-based method; MASS-RAG is training-free but incurs additional inference overhead.
- **vs. Chang et al.**: Single-agent filtering; MASS-RAG compensates for the blind spots of a single perspective through multi-perspective coverage.
- **vs. REPLUG/Self-RAG**: Those works focus on retrieval strategy optimization; MASS-RAG focuses on post-retrieval evidence processing optimization.

## Rating

- **Novelty**: ⭐⭐⭐ The multi-agent filtering idea is valuable but not a breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four benchmarks, ablations, and multiple model sizes, though fair comparison with training-based methods is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework presentation with well-motivated design choices.
- **Value**: ⭐⭐⭐⭐ Offers practical improvements to evidence processing in RAG systems.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)
- [\[ACL 2026\] Beyond Black-Box Interventions: Latent Probing for Faithful Retrieval-Augmented Generation](beyond_black-box_interventions_latent_probing_for_faithful_retrieval-augmented_g.md)

<!-- RELATED:END -->
