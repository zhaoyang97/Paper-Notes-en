---
title: >-
  [Paper Note] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning
description: >-
  [AAAI 2026][Dual-system reasoning] Inspired by dual-process cognitive theory, PRIME is a multi-agent reasoning framework in which a Quick Thinking Agent (System 1) rapidly generates intuitive answers, a Reflection Agent evaluates their confidence, and—when uncertainty is detected—six specialized System 2 agents (Planning / Search / Reading / Hypothesis / Integration / Decision) are triggered for deep knowledge-retrieval reasoning. The framework enables open-source LLaMA 3 to approach GPT-4o performance on medical and multi-hop QA benchmarks.
tags:
  - AAAI 2026
  - Dual-system reasoning
  - fast-and-slow thinking
  - retrieval augmentation
  - multi-agent
  - planning
date: 2026-05-08
content_hash: 7d3480fb3e6274c6
---

# PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning

**Conference**: AAAI 2026
**arXiv**: [2509.22315](https://arxiv.org/abs/2509.22315)
**Code**: Available
**Area**: Information Retrieval
**Keywords**: Dual-system reasoning, fast-and-slow thinking, retrieval augmentation, multi-agent, planning

## TL;DR
Inspired by dual-process cognitive theory, PRIME is a multi-agent reasoning framework in which a Quick Thinking Agent (System 1) rapidly generates intuitive answers, a Reflection Agent evaluates their confidence, and—when uncertainty is detected—six specialized System 2 agents (Planning / Search / Reading / Hypothesis / Integration / Decision) are triggered for deep knowledge-retrieval reasoning. The framework enables open-source LLaMA 3 to approach GPT-4o performance on medical and multi-hop QA benchmarks.

## Background & Motivation

**State of the Field**: LLM reasoning enhancement methods include CoT, RAG, and agent frameworks, among others. However, most approaches uniformly apply slow reasoning to all questions, wasting computational resources.

**Limitations of Prior Work**:
   - Simple questions do not require deep reasoning—invoking System 2 for "What is the capital of France?" is wasteful.
   - Existing RAG methods lack explicit planning—what to retrieve and when to retrieve it are left unaddressed.
   - Single-agent reasoning lacks specialization—the same model must simultaneously handle search, reasoning, and verification.

**Root Cause**: Deep reasoning is effective but expensive—intelligent selection of when to activate it is required.

**Paper Goals**: Design a multi-agent framework that adaptively triggers deep reasoning.

**Starting Point**: Kahneman's dual-process theory—System 1 for fast intuition, System 2 for slow analysis—with a Reflection Agent deciding when to switch.

**Core Idea**: System 1 fast answering + Reflection self-evaluation + System 2 six-agent deep reasoning = efficient and accurate inference.

## Method

### Overall Architecture
Input question → Quick Thinking Agent (decompose sub-questions, answer sequentially) → Reflection Agent (self-assess confidence) → output if confident → otherwise trigger System 2: Planning Agent (formulate reasoning plan) → Hypothesis Agent (generate hypotheses) → Search Agent (retrieve evidence) → Reading Agent (careful extraction) → Integration Agent (merge evidence) → Decision Agent (final judgment).

### Key Designs

1. **Quick Thinking Agent (System 1)**:

    - **Function**: Rapidly generate intuitive answers.
    - **Mechanism**: Decompose the question into sub-questions and answer them sequentially without external retrieval.
    - **Design Motivation**: The majority of questions can be handled by System 1, avoiding unnecessary expensive reasoning.

2. **Reflection Agent (switching gate)**:

    - **Function**: Assess the confidence of System 1 outputs.
    - **Mechanism**: Explicit self-reflection—check whether the answer contains logical gaps, uncertainty, or reliance on unverified assumptions.
    - **Design Motivation**: The key innovation—determines when to switch from fast to slow thinking.

3. **System 2 six-agent reasoning pipeline**:

    - **Function**: Deep knowledge retrieval and multi-step reasoning.
    - **Division of labor**: Planning (formulate reasoning path) → Hypothesis (generate hypotheses) → Search (external retrieval) → Reading (careful evidence reading) → Integration (multi-source synthesis) → Decision (final judgment).
    - **Design Motivation**: Each agent focuses on a single cognitive sub-task—planning ≠ search ≠ reasoning.

### Loss & Training
- No training required—purely prompt-based agent coordination.
- Backbone model: LLaMA 3 (8B / 70B).

## Key Experimental Results

### Medical Reasoning Tasks

| Model / Method | MedQA | MedMCQA | MMLU-Medical | Avg. |
|----------------|-------|---------|--------------|------|
| LLaMA3.1 8B + CoT | 61.51 | 55.15 | 71.63 | 62.76 |
| LLaMA3.1 8B + MedRAG | 63.00 | 56.87 | 74.56 | 64.81 |
| LLaMA3.1 8B + Search-O1 | 73.13 | 62.13 | 79.16 | 71.47 |
| **LLaMA3.1 8B + PRIME** | **76.91** | **67.49** | **83.56** | **75.99** |
| LLaMA3.3 70B + Search-O1 | 83.17 | 73.11 | 87.23 | 81.17 |
| **LLaMA3.3 70B + PRIME** | **87.51** | **78.94** | **92.74** | **86.39** |
| GPT-4 | 83.97 | 69.88 | 89.44 | 81.10 |
| GPT-4o | 85.55 | 74.71 | 90.45 | 83.57 |

- PRIME enables LLaMA3.3 70B to achieve an average score of 86.39%, **surpassing both GPT-4 (81.10%) and GPT-4o (83.57%)**.

### Multi-Hop Reasoning Tasks

| Method | Musique F1 | 2Wiki F1 | HotpotQA F1 |
|--------|-----------|---------|------------|
| Naive RAG | 30.52 | 38.22 | 40.06 |
| Search-O1 | 41.94 | 74.24 | 54.81 |
| **PRIME** | **48.81** | **79.81** | **60.68** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| System 1 only | Strong on simple questions, poor on hard ones—over 80% answered correctly but severe hallucination on the remaining 20% |
| System 2 only (full deep reasoning) | Accurate but slow—computational resources wasted |
| **PRIME (adaptive switching)** | **Accurate and efficient—approximately 60% of questions handled by System 1** |

### Key Findings
- LLaMA3.3 70B + PRIME surpasses GPT-4o (86.39 vs. 83.57)—the multi-agent framework bridges the open- vs. closed-source gap.
- LLaMA3.1 8B + PRIME (75.99) outperforms GPT-4o-mini (74.59)—even an 8B model benefits substantially.
- Approximately 60% of questions are resolved by System 1, saving significant computational resources.
- The quality of the Reflection Agent is critical—false triggers waste System 2 resources; missed triggers lead to incorrect answers.
- On multi-hop reasoning, PRIME outperforms Search-O1 by an average of 6–7 F1 points—reflecting the combined advantage of knowledge retrieval and hypothesis testing.

## Highlights & Insights
- A faithful implementation of dual-process theory in LLMs—not a metaphor but an actual architectural design.
- The specialization of six agents yields more reliable deep reasoning than a single-agent approach.
- Direct practical guidance for reasoning efficiency: not every question warrants deliberate, slow thinking.

## Limitations & Future Work
- The Reflection Agent may misjudge—false negatives (hard questions failing to trigger System 2) and false positives (easy questions incorrectly triggering it) both degrade efficiency.
- The communication overhead of six agents is not cost-effective for simple questions—further cost analysis of System 2 is needed.
- Validation is limited to QA tasks; performance on creative writing, code generation, and other scenarios remains unknown.
- The Search Agent depends on external retrieval quality—retrieval failure can cause the entire System 2 pipeline to fail.
- No persistent memory mechanism exists—similar questions must be reasoned through from scratch each time.

## Related Work & Insights
- **vs. ReAct**: Single-agent loop. PRIME uses multi-agent coordination with adaptive triggering.
- **vs. Self-Consistency**: Multiple sampling with majority voting. PRIME performs one round of deep reasoning.
- The dual-system framework is generalizable to any scenario requiring an efficiency–accuracy trade-off.

## Rating
- Novelty: ⭐⭐⭐⭐ Faithful implementation of dual-system + multi-agent design
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-domain QA + ablation study
- Writing Quality: ⭐⭐⭐⭐ Cognitive-science motivation clearly articulated
- Value: ⭐⭐⭐⭐ Practical value for efficient LLM reasoning

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[AAAI 2026\] ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[AAAI 2026\] Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)

<!-- RELATED:END -->
