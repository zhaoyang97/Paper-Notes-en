---
title: >-
  [Paper Note] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference
description: >-
  [ACL 2026][LLM Reasoning][Multi-Hop Reasoning] Introduces the MARCH benchmark (2,209 multi-hop ambiguous questions) and the CLARION framework, providing the first systematic study of QA challenges at the intersection of ambiguity resolution and multi-step reasoning, revealing critical shortcomings of existing SOTA models on such problems.
tags:
  - ACL 2026
  - LLM Reasoning
  - Multi-Hop Reasoning
  - Ambiguity Resolution
  - Benchmark Construction
  - Hierarchical Uncertainty
  - Agentic Framework
date: 2026-05-08
content_hash: 6d3a07e3ce322f29
---
# MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference

**Conference**: ACL 2026
**arXiv**: [2509.22750](https://arxiv.org/abs/2509.22750)
**Code**: [GitHub](https://github.com/jeonghyunpark2002/MARCH)
**Area**: LLM Reasoning / Question Answering
**Keywords**: multi-hop reasoning, ambiguity resolution, benchmark construction, layered uncertainty, agent framework

## TL;DR

This paper introduces the MARCH benchmark (2,209 multi-hop ambiguous questions) and the CLARION framework, presenting the first systematic study of QA challenges at the intersection of ambiguity interpretation and multi-step reasoning, and revealing severe deficiencies in existing SOTA models on such problems.

## Background & Motivation

**State of the Field**: Multi-hop QA requires models to construct reasoning chains across multiple documents; ambiguous QA requires models to handle polysemous terms and underspecified contexts. Both challenges have been extensively studied in isolation, but their **intersection** remains almost entirely unexplored.

**Limitations of Prior Work**: In real-world user queries, 48.4% contain ambiguity, 17.7% involve multi-hop reasoning, and 13.3% involve both simultaneously. Existing benchmarks, however, focus either solely on single-hop ambiguity (ASQA) or solely on multi-hop reasoning (MuSiQue). When ambiguity arises in intermediate steps of multi-hop reasoning, uncertainty cascades and amplifies—early errors in ambiguity resolution lock the model into incorrect reasoning paths.

**Root Cause**: Ambiguity in multi-hop reasoning can be **latent**—it only becomes apparent after prior steps have been correctly resolved. For example, in the question "What is the best-selling pickup sold by the manufacturer of the 'Mustang'?", the ambiguity of *pickup* (truck vs. guitar pickup) is only discoverable after both interpretations of *Mustang* (car vs. guitar) are retained.

**Paper Goals**: (1) Construct a dedicated benchmark for evaluating multi-hop ambiguous QA; (2) Propose a framework for addressing this problem.

**Core Idea**: Multi-hop ambiguous QA requires models to maintain a "superposition" of multiple interpretation paths throughout the entire reasoning chain, rather than committing prematurely to a single interpretation. CLARION prevents premature pruning of reasoning paths by decoupling ambiguity planning from evidence retrieval.

## Method

### Overall Architecture

The paper makes two contributions: (1) the MARCH benchmark—constructed from MuSiQue via a four-stage pipeline, yielding 2,209 multi-hop ambiguous questions spanning semantic, syntactic, and constraint ambiguity types; and (2) the CLARION framework—a two-stage agent framework in which a Planning Agent first maps all plausible interpretation paths, and a Reasoning Agent then retrieves evidence and performs inference along each path independently.

### Key Designs

1. **MARCH Benchmark Construction Pipeline**:

    - Function: Provides a high-quality evaluation benchmark for multi-hop ambiguous QA.
    - Mechanism: Four-stage construction—(a) unanimous detection of ambiguity types using four LLMs (GPT-4.1, Llama-4, Qwen3-235B, Claude-4); (b) decomposition of clarification questions into atomic sub-questions with Wikipedia evidence retrieval; (c) generation of short answers and comprehensive long answers for each interpretation; (d) consistency-based filtering using three independent LLMs. The final benchmark retains 2,209 samples.
    - Design Motivation: Unanimous multi-LLM detection reduces single-model bias; human validation (Fleiss' $\kappa = 0.92$–$0.95$) ensures label quality.

2. **Multi-hop Ambiguity Taxonomy**:

    - Function: Provides a systematic classification and handling guide for multi-hop ambiguity.
    - Mechanism: Extends standard ambiguity taxonomy to the multi-hop setting—(a) **Semantic ambiguity**: a single mention maps to multiple entities (e.g., *Mustang* → Ford automobile / Fender guitar), requiring *Interpret*; (b) **Syntactic ambiguity**: multiple valid parses yield different inter-hop dependencies (e.g., prepositional phrase attachment), requiring *Resolve*; (c) **Constraint ambiguity**: overly specific qualifiers prune otherwise valid reasoning paths, requiring *Generalize*.
    - Design Motivation: Different ambiguity types demand different resolution strategies; the taxonomy provides principled guidance for method design.

3. **CLARION Framework (CLarifying Ambiguity with Reasoning and InstructiON)**:

    - Function: Addresses multi-hop ambiguous QA by decoupling ambiguity planning from evidential reasoning.
    - Mechanism: Two stages—(a) **Planning Agent**: receives the original question, identifies all ambiguity points, and generates a planning graph covering all plausible interpretation paths, ensuring no legitimate interpretation is discarded; (b) **Reasoning Agent**: independently retrieves evidence and performs inference along each planned path, then synthesizes results across all paths to produce a comprehensive answer.
    - Design Motivation: Standard RAG/CoT approaches tend to commit to a single interpretation at the first hop (premature commitment); CLARION's plan-then-execute separation prevents premature path pruning.

### Loss & Training

Both MARCH and CLARION are training-free. MARCH is a construction-based benchmark; CLARION operates via prompt engineering on existing LLMs. Evaluation employs F1-score, EM, D-F1 (disambiguation F1), ROUGE-L, and LLM-judge metrics.

## Key Experimental Results

### Main Results

| Setting | MuSiQue (multi-hop) | ASQA (ambiguity) | MARCH (intersection) | Notes |
|---|---|---|---|---|
| Existing models | Acceptable | Acceptable | **Significant drop** | Intersection setting far exceeds the difficulty of either individual setting |
| CLARION | — | — | **Significantly outperforms baselines** | Validates the effectiveness of the decoupled strategy |

### Benchmark Statistics

| Metric | Value | Notes |
|---|---|---|
| Total samples | 2,209 | Covers three ambiguity types |
| Ambiguity distribution | Sem: 734, Syn: 739, Const: 736 | Balanced distribution |
| Average hop count | 2.11–2.95 | Syntactic ambiguity has the most hops |
| Human validation agreement | Fleiss' $\kappa = 0.92$–$0.95$ | Extremely high inter-annotator agreement |
| Long-answer validity rate | >90% | Integrates all interpretations |

### Key Findings
- 13.3% of real-world user queries simultaneously involve multi-hop reasoning and ambiguity—this is not a rare edge case.
- Models that perform reasonably on standalone multi-hop or ambiguity tasks suffer sharp performance degradation on the intersection setting (MARCH).
- Models tend to commit to a single interpretation at the first hop (premature commitment), leading to cascading errors.
- CLARION's plan-execute decoupling effectively prevents premature pruning of reasoning paths.

## Highlights & Insights
- **Depth of problem formulation**: The concept of *latent ambiguity*—ambiguity that only becomes apparent after prior steps are correctly resolved—is a particularly insightful contribution.
- **Three-type ambiguity taxonomy**: Each of semantic, syntactic, and constraint ambiguity maps to a distinct resolution action (Interpret / Resolve / Generalize), making the taxonomy operationally useful.
- **Rigorous benchmark construction**: Unanimous agreement among four LLMs combined with human validation ensures high reliability.
- **The 13.3% real-world frequency statistic**: Derived from lmsys-chat-1m, this figure compellingly motivates the practical importance of the problem.
- **Clean design of CLARION**: The plan-then-execute decoupling is conceptually simple yet effective.

## Limitations & Future Work
- MARCH is constructed on top of MuSiQue, inheriting its limitations in domain coverage and hop count.
- CLARION currently operates in a retrieval-augmented setting; open-domain scenarios without retrieval remain unexplored.
- The balanced distribution across the three ambiguity types is artificially controlled and may not fully reflect real-world distributions.
- Future work could investigate the conditions under which a model should proactively seek clarification rather than attempting to cover all interpretations.
- Extension to multilingual multi-hop ambiguity settings is a natural direction.

## Related Work & Insights
- **vs. ASQA**: ASQA addresses only single-hop ambiguity; MARCH is the first to extend the setting to multi-hop ambiguity.
- **vs. MuSiQue**: MuSiQue targets multi-hop reasoning but assumes the absence of ambiguity; MARCH introduces ambiguity on top of its foundation.
- **vs. standard RAG/CoT**: Standard approaches fail on multi-hop ambiguity due to premature commitment; CLARION resolves this through plan-execute decoupling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic definition and evaluation of multi-hop ambiguous QA, addressing an important and previously unstudied problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Encompasses benchmark construction, human validation, model evaluation, and framework comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear, the taxonomy is rigorous, and illustrative examples are vivid.
- Value: ⭐⭐⭐⭐⭐ Both the benchmark and the framework constitute independent contributions with significant impact on the reasoning and ambiguity research communities.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Explicit Trait Inference for Multi-Agent Coordination](explicit_trait_inference_for_multi-agent_coordination.md)
- [\[ACL 2026\] Failure Modes in Multi-Hop QA: The Weakest Link Effect and the Recognition Bottleneck](failure_modes_in_multi-hop_qa_the_weakest_link_effect_and_the_recognition_bottle.md)
- [\[ACL 2025\] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](../../ACL2025/llm_reasoning/beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)
- [\[ACL 2026\] Chain-of-Thought as a Lens: Evaluating Structured Reasoning Alignment between Human Preferences and Large Language Models](chain-of-thought_as_a_lens_evaluating_structured_reasoning_alignment_between_hum.md)
- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](../../NeurIPS2025/llm_reasoning/value-guided_search_for_efficient_chain-of-thought_reasoning.md)

<!-- RELATED:END -->
