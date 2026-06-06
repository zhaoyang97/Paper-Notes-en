---
title: >-
  [Paper Note] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference
description: >-
  [ACL 2026][LLM Evaluation][Multi-hop Reasoning] Proposes the MARCH benchmark (2,209 multi-hop ambiguous questions) and the CLARION framework to systematically investigate QA challenges at the intersection of ambiguity re…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Multi-hop Reasoning"
  - "Ambiguity Resolution"
  - "Benchmark Construction"
  - "Hierarchical Uncertainty"
  - "Agentic Frameworks"
date: 2026-05-08
content_hash: 4089f8cf294b6b80
---

# MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference

**Conference**: ACL 2026  
**arXiv**: [2509.22750](https://arxiv.org/abs/2509.22750)  
**Code**: [GitHub](https://github.com/jeonghyunpark2002/MARCH)  
**Area**: LLM Reasoning / Question Answering  
**Keywords**: Multi-hop Reasoning, Ambiguity Resolution, Benchmark Construction, Hierarchical Uncertainty, Agentic Frameworks

## TL;DR

Proposes the MARCH benchmark (2,209 multi-hop ambiguous questions) and the CLARION framework to systematically investigate QA challenges at the intersection of ambiguity resolution and multi-step reasoning for the first time, revealing significant deficiencies in existing SOTA models on such problems.

## Background & Motivation

**Background**: Multi-hop QA requires models to construct logical chains across multiple documents; ambiguity QA requires models to handle polysemy and insufficient context. While these two challenges have been extensively studied separately, their **intersection** remains almost entirely unexplored.

**Limitations of Prior Work**: In real user queries, 48.4% contain ambiguity, 17.7% involve multi-hop reasoning, and 13.3% simultaneously involve both. However, existing benchmarks either focus solely on single-hop ambiguity (ASQA) or single-path multi-hop reasoning (MuSiQue). When ambiguity occurs at intermediate steps of multi-hop reasoning, uncertainty cascades—early interpretation errors lock in incorrect reasoning paths.

**Key Challenge**: Ambiguity in multi-hop reasoning can be **latent**—it only manifests after previous steps are correctly resolved. For example, in "What is the best-selling pickup sold by the manufacturer of the 'Mustang'?", the ambiguity of "pickup" (truck vs. guitar pickup) can only be discovered after retaining both interpretations of "Mustang" (car vs. guitar).

**Goal**: (1) Construct a specialized benchmark for evaluating multi-hop ambiguous QA; (2) Propose a framework to address this problem.

**Core Idea**: Multi-hop ambiguous QA requires models to maintain a "superposition" of multiple interpretation paths throughout the entire reasoning chain rather than committing to a single interpretation prematurely. CLARION prevents premature pruning of reasoning paths by decoupling ambiguity planning from evidence retrieval.

## Method

### Overall Architecture

This work presents two contributions: (1) The MARCH benchmark—constructed from MuSiQue via a four-stage pipeline, resulting in 2,209 multi-hop ambiguous questions covering semantic, syntactic, and constraint ambiguities; (2) The CLARION framework—a two-stage agentic framework where a Planning Agent first maps out all possible interpretation paths, followed by a Reasoning Agent that retrieves evidence and reasons for each path independently.

### Key Designs

1.  **MARCH Benchmark Construction Pipeline**:

    - **Function**: Provides a high-quality evaluation benchmark for multi-hop ambiguous QA.
    - **Mechanism**: A four-stage process—(a) Unanimous detection of ambiguity types using 4 LLMs (GPT-4.1, Llama-4, Qwen3-235B, Claude-4); (b) Decomposition of clarifying questions into atomic sub-questions and evidence retrieval from Wikipedia; (c) Generation of short answers and comprehensive long-form answers for each interpretation; (d) Consistency filtering using 3 independent LLMs. Finally, 2,209 samples are retained.
    - **Design Motivation**: Unanimous detection by multiple LLMs reduces single-model bias, and manual verification (Fleiss' $\kappa=0.92-0.95$) ensures high label quality.

2.  **Multi-hop Ambiguity Taxonomy**:

    - **Function**: Provides a systematic classification and processing guide for multi-hop ambiguity.
    - **Mechanism**: Extends standard ambiguity classification to multi-hop scenarios—(a) **Semantic Ambiguity**: The same mention maps to multiple entities (e.g., Mustang → Ford vehicle/Fender guitar), requiring "Interpretation"; (b) **Syntactic Ambiguity**: Multiple legal parses lead to different inter-hop dependencies (e.g., prepositional attachment ambiguity), requiring "Resolution"; (c) **Constraint Ambiguity**: Overly specific constraints cause legal reasoning paths to be pruned, requiring "Generalization".
    - **Design Motivation**: Different types of ambiguity require different processing strategies; the taxonomy provides theoretical guidance for method design.

3.  **CLARION Framework (CLarifying Ambiguity with Reasoning and InstructiON)**:

    - **Function**: Handles multi-hop ambiguous QA by decoupling ambiguity planning and evidence reasoning.
    - **Mechanism**: Two stages—(a) **Planning Agent**: Receives the original question, identifies all ambiguity points, and generates a planning graph of all possible interpretation paths, ensuring every legal interpretation is preserved; (b) **Reasoning Agent**: Conducts evidence retrieval and reasoning independently along each planning path, finally synthesizing results from all paths to generate a complete answer.
    - **Design Motivation**: Standard RAG/CoT methods tend to lock in a single interpretation at the first hop (premature commitment). CLARION prevents premature path pruning through the separation of planning and execution.

### Loss & Training

Both MARCH and CLARION are training-free solutions. MARCH is a constructed benchmark, and CLARION is implemented via prompt engineering on existing LLMs. Evaluation utilizes metrics such as F1-score, EM, D-F1 (disambiguation F1), ROUGE-L, and LLM-as-a-judge.

## Key Experimental Results

### Main Results

| Setting | MuSiQue (Multi-hop) | ASQA (Ambiguity) | MARCH (Intersection) | Description |
| :--- | :--- | :--- | :--- | :--- |
| Existing Models | Acceptable | Acceptable | **Significant Drop** | Intersection scenarios are much harder than single-challenge ones |
| CLARION | - | - | **Significantly outperforms baselines** | Validates the effectiveness of the decoupling strategy |

### Benchmark Statistics

| Metric | Value | Description |
| :--- | :--- | :--- |
| Total Samples | 2,209 | Covers three types of ambiguity |
| Ambiguity Distribution | Sem:734, Syn:739, Const:736 | Balanced distribution |
| Average Hops | 2.11-2.95 | Syntactic ambiguity has the most hops |
| Manual Verification Consistency | Fleiss' $\kappa=0.92-0.95$ | Extremely high annotation agreement |
| Long Answer Validity | >90% | Integrates all interpretations |

### Key Findings
- 13.3% of real-world user queries involve both multi-hop reasoning and ambiguity; this is not a rare edge case.
- Models that perform reasonably on isolated multi-hop or ambiguity tasks show a sharp performance decline in the intersection scenario (MARCH).
- Models exhibit a tendency toward "premature commitment," locking into a single interpretation at the first hop, leading to cascading errors.
- The planning-execution decoupling in CLARION effectively prevents the premature pruning of reasoning paths.

## Highlights & Insights
- **Depth of Problem Definition**: The concept of "latent ambiguity" (ambiguity that only manifests if antecedent steps are correctly resolved) is highly insightful.
- **Tripartite Ambiguity Taxonomy**: Defining specific actions (Interpret/Resolve/Generalize) for semantic, syntactic, and constraint ambiguities makes the classification system practically useful.
- **Rigorous Construction Pipeline**: The use of 4-LLM unanimous consensus plus manual verification ensures high quality and reliability.
- **Real-world Significance**: Statistical data from lmsys-chat-1m (13.3% frequency) provides a strong argument for the practical importance of the problem.
- **Elegance of CLARION**: The idea of decoupling planning from execution is simple yet effective for handling branching logic.

## Limitations & Future Work
- Since MARCH is built upon MuSiQue, it inherits the domain and hop-count limitations of the original dataset.
- CLARION currently operates in a retrieval-augmented setting; the open-domain scenario without retrieval remains unexplored.
- The balanced distribution of the three ambiguity types is artificially controlled and may not perfectly reflect the natural distribution.
- Future work could explore conditions under which a model should proactively ask clarifying questions instead of attempting all interpretations.
- The benchmark could be extended to multilingual multi-hop ambiguity scenarios.

## Related Work & Insights
- **vs ASQA**: ASQA focuses only on single-hop ambiguity, while MARCH extends this to the multi-hop domain for the first time.
- **vs MuSiQue**: MuSiQue focuses on multi-hop reasoning but assumes no ambiguity; MARCH introduces ambiguity on top of it.
- **vs Standard RAG/CoT**: Standard methods fail in multi-hop ambiguous scenarios due to premature commitment, which CLARION resolves via decoupled planning and execution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic definition and evaluation of multi-hop ambiguous QA; the problem is important and previously understudied.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes benchmark construction, manual validation, model evaluation, and framework comparison.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definitions are clear, the taxonomy is rigorous, and case analyses are vivid.
- Value: ⭐⭐⭐⭐⭐ Both the benchmark and framework provide independent contributions with significant impact on the reasoning and ambiguity research communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](../../ICLR2026/llm_evaluation/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[ACL 2026\] CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems](clarity_a_framework_and_benchmark_for_conversational_language_ambiguity_and_unan.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)

</div>

<!-- RELATED:END -->
