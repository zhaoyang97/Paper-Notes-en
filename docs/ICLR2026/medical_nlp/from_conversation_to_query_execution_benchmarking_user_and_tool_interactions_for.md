---
title: >-
  [Paper Note] From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents
description: >-
  [ICLR 2026][Medical NLP][EHR] This paper proposes EHR-ChatQA, the first benchmark to evaluate the end-to-end interactive workflow of database agents in electronic health record (EHR) settings — covering ambiguity clarifi…
tags:
  - "ICLR 2026"
  - "Medical NLP"
  - "EHR"
  - "Database Agent"
  - "Interactive QA"
  - "Query Ambiguity"
  - "Value Mismatch"
date: 2026-05-08
content_hash: 7c00b1fcf835ed47
---

# From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents

**Conference**: ICLR 2026
**arXiv**: [2509.23415](https://arxiv.org/abs/2509.23415)  
**Code**: [GitHub](https://github.com/glee4810/EHR-ChatQA)  
**Area**: Medical Informatics / Agent
**Keywords**: EHR, Database Agent, Interactive QA, Query Ambiguity, Value Mismatch

## TL;DR
This paper proposes EHR-ChatQA, the first benchmark to evaluate the end-to-end interactive workflow of database agents in electronic health record (EHR) settings — covering ambiguity clarification, terminology mismatch resolution, SQL generation, and answer return. Evaluation reveals that the strongest model (o4-mini) achieves Pass@5 above 90% but suffers a substantial drop in Pass∧5 (all-success rate), with a gap of up to 60%, exposing critical robustness deficiencies in safety-sensitive clinical domains.

## Background & Motivation

**Background**: LLM-based agents are increasingly applied to structured database interaction. Text-to-SQL benchmarks such as Spider and BIRD evaluate single-turn natural language–to–SQL translation but do not address interactive scenarios.

**Limitations of Prior Work**: (1) **Query ambiguity**: Clinical users frequently pose underspecified questions (e.g., "show me the recent lab results") lacking precise constraints; (2) **Value mismatch**: Clinical terminology often diverges from database entries (e.g., "Lopressor" vs. "metoprolol tartrate"); (3) Existing benchmarks do not assess agents' end-to-end capability to clarify user intent across multiple turns, invoke tools to resolve mismatches, and generate correct SQL.

**Key Challenge**: Single-turn SQL generation is insufficient — real clinical scenarios require agents to proactively ask clarifying questions, search database values, and invoke external knowledge to bridge the gap between user intent and database content.

**Key Insight**: The paper constructs a complete interactive environment — comprising an LLM-simulated user, a tool suite, and a validator — to evaluate the full pipeline from conversation to query execution.

## Method

### Benchmark Construction

1. **Two Interaction Flows**:
    - IncreQA (Incremental Query Refinement): The user progressively adds constraints; the agent must maintain context toward a linear objective.
    - AdaptQA (Adaptive Query Adjustment): The user adjusts search targets based on intermediate results; the agent must handle branching strategies.

2. **Tool Suite**: Schema search, column search, value substring search, value similarity search, web search, and SQL execution.

3. **Simulated User**: Gemini-2.0-Flash (temperature 1.0) with a nested validation–reflection mechanism to ensure behavioral consistency.

4. **Simulation Validator**: Detects whether the simulated user deviates from instructions; reruns the session upon deviation.

### Key Metrics
- **Pass@5**: At least one success in five trials (optimistic evaluation).
- **Pass∧5**: All five trials succeed (robustness evaluation).
- **Gap** = Pass@5 − Pass∧5: Measures instability.

### Dataset Scale
- 366 task instances based on two real-world EHR databases: MIMIC-IV and eICU.
- Covers diverse query ambiguity patterns and value mismatch scenarios.

## Key Experimental Results

### Main Results

| Model | IncreQA Pass@5 | IncreQA Pass∧5 | AdaptQA Pass@5 | AdaptQA Pass∧5 |
|-------|---------------|----------------|----------------|----------------|
| o4-mini | >90% | ~50% | 60–70% | ~20% |
| Gemini-2.5-Flash | >85% | ~45% | 55–65% | ~20% |

### Pass@5 vs. Pass∧5 Gap

| Setting | Max Gap | Note |
|---------|---------|------|
| IncreQA | >38% | Optimistic vs. robust |
| AdaptQA | >36% | Higher instability |
| Overall Max | **~60%** | Extremely unreliable |

### Key Findings
- High Pass@5 indicates that agents are capable of solving tasks, while low Pass∧5 reveals that they cannot do so consistently.
- AdaptQA is more challenging than IncreQA, as it requires flexible strategy adjustment based on intermediate results.
- Value mismatch is the primary failure cause — agents cannot reliably map user terminology to database entries.
- The LLM-simulated user occasionally deviates from instructions; the validator effectively detects and reruns such cases (~10% of sessions).

## Highlights & Insights
- **Pass@5 vs. Pass∧5 Insight**: This dual-metric evaluation exposes a critical safety concern — in the EHR domain, "occasionally correct" is insufficient; "correct every time" is required. A gap of 60% implies that the same agent performs near-randomly on identical tasks across runs.
- **Importance of Value Mismatch**: Prior Text-to-SQL benchmarks largely overlook this issue, yet it is critical in EHR settings — failing to recognize that "Lopressor" and "metoprolol tartrate" refer to the same drug can result in entirely erroneous query results.
- **Interaction Flow Design**: IncreQA tests context retention, while AdaptQA tests flexible adaptation — together they align with distinct patterns of real-world clinical data access.

## Limitations & Future Work
- The LLM-simulated user introduces uncontrollable stochasticity.
- Only two EHR databases are used; more diverse data sources may present additional challenges.
- The tool suite is predefined; real-world deployments may require more flexible tool invocation.
- Evaluation is conducted offline; latency and interactive user experience in live deployment are not considered.

## Related Work & Insights
- **vs. EHRSQL**: EHRSQL addresses single-turn Text-to-SQL; EHR-ChatQA targets multi-turn interactive scenarios.
- **vs. Tau-Bench**: Tau-Bench evaluates general-purpose agent interaction; EHR-ChatQA focuses on domain-specific challenges unique to EHR.
- **vs. MedAgentBench**: MedAgentBench uses unambiguous task instructions and does not require interactive clarification.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First EHR QA benchmark integrating user interaction, tool use, and value mismatch resolution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation covers state-of-the-art models, five repeated trials, and diagnostic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — POMDP formalization and task design are clearly presented.
- **Value**: ⭐⭐⭐⭐⭐ — Carries direct implications for the safety and reliability of EHR agent deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)
- [\[ACL 2026\] Query Pipeline Optimization for Cancer Patient Question Answering Systems](../../ACL2026/medical_nlp/query_pipeline_optimization_for_cancer_patient_question_answering_systems.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](../../ACL2026/medical_nlp/beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)
- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](../../NeurIPS2025/medical_nlp/mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)
- [\[ICLR 2026\] CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)

</div>

<!-- RELATED:END -->
