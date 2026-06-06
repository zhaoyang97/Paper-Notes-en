---
title: >-
  [Paper Note] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification
description: >-
  [ACL 2026][Multi-Agent][Tabular QA] This paper proposes the ODUTQA-MDC task and benchmark, which systematically investigates the detection and multi-turn dialogue clarification of user query ambiguity in open-domain scen…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Tabular QA"
  - "Underspecified Query Clarification"
  - "Multi-turn Dialogue"
  - "Multi-agent Framework"
  - "Text-to-SQL"
date: 2026-05-08
content_hash: 9a22cc13d849d3f9
---

# ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification

**Conference**: ACL 2026  
**arXiv**: [2604.10159](https://arxiv.org/abs/2604.10159)  
**Code**: [GitHub](https://github.com/jensenw1/ODUTQA-MDC)  
**Area**: LLM Evaluation  
**Keywords**: Tabular QA, Underspecified Query Clarification, Multi-turn Dialogue, Multi-agent Framework, Text-to-SQL

## TL;DR
This paper proposes the ODUTQA-MDC task and benchmark, which systematically investigates the detection and multi-turn dialogue clarification of user query ambiguity in open-domain scenarios for the first time. It constructs a large-scale dataset containing 25,105 QA pairs and designs the MAIC-TQA multi-agent framework to achieve end-to-end "detection-clarification-reasoning" for tabular QA.

## Background & Motivation

**Background**: Large language models have advanced the development of Tabular QA, and current Text-to-SQL methods excel on standard datasets such as Spider. Open-domain tabular QA further increases difficulty by requiring autonomous retrieval of relevant tables from large-scale databases.

**Limitations of Prior Work**: In real-world scenarios, user queries are often underspecified—containing spelling errors, vague phrasing, or incomplete information. For example, a user might omit a city name (missing FROM clause), use vague expressions instead of precise column names (unclear SELECT intent), or use abbreviations instead of full names (mismatched WHERE conditions). These ambiguities fundamentally hinder the generation of correct SQL.

**Key Challenge**: Existing research either only detects ambiguity in closed domains (without resolving it) or uses static preset dialogues (PRACTIQ), failing to capture the dynamic and unpredictable nature of real human interaction. There is a lack of appropriate datasets and evaluation frameworks to systematically study the complete "detection-clarification-QA" process.

**Goal**: Define the ODUTQA-MDC task and construct the first comprehensive benchmark, including a large-scale dataset, a fine-grained annotation scheme, and a dynamic clarification interface, alongside a baseline system.

**Key Insight**: Categorize ambiguity according to the SQL structure: table scope ambiguity (FROM), query intent ambiguity (SELECT), query condition ambiguity (WHERE), and mixed types. This classification naturally corresponds to different stages of the Text-to-SQL pipeline.

**Core Idea**: Construct a "detection-clarification-redetection" closed-loop evaluation process, achieving scalable multi-turn interaction evaluation via a dynamic user simulator, while proposing the MAIC-TQA multi-agent framework as a baseline.

## Method

### Overall Architecture
MAIC-TQA adopts a modular multi-agent architecture. The workflow is: SLU module extracts user intent and slot information $\rightarrow$ Scope Verification (SV) Agent validates and clarifies table scope information $\rightarrow$ Table Retrieval (TR) Agent integrates original queries and clarified information to determine the target table $\rightarrow$ SQL Generation and Verification (SGV) Agent generates, executes, and validates SQL queries. Each agent can dynamically trigger clarification dialogues with the user simulator within the process.

### Key Designs

1. **Fine-grained Ambiguity Classification and Annotation System**:
    - **Function**: Supports precise detection and classification of different types of ambiguity in user queries.
    - **Mechanism**: Defines three ambiguity labels: Intent ambiguity (binary classification), Scope ambiguity (triplet annotation `[slot_content, slot_type, error_type]`, where `error_type` includes Missing/Error/Unmatch), and Condition ambiguity (triplet annotation `[slot_content, slot_type, "not exist"]`). Labels correspond one-to-one with SQL clauses.
    - **Design Motivation**: Existing datasets focus only on a single type of ambiguity and do not support mixed types. Fine-grained annotation can precisely locate the source of ambiguity to guide the system in generating targeted clarification questions.

2. **Dynamic Clarification User Simulator**:
    - **Function**: Simulates the process of a real user providing clarification information during multi-turn dialogues.
    - **Mechanism**: Implemented as a callable Python interface, strictly gated by detection accuracy—providing corresponding clarification information only when the system correctly detects the type of ambiguity. Use LLMs to rewrite standard response templates into natural spoken expressions and verify that key information is not lost during rewriting. Provides a dynamic mode (diversified responses) and a fixed mode (standardized responses for replication).
    - **Design Motivation**: Human interaction is costly and lacks consistency and reproducibility. An automated simulator achieves scalable evaluation while maintaining linguistic authenticity. The gating mechanism ensures that the evaluation reflects the system's actual detection capability.

3. **Multi-agent Collaborative Framework (MAIC-TQA)**:
    - **Function**: Completes end-to-end detection, clarification, and answering for underspecified queries.
    - **Mechanism**: Four agents collaborate: the SLU module uses a BERT classifier for intent detection and slot filling; the SV Agent checks whether required slots are missing or invalid and calls database validation functions; the TR Agent integrates dialogue history to generate table summaries and retrieves target tables through exact matching or BM25; the SGV Agent uses 5-shot ICL to generate SQL, checks result validity after execution, and triggers condition clarification when necessary.
    - **Design Motivation**: Decomposing a complex end-to-end task into multiple focused sub-modules, where each module handles specific types of ambiguity, reduces the burden on individual models.

### Loss & Training
The SLU module uses BERT for joint training of intent classification and slot filling. Other agents use in-context learning with LLMs and do not require additional training. Multiple LLM backends (Qwen3 32B/30B, Kimi K2, GLM 4, etc.) are supported.

## Key Experimental Results

### Main Results (Ambiguity Detection)

| Model | FROM Acc. | FROM F1 | WHERE Acc. | WHERE F1 | Mixed Acc. |
|------|-----------|---------|------------|----------|------------|
| Qwen3 32B | 77.66 | 82.82 | 69.59 | 66.02 | 54.96 |
| Qwen3 30B | 75.17 | 85.10 | 75.67 | 78.99 | 58.55 |
| Kimi K2 | 82.60 | 87.95 | 69.02 | 65.54 | 55.51 |
| SELECT (BERT) | 99.78 Acc. | 99.22 F1 | - | - | - |

### Ablation Study (MAIC-TQA vs. SLUTQA Baseline)

| Configuration | Description |
|------|------|
| SLUTQA (No Clarification) | Answers directly from underspecified queries; serves as the baseline without clarification. |
| MAIC-TQA Fixed | Uses standardized clarification responses. |
| MAIC-TQA Dynamic | Uses diversified clarification responses rewritten by LLMs. |

### Key Findings
- SELECT ambiguity is the easiest to detect (BERT reaches 99%+ F1), while FROM and WHERE are more difficult, and Mixed types are the hardest (~55% accuracy).
- Multi-turn dialogue clarification significantly improves QA accuracy, validating the value of the dynamic clarification mechanism.
- Performance in Dynamic mode is slightly lower than in Fixed mode, reflecting the challenges brought by natural language variation.
- All models perform poorly on Mixed types, indicating that simultaneous processing of multiple ambiguities remains an open problem.

## Highlights & Insights
- The task definition is highly complete: from dataset construction and annotation schemes to the evaluation framework (including a dynamic user simulator), forming a reproducible closed-loop research paradigm.
- The design of categorizing ambiguity by SQL clauses is intuitive and practical, allowing detection results to directly guide subsequent SQL generation.
- The gating mechanism of the dynamic clarification simulator is cleverly designed—the system only receives clarification information if it correctly detects the ambiguity, effectively avoiding "leakage" issues.

## Limitations & Future Work
- The dataset covers limited domains (real estate, land auctions, finance), and generalization to other domains needs verification.
- Templated data generation may lead to differences between the query distribution and real user queries.
- Clarification turns are limited to a single round, which may be insufficient for complex ambiguities.
- Performance on Mixed types is low, requiring better methods for joint handling of multiple ambiguities.
- Future directions: Expanding to more domains and languages, allowing multi-turn iterative clarification, and introducing user satisfaction evaluations.

## Related Work & Insights
- **vs. PRACTIQ**: PRACTIQ uses static preset dialogues and does not support dynamic interaction evaluation. The dynamic simulator in this paper is closer to real-world scenarios.
- **vs. AmbiQT/Ambrosia**: These works introduce ambiguity but lack a systematic clarification mechanism and QA evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically define the complete task of open-domain underspecified tabular QA with multi-turn clarification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large dataset scale, fine-grained annotation, and comparison across multiple models.
- Writing Quality: ⭐⭐⭐⭐ Task definition is clear, and method description is detailed.
- Value: ⭐⭐⭐⭐ Fills the gap in datasets and evaluation frameworks in this direction, providing infrastructure value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](../../ICML2026/multi_agent/engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Collaboration](towards_robust_real-world_spreadsheet_understanding_with_multi-agent_multi-forma.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)

</div>

<!-- RELATED:END -->
