---
title: >-
  [Paper Note] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification
description: >-
  [ACL 2026][LLM Evaluation][Tabular Question Answering] This paper introduces the ODUTQA-MDC task and benchmark, the first systematic study of underspecified query detection and multi-turn dialogue-based clarification in…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Tabular Question Answering"
  - "Ambiguous Query Clarification"
  - "Multi-turn Dialogue"
  - "Multi-agent Framework"
  - "Text-to-SQL"
date: 2026-05-08
content_hash: d0b3882512027a38
---

# ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification

**Conference**: ACL 2026
**arXiv**: [2604.10159](https://arxiv.org/abs/2604.10159)
**Code**: [GitHub](https://github.com/jensenw1/ODUTQA-MDC)
**Area**: LLM Evaluation
**Keywords**: Tabular Question Answering, Ambiguous Query Clarification, Multi-turn Dialogue, Multi-agent Framework, Text-to-SQL

## TL;DR
This paper introduces the ODUTQA-MDC task and benchmark, the first systematic study of underspecified query detection and multi-turn dialogue-based clarification in open-domain tabular QA. The authors construct a large-scale dataset of 25,105 QA pairs and propose the MAIC-TQA multi-agent framework to perform end-to-end "detect–clarify–reason" tabular question answering.

## Background & Motivation

**Background**: Large language models have driven advances in Tabular QA, with existing Text-to-SQL methods achieving strong performance on standard benchmarks such as Spider. Open-domain tabular QA further increases difficulty by requiring autonomous retrieval of relevant tables from large-scale databases.

**Limitations of Prior Work**: In real-world scenarios, user queries are frequently underspecified—containing spelling errors, vague expressions, or incomplete information. For example, a user may omit a city name (missing FROM clause), replace a precise column name with a vague description (ambiguous SELECT intent), or use abbreviations instead of full names (mismatched WHERE conditions). Such underspecification fundamentally impedes correct SQL generation.

**Key Challenge**: Existing studies either detect underspecification only in closed-domain settings without resolving it, or rely on statically pre-scripted dialogues (e.g., PRACTIQ), failing to capture the dynamic and unpredictable nature of real user interactions. There is a lack of appropriate datasets and evaluation frameworks for systematically studying the complete "detect–clarify–answer" pipeline.

**Goal**: Define the ODUTQA-MDC task, construct the first comprehensive benchmark—including a large-scale dataset, fine-grained annotation scheme, and dynamic clarification interface—and propose a baseline system.

**Key Insight**: Underspecification is categorized according to SQL clause structure: table-scope ambiguity (FROM), query-intent ambiguity (SELECT), query-condition ambiguity (WHERE), and mixed types. This taxonomy naturally corresponds to different stages of the Text-to-SQL pipeline.

**Core Idea**: Construct a closed-loop evaluation pipeline of "detect–clarify–re-detect," enabling scalable multi-turn interaction evaluation through a dynamic user simulator, alongside the proposed MAIC-TQA multi-agent framework as a baseline.

## Method

### Overall Architecture
MAIC-TQA adopts a modular multi-agent architecture with the following pipeline: an SLU module extracts user intent and slot information → a Scope Validation (SV) Agent verifies and clarifies table-scope information → a Table Retrieval (TR) Agent integrates the original query and clarification information to identify the target table → a SQL Generation and Verification (SGV) Agent generates, executes, and validates SQL queries. Each agent can dynamically trigger clarification dialogues with the user simulator during the pipeline.

### Key Designs

1. **Fine-grained Underspecification Taxonomy and Annotation Scheme**:

    - Function: Supports precise detection and classification of different types of underspecification in user queries.
    - Mechanism: Three underspecification labels are defined: intent ambiguity (binary classification), scope ambiguity (triplet annotation [slot_content, slot_type, error_type], where error_type ∈ {Missing, Error, Unmatch}), and condition ambiguity (triplet annotation [slot_content, slot_type, "not exist"]). Labels correspond one-to-one with SQL clauses.
    - Design Motivation: Existing datasets address only a single type of underspecification and do not support mixed ambiguity. Fine-grained annotation precisely localizes the source of ambiguity, guiding the system to generate targeted clarification questions.

2. **Dynamic Clarification User Simulator**:

    - Function: Simulates the process by which real users provide clarification information across multiple dialogue turns.
    - Mechanism: Implemented as a callable Python interface with strict gating on detection accuracy—clarification information is provided only when the system correctly identifies the type of underspecification. An LLM is used to paraphrase standard response templates into natural, colloquial expressions, with verification that key information is preserved. Both a dynamic mode (diversified responses) and a fixed mode (standardized responses for reproducibility) are provided.
    - Design Motivation: Human interaction is costly and lacks consistency and reproducibility. The automated simulator enables scalable evaluation while maintaining linguistic authenticity. The gating mechanism ensures that evaluation reflects the system's true detection capability.

3. **Multi-agent Collaborative Framework (MAIC-TQA)**:

    - Function: Performs end-to-end detection, clarification, and answering of underspecified queries.
    - Mechanism: Four agents collaborate with distinct roles: the SLU module uses a BERT classifier for intent detection and slot filling; the SV Agent checks whether required slots are missing or invalid and invokes database validation functions; the TR Agent integrates dialogue history to generate table summaries and retrieves target tables via exact matching or BM25; the SGV Agent generates SQL using 5-shot ICL, executes the query, checks result validity, and triggers condition clarification when necessary.
    - Design Motivation: Decomposing the complex end-to-end task into multiple focused sub-modules—each handling a specific type of underspecification—reduces the burden on any individual model.

### Loss & Training
The SLU module employs joint training of BERT for intent classification and slot filling. Other agents use in-context learning with LLMs and require no additional training. Multiple LLM backends are supported (Qwen3 32B/30B, Kimi K2, GLM 4, etc.).

## Key Experimental Results

### Main Results (Underspecification Detection)

| Model | FROM Acc. | FROM F1 | WHERE Acc. | WHERE F1 | Mixed Acc. |
|-------|-----------|---------|------------|----------|------------|
| Qwen3 32B | 77.66 | 82.82 | 69.59 | 66.02 | 54.96 |
| Qwen3 30B | 75.17 | 85.10 | 75.67 | 78.99 | 58.55 |
| Kimi K2 | 82.60 | 87.95 | 69.02 | 65.54 | 55.51 |
| SELECT (BERT) | 99.78 Acc. | 99.22 F1 | - | - | - |

### Ablation Study (MAIC-TQA vs. SLUTQA Baseline)

| Configuration | Description |
|---------------|-------------|
| SLUTQA (no clarification) | Answers directly from underspecified queries; serves as the no-clarification baseline |
| MAIC-TQA Fixed | Uses standardized clarification responses |
| MAIC-TQA Dynamic | Uses LLM-paraphrased diversified clarification responses |

### Key Findings
- SELECT ambiguity is the easiest to detect (BERT achieves 99%+ F1), while FROM and WHERE are more challenging; Mixed types are the hardest (~55% accuracy).
- Multi-turn dialogue clarification substantially improves QA accuracy, validating the value of the dynamic clarification mechanism.
- Performance under dynamic mode is slightly lower than under fixed mode, reflecting the challenges introduced by natural language variation.
- All models perform poorly on Mixed types, indicating that jointly handling multiple types of underspecification remains an open problem.

## Highlights & Insights
- The task definition is comprehensive: from dataset construction and annotation scheme to evaluation framework (including a dynamic user simulator), forming a reproducible closed-loop research paradigm.
- Categorizing underspecification by SQL clause is both intuitive and practical, allowing detection results to directly guide subsequent SQL generation.
- The gating mechanism of the dynamic clarification simulator is elegantly designed—the system receives clarification information only upon correctly detecting the underspecification, avoiding information leakage.

## Limitations & Future Work
- Dataset coverage is limited to specific domains (real estate, land auction, finance); generalization to other domains requires further validation.
- Template-based data generation may introduce a distributional gap between generated queries and real user queries.
- Clarification is limited to a single round, which may be insufficient for complex underspecification.
- Performance on Mixed types remains low, necessitating better methods for jointly handling multiple types of underspecification.
- Future directions include expanding to more domains and languages, enabling multi-round iterative clarification, and incorporating user satisfaction evaluation.

## Related Work & Insights
- **vs. PRACTIQ**: PRACTIQ relies on statically pre-scripted dialogues and does not support dynamic interaction evaluation. The dynamic simulator proposed in this paper more closely approximates real-world scenarios.
- **vs. AmbiQT/Ambrosia**: These works introduce underspecification but lack systematic clarification mechanisms and QA evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic definition of the complete task of open-domain underspecified tabular QA with multi-turn clarification.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale dataset, fine-grained annotations, and multi-model comparisons.
- Writing Quality: ⭐⭐⭐⭐ Task definition is clear and method descriptions are thorough.
- Value: ⭐⭐⭐⭐ Fills a gap in datasets and evaluation frameworks for this research direction, providing infrastructural value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[NeurIPS 2025\] CodeAssistBench (CAB): Dataset & Benchmarking for Multi-turn Chat-Based Code Assistance](../../NeurIPS2025/llm_evaluation/codeassistbench_cab_dataset_benchmarking_for_multi-turn_chat-based_code_assistan.md)
- [\[ACL 2026\] Abstain-R1: Calibrated Abstention and Post-Refusal Clarification via Verifiable RL](abstain-r1_calibrated_abstention_and_post-refusal_clarification_via_verifiable_r.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/llm_evaluation/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)

</div>

<!-- RELATED:END -->
