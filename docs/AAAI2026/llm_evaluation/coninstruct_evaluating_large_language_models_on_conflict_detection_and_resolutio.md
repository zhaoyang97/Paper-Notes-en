---
title: >-
  [Paper Note] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions
description: >-
  [AAAI 2026][LLM Evaluation][instruction conflict detection] This paper proposes ConInstruct, a benchmark for evaluating LLMs' ability to detect and resolve conflicting constraints in instructions. Results show that most…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "instruction conflict detection"
  - "conflict resolution"
  - "constraint satisfaction"
  - "instruction following"
  - "evaluation benchmark"
date: 2026-05-08
content_hash: 7ba39057228ee48b
---

# ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions

**Conference**: AAAI 2026
**arXiv**: [2511.14342](https://arxiv.org/abs/2511.14342)  
**Code**: [GitHub](https://github.com/NLPCode/ConInstruct)  
**Area**: LLM Evaluation
**Keywords**: instruction conflict detection, conflict resolution, constraint satisfaction, instruction following, evaluation benchmark

## TL;DR

This paper proposes ConInstruct, a benchmark for evaluating LLMs' ability to detect and resolve conflicting constraints in instructions. Results show that most proprietary models can detect conflicts reasonably well but rarely notify users explicitly, with DeepSeek-R1 and Claude-4.5-Sonnet achieving the best conflict detection performance (F1 of 91.5% and 87.3%, respectively).

## Background & Motivation

**Background**: Instruction following is a core capability of LLMs. Existing research primarily evaluates model compliance with consistent instructions (e.g., verifiable instructions in IFEval, model-based evaluation in InfoBench) or studies instruction hierarchy adherence (system instructions taking priority over user instructions).

**Limitations of Prior Work**: Prior work assumes all constraints within an instruction are consistent and non-conflicting. In practice, however, users frequently introduce conflicting constraints when writing complex, lengthy instructions—for example, simultaneously requiring "include a specific phrase" and "limit output to 50 words" when the phrase itself already exceeds that limit. LLM behavior under such scenarios has not been systematically studied.

**Key Challenge**: When an instruction contains mutually unsatisfiable constraints, the ideal behavior is to proactively notify the user of the conflict and request clarification. Whether models actually do so—and whether conflict awareness is reflected in their responses—remains an open question.

**Goal**: To systematically evaluate LLMs' conflict detection capability and conflict resolution behavior, filling the gap in instruction-following research under conflicting constraint scenarios.

**Key Insight**: Constructing a benchmark dataset covering 6 constraint types and 9 conflict types, treating "whether a conflict can be detected" and "how the model responds upon detection" as two independent evaluation dimensions.

**Core Idea**: LLMs can generally detect conflicts in instructions but rarely notify users proactively—even the best-performing Claude-4.5-Sonnet does so explicitly in only 45% of cases.

## Method

### Overall Architecture

ConInstruct is constructed in three steps: (1) preparing 100 seed instructions spanning 6 tasks and 35 domains; (2) using GPT-4o to inject constraints of 6 types (content, keyword, phrase, length, format, style) into each instruction; and (3) generating 7–9 conflict pairs per expanded instruction, each consisting of an existing constraint and a newly constructed contradictory one. Two rounds of human quality control ensure that conflicts are unambiguous.

### Key Designs

1. **6 Constraint Types × 9 Conflict Types**: Constraint types include Content (CC), Keyword (KK), Phrase (PP), Length (LL), Format (FF), and Style (SS). Conflicts are categorized into 6 intra-type conflicts (CC, KK, PP, LL, FF, SS) and 3 inter-type conflicts (KP: keyword–phrase, PC: phrase–content, PS: phrase–style). This taxonomy enables fine-grained analysis of detection capability per conflict type.

2. **Conflict Detection Experimental Design**: The conflicting constraint is appended to the end of the expanded instruction, and conflict-containing instructions are mixed with conflict-free ones to form experimental subsets. Each subset contains 100 conflict-free instructions and a corresponding number of instructions with exactly one conflict, forming a binary classification task evaluated using F1. The number of conflicts (1–9) is also controlled to analyze detection behavior under multiple simultaneous conflicts.

3. **Conflict Resolution Behavior Analysis**: LLM responses to conflicting instructions are categorized into three types: (1) generating a response directly without mentioning the conflict; (2) requesting user clarification (RequestC); and (3) resolving the conflict autonomously before responding (ResolveC). The latter two constitute the ideal "explicit conflict acknowledgment" behavior. The distribution across these categories is analyzed for each model under varying numbers of conflicts.

4. **Quality Control**: Two annotators reviewed and revised GPT-4o-generated expanded instructions and conflict pairs to ensure constraint validity and conflict clarity. A third annotator performed a final review. All annotators were independent of the research team.

### Loss & Training

This is a purely evaluative work with no training involved. All models are evaluated directly in a zero-shot setting.

## Key Experimental Results

### Main Results

Single-conflict detection F1 (%):

| Model | CC | KK | PP | LL | FF | SS | KP | PC | PS | Avg. |
|------|------|------|------|------|------|------|------|------|------|------|
| GPT-4o | 91.9 | 91.3 | 88.7 | 88.1 | 79.8 | 89.8 | 75.1 | 83.7 | 76.1 | 84.9 |
| Claude-4.5-Sonnet | 88.5 | 88.5 | 88.5 | 86.0 | 86.5 | 88.5 | 88.5 | 86.8 | 83.6 | 87.3 |
| Claude-3.5-Sonnet | 95.7 | 93.1 | 93.1 | 90.5 | 90.5 | 93.1 | 60.3 | 89.8 | 73.6 | 86.6 |
| **DeepSeek-R1** | 93.1 | **94.1** | 93.6 | 93.1 | 88.1 | **94.1** | **93.1** | 89.1 | 85.3 | **91.5** |
| Llama-3.1-8B | 70.9 | 68.3 | 68.3 | 63.3 | 65.6 | 66.7 | 62.7 | 68.9 | 58.8 | 65.9 |

DeepSeek-R1 ranks first with an average F1 of 91.5% and is the only open-source model capable of matching proprietary model performance.

### Ablation Study

Conflict resolution behavior analysis (GPT-4o, by number of conflicts):

| # Conflicts | Direct Response (no acknowledgment) | Request Clarification | Resolve Autonomously |
|------|------|------|------|
| 1–2 | 97.5% | ~2% | <1% |
| 3–4 | ~90% | ~8% | ~2% |
| 7–9 | ~70% | ~25% | ~5% |

Claude-4.5-Sonnet exhibits the best conflict resolution behavior:

| # Conflicts | Direct Response | Request Clarification | Resolve Autonomously |
|------|------|------|------|
| 1–2 | ~55% | ~36% | ~9% |
| 7–9 | ~20% | ~65% | ~15% |

### Key Findings

- Intra-type conflicts are generally easier to detect than inter-type conflicts, with intra-type average F1 consistently higher across models.
- Format conflicts (FF) and phrase–style conflicts (PS) are the most difficult to detect.
- A higher number of conflicts paradoxically facilitates detection, as multiple conflicts provide stronger signals.
- Among open-source models, only DeepSeek-R1 reaches proprietary model performance; small models (Llama-1B/3B) perform poorly.
- Even the best-performing models tend to silently generate responses rather than notify users when facing a small number of conflicts.

## Highlights & Insights

- The work fills a critical gap in instruction-following evaluation by addressing conflicting constraint scenarios.
- The finding that models "detect but do not notify" is significant—it reveals that current LLM alignment has not adequately accounted for conflict scenarios.
- The 6×9 constraint–conflict type matrix enables fine-grained capability analysis.
- Claude's "request clarification" behavior (36%) substantially exceeds that of other models, reflecting superior user interaction design.

## Limitations & Future Work

- Conflicting constraints are always appended at the end of instructions; this fixed position may make detection artificially easier, and more naturalistic conflict embedding strategies remain to be explored.
- The 100 seed instructions constitute a relatively small corpus, potentially limiting the generalizability of conclusions.
- The quality of conflict resolution is not assessed—when models resolve conflicts autonomously, the appropriateness of their solutions is not evaluated.
- Only 6 constraint types are covered; real-world instructions exhibit more diverse conflict forms (e.g., semantic and logical conflicts).
- Methods for improving conflict acknowledgment behavior through training are not explored.

## Related Work & Insights

- **vs. IFEval**: IFEval evaluates instruction-following compliance under conflict-free instructions, while ConInstruct focuses on behavior under conflicting constraints; the two are complementary.
- **vs. FollowBench**: FollowBench evaluates following capability using multi-level constraints but assumes non-conflicting constraints; ConInstruct relaxes this assumption.
- **vs. Instruction Hierarchy Research**: Instruction hierarchy studies address priority conflicts between system and user instructions, whereas ConInstruct focuses on constraint conflicts within a single instruction—a distinct evaluation dimension.

## Rating

- Novelty: ⭐⭐⭐⭐ Conflict detection and resolution in instructions constitutes a genuinely novel evaluation dimension with a well-defined problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers a wide range of proprietary and open-source models with fine-grained conflict type analysis, though evaluation of conflict resolution quality is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured, with intuitive figures and a compelling problem motivation.
- Value: ⭐⭐⭐⭐ Identifies an important blind spot in current LLM instruction following, with implications for model safety and user experience improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](../../ICLR2026/llm_evaluation/aside_architectural_separation_of_instructions_and_data_in_language_models.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](../../ACL2026/llm_evaluation/novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](../../ACL2026/llm_evaluation/engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[AAAI 2026\] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models](nestr_a_neuro-symbolic_abductive_framework_for_temporal_reasoning_in_large_langu.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](../../ACL2026/llm_evaluation/roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)

</div>

<!-- RELATED:END -->
