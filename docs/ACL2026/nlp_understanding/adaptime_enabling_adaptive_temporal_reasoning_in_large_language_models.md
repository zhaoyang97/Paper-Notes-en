---
title: >-
  [Paper Note] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models
description: >-
  [ACL 2026][NLP Understanding][Temporal reasoning] This paper proposes AdapTime, which abstracts "temporal reasoning" into three reusable atomic actions: reformulate, rewrite…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Temporal reasoning"
  - "Temporal QA"
  - "Adaptive planning"
  - "LLM Planner"
  - "No external tools"
date: 2026-05-08
content_hash: b0de0e60403a1557
---

# AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.24175](https://arxiv.org/abs/2604.24175)  
**Code**: https://github.com/Applied-Machine-Learning-Lab/ACL2026-AdapTime  
**Area**: NLP Understanding / LLM Reasoning  
**Keywords**: Temporal reasoning, Temporal QA, Adaptive planning, LLM Planner, No external tools

## TL;DR
This paper proposes AdapTime, which abstracts "temporal reasoning" into three reusable atomic actions: reformulate, rewrite, and review. An LLM Planner adaptively decides which steps to execute and in what order based on the question and context. Without any external tools, manual rules, or fine-tuning, it significantly improves LLM performance on temporal QA, pushing TimeQA-Easy to 85.4 EM using DeepSeek-V3.

## Background & Motivation

**Background**: Temporal Question Answering (Temporal QA) requires models to answer time-related questions from documents with temporal annotations, such as "What position did Terence Cooper hold between March 1966 and October 1969?". Existing solutions roughly fall into two categories—one relying on external tools (QAaP uses Python dictionaries for check/match, Event-AL uses Python solvers, Step-back uses retrievers), and the other relying on human intervention (Time-CoT requires manual timeline completion, TG-LLM involves manual correction of temporal graphs for hard cases, and TISER similarly depends on temporal graphs).

**Limitations of Prior Work**: (1) External tools and manual rules bind methods to specific data/scenarios, requiring rewrites when switching benchmarks. (2) Existing pipelines use a **fixed process**—running every question through the same "extract-reason-verify" sequence, ignoring differences in question difficulty. This leads to over-processing of simple questions (redundant reasoning, introducing errors) and failure on complex questions due to insufficient steps.

**Key Challenge**: The complexity variance of temporal questions is enormous—simple queries like "Who is the US President" and multi-hop reasoning like "What position did someone hold during a specific period" should essentially follow different reasoning paths. However, traditional fixed pipelines force all questions through the same process.

**Goal**: (1) Extract a set of atomic temporal actions that the **LLM can perform internally**, removing dependence on external tools or humans. (2) Allow the LLM to decide which actions to invoke and in what order based on question characteristics, achieving per-instance adaptive reasoning.

**Key Insight**: The authors found that while existing methods vary in form, they essentially perform one of three tasks: decomposing complex questions into sub-questions, rewriting implicit temporal expressions into structured forms, or fact-checking answers. If these three tasks can be completed via LLM prompting, external tools become redundant.

**Core Idea**: Define three actions (reformulate / rewrite / review) plus an LLM Planner. The Planner decides online whether to execute each action based on the question and intermediate states, upgrading the "fixed pipeline" to "adaptive planning."

## Method

### Overall Architecture
AdapTime consists of 4 modules: 3 atomic reasoning actions (Reformulate / Rewrite / Review) + 1 LLM Planner. Given a document $C$ and a temporal question $Q$, the Planner first decides whether to decompose $Q$ into a sub-question sequence $\{q_1,\ldots,q_n\}$. Next, it determines whether to rewrite $C$ into a structured representation anchored by time. Then, it generates answers $A_i$ for each sub-question based on the rewritten context and aggregates them. Finally, the Planner judges if the confidence in the answer is sufficient to decide whether to trigger a Review for fact-checking. The entire process calls no external APIs and relies on no manual rules.

### Key Designs

1.  **Three Atomic Temporal Reasoning Actions**:
    *   **Function**: Abstract the core operations of existing temporal reasoning methods into a minimal set of actions the LLM can perform autonomously, covering "question decomposition / context rewriting / answer verification."
    *   **Mechanism**: (a) **Reformulate**—prompt the LLM to decompose questions with complex temporal constraints or multi-hops into $Q=\{q_1,\ldots,q_n\}$, where each $q_i$ is answerable individually, and the final answer is aggregated. For example, "What position did Cooper hold during [period]" is split into "What positions did Cooper hold" + "Which position's duration covers the target period". (b) **Rewrite**—prompt the LLM to rewrite implicit temporal expressions like "during his presidency" or "after the war ended" into an explicit timeline anchored to calendar dates (in any form: code/timeline/temporal graph), allowing downstream modules to reason on time-anchored facts. (c) **Review**—prompt the LLM to retrieve original sentences supporting the current answer, correcting the answer if evidence is missing or conflicting.
    *   **Design Motivation**: These three actions cover the key operations of QAaP / Time-CoT / Event-AL / TG-LLM / TISER, but each is implemented purely via prompting, removing dependencies on Python interpreters, manual timelines, or hard-case annotations. The three actions are orthogonal and decoupled, allowing for any combination.

2.  **LLM Planner (Adaptive Planning)**:
    *   **Function**: Dynamically decide which actions to execute or skip based on the semantic structure of the question, the temporal complexity of the context, and the model's confidence in intermediate answers.
    *   **Mechanism**: The Planner is also an LLM prompt (using the same backbone as execution), invoked at each potential action point: determine if $Q$ is multi-hop $\rightarrow$ decide on Reformulate; check if temporal expressions in $C$ are ambiguous or scattered $\rightarrow$ decide on Rewrite; evaluate the credibility of the initial answer $\rightarrow$ decide on Review. Decisions are provided in natural language ("Yes/No + brief reason"), and the model is not forced to complete all steps. Algorithm 1 provides the complete pseudocode, where each step is gated by the Planner's binary decision $d_i$.
    *   **Design Motivation**: Fixed pipelines perform "over-reasoning" on simple questions, introducing noise, yet fall short on hard questions. Letting the Planner inspect the question before deciding the depth of reasoning allows the budget to be spent where it is truly needed. Figure 3 shows that Reformulate frequency is high on TimeQA (mostly decomposable multi-hops), while Rewrite + Review frequency is significantly higher on TempReason-L2/L3 (complex timelines, low initial confidence).

3.  **Zero-tool Reasoning Based on Internal Capabilities**:
    *   **Function**: All actions and decisions are completed using the LLM's internal capabilities without calling external tools, retrievers, or symbolic solvers.
    *   **Mechanism**: "Fact matching" previously handled by Python interpreters is replaced by prompting the LLM to retrieve supporting sentences; "long document compression" previously handled by retrievers is replaced by prompting the LLM to rewrite relevant paragraphs into timeline representations; "hard-case annotations" previously handled manually are replaced by prompting the LLM to re-decompose the problem.
    *   **Design Motivation**: Dependence on external tools is the root cause of poor generalization in current temporal QA methods—once the domain or data source changes, tools/rules must be rebuilt. By internalizing all capabilities within the LLM, the method can migrate zero-shot to new scenarios (evidenced by success on open-domain ArchivalQA).

### Loss & Training
Completely training-free, utilizing only prompt scheduling. Decoding uses top-k=10, temperature=0.7, batch_size=1, and max_new_tokens=512. The paper also attempted to distill a LLaMA-3-8B supervised Planner using 1000 high-quality plans generated by DeepSeek-V3, which performed worse than the prompt-based Planner (31.0 vs 41.5 EM on TimeQA-Easy), suggesting that in-context planning is more robust and less prone to overfitting than fine-tuned planners.

## Key Experimental Results

### Main Results
Two benchmarks: TimeQA (easy/hard, explicit vs. implicit time) + TempReason (L2 time-event alignment / L3 inter-event temporal relations), with 1000 samples each. Three backbones: LLaMA-3-8B, Qwen-3-8B, DeepSeek-V3. Metrics: EM + F1.

| Model | Method | TimeQA-Easy EM | TimeQA-Hard EM | TempReason-L2 EM | TempReason-L3 EM | Avg EM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4 (Closed-source Ref) | – | 71.6 | 54.6 | 45.4 | 43.1 | 54.3 |
| TG-LLM (SOTA Ref) | – | 66.4 | 63.1 | 42.4 | 35.6 | 51.9 |
| LLaMA-3-8B | ICL | 1.1 | 1.7 | 3.8 | 1.8 | 2.1 |
| LLaMA-3-8B | CoT | 29.7 | 31.6 | 18.5 | 16.5 | 24.1 |
| LLaMA-3-8B | **Ours** | **41.5** | **33.3** | **18.7** | 14.5 | **27.0** (+24.9) |
| Qwen-3-8B | CoT | 69.4 | 62.9 | 22.6 | 28.8 | 45.9 |
| Qwen-3-8B | **Ours** | **72.7** | **66.5** | **29.1** | 28.8 | **49.3** (+6.4) |
| DeepSeek-V3 | CoT | 85.3 | 75.6 | 44.8 | 47.0 | 63.2 |
| DeepSeek-V3 | Step-back | 84.4 | 76.4 | 45.8 | 48.8 | 63.9 |
| DeepSeek-V3 | Self-refinement | 77.6 | 76.4 | 44.3 | 41.1 | 60.1 |
| DeepSeek-V3 | **Ours** | **85.4** | **77.7** | **48.0** | **49.8** | **65.1** (+5.5) |

### Ablation Study (DeepSeek-V3)

| Configuration | TimeQA-Easy EM | TempReason-L3 EM |
| :--- | :--- | :--- |
| Full AdapTime | 85.4 | 49.8 |
| w/o Reformulate | 85.0 | 48.9 |
| w/o Rewrite | 84.8 | 47.6 |
| w/o Review | 84.8 | 49.0 |
| w/o LLM Planner (Fixed 3-step) | 85.3 | 48.9 |

### Key Findings
- **Rewrite contributes most**: Adding Rewrite alone boosts the ICL baseline from 80.8 to 86.4 EM. Removing Rewrite also causes the largest drop, suggesting "explicitly time-anchored context representation" is critical for LLM temporal reasoning.
- **Planner value scales with data difficulty**: On TimeQA-Easy, removing the Planner causes almost no drop (forced 3-step works), but on complex tasks like TempReason-L2/L3 where flexible step selection is needed, the Planner provides a stable 0.5–1 EM gain.
- **Minimal token budget increase**: AdapTime averages 4873 tokens/instance, only 12% more than ICL (4345), and far lower than self-refinement (>10000 tokens). This confirms that "triggered-on-demand" planning saves overhead.
- **Consistent gains across scales**: AdapTime significantly outperforms corresponding baselines across 1B, 8B, and V3 scales. Notably, 8B+AdapTime even exceeds closed-source GPT-4 on TimeQA, proving the method's model-agnostic nature.
- **Differentiated step distribution** (Figure 3): TimeQA leans towards Reformulate (multi-hop decomposition), while TempReason-L2/L3 lean towards Rewrite+Review (implicit time + answer uncertainty), validating that the Planner makes task-aware differentiated decisions.

## Highlights & Insights
- **Transforming "Tool Invocation" into "Prompt Scheduling"**: Previous temporal QA outsourced reasoning to external Python/retrievers/solvers. AdapTime proves pure LLMs can handle these sub-tasks and uses an explicit Planner instead of implicit "Chains of Thought" to achieve controllable adaptation. This "internalization" concept can be directly transferred to other domains (math, code, agentic search).
- **Prompt-based Planner > Fine-tuned Planner**: Experimental comparison shows that distillation fine-tuning actually loses generalization ability, suggesting that for planning tasks with scarce data, in-context learning is more robust than SFT—a counter-intuitive finding inspiring for agentic LLM design.
- **Abstraction power of "3 Orthogonal Actions"**: The authors compress core operations from 5 prior works into 3 actions (Table 1), covering the performance upper bounds of existing work. This research paradigm of "ontology abstraction before engineering" is highly valuable.
- **Compatibility with Retrievers**: Section 4.7 shows that BM25+AdapTime still improves by 2 EM on open-domain ArchivalQA, proving AdapTime is an orthogonal enhancement that can be stacked onto RAG pipelines.

## Limitations & Future Work
- The Planner's decision stability depends on the underlying model—the same question may yield different execution paths across runs. The authors admit Planner reliability decreases in smaller models.
- The action set is limited to 3; more complex temporal reasoning (e.g., timezone conversion, relative time expression calculation, cross-lingual time) may require expansions like "Calculate" or "Convert".
- Token overhead for long documents still primarily comes from the original context; performance on ultra-long inputs (>32k) has not been fully verified.
- No interpretability analysis or calibration of Planner decisions was provided—when the Planner makes a mistake and what its error patterns are remains unexplored.

## Related Work & Insights
- **vs QAaP** (Zhu et al. 2023): QAaP converts questions to Python dicts and uses code for matching. AdapTime requires no Python interpreter as all logic is prompt-driven, offering stronger generalization.
- **vs TG-LLM / TISER** (Xiong et al. 2024 / Bazaga et al. 2025): They construct temporal graphs and manually correct hard cases. AdapTime's Rewrite action achieves equivalent capabilities fully automatically.
- **vs Step-back / Self-refine**: Step-back abstracts a step before answering, and Self-refine iterates on reflection, but both are fixed processes. AdapTime introduces per-instance routing via a Planner, which is more granular with only a slight increase in token budget.

## Rating
- Novelty: ⭐⭐⭐⭐ Abstracting core temporal methods into 3 atomic actions + Planner scheduling is a clean redesign.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 benchmarks × 3 backbones + thorough ablations + open-domain extension + case comparisons provide broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Table 1 comparison is clear, Algorithm 1 pseudocode is intuitive, and the appendix includes complete case study examples.
- Value: ⭐⭐⭐⭐ Training-free, pure prompt, directly applicable to any LLM with low reproduction barriers; provides methodological insights for agentic reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)

</div>

<!-- RELATED:END -->
