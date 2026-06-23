---
title: >-
  [Paper Note] BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions
description: >-
  [ICLR 2026][LLM Evaluation][text-to-SQL] BIRD-INTERACT transforms single-turn text-to-SQL evaluation into a dynamic interactive environment featuring a user simulator, knowledge base management, and test case execution. It covers full CRUD operations and intentionally injects ambiguity to evaluate LLM interaction capabilities via two settings: c-Interact (pro
tags:
  - ICLR 2026
  - LLM Evaluation
  - text-to-SQL
  - benchmark
date: 2026-05-08
content_hash: 13bda8eb3f46fbd0
---
# BIRD-INTERACT: Re-imagining Text-to-SQL Evaluation via Lens of Dynamic Interactions

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=nHrYBGujps](https://openreview.net/forum?id=nHrYBGujps)  
**Paper**: [Project Page](https://bird-interact.github.io) (BIRD Team)  
**Code**: https://bird-interact.github.io (Available)  
**Area**: LLM Evaluation / Benchmark / Text-to-SQL  
**Keywords**: text-to-SQL, interactive evaluation, benchmark, user simulator, test-time scaling

## TL;DR
BIRD-INTERACT transforms single-turn text-to-SQL evaluation into a dynamic interactive environment featuring a user simulator, knowledge base management, and test case execution. It covers full CRUD operations and intentionally injects ambiguity to evaluate LLM interaction capabilities via two settings: c-Interact (protocol-guided) and a-Interact (autonomous agent). Even the strongest model, GPT-5, achieved only 8.67% (c) / 17.00% (a) on the 600-question full set, exposing a significant gap where current models can write SQL but struggle to clarify tasks through interaction.

## Background & Motivation

**Background**: LLMs have achieved impressive results on single-turn text-to-SQL benchmarks like Spider and BIRD, directly generating correct SQL from natural language requirements. This has led to numerous LLM-based Natural Language Interface to Database (NLIDB) methods, suggesting that the translation of natural language to SQL is nearing maturity.

**Limitations of Prior Work**: Real-world database interaction is rarely a single "perfect requirement to correct query" transition. It is a stateful, evolving iterative dialogue. User requirements are inherently ambiguous (e.g., "urgent processing"), the initial SQL might require debugging based on feedback, and users often ask follow-up questions dependent on previous results. Existing multi-turn benchmarks fail in two ways: first, they treat dialogue history as **static scripts**, where every LLM receives the same pre-recorded clean trajectory, failing to reward smart interaction strategies or penalize poor ones; second, they have a **narrow task scope**, focusing almost exclusively on read-only SELECT queries while ignoring common Data Management operations like INSERT, UPDATE, DELETE, and ALTER TABLE.

**Key Challenge**: To approach a "production-grade database assistant," evaluations must restore two elements missing from current benchmarks: **dynamic interaction** (clarifying ambiguity and error recovery) and a **full range of operations** (read and write). Scaling such evaluations without human intervention introduces the difficulty of using LLM user simulators without "cheating" (leaking ground-truth SQL) or "drifting" (deviating from the original task).

**Goal**: Build a benchmark that restores "realism" through: ① a high-fidelity, end-to-end automated interactive environment; ② two realistic evaluation settings; ③ a challenging task set covering full CRUD with executable test cases requiring dynamic interaction for resolution.

**Key Insight**: Leveraging LIVESQLBENCH (a single-turn benchmark supporting DML/DDL with sandboxed execution), the authors "dynamicize" it. The core insight is that interaction difficulty can be engineered and measured—ambiguities are manually injected with unique clarifications, and user simulators are constrained via "function-driven" mechanisms to remain flexible but fair.

**Core Idea**: Re-imagine text-to-SQL evaluation via a dynamic lens—coupling databases, hierarchical knowledge bases (HKB), metadata, and a function-driven user simulator into a sandbox. By intentionally creating ambiguity and follow-up inquiries, models are forced to learn how to "ask the right questions, use the right resources, and recover from errors" within a limited interaction budget.

## Method

### Overall Architecture

BIRD-INTERACT is a benchmark that automates dynamic interaction evaluation. The pipeline uses single-turn tasks from LIVESQLBENCH, which are then enhanced by 12 expert annotators who **inject ambiguity** and **append follow-up sub-tasks**. It utilizes a **two-stage function-driven user simulator** to act as a human who provides clarifications and feedback without leaking answers. Evaluation is conducted under **c-Interact** and **a-Interact** settings with limited budgets, where correctness is determined by executable test cases.

The final dataset contains 900 interactive tasks: BIRD-INTERACT-FULL (600 questions, up to 11,796 interactions) and BIRD-INTERACT-LITE (300 questions). Each task includes two sub-tasks ($n=2$): an ambiguous priority sub-task $q_1$ and a state-dependent follow-up sub-task $q_2$.

Formally, interaction is modeled as a collaboration between a text-to-SQL system $S_\theta$ and a user simulator $U_\gamma$ over an environment $E=\{D, M, K\}$ (database $D$, metadata $M$, external knowledge $K$). For sub-task $q_i$, the interaction at turn $t$ is:

$$u_i^t = U_\gamma(h_i^{t-1}, q_i, E), \quad s_i^t = S_\theta(h_i^{t-1}, u_i^t, E), \quad h_i^t = h_i^{t-1} \oplus \langle u_i^t, s_i^t \rangle$$

Where $h_i^t$ is the history and $\oplus$ denotes text concatenation. Subsequent sub-tasks are only released if the priority sub-task passes the test case.

### Key Designs

**1. Ambiguity Injection: Turning Solvable Tasks into Clarification-Required Tasks**

Three types of ambiguity are systematically injected: **Surface-level query ambiguity** (intent-level vagueness or implementation-level underspecification), **Knowledge ambiguity** (creating gaps in the HKB by removing entries or breaking multi-hop reasoning chains), and **Environmental ambiguity** (uncertainty involving NULL values or noise). Each ambiguity is paired with a unique clarification source.

**2. Follow-up Sub-tasks and State Dependency**

Follow-up sub-tasks are designed across 5 categories. Critically, these tasks involve **state dependency**, where $q_2$ may rely on the **database state modified** or **objects created** (e.g., tables/functions) during $q_1$. This shifts the evaluation from independent queries to a stateful, long-term problem-solving process.

**3. Two-Stage Function-Driven User Simulator**

To prevent leakage of ground-truth (GT) SQL and task drift, the simulator is split into two stages. **Stage 1** uses an LLM as a semantic parser to map system clarification requests to one of three actions: `AMB()` (pre-annotated ambiguity), `LOC()` (reasonable SQL-related clarifications located via AST-based retrieval), or `UNA()` (rejecting improper requests). **Stage 2** generates the final response based on the action and GT SQL. This ensures behavior is **predictable and controlled**.

**4. Two Evaluation Settings and Budget Constraints**

- **c-Interact (Protocol-Guided)**: Acts as a dialogue assistant. Sub-tasks follow a $q_1 \to q_2$ sequence. The model gets one debug opportunity per sub-task. Clarification rounds are limited by a budget $\tau_{\text{clar}} = m_{\text{amb}} + \lambda_{\text{pat}}$ ($m_{\text{amb}}$ is the required rounds, $\lambda_{\text{pat}}$ is user patience).
- **a-Interact (Autonomous Agent)**: Follows the ReAct paradigm. The model autonomously decides when to query the DB, check HKB, or ask the user using 9 discrete actions. Budget is calculated as $B = B_{\text{base}} + 2 m_{\text{amb}} + 2\lambda_{\text{pat}}$ ($B_{\text{base}}=6$).

Metrics: **Success Rate (SR)** for sub-tasks and **Normalized Reward**, which weights $q_1$ (70%) and $q_2$ (30%) within $[0, 1]$.

## Key Experimental Results

Evaluations were performed on PostgreSQL 14 using 7 frontier LLMs (2 open, 5 closed), with $\lambda_{\text{pat}}=3$.

### Main Results

Success Rates (SR) on BIRD-INTERACT-FULL (600 tasks), where BI = Business Intelligence and DM = Data Management:

| Setting | Model | Priority SR(%) | Follow-up SR(%) | Reward*(%) | Cost (USD) |
|------|------|------|------|------|------|
| c-Interact | GPT-5 | 14.50 | 8.67 | 12.58 | $0.08 |
| c-Interact | Claude-Sonnet-4 | 22.33 | 14.17 | 18.35 | $0.29 |
| c-Interact | Gemini-2.5-Pro | **25.00** | **16.33** | **20.92** | $0.04 |
| a-Interact | Qwen-3-Coder-480B | 13.33 | 4.17 | 10.58 | $0.07 |
| a-Interact | Claude-Sonnet-4 | 27.83 | 12.67 | 23.28 | $0.51 |
| a-Interact | GPT-5 | **29.17** | **17.00** | **25.52** | $0.24 |

The results show extreme difficulty, with the best models achieving only ~25% reward and end-to-end success rates under 17%.

### Ablation Study

| Analysis | Key Finding |
|------|---------|
| Mode Comparison | GPT-5 varied significantly: 14.50% (c-Interact) vs. 29.17% (a-Interact). |
| Memory Grafting | Transferring clarification history from other models to GPT-5 improved its score significantly. |
| ITS (Scaling) | Claude-3.7-Sonnet performance improved monotonically with more interaction rounds. |
| BI vs DM | BI queries are significantly harder than DM due to complex analytical reasoning. |

### Key Findings
- **GPT-5 Interaction Gap**: GPT-5 performed poorly in the protocol-guided c-Interact but excelled in a-Interact. Memory Grafting proved its bottleneck is communication/clarification strategy, not SQL generation.
- **Subsequent Task Difficulty**: Success rates drop for follow-up tasks due to increased context length and the requirement to reason over modified database states.
- **Interaction as a Test-time Resource**: The Interaction Test-time Scaling (ITS) law suggests that models capable of converting interaction into information gain can approach or exceed performance in idealized single-turn scenarios if given enough rounds.

## Highlights & Insights
- **Engineering Interaction Ability**: By systematic injection of ambiguities that require unique clarifications, the benchmark forces interaction rather than relying on chance.
- **Two-Stage Simulator**: The "Semantic Parsing → Symbolic Action → Controlled Generation" pipeline effectively prevents simulator drift and information leakage.
- **Memory Grafting**: This diagnostic method of "transplanting" dialogue history successfully decouples generation capability from communication capability.

## Limitations & Future Work
- **Scope**: Limited to relational databases (PostgreSQL); NoSQL and cross-database scenarios are not covered.
- **Depth**: Fixed at 2 sub-tasks ($n=2$); real DBA interactions involve longer sessions and evolving multi-objective goals.
- **Simulation**: While controlled, the LLM-based simulator may not fully capture the diversity of real human patience or styles.
- **Cost**: Ambiguity injection requires significant expert manual effort.

## Related Work & Insights
- **Comparison to Spider/BIRD**: BIRD-INTERACT explicitly evaluates the ability to resolve ambiguity and handle evolving requirements through multiple turns.
- **Comparison to SParC/CoSQL**: Unlike previous multi-turn datasets that use static trajectories, BIRD-INTERACT uses a dynamic simulator and introduces state dependency between tasks.
- **Base Framework**: Built upon LIVESQLBENCH, extending it from a single-turn environment to a dynamic interactive one.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCTS-SQL: Light-Weight LLMs can Master the Text-to-SQL through Monte Carlo Tree Search](../../AAAI2026/llm_evaluation/mcts-sql_light-weight_llms_can_master_the_text-to-sql_through_monte_carlo_tree_s.md)
- [\[ACL 2025\] Something's Fishy In The Data Lake: A Critical Re-evaluation of Table Union Search Benchmarks](../../ACL2025/llm_evaluation/somethings_fishy_in_the_data_lake_a_critical_re-evaluation_of_table_union_search.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](../../ACL2026/llm_evaluation/comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)

</div>

<!-- RELATED:END -->
