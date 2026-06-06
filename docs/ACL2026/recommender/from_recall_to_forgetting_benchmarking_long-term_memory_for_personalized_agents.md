---
title: >-
  [Paper Note] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents
description: >-
  [ACL 2026][Recommender Systems][Long-term memory benchmark] Ours proposes the Memora benchmark and the FAMA metric, extending long-term memory evaluation from shallow factual retrieval to memory consolidation and mutatio…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Long-term memory benchmark"
  - "memory consolidation"
  - "memory mutation"
  - "forgetting-aware evaluation"
  - "personalized agent"
  - "Memora"
  - "FAMA"
date: 2026-05-08
content_hash: b6709d637039ab33
---

# From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents

**Conference**: ACL 2026  
**arXiv**: [2604.20006](https://arxiv.org/abs/2604.20006)  
**Code**: [GitHub](https://github.com/geniesinc/Memora)  
**Area**: Long-Term Memory / Personalized Agents  
**Keywords**: Long-term memory benchmark, memory consolidation, memory mutation, forgetting-aware evaluation, personalized agent, Memora, FAMA

## TL;DR

Ours proposes the Memora benchmark and the FAMA metric, extending long-term memory evaluation from shallow factual retrieval to memory consolidation and mutation handling across weeks to months, revealing systemic failures of existing LLMs and memory agents in handling frequent knowledge updates.

## Background & Motivation

**Background**: Personalized agents need to maintain persistent memory across sessions during long-term interactions. Existing benchmarks (LoCoMo, LongMemEval, PersonaMem, etc.) primarily operationalize long-term memory as "cross-session factual retrieval," where 94% of LoCoMo questions and 85% of LongMemEval questions require referencing at most 2 historical sessions.

**Limitations of Prior Work**: Existing benchmarks suffer from two core defects: (1) Minimal demand for memory consolidation—referencing only approximately 1 historical session on average; (2) Memory mutation is rarely tested—LongMemEval contains at most 2 updates, and PersonaMem at most 3. Evaluation metrics only reward memory inclusion and do not penalize the erroneous use of obsolete information.

**Key Challenge**: Human cognition not only recalls but also accumulates experience, reconciles changes, and maintains a coherent world model. However, current evaluation frameworks essentially only test "the ability to retrieve an isolated piece of information from history," failing to expose the true capabilities of models in memory evolution and conflict resolution.

**Goal**: To build an evaluation system that far exceeds existing benchmarks in memory consolidation (up to 309 cross-session references) and memory mutation (up to 94 updates/deletions), and to propose an evaluation metric capable of penalizing the "use of obsolete memory."

**Key Insight**: Categorize memory tasks into three dimensions—Remembering, Reasoning, and Recommending—covering three time spans: weekly, monthly, and quarterly.

**Core Idea**: Long-term memory is not just "remembering" but "knowing when to forget"; the FAMA metric exposes fatal flaws in existing systems by explicitly penalizing reliance on obsolete memory.

## Method

**Overall Architecture**: A simulation-driven pipeline starting from persona seed data, simulating user interaction sequences (Week/Month/Quarter), converting them into multi-turn dialogues, and finally deriving evaluation tasks from memory trajectories.

**Key Designs**:

1. **Memory Evolution Simulation System**
    - **Function**: Generates long-term interaction data with realistic memory dynamics.
    - **Mechanism**: Construct 10 professional personas (engineers, researchers, etc.) and maintain three types of memory (preferences/activities/goals). The simulator updates the persistent memory state after each session, supporting preference drift, repeated activities, and incremental progress on long-term tasks, while including neutral sessions that do not modify memory. The complete memory trajectory is recorded as ground truth.
    - **Design Motivation**: In the Quarterly setting, the average memory consolidation reaches 28.4 (vs. ~1 in existing benchmarks) and mutations reach 14.8 (vs. ~0-2), ensuring that evaluation questions are based on clear memory state transitions.

2. **Multi-Agent Dialogue Generation and Quality Assurance**
    - **Function**: Converts simulated session specifications into high-quality multi-turn dialogues.
    - **Mechanism**: Uses a multi-agent prompting framework to generate dialogues, supporting both memory-neutral and memory-anchored dialogue turns. Quality is ensured through automated memory anchor checks via independent evaluation by 3 LLMs (accepted only if all pass) plus 5% manual verification.
    - **Design Motivation**: To prevent the introduction of untracked false memory details during generation and ensure strict alignment between dialogues and underlying memory trajectories.

3. **Forgetting-Aware Memory Accuracy (FAMA)**
    - **Function**: Unified evaluation of "remembering valid information" and "forgetting obsolete information."
    - **Mechanism**: $FAMA = \max(0, MPA - \lambda \cdot (1 - FAA))$, where $MPA$ is Memory Presence Accuracy and $FAA$ is Forgetting Absence Accuracy. $\lambda = N_{forget} / (N_{presence} + N_{forget})$ provides dynamic weighting by question. Three LLM judges (GPT-4.1, Claude Haiku 4.5, Gemini 2.5 Flash) use majority voting to judge each criterion.
    - **Design Motivation**: Traditional metrics only reward correct recall without penalizing the use of obsolete information. FAMA explicitly constrains forgetting through the $FAA$ term; $\lambda$ dynamic weighting ensures fair comparison across different questions.

## Key Experimental Results

**LLM Base Model Performance (FAMA, 0-100)**:

| Model | Remembering (W/M/Q) | Recommending (W/M/Q) | Reasoning (W/M/Q) |
|------|---------------------|----------------------|--------------------|
| GPT-5.2 | 25.3/19.9/23.4 | 54.8/51.1/53.4 | 4.7/0.0/1.0 |
| Claude Sonnet 4.5 | 27.5/19.4/21.3 | 43.6/39.0/44.0 | 6.7/3.0/5.5 |
| Gemini 3 Pro | 20.4/21.4/17.3 | 45.1/45.9/52.6 | 6.7/4.0/4.0 |

**Long-Term Memory Agent Performance (FAMA)**:

| Agent | Remembering (W/M/Q) | Recommending (W/M/Q) | Reasoning (W/M/Q) |
|-------|---------------------|----------------------|--------------------|
| A-Mem | 71.8/41.9/40.8 | 35.0/37.5/35.0 | 2.0/2.0/5.0 |
| LangMem | 71.2/42.0/39.1 | 48.9/44.1/33.9 | 30.0/14.0/11.0 |
| MemoBase | 43.6/20.1/15.2 | 68.9/58.5/45.6 | 18.0/7.0/1.0 |
| MemoryOS | 51.8/29.8/25.1 | 62.6/48.5/44.0 | 20.7/6.0/5.5 |
| Mem-0 | 40.4/21.1/19.9 | 52.6/36.2/38.5 | 16.0/0.0/2.0 |

**Key Findings**:
- **Reasoning almost completely fails**: Reasoning scores for all models and agents are mostly <10, with GPT-5.2 even scoring 0 on Monthly.
- **Memory agents significantly outperform LLMs in Remembering** (e.g., A-Mem 71.8 vs. GPT-5.2 25.3), but show limited advantage in Reasoning.
- **Performance consistently declines as time spans increase**: From Weekly to Quarterly, Remembering scores for all systems drop significantly.
- **Effect of reasoning tokens is unstable**: Helpful in some scenarios (e.g., Gemini Monthly Reasoning 4 → 10), but overall improvements are limited.
- **All systems frequently reuse invalidated memories**, exposing fundamental difficulties in maintaining a consistent belief state under high consolidation and mutation pressure.

## Highlights & Insights

- **Evaluation Paradigm Innovation**: Upgrades long-term memory from "Do you remember?" to "Do you know what to forget?", with the FAMA metric being elegantly designed and practical.
- **Simulation-Driven Construction**: A pipeline of structured seeds → session simulation → dialogue generation → automated verification enables large-scale, high-quality benchmark construction.
- **Reasoning is the Biggest Bottleneck**: Synthetic reasoning across timelines is the fatal weakness of all existing systems.
- **Unprecedented Benchmark Scale**: Average of 1991 sessions and 1171.4 memory operations per persona in the Quarterly setting.

## Limitations & Future Work

- Based on simulated data rather than real user interactions, which may pose ecological validity issues.
- Evaluation relies on LLM-as-a-judge (though showing 88.3% agreement with human annotation).
- All agents use a unified GPT-4o-mini backend, which may underestimate the performance of certain agents under stronger LLMs.
- The effects of memory compression/summarization strategies were not explored.

## Related Work & Insights

- **LoCoMo (Maharana et al., 2024)**: Multi-session dialogue benchmark; 94% of questions require ≤2 sessions.
- **LongMemEval (Wu et al., 2024)**: Million-token evaluation, but with limited memory updates.
- **PersonaMem (Jiang et al., 2025)**: Personalized decision evaluation, with at most 3 memory mutations.
- **MemoryAgentBench (Hu et al., 2025)**: Agentic memory evaluation.
- **Insight**: The core challenge of long-term memory is not capacity, but rather temporal consistency management and knowledge state maintenance; evaluating AI memory capacity should not only look at "how much is remembered," but more importantly at "how to choose when information is contradictory."

## Rating

- **Novelty**: ★★★★☆ — FAMA metric and large-scale memory mutation evaluation are major contributions.
- **Experimental Thoroughness**: ★★★★☆ — Covers 4 LLMs and 6 memory agents comprehensively.
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated with solid quantitative comparisons to prior work.
- **Value**: ★★★★★ — Fills a critical gap in the evaluation of personalized agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)
- [\[ICML 2026\] RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents](../../ICML2026/recommender/rgmem_renormalization_group-inspired_memory_evolution_for_language_agents.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)
- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](memrec_collaborative_memory-augmented_agentic_recommender_system.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)

</div>

<!-- RELATED:END -->
