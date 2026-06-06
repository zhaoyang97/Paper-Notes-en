---
title: >-
  [Paper Note] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents
description: >-
  [ACL 2026][Recommender Systems][long-term memory benchmark] This paper proposes the Memora benchmark and the FAMA metric, extending long-term memory evaluation beyond shallow fact retrieval to memory consolidation and mu…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "long-term memory benchmark"
  - "memory consolidation"
  - "memory mutation"
  - "forgetting-aware evaluation"
  - "personalized agent"
  - "Memora"
  - "FAMA"
date: 2026-05-08
content_hash: 2045f4ae5ed5a350
---

# From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents

**Conference**: ACL 2026  
**arXiv**: [2604.20006](https://arxiv.org/abs/2604.20006)  
**Code**: [GitHub](https://github.com/geniesinc/Memora)  
**Area**: Long-Term Memory / Personalized Agents  
**Keywords**: long-term memory benchmark, memory consolidation, memory mutation, forgetting-aware evaluation, personalized agent, Memora, FAMA

## TL;DR

This paper proposes the Memora benchmark and the FAMA metric, extending long-term memory evaluation beyond shallow fact retrieval to memory consolidation and mutation handling spanning weeks to months, revealing systematic failures of existing LLMs and memory agents under frequent knowledge updates.

## Background & Motivation

**State of the Field**: Personalized agents require persistent cross-session memory over long-term interactions. Existing benchmarks (LoCoMo, LongMemEval, PersonaMem, etc.) primarily operationalize long-term memory as "cross-session fact retrieval," with 94% of LoCoMo questions and 85% of LongMemEval questions requiring reference to at most 2 historical sessions.

**Limitations of Prior Work**: Existing benchmarks suffer from two core deficiencies: (1) minimal memory consolidation demands—on average requiring reference to only ~1 historical session; and (2) near-absent memory mutation testing—LongMemEval supports at most 2 updates and PersonaMem at most 3. Evaluation metrics reward memory presence without penalizing erroneous use of outdated information.

**Root Cause**: Human cognition not only recalls but also accumulates experience, reconciles changes, and maintains a coherent world model. Current evaluation frameworks essentially only test "whether an isolated piece of historical information can be retrieved," failing to expose models' true capabilities in memory evolution and conflict resolution.

**Paper Goals**: To construct an evaluation framework that substantially surpasses existing benchmarks in both memory consolidation (up to 309 cross-session references) and memory mutation (up to 94 updates/deletions), and to propose an evaluation metric that penalizes reliance on outdated memories.

**Starting Point**: Memory tasks are defined along three dimensions—Remembering, Reasoning, and Recommending—covering three temporal granularities: weekly, monthly, and quarterly.

**Core Idea**: Long-term memory is not merely about "remembering" but about "knowing when to forget." The FAMA metric exposes critical weaknesses in existing systems by explicitly penalizing dependence on stale memories.

## Method

**Overall Architecture**: A simulation-driven pipeline that begins from persona seed data, simulates user interaction sequences (weekly/monthly/quarterly), converts them into multi-turn dialogues, and derives evaluation tasks from memory trajectories.

**Key Designs**:

1. **Memory Evolution Simulation System**
    - **Function**: Generates long-term interaction data with realistic memory dynamics.
    - **Mechanism**: Ten professional personas (engineer, researcher, etc.) are constructed, each maintaining three memory categories (preferences/activities/goals). The simulator updates persistent memory states after each session, supporting preference drift, recurring activities, and progressive long-term tasks, while also including memory-neutral sessions. Complete memory trajectories are recorded as ground truth.
    - **Design Motivation**: Under the quarterly setting, average memory consolidation reaches 28.4 (vs. ~1 in existing benchmarks) and mutation reaches 14.8 (vs. ~0–2), ensuring that evaluation questions have a well-defined memory state basis.

2. **Multi-Agent Dialogue Generation and Quality Assurance**
    - **Function**: Converts simulated session specifications into high-quality multi-turn dialogues.
    - **Mechanism**: A multi-agent prompting framework generates dialogues supporting both memory-neutral and memory-anchored conversation turns. Quality is ensured through automated memory-anchoring checks evaluated independently by 3 LLMs (acceptance requires unanimous agreement) plus 5% manual verification.
    - **Design Motivation**: Prevents the generation process from introducing untracked spurious memory details, ensuring strict alignment between dialogues and the underlying memory trajectories.

3. **Forgetting-Aware Memory Accuracy (FAMA)**
    - **Function**: Jointly evaluates "retaining valid information" and "forgetting outdated information."
    - **Mechanism**: $\text{FAMA} = \max(0,\ \text{MPA} - \lambda \cdot (1 - \text{FAA}))$, where MPA is Memory Presence Accuracy, FAA is Forgetting Absence Accuracy, and $\lambda = N_{\text{forget}} / (N_{\text{presence}} + N_{\text{forget}})$ provides dynamic per-question weighting. Three LLM judges (GPT-4.1, Claude Haiku 4.5, Gemini 2.5 Flash) evaluate each criterion by majority vote.
    - **Design Motivation**: Traditional metrics reward correct recall without penalizing the use of stale information. FAMA explicitly constrains forgetting capability via the FAA term, while dynamic $\lambda$ weighting ensures fair comparison across questions with varying mutation rates.

## Key Experimental Results

**LLM Base Model Performance (FAMA, 0–100)**:

| Model | Remembering (W/M/Q) | Recommending (W/M/Q) | Reasoning (W/M/Q) |
|-------|---------------------|----------------------|--------------------|
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
- **Reasoning nearly completely fails**: Reasoning scores for all models and agents are mostly below 10; GPT-5.2 scores 0 on the Monthly setting.
- **Memory agents significantly outperform LLMs on Remembering** (e.g., A-Mem 71.8 vs. GPT-5.2 25.3), but the advantage on Reasoning is limited.
- **Performance consistently degrades as temporal span increases**: Remembering scores drop substantially for all systems from Weekly to Quarterly.
- **Reasoning tokens yield unstable benefits**: Helpful in certain scenarios (e.g., Gemini Monthly Reasoning improves from 4 to 10), but overall improvement is limited.
- **All systems frequently reuse invalidated memories**, exposing the fundamental difficulty of maintaining consistent belief states under high consolidation and mutation pressure.

## Highlights & Insights

- **Evaluation paradigm shift**: Long-term memory evaluation is advanced from "Do you remember?" to "Do you know what to forget?"; the FAMA metric is both elegant and practically useful.
- **Simulation-driven construction methodology**: The structured pipeline of seed data → session simulation → dialogue generation → automated verification enables large-scale, high-quality benchmark construction.
- **Reasoning is the dominant bottleneck**: Cross-timeline information synthesis and reasoning represents the critical weakness of all existing systems.
- **Unprecedented benchmark scale**: Under the quarterly setting, each persona averages 1,991 sessions and 1,171.4 memory operations.

## Limitations & Future Work

- The benchmark is based on simulated rather than real user interactions, which may raise ecological validity concerns.
- Evaluation relies on LLM-as-judge (despite 88.3% agreement with human annotation).
- All agents use a unified GPT-4o-mini backend, potentially underestimating the performance of certain agents with stronger LLMs.
- The effects of memory compression and summarization strategies are not explored.

## Related Work & Insights

- **LoCoMo (Maharana et al., 2024)**: Multi-session dialogue benchmark; 94% of questions require ≤2 sessions.
- **LongMemEval (Wu et al., 2024)**: Million-token evaluation, but with limited memory updates.
- **PersonaMem (Jiang et al., 2025)**: Personalized decision-making evaluation; memory mutations capped at 3.
- **MemoryAgentBench (Hu et al., 2025)**: Agentic memory evaluation.
- **Insights**: The core challenge of long-term memory lies not in capacity but in temporal consistency management and knowledge state maintenance. Evaluating AI memory should focus not only on "how much is remembered" but on "how conflicting information is prioritized and reconciled."

## Rating

- **Novelty**: ★★★★☆ — The FAMA metric and large-scale memory mutation evaluation constitute significant contributions.
- **Experimental Thoroughness**: ★★★★☆ — Covers 4 LLMs and 6 memory agents comprehensively.
- **Writing Quality**: ★★★★☆ — Motivation is clearly articulated with well-quantified comparisons against prior work.
- **Value**: ★★★★★ — Fills a critical gap in evaluation of personalized agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ICLR 2026\] In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](../../ICLR2026/recommender/in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)
- [\[AAAI 2026\] SlideTailor: Personalized Presentation Slide Generation for Scientific Papers](../../AAAI2026/recommender/slidetailor_personalized_presentation_slide_generation_for_scientific_papers.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

</div>

<!-- RELATED:END -->
