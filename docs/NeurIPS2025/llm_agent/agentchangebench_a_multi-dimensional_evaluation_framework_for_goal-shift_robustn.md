---
title: >-
  [Paper Note] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness
description: >-
  [NeurIPS 2025][LLM Agent][benchmark] AgentChangeBench is the first benchmark that systematically evaluates the adaptability of LLM agents when user goals shift mid-conversation: 315 base tasks × 9 variants = 2,835 sequences, spanning 3 enterprise domains (banking/retail/airline) and 5 user personas. It introduces 4 complementary metrics including GSRT (Goal-Shift Recovery Time), revealing efficiency and robustness gaps masked by high pass@k—e.g., GPT-4o achieves 92.2% airline recovery rate yet 89.1% retail redundancy rate.
tags:
  - NeurIPS 2025
  - LLM Agent
  - benchmark
  - goal shift
  - multi-turn dialogue
  - agent robustness
  - tool calling
date: 2026-05-08
content_hash: ad551498c7a895eb
---

# AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness

**Conference**: NeurIPS 2025
**arXiv**: [2510.18170](https://arxiv.org/abs/2510.18170)
**Code**: Available (released with paper)
**Area**: LLM Agent
**Keywords**: benchmark, goal shift, multi-turn dialogue, agent robustness, tool calling

## TL;DR
AgentChangeBench is the first benchmark that systematically evaluates the adaptability of LLM agents when user goals shift mid-conversation: 315 base tasks × 9 variants = 2,835 sequences, spanning 3 enterprise domains (banking/retail/airline) and 5 user personas. It introduces 4 complementary metrics including GSRT (Goal-Shift Recovery Time), revealing efficiency and robustness gaps masked by high pass@k—e.g., GPT-4o achieves 92.2% airline recovery rate yet 89.1% retail redundancy rate.

## Background & Motivation

**State of the Field**: Existing agent benchmarks (τ-bench, τ²-bench, AgentBench) assume user goals remain fixed throughout a conversation and rely primarily on pass@k as the evaluation metric.

**Limitations of Prior Work**: (a) In real-world scenarios, users frequently change their goals—a banking customer may shift from checking account balances to disputing fraud, while an airline customer may switch from flight lookup to rebooking—yet no benchmark systematically tests this "goal-shift" capability; (b) pass@k is a binary metric that cannot distinguish between "immediately adapting to the new goal" and "eventually adapting after 10 turns"; (c) high tool-call accuracy does not imply efficiency—agents may issue large numbers of redundant calls.

**Root Cause**: pass@k compresses all success/failure signals into a single number, obscuring substantial variation in recovery speed, tool efficiency, and redundancy—dimensions that are critical for enterprise deployment in terms of both cost and user experience.

**Paper Goals**: To construct a benchmark with explicit "goal-shift sequences" and multi-dimensional evaluation metrics that quantify agent adaptability under dynamic user goals.

**Starting Point**: Each task is defined as an ordered goal sequence (e.g., `["authentication","transactions","dispute"]`); a persona-conditioned user simulator naturally triggers goal shifts during dialogue, and the multi-stage GSRT metric quantifies the recovery process.

**Core Idea**: Replace pass@k with explicit goal sequences and four-dimensional metrics (success rate / efficiency / redundancy rate / recovery time) to reveal the true robustness of agents under dynamic goals.

## Method

### Overall Architecture
AgentChangeBench consists of two components—a dataset and an evaluation protocol. The dataset comprises 315 tasks (banking 50 + airline 100 + retail 165), each specifying a persona, known/unknown information, and an ordered goal list. The evaluation protocol runs simulations using the τ²-bench harness, where a persona-conditioned user simulator naturally triggers goal shifts (e.g., after completing one topic or when the agent asks "Is there anything else I can help you with?"), and agent performance is then assessed using 4 metrics.

### Key Designs

1. **Four-Dimensional Evaluation Metrics**:

    - **TSR (Task Success Rate)**: Weighted average = 0.25 × information communicated + 0.45 × action executed + 0.30 × policy compliance. Unlike the binary pass@k, TSR awards partial credit for incremental progress.
    - **TUE (Tool Use Efficiency)**: $TUE = 0.6T + 0.4P$, where $T$ is tool-call correctness and $P$ is parameter validity. In experiments, $P$ is nearly saturated (0.986), so variation is primarily driven by $T$.
    - **TCRR (Tool-Call Redundancy Rate)**: The proportion of repeated calls with identical tool name and parameters within a 3-turn window. Directly measures waste—high TCRR implies greater API cost and longer conversations.
    - **GSRT (Goal-Shift Recovery Time)**: Decomposes recovery into three stages—*acknowledgment* (how many turns until the agent acknowledges the new goal), *tool* (how many turns until the first relevant tool call), and *outcome* (how many turns until the new goal is completed). Recovery is counted as successful if the agent acknowledges the new goal without escalating to a human agent.
    - **Design Motivation**: The four metrics cover four complementary dimensions—Can the agent succeed? (TSR), Did it use the right tools? (TUE), Was there any waste? (TCRR), How quickly did it adapt? (GSRT).

2. **Goal Sequence Design**:

    - **Function**: Each task's JSON schema explicitly declares `goal_shifts: {required_shifts: k, goals: [g1,...,g{k+1}]}`.
    - **Mechanism**: Goal shifts are triggered by the user simulator at natural conversational junctures (after turn 4, after issue resolution, or when the agent poses a question); the agent receives no explicit signal.
    - **Coverage**: Over 150 goal labels (e.g., reservation / baggage / cancellation / returns / fraud_response).

3. **5 User Personas**:

    - EASY_1 (polite, detailed), EASY_2 (distracted, casual), MEDIUM_1 (business-like, impatient), MEDIUM_2 (curious, asks many questions), HARD_1 (skeptical, demands proof).
    - **Design Motivation**: Different personas elicit distinct goal-shift patterns and agent response behaviors.

## Key Experimental Results

### Main Results (TSR, Cross-Domain and Cross-Model)

| Domain | GPT-4o | Claude-3.7-Sonnet | Gemini-2.5-Flash |
|--------|--------|-------------------|-----------------|
| Banking | 51.25% | **57.54%** | 47.36% |
| Airline | 62.19% | **65.14%** | 46.98% |
| Retail | 56.48% | **79.57%** | 58.03% |

### Goal-Shift Robustness (New Tasks)

| Domain | Model | TSR | Recovery Rate | TCRR↓ |
|--------|-------|-----|--------------|-------|
| Airline | GPT-4o | 59.53% | **92.2%** | **13.54%** |
| Airline | Claude-3.7 | **69.90%** | 79.2% | 24.11% |
| Airline | Gemini-2.5 | 39.97% | 48.6% | 14.46% |
| Retail | GPT-4o | 50.68% | 88.0% | 89.14% ⚠️ |
| Retail | Claude-3.7 | **79.57%** | **89.5%** | 65.38% |
| Retail | Gemini-2.5 | 51.26% | 53.5% | 66.45% |

### Persona Analysis

| Persona | TSR | TUE | GSRT Recovery |
|---------|-----|-----|--------------|
| MEDIUM_2 (curious) | **0.580** | 0.990 | **0.756** |
| MEDIUM_1 (impatient) | 0.554 | 0.978 | 0.916 |
| EASY_1 (polite) | 0.533 | 0.960 | 0.849 |
| EASY_2 (distracted) | 0.475 | 0.971 | 0.585 |
| HARD_1 (skeptical) | 0.430 | 0.946 | 0.585 |

### Key Findings
- **High recovery rate ≠ high efficiency**: GPT-4o achieves the highest airline recovery rate (92.2%), yet its retail redundancy rate reaches 89.1%—in the retail domain the agent repeatedly queries the same tools and, although it ultimately completes the task, does so extremely wastefully.
- **Gemini collapses under goal shifts**: Its airline recovery rate is only 48.6%, far below GPT-4o's 92.2%. Gemini tends to continue executing prior plans rather than responding to the new goal.
- **pass@k masks critical differences**: pass@k for new tasks frequently drops to 0.0 (e.g., GPT-4o and Gemini on airline/retail), whereas TSR remains in the 40–60% range, indicating that agents make substantial partial progress—progress that pass@k entirely fails to capture.
- **Parameter validity is saturated**: All models exhibit parameter validity $P \approx 0.986$; variation in TUE is driven primarily by a long-tail distribution in tool correctness $T$.
- **Persona has a significant impact**: HARD_1 (skeptical) achieves a TSR of only 0.430, while MEDIUM_2 (curious) reaches 0.580. Skeptical users generate longer conversations (avg. 19.2 turns) with lower recovery rates.
- **Domain difficulty varies substantially**: Banking is the hardest (multi-step authentication + complex policies), airline is intermediate (fastest recovery), and retail exhibits the most severe redundancy.

## Highlights & Insights
- **The three-stage GSRT decomposition** is the core methodological contribution: decomposing "recovery" into acknowledgment → tool → outcome enables researchers to pinpoint where an agent stalls—whether it is slow to detect goal changes, slow to select appropriate tools, or slow to execute.
- **TCRR fills an evaluation gap**: Existing benchmarks entirely overlook redundancy, yet redundancy directly affects API costs and user wait times. A retail redundancy rate of 89% would be wholly unacceptable in a production environment.
- **Explicit vs. implicit goal shifts**: The current benchmark only tests explicit goal shifts (where the user clearly signals a topic change); the harder case of implicit goal drift (where user intent gradually changes in meaning) is left for future work.

## Limitations & Future Work
- Goal shifts follow preset sequences and are typically triggered explicitly; implicit goal drift, conflicting goals, and concurrent multi-goal scenarios are not evaluated.
- All 5 personas are relatively cooperative; adversarial, deceptive, or non-cooperative users are absent.
- Only 3 closed-source commercial models are evaluated (GPT-4o / Claude-3.7 / Gemini-2.5); open-source models are not included for comparison.
- Coverage is limited to customer service API tools; more complex tool types such as code execution and web browsing are not addressed.
- Many of the 315 tasks reuse templates from τ/τ²-bench, limiting originality.

## Related Work & Insights
- **vs. τ-bench**: 234 tasks × 2 domains, no goal shifts, only pass@k. AgentChangeBench adds explicit goal sequences and 4-dimensional metrics.
- **vs. τ²-bench**: 105 tasks × 1 domain × 5 personas, with implicit goal shifts. AgentChangeBench introduces explicit goal sequence declarations and GSRT.
- **vs. AgentBench**: 8 environments but all goals are static, no personas, and no redundancy evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark to systematically evaluate goal-shift robustness; the GSRT metric design is particularly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 315 tasks × 3 models × 5 personas × 4 metrics with thorough analysis.
- Writing Quality: ⭐⭐⭐⭐ Metric definitions are rigorous and failure mode analysis is clear.
- Value: ⭐⭐⭐⭐ Provides direct guidance for enterprise agent deployment decisions.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] CORE: Full-Path Evaluation of LLM Agents Beyond Final State](core_full-path_evaluation_of_llm_agents_beyond_final_state.md)
- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[NeurIPS 2025\] AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents](agentauditor_humanlevel_safety_and_security_evaluation_for_l.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)

<!-- RELATED:END -->
