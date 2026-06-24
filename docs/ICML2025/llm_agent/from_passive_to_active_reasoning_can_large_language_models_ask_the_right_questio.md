---
title: >-
  [Paper Note] From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?
description: >-
  [ICML 2025][LLM Agent][Active Reasoning] This paper introduces AR-Bench, a benchmark specifically designed to evaluate the active reasoning capabilities of LLMs. It features three task families: detective cases, situation puzzles, and guessing numbers. Experiments reveal that state-of-the-art models such as GPT-4o perform far worse than humans in scenarios where they must actively ask questions to retrieve missing information, exposing a massive gap between passive and active…
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "Active Reasoning"
  - "benchmark"
  - "information acquisition"
  - "multi-turn interaction"
  - "reasoning evaluation"
date: 2026-05-08
content_hash: 3feecd6ebf62067b
---

# From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?

**Conference**: ICML 2025  
**arXiv**: [2506.08295](https://arxiv.org/abs/2506.08295)  
**Code**: [https://github.com/tmlr-group/AR-Bench](https://github.com/tmlr-group/AR-Bench)  
**Area**: LLM Agent  
**Keywords**: Active Reasoning, benchmark, information acquisition, multi-turn interaction, reasoning evaluation

## TL;DR
This paper introduces AR-Bench, a benchmark specifically designed to evaluate the active reasoning capabilities of LLMs. It features three task families: detective cases, situation puzzles, and guessing numbers. Experiments reveal that state-of-the-art models such as GPT-4o perform far worse than humans in scenarios where they must actively ask questions to retrieve missing information, exposing a massive gap between passive and active reasoning.

## Background & Motivation
**Background**: LLMs excel in passive reasoning tasks such as mathematics and coding, where sufficient context and information are provided to deduce the answer.

**Limitations of Prior Work**: A large number of real-world scenarios require operating under incomplete information—asking for user preferences in travel planning, or inquiring about symptoms in medical diagnosis—requiring models to actively acquire information.

**Key Challenge**: Existing evaluations focus almost entirely on passive reasoning. The few existing active reasoning datasets (e.g., 20 Questions, MediQ) are either too simple or lack symbolic feedback and complex reasoning.

**Goal**: To construct a comprehensive active reasoning benchmark, AR-Bench, to systematically evaluate the ability of LLMs to acquire critical information through multi-turn interactions and reason out answers.

**Key Insight**: Designing three complementary task families—detective cases (commonsense reasoning), situation puzzles (lateral reasoning), and guessing numbers (symbolic reasoning)—covering different feedback types.

**Core Idea**: The essence of active reasoning is not "deducing the right answer," but "asking the right questions" to acquire missing information.

## Method

### Overall Architecture
AR-Bench consists of 6,040 puzzles categorized into three task families. The evaluation pipeline simulates multi-turn conversations: the evaluated model acts as a player that asks questions to an NPC (Llama-3.1-405B or rule-based functions) within 25 turns to retrieve information and eventually provides an answer. Two evaluation schemes are designed: outcome metrics (accuracy/F1/exact match) and process metrics (key question coverage).

### Key Designs

1. **Detective Cases (DC)**:

    - **Function**: Simulates a detective interrogating 5 suspects, each possessing a different role (helpful or deceptive).
    - **Mechanism**: The player takes turns to select suspects and ask them questions, collecting clues to eventually pinpoint the real culprit.
    - **Design Motivation**: To test the model's commonsense reasoning and information integration capabilities under complex and noisy feedback.
    - **Scale**: 400 train / 100 test, averaging 564 tokens per problem, with an answer space of 5.

2. **Situation Puzzles (SP)**:

    - **Function**: Classic lateral thinking puzzles where the player reconstructs the truth behind an unconventional premise using Yes/No questions.
    - **Mechanism**: Requires indirect and creative thinking to assemble a complete story from fragmented clues.
    - **Design Motivation**: To test logical reasoning capabilities under an open (infinite) answer space.
    - **Scale**: 400 train / 100 test, averaging 178 tokens.

3. **Guessing Numbers (GN)**:

    - **Function**: Guessing a 4-digit unique number, receiving feedback of "how many digits are correct in position + how many are correct in value but in the wrong position" for each guess.
    - **Mechanism**: Each guess serves as an information query; the model needs to maximize information gain and narrow down the hypothesis space through symbolic reasoning.
    - **Design Motivation**: To test the ability to perform systematic reasoning utilizing precise symbolic feedback.
    - **Scale**: 5,040 possible answers.

4. **Dataset Construction and Evaluation**:

    - **Function**: A four-stage automated generation process (core sampling $\rightarrow$ tree expansion $\rightarrow$ key question extraction $\rightarrow$ puzzle synthesis) followed by human verification.
    - **Process Evaluation Formula**: $\text{Score}(Q, s_t) = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(f(s_t, q_i) = 1)$
    - **Judge function** $f$ is implemented by Llama-3.1-405B to determine whether the current conversation state is sufficient to answer each key question.

### Evaluation Methodology
The evaluation covers 8 models (Llama-3.1-8B/70B/405B, Qwen-2.5-3B/7B, QwQ-32B, GPT-4o-mini, GPT-4o) and 6+2 methods (zero-shot, few-shot, few-shot+instruction, ToT, SFT, DPO, Proactive CoT, UoT).

## Key Experimental Results

### Main Results

| Model/Method | DC Accuracy | SP F1 | GN Exact Match |
|-----------|----------|-------|------------|
| GPT-4o (zero-shot) | ~60% | ~50% | 35% |
| Llama-3.1-405B | Upper-middle | Upper-middle | Middle |
| Llama-3.1-8B + SFT | Average | Average | **0%** |
| Llama-3.1-8B + DPO | Average | Below zero-shot | Below zero-shot |
| Proactive CoT | Marginal gain (SP) | — | — |
| UoT | Below zero-shot | — | — |
| **Human** | **Significantly higher than all models** | **Significantly higher than all models** | **Significantly higher than all models** |

### Process Analysis (Information Acquisition Efficiency)

| Interaction Phase | Average Process Score Gain | Explanation |
|---------|---------------|------|
| Turns 5-10 | +7.7% | Rapid early progress |
| Turns 20-25 | +2.5% | Severe stagnation in later stages |
| First 50 turns scaling | +45.8% | Cumulative process score growth |
| 50-100 turns scaling | Only +6.7% | Clear diminishing marginal returns |

### Error Pattern Analysis

| Task | Error Type | Llama-8B | GPT-4o |
|------|---------|---------|--------|
| DC | Timeline Misunderstanding | 10% | 31% |
| DC | Ignoring Evidence | 61% | 15% |
| SP | Unsupported Assumptions | 90% | 72% |
| GN | Feedback Comprehension Error | 78% | 61% |
| GN | Incomplete Testing | 81% | 55% |

### Key Findings
- GPT-4o leads on passive reasoning leaderboards, but only achieves 35% exact match on GN in active reasoning.
- SFT scores 0 on GN, and DPO performs worse than zero-shot on both SP and GN.
- Models tend to ask vague and repetitive questions, "converging to local optima" in later turns.
- Even scaling up to 100 turns of interaction fails to fully solve the tasks.
- Verifier reliability varies by task: deterministic verification works best for GN, while heuristic verification is less effective for DC/SP.
- Larger models are stronger in both information retrieval (quality of questions) and information processing (reasoning with incomplete information).

## Highlights & Insights
- Systematically distinguishes and evaluates passive reasoning vs. active reasoning for the first time, filling a major evaluation gap.
- Core Insight: **Reasoning capability $\neq$ Asking capability**—models excel at deriving answers from known information, but struggle to identify and acquire missing information.
- The process evaluation metric is elegantly designed, evaluating not just the final answer but also tracking the quality of information acquisition in each turn.
- The three tasks cover different types of feedback (narrative/boolean/symbolic) and answer space dimensions (5/$\infty$/5040).
- "Rapid early progress followed by severe late stagnation" points to fundamental challenges in context management and long-term strategic planning.

## Limitations & Future Work
- NPC response quality relies on Llama-3.1-405B, which may introduce evaluation bias.
- The three tasks remain simplified toy scenarios, still far from real-world applications (e.g., medical diagnosis).
- Focuses more on revealing problems rather than presenting solutions, lacking novel methods to effectively improve active reasoning.
- Does not explore more practical information acquisition methods such as RAG or tool calling.

## Related Work & Insights
- Comparison with Proactive CoT and UoT shows that existing active reasoning methods have limited effectiveness in complex scenarios.
- Points to several research directions: interactive learning, real-time feedback loops, and environment-aware training objectives.
- Connection to long-context research: degradation in later stages may stem from attention dilution due to overly long contexts.
- Insight: Future LLM agents need the capability to "ask good questions", rather than just "answer questions well".

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulates and evaluates the passive vs. active reasoning paradigm for the first time, opening up a new evaluation dimension.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models $\times$ 8 methods $\times$ 3 tasks, complemented by human controls, process analysis, error pattern analysis, and multi-angle ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, observations numbered for easy citation, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Exposes a fundamental blind spot in LLM reasoning capabilities, with profound implications for agent system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On Information Self-Locking in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICML2026/llm_agent/on_information_self-locking_in_reinforcement_learning_for_active_reasoning_of_ll.md)
- [\[ICLR 2026\] GPS: Graph-guided Proactive Information Seeking in Large Language Models](../../ICLR2026/llm_agent/gps_graph-guided_proactive_information_seeking_in_large_language_models.md)
- [\[ICLR 2026\] Can Language Models Discover Scaling Laws?](../../ICLR2026/llm_agent/can_language_models_discover_scaling_laws.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](../../ACL2025/llm_agent/toolhop_multi_hop_tool_use.md)

</div>

<!-- RELATED:END -->
