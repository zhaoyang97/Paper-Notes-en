---
title: >-
  [Paper Note] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
description: >-
  [ICLR 2026][Code Intelligence][Agent Memory] This paper proposes ReasoningBank, a memory framework that distills generalizable reasoning strategies from both successful and failed experiences as judged by the agent itsel…
tags:
  - "ICLR 2026"
  - "Code Intelligence"
  - "Agent Memory"
  - "Reasoning Strategy"
  - "Test-Time Scaling"
  - "Self-Evolving"
  - "Experiential Learning"
date: 2026-05-08
content_hash: d168899643d3d25c
---

# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory

**Conference**: ICLR 2026
**arXiv**: [2509.25140](https://arxiv.org/abs/2509.25140)
**Code**: [google-research/reasoning-bank](https://github.com/google-research/reasoning-bank)
**Area**: Code Intelligence
**Keywords**: Agent Memory, Reasoning Strategy, Test-Time Scaling, Self-Evolving, Experiential Learning

## TL;DR
This paper proposes ReasoningBank, a memory framework that distills generalizable reasoning strategies from both successful and failed experiences as judged by the agent itself, and introduces memory-aware test-time scaling (MaTTS) to establish a synergy between memory and test-time scaling. The approach consistently outperforms baselines on WebArena, Mind2Web, and SWE-Bench (up to 34.2% relative improvement) while reducing interaction steps by 16%.

## Background & Motivation
As LLM agents are increasingly deployed in persistent, real-world roles, they naturally encounter continuous streams of tasks. A critical limitation, however, is their inability to learn from accumulated interaction history—each new task is approached from scratch, forcing agents to discard valuable insights and repeat past mistakes.

Existing agent memory approaches suffer from two major shortcomings:

**Storing only raw trajectories or successful patterns**: Synapse stores raw trajectories as in-context memory, while AWM extracts workflows from trajectories, yet neither distills higher-level, transferable reasoning patterns.

**Neglecting the value of failure experiences**: An overemphasis on successful experiences prevents agents from learning from their own failures.

Core Idea: Both successful and failed experiences are distilled into generalizable reasoning strategies (rather than concrete operational steps) and stored in a structured memory bank. Combined with test-time scaling, this generates rich contrastive signals that further improve memory quality.

## Method

### Overall Architecture
ReasoningBank is a closed-loop memory system: the agent receives a new task → retrieves relevant memories from ReasoningBank → uses them to guide decision-making and execution → constructs new memories from the resulting experience → merges them back into ReasoningBank. The entire process requires no ground-truth labels and relies solely on LLM-as-a-judge self-evaluation.

### Key Designs

1. **Memory Schema (Structured Memory Unit)**: Each memory entry consists of three components:

    - **Title**: Briefly identifies the core strategy or reasoning pattern
    - **Description**: A one-sentence summary
    - **Content**: Distilled reasoning steps, decision rationale, or operational insights

   The schema is designed to be both human-readable and machine-usable—more abstract than raw trajectories (capturing common patterns) yet concrete enough to be actionable (including reasoning steps).

2. **Three-Step Closed-Loop Pipeline**:

    - **Memory Retrieval**: Embedding-based similarity search retrieves top-k relevant memory entries from ReasoningBank and injects them into the agent's system prompt.
    - **Memory Construction**: After task completion, LLM-as-a-judge evaluates each trajectory as success or failure. Successful trajectories contribute "validated strategies," while failed trajectories contribute "counterfactual signals and pitfall warnings." Up to 3 memory entries are extracted per trajectory.
    - **Memory Consolidation**: New memory entries are directly appended to ReasoningBank (a deliberately minimal merging strategy to isolate the effect of memory content).

3. **MaTTS: Memory-Aware Test-Time Scaling**:
   ReasoningBank is integrated with test-time scaling to establish a bidirectional synergy:
    - **Parallel Scaling**: Multiple trajectories are generated for the same query; self-contrast compares outcomes across trajectories to identify consistent reasoning patterns and filter spurious solutions, providing diverse contrastive signals that make memories more reliable.
    - **Sequential Scaling**: Within a single trajectory, iterative self-refinement captures intermediate reasoning attempts, corrections, and insights as valuable memory signals.

   **Key Distinction**: Vanilla TTS processes multiple trajectories independently and extracts memories from each in isolation (suboptimal); MaTTS leverages the intrinsic contrastive signals arising from redundant exploration to curate higher-quality memories. Better memories guide scaling toward more promising trajectories, while richer experiences forge stronger memories—forming a positive feedback loop.

### Loss & Training
- No training required: the entire system is based on LLM in-context learning.
- Backbone: Gemini-2.5-flash/pro, Claude-3.7-sonnet
- Environments: BrowserGym (web browsing), Bash-only (SWE)
- ReAct-style agent with top-1 retrieval by default

## Key Experimental Results

### Main Results — WebArena

| Method | Shopping SR | Admin SR | Gitlab SR | Reddit SR | Overall SR | Steps |
|--------|------------|---------|----------|----------|------------|-------|
| No Memory (Gemini-2.5-flash) | 39.0 | 44.5 | 33.9 | 55.7 | 40.5 | 9.7 |
| Synapse | 40.6 | 45.1 | 35.6 | 59.4 | 42.1 | 9.2 |
| AWM | 44.4 | 46.7 | 37.2 | 62.3 | 44.1 | 9.0 |
| **ReasoningBank** | **49.7** | **51.1** | **40.6** | **67.0** | **48.8** | **8.3** |
| No Memory (Gemini-2.5-pro) | 45.5 | 51.1 | 35.0 | 71.7 | 46.7 | 8.8 |
| **ReasoningBank** (pro) | **51.9** | **56.6** | **44.4** | **80.2** | **53.9** | **7.4** |

### SWE-Bench-Verified

| Method | Resolve Rate | Steps |
|--------|-------------|-------|
| No Memory (Gemini-2.5-flash) | 34.2 | 30.3 |
| ReasoningBank | **38.8** | **27.5** |
| No Memory (Gemini-2.5-pro) | 54.0 | 21.1 |
| ReasoningBank (pro) | **57.4** | **19.8** |

### MaTTS Scaling Experiment (WebArena-Shopping, k = scaling factor)

| Configuration | k=1 | k=3 | k=5 |
|--------------|-----|-----|-----|
| MaTTS w/o memory (parallel) | 39.0 | 40.6 | 42.2 |
| MaTTS w/o aggregation (vanilla TTS) | 49.7 | 52.4 | 52.4 |
| **MaTTS (parallel)** | **49.7** | 53.5 | **55.1** |
| **MaTTS (sequential)** | 49.7 | **54.5** | 54.5 |

### Ablation Study

| Configuration | Key Metric | Note |
|--------------|-----------|------|
| Successful trajectories only | ReasoningBank: 46.5 SR | Using only successes already outperforms baselines |
| Success + failure trajectories | ReasoningBank: 49.7 SR | Failure experiences contribute an additional 3.2 pp gain |
| Synapse + failure trajectories | 41.7 SR | Synapse cannot effectively exploit failure signals |
| AWM + failure trajectories | 42.2 SR (degraded) | AWM processing failures leads to performance drop |
| Retrieval count k=1/2/3/4 | 49.7/46.0/45.5/44.4 | k=1 is optimal; excessive memories introduce noise |

### Key Findings
- ReasoningBank consistently outperforms all baselines across **all datasets and all backbones**.
- Substantial efficiency gains: successful cases average 2.1 fewer steps (26.9% relative reduction), indicating that memory helps agents converge on correct solutions faster.
- Advantages are especially pronounced in cross-domain generalization (Multi subset, Mind2Web cross-domain).
- MaTTS synergy: only ReasoningBank benefits from scaling (Pass@1 increases from 49.7 to 50.8), while weaker memory degrades under scaling.

## Highlights & Insights
- **Failure experiences are fully exploited for the first time**: Unlike prior methods that rely solely on successful trajectories, ReasoningBank demonstrates that counterfactual signals from failures are a more powerful source of generalization.
- **Emergent Behaviors**: Memory entries naturally evolve—from low-level execution strategies → adaptive checking → compositional reasoning—exhibiting dynamics analogous to emergent learning in reinforcement learning.
- **Experience-driven scaling as a new scaling dimension**: Traditional scaling merely increases computation; MaTTS couples memory quality with scaling, opening a new avenue for improving agent capabilities.
- **Design simplicity**: The entire system requires no training and relies solely on in-context learning, embedding retrieval, and LLM judging, making it straightforward to deploy.

## Limitations & Future Work
- Reliance on LLM-as-a-judge for correctness signals means that judge errors may lead to memory contamination.
- The memory consolidation strategy is overly simplistic (direct append); at large scale, memory pool growth may degrade retrieval efficiency.
- Retrieval relies on embedding similarity alone, lacking a reasoning-aware retrieval mechanism.
- Memory forgetting and update strategies remain unexplored (stale memories may introduce interference).
- Validation is limited to web browsing and SWE environments; applicability to other agent settings (e.g., embodied environments) has yet to be investigated.

## Related Work & Insights
- **vs. Synapse**: Stores raw trajectories as exemplars; memory granularity is too coarse and lacks transferability.
- **vs. AWM (Agent Workflow Memory)**: Extracts workflows from successful trajectories, but (1) uses only successful experiences and (2) transfers poorly across domains (degrades to 3.4 SR on the Multi subset).
- **vs. ExpeL**: Also leverages successes and failures, but memories are stored as tips rather than the structured reasoning strategies of ReasoningBank.
- **Insight**: An agent's memory system should mirror human cognition—recording not only "how to succeed" but also "why failures occur" and "abstract decision-making principles."

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[ICLR 2026\] CARD: Towards Conditional Design of Multi-agent Topological Structures](card_towards_conditional_design_of_multi-agent_topological_structures.md)
- [\[ICLR 2026\] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)

</div>

<!-- RELATED:END -->
