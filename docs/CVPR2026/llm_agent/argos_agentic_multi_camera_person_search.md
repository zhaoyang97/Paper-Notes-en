---
title: >-
  [Paper Note] ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search
description: >-
  [CVPR 2026][LLM Agent][multi-camera search] This paper proposes ARGOS, the first benchmark and framework that redefines multi-camera person search as an interactive reasoning problem. An agent conducts multi-turn dialogue with witnesses, invokes spatiotemporal tools, and eliminates candidates under information asymmetry. The benchmark comprises 2,691 tasks across 3 progressive tracks.
tags:
  - CVPR 2026
  - LLM Agent
  - multi-camera search
  - agentic reasoning
  - spatiotemporal topology graph
  - interactive dialogue
  - person search
date: 2026-05-08
content_hash: 2d15060532b38eb6
---

# ARGOS: Who, Where, and When in Agentic Multi-Camera Person Search

**Conference**: CVPR 2026
**arXiv**: [2604.12762](https://arxiv.org/abs/2604.12762)
**Code**: None
**Area**: LLM Agent
**Keywords**: multi-camera search, agentic reasoning, spatiotemporal topology graph, interactive dialogue, person search

## TL;DR
This paper proposes ARGOS, the first benchmark and framework that redefines multi-camera person search as an interactive reasoning problem. An agent conducts multi-turn dialogue with witnesses, invokes spatiotemporal tools, and eliminates candidates under information asymmetry. The benchmark comprises 2,691 tasks across 3 progressive tracks.

## Background & Motivation

1. **Background**: Multi-camera person search is a fundamental requirement in surveillance. Traditional person re-identification relies on clear visual queries; text-driven and interactive methods use only appearance descriptions. Existing spatial reasoning benchmarks and agent evaluation frameworks are limited to single images or general-purpose scenarios.
2. **Limitations of Prior Work**: Existing methods lack active question-planning capabilities and cannot leverage spatiotemporal cues provided by witnesses (e.g., "I saw them in the warehouse, then near the hall a few minutes later"). No method simultaneously integrates multimodal interaction, spatial localization, and temporal reasoning.
3. **Key Challenge**: Real-world person search is inherently an active reasoning problem—requiring decisions about what to ask, when to invoke tools, and how to interpret ambiguous answers under information asymmetry—yet existing benchmarks and methods reduce it to passive visual matching.
4. **Goal**: Define the interactive multi-camera person search task and construct a progressive benchmark encompassing semantic perception (Who), spatial reasoning (Where), and temporal reasoning (When).
5. **Key Insight**: The camera network is encoded as a Spatiotemporal Topology Graph (STTG), serving as both the structural backbone for task construction and a localization tool for the agent, enabling temporal feasibility reasoning under physical constraints.
6. **Core Idea**: A four-module LLM-driven agent (Analyze → Plan → Interview → Interpret) performs multi-turn dialogue reasoning over the STTG, using tool calls to eliminate infeasible candidates.

## Method

### Overall Architecture
The agent receives an initial witness statement and a pedestrian gallery $\mathcal{G}$, and identifies the target person through multi-turn dialogue within a limited number of turns. Three action types are available: querying visual attributes, querying spatial location, and invoking temporal reasoning. The STTG provides camera connectivity and empirically validated transition times.

### Key Designs

1. **Spatiotemporal Topology Graph (STTG)**:
    - **Function**: Encodes the physical connectivity and temporal constraints of the camera network.
    - **Mechanism**: A directed weighted graph $\mathcal{T} = (V, E)$, where nodes are cameras (with region labels) and edges carry type (OVERLAP = shared field of view, SOFT_ADJ = soft adjacency, TRAVEL = long-distance) and transition time statistics $(t_{\min}, t_{\text{med}}, t_{\max}, n)$. Connected components of OVERLAP edges define regions. The STTG serves a dual role: the benchmark generates ground-truth tasks from it, and the agent uses it as an environmental representation for reasoning.
    - **Design Motivation**: Transforms the otherwise vague question "how long does it take to go from A to B" into a computable graph constraint, grounding temporal reasoning in verifiable evidence.

2. **Three-Track Progressive Benchmark**:
    - **Function**: Hierarchically evaluates progressive capabilities from perception to spatiotemporal reasoning.
    - **Mechanism**: Track 1 Who (989 tasks) tests semantic perception—the agent receives the full dialogue log and extracts attributes to filter the gallery. Track 2 Where (550 tasks) tests spatial reasoning—a witness reports seeing the target in a region, and the agent localizes the specific sub-region through spatial and attribute queries (oracle average: 2.02 turns). Track 3 When (1,152 tasks) tests temporal reasoning—a witness reports two sightings at different times and locations, and the agent uses the STTG to eliminate candidates with infeasible transition times (oracle average: 1.89 turns). Turn-Weighted Success (TWS) jointly measures accuracy and turn efficiency.
    - **Design Motivation**: The progressive design enables precise diagnosis of capability bottlenecks; TWS is inspired by SPL from embodied navigation.

3. **Four-Module LLM Agent**:
    - **Function**: Efficiently completes person search through a structured reasoning pipeline.
    - **Mechanism**: The Analyst queries the gallery and computes attribute elimination power; the Planner decides the next action; the Interviewer executes actions via tools (8 tools: gallery query, region structure retrieval, witness interaction, STTG temporal feasibility check, filtering/prediction); the Interpreter parses witness responses and applies filters. A key design is the "information boundary": the agent does not know which attributes a witness can answer (only 3 of 21 are observable) and must make strategic decisions under uncertainty.
    - **Design Motivation**: Decomposes complex reasoning into controllable modular steps, each with well-defined responsibilities.

### Loss & Training
No training is performed. Frozen LLM backbones (GPT-5.2, GPT-4o, GPT-o5-mini, Claude Sonnet 4) are used for direct inference at temperature 0.0 with a budget of 20 turns.

## Key Experimental Results

### Main Results

| Backbone | Track 2 TWS | Track 2 Top-1 | Track 3 TWS | Track 3 Top-1 |
|---------|------------|--------------|------------|--------------|
| Oracle | 1.000 | 100.0% | 1.000 | 100.0% |
| GPT-5.2 | 0.338 | 73.1% | **0.590** | **88.2%** |
| Claude Sonnet 4 | **0.383** | **76.0%** | 0.548 | 82.8% |
| GPT-4o | 0.323 | 74.5% | 0.567 | 80.6% |

### Ablation Study

| Configuration | Track 3 TWS | Note |
|------|------------|------|
| Full toolset | 0.590 | GPT-5.2 |
| w/o spatiotemporal tools | ~0.30 | −49.6 pp |
| w/o attribute analysis tools | ~0.45 | Degraded strategy selection |

### Key Findings
- **The benchmark is far from solved**: The best TWS is only 0.383 (Track 2) and 0.590 (Track 3), compared to an oracle of 1.0.
- **Removing tools causes a large performance drop** (49.6 percentage points), demonstrating that domain-specific tools are critical.
- **Spatial reasoning is the largest bottleneck**: Track 2 TWS is substantially lower than Track 3, as spatial disambiguation requires more turns and heavier reliance on strategic planning.

## Highlights & Insights
- **Redefining person search as interactive reasoning** is a perspective innovation: it shifts from passive visual matching to active dialogue reasoning, more closely reflecting real-world interactions between humans and surveillance systems.
- **The dual role of the STTG** is an elegant design: it serves simultaneously as the structural backbone for dataset construction (ensuring tasks have unambiguous ground truth) and as the agent's reasoning tool (providing computable spatiotemporal constraints).
- **The information boundary design** adds strategic depth: the agent does not know what the witness can answer and must intelligently explore under a limited budget.

## Limitations & Future Work
- The witness simulator is deterministic (fixed templates), lacking the noise and ambiguity of real human responses.
- Only 3 observable attributes are used (gender, upper-body color, lower-body color); in practice, witnesses may provide richer descriptions.
- The scale of 16 cameras is small; scalability to large camera networks has not been validated.
- Future work could incorporate visual understanding, enabling the agent to extract information directly from camera feeds.

## Related Work & Insights
- **vs. Traditional Re-ID**: Re-ID retrieves matches given a query image; ARGOS actively narrows candidates through dialogue and reasoning, representing a fundamentally different mode of information acquisition.
- **vs. GT-Bench / VS-Bench**: Those are multi-agent game-theoretic benchmarks; ARGOS is a single-agent reasoning benchmark in a structured environment, emphasizing tool use and spatiotemporal constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to define multi-camera person search as an interactive reasoning problem; STTG design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four LLMs and three tracks, though comparisons with traditional Re-ID methods are absent.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; track design logic is coherent.
- Value: ⭐⭐⭐⭐ Opens a new paradigm for person search; the benchmark has long-term value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_audiovisual_entity.md)
- [\[CVPR 2026\] HAVEN: Hierarchical Long Video Understanding with Audiovisual Entity Cohesion and Agentic Search](haven_hierarchical_long_video_understanding_with_audiovisual_entity_cohesion.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](../../ACL2026/llm_agent/when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)

<!-- RELATED:END -->
