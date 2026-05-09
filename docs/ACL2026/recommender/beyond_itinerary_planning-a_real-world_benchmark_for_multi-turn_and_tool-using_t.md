---
title: >-
  [Paper Note] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks
description: >-
  [ACL 2026][Recommender Systems][travel planning benchmark] This paper proposes TravelBench, the first travel planning benchmark that integrates real user queries, implicit user preferences, multi-turn interaction, unsolvable task recognition, and 10 real-world tools. It enables reproducible evaluation through a sandbox environment and reveals that state-of-the-art models exhibit uneven performance across different capability dimensions.
tags:
  - ACL 2026
  - Recommender Systems
  - travel planning benchmark
  - tool use
  - multi-turn dialogue
  - implicit preferences
  - unsolvable tasks
date: 2026-05-08
content_hash: 3a8a7f8b5d71c3d9
---

# Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks

**Conference**: ACL 2026
**arXiv**: [2512.22673](https://arxiv.org/abs/2512.22673)
**Code**: [GitHub](https://github.com/small-xiangcheng/TravelBench)
**Area**: Recommender Systems
**Keywords**: travel planning benchmark, tool use, multi-turn dialogue, implicit preferences, unsolvable tasks

## TL;DR

This paper proposes TravelBench, the first travel planning benchmark that integrates real user queries, implicit user preferences, multi-turn interaction, unsolvable task recognition, and 10 real-world tools. It enables reproducible evaluation through a sandbox environment and reveals that state-of-the-art models exhibit uneven performance across different capability dimensions.

## Background & Motivation

**State of the Field**: Travel planning serves as an ideal testbed for evaluating LLM agents' multi-step reasoning, tool use, and user interaction capabilities. Existing benchmarks such as TravelPlanner and ChinaTravel have made progress but still suffer from critical limitations.

**Limitations of Prior Work**: (1) User preferences and constraints are typically predefined and injected into instructions or incrementally revealed by a simulator, preventing dynamic elicitation of users' implicit preferences. (2) Most benchmarks cover only itinerary planning, neglecting diverse real-world travel needs such as POI exploration, route planning, and plan comparison. (3) Benchmarks either do not support tool use or rely on synthetic queries that fail to reflect real data distributions. (4) Evaluation of unsolvable tasks is absent—yet agents in practice must recognize the boundaries of their capabilities.

**Root Cause**: The performance reported on existing benchmarks does not faithfully reflect agent behavior in real-world travel planning, as these benchmarks diverge substantially from actual user needs in terms of task scope, interaction modality, and evaluation coverage.

**Paper Goals**: To construct a genuinely real-world travel planning benchmark that comprehensively evaluates three core agent capabilities: solving tasks autonomously, eliciting implicit preferences through interaction, and recognizing capability boundaries.

**Starting Point**: Queries and user preferences are collected from real user logs of Alibaba's Amap platform; 10 real travel tools are integrated; and approximately 200,000 cached tool-call trajectories are constructed.

**Core Idea**: To expand travel planning benchmarks from "itinerary planning" to a broader scope covering POI exploration, route planning, and plan comparison, while introducing two entirely new evaluation dimensions: multi-turn elicitation of implicit preferences and unsolvable task recognition.

## Method

### Overall Architecture

TravelBench comprises three subsets: 500 single-turn queries (solved autonomously by the agent using tools), 500 multi-turn queries (requiring interaction with the user to elicit implicit preferences), and 100 unsolvable queries (requiring recognition of missing tools or information). A user simulator driven by an LLM and user profile executes multi-turn interactions. The agent operates within a sandbox environment equipped with 10 real tools. Evaluation follows a three-tier protocol: LLM-as-judge, tool-call error penalty, and meta-review calibration.

### Key Designs

1. **Implicit Preferences and Multi-Turn Interaction**:

    - Function: Evaluates the agent's ability to proactively elicit preferences not explicitly expressed by the user.
    - Mechanism: Preference information is extracted from real user data after anonymization to construct user profiles (gender, family structure, lifestyle, etc.). Profiles are held exclusively by the user simulator; the agent must obtain them through multi-turn questioning. Three models × two trials are used to determine whether each query is single-turn or multi-turn.
    - Design Motivation: In prior benchmarks, preferences are either predefined and injected or incrementally revealed by the simulator, providing no support for the agent to actively explore and elicit preferences.

2. **Unsolvable Task Subset**:

    - Function: Evaluates the agent's ability to recognize the boundaries of its own capabilities.
    - Mechanism: Three models (GPT-5.1, Qwen3-235B, Qwen-Plus) annotate each query as solvable or unsolvable. Queries unanimously deemed unsolvable constitute the unsolvable subset, categorized into three types: missing tool support, missing necessary context, and no clear actionable intent. The agent outputs a special label `[Unsolved]` to signal recognition.
    - Design Motivation: In real-world scenarios, agents must know when to say "I cannot do this" rather than producing incorrect answers.

3. **Reproducible Sandbox and Tool Caching**:

    - Function: Ensures evaluation stability and reproducibility.
    - Mechanism: Multiple models are run against real APIs to cache approximately 200,000 tool-call trajectories. During evaluation, cache matches are prioritized; on cache misses, consistent tool responses are simulated via embedding retrieval and in-context learning. Strict parameter validation is enforced and tool-call error rates are recorded.
    - Design Motivation: Direct calls to external APIs yield unstable results, undermining reproducibility and fair comparison.

### Evaluation Protocol

A three-tier evaluation is employed: (1) rule-based criteria compute accuracy for the unsolvable subset; (2) LLM-as-judge scores single-turn (3 dimensions) and multi-turn (4 dimensions, adding `user_interaction`) responses on a 1–5 scale; (3) meta-review calibrates over-inflated scores, and tool-call error rates are applied as a penalty. The final score is the average across the three subsets.

## Key Experimental Results

### Main Results

| Model | Multi-Turn (penalized) | Single-Turn (penalized) | Unsolvable | Total |
|-------|----------------------|------------------------|------------|-------|
| Qwen-Plus | 62.56 | 82.64 | 83.67 | **76.29** |
| GPT-5.1 | 71.31 | 73.81 | 80.00 | 75.04 |
| Kimi-K2-Th | 71.83 | 77.31 | 73.67 | 74.27 |
| DeepSeek-V3.2 | 82.80 | 83.29 | 51.33 | 72.47 |
| Qwen3-235B-It | 61.78 | 70.74 | 80.00 | 70.84 |
| DeepSeek-R1 | 35.67 | 76.93 | 83.67 | 65.42 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Score stability | std ≈ 0.01 | Extremely low standard deviation across 3 repeated runs |
| Offline vs. online scoring | < 1-point difference | Cached sandbox and real-API evaluation are highly consistent |
| Human validation | 97% label agreement | Unsolvable and single/multi-turn labels closely match human judgments |
| Judge MAE | 0.52 vs. 0.48 (inter-human) | Approaches the level of inter-human disagreement |

### Key Findings

- Even the strongest model achieves only ~76 points, indicating that real-world travel planning remains challenging.
- Capability imbalance is pervasive: DeepSeek-V3.2 leads on single- and multi-turn tasks but performs poorly on unsolvable recognition (51.33), while Kimi-K2-0925 achieves the highest unsolvable recognition (94) but poor task completion.
- Reasoning models generally exhibit lower tool-call error rates than instruction-following models, yet perform worse on unsolvable tasks—stronger reasoning makes models more reluctant to give up.
- Tool-call error rates are higher in multi-turn tasks than in single-turn tasks, suggesting that multi-turn interaction increases the difficulty of tool use.
- The tool penalty has the greatest impact on MiniMax-M2 (78→67), effectively distinguishing trajectories that appear correct but suffer from unreliable tool use.

## Highlights & Insights

- **Genuinely real-world grounding**: Derived from real Amap user logs, covering 32 provincial regions and 243 cities, with a task distribution that reflects authentic user needs.
- **Unified evaluation of three core capabilities**: Autonomous problem solving, interactive preference elicitation, and boundary recognition—constituting a complete capability profile for practical agents.
- **Innovation in tool penalty mechanism**: Evaluation considers not only whether a task is completed but also whether the tool-use process is reliable.
- **Significance of the capability imbalance finding**: Highlights the tension in current models between strong reasoning and knowing when to abstain.

## Limitations & Future Work

- **Coverage limited to Chinese travel scenarios**: Geographic and cultural scope is restricted; international travel needs are not addressed.
- **Limitations of the user simulator**: LLM-simulated user interactions may deviate from real user behavior.
- **Dependence on LLM judge**: Despite high human-validation agreement, blind spots may remain.
- Future directions include extending to international travel, introducing more complex dynamic constraint changes, and studying how agents can better balance capability and boundary recognition.

## Related Work & Insights

- **vs. TravelPlanner**: The first travel planning benchmark, but now largely solved by solver-based methods; tasks are too simple.
- **vs. ChinaTravel**: Introduces real queries and stricter constraints but does not support multi-turn interaction or unsolvable tasks.
- **vs. COMPASS**: Focuses on soft preference optimization but does not use real queries or tools.

## Rating

- Novelty: ⭐⭐⭐⭐ Implicit preference elicitation and unsolvable task recognition are important new dimensions; task coverage far exceeds prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 12+ models with stability analysis, human validation, and online/offline comparison.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough description of the evaluation protocol.
- Value: ⭐⭐⭐⭐ Provides the most comprehensive benchmark for travel planning agent evaluation; the capability imbalance findings offer meaningful guidance for agent design.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints](../../NeurIPS2025/recommender/wide-horizon_thinking_and_simulation-based_evaluation_for_real-world_llm_plannin.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[ACL 2026\] Content Fuzzing for Escaping Information Cocoons on Social Media](content_fuzzing_for_escaping_information_cocoons_on_digital_social_media.md)

<!-- RELATED:END -->
