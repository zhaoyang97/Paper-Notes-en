---
title: >-
  [Paper Note] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks
description: >-
  [ACL 2026][LLM Evaluation][Travel Planning Benchmark] Ours proposes TravelBench, the first travel planning benchmark integrating real-world user queries, implicit user preferences, multi-turn interactions…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Travel Planning Benchmark"
  - "Tool-Using"
  - "Multi-turn Dialogue"
  - "Implicit Preferences"
  - "Unsolvable Tasks"
date: 2026-05-08
content_hash: 1aa9f2d3e014b78a
---

# Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks

**Conference**: ACL 2026  
**arXiv**: [2512.22673](https://arxiv.org/abs/2512.22673)  
**Code**: [GitHub](https://github.com/small-xiangcheng/TravelBench)  
**Area**: Recommender Systems  
**Keywords**: Travel Planning Benchmark, Tool-Using, Multi-turn Dialogue, Implicit Preferences, Unsolvable Tasks

## TL;DR

Ours proposes TravelBench, the first travel planning benchmark integrating real-world user queries, implicit user preferences, multi-turn interactions, unsolvable task identification, and 10 real-world tools. Through a sandbox environment, it achieves reproducible evaluation and reveals unbalanced performance of state-of-the-art models across different capability dimensions.

## Background & Motivation

**Background**: Travel planning is an ideal testing scenario for evaluating LLM Agent capabilities in multi-step reasoning, tool usage, and user interaction. Existing benchmarks (TravelPlanner, ChinaTravel, etc.) have made progress but still possess critical deficiencies.

**Limitations of Prior Work**: (1) User preferences and constraints are typically predefined, injected via instructions, or gradually revealed by a simulator, failing to dynamically elicit implicit user preferences; (2) Most benchmarks only cover itinerary planning, ignoring diverse real-world travel needs such as POI exploration, route planning, and option comparison; (3) They either do not support tool usage or rely on synthetic queries, failing to reflect real-world data distributions; (4) Lack of evaluation for unsolvable tasks—Agents must identify capability boundaries in actual scenarios.

**Key Challenge**: The performance of existing benchmarks does not truly reflect Agent performance in actual travel planning because they significantly differ from real needs in task scope, user interaction modes, and evaluation coverage.

**Goal**: Construct a "truly real-world oriented" travel planning benchmark to comprehensively evaluate three core capabilities of Agents: independent problem solving, eliciting implicit preferences through interaction, and identifying capability boundaries.

**Key Insight**: Collect queries and preferences from real-world user logs of Alibaba's Amap, integrate 10 real-world travel tools, and construct approximately 200,000 cached tool-calling trajectories.

**Core Idea**: Expand travel planning benchmarks from "itinerary planning" to cover multi-domain tasks like POI exploration, route planning, and option comparison, while introducing two entirely new dimensions: multi-turn elicitation of implicit preferences and identification of unsolvable tasks.

## Method

### Overall Architecture

TravelBench consists of three subsets: 500 single-turn queries (solved independently by Agent using tools), 500 multi-turn queries (requiring interaction to elicit implicit preferences), and 100 unsolvable queries (requiring identification of missing tools or information). The user simulator is driven by LLM + user personas, and the Agent executes reasoning in a sandbox environment integrated with 10 real-world tools. Evaluation adopts a three-tier protocol: LLM-as-judge + tool-calling error penalty + meta-review calibration.

### Key Designs

1. **Implicit Preferences and Multi-turn Interaction**:

    - **Function**: Evaluate the Agent's ability to proactively elicit preferences not explicitly stated by the user.
    - **Mechanism**: Anonymized preference information is extracted from real user data to construct user personas (gender, family structure, lifestyle, etc.). Personas are held only by the user simulator; the Agent must obtain them through multi-turn questioning. A query is determined as single-turn or multi-turn by 3 models across 2 trials.
    - **Design Motivation**: Preferences in existing benchmarks are either predefined or gradually revealed by simulators, not supporting proactive exploration and elicitation by Agents.

2. **Unsolvable Task Subset**:

    - **Function**: Evaluate the Agent's capability to identify its own capability boundaries.
    - **Mechanism**: Three models (GPT-5.1, Qwen3-235B, Qwen-Plus) label whether each query is solvable. Queries consistently judged as unsolvable by all three models form the unsolvable subset, categorized by three reasons: lack of tool support, lack of necessary context, or no clear executable intent. The Agent must output a special tag `[Unsolved]` for identification.
    - **Design Motivation**: In real scenarios, an Agent must know when to say "I cannot do this" rather than forcing a wrong answer.

3. **Reproducible Sandbox and Tool Caching**:

    - **Function**: Ensure stability and reproducibility of the evaluation.
    - **Mechanism**: Multiple models run on real APIs to cache approximately 200,000 tool-calling trajectories. During evaluation, matching from the cache is prioritized; if a cache miss occurs, consistent tool responses are simulated via embedding retrieval + ICL. Strict parameter validation is performed to record tool-calling error rates.
    - **Design Motivation**: Directly calling external APIs yields unstable results, affecting reproducibility and fair comparison.

### Evaluation Protocol

Three-tier evaluation: (1) Rule-based accuracy calculation for the unsolvable subset; (2) LLM-as-judge scores 1-5 for single-turn (3 dimensions) and multi-turn (4 dimensions, adding user_interaction); (3) Meta-review calibration of over-evaluated scores + tool-calling error rate penalty. The final score is the average of the three subsets.

## Key Experimental Results

### Main Results

| Model | Multi-turn (After Penalty) | Single-turn (After Penalty) | Unsolvable | Total Score |
|------|------------|------------|--------|------|
| Qwen-Plus | 62.56 | 82.64 | 83.67 | **76.29** |
| GPT-5.1 | 71.31 | 73.81 | 80.00 | 75.04 |
| Kimi-K2-Th | 71.83 | 77.31 | 73.67 | 74.27 |
| DeepSeek-V3.2 | 82.80 | 83.29 | 51.33 | 72.47 |
| Qwen3-235B-It | 61.78 | 70.74 | 80.00 | 70.84 |
| DeepSeek-R1 | 35.67 | 76.93 | 83.67 | 65.42 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Score Stability | std ≈ 0.01 | Minimal standard deviation across 3 repeated runs. |
| Offline vs. Online Scoring | < 1 point difference | Cached sandbox and real API evaluations are highly consistent. |
| Human Verification | 97% label agreement | Unsolvable and single/multi-turn labels align closely with human judgment. |
| Judge MAE | 0.52 vs. Human-Human 0.48 | Close to inter-human disagreement levels. |

### Key Findings
- Even the strongest models score only around 76, indicating that real-world travel planning remains challenging.
- Capability imbalance is prevalent: DeepSeek-V3.2 is strongest in single/multi-turn but poor at identifying unsolvability (51.33); Kimi-K2-0925 is strongest at unsolvability (94) but poor at task completion.
- Tool-calling error rates for reasoning models are generally lower than for instruction-following models, yet they tend to underperform on unsolvable tasks—strong reasoning makes models more "reluctant to give up."
- Tool-calling error rates for multi-turn tasks are higher than for single-turn, suggesting that multi-turn interaction increases the difficulty of tool usage.
- Tool penalties impact MiniMax-M2 the most (78→67), effectively distinguishing trajectories that "appear correct but have tool usage issues."

## Highlights & Insights
- **Truly Real-World Oriented**: Starting from real Amap logs, covering 32 provincial regions and 243 cities, the task distribution reflects real user needs.
- **Unified Evaluation of Three Core Capabilities**: Independent problem-solving, interactive preference elicitation, and boundary recognition—forming a complete capability profile for Agent practicalization.
- **Innovation in Tool Penalty Mechanism**: Evaluation looks not only at task completion but also at whether the tool-using process is reliable.
- **Importance of Capability Imbalance Findings**: Points out the tension in current models between "strong reasoning vs. knowing when to give up."

## Limitations & Future Work
- **Covers only China Travel Scenarios**: Limited geographical and cultural scope; international travel needs are not covered.
- **User Simulator Limitations**: User interactions simulated by LLMs may deviate from real user behavior.
- **Evaluation Relies on LLM Judge**: Although human verification shows high consistency, blind spots may still exist.
- Future Directions: Expand to international travel, introduce more complex dynamic constraint changes, and study how Agents can better balance capability and boundary recognition.

## Related Work & Insights
- **vs. TravelPlanner**: The first travel planning benchmark but largely solved by solver methods; tasks are too simple.
- **vs. ChinaTravel**: Introduces real queries and stricter constraints but does not support multi-turn interaction and unsolvable tasks.
- **vs. COMPASS**: Focuses on soft preference optimization but does not use real queries and tools.

## Rating
- Novelty: ⭐⭐⭐⭐ Elicitation of implicit preferences and identification of unsolvable tasks are important new dimensions; task coverage far exceeds prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 12+ models, includes stability analysis, human verification, and online/offline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed evaluation protocol descriptions.
- Value: ⭐⭐⭐⭐ Provides the most comprehensive benchmark for travel planning Agent evaluation; findings on capability imbalance guide Agent design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)

</div>

<!-- RELATED:END -->
