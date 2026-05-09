---
title: >-
  [Paper Note] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
description: >-
  [NeurIPS 2025][LLM Reasoning][Temporal Reasoning] This paper introduces TimE, a multi-level temporal reasoning benchmark comprising 38,522 QA pairs across three real-world scenarios — knowledge-intensive (Wiki), dynamic events (News), and multi-turn dialogue (Dial) — with 11 fine-grained subtasks for systematic evaluation of LLMs' temporal reasoning capabilities. A manually annotated subset, TimE-Lite, is also released.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - Temporal Reasoning
  - Benchmark Evaluation
  - Large Language Models
  - Multi-level Tasks
  - Real-World Scenarios
date: 2026-05-08
content_hash: eed07b09fbe9332a
---

# TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios

**Conference**: NeurIPS 2025
**arXiv**: [2505.12891](https://arxiv.org/abs/2505.12891)
**Code**: [GitHub](https://github.com/ShaohangWei/TimE)
**Area**: LLM Reasoning
**Keywords**: Temporal Reasoning, Benchmark Evaluation, Large Language Models, Multi-level Tasks, Real-World Scenarios

## TL;DR
This paper introduces TimE, a multi-level temporal reasoning benchmark comprising 38,522 QA pairs across three real-world scenarios — knowledge-intensive (Wiki), dynamic events (News), and multi-turn dialogue (Dial) — with 11 fine-grained subtasks for systematic evaluation of LLMs' temporal reasoning capabilities. A manually annotated subset, TimE-Lite, is also released.

## Background & Motivation

1. **Background**: LLMs have achieved remarkable progress in mathematical and code reasoning, yet temporal reasoning remains a persistent challenge.

2. **Limitations of Prior Work**: Existing temporal reasoning benchmarks (TimeBench, TRAM) focus primarily on simplified scenarios, overlooking three key real-world challenges: dense temporal information, rapidly evolving event dynamics, and complex temporal dependencies in social interactions.

3. **Key Challenge**: Temporal reasoning is inherently hierarchical (basic understanding → expression-level reasoning → complex relational reasoning), yet existing benchmarks lack this stratified evaluation design.

4. **Goal**: To construct a comprehensive temporal reasoning benchmark covering diverse real-world scenarios and multi-level task structures.

5. **Key Insight**: Three data sources (Wikidata, news articles, and ultra-long dialogues) are employed to simulate the process by which humans leverage temporal concepts to understand the world.

6. **Core Idea**: A three-level progressive framework — Level 1: basic temporal understanding and retrieval; Level 2: temporal expression reasoning; Level 3: complex temporal relational reasoning.

## Method

### Overall Architecture
Three sub-datasets address three distinct real-world challenges: TimE-Wiki (knowledge-intensive, 13,848 QA pairs), TimE-News (dynamic events, 19,958 QA pairs), and TimE-Dial (multi-turn dialogue, 4,716 QA pairs). Data construction employs rule-based templates combined with DeepSeek-V3/R1 for QA synthesis, and the STARC framework for distractor generation. TimE-News uses BM25, Vector, and Hybrid RAG retrievers to process long-form news documents.

### Key Designs

1. **Three-Level Task Taxonomy**:
   - Level 1: Extract, Localization, Computation, DurationCompare, OrderCompare (5 subtasks)
   - Level 2: Explicit Reasoning, Order Reasoning, Relative Reasoning (3 subtasks)
   - Level 3: Co-temporality, Timeline, Counterfactual Reasoning (3 subtasks)
   - **Design Motivation**: Mirrors the human cognitive process of capturing temporal concepts → reasoning over implicit expressions → understanding complex relational structures.

2. **Data Construction Pipeline**:
   - **Function**: Ensures data quality and diversity.
   - **Mechanism**: Temporal facts are collected → timelines are constructed → QA pairs are synthesized using rule-based templates and DeepSeek-V3/R1, with the STARC framework generating distractors.
   - **Design Motivation**: Different data sources require tailored construction strategies; the News subset employs RAG to handle ultra-long documents.

3. **TimE-Lite**:
   - **Function**: Provides a high-quality, human-verified subset.
   - **Mechanism**: 1,071 pairs are systematically sampled from TimE and annotated by 3 domain experts, achieving an inter-annotator agreement rate of 89.13%.
   - **Design Motivation**: Ensures evaluation reliability and enables efficient model assessment.

### Loss & Training
- **Evaluation Metrics**: F1/Exact Match (EM) for free-form QA; Macro F1 for multiple-choice questions.
- **Decoding Strategy**: Greedy search.
- **Evaluated Models**: 24 models, including both reasoning and non-reasoning models.

## Key Experimental Results

### Main Results

| Model | TimE-Wiki Level 3 | TimE-News Timeline | TimE-Dial Extract |
|---|---|---|---|
| o3-mini | ~52% Avg | <30% | ~40% |
| Qwen2.5-72B | ~50% Avg | ~27% | ~40% |
| DeepSeek-R1 | ~55% Avg | 33% | — |

### Key Findings
- **Knowledge-intensive scenarios**: Models perform poorly on implicit temporal expressions and cross-event relational tasks (o3-mini achieves only 52% on Order Reasoning).
- **Dynamic events**: No model exceeds 30% on the Timeline task (ordering 3 events).
- **Multi-turn dialogue**: Temporal retrieval and localization accuracy is approximately 40%, substantially lower than other sub-datasets.
- Reasoning models exhibit a clear advantage on computation-type tasks but show limited gains in temporal relational understanding.
- Test-time scaling (TTS) offers marginal benefit for temporal reasoning overall.
- TTS effects are inconsistent: R1-Distill-Qwen-14B improves by 24.44%/11.33%/12.0% on OrderCompare/DurationCompare/Counterfactual in TimE-Dial, yet degrades by 3.36%/8.16% on Extract/Localization in TimE-Wiki — suggesting that systematic context traversal strategies may induce over-thinking loops.
- Retriever selection significantly affects temporal reasoning: GPT-4o with the Hybrid retriever underperforms BM25/Vector by over 10% on Timeline, indicating that accurate temporal fact retrieval is critical for complex event reasoning.
- Basic temporal retrieval abilities (Extract/Localization) correlate with nearly all higher-level temporal reasoning tasks at coefficients exceeding 0.5; cluster analysis confirms retrieval as a foundational prerequisite for reasoning.

## Highlights & Insights
- The first benchmark to systematically cover all three real-world temporal reasoning challenges.
- Reveals that even the strongest reasoning models exhibit significant deficiencies in temporal reasoning.
- Memory-anchored temporal expressions in dialogue (e.g., "last Saturday") pose unique challenges for models.
- Supports a public leaderboard for continuous community evaluation.
- TimE-Lite provides 1,071 human-verified QA pairs annotated by 3 experts with an inter-annotator agreement of 89.13%, ensuring evaluation reliability.
- Broad evaluation across 24 models — covering both reasoning models (o3-mini, DeepSeek-R1) and non-reasoning models — using greedy search decoding.

## Limitations & Future Work
- QA synthesis relies on LLMs, which may introduce quality biases.
- The News subset depends on RAG; retrieval quality directly impacts evaluation validity.
- Multimodal temporal reasoning scenarios are not covered.
- Task difficulty calibration may need to be dynamically adjusted according to model capability.
- In TimE-Dial, small-scale models (8B) achieve only ~30–40% on Extract and Localization tasks, far below the ~60–70% observed on TimE-Wiki, highlighting that memory-anchored temporal expressions in multi-turn dialogue (e.g., "last time we talked," "two days later") constitute a distinct challenge.
- Cluster analysis naturally partitions the 11 subtasks into three groups — basic retrieval (Extract/Localization), reasoning (Reasoning/Compare), and complex relational (Timeline/Counterfactual) — with progressively increasing difficulty across groups.

## Related Work & Insights
- **vs. TimeBench**: TimeBench aggregates 10 existing datasets with relatively simple tasks; TimE is constructed in a unified manner with substantially higher difficulty.
- **vs. TRAM**: TRAM evaluates only event sequence understanding; TimE covers 11 subtasks more comprehensively.
- **vs. TCELongBench**: Focuses on partial temporal aspects of the news domain only; TimE spans three distinct scenarios.
- **vs. TReMu**: TReMu addresses temporal localization in dialogue only; TimE-Dial covers a broader range of task types.
- **vs. RealTimeQA/FreshLLM**: These works focus on knowledge currency rather than temporal reasoning per se; TimE targets reasoning capability evaluation.
- **Data Scale**: With 38,522 QA pairs in total, TimE is the largest dedicated temporal reasoning benchmark to date. Wikidata provides dense temporal facts, news articles cover dynamic event streams, and ultra-long dialogues capture temporal dependencies in social interactions.

## Rating

### Implementation Details
38,522 QA pairs spanning 3 scenarios (Wiki/News/Dial) × 3 levels × 11 subtasks.
TimE-Lite contains 1,071 human-verified QA pairs with an inter-annotator agreement of 89.13% among 3 experts.
24 models evaluated using greedy search decoding, with F1/EM (free-form QA) and Macro F1 (multiple-choice).
- **Novelty**: ⭐⭐⭐⭐ Novel multi-level, multi-scenario benchmark design for temporal reasoning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive evaluation across 24 models.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical gap in temporal reasoning evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[NeurIPS 2025\] ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning in LLMs](reasonfluxprm_trajectoryaware_prms_for_long_chainofthought_r.md)
- [\[NeurIPS 2025\] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)

</div>

<!-- RELATED:END -->
