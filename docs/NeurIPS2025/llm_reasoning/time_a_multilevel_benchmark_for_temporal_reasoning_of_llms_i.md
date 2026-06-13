---
title: >-
  [Paper Note] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios
description: >-
  [NeurIPS 2025][LLM Reasoning][Temporal Reasoning] This paper introduces TimE, a multi-level temporal reasoning benchmark comprising 38,522 QA pairs across three real-world scenarios — knowledge-intensive (Wiki)…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Temporal Reasoning"
  - "Multi-level Benchmark"
  - "Real-World Scenarios"
  - "Knowledge-Intensive"
  - "Long Dialogue"
date: 2026-05-08
content_hash: f63ee66c6aa2fb85
---

# TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.12891](https://arxiv.org/abs/2505.12891)  
**Code**: [GitHub](https://github.com/) / [HuggingFace Dataset](https://huggingface.co/)  
**Area**: LLM Reasoning / Temporal Reasoning
**Keywords**: Temporal Reasoning, Multi-level Benchmark, Real-World Scenarios, Knowledge-Intensive, Long Dialogue

## TL;DR

This paper introduces TimE, a multi-level temporal reasoning benchmark comprising 38,522 QA pairs across three real-world scenarios — knowledge-intensive (Wiki), dynamic news (News), and long dialogue (Dial) — and three progressively difficult levels with 11 sub-tasks. A comprehensive evaluation of 24 LLMs reveals that even the strongest reasoning models exhibit significant deficiencies on complex tasks such as timeline construction and counterfactual reasoning.

## Background & Motivation

**Background**: Temporal reasoning is a critical capability for LLMs to understand the real world. Existing benchmarks such as TimeBench and TRAM primarily focus on simplified scenarios (e.g., basic temporal commonsense, temporal relations within short texts), and their task designs are relatively simple, posing insufficient challenge to current LLMs.

**Limitations of Prior Work**: Real-world temporal reasoning presents three major challenges: (1) knowledge-intensive scenarios feature high-density temporal information with complex entity relationships; (2) news events evolve rapidly, with details changing over time; (3) multi-turn dialogues involve complex temporal dependencies spanning long contexts. Existing benchmarks fail to address these challenges. Furthermore, temporal reasoning is inherently a hierarchical capability framework — ranging from basic comprehension to complex relational inference — yet prior work typically addresses only a single dimension.

**Key Challenge**: How to design a temporal reasoning benchmark that simultaneously covers real-world complexity and enables systematic, hierarchical evaluation? Existing benchmarks are either too simple (e.g., basic tasks in TimeBench) or narrowly focused (e.g., TReMu addresses only temporal localization in dialogue), lacking a unified framework.

**Goal**: (1) The absence of a temporal reasoning benchmark spanning diverse real-world scenarios; (2) the lack of systematic multi-level evaluation from basic to complex reasoning; (3) the lack of high-quality human-annotated subsets for reliable assessment.

**Key Insight**: Starting from three real-world data sources — the Wikidata knowledge graph, online news, and multi-turn long dialogues — the paper designs a three-level progressive task taxonomy, complemented by the human-annotated TimE-Lite subset as a high-quality evaluation anchor.

**Core Idea**: Construct a large-scale temporal reasoning benchmark covering three progressive levels — temporal understanding → temporal expression reasoning → complex temporal relation reasoning — across three distinct real-world scenarios.

## Method

### Overall Architecture

TimE consists of three sub-datasets: (1) TimE-Wiki (13,848 QA) — constructed from the Wikidata temporal knowledge graph to evaluate temporal reasoning in knowledge-intensive scenarios; (2) TimE-News (19,958 QA) — based on temporally complex events (TCEs) in online news to assess dynamic event understanding; (3) TimE-Dial (4,716 QA) — derived from ultra-long multi-turn dialogues to evaluate temporal reasoning in interactive settings. All sub-datasets share the same three-level task taxonomy.

### Key Designs

1. **Three-Level Progressive Task Taxonomy**:

    - Function: Systematically evaluates temporal reasoning capability from basic to complex levels.
    - Mechanism: Level-1 (Basic Temporal Understanding & Retrieval) comprises 5 sub-tasks — Extract (temporal expression extraction), Localization (event-time mapping), Computation (duration calculation), DurationCompare (duration comparison), and OrderCompare (temporal sequence ordering). Level-2 (Temporal Expression Reasoning) comprises 3 sub-tasks — Explicit Reasoning (inference of non-explicitly mentioned times), Order Reasoning (ordinal temporal localization), and Relative Reasoning (relative temporal reference inference). Level-3 (Complex Temporal Relation Reasoning) comprises 3 sub-tasks — Co-temporality (simultaneity identification), Timeline (multi-event chronological ordering), and Counterfactual (counterfactual temporal reasoning).
    - Design Motivation: Simulates the cognitive process by which humans process temporal information — first capturing and understanding, then reasoning over implicit expressions, and finally resolving complex relations.

2. **Multi-Source Data Construction Pipeline**:

    - Function: Automatically generates high-quality QA pairs from real-world data sources.
    - Mechanism: For TimE-Wiki, SLING is used to parse Wikidata and extract temporal fact quadruples, constructing a multi-hop temporal knowledge graph, with DeepSeek-V3 generating natural language contexts. For TimE-News, relevant passages are retrieved via RAG from an existing temporally complex event dataset. For TimE-Dial, LLMs summarize event graphs from long dialogues and normalize temporal expressions. QA generation combines rule-based templates and LLMs (DeepSeek-V3/R1); for open-ended questions, distractor options are generated via the STARC framework.
    - Design Motivation: The hybrid rule-LLM approach ensures logical correctness and naturalness of QA pairs; distractor options improve evaluation discriminability.

3. **TimE-Lite Human-Annotated Subset**:

    - Function: Provides a high-quality evaluation anchor.
    - Mechanism: A random sample of 1,071 QA pairs is drawn from the full dataset and subjected to multi-round review and answer validation by three professional annotators, yielding 943 high-quality QA pairs. The agreement rate between automatically generated data and human annotations reaches 89.13%, validating the quality of the data pipeline.
    - Design Motivation: Automatically generated data may contain noise; the human-annotated subset provides a reliable evaluation standard.

## Key Experimental Results

### Main Results (TimE-Wiki, Representative Models)

| Model | Extract | Localiz. | Compute | OrderComp | ExplReas | Timeline | Counterfact |
|------|---------|---------|---------|-----------|----------|----------|-------------|
| Qwen2.5-72B-Inst | 81.70 | 83.84 | 41.37 | 84.22 | 70.13 | 4.08 | 50.68 |
| QwQ-32B | 74.99 | 67.75 | 49.59 | 93.53 | 60.61 | 25.38 | 53.13 |
| o3-mini | 96.67 | 80.83 | 49.17 | 93.33 | 82.24 | 33.33 | 52.07 |
| DeepSeek-R1 | 96.67 | 77.61 | 46.39 | 93.33 | 78.20 | 33.33 | 55.71 |

### Ablation Study (TimE-Dial)

| Model | Extract | Localiz. | DurComp | OrdComp | Timeline |
|------|---------|---------|---------|---------|----------|
| Qwen2.5-14B-Inst | 38.85 | 30.83 | 42.00 | 47.78 | 0.00 |
| R1-Distill-14B | 40.40 | 18.34 | 53.33 | 72.22 | 0.22 |
| o3-mini | 41.41 | 45.30 | 56.67 | 86.67 | 10.00 |

### Key Findings

- **Timeline is the primary bottleneck**: Even the strongest model, o3-mini, achieves only 33.33% on TimE-Wiki, while smaller models approach 0%. Ordering multiple events chronologically requires simultaneous information retrieval and global sorting.
- **Test-time scaling benefits logical reasoning but yields inconsistent gains on retrieval tasks**: R1-Distill improves OrderCompare by 24.44% over the corresponding non-reasoning model, but performance on Localization actually degrades (due to circular reasoning caused by overthinking).
- **Basic temporal retrieval ability is significantly correlated with higher-level reasoning tasks**: Cluster analysis shows that Extract and Localization have correlation coefficients > 0.5 with nearly all other tasks.
- **Retriever selection has a substantial impact on news scenarios**: Performance gaps of 10%+ are observed for the same model under different retrievers.
- **Temporal localization in long dialogue scenarios is extremely difficult**: Multi-turn dialogues exceeding 15k tokens combined with memory-based relative temporal expressions cause a sharp drop in localization accuracy.

## Highlights & Insights

- **Elegantly designed hierarchical evaluation framework**: The three-level task design, progressing from shallow to deep cognitive processing, combined with three data sources of distinct difficulty profiles, enables precise identification of model weaknesses in temporal reasoning. This "capability-level × scenario-level" evaluation matrix is transferable to the assessment of other complex reasoning abilities.
- **Timeline task exposes a fundamental deficiency**: Universally poor performance on multi-event ordering indicates that current LLMs lack the ability to construct global temporal structures, pointing to an important research direction.
- **The dual nature of test-time scaling**: Extended chain-of-thought in reasoning models benefits logical inference but may cause overthinking-induced errors on simple retrieval tasks — providing empirical evidence for the rational allocation of test-time compute.

## Limitations & Future Work

- QA pairs are primarily generated via LLMs and rules, which may introduce systematic biases (e.g., preference for certain types of temporal expressions).
- TimE-News employs RAG-based retrieval, introducing retrieval quality as a confounding variable that makes it difficult to assess temporal reasoning ability in isolation.
- Dialogue data sources are derived from synthetic long dialogues (LoCoMo, RealTalk), which may exhibit distribution shifts relative to authentic multi-turn conversations.
- Multilingual temporal reasoning ability is not evaluated.
- The design of the Counterfactual task is relatively basic (e.g., "event delayed by 3 years"); more complex counterfactual scenarios are not addressed.

## Related Work & Insights

- **vs TimeBench (Chu 2024)**: Aggregates 10 datasets but with relatively simple tasks; inconsistent evaluation contexts may introduce bias. TimE provides a unified framework with more challenging tasks.
- **vs TRAM (Wang 2024)**: Focuses on event sequence understanding but poses insufficient challenge to current models. TimE's Level-3 tasks are substantially more difficult.
- **vs TCELongBench (Zhang 2024)**: Restricted to temporal understanding in news scenarios. TimE spans three scenarios with a more systematic hierarchical structure.
- **vs TReMu**: Covers only temporal localization and long-range dependency in dialogue. TimE-Dial encompasses a more comprehensive set of temporal reasoning sub-tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The evaluation matrix combining three progressive levels with three distinct scenarios is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 24 models × 11 sub-tasks × 3 datasets.
- Writing Quality: ⭐⭐⭐⭐ Task definitions are clear and the data construction pipeline is well-documented.
- Value: ⭐⭐⭐⭐ Reveals fundamental deficiencies in LLM temporal reasoning and provides an important evaluation resource for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 6: Self-Evaluating LLMs - Step-Level Confidence Estimation for Multi-Step Tasks](value-guided_search_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ACL 2026\] Efficient Test-Time Scaling via Temporal Reasoning Aggregation](../../ACL2026/llm_reasoning/efficient_test-time_scaling_via_temporal_reasoning_aggregation.md)
- [\[ACL 2026\] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation](../../ACL2026/llm_reasoning/mtr-bench_a_comprehensive_benchmark_for_multi-turn_reasoning_evaluation.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](../../ICML2026/llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)

</div>

<!-- RELATED:END -->
