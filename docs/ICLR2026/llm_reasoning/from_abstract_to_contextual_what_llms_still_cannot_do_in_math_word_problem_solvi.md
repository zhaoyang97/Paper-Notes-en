---
title: >-
  [Paper Note] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics
description: >-
  [ICLR 2026][LLM Reasoning][Mathematical Reasoning] This paper introduces ContextMATH, a benchmark that transforms abstract AIME/MATH-500 problems into two variants — Scenario Grounding (SG) and Complexity Scaling (CS) —…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Mathematical Reasoning"
  - "Contextual Reasoning"
  - "Problem Formulation"
  - "Benchmark"
  - "LLM Evaluation"
  - "AIME"
date: 2026-05-08
content_hash: f4163e4390dd22e5
---

# From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics

**Conference**: ICLR 2026
**arXiv**: [2601.23048](https://arxiv.org/abs/2601.23048)
**Code**: Not released
**Area**: LLM Reasoning
**Keywords**: Mathematical Reasoning, Contextual Reasoning, Problem Formulation, Benchmark, LLM Evaluation, AIME

## TL;DR

This paper introduces ContextMATH, a benchmark that transforms abstract AIME/MATH-500 problems into two variants — Scenario Grounding (SG) and Complexity Scaling (CS) — and reveals that even top-tier models such as GPT-5 and DeepSeek-R1 suffer accuracy drops of 13–34% on contextual mathematical reasoning, with errors attributable primarily to problem formulation rather than computational reasoning.

## Background & Motivation

LLMs have achieved near-perfect scores on mathematical benchmarks such as AIME and MATH-500, even reaching IMO gold-medal level. However, these successes are confined to **well-structured abstract problems** — those that present equations and conditions directly.

Real-world mathematical applications (financial analysis, scientific research, engineering design) rarely present ready-made equations; they typically require **extracting the mathematical core from concrete narrative scenarios before solving**. The authors term this capability **Contextual Mathematical Reasoning**.

Existing benchmarks focus almost exclusively on abstract problems (GSM8K, MATH, AIME), and even those containing simple narratives (e.g., "Jack had 8 pens...") remain superficial. This leaves a critical open question: **Can the strong performance of LLMs on abstract benchmarks transfer to contextualized, modeling-required mathematical problems?**

Collecting real-world mathematical problems is costly and difficult to scale. The authors therefore adopt a **controlled transformation strategy** — systematically converting each problem from existing benchmarks (ensuring correctness) into contextual variants.

## Method

### Overall Architecture

ContextMATH is built upon AIME 2024, AIME 2025, and MATH-500 (retaining only problems with difficulty $\geq 3$), transforming each original problem into two variants:

1. **Scenario Grounding (SG)**: Embeds the abstract mathematical structure into a concrete narrative without increasing reasoning complexity.
2. **Complexity Scaling (CS)**: Conceals explicit conditions within sub-problems, requiring additional reasoning steps to recover the original conditions.

### Key Designs

**SG Construction**: A multi-step prompting pipeline guides an LLM (o1-mini) to map all abstract mathematical elements to real-world entities (e.g., "variable $x$" → "initial number of oil barrels") and define interaction rules among them. The mathematical core is preserved; only narrative context is added.

**CS Construction**: Explicit conditions are encoded as outputs of simple, self-contained sub-problems. Strategies include:
- Encoding numerical values as solutions to number-theoretic or combinatorial problems (e.g., "25 indicator lights" → "the number of unique pairings of indicator lights is exactly 300")
- Replacing explicit functions/constants with variables that must be determined from data points
- Rephrasing geometric relationships as physical or structural descriptions

**Quality Control**: Three expert annotators (holding advanced degrees in computer science with competitive mathematics backgrounds) independently reviewed each problem:
- Assessed narrative plausibility and clarity
- Independently modeled the abstract mathematical problem from the scenario to verify equivalence
- Tested problems on Gemini and GPT-5
- Resolved disagreements through discussion led by the annotator with the strongest mathematical background

SG and CS variants average 133 and 176 words respectively, well within the processing capacity of current LLMs.

### Formulation Analysis Framework

Beyond accuracy, three metrics are defined to assess problem formulation capability:

**Formulation Accuracy**: The proportion of cases in which the model correctly translates the scenario into a mathematical formulation.

**Formulation Necessity**:

$$P(F=\text{True} \mid R=\text{True})$$

The degree to which correct reasoning depends on correct formulation.

**Formulation Sufficiency**:

$$P(R=\text{True} \mid F=\text{True})$$

The degree to which correct formulation leads to correct reasoning.

### Loss & Training

Training experiments are conducted on the Qwen3-Base series under three settings:
- $\text{SFT}_{\text{Ori}}$: Original data only (50k)
- $\text{SFT}_{\text{Syn}}$: Synthetic scenario data only (50k)
- $\text{SFT}_{\text{Mix}}$: Mixed (100k)

A dedicated formulation model training approach is also explored.

## Key Experimental Results

### Main Results

**Top closed-source models on AIME (single-run accuracy %)**:

| Model | AIME24 Ori | AIME24 SG | AIME24 CS | AIME25 Ori | AIME25 SG | AIME25 CS |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| DeepSeek-R1 | 93.3 | 70.0 (-25%) | 66.7 (-29%) | 86.7 | 73.3 (-15%) | 53.3 (-38%) |
| GPT-5 | 90.0 | 83.3 (-7%) | 80.0 (-11%) | 90.0 | 80.0 (-11%) | 66.7 (-26%) |
| Gemini 2.5 Pro | 83.3 | 73.3 (-12%) | 76.7 (-8%) | 83.3 | 56.7 (-32%) | 50.0 (-40%) |
| o3 | 83.3 | 70.0 (-16%) | 66.7 (-20%) | 76.7 | 70.0 (-9%) | 60.0 (-22%) |
| QwQ-plus | 86.7 | 56.7 (-35%) | 46.7 (-46%) | 73.3 | 53.3 (-27%) | 43.3 (-41%) |

**Open-source models (16-sample average accuracy %)**:

| Model | AIME24 Ori | AIME24 SG | AIME24 CS | AIME25 SG | AIME25 CS |
|-------|-----------|-----------|-----------|-----------|-----------|
| Qwen3-32B | 81.2 | 67.9 (-16%) | 57.1 (-30%) | 54.4 (-22%) | 45.0 (-36%) |
| Qwen3-8B | 73.8 | 61.5 (-16%) | 42.9 (-42%) | 48.3 (-25%) | 35.8 (-45%) |
| Qwen3-4B | 70.4 | 52.5 (-25%) | 34.6 (-51%) | 39.6 (-38%) | 33.8 (-47%) |
| AReaL-boba-2-32B | 81.5 | 65.4 (-20%) | 58.3 (-29%) | 55.0 (-29%) | 43.8 (-43%) |

On average, open-source models decline by 13% on SG and 34% on CS; closed-source models decline by 13% and 20%, respectively.

### Ablation Study

**Formulation capability analysis (selected models)**:

| Model | Formulation Acc. Avg | Necessity Avg | Sufficiency Avg |
|-------|---------------------|--------------|----------------|
| Qwen3-0.6B | 42.8 | 56.1 | 13.5 |
| Qwen3-4B | 61.6 | 79.2 | 61.3 |
| Qwen3-8B | 73.8 | 83.8 | 60.7 |
| Qwen3-32B | 75.0 | 81.9 | 64.9 |
| GPT-5 | 81.4 | 85.6 | 82.7 |

**Training experiments (Qwen3-14B-Base, average accuracy %)**:

| Setting | Average |
|---------|---------|
| Base | 29.4 |
| + SFT_Ori | 55.5 (+26.1%) |
| + SFT_Syn | 60.4 (+31.0%) |
| + SFT_Mix | **61.3** (+31.9%) |

**Failure of dedicated formulation model training**:

| Reasoning Model | No Formulation | Untuned Formulation 8B | Tuned Formulation 8B |
|----------------|---------------|----------------------|---------------------|
| Qwen3-8B | 53.9 | 48.9 | 20.8 |
| Qwen3-14B | 57.7 | 51.8 | 21.8 |

Performance collapses after training the dedicated formulation model, indicating that formulation capability cannot be effectively learned from scenario–original paired data.

### Key Findings

1. **Contextual complexity is a universal bottleneck**: Even GPT-5 suffers a 26% drop on AIME25-CS.
2. **Scale mitigates but does not resolve the problem**: 1.5B models drop 77% vs. 29% for 32B on CS, yet the gap remains substantial.
3. **Error analysis: formulation errors account for ~80%**, far exceeding computational, logical, and other error types.
4. **Formulation is a necessary condition**: Necessity consistently exceeds accuracy (Qwen3-8B: 83.8% vs. 73.8%).
5. **Formulation is not sufficient**: Sufficiency lags behind necessity; even GPT-5 achieves only 82.7%.
6. **Subsequent RL specialization may be harmful**: Further SFT/RL improves scores on original problems but amplifies the contextual performance gap.
7. **Training on scenario data helps but is insufficient**: $\text{SFT}_{\text{Mix}}$ performs best, yet substantial unresolved gaps remain.

## Highlights & Insights

1. **Benchmark design is conceptually strong**: The SG and CS dimensions form a progressive probe that disentangles "contextual understanding" from "condition recovery."
2. **The three-tier quantitative framework** (accuracy–necessity–sufficiency) rigorously characterizes the dual bottleneck of formulation and reasoning.
3. **A counterintuitive finding is revealed**: Specialized post-training via RL may overfit to canonical formats, thereby degrading contextual reasoning.
4. **Negative results are equally valuable**: The failure of dedicated formulation model training demonstrates that formulation capability cannot be straightforwardly learned from paired data.
5. **Evaluation is comprehensive in scale**: 61 models (46 open-source + 15 closed-source), including GPT-5.

## Limitations & Future Work

1. **Limited benchmark scale**: Built upon AIME (30 problems per year) and a subset of MATH-500, resulting in relatively small data volume.
2. **Construction relies on LLM + human review**: Difficult to scale.
3. **CS variants not constructed for MATH-500**: Some simpler problems are not amenable to further transformation.
4. **Closed-source models evaluated with single runs**: API constraints preclude multiple sampling.
5. The framework could be extended to contextual reasoning evaluation in other domains (physics, economics).
6. Curriculum learning strategies that expose models to both abstract and contextual variants during training warrant further exploration.

## Related Work & Insights

- **GSM8K/MATH/AIME**: ContextMATH directly builds contextual variants upon these benchmarks.
- **Math-Perturb** (Huang et al., 2025): Tests generalization by perturbing surface-level parameters; ContextMATH operates at a deeper level by altering the mode of presentation.
- **SWE-bench/WebArena**: Real-world scenario evaluations in other domains; ContextMATH represents an analogous effort in mathematics.
- Key insight: Abstract capability $\neq$ applied capability — this gap is particularly pronounced in the mathematical domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual SG/CS design and formulation analysis framework are original.
- **Technical Depth**: ⭐⭐⭐⭐ — The necessity/sufficiency analysis framework is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 61-model evaluation combined with training experiments and formulation analysis; exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with concise insights.
- **Value**: ⭐⭐⭐⭐ — Provides direct guidance for evaluating and training LLM mathematical capabilities.
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ (4.5/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs](geogrambench_benchmarking_the_geometric_program_reasoning_in_modern_llms.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)
- [\[ICLR 2026\] Harder Is Better: Boosting Mathematical Reasoning via Difficulty-Aware GRPO and Multi-Aspect Question Reformulation](harder_is_better_boosting_mathematical_reasoning_via_difficulty-aware_grpo_and_m.md)
- [\[ICLR 2026\] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)
- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)

</div>

<!-- RELATED:END -->
