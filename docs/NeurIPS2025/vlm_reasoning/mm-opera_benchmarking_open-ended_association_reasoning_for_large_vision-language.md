---
title: >-
  [Paper Note] MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models
description: >-
  [NeurIPS 2025][VLM Reasoning][association reasoning] This paper proposes MM-OPERA, an open-ended association reasoning benchmark comprising 11,497 instances. It evaluates the association reasoning capabilities of LVLMs through two tasks — Remote-Item Association (RIA) and In-Context Association (ICA) — and introduces an LLM-as-a-Judge scoring strategy alongside a process reward evaluation method. The benchmark reveals that even the strongest current LVLMs remain significantly…
tags:
  - "NeurIPS 2025"
  - "VLM Reasoning"
  - "association reasoning"
  - "open-ended evaluation"
  - "LLM-as-a-Judge"
  - "process reward"
  - "divergent thinking"
  - "convergent thinking"
date: 2026-05-08
content_hash: 7e82d3510aa22821
---

# MM-OPERA: Benchmarking Open-ended Association Reasoning for Large Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.26937](https://arxiv.org/abs/2510.26937)  
**Code**: [https://github.com/MM-OPERA-Bench/MM-OPERA](https://github.com/MM-OPERA-Bench/MM-OPERA)  
**Area**: Multimodal VLM / Association Reasoning / Evaluation Benchmark
**Keywords**: association reasoning, open-ended evaluation, LLM-as-a-Judge, process reward, divergent thinking, convergent thinking

## TL;DR
This paper proposes MM-OPERA, an open-ended association reasoning benchmark comprising 11,497 instances. It evaluates the association reasoning capabilities of LVLMs through two tasks — Remote-Item Association (RIA) and In-Context Association (ICA) — and introduces an LLM-as-a-Judge scoring strategy alongside a process reward evaluation method. The benchmark reveals that even the strongest current LVLMs remain significantly behind humans.

## Background & Motivation

**Background**: LVLMs have made remarkable progress in visual understanding, language generation, and multi-step reasoning. However, **associative intelligence** — a cornerstone of human creative thinking and knowledge integration — remains severely underexplored in existing evaluations.

**Limitations of Prior Work**: The prior work "Labyrinth of Links" evaluates associative memory only through closed-form multiple-choice questions, which (1) may inadvertently hint at answers through fixed options, masking the model's true capabilities, and (2) cannot assess complex multi-step associative reasoning or divergent thinking.

**Core Problem**: Open-ended association reasoning is critical for real-world applications such as scientific discovery, creative design, personalized education, and innovative problem-solving. A benchmark free of predefined constraints is needed to rigorously evaluate the associative reasoning capabilities of LVLMs.

**Cognitive Science Foundation**: Association arises from the interplay between convergent thinking (identifying the optimal connection) and divergent thinking (generating multiple unique ideas). The Remote Associates Test (RAT) is a classic instrument but measures only single-hop convergent thinking. MM-OPERA extends this to multi-step reasoning structures.

## Method

### Task Design

**1. Remote-Item Association (RIA)**:
- Given two seemingly unrelated elements (images, text, or mixed modalities), the model must identify a meaningful connection between them.
- Example: an image of an armadillo + an image of Kevlar fabric → shared "protective function."
- Encourages cross-domain reasoning and permits multiple valid associative paths.

**2. In-Context Association (ICA)**:
- Extends RIA to in-context learning: the model first infers the associative pattern from a given element pair, then transfers that pattern to a new element.
- Example: bald eagle ↔ basketball (U.S. national symbol ↔ sport of national origin) → lion → ? (British symbol → football).
- Tests pattern abstraction and cross-domain transfer capabilities.

### Dataset Statistics
- **Total**: 11,497 instances (RIA: 8,021; ICA: 3,476)
- **Hierarchical capability taxonomy**: 3 levels (L-1: perception/concept; L-2: six dimensions; L-3: thirteen dimensions)
- **3 relation types**: Relation, Mutual Element, Metaphor
- **Association reasoning paths**: represented as directed paths; hop count reflects complexity
- **Diversity**: 15 languages, 22 thematic domains, multicultural backgrounds

### LLM-as-a-Judge Evaluation Strategy

**Holistic Score (0–4)**:
- 4: accurate, logically consistent, insightful, matching reference answer quality
- 3: reasonable understanding but lacking key insights or completeness
- 2: some relevance but lacking depth
- 1: vague, uncertain, or incomplete
- 0: contains factual errors

**Evaluation Metrics**:
- Score Rate (SR): average score as a percentage
- High Score Rate (HR-4): proportion of responses scoring 4
- HR-3: proportion of responses scoring ≥ 3
- $\triangle$HR = HR-3 − HR-4: reflects **divergent thinking** capacity

**Process Reward Evaluation (PR-Judge)**:
Model responses are restructured as associative paths $P = (s_1, s_2, \ldots, s_n)$, and each step is evaluated along three dimensions:
- **Reasonableness** $R_t \in [0,1]$: fluency and logical coherence of the reasoning
- **Uniqueness** $D_t \in [0,1]$: clarity of conceptual boundaries
- **Knowledge** $K_t \in \{0,1\}$: whether domain knowledge is demonstrated

Per-step association quality: $s_t = \alpha R_t D_t + (1-\alpha) K_t$

Overall reasoning score: $S_r = \sum_{t=1}^{n} s_t \delta^t$ (where $\delta$ is a cognitive decay factor that favors efficient reasoning paths)

## Key Experimental Results

### RIA Task (Representative Models)

| Model | SR(%) | HR-4(%) | HR-3(%) | △HR(%) |
|------|-------|---------|---------|--------|
| Gemini-2.5-Pro-Preview | 60.05 | 23.89 | 41.75 | 17.86 |
| o4-mini | 60.33 | 19.86 | 37.89 | 18.03 |
| GPT-4o | 59.72 | 10.89 | 28.83 | 17.94 |
| Gemini-2.0-Flash-Thinking | 59.11 | 17.73 | 36.60 | 18.87 |
| Qwen2.5-VL-7B | 52.28 | 5.35 | 20.36 | 15.00 |
| **Human** | **61.88** | **22.84** | **48.97** | **26.13** |

### ICA Task

| Model | SR(%) | HR-4(%) | HR-3(%) | △HR(%) |
|------|-------|---------|---------|--------|
| Gemini-2.5-Pro-Preview | 63.09 | 12.85 | 41.15 | 28.30 |
| o4-mini | 61.55 | 10.24 | 36.60 | 26.36 |
| GPT-4o | 58.26 | 6.27 | 29.62 | 23.35 |
| **Human** | **68.69** | **31.65** | **61.47** | **29.82** |

### Key Findings
1. **LVLMs lag significantly behind humans**: On ICA, the strongest model achieves HR-4 of only 12.85% vs. 31.65% for humans.
2. **Creativity gap**: Model $\triangle$HR is approximately 12%–20%, compared to 26%–30% for humans, indicating a substantial divergent thinking deficit.
3. **ICA is harder than RIA**: Most models score lower on ICA, suggesting that pattern abstraction and transfer pose greater challenges.
4. **Conservative reasoning vs. associative flexibility**: Gemini-1.5-Pro (conservative style) underperforms Flash (fast style), as excessive fact-checking and ethical considerations constrain creative association.
5. **Process evaluation**: Models perform adequately on reasonableness (50%–80%) but are severely deficient in uniqueness (fewer than half exceed 75%).

## Highlights & Insights
- ⭐⭐⭐⭐ **Fills an evaluation gap**: The first large-scale open-ended association reasoning benchmark, grounded in a solid cognitive science foundation.
- ⭐⭐⭐⭐ **Methodological innovation**: PR-Judge enables differentiation of reasoning path quality across responses that converge on the same conclusion via different routes.
- ⭐⭐⭐⭐ **Insightful findings**: Observations such as the trade-off between conservative reasoning and associative flexibility, and the uniqueness bottleneck, offer actionable guidance for model improvement.
- ⭐⭐⭐ **Multi-dimensional analysis**: Sensitivity tests (image replacement, text replacement, order sensitivity) + judge validation + diversity analysis.

## Limitations & Future Work
1. Reference answers serve only as heuristic baselines; open-ended evaluation still depends on the reliability of LLM-as-a-Judge.
2. The human baseline is drawn from a college student sample, which may not fully represent general human association reasoning.
3. The hyperparameter choices of $\alpha=0.9$ and $\delta=0.9$ lack systematic ablation.
4. The benchmark currently evaluates association reasoning only; it does not explore how the findings can be leveraged to concretely improve models' associative capabilities.
5. Some culturally and linguistically specific associations may disadvantage non-native models.

## Rating
⭐⭐⭐⭐ A benchmark work of considerable depth and breadth that systematically introduces association reasoning theory from cognitive psychology into LVLM evaluation. The task design is rigorous, the evaluation methodology is multi-layered, and the experimental analysis is thorough. The work exposes important shortcomings of current LVLMs in creative thinking and knowledge integration, providing clear directions for future model development.

## Related Work & Insights

## Rating

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning](../../ACL2025/vlm_reasoning/benchmarking_and_improving_large_vision-language_models_for_fundamental_visual_g.md)
- [\[NeurIPS 2025\] Recognition through Reasoning: Reinforcing Image Geo-localization with Large Vision-Language Models](recognition_through_reasoning_reinforcing_image_geo-localization_with_large_visi.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](../../ACL2026/vlm_reasoning/geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ICLR 2026\] PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts](../../ICLR2026/vlm_reasoning/puzzleworld_a_benchmark_for_multimodal_open-ended_reasoning_in_puzzlehunts.md)
- [\[NeurIPS 2025\] Sherlock: Self-Correcting Reasoning in Vision-Language Models](sherlock_selfcorrecting_reasoning_in_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
