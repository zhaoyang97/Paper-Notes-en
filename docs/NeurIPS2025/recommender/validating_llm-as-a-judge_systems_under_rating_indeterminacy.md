---
title: >-
  [Paper Note] Validating LLM-as-a-Judge Systems under Rating Indeterminacy
description: >-
  [NeurIPS 2025][Recommender Systems][LLM-as-a-Judge] This paper proposes a framework for validating LLM-as-a-Judge systems under rating indeterminacy, replacing forced-choice rating with a "response set" multi-label rating scheme, achieving up to 31% performance improvement in the selected judge system.
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "LLM-as-a-Judge"
  - "rating indeterminacy"
  - "validation framework"
  - "multi-label evaluation"
  - "forced-choice bias"
date: 2026-05-08
content_hash: 5863b8228c9279bd
---

# Validating LLM-as-a-Judge Systems under Rating Indeterminacy

**Conference**: NeurIPS 2025
**arXiv**: [2503.05965](https://arxiv.org/abs/2503.05965)  
**Code**: None  
**Area**: LLM Evaluation / Recommender Systems
**Keywords**: LLM-as-a-Judge, rating indeterminacy, validation framework, multi-label evaluation, forced-choice bias

## TL;DR

This paper proposes a framework for validating LLM-as-a-Judge systems under rating indeterminacy, replacing forced-choice rating with a "response set" multi-label rating scheme, achieving up to 31% performance improvement in the selected judge system.

## Background & Motivation

The LLM-as-a-Judge paradigm has become a mainstream approach for evaluating generative AI outputs, yet validating such systems poses fundamental challenges:

**Rating Indeterminacy**: For many evaluation items, the rating rubric admits multiple valid interpretations, and both human annotators and LLMs may assign different yet equally "correct" ratings to the same item.

**Forced-Choice Bias**: Existing methods require raters to select a single rating (forced-choice), obscuring the inherent uncertainty in ratings.

**Validation Distortion**: Humans and LLMs handle rating uncertainty differently, causing severe bias in validations based on forced-choice ratings.

## Method

### Overall Architecture

1. Analyze failure modes of forced-choice rating under rating indeterminacy.
2. Propose a "response set" multi-label rating scheme.
3. Establish theoretical connections between different rating schemes and validation metrics.
4. Conduct empirical validation across 11 real-world tasks and 9 commercial LLMs.

### Key Designs

1. **Formalization of Rating Indeterminacy**:

    - Define the "plausible rating set": for item $x$, there may exist multiple plausible ratings $R(x) \subseteq \{1,...,K\}$.
    - Rating indeterminacy arises when $|R(x)| > 1$.
    - Forced-choice requires $|R(x)| = 1$, which is inconsistent with reality.

2. **Response Set Rating Scheme**:

    - Raters annotate all plausible ratings rather than selecting only one.
    - For example, a response may be rated as "both 3 and 4 are reasonable," yielding annotation $\{3, 4\}$.
    - This preserves the uncertainty information inherent in ratings.

3. **Validation Metric Correction**:

    - Conventional metrics (e.g., agreement rate) are biased under indeterminacy.
    - A corrected human-judge agreement metric is proposed.
    - Theoretical proof shows that validation under the response set scheme is unbiased.

4. **Judge System Selection**:

    - Conventional methods select the judge with the highest agreement, leading to suboptimal choices.
    - The proposed method re-evaluates judges under the response set scheme, identifying the truly optimal system.

### Loss & Training

No model training is involved. The core contribution is methodological improvement in validation and evaluation.

## Key Experimental Results

### Main Results (11 Rating Tasks × 9 LLMs)

| Validation Scheme | True Rank of Selected Judge (Median) | Performance Gap vs. Best Judge (%) | Rate of Selecting Best Judge |
|---|---|---|---|
| Forced-choice + Majority Vote | 5th / 9 | −31% | 11% |
| Forced-choice + Weighted Aggregation | 4th / 9 | −25% | 18% |
| **Response Set (Ours)** | **1st / 9** | **0%** | **72%** |

### Comparison of Different LLMs as Judge

| LLM Judge | Forced-Choice Agreement ↑ | Response Set Agreement ↑ | Rank Change |
|---|---|---|---|
| GPT-4o | 0.72 | 0.81 | 3→1 |
| Claude-3.5 | 0.75 | 0.78 | 1→2 |
| GPT-4 | 0.71 | 0.77 | 4→3 |
| Gemini-1.5 | 0.73 | 0.74 | 2→4 |
| GPT-3.5 | 0.65 | 0.68 | 5→5 |
| Llama-3-70B | 0.62 | 0.66 | 6→6 |
| Mixtral-8x7B | 0.58 | 0.63 | 7→7 |
| Llama-3-8B | 0.52 | 0.55 | 8→8 |
| Phi-3 | 0.48 | 0.51 | 9→9 |

### Distribution of Rating Indeterminacy

| Task Category | High-Indeterminacy Item Ratio (%) | Forced-Choice Bias (%) | Correction Rate of Proposed Method (%) |
|---|---|---|---|
| Safety Evaluation | 42.5 | 28.3 | 85.2 |
| Creative Writing | 55.8 | 35.1 | 82.5 |
| Factual Accuracy | 18.2 | 12.5 | 91.8 |
| Dialogue Quality | 48.3 | 31.2 | 83.8 |
| Code Quality | 25.5 | 15.8 | 89.5 |

### Key Findings

1. Forced-choice rating introduces judge selection bias of up to 31%, a problem that has been severely underestimated.
2. Tasks with high indeterminacy (e.g., creative writing, safety evaluation) exhibit the most severe bias.
3. The rankings of top-tier LLMs shift substantially (GPT-4o rises from 3rd to 1st), suggesting that conclusions from existing evaluations may be unreliable.
4. The additional annotation cost of response set rating is manageable (approximately a 20% increase in time).

## Highlights & Insights

- **Reveals a previously overlooked fundamental problem**: The impact of rating indeterminacy on LLM-as-a-Judge validation has received virtually no systematic study prior to this work.
- **Practical recommendations**: Concrete improvements to rating schemes are provided and can be directly applied.
- **Large-scale experiments**: 15,075 benchmarking experiments lend statistical credibility to the conclusions.
- **Thought-provoking rank reversals**: Conclusions drawn from existing LLM leaderboards may warrant re-examination.

## Limitations & Future Work

1. Response set annotation requires more rigorous rater training and incurs higher costs.
2. Only ordinal rating scales are considered; non-ordinal evaluation formats (e.g., open-ended feedback) are not addressed.
3. Evaluation tasks are conducted in English only; multilingual settings may exhibit different indeterminacy patterns.
4. The theoretical framework assumes complete response sets, whereas raters may omit plausible ratings in practice.

## Related Work & Insights

- **Zheng et al. (2024)**: MT-Bench, the seminal work on LLM-as-a-Judge.
- **Annotation disagreement**: Research on annotation inconsistency in the NLP community.
- **Calibration**: Work on confidence calibration of LLMs.
- **Inter-annotator agreement**: Metrics such as Cohen's κ in conventional NLP evaluation.

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 5 |
| Value | 5 |
| Overall Recommendation | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems](../../ACL2025/recommender/cove_compressed_vocabulary_expansion_makes_better_llm-based_recommender_systems.md)
- [\[ICLR 2026\] Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](../../ICLR2026/recommender/continual_low-rank_adapters_for_llm-based_generative_recommender_systems.md)
- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[ICLR 2026\] Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning](../../ICLR2026/recommender/rank-grpo_training_llm-based_conversational_recommender_systems_with_reinforceme.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](../../AAAI2026/recommender/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)

</div>

<!-- RELATED:END -->
