---
title: >-
  [Paper Note] Idiom Understanding as a Tool to Measure the Dialect Gap
description: >-
  [ACL 2026][LLM Evaluation][Dialect gap] This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Standard French MFrCoE). Evaluation on 111 LLMs reveals that 65.77% o…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Dialect gap"
  - "Idiom understanding"
  - "Quebec French"
  - "Benchmark dataset"
  - "Multilingual evaluation"
date: 2026-05-08
content_hash: eb266743107b20d4
---

# Idiom Understanding as a Tool to Measure the Dialect Gap

**Conference**: ACL 2026  
**arXiv**: [2510.05026](https://arxiv.org/abs/2510.05026)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Dialect gap, Idiom understanding, Quebec French, Benchmark dataset, Multilingual evaluation

## TL;DR
This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Standard French MFrCoE). Evaluation on 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms than on standard French, quantifying the dialect gap phenomenon.

## Background & Motivation

**Background**: Idiom understanding and dialect understanding are mature evaluation benchmark areas in NLP. While LLMs perform well on standard French (Metropolitan French), research on their capabilities in other French dialects is minimal.

**Limitations of Prior Work**: (1) Existing idiom datasets mostly focus on a single standard language variety, lacking dialectal coverage. (2) Although the dialect gap has been validated in languages such as Arabic and Bengali, local idioms have not been used as probes for dialect understanding. (3) Model proficiency in authoritative dialects does not guarantee the ability to understand regional dialect-specific expressions.

**Key Challenge**: While grammatical and syntactic rules of dialects can be approximately inferred from standard languages, dialectal idioms originate from local culture and history. They cannot be derived from standard language training, constituting an essential challenge for dialect understanding.

**Goal**: (1) Construct idiom understanding benchmarks for both Quebec French and Standard French; (2) Utilize dialectal idioms as a tool to quantify the dialect gap in LLMs.

**Key Insight**: By combining idiom understanding with dialect understanding—given that dialectal idioms are unique cultural products that do not generalize from standard language training—the performance gap in idiom understanding directly reflects the dialect capability gap.

**Core Idea**: Use local idiom understanding as a probe for dialect capability, constructing paired standard/dialect idiom benchmarks to quantify the dialect gap.

## Method

### Overall Architecture
The authors construct three benchmark datasets and design a classification task (given an idiom and multiple definitions, the model selects the correct one). They evaluate 111 LLMs and compare the performance differences between Standard French and Quebec French.

### Key Designs

1.  **QFrCoRE (Quebec French Expressions Corpus)**:
    - **Function**: Evaluates LLM understanding of Quebec multi-word idiomatic expressions.
    - **Mechanism**: 4,633 idiomatic expressions and their definitions were extracted via OCR from authoritative sources like the *Dictionary of Quebec Expressions*, followed by Azure OCR, regex cleaning, and manual deduplication. The task is multiple-choice definition matching.
    - **Design Motivation**: Multi-word idioms are core carriers of dialect culture, and their meanings are usually unrelated to the literal meanings of the constituent words.

2.  **QFrCoRT (Quebec French Terms Corpus)**:
    - **Function**: Evaluates LLM understanding of Quebec word-level dialectal terms.
    - **Mechanism**: 171 dialectal terms and definitions were manually extracted from five online Quebec linguistic resources, excluding English loanwords to ensure pure dialect understanding testing.
    - **Design Motivation**: Word-level dialectal terms complement phrase-level assessments, covering dialect understanding at different granularities.

3.  **MFrCoE (Metropolitan French Expressions Corpus)**:
    - **Function**: Serves as a control benchmark for standard French to quantify the dialect gap.
    - **Mechanism**: 4,938 standard French idioms were constructed from sources like *The 1001 Favorite Expressions of the French*. It maintains the same evaluation format as QFrCoRE.
    - **Design Motivation**: The gap can only be quantified by simultaneously evaluating standard and dialectal versions, rather than merely reporting absolute dialectal performance.

### Evaluation
The study uses a zero-shot classification task where the model must choose the correct definition of an idiom from given options. The accuracy difference for each model is compared between MFrCoE and QFrCoRE.

## Key Experimental Results

### Main Results
The distribution of the dialect gap across 111 LLMs:

| Indicator | Value |
|-----------|-------|
| Proportion of models significantly worse on dialect | 65.77% |
| Proportion of models significantly better on dialect | 9.0% |
| Proportion of models with no significant difference | 25.23% |
| Standard French average accuracy | Higher (Baseline) |
| Quebec French average accuracy | Significantly lower than standard French |

### Ablation Study

| Analysis Dimension | Findings |
|--------------------|----------|
| Model Scale | Large models show smaller dialect gaps but do not eliminate them. |
| Idiom Type | Culture-specific idioms exhibit the largest gap. |
| QFrCoRT vs QFrCoRE | Dialect gaps are consistent across both word-level and phrase-level. |

### Key Findings
- Proficiency in standard French does not guarantee regional dialect understanding; 65.77% of models exhibit a significant dialect gap.
- Only 9% of models perform better on the dialect, indicating that dialect preference is a rare exception.
- The dialect gap is most severe for culture-specific idioms, validating the hypothesis that idioms are effective probes for dialect understanding.

## Highlights & Insights
- The evaluation approach of combining idiom understanding with dialect understanding is original and can be extended to any language with regional idioms.
- The detailed description of the dataset construction methodology makes it reproducible for other dialects (e.g., Swiss French, Belgian French).
- The large-scale evaluation of 111 models provides statistically reliable conclusions.

## Limitations & Future Work
- The study focuses only on two dialectal varieties of French; generalizability remains to be verified.
- The evaluation task is limited to a multiple-choice matching format and does not test open-ended idiom usage.
- The correlation between the proportion of dialectal corpora in model training data and the dialect gap was not analyzed.
- Future work could expand the scope to other multi-dialectal languages such as English (US vs UK vs AU) and Spanish.

## Related Work & Insights
- **vs Kantharuban et al. (Dialect Gap Research)**: While they use general NLP tasks to measure gaps, this work uses idiom understanding as a more precise probe.
- **vs Kim et al. (Idiom Understanding Mechanism)**: They study whether LLMs memorize or reason about idioms; this work focuses on the disparity in understanding between dialects.
- **vs Sørensen & Nimb (Danish Idioms)**: They evaluate a single language; this work provides a methodology for quantifying gaps through standard-dialect pairings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of using idioms as dialect probes is novel and generalizable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ The evaluation involving 111 models is very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear and the dataset construction is well-documented.
- **Value**: ⭐⭐⭐⭐ This work provides a practical contribution to multilingual fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[NeurIPS 2025\] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](../../NeurIPS2025/llm_evaluation/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/llm_evaluation/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)

</div>

<!-- RELATED:END -->
