---
title: >-
  [Paper Note] Idiom Understanding as a Tool to Measure the Dialect Gap
description: >-
  [ACL 2026][LLM Evaluation][dialect gap] Three new French idiom understanding benchmark datasets are proposed — QFrCoRE and QFrCoRT for Quebec French, and MFrCoE for standard French. Evaluation across 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms than on standard French idioms, quantifying the dialect gap phenomenon.
tags:
  - ACL 2026
  - LLM Evaluation
  - dialect gap
  - idiom understanding
  - Quebec French
  - benchmark dataset
  - multilingual evaluation
date: 2026-05-08
content_hash: 3607c1b0e2639f81
---

# Idiom Understanding as a Tool to Measure the Dialect Gap

**Conference**: ACL 2026
**arXiv**: [2510.05026](https://arxiv.org/abs/2510.05026)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: dialect gap, idiom understanding, Quebec French, benchmark dataset, multilingual evaluation

## TL;DR
Three new French idiom understanding benchmark datasets are proposed — QFrCoRE and QFrCoRT for Quebec French, and MFrCoE for standard French. Evaluation across 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms than on standard French idioms, quantifying the dialect gap phenomenon.

## Background & Motivation

**State of the Field**: Idiom understanding and dialect understanding are each well-established benchmark domains in NLP. LLMs perform well on standard French (Parisian French), yet their capabilities on other French dialects remain largely unexplored.

**Limitations of Prior Work**: (1) Existing idiom datasets predominantly focus on a single standard language variety, lacking dialectal coverage. (2) Although the dialect gap has been studied in languages such as Arabic and Bengali, local idioms have not been used as probes for dialectal comprehension. (3) Model proficiency on a prestige dialect does not guarantee understanding of regionally specific expressions.

**Root Cause**: While the grammar and syntax of a dialect can be approximately inferred from the standard language, dialectal idioms originate from local culture and history and cannot be derived from standard-language training data, constituting an intrinsic challenge for dialect understanding.

**Paper Goals**: (1) Construct idiom understanding benchmarks for both Quebec French and standard French. (2) Use dialectal idioms as a tool to quantify the dialect gap in LLMs.

**Starting Point**: Idiom understanding and dialect understanding are combined by exploiting the fact that dialectal idioms are culturally specific artifacts that cannot be generalized from standard-language training, making idiom understanding performance gaps a direct reflection of dialect capability gaps.

**Core Idea**: Local idiom understanding is used as a probe for dialectal competence, and paired standard/dialectal idiom benchmarks are constructed to quantify the dialect gap.

## Method

### Overall Architecture
Three benchmark datasets are constructed around a classification task in which a model, given an idiom and multiple candidate definitions, selects the correct definition. Evaluation is conducted across 111 LLMs, and performance on standard French is compared against performance on Quebec French.

### Key Designs

1. **QFrCoRE (Quebec French Corpus of Regional Expressions)**:

    - Function: Evaluates LLMs' understanding of Quebec multi-word idiomatic expressions.
    - Mechanism: 4,633 idiomatic expressions and their definitions are extracted via OCR from authoritative sources such as the *Dictionnaire des expressions québécoises*, followed by Azure OCR processing, regex-based cleaning, and manual deduplication. The task is formatted as multiple-choice definition matching.
    - Design Motivation: Multi-word idioms are core carriers of dialectal culture, and their meanings are typically unrelated to the literal senses of their constituent words.

2. **QFrCoRT (Quebec French Corpus of Regional Terms)**:

    - Function: Evaluates LLMs' understanding of single-word Quebec dialectal terms.
    - Mechanism: 171 dialectal vocabulary items and their definitions are manually extracted from five online Quebec linguistic resources; English loanwords are excluded to ensure pure dialectal understanding is tested.
    - Design Motivation: Single-word dialectal terms complement phrase-level evaluation, covering dialectal comprehension at different levels of granularity.

3. **MFrCoE (Metropolitan French Corpus of Expressions)**:

    - Function: Serves as a standard French control benchmark to quantify the dialect gap.
    - Mechanism: 4,938 standard French idioms are compiled from sources such as *Les 1001 expressions préférées des Français*, using the same evaluation format as QFrCoRE.
    - Design Motivation: Quantifying the gap requires evaluating both standard and dialectal varieties simultaneously, rather than reporting only absolute dialectal performance.

### Evaluation Protocol
A zero-shot classification setup is used, in which models select the correct definition of an idiom from a set of provided options. Accuracy on MFrCoE versus QFrCoRE is compared for each model.

## Key Experimental Results

### Main Results
Distribution of the dialect gap across 111 LLMs:

| Metric | Value |
|--------|-------|
| Models performing significantly worse on the dialect | 65.77% |
| Models performing significantly better on the dialect | 9.0% |
| Models showing no significant difference | 25.23% |
| Average accuracy on standard French | Higher (baseline) |
| Average accuracy on Quebec French | Significantly lower than standard French |

### Ablation Study

| Analysis Dimension | Finding |
|--------------------|---------|
| Model scale | Larger models exhibit smaller dialect gaps, though the gap is not eliminated |
| Idiom type | Culturally specific idioms show the largest gap |
| QFrCoRT vs QFrCoRE | Single-word and phrase-level dialect gaps are consistent |

### Key Findings
- Proficiency in standard French does not guarantee comprehension of regional dialect expressions — 65.77% of models exhibit a significant dialect gap.
- Only 9% of models perform better on the dialect, indicating that dialectal advantage is an exceptional case.
- The dialect gap is most severe for culturally specific idioms, validating the hypothesis that idioms serve as effective probes for dialect understanding.

## Highlights & Insights
- The approach of combining idiom understanding with dialect evaluation is original and generalizable to any language that has local idiomatic expressions.
- The detailed description of the dataset construction methodology enables replication for other French dialects, such as Swiss French or Belgian French.
- The large-scale evaluation across 111 models yields statistically reliable conclusions.

## Limitations & Future Work
- The study focuses solely on two dialectal varieties of French, and generalizability remains to be validated.
- The evaluation task is limited to a multiple-choice definition-matching format and does not assess open-ended idiom production.
- No analysis is conducted on the correlation between the proportion of dialectal training data and the dialect gap.
- Future work could extend the framework to languages with multiple dialects, such as English (US vs. UK vs. AU) or Spanish.

## Related Work & Insights
- **vs. Kantharuban et al. (dialect gap research)**: Their work measures the dialect gap using general NLP tasks, whereas this paper employs idiom understanding as a more targeted probe.
- **vs. Kim et al. (idiom understanding mechanisms)**: Their work investigates whether LLMs memorize or reason about idioms; this paper focuses on cross-dialectal differences in idiom comprehension.
- **vs. Sørensen & Nimb (Danish idioms)**: Their work evaluates a single language variety, whereas this paper introduces a methodology for quantifying the gap through standard–dialect pairing.

## Rating
- Novelty: ⭐⭐⭐⭐ The use of idioms as dialectal probes is original and broadly applicable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation across 111 models is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed dataset construction descriptions.
- Value: ⭐⭐⭐⭐ Makes a meaningful contribution to multilingual fairness research.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[NeurIPS 2025\] Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](../../NeurIPS2025/llm_evaluation/mind_the_gap_removing_the_discretization_gap_in_differentiable_logic_gate_networ.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[ICLR 2026\] Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces](../../ICLR2026/llm_evaluation/discount_model_search_for_quality_diversity_optimization_in_high-dimensional_mea.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)

<!-- RELATED:END -->
