---
title: >-
  [Paper Note] Idiom Understanding as a Tool to Measure the Dialect Gap
description: >-
  [ACL 2026 Findings][LLM Evaluation][Dialect Gap] This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Standard French MFrCoE). Evaluation of 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms than on standard French, quantifying the dialect gap phenomenon.
tags:
  - "ACL 2026 Findings"
  - "LLM Evaluation"
  - "Dialect Gap"
  - "Idiom Understanding"
  - "Quebec French"
  - "Benchmark Datasets"
  - "Multilingual Evaluation"
date: 2026-05-08
content_hash: adda9b9e6d9d3b73
---

# Idiom Understanding as a Tool to Measure the Dialect Gap

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.05026](https://arxiv.org/abs/2510.05026)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Dialect Gap, Idiom Understanding, Quebec French, Benchmark Datasets, Multilingual Evaluation

## TL;DR
This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Standard French MFrCoE). Evaluation of 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms than on standard French, quantifying the dialect gap phenomenon.

## Background & Motivation

**Background**: Idiom understanding and dialect understanding are mature evaluation benchmark fields in NLP. LLMs perform well in standard French (Parisian French), but research on their capabilities in other French dialects is minimal.

**Limitations of Prior Work**: (1) Existing idiom datasets mostly focus on a single standard language variant and lack dialect coverage; (2) Although dialect gap research has been validated in languages like Arabic and Bengali, it has not utilized local idioms as probes for dialect understanding; (3) Model proficiency in authoritative dialects does not guarantee the ability to understand specific expressions of regional dialects.

**Key Challenge**: Grammatical and syntactic rules of dialects can be approximately inferred from the standard language, but dialectal idioms originate from local culture and history and cannot be derived from standard language training, constituting an essential challenge for dialect understanding.

**Goal**: (1) Construct idiom understanding benchmarks for Quebec French and standard French; (2) Use dialectal idioms as a tool to quantify the dialect gap of LLMs.

**Key Insight**: Combining idiom understanding with dialect understanding—dialectal idioms are unique cultural products of dialects that cannot generalize from standard language training, thus the performance gap in idiom understanding directly reflects the dialectal capability gap.

**Core Idea**: Use local idiom understanding as a probe for dialectal capability, and construct paired standard/dialect idiom benchmarks to quantify the dialect gap.

## Method

### Overall Architecture

The core mechanism of this paper is to treat idiom understanding as a "litmus test" for probing dialectal capability: while dialect grammar and syntax can be approximately extrapolated from standard languages, dialectal idioms are rooted in local cultural history and cannot be generalized from standard training. Therefore, the comprehension drop between standard and dialectal idioms can be directly used as a quantitative reading of the dialect gap. For this purpose, the authors constructed three paired benchmarks—QFrCoRE (phrase-level) and QFrCoRT (word-level) for Quebec French, and MFrCoE (control group) for Standard French—all utilizing a zero-shot definition matching task: given an idiom and several candidate definitions, the model selects the correct meaning. Finally, the accuracy difference between MFrCoE and QFrCoRE was compared across 111 LLMs, with the difference serving as the dialect gap for each model.

### Key Designs

**1. QFrCoRE (Québec French Expressions Corpus): The primary probe for phrase-level dialectal idioms**

Multi-word idioms are the core cultural carriers of dialects; their meanings are usually unrelated to the literal meanings of the constituent words, making them the hardest part for standard language training to cover. Thus, it serves as the primary test set for the dialect gap. The authors extracted data from authoritative sources like the *Dictionary of Quebec Expressions* via Azure OCR, followed by regex cleaning and manual de-duplication, resulting in 4,633 idiomatic expressions and definitions for the multiple-choice definition matching task.

**2. QFrCoRT (Québec French Terminology Corpus): Granular supplement for word-level dialectal terms**

Beyond phrase-level evaluation, it's necessary to verify finer-grained dialect understanding. Thus, 171 word-level dialectal terms and definitions were manually extracted from five online Quebec linguistic resources. English loanwords were intentionally excluded during extraction to ensure the test measures pure dialectal understanding rather than mixed usage, allowing cross-verification of whether the dialect gap is consistent across word and phrase levels.

**3. MFrCoE (Metropolitan French Expressions Corpus): Control benchmark for quantifying the gap**

Reporting absolute performance on dialects alone is insufficient—it requires a homogeneous standard language benchmark for subtraction to make the gap meaningful. The authors constructed 4,938 standard French idioms from sources like *1001 Favorite Expressions of the French* and maintained the exact same evaluation format as QFrCoRE, allowing accuracies to be directly subtracted to turn the "dialect gap" into a comparable numerical value.

## Key Experimental Results

### Main Results
Distribution of dialect gaps across 111 LLMs:

| Metric | Value |
|------|------|
| Proportion of models performing significantly worse on dialect | 65.77% |
| Proportion of models performing significantly better on dialect | 9.0% |
| Proportion of models with no significant difference | 25.23% |
| Average Accuracy on Standard French | High (Baseline) |
| Average Accuracy on Quebec French | Significantly lower than Standard French |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| Model Scale | Larger models have smaller dialect gaps but do not eliminate them |
| Idiom Type | Culture-specific idioms show the largest gap |
| QFrCoRT vs QFrCoRE | Word-level and phrase-level dialect gaps are consistent |

### Key Findings
- Proficiency in standard French does not guarantee regional dialect understanding capability—65.77% of models exhibit a significant dialect gap.
- Only 9% of models perform better on the dialect, suggesting that dialectal preference is a rare phenomenon.
- The dialect gap is most severe for culture-specific idioms, validating the hypothesis that "idioms are effective probes for dialect understanding."

## Highlights & Insights
- The evaluation approach of cleverly combining idiom understanding with dialect understanding is original and can be extended to any language with regional idioms.
- Detailed methodology for dataset construction makes it replicable for other dialects (e.g., Swiss French, Belgian French).
- Large-scale evaluation of 111 models provides statistically reliable conclusions.

## Limitations & Future Work
- Focuses only on two dialectal variants of one language (French); generalization remains to be verified.
- The evaluation task is limited to a multiple-choice definition matching format and does not test open-ended idiom usage.
- The study did not analyze the correlation between the proportion of dialectal data in model training sets and the dialect gap.
- Future work can extend to other multi-dialect languages such as English (US vs. UK vs. AU) and Spanish.

## Related Work & Insights
- **vs Kantharuban et al. (Dialect Gap Research)**: They use general NLP tasks to measure dialect gaps; this paper uses idiom understanding as a more precise probe.
- **vs Kim et al. (Idiom Understanding Mechanism)**: They study whether LLMs memorize or reason about idioms; this paper focuses on comprehension differences between dialects.
- **vs Sørensen & Nimb (Danish Idioms)**: They evaluate a single language; this paper provides a methodology for quantifying gaps through standard-dialect pairing.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using idioms as dialect probes is novel and generalizable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation of 111 models is very thorough.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed descriptions of dataset construction.
- Value: ⭐⭐⭐⭐ Practical contribution to multilingual fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](../../ICML2026/llm_evaluation/on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ICLR 2026\] The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas](../../ICLR2026/llm_evaluation/the_ideation-execution_gap_execution_outcomes_of_llm-generated_versus_human_rese.md)
- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)

</div>

<!-- RELATED:END -->
