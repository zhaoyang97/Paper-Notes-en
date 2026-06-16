---
title: >-
  [Paper Note] Idiom Understanding as a Tool to Measure the Dialect Gap
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Metropolitan French MFrCoE). Evaluation across 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms compared to standard French, quantifying the dialect gap phenomenon.
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: e86cfe1eed53cee0
---
# Idiom Understanding as a Tool to Measure the Dialect Gap

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.05026](https://arxiv.org/abs/2510.05026)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Dialect Gap, Idiom Understanding, Quebec French, Benchmark Datasets, Multilingual Evaluation

## TL;DR
This paper proposes three new French idiom understanding benchmark datasets (Quebec French QFrCoRE/QFrCoRT and Metropolitan French MFrCoE). Evaluation across 111 LLMs reveals that 65.77% of models perform significantly worse on dialectal idioms compared to standard French, quantifying the dialect gap phenomenon.

## Background & Motivation

**Background**: Idiom understanding and dialect understanding are established evaluation benchmark fields in NLP. LLMs perform well in Metropolitan French (Parisian French), but research on their capabilities in other French dialects is extremely sparse.

**Limitations of Prior Work**: (1) Most existing idiom datasets focus on a single standard language variant and lack dialect coverage; (2) Although the dialect gap has been validated in languages such as Arabic and Bengali, local idioms have not been utilized as probes for dialect understanding; (3) Proficiency in a dominant dialect does not guarantee a model's ability to understand unique regional expressions.

**Key Challenge**: While the grammar and syntax of a dialect can be approximately inferred from the standard language, dialectal idioms are rooted in local culture and history. They cannot be derived from standard language training, thus constituting an essential challenge for dialect understanding.

**Goal**: (1) Construct idiom understanding benchmarks for Quebec French and Metropolitan French; (2) Utilize dialectal idioms as a tool to quantify the dialect gap in LLMs.

**Key Insight**: By combining idiom understanding with dialect understanding—recognizing that dialectal idioms are unique cultural products that do not generalize from standard language training—the performance gap in idiom understanding can directly reflect the gap in dialect capability.

**Core Idea**: Treat local idiom understanding as a probe for dialect capability and construct paired standard/dialect idiom benchmarks to quantify the dialect gap.

## Method

### Overall Architecture

The core mechanism of this study is to use idiom understanding as a "litmus test" for dialect capability. While dialectal grammar and syntax may be extrapolated from the standard language, dialectal idioms are deeply embedded in local culture and history and struggle to generalize from standard training. Therefore, the comprehension disparity between standard and dialectal idioms serves as a quantified metric for the dialect gap. The authors construct three paired benchmarks—QFrCoRE (phrase-level) and QFrCoRT (word-level) for Quebec French, and MFrCoE (control group) for Metropolitan French. A unified zero-shot definition matching task is adopted: given an idiom and several candidate definitions, the model must select the correct meaning. Finally, the accuracy difference between MFrCoE and QFrCoRE is compared across 111 LLMs, with this difference representing each model's dialect gap.

### Key Designs

**1. QFrCoRE (Quebec French Expression Corpus): Primay probe for phrase-level dialect idioms**  
Multi-word idioms are core carriers of dialect culture, and their meanings are usually unrelated to the literal definitions of their constituent words. This represents the area most difficult for standard language training to cover, making it the primary test set for the dialect gap. The authors extracted 4,633 idiomatic expressions and definitions from authoritative sources such as the *Dictionnaire des expressions québécoises* using Azure OCR, followed by regex cleaning and manual deduplication.

**2. QFrCoRT (Quebec French Terminology Corpus): Granular supplement for word-level dialect terms**  
To verify dialect understanding at a finer granularity beyond the phrase level, the authors manually extracted 171 word-level dialectal terms and definitions from five online Quebec linguistic resources. English loanwords were intentionally excluded during extraction to ensure the measuring of pure dialect comprehension rather than "Franglais," allowing word-level and phrase-level results to cross-verify the consistency of the dialect gap.

**3. MFrCoE (Metropolitan French Expression Corpus): Control benchmark for quantifying the gap**  
Reporting absolute performance on a dialect is insufficient without a homogeneous standard language benchmark for comparison. The authors constructed 4,938 Metropolitan French idioms from sources such as *Les 1001 expressions préférées des français*. By maintaining the exact same evaluation format as QFrCoRE, the accuracies can be directly subtracted, resulting in a comparable numerical "dialect gap" value.

## Key Experimental Results

### Main Results
Distribution of the dialect gap across 111 LLMs:

| Metric | Value |
|------|------|
| Proportion of models significantly worse on dialect | 65.77% |
| Proportion of models significantly better on dialect | 9.0% |
| Proportion of models with no significant difference | 25.23% |
| Average Accuracy (Metropolitan French) | High (Baseline) |
| Average Accuracy (Quebec French) | Significantly lower than Metropolitan |

### Ablation Study

| Analysis Dimension | Finding |
|---------|------|
| Model Scale | Gaps are smaller in larger models but do not disappear |
| Idiom Type | Culture-specific idioms show the largest gap |
| QFrCoRT vs QFrCoRE | Dialect gaps are consistent across word and phrase levels |

### Key Findings
- Proficiency in Metropolitan French does not guarantee regional dialect understanding capabilities—65.77% of models exhibit a significant dialect gap.
- Only 9% of models perform better on the dialect, indicating that dialectal preference is extremely rare.
- The dialect gap is most severe for culture-specific idioms, validating the hypothesis that "idioms are effective probes for dialect understanding."

## Highlights & Insights
- The original evaluation approach of combining idiom understanding with dialect understanding is highly generalizable to any language with regional idioms.
- The detailed description of the dataset construction methodology allows for replication in other dialects (e.g., Swiss or Belgian French).
- The large-scale evaluation of 111 models provides statistically reliable conclusions.

## Limitations & Future Work
- The study focuses only on two variants of the French language; the generalizability to other languages remains to be verified.
- The evaluation task is limited to a multiple-choice definition matching format and does not test open-ended idiom usage.
- The correlation between the proportion of dialectal corpora in the training data and the resulting dialect gap was not analyzed.
- Future work could extend to multi-dialect languages such as English (US vs. UK vs. AU) or Spanish.

## Related Work & Insights
- **vs. Kantharuban et al. (Dialect Gap Research)**: While they use general NLP tasks to measure dialect gaps, this work uses idiom understanding as a more precise probe.
- **vs. Kim et al. (Idiom Understanding Mechanisms)**: They investigate whether LLMs rely on memory or reasoning for idioms; this work focuses on comprehension differences between dialects.
- **vs. Sørensen & Nimb (Danish Idioms)**: They evaluate a single language; this work provides a methodology for quantifying gaps through standard-dialect pairing.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using idioms as dialect probes is novel and extensible.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The large-scale evaluation of 111 models is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed description of dataset construction.
- Value: ⭐⭐⭐⭐ Provides a practical contribution to multilingual fairness research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2025\] Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](../../ACL2025/llm_evaluation/retrieval_models_arent_tool-savvy_benchmarking_tool_retrieval_for_large_language.md)
- [\[ICML 2026\] On Effectiveness and Efficiency of Agentic Tool-calling and RL Training](../../ICML2026/llm_evaluation/on_effectiveness_and_efficiency_of_agentic_tool-calling_and_rl_training.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[AAAI 2026\] Beyond Accuracy: A Cognitive Load Framework for Mapping the Capability Boundaries of Tool-use Agents](../../AAAI2026/llm_evaluation/beyond_accuracy_a_cognitive_load_framework_for_mapping_the_c.md)

</div>

<!-- RELATED:END -->
