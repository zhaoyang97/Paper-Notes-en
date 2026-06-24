---
title: >-
  [Paper Note] Has Machine Translation Evaluation Achieved Human Parity?
description: >-
  [ACL 2025][Multilingual & Machine Translation][Machine Translation Evaluation] Introduces human performance baselines to the rankings of the WMT Metrics Shared Task for the first time, finding that state-of-the-art automatic metrics often rank on par with or even higher than human evaluators. However, it argues that claiming "human parity" is premature and discusses the fundamental difficulties of measuring progress in MT evaluation.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Machine Translation Evaluation"
  - "Human Baseline"
  - "Meta-evaluation"
  - "Automatic Metrics"
  - "Human Parity"
date: 2026-05-08
content_hash: b596a4db348414fd
---

# Has Machine Translation Evaluation Achieved Human Parity?

**Conference**: ACL 2025  
**arXiv**: [2506.19571](https://arxiv.org/abs/2506.19571)  
**Code**: [https://github.com/SapienzaNLP/human-parity-mt-eval](https://github.com/SapienzaNLP/human-parity-mt-eval)  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation Evaluation, Human Baseline, Meta-evaluation, Automatic Metrics, Human Parity  

## TL;DR
Introduces human performance baselines to the rankings of the WMT Metrics Shared Task for the first time, finding that state-of-the-art automatic metrics often rank on par with or even higher than human evaluators. However, it argues that claiming "human parity" is premature and discusses the fundamental difficulties of measuring progress in MT evaluation.

## Background & Motivation
**Background**: In MT evaluation, the performance of automatic metrics is measured by their consistency with human judgments. In recent years, neural metrics (BLEURT, COMET, MetricX) and LLM-based metrics (GEMBA-MQM) have shown increasing correlation with human judgments, approaching or even exceeding inter-annotator agreement.

**Limitations of Prior Work**: Unlike NLU tasks (such as HellaSwag or MMLU), MT evaluation has never established a human performance baseline. Without a human upper bound reference, it is impossible to determine how far automatic metrics actually are from human performance.

**Key Challenge**: As the performance of automatic metrics continues to improve, without a human baseline, we cannot even determine whether the differences in metric scores are meaningful—does a higher ranking indicate truly stronger evaluation capability, or just better fitting to the style of specific annotators?

**Goal**: To quantify the gap between MT automatic metrics and human evaluators, and to discuss the implications if this gap has indeed disappeared.

**Key Insight**: Leverage multi-year human annotation data accumulated by WMT (with different annotation protocols and annotators), and include human annotators as "evaluators" in the metric rankings to compare them directly with automatic metrics.

**Core Idea**: Use inter-annotator agreement as the human performance reference, and unifiedly rank human and automatic evaluators using the meta-evaluation strategy from WMT 2024. The study finds that while automatic metrics have reached human-level performance, it warns that claiming human parity requires caution.

## Method

### Overall Architecture
Using test sets from WMT 2020-2024, multiple human annotators across four annotation protocols (MQM, ESA, pSQM, DA+SQM) and all participating automatic metrics are collected. Taking MQM annotations as the gold standard, other human annotators and automatic metrics are unified under a single ranking system. The core output is a unified ranking table of evaluators across various years and language directions.

### Key Designs

1. **Disjoint Annotators Constraint**:

    - **Function**: Ensure that there is no annotator overlap among the human baselines.
    - **Mechanism**: Since WMT test sets are usually annotated by multiple annotators dividing different segments, direct combination might lead to a single annotator contributing to multiple "evaluators", artificially inflating consistency. This is addressed by solving an optimization problem: finding the largest subset of segments and annotator partitions such that each group of annotators covers all segments with zero overlap between groups.
    - **Design Motivation**: To avoid overestimating the human baseline performance and to ensure fair comparisons among humans.

2. **Dual Meta-Evaluation Strategy**:

    - **Function**: Evaluate all evaluators from two complementary perspectives.
    - **Mechanism**: (1) SPA (Soft Pairwise Accuracy) measures the evaluator's ability to rank MT systems—consistency with gold-standard system-level rankings; (2) $\text{acc}^*_{eq}$ (Pairwise Accuracy with Tie Calibration) measures the fine-grained ranking ability for different translations of the same source sentence.
    - **Design Motivation**: System-level and translation-level evaluations reflect different dimensions of capability, and human vs. automatic metrics may perform differently across these dimensions.

3. **Cross-year and Cross-language Direction Analysis**:

    - **Function**: Cover 7 test sets across 4 years (2020-2024) and 4 language directions.
    - **Mechanism**: Independently calculate rankings and statistical significance clustering on each test set to observe whether the relative positioning of humans vs. metrics remains stable.
    - **Design Motivation**: To avoid the contingency of a single test set and to verify the generalizability of the findings.

## Key Experimental Results

### Main Results
The rankings of human evaluators and top automatic metrics on representative test sets (SPA / $\text{acc}^*_{eq}$ rank):

| Test Set | Evaluator | SPA Rank | $\text{acc}^*_{eq}$ Rank |
|--------|--------|----------|--------------------------|
| 2020 en→de | MQM-2020-2 (Human) | 1 | 1 |
| 2020 en→de | BLEURT-0.2 | 2 | 4 |
| 2022 en→de | MQM-2022-3 (Human) | 1 | 1 |
| 2022 en→de | MetricX-23-QE-XXL | 1 | 3 |
| 2023 en→de | GEMBA-MQM | 1 | 5 |
| 2023 en→de | MQM-2023-2 (Human) | 1 | 6 |
| 2023 en→de | DA+SQM (Human) | 2 | 14 |
| 2024 en→es | CometKiwi-XXL | 1 | 4 |
| 2024 en→es | ESA (Human) | 2 | 8 |

### Comparison of Key Findings

| Observation Dimension | Findings |
|----------|------|
| SPA Rank | Human evaluators typically share the same statistical significance cluster as top automatic metrics. |
| $\text{acc}^*_{eq}$ Rank | Human evaluators are frequently outperformed by automatic metrics, particularly by annotators using non-MQM protocols. |
| Cross-year Trend | Humans had a clear lead in 2020, but automatic metrics frequently outperformed humans in 2023-2024. |
| DA+SQM Protocol | The worst-performing human protocol, often ranking in the middle-to-lower tier, likely due to low annotation quality. |

### Key Findings
- **Human evaluators are not always superior to automatic metrics**: Humans typically tie with the best metrics on SPA, but frequently lag behind on $\text{acc}^*_{eq}$.
- **MQM annotators perform best, but only held a clear lead in 2020**: As metrics improved, MQM annotators were also caught up with by 2023-2024.
- **DA+SQM is the weakest human protocol**: It often ranks outside the top 10, exposing the issues associated with low-quality annotation.
- **The fluency-only metric sentinel-cand-mqm surprisingly ties with ESA human annotators**: This suggests that the translation differences in current test sets might be minor nuances strictly at the fluency level, indicating that the test sets are too simple.
- The meta-evaluation metrics themselves exhibit bias: $\text{acc}^*_{eq}$ favors evaluators with continuous score distributions, which disadvantages human annotators who produce discrete scores.

## Highlights & Insights
- **Establishing a human performance reference for MT evaluation for the first time**: This finally brings a human baseline to the MT evaluation field, similar to NLU tasks, enabling meaningful discussions on "human parity."
- **The prudent discussion of "human parity" is more valuable than the results themselves**: It points out three confounding factors—test set difficulty, annotation quality, and meta-evaluation metric bias—reminding the community not to declare victory prematurely.
- **Proposes a potential "ceiling effect" that MT evaluation might face**: If metrics are already as good as humans (or if humans do not agree with each other), what does a higher ranking really mean? Is it truly better, or is it just fitting a specific gold-standard annotator better?

## Limitations & Future Work
- Limited in scope by being restricted to test sets with multiple human annotations (only 7 test sets across 4 language directions).
- For some test sets, after restricting to disjoint annotators, the number of segments becomes very small (e.g., only 145 segments for 2023 en→de), which raises concerns about statistical reliability.
- Has not tested whether automatic metrics still achieve human-level performance in out-of-domain areas (legal, medical, etc.).
- Did not conduct an in-depth analysis of human vs. metric differences across various error types (terminology, gender, numbers, etc.).

## Related Work & Insights
- **vs WMT Metrics Shared Task**: Over the years, WMT has only ranked automatic metrics; this work is the first to include humans in the rankings for a fair comparison.
- **vs Perrella et al. (2024)**: The only prior work attempting to compare humans and metrics, but it only used the low-quality DA+SQM protocol, leading to unreliable conclusions. This work covers 4 protocols.
- This work holds significant "meta-reflective" value for the MT evaluation community—how do we ensure we can still measure progress?

## Rating
- Novelty: ⭐⭐⭐⭐ Establishes a systematic human baseline for MT evaluation for the first time, offering a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple years and languages with statistical significance analysis, though test set coverage and size are limited.
- Writing Quality: ⭐⭐⭐⭐⭐ In-depth discussion, rigorous argumentation, and provides forward-looking insights for the community.
- Value: ⭐⭐⭐⭐ Touches upon the fundamental issues of MT evaluation, exerting an important influence on the future direction of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] AskQE: Question Answering as Automatic Evaluation for Machine Translation](askqe_question_answering_as_automatic_evaluation_for_machine_translation.md)
- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Context Augmented Token-Level Post-Editing for Human Interpreting](context_augmented_token-level_post-editing_for_human_interpreting.md)

</div>

<!-- RELATED:END -->
