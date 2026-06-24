---
title: >-
  [Paper Note] Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?
description: >-
  [ACL 2025][Text Generation][Grammatical Error Correction Evaluation] This paper points out a fundamental discrepancy between current automatic evaluation and human evaluation in GEC regarding the aggregation pipeline "from sentence-level scores to system rankings." Specifically, human evaluation relies on sentence-level pairwise comparisons combined with the TrueSkill ranking algorithm, whereas automatic evaluation typically uses average absolute scores followed by sorting. B…
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Grammatical Error Correction Evaluation"
  - "TrueSkill"
  - "Meta-Evaluation"
  - "Sentence-level Comparison"
  - "Rank Aggregation"
date: 2026-05-08
content_hash: f11d69b079f35d47
---

# Rethinking Evaluation Metrics for Grammatical Error Correction: Why Use a Different Evaluation Process than Human?

**Conference**: ACL 2025  
**arXiv**: [2502.09416](https://arxiv.org/abs/2502.09416)  
**Code**: [gotutiyan/gec-metrics](https://github.com/gotutiyan/gec-metrics)  
**Area**: NLP / Text Generation  
**Keywords**: Grammatical Error Correction Evaluation, TrueSkill, Meta-Evaluation, Sentence-level Comparison, Rank Aggregation

## TL;DR

This paper points out a fundamental discrepancy between current automatic evaluation and human evaluation in GEC regarding the aggregation pipeline "from sentence-level scores to system rankings." Specifically, human evaluation relies on sentence-level pairwise comparisons combined with the TrueSkill ranking algorithm, whereas automatic evaluation typically uses average absolute scores followed by sorting. By adopting TrueSkill aggregation for automatic evaluation to bridge this gap, this study substantially improves the correlation of most metrics with human evaluation on the SEEDA benchmark, even allowing BERT-level metrics to outperform GPT-4.

## Background & Motivation

**Background**: The field of Grammatical Error Correction (GEC) has developed various automatic evaluation metrics, including edit-based metrics like ERRANT/PT-ERRANT, n-gram-based metrics like GLEU+/GREEN, and neural-based sentence-level metrics like SOME/IMPARA/Scribendi Score. The core objective of these metrics is to rank GEC systems and ensure consistency with human evaluation rankings; researchers typically employ Spearman/Pearson correlation coefficients to measure the alignment between metric rankings and human rankings.

**Limitations of Prior Work**: Despite the goal of "reproducing human evaluation rankings," automatic evaluation and human evaluation employ completely different pipelines to aggregate sentence-level results into system rankings. Human evaluation performs relative pairwise comparisons of multiple system outputs for each sentence, and then aggregates them into system rankings using algorithms such as TrueSkill. Conversely, automatic evaluation computes an absolute score for each sentence and then averages or sums these scores at the corpus level to sort systems. These two aggregation pipelines mathematically yield different ranking results. Particularly when the performance gap between systems is minuscule, the average of absolute scores can be dominated by individual outliers, whereas pairwise comparisons are more robust.

**Key Challenge**: Although the goal of automatic evaluation is to align with human evaluation, they employ different aggregation methods, a "pipeline gap" that has long been overlooked. While prior research (e.g., Kobayashi et al. 2024a) utilized TrueSkill for their proposed metrics, they still applied traditional average aggregation to other baseline metrics, leading to an unfair comparison and failing to explicitly address the fundamental nature of this discrepancy.

**Goal**: To systematically identify and bridge the gap between human and automatic evaluation in their aggregation pipelines: unifying all automatic evaluation metrics under the same TrueSkill aggregation approach as human evaluation, and validating the impact of this change on the standard meta-evaluation benchmark SEEDA.

**Key Insight**: The authors' observation is remarkably straightforward: since the target is to align automatic rankings with human rankings, why not directly adopt the same ranking pipeline as human evaluation? Specifically, sentence-level scores of existing metrics are first used to perform pairwise comparisons (the system with the higher score wins), and then all pairwise comparison outcomes are fed into the TrueSkill algorithm to compute system rankings. This method applies to any existing metric without altering the underlying metric calculations.

**Core Idea**: To replace traditional average aggregation with the same TrueSkill aggregation method used in human evaluation for automatic GEC evaluation, thereby significantly improving ranking alignment with human evaluation without modifying the metrics themselves.

## Method

### Overall Architecture

The input consists of sentence-level error correction outputs from $N$ GEC systems on the same test set, along with the sentence-level scores computed by existing automatic evaluation metrics. Traditional methods average the sentence-level scores of each system to obtain a corpus-level score, which is then sorted to produce system rankings. Conversely, the proposed method performs $N(N-1)$ pairwise comparisons of the $N$ systems' scores for each sentence (the system with the higher score "wins," and equal scores result in a "tie"). Subsequently, the pairwise comparison results across all sentences are aggregated and fed into the TrueSkill ranking algorithm to output the final system rankings. This entire approach can be seen as wrapping a layer of aggregation conversion over existing metrics, leaving the underlying metric computation logic completely unchanged.

### Key Designs

1. **Conversion from Sentence-level Scores to Pairwise Comparisons**:

    - Function: To convert absolute scores output by automatic metrics into relative comparison results consistent with the format of human evaluation.
    - Mechanism: For each input sentence, assuming there are $N$ systems yielding corrected outputs with metric scores $s_1, s_2, \ldots, s_N$, for all $N(N-1)$ ordered pairs $(i, j)$, system $i$ wins if $s_i > s_j$, a tie occurs if $s_i = s_j$, and system $j$ wins if $s_i < s_j$. This step transforms the absolute scoring problem into a relative ranking problem, perfectly aligning with how annotators perform pairwise comparisons in human evaluation.
    - Design Motivation: The distribution of absolute scores may vary significantly across different sentences (e.g., n-gram metrics fluctuate widely on short sentences), so direct averaging can be skewed by extreme values. Pairwise comparisons inherently possess a normalization effect—each comparison only cares about "which is better" rather than "by how much," making it insensitive to score scales.

2. **TrueSkill Ranking Algorithm Aggregation**:

    - Function: To estimate the "true skill" of each system from a large number of pairwise comparison results and output a global ranking.
    - Mechanism: TrueSkill is a Bayesian ranking algorithm proposed by Microsoft that maintains a skill distribution $\mathcal{N}(\mu, \sigma^2)$ for each system. Upon observing a pairwise comparison result, it adjusts the $\mu$ and $\sigma$ of the two systems using Bayesian update rules. After processing all comparisons, the final ranking is obtained by sorting systems based on $\mu$. This algorithm is naturally robust to outliers and can handle ties.
    - Design Motivation: Since TrueSkill is the aggregation method adopted by SEEDA human evaluation, the core thesis of this paper is that "automatic evaluation should employ the same aggregation method as human evaluation," hence TrueSkill is directly reused. The authors also emphasize that if human evaluation shifts to Expected Wins or other algorithms in the future, automatic evaluation should follow suit.

3. **Applicability and Generalization Design**:

    - Function: To ensure that the method can be seamlessly applied to any existing automatic GEC evaluation metric.
    - Mechanism: The method does not modify the internal computation logic of any metric, only altering the "final step" from sentence-level scores to system rankings. Whether it is edit-based ERRANT, n-gram-based GLEU+, or neural-based IMPARA, any metric can be leveraged as long as it outputs sentence-level scores. It has been integrated into the open-source library `gec-metrics`.
    - Design Motivation: To lower the barrier to adoption, allowing researchers to obtain better ranking quality without having to develop new metrics.

### Loss & Training

This paper does not involve model training; instead, it proposes an improvement to the evaluation pipeline. The core adjustment lies solely in the inference/evaluation phase: replacing "average score -> sorting" with "pairwise comparison -> TrueSkill".

## Key Experimental Results

### Main Results

Spearman correlation coefficient $\rho$ on the SEEDA benchmark (compared with human TrueSkill ranking):

| Metric | SEEDA-S Base (w/o TS) | SEEDA-S Base (w/ TS) | Gain |
|------|----------------------|---------------------|------|
| ERRANT | 0.343 | **0.706** | +0.363 |
| PT-ERRANT | 0.629 | **0.797** | +0.168 |
| GLEU+ | 0.902 | 0.846 | -0.056 |
| GREEN | 0.881 | 0.846 | -0.035 |
| SOME | 0.867 | **0.881** | +0.014 |
| IMPARA | 0.902 | **0.923** | +0.021 |
| Scribendi | 0.636 | **0.762** | +0.126 |
| GPT-4-S (fluency) | — | 0.874 | — |

### Ablation Study

Spearman $\rho$ under the SEEDA-S +Fluency setting (extended evaluation including fluency references and GPT-3.5 outputs):

| Metric | w/o TrueSkill | w/ TrueSkill | Note |
|------|--------------|-------------|------|
| IMPARA | 0.938 | **0.952** | Outperforms GPT-4-S (0.916) |
| SOME | 0.916 | **0.925** | Outperforms GPT-4-E (0.908) |
| ERRANT | -0.156 | 0.095 | Shifts from negative correlation to a weak positive correlation |
| Scribendi | 0.714 | **0.859** | Significant improvement of +0.145 |

### Key Findings

- **Edit-level metrics benefit the most**: ERRANT on SEEDA-S Base soared from $\rho=0.343$ to $0.706$, an improvement of over 0.36, indicating that traditional average aggregation severely underestimates the ranking capabilities of edit-level metrics.
- **N-gram metrics are unsuitable**: GLEU+ and GREEN actually declined when using TrueSkill. This is because the sentence-level scores of these metrics are of poor quality (e.g., brevity penalty is unstable on short sentences, and the n-gram geometric mean is highly sensitive to short sentences), failing to support the accuracy of pairwise comparisons.
- **BERT-level metrics can outperform GPT-4**: IMPARA achieved $\rho=0.952$ under the +Fluency setting, outperforming GPT-4's 0.916. This demonstrates that with an appropriate aggregation pipeline, lightweight BERT-based metrics are fully capable of matching or even surpassing LLM-based evaluation.
- **Window analysis**: IMPARA aligns exceptionally well in low-ranking system zones, whereas ERRANT, despite overall improvements, still struggles with top-ranking systems (including those with heavily rewritten outputs from GPT-3.5).

## Highlights & Insights

- **Extremely simple yet remarkably effective**: The entire method merely wraps a TrueSkill ranking layer over the sentence-level scores of existing metrics. Without altering the metrics themselves, it significantly enhances ranking quality. This approach of "fixing the evaluation pipeline rather than inventing new metrics" is highly practical.
- **Reveals a long-overlooked systematic issue**: For years, the GEC community has conducted meta-evaluation using an aggregation method inconsistent with human evaluation, resulting in systematic underestimation of existing metrics. This finding could drive a paradigm shift in GEC evaluation.
- **Inspires other NLG tasks**: Similar "aggregation discrepancy" issues may also exist in the meta-evaluations of other NLG tasks, such as machine translation and summarization. The proposed methodology can be directly migrated.

## Limitations & Future Work

- **Only applicable to system rankings**: The method cannot analyze the specific strengths and weaknesses of individual systems (such as precision vs. recall); such fine-grained analyses still rely on traditional corpus-level aggregation.
- **Requires raw outputs from all systems**: TrueSkill requires pairwise comparisons among all compared systems, meaning researchers cannot directly cite scores reported in prior literature and must reproduce the outputs of all systems.
- **Underlying issues of n-gram metrics remain unresolved**: GLEU+/GREEN deteriorated after adopting TrueSkill, indicating inherent flaws in their sentence-level scoring quality, which this paper does not attempt to resolve.
- **Validated active only on a single benchmark (SEEDA)**: Generalization needs to be validated across a wider variety of meta-evaluation datasets (e.g., the Expected Wins setting of CoNLL-2014).

## Related Work & Insights

- **vs Kobayashi et al. (2024a)**: Although they also employed TrueSkill, it was restricted to their proposed LLM metric while other baselines still relied on traditional aggregation, leading to an unfair comparison. This work unifies the aggregation method across all metrics.
- **vs GPT-4 Evaluation**: GPT-4 evaluation inherently utilizes TrueSkill (being sentence-level pairwise comparisons), but it is cost-prohibitive. This study proves that inexpensive BERT-level metrics can achieve comparable results once their aggregation processes are aligned.
- **Implications for NLG Evaluation**: Similar pipeline discrepancies might be widespread across tasks like machine translation (MT) and summarization, warranting systematic investigation.

## Rating
- Novelty: ⭐⭐⭐ The method itself is a simple modification of "using TrueSkill instead of averaging," which offers limited technical novelty but strong insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 7 metrics, 4 SEEDA configurations, window analysis, and other multi-dimensional validations.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, diagrams are intuitive, and conclusions are explicit.
- Value: ⭐⭐⭐⭐ Uncovers a long-ignored systematic bias in GEC evaluation, holding direct practical value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] gec-metrics: A Unified Library for Grammatical Error Correction Evaluation](gec-metrics_a_unified_library_for_grammatical_error_correction_evaluation.md)
- [\[ACL 2025\] IMPARA-GED: Grammatical Error Detection is Boosting Reference-free Grammatical Error Quality Estimator](impara-ged_grammatical_error_detection_is_boosting_reference-free_grammatical_er.md)
- [\[ACL 2025\] Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](enhancing_text_editing_for_grammatical_error_correction_arabic_as_a_case_study.md)
- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] A Representation Level Analysis of NMT Model Robustness to Grammatical Errors](a_representation_level_analysis_of_nmt_model_robustness_to_grammatical_errors.md)

</div>

<!-- RELATED:END -->
