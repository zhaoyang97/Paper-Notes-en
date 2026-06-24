---
title: >-
  [Paper Note] HalluLens: LLM Hallucination Benchmark
description: >-
  [ACL 2025][Hallucination Detection][Hallucination Evaluation] Proposes HalluLens, a hallucination benchmark that clearly distinguishes hallucination from factuality, establishes a clear taxonomy of extrinsic hallucination (inconsistency with training data) and intrinsic hallucination (inconsistency with input context), introduces three dynamically regenerable extrinsic hallucination evaluation tasks, and comprehensively analyzes the limitations of existing benchmarks.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Hallucination Evaluation"
  - "Extrinsic Hallucination"
  - "Intrinsic Hallucination"
  - "Dynamic Test Set"
  - "Benchmark"
date: 2026-05-08
content_hash: 0ccf02087ff5ce29
---

# HalluLens: LLM Hallucination Benchmark

**Conference**: ACL 2025  
**arXiv**: [2504.17550](https://arxiv.org/abs/2504.17550)  
**Code**: [github](https://github.com/facebookresearch/HalluLens)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Evaluation, Extrinsic Hallucination, Intrinsic Hallucination, Dynamic Test Set, Benchmark

## TL;DR

Proposes HalluLens, a hallucination benchmark that clearly distinguishes hallucination from factuality, establishes a clear taxonomy of extrinsic hallucination (inconsistency with training data) and intrinsic hallucination (inconsistency with input context), introduces three dynamically regenerable extrinsic hallucination evaluation tasks, and comprehensively analyzes the limitations of existing benchmarks.

## Background & Motivation

LLM hallucination is a core obstacle limiting their widespread application, but existing research suffers from severe conceptual confusion and insufficient evaluation:

**Inconsistent Definitions**: Existing taxonomies (Huang et al., 2023; Zhang et al., 2023) conflate hallucination with factuality. Factuality concerns whether the generated content conforms to real-world knowledge, whereas hallucination should focus on whether the generated content is consistent with the model's training data or input context. These two require different evaluation and mitigation strategies.

**Neglected Extrinsic Hallucinations**: Existing benchmarks primarily focus on intrinsic hallucinations (such as unfaithfulness in document summarization), while the evaluation of extrinsic hallucinations (inconsistency between generated content and training data) remains virtually unaddressed. As LLMs increasingly generate free-form text based on task instructions, extrinsic hallucinations become even more critical.

**Benchmark Saturation due to Data Contamination**: Static test sets are easily incorporated into the training data of subsequent models, leading to artificially inflated benchmark scores. Widely used benchmarks like TruthfulQA have already suffered from severe contamination.

**Issues with TruthfulQA**: Approximately 25% of the samples rated as incorrect by MC1 could actually be correct; there are also issues with outdated answers, subjective questions, and inaccurate gold-standard answers.

## Method

### Overall Architecture

HalluLens consists of two parts: (a) **newly introduced extrinsic hallucination evaluations**, containing three dynamically generated tasks; (b) **integrated intrinsic hallucination evaluations**, selecting three unsaturated existing benchmarks.

### Key Designs

1. **Hallucination Taxonomy**:

    - **Extrinsic Hallucinations**: Generated content is inconsistent with the training data, where the model attempts to fill in knowledge gaps.
    - **Intrinsic Hallucinations**: Generated content is inconsistent with the input context, where the model fails to correctly understand the input.
    - **Factuality** (excluded): Correctness issues that require verification from external knowledge sources, which do not fall under the category of hallucination.
    - Key Distinction: If real-world changes cause information in the training data to be outdated, the model answering based on the training data is not considered a hallucination.

2. **PreciseWikiQA Task**: Evaluates the model's extrinsic hallucination rate in short-form question answering.

    - Dynamically generates 5,000 QA pairs from the GoodWiki dataset (composed of 44,754 high-quality Wikipedia pages).
    - Uses harmonic centrality to control difficulty (10 levels), with 500 pages per level.
    - Three metrics: False Refusal Rate, Hallucination Rate given Non-Refusal, and overall Accuracy.
    - The automatically generated gold-standard answers are 97.2% correct.

3. **LongWiki Task**: Evaluates extrinsic hallucinations in long-form text generation.

    - Dynamically generates 250 paragraph-level questions (difficulty levels 5-9, avoiding long-tail knowledge).
    - Evaluation Pipeline: Claim extraction → Reference evidence selection (Wikipedia page retrieval) → Claim verification.
    - Metrics: Precision, Recall@32, F1@32.

4. **NonExistentRefusal Task**: Evaluates the model's tendency to hallucinate when facing non-existent entities.

    - MixedEntities subtask: Generates non-existent names by mixing real plant, animal, or drug names (8,000 samples).
    - GeneratedEntities subtask: LLMs take turns generating fictional businesses, events, or brand names (1,950 samples).
    - Metric: False Acceptance Rate (the lower, the better).

### Loss & Training

This work is an evaluation benchmark rather than a training method. The core design principles are:
- **Dynamic Test Set**: Questions are regenerated for each evaluation to prevent data contamination.
- **Reproducibility**: Harmonic centrality is used to control the difficulty distribution, ensuring stable results across different versions of the test set.
- **Automated Evaluation**: LLaMA-3.1-70B-Instruct is used as the judge (96.67% accuracy for refusal determination, and 95.56% accuracy for correctness determination).

## Key Experimental Results

### Main Results

**PreciseWikiQA (13 Models)**:

| Model | False Refusal Rate | Hallucination Rate (Non-Refusal) | Accuracy |
|------|----------|------------|-------|
| GPT-4o | 4.13% | 45.15% | **52.59%** |
| Llama-3.1-405B | 56.77% | **26.84%** | 31.62% |
| Llama-3.3-70B | 20.01% | 50.19% | 39.84% |
| Qwen2.5-7B | 13.85% | 85.22% | 12.73% |
| Mistral-7B | 7.77% | 81.19% | 17.34% |

**LongWiki**:

| Model | F1@32 | Precision | Recall@32 |
|------|-------|-----------|-----------|
| GPT-4o | **75.80** | **71.03** | **84.89** |
| Llama-3.1-405B | 61.98 | 56.94 | 74.44 |
| Qwen2.5-14B | 60.11 | 52.84 | 74.05 |

**NonExistentRefusal (False Acceptance Rate, lower is better)**:

| Model | MixedEntities | GeneratedEntities | Average |
|------|-------------|-------------------|-----|
| Llama-3.1-405B | **11.48%** | **2.28%** | **6.88%** |
| Llama-3.1-8B | 19.78% | 6.58% | 13.18% |
| GPT-4o | 65.89% | 18.74% | 42.31% |
| Mistral-7B | 94.74% | 77.98% | 86.36% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Different Difficulty Levels (PreciseWikiQA) | Refusal rate is highest for long-tail knowledge | Llama/Claude are more inclined to refuse on long-tail knowledge |
| Different Location Frequencies (NonExistent) | Most hallucinations occur in mid-frequency locations | Near the knowledge boundary, where model uncertainty is highest |
| TruthfulQA Misjudgment Analysis | ~25% misjudgment | The log-probability summation method of MC1 has serious flaws |
| Dynamic Test Set Stability | <1.01% standard deviation | The model rankings across three runs of PreciseWikiQA remain consistent |

### Key Findings

1. **Refusal-Hallucination Trade-off**: Llama-3.1-405B achieves the lowest hallucination rate (26.84%) but suffers from the highest refusal rate (56.77%). Conversely, GPT-4o exhibits fewer refusals but higher hallucination rates, resulting in GPT-4o leading in overall accuracy.
2. **Inconsistent Scaling Law across Model Families**: Larger models within the same family typically outperform smaller ones, but this pattern does not consistently hold across different families (e.g., Gemma-2-9B performs closely to Qwen2.5-14B).
3. **Changes in Llama-3.3-70B**: Compared to Llama-3.1-70B, the refusal rate drops significantly (52% → 20%), while the hallucination rate increases (37% → 50%). This indicates that instruction tuning strategies significantly impact hallucination behavior.
4. **TruthfulQA is No Longer Reliable**: Findings show incorrect gold-standard answers, systematic biases in the evaluation method (MC1 log-probabilities), and outdated time-sensitive questions.

## Highlights & Insights

- **Enormous Contribution in Conceptual Clarification**: Clarifies the distinction between hallucination and factuality for the first time, arguing that they require distinct benchmarks and mitigation strategies. This conceptual framework is highly valuable for guiding future research in this field.
- **Ingenious Dynamic Test Set Design**: Balances leakage prevention and reproducibility through difficulty control using harmonic centrality, dynamic question generation, and automated evaluation pipelines.
- **In-depth Criticism of TruthfulQA**: Unveils around 25% misjudgments through sample-by-sample analysis, providing an empirical basis for the community to re-evaluate existing benchmarks.
- **Clever Design of NonExistentRefusal**: Tests whether models know "what they do not know", directly probing the essence of hallucination.

## Limitations & Future Work

1. **Assumptions in Extrinsic Hallucination Verification**: Assumes Wikipedia was included in the training data of all evaluated models, which might not be completely true for some models.
2. **Lack of Dynamic Test Sets for Intrinsic Hallucinations**: The authors acknowledge that creating dynamic test sets for intrinsic hallucinations remains an open question.
3. **Limited Evaluation Scope**: Only covers text-based hallucinations, excluding multimodal hallucinations.
4. **Judge Model Bias**: Using LLaMA-3.1-70B as a judge might introduce systematic biases.
5. **Domain Bias in NonExistentRefusal**: Gemma models refuse to answer all drug-related questions, failing to distinguish whether they exist or not.

## Related Work & Insights

- **SimpleQA (Wang et al., 2024)**: A factuality benchmark that can be adapted as an extrinsic hallucination benchmark by modifying the evaluation metrics.
- **FaithEval (Ming et al., 2024)**: Evaluates intrinsic hallucinations within noisy or counterfactual contexts.
- **ANAH 2.0 (Gu et al., 2024)**: Evaluates intrinsic hallucinations within factually accurate input contexts.
- **Insights**: The design of evaluation benchmarks should explicitly define the "reference source" (training data vs. external knowledge vs. input context), as different reference sources correspond to distinct types of problems.

## Rating

- Novelty: ⭐⭐⭐ Tasonomy and the design of extrinsic hallucination evaluation tasks are novel, and the analysis of TruthfulQA is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 13 models, three new tasks, three existing benchmark analyses, and stability validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual clarification, rich figures and tables, and rigorous argumentation.
- Value: ⭐⭐⭐⭐⭐ Establishes a clear taxonomic framework and new evaluation standards for hallucination research, providing important guidance for the development of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Rethinking Evaluation for LLM Hallucination Detection: A Desiderata, A New RAG-based Benchmark, New Insights](../../ACL2026/hallucination/rethinking_evaluation_for_llm_hallucination_detection_a_desiderata_a_new_rag-bas.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models](cchall_a_novel_benchmark_for_joint_cross-lingual_and_cross-modal_hallucinations_.md)
- [\[ACL 2025\] Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge](correcting_hallucinations_in_news_summaries_exploration_of_self-correcting_llm_m.md)
- [\[ICML 2025\] Steer LLM Latents for Hallucination Detection](../../ICML2025/hallucination/steer_llm_latents_for_hallucination_detection.md)

</div>

<!-- RELATED:END -->
