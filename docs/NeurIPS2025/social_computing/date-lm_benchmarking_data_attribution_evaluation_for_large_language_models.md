---
title: >-
  [Paper Note] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models
description: >-
  [NeurIPS 2025][Social Computing][data attribution] DATE-LM introduces the first unified benchmark for evaluating data attribution methods in LLMs. Through three application-driven tasks—training data selection…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "data attribution"
  - "benchmark"
  - "LLM"
  - "data selection"
  - "factual provenance"
date: 2026-05-08
content_hash: 233990f04fa09f77
---

# DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2507.09424](https://arxiv.org/abs/2507.09424)  
**Code**: [GitHub](https://github.com/DataAttributionEval/DATE-LM)  
**Area**: Social Computing / Data Attribution
**Keywords**: data attribution, benchmark, LLM, data selection, factual provenance

## TL;DR
DATE-LM introduces the first unified benchmark for evaluating data attribution methods in LLMs. Through three application-driven tasks—training data selection, toxicity filtering, and factual attribution—it systematically compares multiple attribution approaches, finding that no single method dominates across all tasks and that simple baselines can match attribution methods in certain settings.

## Background & Motivation

**Background** Data attribution methods quantify the influence of training data on model outputs, and have become increasingly important for dataset curation, model interpretability, and data valuation. LLM-specific methods such as LESS and MATES have emerged in recent years.

**Limitations of Prior Work** The complexity of LLM training stacks makes fair comparison across methods difficult; retraining-based evaluation protocols incur prohibitive computational costs; existing work lacks comprehensive application-driven evaluation and often omits comparisons against non-attribution baselines.

**Key Challenge** The practical utility of data attribution methods at LLM scale remains unclear—their cost-effectiveness across diverse applications and whether the high computational overhead is justified are open questions.

**Goal** To provide a unified and scalable benchmarking platform enabling fair, reproducible comparison of data attribution methods across multiple LLM architectures and tasks.

**Key Insight** Design a modular three-stage evaluation pipeline (attribution scoring → subset selection → task evaluation), paired with released pre-training checkpoints and a public leaderboard to lower the barrier to evaluation.

**Core Idea** Through a unified pipeline and application-driven task design, enable the first large-scale, systematic comparison of data attribution methods for LLMs.

## Method

### Overall Architecture
DATE-LM comprises a unified three-stage pipeline: (1) **Attribution Scoring**—method $\tau$ scores each sample in training set $\mathcal{D}$ with respect to reference set $\mathcal{D}_{ref}$, yielding $\mathcal{S} = \tau(\mathcal{D}, \mathcal{D}_{ref}, \theta)$; (2) **Subset Selection**—selects subset $\mathcal{D}_s$ from $\mathcal{S}$ via top-k or probabilistic sampling; (3) **Task Evaluation**—runs downstream tasks on $\mathcal{D}_s$ and computes metrics. The pipeline is agnostic to $\tau$; new methods need only provide scores to be integrated.

### Key Designs

1. **Training Data Selection Task**:
    - Function: Evaluate how well attribution methods identify high-quality training data to improve LLM capabilities.
    - Mechanism: The pre-training setting uses FineWeb + LAMBADA with Gumbel-top-k sampling to ensure diversity, evaluated on 7 tasks including SciQ, ARC, and BoolQ; the fine-tuning setting uses Tulu 3 with MMLU/GSM8K/BBH. Short training runs (200-step WSD schedule) replace full pre-training to reduce cost.
    - Design Motivation: Uniform adoption of Gumbel-top-k isolates the effect of the attribution scoring function itself, preventing confounds from different selection strategies (clustering, multi-armed bandits, etc.).

2. **Toxicity/Bias Filtering Task**:
    - Function: Evaluate how well attribution methods detect harmful samples in the training set.
    - Mechanism: Constructs $\mathcal{D} = \mathcal{D}_{benign} \cup \mathcal{D}_{unsafe}$ (10K benign + <100 harmful) and evaluates with AUPRC. Introduces **heterogeneous filtering**—inserting safety-aligned "distractor" samples (e.g., refusals to harmful queries) into the benign data, making harmful samples harder to distinguish.
    - Design Motivation: Homogeneous filtering is overly simplistic; the heterogeneous setting better reflects real-world safety alignment scenarios, providing a more rigorous and meaningful test.

3. **Factual Attribution Task**:
    - Function: Trace factual content in LLM outputs back to supporting evidence in the training data.
    - Mechanism: Based on the ROME dataset, evaluated with Recall@50 and MRR. Introduces a **counterfactual setting**—replacing entities in supporting evidence with related but incorrect ones (e.g., Microsoft→Google) to break lexical overlap and force methods to capture semantic contributions.
    - Design Motivation: Prior benchmarks are heavily biased toward lexical overlap detection, allowing BM25 to achieve high scores trivially; the counterfactual setting enables fairer evaluation.

## Key Experimental Results

### Main Results — Pre-training Data Selection (1B model, 30K steps)

| Method | Avg. (7 tasks) | FLOPS |
|--------|---------------|-------|
| Random | 49.83 | 1× |
| BM25 | 50.26 | 1× |
| Grad Sim | 50.26 | 11× |
| MATES | 50.13 | 1.13× |
| EDU | **50.63** | 1.07× |

### Toxicity Filtering (Average AUPRC)

| Method | Homogeneous | Heterogeneous |
|--------|------------|---------------|
| WildGuard (baseline) | **0.827** | **0.817** |
| LESS | 0.704 | 0.515 |
| Grad Sim | 0.584 | 0.466 |

### Ablation Study — Gumbel Temperature Sensitivity

| Temperature | Effect |
|-------------|--------|
| Too low (0.1) | Insufficient diversity; some methods fall below Random |
| Moderate (0.5–1.0) | Most methods achieve optimal performance |
| Too high (2.0) | Excessive randomness; degenerates to uniform sampling |

### Key Findings
- No single attribution method dominates across all tasks.
- Simple non-attribution baselines (EDU, WildGuard) match or outperform attribution methods on multiple tasks.
- Attribution method performance drops sharply under heterogeneous filtering (e.g., LESS declines from 0.704 to 0.515).
- Evaluation design choices (Gumbel temperature, reference set composition) have a substantial impact on results.

## Highlights & Insights
- The first unified LLM data attribution benchmark with a public leaderboard, including a complete evaluation pipeline and released pre-training checkpoints.
- Counterfactual factual attribution and heterogeneous toxicity filtering reveal systematic biases in prior evaluations.
- The cost-effectiveness analysis of high-cost attribution methods versus low-cost baselines carries important practical implications.

## Limitations & Future Work
- Evaluation is limited to 1B–8B models; conclusions may differ at larger scales.
- Short-training proxies may not reflect the effects of long-horizon training.
- Fine-tuning evaluation considers only single-task settings; multi-task scenarios remain to be explored.

## Related Work & Insights
- **vs. TRAK/LDS**: DATE-LM replaces costly leave-one-out evaluation with application-driven evaluation, which is more practically grounded but represents a different evaluation perspective.
- **vs. LESS**: LESS performs above average but not comprehensively best on this benchmark, highlighting the need for more diverse evaluation perspectives.
- **vs. Quanda**: Quanda targets general ML; DATE-LM is specifically designed for LLMs and lowers the barrier to entry by providing released checkpoints.

## Rating
- Novelty: ⭐⭐⭐⭐ First comprehensive LLM data attribution benchmark; counterfactual setting and heterogeneous filtering are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Eight methods, three major tasks, multiple model scales; rigorous and comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated design choices.
- Value: ⭐⭐⭐⭐ Provides a much-needed standardized evaluation tool and important empirical insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [\[ACL 2026\] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](../../ACL2026/social_computing/smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)
- [\[ACL 2026\] Inertia in Moral and Value Judgments of Large Language Models](../../ACL2026/social_computing/inertia_in_moral_and_value_judgments_of_large_language_models.md)
- [\[NeurIPS 2025\] Any Large Language Model Can Be a Reliable Judge: Debiasing with a Reasoning-based Bias Detector](any_large_language_model_can_be_a_reliable_judge_debiasing_w.md)

</div>

<!-- RELATED:END -->
