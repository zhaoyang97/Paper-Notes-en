---
title: >-
  [Paper Note] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models
description: >-
  [ICCV 2025][Time Series][reward model] This paper proposes VLRMBench, a comprehensive and challenging benchmark for vision-language reward models (VLRMs) comprising 12,634 questions across 12 tasks…
tags:
  - "ICCV 2025"
  - "Time Series"
  - "reward model"
  - "vision-language understanding"
  - "benchmark"
  - "process reasoning"
  - "multimodal evaluation"
date: 2026-05-08
content_hash: a54cd931b221df05
---

# VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models

**Conference**: ICCV 2025
**arXiv**: [2503.07478](https://arxiv.org/abs/2503.07478)  
**Code**: [https://github.com/JCruan519/VLRMBench](https://github.com/JCruan519/VLRMBench)  
**Area**: Time Series / Multimodal Evaluation
**Keywords**: reward model, vision-language understanding, benchmark, process reasoning, multimodal evaluation

## TL;DR

This paper proposes VLRMBench, a comprehensive and challenging benchmark for vision-language reward models (VLRMs) comprising 12,634 questions across 12 tasks, covering three dimensions: process understanding, outcome judgment, and criticism generation. Extensive experiments on 26 models reveal significant deficiencies in current VLRMs.

## Background & Motivation

Reward models (RMs) play a critical role in both the training and inference stages of large models: they are used to filter high-quality data before training, to guide preference optimization during training (e.g., RLAIF), and to enable test-time scaling (TTS) at inference. However, existing VLRM benchmarks suffer from serious limitations:

**Narrow evaluation dimensions**: VLRewardBench covers only pairwise comparison, which is insufficient for a comprehensive assessment of VLRM capabilities.

**Absence of step-level annotations**: Most benchmarks do not include labels at the level of individual reasoning steps.

**Focus on language-only settings**: Existing RM benchmarks (e.g., PRMBench, ProcessBench) target purely textual inputs and are not applicable to vision-language scenarios.

**Insufficient challenge**: Overly simple benchmarks fail to expose latent weaknesses of VLRMs.

## Method

### Overall Architecture

The construction of VLRMBench follows a three-stage pipeline: (1) data collection and filtering; (2) reasoning process generation and step segmentation; (3) task design based on three overarching themes, yielding 12 tasks in total.

### Key Designs

1. **Collaborative Data Filtering & Generation Pipeline**:

    - **Quality filtering**: Qwen2VL-7B is prompted to answer questions without images; samples answered correctly are considered low-quality (i.e., solvable by text alone) and discarded.
    - **Difficulty filtering**: Samples that Qwen2VL-7B answers correctly even with images are deemed too easy and discarded. This reduces the pool from 16,550 to 6,715 samples.
    - **Reasoning process generation**: QVQ-72B-preview generates reasoning chains; only samples with correct final answers are retained.
    - **Step segmentation**: GPT-4o performs semantic-level step segmentation.
    - **Human verification**: Three doctoral students validate and correct errors in the generated reasoning processes.
    - The final dataset retains 1,000 high-quality samples spanning mathematical reasoning, hallucination understanding, and multi-image understanding.

2. **Step-based Tasks (8 tasks)**: Evaluate a VLRM's ability to understand reasoning processes.

    - **SC (Step Correctness)**: Detects injected errors in reasoning steps.
    - **RD (Redundancy Detection)**: Identifies redundant information in reasoning processes.
    - **CM (Confidence Misdirection)**: Inserts high-confidence expressions into erroneous steps to test robustness.
    - **EH (Existential Hallucination)**: Detects entities mentioned in reasoning that do not exist in the image.
    - **AH (Attribute Hallucination)**: Detects incorrect descriptions of entity attributes.
    - **DE (Detail Error)**: Detects fine-grained errors in numerical computations or symbols.
    - **SR (Spatial Relationship)**: Detects errors in spatial relationship descriptions.
    - **IC (Image Confusion)**: Detects incorrect image references in multi-image tasks.

3. **Outcome-based Tasks (2 tasks)**: Evaluate a VLRM's ability to judge final outcomes.

    - **MJ (Multi-solution Judgment)**: Compares the quality of different reasoning processes for the same question.
    - **FF (Forecasting Future)**: Predicts the correctness of the final answer based on the first $m$ reasoning steps.

4. **Criticism-based Tasks (2 tasks)**: Evaluate a VLRM's ability to analyze and correct errors.

    - **ERA (Error Reason Analysis)**: Analyzes the cause of erroneous reasoning steps.
    - **EC (Error Correction)**: Directly corrects errors and produces a revised reasoning chain.

### Loss & Training

As a benchmarking study, no model training is required. Evaluation metrics include:
- Step-based tasks: F1-Score (balancing precision and recall)
- Outcome-based tasks: Accuracy
- Criticism-based tasks: Win Rate (using GPT-4o as the judge)

## Key Experimental Results

### Main Results

**Average F1-Score on step-based tasks (selected models)**:

| Model | SC | RD | CM | EH | AH | DE | Step Avg. |
|------|-----|-----|-----|-----|-----|-----|---------|
| GPT-4o | 73.7 | 50.6 | 66.6 | 57.6 | 58.6 | 71.8 | 62.4 |
| Claude-3.5-Sonnet | 70.8 | 53.7 | 65.7 | 63.9 | 62.8 | 63.4 | 62.9 |
| Qwen2.5VL-72B | 72.8 | 41.7 | 70.4 | 64.6 | 59.9 | 72.4 | 62.6 |
| Qwen2.5VL-7B | 43.4 | 33.2 | 37.8 | 22.8 | 23.9 | 45.5 | 33.4 |
| InternVL2.5-8B | 36.6 | 28.4 | 31.1 | 21.9 | 21.2 | 36.5 | 28.6 |
| Ovis2-34B | 65.3 | 51.1 | 64.5 | 54.5 | 51.6 | 59.6 | 57.0 |

**Performance on outcome-based and criticism-based tasks**:

| Model | MJ(Acc) | FF(Acc) | Outcome Avg. | ERA(WinRate) | EC(WinRate) |
|------|---------|---------|---------|-------------|-------------|
| GPT-4o | 58.4 | 76.0 | 66.3 | 0.0 | 0.0 |
| Claude-3.5-Sonnet | 82.2 | 75.1 | 79.0 | 60.6/25.5/13.9 | 21.2/53.9/24.9 |
| Qwen2.5VL-72B | 65.6 | 80.2 | 72.1 | 74.1/15.1/10.8 | 15.6/77.0/7.3 |
| Qwen2.5VL-7B | 26.0 | 70.7 | 46.0 | 37.7/22.0/40.3 | 9.3/51.9/38.8 |

### Ablation Study

**Effect of model scale on step-based and outcome-based tasks**:

| Group | Scale | Step Avg. | Outcome Avg. |
|--------|------|---------|---------|
| Small (<10B) | 2B–8B | 29.8 | 45.2 |
| Medium (10–40B) | 11B–38B | 45.9 | 54.7 |
| Large (>40B) | 72B–90B | 46.1 | 56.8 |
| Closed-source | — | 62.4+ | 66.3+ |

### Key Findings

- **Even the state-of-the-art GPT-4o achieves only 76.0% accuracy on the FF (Forecasting Future) task** and an average F1 of 62.4% on step-based tasks.
- **Open-source models are closing the gap with closed-source counterparts**: Qwen2.5VL-72B outperforms GPT-4o on criticism tasks (ERA win rate 74.1% vs. 0.0%).
- **The CM task confirms that VLRMs are susceptible to high-confidence expressions**: CM F1 scores are consistently lower than SC scores across all models.
- **Redundancy Detection (RD) is the most challenging step-based task**: all models record their lowest F1 scores on RD.
- Performance improves substantially from small to medium scale (29.8→45.9), but the gain from medium to large scale is marginal (45.9→46.1).

## Highlights & Insights

- **Comprehensive 12-task design**: The three-dimensional evaluation framework—covering process, outcome, and criticism—substantially surpasses existing benchmarks that rely solely on pairwise comparison.
- **Dual filtering mechanism**: Answering correctly without images indicates low quality; answering correctly with images indicates insufficient difficulty. This ensures that only high-quality, high-difficulty samples are retained.
- **Innovation of the Confidence Misdirection task**: Tests whether models are misled by confidence-laden expressions such as "definitely" and "without a doubt."
- **Practical value of Forecasting Future**: The ability to predict reasoning correctness early could significantly accelerate TTS inference.

## Limitations & Future Work

- Reasoning processes are generated by QVQ-72B-preview, which may introduce biases inherent to that model.
- Mathematical reasoning samples dominate the dataset; hallucination and multi-image samples are comparatively scarce.
- Using GPT-4o as the judge for criticism-based tasks introduces a risk of evaluator bias.
- The benchmark currently evaluates only text-form reward models and does not consider scalar-valued RMs.
- The absence of a dynamic update mechanism means models may overfit to a fixed test set over time.

## Related Work & Insights

- PRMBench: Fine-grained evaluation of process reward models in the text-only domain.
- VLRewardBench: The first vision-language RM benchmark, but limited to a single task type.
- ProcessBench: Mathematical reasoning evaluation for step-level error detection.
- Insight: Evaluating the capabilities of reward models requires a multi-dimensional framework such as VLRMBench; single-task evaluation (e.g., pairwise comparison) is far from sufficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First comprehensive VLRM benchmark with a uniquely designed set of 12 tasks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extensive evaluation across 26 models, analyzed in four groups (small/medium/large/closed-source).
- **Writing Quality**: ⭐⭐⭐⭐ Task designs are clearly articulated, with rich tables and figures.
- **Value**: ⭐⭐⭐⭐ Fills the gap in comprehensive VLRM evaluation and exposes critical weaknesses of current models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/time_series/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[NeurIPS 2025\] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](../../NeurIPS2025/time_series/causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)
- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](../../ICLR2026/time_series/timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[NeurIPS 2025\] NSW-EPNews: A News-Augmented Benchmark for Electricity Price Forecasting with LLMs](../../NeurIPS2025/time_series/nsw-epnews_a_news-augmented_benchmark_for_electricity_price_forecasting_with_llm.md)

</div>

<!-- RELATED:END -->
