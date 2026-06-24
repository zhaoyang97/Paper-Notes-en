---
title: >-
  [Paper Note] How Much Can We Forget about Data Contamination?
description: >-
  [ICML 2025][LLM Evaluation][Data contamination] This work systematically quantifies the impact of data contamination on LLM benchmark evaluation through controlled experiments. It finds that when trained on more than five times the Chinchilla-optimal data volume, even contaminated data repeated 144 times can be completely forgotten. It further demonstrates that weight decay is the key mechanism driving forgetting, leading to the inference that large models like Llama 3 405B h…
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Data contamination"
  - "forgetting"
  - "benchmark overfitting"
  - "Chinchilla scaling"
  - "weight decay"
  - "AdamW"
date: 2026-05-08
content_hash: 78df06b2c3ea50b8
---

# How Much Can We Forget about Data Contamination?

**Conference**: ICML 2025  
**arXiv**: [2410.03249](https://arxiv.org/abs/2410.03249)  
**Code**: [GitHub](https://github.com/tml-tuebingen/forgetting-contamination/)  
**Area**: LLM Evaluation  
**Keywords**: Data contamination, forgetting, benchmark overfitting, Chinchilla scaling, weight decay, AdamW, LLM evaluation

## TL;DR

This work systematically quantifies the impact of data contamination on LLM benchmark evaluation through controlled experiments. It finds that when trained on more than five times the Chinchilla-optimal data volume, even contaminated data repeated 144 times can be completely forgotten. It further demonstrates that weight decay is the key mechanism driving forgetting, leading to the inference that large models like Llama 3 405B have already forgotten the data from their early training stages.

## Background & Motivation

One of the core principles of machine learning is that models should not be trained on test sets. However, training data for large language models is often scraped from the internet, inevitably containing benchmark evaluation data—namely, data contamination. Overlaps between training data and benchmarks have been reported in models such as GPT-3 and Llama 3.

Key gaps remain in the current understanding of data contamination:

1. **Whether small-scale contamination inevitably invalidates evaluation** remains unclear—modern models train for over millions of gradient steps, and using contaminated data at a certain step during training does not definitely affect the final evaluation.
2. **Memorization research** shows that samples need to be repeated multiple times to be remembered, and **knowledge acquisition research** also finds that facts need to be paraphrased multiple times to be learned.
3. The size of training data has far exceeded Chinchilla-optimal levels (for example, Llama 3 70B uses more than 10 times Chinchilla data), but the contamination literature has rarely considered the dilution effect of this data abundance on contamination.

The core problem of this paper: **Under what conditions does data contamination indeed invalidate benchmark evaluation?**

## Method

### Overall Architecture

A controlled experimental design is adopted: language models (up to 1.6B parameters) are trained from scratch, with benchmark questions explicitly inserted into the training data. Scaling experiments are conducted along three dimensions:

- **Model Parameters**: 124M → 350M → 774M → 1.6B
- **Training Tokens**: 1× → 15× Chinchilla
- **Contamination Repetitions**: 4 → 12 → 32 → 144 times

The training data consists of FineWeb-Edu 100BT, mixed with seven benchmarks (ARC-Easy, SocialIQA, WinoGrande, PiQA, BoolQ, MMLU, HellaSwag).

### Contamination Insertion Method

- Benchmark questions are divided into multiple subsets (2,000–10,000 questions).
- A holdout set of 10,000 questions is reserved and never added to the training data.
- Other subsets are randomly inserted into the training data with different repetition counts (4/12/32/144).
- Exact contamination is adopted—the training data is completely identical to the evaluation data.
- Near-duplicate questions in benchmarks are filtered out (based on Levenshtein distance).

### Forgetting Experimental Design

In the 15× Chinchilla training, contamination is concentrated between the 1st and 2nd Chinchilla intervals of tokens. Subsequently, the decay of the cross-entropy loss difference is observed during further training.

Key Variants:
- Continued training on new data vs. repeated training on a fixed set of 100M tokens
- The impact of contamination data distribution: early, middle, late, or uniformly distributed throughout training

### Theoretical Analysis of Weight Decay and Forgetting

Parameter updates of the AdamW optimizer can be decomposed into weight decay updates and gradient updates. By iterative expansion, the final model weights can be represented as:

$$\theta_T = w_0^T \theta_0 - \sum_{t=1}^T w_t^T \gamma_t \hat{g}_t$$

where the cumulative weight decay is:

$$w_{t_1}^{t_2} = \prod_{i=t_1+1}^{t_2} (1 - \gamma_i \lambda)$$

**Proposition 1**: When $T \geq \frac{\log(1/\epsilon)}{\lambda \gamma_{\text{avg}}}$, $w_{t_1}^{t_2} \leq \epsilon$.

Implication: After a sufficient number of gradient steps, the contribution of early gradient updates to the final model weights approaches zero.

## Key Experimental Results

### Three-Dimensional Scaling Experiment

Absolute accuracy of Chinchilla-optimal models (holdout vs. contaminated):

| Model | Holdout | 4× | 12× | 32× | 144× |
|---|---|---|---|---|---|
| 124M | 42.22 | 48.14 | 56.92 | 80.70 | 96.45 |
| 350M | 44.72 | 55.69 | 69.90 | 89.20 | 95.50 |
| 774M | 49.16 | 64.76 | 81.30 | 92.95 | 96.05 |
| 1.6B | 52.06 | 67.61 | 82.32 | 91.85 | 95.40 |

Key Findings:
- Overfitting increases as the **parameter size increases** (from 124M to 1.6B, the accuracy gap for 4× contamination grows from 6 to 15 percentage points).
- Overfitting decreases as the **number of training tokens increases**—under 15× Chinchilla, the impact of 12-repetition contamination completely vanishes.
- Overfitting increases as the **number of repetitions increases**.

### Forgetting Experiments

- After inserting 144× contamination concentrated between the 1st and 2nd Chinchilla intervals and continuing training on 5 Chinchilla equivalents of new data, the contamination effect **completely vanishes**.
- Key condition: training must continue on a **stream of new data**; if trained for multiple epochs on a fixed dataset, forgetting stabilizes at a non-zero level.
- Contamination uniformly distributed throughout the entire training process is harder to forget than that concentrated at the end (spacing effect).

### OLMo-7B Validation

- After inserting 4× full benchmark contamination into an intermediate checkpoint of OLMo-7B, the average accuracy improved by 17 percentage points.
- After continuing pre-training for 13% of the remaining training time, the contamination effect on WinoGrande and ARC-Easy is no longer significant, and about 2 percentage points remain for HellaSwag and PiQA.
- The forgetting curves of 1B and 7B models align closely when scaled by the parameter ratio (5.9×)—indicating that forgetting exhibits scaling behavior.

### Weight Decay Experiments

Under four different weight decay parameters ({50, 5, 1, 0.1}):
- Larger weight decay leads to faster forgetting (with the x-axis ranging from 120 steps to 62,500 steps).
- **Empirical forgetting is always faster than cumulative weight decay**—weight decay serves as an upper bound on forgetting.

### Extrapolation to Large-Scale Training

By analyzing only the learning rate schedule and weight decay parameters to calculate cumulative weight decay:
- **OLMo-7B**: The gradient contribution of the first 40% of the training data has decayed to zero.
- **Llama 3 405B**: The first 10% of the training data may have already been forgotten.

## Highlights & Insights

1. **Significant Practical Implications**: For modern LLMs trained on over 5× Chinchilla data, small-scale contamination may have a negligible impact on evaluation; this provides a new perspective on the reliability of benchmark evaluations.

2. **Spaced Repetition Effect**: Contamination uniformly distributed throughout training is harder to forget than that concentrated at the end, mirroring the "spaced repetition effect" in human learning.

3. **Bridging Theory and Experiments**: Cumulative weight decay in AdamW provides a theoretical tool to estimate the degree of forgetting without requiring actual retraining.

4. **New Data is Key to Forgetting**: Multi-epoch training (reusing the same data) does not lead to complete forgetting; only the continuous injection of new data can achieve this.

## Limitations & Future Work

- The experiments target benchmark questions; the conclusions may not apply to privacy scenarios (the forgetting of random strings or personally identifiable information may differ).
- While findings on small models are validated with OLMo-7B, those on larger-scale models are only inferred through cumulative weight decay.
- The impact of contamination during the instruction-tuning phase is not considered.
- Only exact contamination is evaluated; paraphrased contamination or partial contamination is not studied.

## Related Work & Insights

- **Data Contamination Detection**: Brown 2020 (n-gram matching), Oren 2024 (inference-based detection), Jiang 2024 (controlled contamination experiments)
- **Forgetting Studies**: Tirumala 2022 (exponentially slow forgetting), Jagielski 2023 (empirical forgetting), Toneva 2019 (unforgettable examples)
- **Data Attribution**: Koh & Liang 2017 (influence functions), Grosse 2023

## Rating

⭐⭐⭐⭐ — The research question is important and the experimental design is rigorous. The three-dimensional scaling analysis is clear and comprehensive, and the theoretical tool (cumulative weight decay) is practical. It provides direct guidance for the LLM evaluation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ICLR 2026\] Detecting Data Contamination in LLMs via In-Context Learning](../../ICLR2026/llm_evaluation/detecting_data_contamination_in_llms_via_in-context_learning.md)
- [\[ACL 2025\] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](../../ACL2025/llm_evaluation/antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ICLR 2026\] Detecting Data Contamination from Reinforcement Learning Post-training for Large Language Models](../../ICLR2026/llm_evaluation/detecting_data_contamination_from_reinforcement_learning_post-training_for_large.md)

</div>

<!-- RELATED:END -->
