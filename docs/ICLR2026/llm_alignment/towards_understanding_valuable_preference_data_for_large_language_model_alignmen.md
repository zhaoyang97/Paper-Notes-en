---
title: >-
  [Paper Note] Towards Understanding Valuable Preference Data for Large Language Model Alignment
description: >-
  [ICLR 2026][LLM Alignment][Preference data selection] This work studies preference data quality from a model-dependent perspective. It proposes Truncated Influence Functions (TIF)…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Preference data selection"
  - "influence function"
  - "DPO"
  - "data quality"
  - "model dependency"
date: 2026-05-08
content_hash: ba7b830cbae50097
---

# Towards Understanding Valuable Preference Data for Large Language Model Alignment

**Conference**: ICLR 2026
**arXiv**: [2510.13212](https://arxiv.org/abs/2510.13212)  
**Code**: [GitHub](https://github.com/tmlr-group/TIF_LossDiff-IRM)  
**Area**: LLM Alignment
**Keywords**: Preference data selection, influence function, DPO, data quality, model dependency

## TL;DR
This work studies preference data quality from a model-dependent perspective. It proposes Truncated Influence Functions (TIF), revealing that data with medium IF values—rather than high IF values as conventionally assumed—is most valuable. Two lightweight proxy metrics, LossDiff and IRM, are designed to approximate TIF. The combined LossDiff-IRM selector achieves an average WinRate improvement of 13.58% using only 50–64% of the data, with consistent effectiveness across multiple LLM families and alignment benchmarks.

## Background & Motivation

**Background**: LLM alignment relies on high-quality preference data. Existing methods filter data using external reward models or GPT-4, implicitly assuming that data quality is an intrinsic property of the data itself—an assumption that ignores the influence of the model and training configuration on data value.

**Limitations of Prior Work**: (1) External filtering (GPT-4/reward model) treats data quality as model-agnostic; the same data may be beneficial for one model but harmful for another. (2) Classical influence functions (IF) suffer from validation-set overfitting in preference alignment (high-IF data is not necessarily optimal). (3) Exact IF computation requires gradient access, which is infeasible for large models.

**Key Challenge**: Preference alignment is an open-ended task with no ground-truth answers; validation gradients serve only as an imperfect proxy for human preferences. Traditional IF assumes high-IF data equals good data, but in preference alignment this leads to overfitting—the model overfits to a small number of high-IF samples with extremely large margins, degrading performance on other samples.

**Goal**: (a) What preference data is truly valuable? (b) How can valuable data be identified efficiently? (c) How can data selection be adapted to a specific target model?

**Key Insight**: Training data is partitioned into small/medium/large IF groups; analysis of training dynamics shows that medium-IF data yields the most stable alignment—motivating TIF, which retains only the intermediate interval, and lightweight positively correlated proxy metrics to approximate TIF.

**Core Idea**: The value of preference data is model-dependent, and data with medium influence is most valuable—neither too easy nor too hard, but "just right."

## Method

### Overall Architecture
**Input**: Raw preference dataset. **Output**: A model-dependent, high-quality subset.

**Pipeline**: 1-epoch warm-up training → compute LossDiff and IRM → select data in the intersection of both intermediate intervals → continue training for 2 epochs.

### Key Designs

1. **Truncated Influence Function (TIF)**

    - **Function**: Corrects the overfitting problem of classical IF in preference alignment.
    - **Mechanism**: IF scores are divided by percentile into small/medium/large groups. Empirical analysis shows: small-IF data corresponds to noise/ambiguity (eval loss increases, reward margin becomes negative after training); large-IF data leads to overfitting (eval loss first decreases then increases, a few pairs are over-optimized to extreme margins); **medium-IF data** is optimal (eval loss decreases steadily, margin increases steadily). TIF is defined as: $\text{TIF}(d) = \mathbb{I}[\delta_{small} < \text{IF}(d) < \delta_{large}]$
    - **Design Motivation**: Since preference alignment is open-ended, validation gradients are imperfect proxies for human preference. Extreme IF values in either direction reflect low-quality data. This is counter-intuitive relative to the classification setting where high IF implies good data, yet is well-grounded for alignment.

2. **Loss Difference (LossDiff) — Validation-Dependent Proxy**

    - **Function**: Approximates IF via forward passes, avoiding gradient computation.
    - **Mechanism**: An auxiliary model $\pi_{\theta_{val}}$ is trained on the validation set; LossDiff is then computed as $\text{LossDiff}(d) = \ell(\theta; d) - \ell(\theta_{val}; d)$. Intuitively, a large LossDiff indicates that moving from $\theta$ toward $\theta_{val}$ reduces the loss on sample $d$, implying alignment with the validation objective.
    - **Design Motivation**: LossDiff is proven to be positively correlated with IF (Pearson $r = 0.77$). Only two forward passes are required; no backpropagation is needed.

3. **Implicit Reward Margin (IRM) — Validation-Free Proxy**

    - **Function**: Evaluates data quality using only the current model's internal signal.
    - **Mechanism**: $\text{IRM}(d) = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$, i.e., the argument inside the sigmoid in the DPO loss.
    - **Design Motivation**: IRM measures the model's preference strength for chosen over rejected responses. It is positively correlated with IF ($r = 0.67$), though weaker than LossDiff due to the absence of validation information. Its key advantage is requiring no validation set whatsoever.

4. **LossDiff-IRM Combined Selector**

    - **Function**: Selects data in the intersection of both metrics' intermediate intervals.
    - **Mechanism**: Data satisfying both LossDiff and IRM falling within intermediate percentile ranges are retained. Since the two metrics have different sources of error (one validation-dependent, the other not), their intersection mutually cancels errors.
    - **Design Motivation**: Each individual TIF approximation has limited precision (overlap ~0.67–0.70); the combination raises overlap to 0.73–0.78.

### Loss & Training
- **Warm-up**: DPO training on the full dataset for 1 epoch.
- Train an auxiliary model $\pi_{\theta_{val}}$ on the validation set for 1 epoch.
- Compute LossDiff (two forward passes) and IRM (one forward pass).
- Select data according to the LossDiff-IRM criterion (retaining 50–64%).
- Continue DPO training on the selected subset for 2 epochs.

## Key Experimental Results

### Main Results: LossDiff-IRM vs. Baselines (DPO)

| Method | Data | UltraFeedback WR | AlpacaEval WR | Vicuna WR | Arena-Hard WR |
|---|---|---|---|---|---|
| Full Data (Llama-3.1-8B) | 100% | 77.61 | 78.41 | 73.75 | 81.39 |
| GPT4 Filter | 64% | 80.57 | 81.09 | 80.31 | 84.30 |
| Reward Model Filter | 64% | 82.68 | 83.76 | 76.88 | 86.19 |
| **LossDiff-IRM** | **64%** | **83.97** | **87.08** | **86.88** | **88.40** |

### Ablation Study: Training Dynamics by IF Group (TIF Validation)

| IF Interval | Train Loss | Eval Loss | Eval Margin | Effect |
|---|---|---|---|---|
| Small-IF | Decreases | Increases | Negative | Harmful (noise/ambiguity) |
| Large-IF | Decreases | Decreases then increases | Continuously increases | Overfitting (few pairs over-optimized) |
| **Medium-IF** | **Decreases** | **Steadily decreases** | **Steadily increases** | **Optimal** |

### Key Findings
- **Model dependency validated**: The IF value distributions for the same data differ between Qwen-0.6B and Llama-1B; some samples benefit one model while harming the other.
- **Medium-IF optimality as a key finding**: Challenges the conventional assumption that high IF equals good data; in preference alignment, medium-IF data is most valuable.
- **High efficiency of LossDiff-IRM**: IF computation on Llama-1B requires ~10 hours; LossDiff-IRM requires only ~5 minutes (120× speedup).
- **Generalization across models and methods**: Consistent effectiveness across Llama-3.1-8B, Qwen3-8B, and the Pythia series, as well as across DPO and SLiC alignment methods.
- **Combination outperforms individuals**: TIF overlap of LossDiff-IRM (0.73–0.78) > LossDiff alone (0.66–0.70) > IRM alone (0.60–0.70).

## Highlights & Insights
- The claim that **"data quality is a property of the model"** overturns the dominant assumption in preference data research. Existing filtering pipelines (GPT-4/RM) are model-agnostic, yet this work demonstrates that data selection should be tailored to each target model.
- The **"Goldilocks effect" of medium-IF optimality** is highly insightful: small-IF data is noise, large-IF data causes overfitting, and only data of "just right" difficulty is most beneficial. This resembles curriculum learning but is more theoretically grounded.
- The **LossDiff "validation-aligned auxiliary model" design** is elegant: a model trained on the validation set serves as a proxy direction, and loss differences provide a closed-form approximation of IF. This approach is transferable to any scenario requiring efficient data valuation.
- **Combining two proxy metrics to cancel errors** resembles ensemble reasoning, but exploits complementary signal sources (validation-dependent vs. validation-free).

## Limitations & Future Work
- The warm-up phase still requires training on the full dataset for one epoch, incurring non-trivial cost at scale.
- Percentile thresholds for TIF require manual tuning and may need adjustment across different datasets.
- Experiments assume access to a high-quality validation set, which may be difficult to obtain in practice.
- Validation on models larger than 8B parameters is not sufficiently demonstrated.

## Related Work & Insights
- **vs. Morimura/Deng et al. (external RM filtering)**: These treat data quality as an intrinsic property and do not adapt to the target model. LossDiff-IRM is model-dependent and computationally more efficient.
- **vs. Pattnaik (curriculum)**: GPT-4 scores are used for curriculum construction, but such scores are model-agnostic. The ranking produced by LossDiff-IRM varies with the model.
- **vs. classical influence functions (Koh & Liang)**: In classification, high IF implies good data. In preference alignment, truncated IF (medium interval) is superior—a domain-specific finding that is novel to this field.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — "Data quality as a model property" and "medium-IF optimality" are both important new insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across multiple model families (Llama/Qwen/Pythia), benchmarks, and alignment methods.
- Writing Quality: ⭐⭐⭐⭐ — Analysis-driven, progressively structured, and logically clear.
- Value: ⭐⭐⭐⭐⭐ — Paradigm-level impact on data selection for LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment](semantic-aware_wasserstein_policy_regularization_for_large_language_model_alignm.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[ICLR 2026\] Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization](alignment_through_meta-weighted_online_sampling_bridging_the_gap_between_data_ge.md)
- [\[ACL 2026\] Alignment Data Map for Efficient Preference Data Selection and Diagnosis](../../ACL2026/llm_alignment/alignment_data_map_for_efficient_preference_data_selection_and_diagnosis.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
