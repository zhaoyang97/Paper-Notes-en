---
title: >-
  [Paper Note] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding
description: >-
  [ACL 2025][Model Extrapolation] This paper proposes EpiCoDe, a training-free method combining Model Extrapolation and Contrastive Decoding. It enhances the performance of fine-tuned models in data-scarce scenarios through parameter-space extrapolation and inference-time logit contrast, while providing a theoretical analysis framework from the perspective of logit errors.
tags:
  - "ACL 2025"
  - "Model Extrapolation"
  - "Contrastive Decoding"
  - "Data Scarcity"
  - "Training-free Enhancement"
  - "Logit Analysis"
date: 2026-05-08
content_hash: 90e48b427cbc165e
---

# EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding

**Conference**: ACL 2025  
**arXiv**: [2506.03489](https://arxiv.org/abs/2506.03489)  
**Code**: None  
**Area**: Others  
**Keywords**: Model Extrapolation, Contrastive Decoding, Data Scarcity, Training-free Enhancement, Logit Analysis

## TL;DR

This paper proposes EpiCoDe, a training-free method combining Model Extrapolation and Contrastive Decoding. It enhances the performance of fine-tuned models in data-scarce scenarios through parameter-space extrapolation and inference-time logit contrast, while providing a theoretical analysis framework from the perspective of logit errors.

## Background & Motivation

The superior performance of Large Language Models (LLMs) heavily relies on large-scale and high-quality training data. However, in domains such as law and medicine, available data is often limited due to privacy constraints and annotation costs. This data scarcity leads to insufficient fine-tuning, leaving model performance far below its potential upper bound.

Existing training-free enhancement methods mainly fall into two categories:

**Model Extrapolation**: Leverages two checkpoints from different training stages to linearly extrapolate in the parameter space, yielding a stronger model.

**Contrastive Decoding**: Contrasts the logit distribution differences between strong and weak models during inference to eliminate shared error patterns.

However, both methods have limitations—model extrapolation sometimes fails, and contrastive decoding can even degrade performance on certain tasks. More importantly, **no prior work has explored combining the two**, nor has there been a theoretical framework to explain why contrastive decoding succeeds or fails. EpiCoDe is designed to address these gaps.

## Method

### Overall Architecture

EpiCoDe operates in two stages and requires absolutely no additional training:

**Stage 1: Model Extrapolation**  
Collect two checkpoints during the fine-tuning process:
- $\theta^{early}$: Early-stage model (e.g., trained for 1 epoch)
- $\theta^{ft}$: Fully fine-tuned model (e.g., trained for 2 epochs)

An enhanced model is obtained via linear extrapolation:
$$\theta^{ep} = \theta^{ft} + \mu(\theta^{ft} - \theta^{early})$$
where $\mu > 0$ is a hyperparameter controlling the extrapolation step size.

**Stage 2: Contrastive Decoding**  
During inference, the extrapolated model $\theta^{ep}$ serves as the strong model and the fine-tuned model $\theta^{ft}$ as the weak model. The logit differences are calculated to adjust the prediction:
$$L_{CD}(x_{<i}) = L_{ep}(x_{<i}) + \lambda \cdot (L_{ep}(x_{<i}) - L_{ft}(x_{<i}))$$
where $\lambda > 0$ controls the contrast strength.

### Key Designs

1. **Weak Model Selection Strategy**: Instead of using the initial model $\theta^{init}$ or the early model $\theta^{early}$, the nearest model $\theta^{ft}$ is selected as the weak model. The rationale is that models closer in the parameter space share more consistent error patterns, allowing the contrastive subtraction to effectively eliminate shared errors.

2. **Decoding Constraint**: A threshold $\alpha = 0.1$ is introduced to restrict token selection only to candidates with high logit scores from the strong model, preventing anomalous inflation of contrastive values caused by extremely low logits from the weak model.

3. **Staged Hyperparameter Search**: The optimal $\mu$ (for model extrapolation) is first searched on the validation set, and then $\mu$ is frozen to search for $\lambda$ (for contrastive decoding intensity) rather than conducting a joint grid search, ensuring a fair contrast.

### Loss & Training

EpiCoDe itself does not involve any additional training. The underlying fine-tuning uses a standard setup: AdamW optimizer ($\beta_1=0.9$, $\beta_2=0.95$), learning rate of 3e-5, batch size of 128, and is trained for 2 epochs.

### Theoretical Analysis Framework

The paper proposes a new theoretical framework to analyze contrastive decoding from the perspective of logit errors:

Let $\theta^*$ be the hypothetical optimal model. Define $\delta(x, \theta) = L(x|\theta) - L(x|\theta^*)$ as the logit error. Assume that the error of the strong model follows $N(0, \epsilon^2)$ and the error of the weak model follows $N(0, (k\epsilon)^2)$, where $k > 1$.

**Case 1 (Locality Holds)**: When the errors of the strong and weak models are highly positively correlated, contrastive decoding can reduce the error variance from $\epsilon$ to $(1-\lambda(k-1))\epsilon$. This corresponds precisely to the case of $\theta^{ep}$ and $\theta^{ft}$—extrapolation guarantees parameter space locality, which in turn ensures consistent error patterns.

**Case 2 (Inconsistent Errors)**: When the errors of the strong and weak models are independent (such as when using the un-fine-tuned $\theta^{init}$), the error variance under contrastive decoding instead increases to $\sqrt{(1+\lambda)^2\epsilon^2 + \lambda^2 k^2 \epsilon^2}$, leading to performance degradation.

## Key Experimental Results

### Main Results

| Method | Law (Acc) | Math (Acc) | Logic (Acc) | Average |
|------|-----------|------------|-------------|------|
| **Deepseek-7B** | | | | |
| Finetune | 64.78 | 27.28 | 57.22 | 49.76 |
| ME only | 65.42 (+0.64) | 27.12 (-0.17) | 58.89 (+1.67) | 50.48 |
| CD only | 65.29 (+0.51) | 26.88 (-0.41) | 58.46 (+1.24) | 50.21 |
| **EpiCoDe** | **65.51 (+0.73)** | **27.81 (+0.53)** | **59.05 (+1.83)** | **50.79** |
| **Llama-3.2-3B** | | | | |
| Finetune | 62.13 | 48.45 | 53.45 | 54.68 |
| ME only | 62.73 (+0.60) | 49.74 (+1.29) | 55.11 (+1.66) | 55.77 |
| CD only | 63.38 (+1.25) | 53.13 (+4.68) | 56.62 (+3.17) | 57.71 |
| **EpiCoDe** | **63.79 (+1.66)** | **54.31 (+5.86)** | **57.48 (+4.03)** | **58.53** |

### Ablation Study: Weak Model Selection

| Weak Model | Law | Math | Logic |
|--------|-----|------|-------|
| $\theta^{ft}$ (Qwen2-7B) | 70.25 (+1.22) | 58.71 (+1.59) | 68.07 (+1.40) |
| $\theta^{early}$ (Qwen2-7B) | 69.63 (+0.60) | 58.27 (+1.15) | 66.75 (+0.08) |
| $\theta^{init}$ (Qwen2-7B) | 68.70 (-0.33) | 55.96 (-1.16) | 66.78 (+0.11) |
| $\theta^{ft}$ (Qwen2-1.5B) | 69.60 (+0.57) | 49.39 (-7.73) | 67.91 (+1.24) |

### Key Findings

1. **EpiCoDe consistently outperforms ME or CD alone across all models and tasks**, showing improvements in all but 3 out of 120 experiments.
2. While using ME or CD alone frequently fails on math tasks (such as on DeepSeek for the Math task), EpiCoDe remains consistently effective.
3. **Improvements mainly stem from hard samples**: In the legal QA task, the largest improvement (+1.84%) is observed on the "hard" subset (which requires the longest CoT), whereas performance on the "easy" subset remains almost unchanged.
4. Using $\theta^{init}$ as the weak model leads to performance degradation, validating the theoretical prediction that inconsistent errors cause adverse effects.
5. Paired t-tests indicate that out of 12 experimental groups, EpiCoDe statistically outperforms ME alone in 7 groups and CD alone in 11 groups with 95% confidence.

## Highlights & Insights

- **Extremely simple yet effective**: Requires no additional training or data, utilizing only pre-existing checkpoints for parameter-space extrapolation and inference-time logit contrast.
- **Solid theoretical contribution**: Establishes a theoretical framework for contrastive decoding from the perspective of logit error variance, quantitatively explaining why choosing a close neighbor in the parameter space as the weak model is superior.
- **"Locality" is the core insight**: Model extrapolation naturally preserves the locality between $\theta^{ep}$ and $\theta^{ft}$, which subsequently guarantees the simplicity and effectiveness of contrastive decoding—making the two methods perfectly complementary.

## Limitations & Future Work

1. Validated only on models ranging from 1.5B to 7B parameters; the efficacy on larger models (e.g., 70B+) remains unknown.
2. Tasks are limited to Chinese legal QA, mathematics, and logical reasoning, leaving English tasks and broader NLP tasks uncovered.
3. The hyperparameters $\mu$ and $\lambda$ require validation-set tuning, increasing search costs.
4. The theoretical framework assumes normally distributed errors, which may be more complex in real-world scenarios.
5. The extrapolation magnitude $\mu$ must be small ($\mu \ll 1$), which limits the potential upper bound of performance improvements.

## Related Work & Insights

- **Weak-to-Strong Extrapolation** (Zheng et al., 2024): Treats RLHF models as a merge between SFT models and unknown super-strong models, then performs reverse extrapolation. EpiCoDe extends this concept to data-scarce scenarios.
- **Contrastive Decoding** (Li et al., 2023; O'Brien & Lewis, 2023): Original CD utilizes pairs of models of different sizes from the same family. EpiCoDe innovatively uses checkpoint pairs from different stages of the same model and provides a theoretical explanation.
- **Insights**: This two-stage paradigm of "parameter-space operation + inference-time correction" could be generalized to other scenarios, such as inference optimization following model merging.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of model extrapolation and contrastive decoding is novel, and the theoretical analysis offers a fresh perspective, though each individual technique builds on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, featuring 4 models $\times$ 3 tasks $\times$ 10 runs, paired t-tests, robustness analysis, and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with rigorous logical flow (despite some LaTeX noise in theoretical derivations) and easy-to-understand tables/charts.
- **Value**: ⭐⭐⭐⭐ — Simple and practical, suitable for plug-and-play in low-resource settings. The theoretical framework provides inspiring insights for understanding contrastive decoding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)
- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)
- [\[ACL 2025\] Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability](rationales_are_not_silver_bullets_measuring_the_impact_of_rationales_on_model_pe.md)
- [\[ACL 2025\] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)
- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](../../AAAI2026/others/measuring_model_performance_in_the_presence_of_an_intervention.md)

</div>

<!-- RELATED:END -->
