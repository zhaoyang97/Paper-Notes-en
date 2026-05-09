---
title: >-
  [Paper Note] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums
description: >-
  [ICLR 2026][LLM Pretraining][training re-evaluation curves] This paper proposes the Training Re-evaluation Curve (TREC) as a diagnostic tool that analyzes the loss of a fully trained model evaluated on training data at each timestep, thereby guiding optimal placement of high-quality data. The paper further demonstrates that the shape of TREC can be predicted via the implicit EMA coefficient of AdamW, enabling curriculum design without any actual training runs.
tags:
  - ICLR 2026
  - LLM Pretraining
  - training re-evaluation curves
  - data curriculum learning
  - AdamW timescale
  - high-quality data placement
  - continual pretraining
date: 2026-05-08
content_hash: 157aeee74635f648
---

# Predicting Training Re-evaluation Curves Enables Effective Data Curriculums

**Conference**: ICLR 2026
**arXiv**: [2509.25380](https://arxiv.org/abs/2509.25380)
**Code**: None
**Area**: LLM Pretraining
**Keywords**: training re-evaluation curves, data curriculum learning, AdamW timescale, high-quality data placement, continual pretraining

## TL;DR

This paper proposes the Training Re-evaluation Curve (TREC) as a diagnostic tool that analyzes the loss of a fully trained model evaluated on training data at each timestep, thereby guiding optimal placement of high-quality data. The paper further demonstrates that the shape of TREC can be predicted via the implicit EMA coefficient of AdamW, enabling curriculum design without any actual training runs.

## Background & Motivation

Modern LLM training commonly adopts multi-stage data curriculum strategies, introducing high-quality, domain-specific, or recent data near the end of pretraining (i.e., the annealing phase). This practice rests on the assumption that presenting data when the learning rate approaches zero maximizes its effect. However, this assumption lacks theoretical grounding, and "many interesting questions about the optimal data distribution for pretraining remain unanswered" (Anil et al., 2023).

In practice, determining the optimal timing for introducing high-quality data relies primarily on heuristics or expensive ablation studies. For instance, Llama-3 405B annealed on the GSM8k training set without measurable benefit, while OLMo-2 13B used a high-quality mixture only during the final 5.7% of training. The effectiveness of different strategies varies substantially, yet a unified theoretical framework for explaining and predicting these differences is absent.

The central insight of this paper is that high-quality data need not always be placed at the end of training; rather, it should be placed where the model **retains that data most effectively**—i.e., at the minimum of TREC.

## Method

### Overall Architecture

The methodology proceeds in three stages:
1. **Define and validate TREC**: demonstrate that the TREC minimum genuinely corresponds to the optimal data placement position.
2. **Identify the governing factor of TREC**: show that the EMA timescale $\tau$ of AdamW dominates the shape of TREC.
3. **Construct a predictive model**: combine the EMA coefficient with a training-progress correction term to accurately predict TREC prior to training.

### Key Designs

1. **Definition of the Training Re-evaluation Curve (TREC)**:

   Given a sequence of training batches $B_1, \ldots, B_T$ sampled i.i.d. from distribution $D$ and the final model parameters $\theta_T$, TREC is defined as:

   $$\mathcal{L}_{re}(t) := \mathcal{L}(B_t; \theta_T)$$

   That is, the loss is recomputed on each training batch using the final model. A lower TREC value at a given timestep indicates that the final model has "memorized" the data at that step more deeply. The core hypothesis is: **placing high-quality data at the TREC minimum maximizes its contribution to the target task**.

2. **TREC Shape Governed by the AdamW Timescale $\tau$**:

   The AdamW parameters $\theta_t$ can be viewed as an exponential moving average (EMA) of weight updates with timescale:

   $$\tau = \frac{1}{\eta \lambda T} = \frac{B}{\eta \lambda D}$$

   where $\eta$ is the learning rate, $\lambda$ is the weight decay, $T$ is the total number of steps, $B$ is the batch size, and $D$ is the total number of tokens. Experiments show that regardless of whether $\tau$ is varied via $\eta$, $\lambda$, or $B$, the TREC shape remains consistent as long as $\tau$ matches. This finding holds across models ranging from 111M to 3.3B parameters (spanning $1000\times$ in compute).

3. **Predictive Model for TREC**:

   Although the EMA coefficient $c(\hat{t})$ reflects each step's contribution to the final weights, the effectiveness of early gradients decays due to "minimizer drift." The paper proposes:

   $$\hat{\mathcal{L}}_{re}(\hat{t}) = 1 - c(\hat{t})^p \cdot \hat{t}^m$$

   where $\hat{t} = t/T$ is the fractional training progress, $p$ (fixed at 0.5) controls the EMA contribution strength, and $m$ (the training-progress exponent) controls when TREC begins to reflect the EMA. The optimal $m^*$ follows a power-law relation:

   $$m^* = C \cdot (TPP)^{\mu_1} \cdot (\tau)^{\mu_2}$$

   where TPP (tokens-per-parameter) and $\tau$ are the two key variables. The power law fitted at the 111M scale retains ${\sim}98\%$ Pearson correlation at the 3.3B scale.

### Loss & Training

The paper does not introduce a new loss function; instead, it provides an **optimal data ordering strategy** for existing AdamW training. Key findings include:
- Under step-decay learning rate schedules, the TREC trough occurs before the learning rate drop rather than at the end of training.
- Under linear decay-to-zero (D2Z) schedules, the TREC trough occurs at approximately 60–80% through training.
- The absolute depth of the TREC trough decreases as TPP increases, suggesting that overtrained models are less capable of memorizing specific data.

## Key Experimental Results

### Main Results: Data Placement Validation (610M model, 82 TPP)

For each learning rate schedule, ten models are trained with a 5B code-mixture (CB) dataset inserted into a different 10% segment of training.

| LR Schedule | Optimal Placement | Coincides with TREC Minimum | Gain vs. Uniform Mixing |
|---|---|---|---|
| Step-decay (drop at 70%) | Segment 6–7 (60–70%) | ✓ | Clearly better than uniform |
| $10\times$ linear decay | Last segment (90–100%) | ✓ | Clearly better than uniform |
| Decay-to-zero (D2Z) | Last segment | ✓ | Clearly better than uniform |

### TREC Prediction Accuracy

| Model Scale | $m^*$ Prediction $R^2$ | TREC Shape Pearson $r_p$ |
|---|---|---|
| 111M | 98.9% | 96.6% |
| 266M | 97.2% | 97.5% |
| 610M | 98.7% | 98.4% |
| 1.7B | 89.0% | 98.7% |
| 3.3B | 76.7% | 98.6% |

### Sparse MoE Experiments (111M base model)

| Num. Experts $E$ | Effective TPP | TREC Behavior |
|---|---|---|
| 1 (dense) | 20 | Shallowest trough |
| 4 | 5 | Deeper and earlier trough |
| 8 | 2.5 | Deeper and earlier |
| 32 | 0.625 | Deepest and earliest trough |

### 3.9B Continual Pretraining

| Configuration | Math Validation Performance |
|---|---|
| High-quality data placed in the middle (TREC trough) | Best (across all LR schedules) |
| High-quality data placed at the end | Second best |
| No high-quality data | Baseline |

### Ablation Study

| Ablation Dimension | Key Finding |
|---|---|
| Varying $\beta_1$, $\beta_2$ | TREC shape remains nearly unchanged, confirming $\tau$ as the dominant factor |
| Batch size $> B_{crit}$ | TREC shape deviates significantly; influence of individual batches weakens |
| Increasing TPP | TREC trough depth decreases (reduced memorization capacity) |
| Across different schedules | TREC shapes align when $\tau$ is matched |

### Key Findings

1. **TREC minimum $\neq$ end of training**: Particularly under step-decay schedules, high-quality data should be placed before the learning rate drop, not at the end.
2. **$\tau$ is the universal key**: Regardless of whether the learning rate, weight decay, or batch size is varied, TREC shapes match whenever $\tau$ matches.
3. **Cross-scale predictability**: The $m^*$ power law fitted on 111M models generalizes to 3.3B ($1000\times$ compute).
4. **Explains Llama-3 405B's failure**: The annealing phase involved only 3 optimization steps with a near-zero LR, making the EMA coefficient essentially zero.
5. **MoE experts reduce effective TPP**: This leads to stronger memorization; TREC analysis can guide data strategies for MoE models.

## Highlights & Insights

- **TREC is a minimal yet profound diagnostic tool**: Simply re-evaluating training data with the final model suffices to reveal the temporal structure of data influence.
- **Solid theoretical foundation**: The power-law predictive model is grounded in clear theoretical motivation (minimizer drift on a quadratic loss surface) and validated at scales from 111M to 3.9B.
- **High practical value**: Provides practitioners with a method for determining optimal data ordering without expensive ablation experiments.
- **Natural integration with existing work**: Successfully explains data strategy choices in published training recipes such as OLMo-2, Feng et al., and Pangu-Ultra.
- **Direct guidance for CPT/SFT scenarios**: TREC prediction applies not only to pretraining but also to continual pretraining (CPT).

## Limitations & Future Work

1. **Optimizer scope**: The predictive model is designed specifically for AdamW; extending it to non-EMA optimizers such as Adagrad, Adafactor, and SGD remains an open problem.
2. **TREC absolute values are not comparable across schedules**: TREC reliably guides placement within a given schedule, but cross-schedule comparison of absolute values is invalid.
3. **Shape predicted, not magnitude**: The current model normalizes and compares shapes; predicting absolute TREC magnitude remains unexplored.
4. **No systematic analysis of data type differences**: Differences in TREC across factual vs. reasoning-oriented, or instruction vs. narrative content, have not been studied.
5. **Anomalous behavior at high learning rates**: In the 3.9B CPT experiments, $\eta = 0.015$ produced the deepest TREC trough but the worst validation performance; the underlying mechanism is unclear.

## Related Work & Insights

- **Complementary to AdEMAMix (Pagliardini et al., 2024)**: The latter designs a slow-forgetting optimizer, whereas this paper exploits the forgetting structure to guide data placement.
- **Orthogonal to data mixing laws (Ye et al., 2024)**: The latter addresses *what* data to include; this paper addresses *when* to present it.
- **Provides a complementary perspective to Scaling Collapse (Qiu et al., 2025)**: Both use normalized compute/training progress, but pursue different objectives.
- **Offers practical recommendations for reproducibility**: The "memorization window" concept from Falcon-H1 can be precisely characterized using TREC.
- **Inspires the design of "forgetting-free" LR schedules**: While it is theoretically possible to design schedules that flatten TREC, a certain degree of forgetting is beneficial in practice.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (TREC is a novel and elegant concept that perfectly bridges theory and practice)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (600+ TRECs, 111M to 3.9B, multiple schedules and hyperparameters)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear logic, progressive exposition, polished figures)
- Value: ⭐⭐⭐⭐⭐ (Directly actionable guidance for LLM training data strategies)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training](common_corpus_ethical_data_for_llm_pretraining.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)
- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[NeurIPS 2025\] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation](../../NeurIPS2025/llm_pretraining/prescribe_predicting_single-cell_responses_with_bayesian_estimation.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)

<!-- RELATED:END -->
