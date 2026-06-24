---
title: >-
  [Paper Note] Rethinking the Bias of Foundation Model under Long-tailed Distribution
description: >-
  [ICML 2025][AI Safety][Long-tailed learning] This work reveals that fine-tuning foundation models on long-tailed tasks is doubly affected by "parameter imbalance" (pre-training data bias) and "data imbalance" (downstream data bias). It discovers that parameter imbalance is more critical and cannot be resolved by existing logit adjustment methods. It proposes a method based on causal backdoor adjustment to eliminate the confounding effect of incomplete semantic factors…
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Long-tailed learning"
  - "Foundation model bias"
  - "Parameter imbalance"
  - "Causal inference"
  - "Backdoor adjustment"
date: 2026-05-08
content_hash: e0b6a2874fc66adf
---

# Rethinking the Bias of Foundation Model under Long-tailed Distribution

**Conference**: ICML 2025  
**arXiv**: [2501.15955](https://arxiv.org/abs/2501.15955)  
**Code**: To be confirmed  
**Area**: AI Safety  
**Keywords**: Long-tailed learning, Foundation model bias, Parameter imbalance, Causal inference, Backdoor adjustment

## TL;DR

This work reveals that fine-tuning foundation models on long-tailed tasks is doubly affected by "parameter imbalance" (pre-training data bias) and "data imbalance" (downstream data bias). It discovers that parameter imbalance is more critical and cannot be resolved by existing logit adjustment methods. It proposes a method based on causal backdoor adjustment to eliminate the confounding effect of incomplete semantic factors, achieving an average improvement of approximately 1.67% across three long-tailed benchmarks.

## Background & Motivation

### 1. Background of Long-Tailed Learning

Real-world data often follows a long-tailed distribution, where head classes have abundant samples and tail classes have very few. Fine-tuning foundation models (e.g., CLIP) has become the dominant paradigm in long-tailed learning, with methods like LIFT, LPT, and VL-LTR employing PEFT to preserve pre-trained knowledge.

### 2. Overlooked Pre-training Bias

These methods only focus on downstream data imbalance, neglecting the bias inherent to the foundation model itself, as pre-training data (such as LAION) also follows a long-tailed distribution. Consequently, the fine-tuned model is influenced by a dual long-tailed distribution from both upstream and downstream phases.

### 3. Parameter Imbalance vs. Data Imbalance

The authors decompose the bias into two categories:
- **Parameter Imbalance**: Class imbalance in the pre-training data causes pre-trained weights to bias toward certain classes (since pre-training data is inaccessible, it can only be indirectly perceived through parameters).
- **Data Imbalance**: Class imbalance inherent in the downstream training data itself.

Empirical observations show that parameter imbalance has a more significant impact, and existing re-balancing techniques (e.g., Logit Adjustment) can only alleviate data imbalance but fail to address parameter imbalance.

### 4. Core Idea

A causal graph is constructed to identify "incomplete semantic factors" as confounding variables, which mislead the model into learning spurious correlations between samples and labels rather than true causal relationships. Backdoor adjustment is then applied to eliminate this confounding effect.

## Method

### Overall Architecture

1. Perform zero-shot inference on the foundation model, using GLA to estimate the pre-training label prior $\hat{\mathbb{P}}_P(Y)$.
2. Analyze the cross-influence of parameter imbalance and data imbalance.
3. Construct a causal structural graph: Input $X$ → Label $Y$, where incomplete semantic factors $Z$ act as a confounding variable affecting both $X$ and $Y$.
4. Apply backdoor adjustment to learn $P(Y|do(X))$ instead of $P(Y|X)$, thereby eliminating spurious correlations.

### Key Designs

#### 1. Dual Imbalance Analysis

- Quantify parameter imbalance using the differences in zero-shot performance among various CLIP variants (CLIP, OpenCLIP, MetaCLIP).
- Group data by crossing data imbalance and parameter imbalance, revealing that dual-tail classes are the most severely affected.
- Extend GLA to the training phase (GLA-Train) and observe that it fails to alleviate parameter imbalance.

#### 2. Incomplete Semantic Factors and Causal Analysis

- When a category belongs to the tail due to parameter imbalance, the foundation model captures only partial semantic features (e.g., learning only "dog head" rather than the complete "dog").
- These incomplete features act as the confounding variable $Z$, inducing spurious correlations.
- Construct SCM: $X \leftarrow Z \rightarrow Y$, $X \rightarrow Y$.

#### 3. Backdoor Adjustment

- Apply the backdoor criterion of do-calculus: $P(Y|do(X)) = \sum_z P(Y|X,Z=z)P(Z=z)$.
- Marginalize the incomplete semantic factors in the feature space to learn the true causal effect.
- In practical implementation, this is achieved by fine-tuning the PEFT adapters and replacing the standard CE loss with a backdoor adjustment loss.

## Key Experimental Results

### Main Results

| Dataset | Method | Many | Medium | Few | Overall |
|--------|------|------|--------|-----|---------|
| ImageNet-LT | LIFT (PEFT baseline) | 76.2 | 72.1 | 66.8 | 72.6 |
| ImageNet-LT | GLA (logit adjustment) | 76.8 | 73.0 | 68.5 | 73.5 |
| ImageNet-LT | **Ours** | **77.5** | **73.9** | **69.7** | **74.2** |
| Places365-LT | LIFT | 45.2 | 43.8 | 44.5 | 44.3 |
| Places365-LT | **Ours** | **46.9** | **45.3** | **46.1** | **45.8** |
| iNaturalist2018 | LIFT | 78.3 | 76.1 | 74.2 | 76.0 |
| iNaturalist2018 | **Ours** | **80.5** | **78.2** | **76.3** | **78.0** |

Achieving gains of +1.6%, +1.5%, and +2.0% on the three datasets, respectively.

### Ablation Study

| Configuration | Target | ImageNet-LT | Note |
|------|---------|-------------|------|
| CE (Without adjustment) | — | 71.8 | Baseline |
| LA (Data imbalance only) | Data imbalance | 73.0 | Helpful for tail classes but limited |
| GLA-Train (Training phase) | Parameter + Data | 73.2 | Minimal improvement on parameter imbalance |
| GLA-ZS + GLA-FT (Inference phase) | Parameter + Data | 73.5 | Inference-phase logit adjustment is slightly better |
| **Ours (Backdoor adjustment)** | **Causal deconfounding** | **74.2** | **Fundamentally addresses the confounding issue** |

### Key Findings

- Although LA improves tail-class classifiers, it barely enhances feature quality (KNN accuracy only increases marginally), showing that parameter imbalance is rooted in the feature representation layer rather than the classification head.
- Dual-tail classes (which lie in the tail of both parameter and data distributions) suffer the most severe performance drop and require targeted strategies.
- Backdoor adjustment fundamentally improves feature representation instead of just the decision boundaries by eliminating confounding in the feature space.

## Highlights & Insights

- **Depth of Problem Definition**: This work systematically distinguishes between parameter imbalance and data imbalance in foundation model fine-tuning for the first time, filling a key cognitive gap.
- **Introduction of Causal Perspective**: Incomplete semantic factors are modeled as confounding variables, providing a principled explanation from the perspective of causal inference.
- **Counter-Intuitive Discovery**: Logit adjustment (GLA-Train) fails to address parameter imbalance during the training phase, indicating that the bias is embedded within the feature space rather than the decision boundary.
- **Cross-Model Validation**: The consistent existence of parameter imbalance is validated across three different foundation models: CLIP, OpenCLIP, and MetaCLIP.

## Limitations & Future Work

- Backdoor adjustment requires estimating the distribution of incomplete semantic factors, which can be limited when the semantic dimensions are extremely high or difficult to estimate.
- The parameter imbalance of the text encoder in multimodal scenarios is not considered.
- The evaluation is restricted to the PEFT setting; its effectiveness under full parameter fine-tuning remains to be explored.
- Future work could explore the joint application with GLA to de-bias from both the logit level and the feature level.

## Related Work & Insights

- **vs. LIFT/LPT**: These methods neglect pre-training data bias, whereas this work introduces the dimension of "parameter imbalance".
- **vs. GLA (Zhu et al. 2024)**: While GLA is effective for inference-phase logit adjustment, this paper demonstrates that GLA is ineffective against parameter imbalance during the training phase.
- **vs. Causal Long-Tailed Learning**: Prior causal methods primarily address data imbalance, whereas this work is the first to apply causal inference to the dual imbalance of foundation models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Decouples the dual imbalance of foundation models and provides a causal solution for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive validation across three standard benchmarks, multiple ablation studies, and multiple models.
- Writing Quality: ⭐⭐⭐⭐ Coherent and progressive analysis of the problem with clear causal modeling.
- Value: ⭐⭐⭐⭐⭐ Offers significant methodological insights into the foundation model fine-tuning paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts](avoiding_leakage_poisoning_concept_interventions_under_distribution_shifts.md)
- [\[ICLR 2026\] FaLW: A Forgetting-aware Loss Reweighting for Long-tailed Unlearning](../../ICLR2026/ai_safety/falw_a_forgetting-aware_loss_reweighting_for_long-tailed_unlearning.md)
- [\[CVPR 2026\] FedCART: Tackling Long-Tailed Distributions in Federated Adversarial Training via Classifier Refinement](../../CVPR2026/ai_safety/fedcart_tackling_long-tailed_distributions_in_federated_adversarial_training_via.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)

</div>

<!-- RELATED:END -->
