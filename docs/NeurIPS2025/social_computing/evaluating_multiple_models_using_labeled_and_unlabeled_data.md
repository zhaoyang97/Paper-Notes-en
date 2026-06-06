---
title: >-
  [Paper Note] Evaluating Multiple Models Using Labeled and Unlabeled Data
description: >-
  [NeurIPS 2025][Social Computing][semi-supervised evaluation] This paper proposes **SSME (Semi-Supervised Model Evaluation)**, which leverages a small amount of labeled data and a large amount of unlabeled data to estimat…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "semi-supervised evaluation"
  - "model evaluation"
  - "mixture model"
  - "unlabeled data"
  - "classifier performance"
date: 2026-05-08
content_hash: c75a7c54d88e04e0
---

# Evaluating Multiple Models Using Labeled and Unlabeled Data

**Conference**: NeurIPS 2025
**arXiv**: [2501.11866](https://arxiv.org/abs/2501.11866)  
**Code**: To be confirmed  
**Area**: Social Computing
**Keywords**: semi-supervised evaluation, model evaluation, mixture model, unlabeled data, classifier performance

## TL;DR

This paper proposes **SSME (Semi-Supervised Model Evaluation)**, which leverages a small amount of labeled data and a large amount of unlabeled data to estimate the joint distribution $P(y, \mathbf{s})$ of multiple classifiers via a semi-supervised mixture model, enabling accurate classifier performance evaluation with errors reduced to 1/5 of those incurred when using labeled data alone.

## Background & Motivation

The central dilemma in machine learning evaluation is that **large-scale labeled data is fundamental to evaluation, yet acquiring labels is prohibitively costly or infeasible in many domains** (e.g., healthcare, content moderation, molecular property prediction). At the same time, unlabeled data is typically abundant.

Modern ML practice exacerbates this issue: model hubs such as HuggingFace provide a vast collection of off-the-shelf classifiers, yet practitioners face a large number of trained models with insufficient labeled data to evaluate them.

Limitations of existing methods:
- Using labeled data only: small sample sizes lead to high evaluation variance
- Single-classifier methods (e.g., SPE, Active-Testing): do not exploit mutual information across multiple classifiers
- Annotator models such as Dawid-Skene: rely solely on discrete labels, discarding probabilistic prediction information
- AutoEval / Pseudo-Labeling: learn a mapping from labeled data and extrapolate, ignoring the advantages of joint learning

SSME is the first method to simultaneously exploit three information sources: **(i) multiple classifiers, (ii) continuous probability scores, and (iii) unlabeled data**.

## Method

### Overall Architecture

SSME proceeds in two steps:

**Step 1: Estimate the joint distribution $P(y, \mathbf{s})$**

A semi-supervised mixture model is employed, where each mixture component corresponds to a true class. Both labeled and unlabeled data are used jointly to maximize the log-likelihood:

$$\underset{\theta}{\text{argmax}}\left[\sum_{i=1}^{n_\ell}\log[P_\theta(\mathbf{s}_i|y_i)P_\theta(y_i)] + \lambda_U\sum_{j=1}^{n_u}\log\sum_{k=1}^{K}[P_\theta(\mathbf{s}_j|y_j=k)P_\theta(y_j=k)]\right]$$

where $\lambda_U$ controls the weight of unlabeled data (fixed to 1), and $\mathbf{s}_i = [f_1(x_i), \ldots, f_M(x_i)]$ denotes the concatenated scores from all classifiers.

**Step 2: Estimate classifier performance using $P(y, \mathbf{s})$**

After fitting the density, labels for unlabeled samples are drawn via $P_\theta(y|\mathbf{s})$, and any standard metric (Accuracy, AUC, AUPRC, ECE, etc.) is subsequently computed.

### Key Designs

**Additive Log-Ratio (ALR) Transform**: Classifier outputs lie on the probability simplex and exhibit boundary bias. SSME applies the ALR transform to map them to the unbounded space $\mathbb{R}^{K-1}$, enabling more accurate density estimation.

**Kernel Density Estimation (KDE)**: Gaussian kernels are used to parameterize the class-conditional distribution $P_\theta(\mathbf{s}|y)$, with bandwidths estimated via an improved Sheather-Jones algorithm. KDE makes no parametric assumptions, accommodating the diversity of classifier prediction distributions.

**EM Optimization**: 1000 iterations are performed. The E-step computes the posterior $\gamma_{ik}$ of each data point belonging to each component (labels for labeled data are directly fixed to ground truth); the M-step updates class priors and density parameters.

### Loss & Training

Theoretical analysis derives error bounds under a binary Gaussian mixture model setting:

$$|\text{AUC}_k - \widehat{\text{AUC}}_k| \leq \Phi\left(\frac{\mathbf{c}_k}{\sqrt{2}}\right) - \Phi\left(\frac{\mathbf{c}_k - \epsilon_\mathbf{c}}{\sqrt{2}}\right)$$

where $\epsilon_\mathbf{c} \lesssim \frac{1}{p}\left(\sqrt{\frac{d}{\|c\|^2 n_u}} + \|c\|e^{-\frac{1}{2}n_l\|c\|^2(\cdot)}\right)$

Key implications:
- Increasing $n_u$ → lower error (unlabeled data is beneficial)
- Larger $\|c\|$ → lower error (more accurate classifiers yield better estimates)
- Larger $d$ (more classifiers) → lower error, provided the separation gain outpaces the dimensionality cost

## Key Experimental Results

### Main Results

**Experimental Setup**: 5 binary classification datasets (three tasks from MIMIC-IV, CivilComments, OGB-SARS-CoV, MultiNLI, AG News), 20/50/100 labeled samples + 1000 unlabeled samples, compared against 8 baselines.

**Core Results ($n_\ell=20, n_u=1000$)**:

| Method | Error Reduction Relative to Labeled Only |
|--------|------------------------------------------|
| SSME (Ours) | **5.1×** |
| Second-best baseline | 2.4× |
| Labeled only | 1.0× (baseline) |

**Per-metric Performance**:
- Accuracy estimation: SSME reduces error by 5.6×; second-best method 2.0×
- ECE estimation: SSME reduces error by 7.2× (largest margin)
- AUC estimation: SSME reduces error by 2.9×; second-best 2.6×
- AUPRC estimation: SSME reduces error by 2.2× (smallest margin)

**Absolute Error**: With 20 labeled + 1000 unlabeled samples, SSME estimates accuracy with an error of only 1.5 percentage points (second-best: 3.4 percentage points).

### Ablation Study

**SSME-M (Marginal Fitting)**: Fitting only the marginal $P(y|s)$ for individual classifiers performs substantially worse than joint fitting, confirming that multiple classifiers provide complementary information.

**Effective Sample Size (ESS)**: With 20 labeled + 1000 unlabeled samples, SSME in ECE estimation is equivalent to **539** labeled samples; the second-best method is equivalent to only 110.

**Effect of Labeled Data Size**: As labeled samples increase from 20 → 50 → 100, the relative advantage of SSME decreases from 5.6× to 3.0× to 1.6×, yet SSME consistently leads.

### Key Findings

1. SSME achieves the best or tied performance in 51 out of 60 combinations (dataset × metric × labeled size)
2. ECE estimation benefits most—because ECE relies on binning statistics, which exhibit extremely high variance under few labeled samples
3. Joint fitting of multiple classifiers substantially outperforms fitting each classifier individually (SSME vs. SSME-M)
4. All three theoretically predicted trends are empirically verified
5. Case Study: evaluation error for LLM classifiers is reduced by 2.3×; subgroup evaluation error along the gender dimension is reduced by 5.3×

## Highlights & Insights

- **Unified Framework**: The first method to jointly exploit three information sources—multiple classifiers, continuous probability scores, and unlabeled data
- **Theory–Practice Alignment**: Error bounds are rigorously derived and empirically validated
- **Strong Practicality**: With only 20 labeled samples, SSME achieves evaluation accuracy comparable to 500+ labeled samples
- **Broad Applicability**: Validated across healthcare, NLP, chemistry, and content moderation domains
- **Fairness Application**: Directly applicable to subgroup performance evaluation, offering value for algorithmic fairness auditing

## Limitations & Future Work

- KDE may degrade in high-dimensional settings (many classifiers or many classes)
- The method assumes labeled and unlabeled data are identically distributed—performance under distribution shift is unverified
- The optimality of $\lambda_U = 1$ has not been thoroughly investigated
- The scope is limited to classification tasks; extension to regression or generative task evaluation is not addressed
- The computational efficiency of KDE may become a bottleneck at very large scales

## Related Work & Insights

- Core distinction from Dawid-Skene-style methods: SSME exploits continuous probability scores rather than discrete labels, carrying substantially more information
- Complementary to Prediction-Powered Inference (Angelopoulos et al., 2023), which focuses on confidence intervals
- Implications for LLM-as-evaluator: in low-annotation settings, SSME is more reliable and general than LLM self-evaluation
- Future integration with active learning is promising: SSME can identify high-uncertainty samples to prioritize for annotation

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Elegant integration of three information sources to address a practical problem
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Complete error bound derivation and UL+ analysis
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 5 datasets, 8 baselines, multiple metrics and scenarios
- **Practicality**: ⭐⭐⭐⭐⭐ — Extremely valuable in annotation-scarce settings
- **Overall**: 8.5/10

## Related Work & Insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)
- [\[NeurIPS 2025\] Don't Let It Fade: Preserving Edits in Diffusion Language Models via Token Timestep Allocation](dont_let_it_fade_preserving_edits_in_diffusion_language_mode.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](../../AAAI2026/social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)

</div>

<!-- RELATED:END -->
