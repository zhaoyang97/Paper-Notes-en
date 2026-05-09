---
title: >-
  [Paper Note] Reliably Detecting Model Failures in Deployment Without Labels
description: >-
  [NeurIPS 2025][LLM Evaluation][Post-deployment degradation monitoring] This paper proposes D3M (Disagreement-Driven Deterioration Monitoring), a three-stage model monitoring algorithm based on variational Bayesian posterior sampling, which reliably detects model performance degradation in label-free, training-data-free deployment settings while maintaining low false positive rates under non-degrading distribution shifts.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Post-deployment degradation monitoring
  - model disagreement
  - variational Bayes
  - label-free detection
  - clinical AI
date: 2026-05-08
content_hash: d20693d6e9596a0e
---

# Reliably Detecting Model Failures in Deployment Without Labels

**Conference**: NeurIPS 2025
**arXiv**: [2506.05047](https://arxiv.org/abs/2506.05047)
**Code**: [GitHub](https://github.com/teivng/d3m)
**Area**: Model Monitoring / Distribution Shift Detection / Trustworthy AI
**Keywords**: Post-deployment degradation monitoring, model disagreement, variational Bayes, label-free detection, clinical AI

## TL;DR

This paper proposes D3M (Disagreement-Driven Deterioration Monitoring), a three-stage model monitoring algorithm based on variational Bayesian posterior sampling, which reliably detects model performance degradation in label-free, training-data-free deployment settings while maintaining low false positive rates under non-degrading distribution shifts.

## Background & Motivation

ML models deployed in production face distribution shift, yet **not all shifts lead to performance degradation**. The central challenge is designing a monitoring mechanism that, without access to labels, can distinguish *degrading shifts* (requiring retraining) from *benign shifts* (where the model still performs well).

Desiderata and limitations of existing methods: (1) **Label-free operation** — distribution shift detection methods (MMD-D, H-divergence, etc.) operate without labels but suffer from high false positive rates under non-degrading shifts; (2) **No training data required** — model disagreement frameworks (e.g., Detectron) require continuous access to training data to compute disagreement statistics, which is infeasible under privacy regulations and edge deployment; (3) **Robustness** — high detection rate for degrading shifts, low false positive rate for benign shifts. D3M is the **only method in the literature that simultaneously satisfies all three desiderata** (Table 1).

- **Core Idea**: Replace full Bayesian neural networks or model fine-tuning with a Variational Bayesian Last Layer (VBLL), and approximate maximum disagreement by sampling from the posterior distribution, thereby eliminating dependence on training data.

## Method

### Overall Architecture

D3M operates in three stages: (1) **Train** — train a feature extractor with a VBLL to model the posterior predictive distribution (PPD) over logits; (2) **Calibrate** — establish a reference distribution $\Phi$ of maximum disagreement rates on an in-distribution (ID) validation set via bootstrap sampling; (3) **Deploy** — compute the maximum disagreement rate $\tilde{\phi}$ on deployment data and apply a quantile test to detect degradation.

### Key Designs

1. **VBLL Posterior Predictive Distribution (Section 2.2, Step 1)**: The model comprises a feature extractor $\operatorname{FE}_\theta: \mathcal{X} \to \mathbb{R}^d$ and a VBLL $\operatorname{VBLL}_\theta: \mathbb{R}^d \to \mathcal{P}(\mathbb{R}^C)$. For each sample, the model outputs a Gaussian posterior over logits: $q_\theta(z|x) = \mathcal{N}(z|\mu_\theta(x), \operatorname{diag}(\sigma^2_\theta(x)))$. Compared to full Bayesian networks, VBLL applies variational inference only to the last layer, yielding computational efficiency at the cost of reduced sampling diversity. The training objective is ELBO maximization: $\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{z \sim q_\theta}[\log \operatorname{softmax}(z)_y] - \operatorname{KL}[q_\theta(z|x) \| p(z)]$.

2. **Calibration Stage — Maximum Disagreement Rate (Step 2)**: Over $T$ bootstrap rounds, each round samples $m$ ID examples; for each sample $x_i$, $K$ logit samples $z_i^{(k)}$ are drawn from the posterior $q_\theta(\cdot|x_i)$, passed through temperature-scaled softmax, and used to sample class labels $\hat{y}_i^{(k)}$ from a Categorical distribution. The disagreement rate with respect to the base model's mean prediction $\bar{y}_i$ is computed as $\operatorname{DisRate}(k) = \frac{1}{m}\sum_i \mathbb{1}\{\hat{y}_i^{(k)} \neq \bar{y}_i\}$, and the per-round maximum $\phi_t = \max_k \operatorname{DisRate}(k)$ is recorded. After $T$ rounds, the reference distribution $\Phi = \{\phi_t\}_{t \in [T]}$ is obtained.

3. **Deployment Monitoring (Step 3)**: A deployment batch $\mathcal{D}_{\text{te}}^m$ is collected, and the same procedure is applied to compute $\tilde{\phi}$. An alert is raised if $\tilde{\phi} \geq \operatorname{Quantile}_{1-\alpha}(\Phi)$. Under no distribution shift, the probability that $\tilde{\phi}$ exceeds this threshold is exactly $\alpha$, directly controlling the false positive rate.

4. **Diversity Enhancement Strategies**: Because VBLL only variationalizes the last layer, posterior sample diversity may be insufficient. Two techniques are employed: (a) temperature scaling $\operatorname{softmax}(z^{(k)}/\tau)$ to increase softmax output diversity; (b) sampling labels from a Categorical distribution (rather than taking argmax) to amplify the disagreement signal.

### Loss & Training

The training stage maximizes the ELBO. The base model's mean prediction is not modified during calibration or deployment, preserving the original generalization guarantees. Temperature $\tau$, bootstrap size $m$, and number of posterior samples $K$ are tunable hyperparameters that must remain consistent between training and deployment.

## Key Experimental Results

### Main Results (Degrading Shifts — Higher TPR is Better)

| Dataset | Query Size | D3M | Detectron | MMD-D | H-Div | BBSD |
|---------|-----------|-----|-----------|-------|-------|------|
| UCI Heart | 10 | .38±.19 | .24±.04 | .09±.03 | .15±.04 | .13±.03 |
| UCI Heart | 50 | .69±.33 | .82±.04 | .27±.04 | .37±.05 | .46±.05 |
| CIFAR-10.1 | 50 | .74±.12 | .83±.04 | .05±.02 | .04±.02 | .12±.03 |
| Camelyon17 | 10 | .89±.20 | .97±.02 | .42±.05 | .03±.02 | .16±.04 |
| Camelyon17 | 50 | .99±.02 | .96±.02 | .69±.05 | .23±.04 | .87±.03 |

### Ablation Study (GEMINI Clinical Data)

| Scenario | Metric | D3M | Other Baselines | Notes |
|----------|--------|-----|----------------|-------|
| Temporal shift (non-degrading) | FPR@α=0.05 | <0.05 | Most >0.05 | D3M resists benign shifts |
| Age shift (degrading), mix ratio 0.75 | TPR | ~0.90 | ~0.90 | On par with strongest baseline |
| Age shift (degrading), mix ratio 0.25 | TPR | ~0.45 | ~0.40 | Competitive under low mixture ratio |

### Key Findings

- **D3M's primary advantage is low FPR under non-degrading shifts**: In the GEMINI temporal shift experiment, where model performance did not degrade despite a genuine distribution shift (including the COVID period), D3M maintained a low FPR while most baselines produced false alarms.
- **High variance at small query sizes**: At query size 10 on UCI datasets, D3M's TPR variance (±0.19) is substantially larger than Detectron's (±0.04), reflecting inherent noise from the sampling strategy.
- **D3M requires neither training data nor gradient updates**: Detectron requires continuous access to training data and gradient-based fine-tuning, whereas D3M relies solely on forward passes.
- **At larger query sizes (100/200), D3M matches the strongest baselines**: Both performance and confidence intervals improve with larger sample sizes.

## Highlights & Insights

- First monitoring mechanism to simultaneously satisfy the triple requirements of label-free operation, no training data access, and theoretical guarantees.
- The use of VBLL is particularly elegant — it avoids the computational overhead of full Bayesian networks while providing sufficient posterior uncertainty.
- The three-stage design fully decouples training, calibration, and deployment, making it well-suited for real-world deployment pipelines.
- Validation on real clinical data (GEMINI) demonstrates selective alerting capability — raising alarms when warranted and remaining silent otherwise.

## Limitations & Future Work

- High variance at small query sizes is the primary weakness, attributed to insufficient sampling diversity of VBLL.
- Temperature $\tau$ requires sweeping — values that are too large cause overfitting to the reference distribution, resulting in elevated FPR.
- Theoretical guarantees are derived for the idealized oracle version of D3M; the practical approximation may oversample outside $\mathcal{H}_p$.
- Only classification tasks are supported; degradation detection for regression requires a different definition of disagreement.
- The quality of the pretrained feature extractor directly affects detection performance (e.g., ImageNet-pretrained ResNet works well on Camelyon17).

## Related Work & Insights

- D3M extends the line of model disagreement frameworks (Ginsberg 2023, Rosenfeld 2024), with the key distinction of eliminating training data dependence.
- It leverages VBLL (Harrison et al. 2024) for efficient uncertainty estimation, but uses it in a non-standard manner — not for OOD detection, but to identify maximum disagreement.
- D3M is complementary to test-time adaptation (TTA, Wang 2020): D3M detects degradation, while TTA addresses it.

## Core Problem

1. **Train**: End-to-end training of FE + VBLL by maximizing ELBO.
2. **Calibrate**: $T$ rounds of bootstrap sampling over ID data to compute the maximum disagreement rate set $\Phi$.
3. **Deploy**: Collect deployment data, compute $\tilde{\phi}$, and compare against $\text{Quantile}_{1-\alpha}(\Phi)$.
4. **Key formulas**: $\operatorname{DisRate}(k) = \frac{1}{m}\sum_{i=1}^m \mathbb{1}\{\hat{y}_i^{(k)} \neq \bar{y}_i\}$, $\phi_t = \max_k \operatorname{DisRate}(k)$

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of VBLL and disagreement is novel; the formalization of three desiderata is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Standard benchmarks plus real-world clinical GEMINI data, covering both degrading and non-degrading scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation and methodology are clearly described, though theoretical content is largely relegated to the appendix.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a real deployment pain point; the GEMINI experiments are particularly compelling.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[NeurIPS 2025\] Aggregation Hides OOD Generalization Failures from Spurious Correlations](aggregation_hides_out-of-distribution_generalization_failures_from_spurious_corr.md)
- [\[NeurIPS 2025\] Model-Behavior Alignment under Flexible Evaluation: When the Best-Fitting Model Isn't the Right One](model-behavior_alignment_under_flexible_evaluation_when_the_best-fitting_model_i.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[ICLR 2026\] Subliminal Signals in Preference Labels](../../ICLR2026/llm_evaluation/subliminal_signals_in_preference_labels.md)

<!-- RELATED:END -->
