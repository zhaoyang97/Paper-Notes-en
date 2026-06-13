---
title: >-
  [Paper Note] Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling
description: >-
  [CVPR 2026][Optimization][Conditional conformal prediction] This paper proposes CFC (Conditional Factuality Control), a post-hoc conformal framework that learns feature-conditional acceptance threshold functions via augm…
tags:
  - "CVPR 2026"
  - "Optimization"
  - "Conditional conformal prediction"
  - "hallucination control"
  - "inference-time sampling"
  - "PAC certificates"
  - "set-valued output"
date: 2026-05-08
content_hash: aca9ac09fb94b01d
---

# Conditional Factuality Controlled LLMs with Generalization Certificates via Conformal Sampling

**Conference**: CVPR 2026
**arXiv**: [2603.27403](https://arxiv.org/abs/2603.27403)  
**Code**: Available  
**Area**: Multimodal VLM / LLM Reliability
**Keywords**: Conditional conformal prediction, hallucination control, inference-time sampling, PAC certificates, set-valued output

## TL;DR

This paper proposes CFC (Conditional Factuality Control), a post-hoc conformal framework that learns feature-conditional acceptance threshold functions via augmented quantile regression, providing conditional coverage guarantees (rather than merely marginal guarantees) for LLM sampled outputs. The authors further derive a PAC-style finite-sample certificate CFC-PAC, and validate the approach on synthetic data, reasoning/QA benchmarks, and VLM settings.

## Background & Motivation

LLMs exhibit strong capabilities on reasoning and generation tasks, yet hallucination renders their outputs unreliable. Inference-time multi-sample generation with reranking can improve accuracy, but lacks **formal reliability guarantees**. Conformal Prediction (CP) is a natural candidate—model-agnostic and distribution-free, it constructs set-valued predictions containing the correct answer under an exchangeability assumption.

**Core Problem: Heterogeneity of Marginal Guarantees**

Existing CP methods applied to LLMs rely on a **single global threshold**, providing only marginal coverage guarantees (holding on average across all prompts). This leads to:

**Under-coverage on hard prompts**: Difficult prompts such as long math problems and rare entities are systematically under-covered, with reliability left unguaranteed.

**Over-coverage on easy prompts**: Simple prompts receive unnecessarily excessive coverage, causing prediction sets to inflate and wasting computation.

**Compromise of the global threshold**: A single threshold must trade off between easy and hard regions of the feature space, resulting in subgroup calibration bias and poor sample efficiency.

**Motivation**: **Conditional coverage** is needed—guarantees that hold not only on average, but also conditionally on prompt features. Conditional coverage is strictly stronger than marginal coverage and directly addresses reliability for systematically difficult subgroups. At the same time, prediction sets should remain as compact as possible to preserve the computational efficiency of sampling-based inference.

## Method

### Overall Architecture

CFC is a post-processing layer placed on top of any LLM sampler. The pipeline is as follows:
1. Given a test prompt $X$, sample $M$ candidates $C(X) = \{Y_j\}_{j=1}^M$ from a base generator $\pi$
2. Score each candidate with a verifier $V(X, y) \in [0,1]$ (lower is better)
3. Learn a feature-conditional acceptance threshold $\hat{\lambda}_\alpha(X)$
4. Return the prediction set $\hat{C}_\alpha(X) = \{y \in C(X) : V(X,y) \leq \hat{\lambda}_\alpha(X)\}$

The core innovation is learning $\hat{\lambda}_\alpha(X)$ via **augmented quantile regression** rather than using a global threshold.

### Key Designs

1. **Latent Success Score**: Defined as the best verifier score among correct answers in the candidate set:

    $S(X) := \inf\{V(X,y) : y \in C(X),\; A(X,y) = 1\}$

   The prediction set contains at least one correct answer if and only if $S(X) \leq \lambda(X)$. CFC aims to learn a feature-conditional $\lambda(\cdot)$ such that $S(X) \leq \lambda(X)$ holds with high probability.

2. **Augmented Quantile Regression**: Building on the function-class conditional conformal framework of Gibbs et al., for a candidate score $s \in [0,1]$, the following is solved:

    $\beta_s = \arg\min_{\beta \in \mathbb{R}^d} \left[\frac{1}{N+1}\sum_{i=1}^N \rho_{1-\alpha}(S_i - \Phi(X_i)^\top \beta) + \frac{1}{N+1}\rho_{1-\alpha}(s - \Phi(X_{N+1})^\top \beta)\right]$

   where $\rho_{1-\alpha}$ is the pinball loss and $\Phi(X)$ is a feature map. The key step is to take the **largest fixed point** of the mapping $g_X(s) = \Phi(X)^\top \beta_s$ as the deployment threshold:

    $\hat{\lambda}_\alpha(X) = \sup\{s \in [0,1] : s \leq g_X(s)\}$

   This makes the threshold adaptive to prompt features (difficulty): hard prompts receive a more lenient threshold (allowing more candidates to pass), while easy prompts receive a stricter threshold.

3. **CFC-PAC High-Probability Certificate**: CFC's conditional coverage is an expectation-level guarantee. CFC-PAC further provides a PAC-style finite-sample certificate: Ridge regularization $\frac{\lambda}{2}\|\beta\|_2^2$ is added and the nominal risk level is shrunk:

    $\alpha_{\text{eff}} = \alpha - \varepsilon_N(\delta), \quad \varepsilon_N(\delta) = O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right)$

   With probability at least $1-\delta$, the deployed rule achieves coverage of at least $1-\alpha$.

4. **Efficiency Analysis**: Under mild assumptions (monotonicity and concavity of the score distribution), it is proved that the expected prediction set size of the oracle conditional rule is strictly smaller than that of marginal CP:

    $\mathbb{E}[G_X(\lambda^*(X))] \leq \mathbb{E}[G_X(\bar{\lambda}_\alpha)]$

   with the inequality being strict when $\mathbb{P}(q_\alpha(X) \neq \bar{\lambda}_\alpha) > 0$. CFC asymptotically inherits this efficiency when the quantile regression is consistent (Theorem 4.4).

### Loss & Training

- CFC is a purely post-hoc method that **does not fine-tune the base model**
- Only augmented quantile regression (pinball loss) is fitted on the calibration set
- At deployment, only a fixed-point threshold computation is required, with negligible computational overhead

## Key Experimental Results

### Main Results

Synthetic data ($\alpha = 0.10$, $N_{\text{cal}} = 10000$):

| Method | ECR | APSS↓ | GSC↑ | Note |
|--------|-----|-------|------|------|
| TopK | 90.6 | 16.00 | 58.2 | Fixed-size set |
| ICP (standard conformal) | 90.2 | 16.71 | 57.4 | Single global threshold |
| Learnt CP | 90.2 | 15.72 | 84.3 | Learned threshold without conformal correction |
| **CFC** | **90.3** | **15.53** | **88.7** | Conditional conformal |
| **CFC-PAC** | 90.8 | 15.87 | **89.1** | +PAC high-probability certificate |

CFC substantially improves worst-group coverage (GSC) from 57.4% (ICP) to 88.7%, while producing smaller prediction sets (15.53 vs. 16.71).

TriviaQA ($\alpha = 0.25$):

| Method | ECR | GSC↑ | APSS↓ |
|--------|-----|------|-------|
| TopK | 73.4 | 55.9 | 1.00 |
| ICP | 74.9 | 56.7 | 1.08 |
| Learnt CP | 74.7 | 74.0 | 1.22 |
| **CFC** | **72.7** | **65.2** | **1.03** |

### Ablation Study

| Configuration | GSC↑ | APSS↓ | Note |
|---------------|------|-------|------|
| ICP (global threshold) | 57.4 | 16.71 | Baseline: marginal guarantee |
| Learnt CP (learned threshold) | 84.3 | 15.72 | Learning helps but is insufficient |
| CFC (conditional conformal) | 88.7 | 15.53 | Conformal correction yields further gains |
| CFC-PAC | 89.1 | 15.87 | High-probability certificate; slightly larger sets |

### Key Findings

- **Learning better score thresholds alone is insufficient**: Learnt CP already substantially outperforms ICP (GSC 84.3 vs. 57.4), but the absence of conformal correction still falls short of CFC's subgroup reliability (88.7)
- **Visualization of conditional thresholds** confirms the intuition: easy prompts receive strict thresholds and hard prompts receive lenient ones—precisely the mechanism that corrects systematic under-coverage of difficult inputs by global thresholds
- Conditional rules empirically reduce average prediction set size, validating the efficiency theorem
- CFC transfers directly to VLMs (Flickr8k) without retraining the base model

## Highlights & Insights

- **Conditional conformal prediction for LLM hallucination control** is a natural and valuable research direction
- **Theoretically rigorous**: conditional coverage theorem (Thm 4.1) + PAC certificate (Thm 4.2) + efficiency analysis (Prop 4.3, Thm 4.4) form a cohesive theoretical contribution
- **Practical significance**: For safety-critical applications (medical QA, legal reasoning, etc.), conditional coverage is more meaningful than marginal coverage—systematic under-coverage of hard questions is unacceptable
- The fully post-hoc, model-agnostic design makes the framework broadly applicable without modifying any base model

## Limitations & Future Work

- The choice of feature map $\Phi(X)$ has a large impact on performance, yet no automated selection mechanism is provided
- The linear assumption in quantile regression may be limiting in high-dimensional feature spaces
- Experiments are relatively small-scale (TriviaQA + GSM8K + Flickr8k); scalability to large-scale LLM settings remains to be verified
- The PAC convergence rate $O(\sqrt{\log(1/\delta)/N})$ may be loose when the calibration set is small
- An external verifier is required for scoring, and verifier quality itself becomes a bottleneck

## Related Work & Insights

- Builds on the function-class conditional conformal framework of Gibbs et al., instantiating it for the LLM sampling setting
- Compared to existing LLM CP methods such as conformal factuality and TopK, the core contribution is conditionalization
- The efficiency analysis makes an independent theoretical contribution to the CP literature
- Transfer experiments to the VLM setting provide a new perspective on multimodal reliability

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of conditional conformal prediction and LLM hallucination control is novel, though the theoretical framework is largely inherited from Gibbs et al.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic + real + VLM multi-setting validation, though the scale is relatively small
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory is presented clearly and rigorously; the logical chain from motivation to method to experiments is highly coherent
- **Value**: ⭐⭐⭐⭐ — Offers direct theoretical and practical value for safe LLM deployment, though large-scale validation is still needed

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Colorful Pinball: Density-Weighted Quantile Regression for Conditional Guarantee of Conformal Prediction](../../ICML2026/optimization/colorful_pinball_density-weighted_quantile_regression_for_conditional_guarantee_.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[ICLR 2026\] Conformal Prediction Adaptive to Unknown Subpopulation Shifts](../../ICLR2026/optimization/conformal_prediction_adaptive_to_unknown_subpopulation_shifts.md)
- [\[ICML 2026\] Interpretability and Generalization Bounds for Learning Spatial Physics](../../ICML2026/optimization/interpretability_and_generalization_bounds_for_learning_spatial_physics.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](../../ICML2026/optimization/learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)

</div>

<!-- RELATED:END -->
