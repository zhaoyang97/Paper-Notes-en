---
title: >-
  [Paper Note] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference
description: >-
  [ICML 2026][Causal Inference][proper scoring rule] This paper proposes a general framework: by matching the local second-order curvature of the training loss $w_\ell(p)$ with the curvature of the downstream task error $w…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "proper scoring rule"
  - "IPW"
  - "ATE"
  - "propensity score"
  - "canonical link"
date: 2026-05-08
content_hash: 75da81de901860bf
---

# Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference

**Conference**: ICML 2026  
**arXiv**: [2606.03332](https://arxiv.org/abs/2606.03332)  
**Code**: To be confirmed (Paper署名 "tailored-psr", repository URL not explicitly provided)  
**Area**: Causal Inference / Probability Estimation / IPW / Strictly proper scoring rule  
**Keywords**: proper scoring rule, IPW, ATE, propensity score, canonical link

## TL;DR
This paper proposes a general framework: by matching the local second-order curvature of the training loss $w_\ell(p)$ with the curvature of the downstream task error $w_{\text{task}}(p)$, a strictly proper scoring rule "geometrically aligned" with the downstream task can be derived. Applying this to ATE estimation via IPW yields a closed-form loss and a closed-form canonical activation function (solved via a quartic equation). It consistently outperforms log-loss and covariate balancing baselines on IHDP, Jobs, Kang-Schafer, and ACIC 2017.

## Background & Motivation

**Background**: Many ML pipelines are "two-stage"—first estimating a conditional probability $\hat p(x)$, then feeding it as input to a downstream estimator (e.g., classification, risk assessment, or IPW/AIPW in causal inference). The first stage almost exclusively uses log-loss for training, as it corresponds to KL divergence and is statistically "natural."

**Limitations of Prior Work**: Log-loss is completely task-agnostic regarding the downstream task. This is particularly fatal in IPW-based causal inference: when the true propensity score $e(x)$ approaches $0$ or $1$, the bias and variance of the IPW estimator explode due to $1/\hat e$ and $1/(1-\hat e)$ terms, yet the penalty of log-loss near the boundaries is insufficient. A model may achieve low log-loss while yielding poor downstream ATE estimates.

**Key Challenge**: Existing remedies follow two paths, both suboptimal: (i) Post-processing heuristics: trimming or clipping, which sacrifices consistency for variance reduction; (ii) Covariate balancing (CBPS, Entropy Balancing, SBW, CBSR), which injects moment constraints into training. However, the theoretical link between "covariate balance" and "downstream estimation bias minimization" is weak (Bruns-Smith & Feller, 2022). Furthermore, methods like CBSR break the pairing between proper scoring rules and their canonical links, leading to gradient explosion and making them nearly incompatible with deep networks.

**Goal**: Can a training loss be constructed such that its geometry on the probability simplex directly mirrors the sensitivity of the downstream task to probability estimation errors?

**Key Insight**: Proper scoring rule theory indicates that every strictly proper scoring rule is uniquely characterized by a non-negative weight function $w_\ell(q) = H_\ell''(q)$, where its induced divergence $d_\ell(p,q)$ has a second-order curvature at $q=p$ precisely equal to $w_\ell(p)$. If the upper bound of the downstream task error can also be written as a divergence $d_{\text{task}}$, one only needs to equate the two curvatures to ensure the local penalty of the loss is "weighted by downstream sensitivity."

**Core Idea**: Derive task-specific losses using the curvature matching equation $w_\ell(p) = w_{\text{task}}(p)$ and solve the associated differential equation $(\sigma_\ell^{-1})'(p) = w_\ell(p)$ to obtain the canonical activation function, ensuring training stability.

## Method

### Overall Architecture

The framework consists of a three-step derivation chain:
1. **Upper Bound**: Find a divergence-type upper bound $\mathbb{E}[d_{\text{task}}(p,\hat p)]$ for the downstream task error $\mathcal{E}(\theta,\hat\theta)$.
2. **Curvature Matching**: Compute $w_{\text{task}}(p) = \partial_q^2 d_{\text{task}}(p,q)|_{q=p}$, set $w_\ell = w_{\text{task}}$, and derive the unique proper scoring rule $\ell$ using the integral construction from Buja et al. (2005).
3. **Canonical Pairing**: Solve the differential equation $(\sigma_\ell^{-1})'(p) = w_\ell(p)$ to obtain the activation function $\sigma_\ell$. When paired with $\ell$, the gradient with respect to the logit simplifies to $\partial \ell / \partial z = p - y$, eradicating gradient explosion.

The input consists of $(X_i, T_i)$ sample pairs, and the output is a propensity score model $\hat e(x) = \sigma_\ell(f_\theta(x))$, which is then substituted into IPW/Hajek/AIPW to estimate the ATE.

### Key Designs

1. **General Curvature Matching Framework (Section 3.1)**:
    - **Function**: Automatically derives the most suitable proper scoring rule for any "two-stage" estimation problem (where the first stage estimates probability $p$ and the second stage computes a downstream estimator $\hat\theta$ using $\hat p$).
    - **Mechanism**: Assume the downstream error is bounded by $\mathcal{E}(\theta,\hat\theta) \le \mathbb{E}[d_{\text{task}}(p,\hat p)]$. A Taylor expansion at $q=p$ gives $d_{\text{task}}(p,q) = \tfrac{1}{2} w_{\text{task}}(p)(p-q)^2 + o((p-q)^2)$. The induced divergence of any proper scoring rule has an identical expansion $d_\ell(p,q) = \tfrac{1}{2} w_\ell(p)(p-q)^2 + o((p-q)^2)$. By setting $w_\ell(p) = w_{\text{task}}(p)$ and integrating $H_\ell''(q) = w_\ell(q)$ twice to obtain the entropy function $H_\ell$, the final $\ell$ is given by $\ell(y,q) = -H_\ell(q) - H_\ell'(q)(y - q)$.
    - **Design Motivation**: The additive decomposition of proper scoring rules $\mathbb{E}[\ell(T,\hat p)] = \mathbb{E}[d_\ell(p,\hat p)] + \mathbb{E}[H_\ell(p)]$ ensures that minimizing the training loss is equivalent to minimizing $\mathbb{E}[d_\ell]$. Combined with curvature matching, this forms a provable link from "training loss to the second-order upper bound of downstream error." Unlike covariate balancing methods that provide only "surrogate objectives," this link is explicit and direct.

2. **IPW-tailored Loss (Section 3.2)**:
    - **Function**: Implements the general framework for ATE estimation via IPW, providing closed-form task-specific weights and losses.
    - **Mechanism**: Performs a bias-variance decomposition of the MSE of the IPW estimator $\hat\tau_{\text{ATE}}$. The Bias term is decomposed via Cauchy-Schwarz into $\mathbb{E}[(e/\hat e - 1)^2]$, corresponding to $d_{\text{bias}}$; the variance term, under the assumption of bounded second moments, yields $\mathbb{E}[Y(1)^2 e (1/\hat e - 1/e)^2]$, corresponding to $d_{\text{var}}$. Computing the second derivative at $q=p$ gives $w_{\text{task}}(p) = \big(\tfrac{2}{p^2} + \tfrac{2}{(1-p)^2}\big) + \big(\tfrac{2}{p^3} + \tfrac{2}{(1-p)^3}\big)$, where the first pair comes from bias and the second from variance. Compared to $w(q) = \tfrac{1}{q(1-q)}$ for log-loss, the new weights explode near $q \to 0$ or $q \to 1$ at rates of $1/p^2$ and $1/p^3$, respectively, severely penalizing boundary errors. The paper also notes that the MSE upper bounds for Hajek and AIPW estimators share the same $d_{\text{task}}$ (differing only by scaling constants), making the loss "universal" for these estimators.
    - **Design Motivation**: The fragility of IPW at small propensity scores is a classic pain point. Previous approaches either used trimming (losing consistency) or changed the estimator (e.g., Overlap Weighting, which changes the target). Ours embeds this $1/p^3$ sensitivity directly into the loss during the training phase, representing an ante-hoc rather than post-hoc fix.

3. **Canonical Probability Mapping $\sigma_\ell$ (Section 3.2.4)**:
    - **Function**: Replaces the sigmoid as the final activation function to ensure stable training gradients.
    - **Mechanism**: If the new loss were directly connected to a sigmoid, the logit gradient $\partial \ell / \partial z = w_\ell(p)(p-y) \sigma'(z)$, where $\sigma'(z) = p(1-p) \approx p$, and $w_\ell(p) \approx 1/p^3$. Thus, the gradient would explode at $1/p^2$ in critical boundary regions. The solution is the canonical link: solving $(\sigma_\ell^{-1})'(p) = w_\ell(p)$ yields $z = \int (2/p^2 + 2/(1-p)^2 + 2/p^3 + 2/(1-p)^3) dp$. Letting $u = 1/[p(1-p)]$, the inversion reduces to the largest real root of a quartic equation $u^4 - 12u^2 - 16u - z^2 = 0$ (solvable in closed form). After pairing, the gradient miraculously simplifies to $\partial \ell / \partial z = p - y$—a simple linear residual that never explodes or vanishes.
    - **Design Motivation**: Methods like CBSR often fail to scale to deep learning because using sigmoid with a custom loss results in uncontrollable gradients. By strictly maintaining the canonical pairing, the new loss becomes a drop-in replacement for log-loss, allowing seamless integration with any gradient-based learner like MLPs or XGBoost. The paper also mentions a trick: using the quartic root for the forward pass while bypassing auto-diff in the backward pass with the manual $p - y$ calculation further reduces overhead.

### Loss & Training
- **Training Objective**: Tailored proper scoring rule $\ell$ (derived from $w_\ell = w_{\text{task}}$), paired with the final layer activation $\sigma_\ell$ (closed-form solution of the quartic equation).
- **Evaluation Procedure**: 10-fold cross-fitting; downstream ATE calculated using IPW, Hajek, and AIPW estimators.
- **Variant**: A "Bias-Only Loss" obtained by removing the variance term ($1/p^3$), which reduces the inversion to a quadratic equation; the full MSE version performed better in practice.

## Key Experimental Results

### Main Results

**Standard Benchmarks (IHDP / Jobs / Kang-Schafer, linear backbone)**:

| Metric | Tailored ℓ (Ours) | Logistic (MLE) | Trim/Clip | CBPS / CBSR | EB / SBW |
|------|--------------------|----------------|-----------|-------------|----------|
| Std. MAE Dist. | Leftmost (Best) | Moderate | Competitive in some cases | Unstable | Unstable |
| Std. RMSE Dist. | Leftmost | Moderate | Competitive in some cases | Unstable | Unstable |
| Mean Bias | Closest to 0 | Significant deviation | Biased | Inconsistent | Inconsistent |
| Mean Rank (over IPW/Hajek/AIPW) | **1st for all three** | Mid-tier | Occasionally high | Unstable | Unstable |

Highlight: The largest improvement occurs in the most fragile vanilla IPW estimator—confirming that increased boundary punishment directly addresses IPW's core weakness.

**ACIC 2017 ($N=4802$, $d=58$, 32 DGPs)**:

| Backbone | Configs where Ours beats log-loss / Total |
|----------|------------------------------------------|
| MLP | **95 / 96** (≈ 99%) |
| XGBoost | 62 / 96 (≈ 65%) |

Total 96 = 32 DGPs × 3 downstream estimators (IPW / Hajek / AIPW).

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full Tailored Loss (Bias + Var terms) | Best RMSE | $w = 2/p^2 + 2/(1-p)^2 + 2/p^3 + 2/(1-p)^3$ |
| Bias-Only Loss | Slightly worse but > log-loss | No $1/p^3$ term, quadratic inversion, simpler engineering |
| Tailored loss + Standard Sigmoid | Unstable training / Gradient explosion | Validates the necessity of canonical pairing (Appendix B.1) |
| log-loss baseline | Significant lag under high selection | Insufficient boundary penalty |

### Key Findings
- Gains are highly correlated with **selection strength**: The RMSE improvement relative to log-loss is most significant when true propensity scores are concentrated near boundaries (strong selection); it rarely degrades performance in weak selection scenarios. This aligns perfectly with theory—the loss only applies pressure where IPW is truly fragile.
- Canonical pairing is not optional: Ablations show that using sigmoid with the new loss lead to $1/p^2$ gradient explosion, preventing convergence. This explains why CBSR is rarely used in deep networks.
- The same loss applies to IPW, Hajek, and AIPW estimators (sharing a common $d_{\text{task}}$ upper bound up to a constant), which was validated by experiments.
- While achieving near-total victory on MLPs (95/96), the win rate on XGBoost was lower (62/96)—likely because boosting step sizes inherently mitigate some boundary gradient explosion issues, absorbing part of the gains.

## Highlights & Insights
- **Second-order marriage of "Probability Estimation" and "Downstream Task"**: Using Taylor second-order curvature as an informational bridge to align loss geometry with task geometry. This recipe moves beyond "heuristics + post-processing" and is transferable to ATT, CATE, TMLE, and other two-stage ML pipelines.
- **Closed-form + Canonical Double Closure**: Both the loss and the activation function (quartic roots) are closed-form. The gradient simplifies to $p - y$. This means it can practically be a one-line replacement for log-loss, offering much higher utility than covariate balancing methods.
- **Paradigm Value of "Task-aware Loss"**: Compared to approaches like DML that "debias at the estimator level," this work "aligns at the probability level." The two are orthogonal and can be combined. This opens a door: any downstream task that can be expressed via $d_{\text{task}}$ can utilize this pipeline.
- **Honest reporting of failure cases**: The 65% win rate on XGBoost, the slight inferiority of Bias-Only, and the gain stratification under selection strength were reported honestly, facilitating better selection of applicable scenarios for future researchers.

## Limitations & Future Work
- **Local Curvature Assumption**: All conclusions rely on a local Taylor expansion of $\hat p$ near $p$. The author acknowledges there is "no global upper bound guarantee far from $p$." If the model is severely underfit, the second-order alignment may fail.
- **Canonical Mapping not always Closed-form**: Here, IPW happened to result in a quartic equation with closed-form roots. For other downstream tasks (e.g., certain CATE estimators), the inversion may lack an analytical solution, requiring numerical root-finding.
- **Computational Cost**: The forward pass for quartic roots is slower than sigmoid. While the backward pass is simplified via $p-y$, total training time is still slightly higher than log-loss baselines (Appendix B.4).
- **Uncovered Downstream Tasks**: The study only covered ATE for IPW/Hajek/AIPW. ATT, CATE, and TMLE were left as future work, requiring new derivations for their respective $d_{\text{task}}$.
- **Strong Second-Moment Assumption**: The derivation of $d_{\text{var}}$ depends on $\mathbb{E}[Y(1)^2 | X] \le M^2$, which may need re-evaluation in heavy-tailed outcome scenarios.

## Related Work & Insights
- **vs Covariate Balancing (CBPS / EB / SBW / Kernel Balancing)**: These use "covariate balance" as a surrogate for bias, with a weak theoretical link. Ours directly aligns with the second-order curvature of the downstream MSE, providing a tighter connection.
- **vs CBSR (Zhao 2019)**: CBSR also uses the proper scoring rule framework but forces moment constraints, breaking canonical pairing and causing gradient explosion in deep networks. Ours is the "engineered-to-work" version of CBSR.
- **vs DML (Chernozhukov 2018)**: DML uses Neyman orthogonality and cross-fitting at the estimator level to remove bias, but the nuisance model's training objective remains unchanged. This work modifies the nuisance model's objective, making them orthogonal and combinable.
- **vs Trimming / Clipping / Overlap Weighting**: These are post-hoc heuristics or changes to the estimand. This work is ante-hoc and leaves the estimand unchanged.
- **vs Calibration (Deshpande & Kuleshov 2024, etc.)**: Calibration is a post-processing step for probabilities; this work aligns the geometry during the training phase.
- **Transferable Insight**: Any pipeline where "probabilities are estimated first, then substituted into complex downstream formulas" (medical risk scores into decision curves, CTR into revenue estimates, recommendation scores into ranking metrics) can attempt to derive $d_{\text{task}}$ and apply this curvature matching recipe.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulating proper scoring rule "curvature matching" as a general recipe for task-aware losses is clear and theoretically sound, particularly with the first closed-form "tailored loss + canonical link" for IPW.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers IHDP/Jobs/Kang-Schafer and the high-dimensional ACIC 2017 benchmark across three estimators and two backbones. It is only slightly limited by focusing solely on the ATE estimand.
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations, motivations, and limitations are all well-explained. Figure 1 ($d_\ell$ vs $d_{\text{task}}$ vs $d_{\text{KL}}$) instantly illustrates why KL under-punishes at the boundaries. Remarks cover practical engineering tricks effectively.
- Value: ⭐⭐⭐⭐⭐ Provides a true "drop-in replacement for log-loss" with direct value for causal inference practice while establishing a reusable paradigm for "task-aware probabilistic learning."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)

</div>

<!-- RELATED:END -->
