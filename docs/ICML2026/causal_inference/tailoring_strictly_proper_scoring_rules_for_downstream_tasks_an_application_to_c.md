---
title: >-
  [Paper Note] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference
description: >-
  [ICML 2026][Causal Inference][proper scoring rule] This paper proposes a universal framework: by matching the local second-order curvature of the training loss $w_\ell(p)$ with the curvature of the downstream task error $w_{\text{task}}(p)$, one can derive a strictly proper scoring rule that is "geometrically aligned" with the downstream task. Applying this to IPW esti
tags:
  - ICML 2026
  - Causal Inference
  - proper scoring rule
  - IPW
  - ATE
  - canonical link
date: 2026-05-08
content_hash: 8828bf0977332585
---
# Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference

**Conference**: ICML 2026  
**arXiv**: [2606.03332](https://arxiv.org/abs/2606.03332)  
**Code**: To be confirmed (Paper name "tailored-psr", repository URL not explicitly given)  
**Area**: Causal Inference / Probability Estimation / IPW / Strictly Proper Scoring Rule  
**Keywords**: proper scoring rule, IPW, ATE, propensity score, canonical link

## TL;DR
This paper proposes a universal framework: by matching the local second-order curvature of the training loss $w_\ell(p)$ with the curvature of the downstream task error $w_{\text{task}}(p)$, one can derive a strictly proper scoring rule that is "geometrically aligned" with the downstream task. Applying this to IPW estimation of ATE yields a closed-form loss and a closed-form canonical activation function (solving a quartic equation), which consistently outperforms log-loss and covariate balancing baselines on IHDP, Jobs, Kang-Schafer, and ACIC 2017.

## Background & Motivation

**Background**: Many ML pipelines are "two-stage" — first estimating a conditional probability $\hat p(x)$, then feeding it as input to a downstream estimator (e.g., classification, risk assessment, or IPW/AIPW in causal inference). The first stage almost exclusively uses log-loss for training because it corresponds to KL divergence, which is statistically "natural."

**Limitations of Prior Work**: Log-loss is entirely task-agnostic regarding the downstream task. This is particularly fatal in IPW causal inference: when the true propensity score $e(x)$ approaches $0$ or $1$, the bias and variance of the IPW estimator explode due to $1/\hat e$ and $1/(1-\hat e)$ terms, yet the penalty of log-loss near the boundaries is far from sufficient. A model may have a low log-loss but a poor downstream ATE estimate.

**Key Challenge**: Existing remedies follow two paths, neither of which is clean — (i) post-processing heuristics: trimming/clipping, which sacrifices consistency for variance; (ii) covariate balancing (CBPS / Entropy Balancing / SBW / CBSR), which forces moment constraints into training. However, the theoretical link between "covariate balance" and "downstream bias minimization" is weak (Bruns-Smith & Feller, 2022), and approaches like CBSR break the pairing between proper scoring rules and their canonical links, leading to gradient explosion, making them nearly impossible to use with deep networks.

**Goal**: Can a training loss be constructed such that its geometry on the probability simplex directly mirrors the sensitivity of the downstream task to probability estimation errors?

**Key Insight**: The theory of proper scoring rules tells us that every strictly proper scoring rule is uniquely characterized by a non-negative weight function $w_\ell(q) = H_\ell''(q)$, and the second-order curvature of its induced divergence $d_\ell(p,q)$ at $q=p$ is exactly $w_\ell(p)$. If the upper bound of the downstream task error can also be written as a divergence $d_{\text{task}}$, one only needs to equate the two curvatures to make the local penalty of the loss "weighted by downstream sensitivity."

**Core Idea**: Derive task-specific losses using the curvature matching equation $w_\ell(p) = w_{\text{task}}(p)$, and solve the accompanying differential equation $(\sigma_\ell^{-1})'(p) = w_\ell(p)$ to obtain the canonical activation function, ensuring training stability.

## Method

### Overall Architecture

Ours addresses the misalignment between training loss and downstream tasks in "two-stage" estimation. While the first stage estimates a probability $p$ and the second stage substitutes $\hat p$ into a downstream estimator to calculate $\hat\theta$, the first stage typically uses task-agnostic log-loss. The authors transform this by expressing both the "local sensitivity of the training loss to probability errors" and the "local sensitivity of the downstream task to probability errors" as identical second-order curvatures and equating them — thus the geometry of the loss directly mirrors the geometry of the downstream task. For IPW estimation of ATE, the input is $(X_i, T_i)$ sample pairs, the output is a propensity score model $\hat e(x) = \sigma_\ell(f_\theta(x))$, which is then substituted into IPW / Hajek / AIPW to calculate ATE.

### Key Designs

**1. Universal Curvature Matching Framework: Geometrically Aligning Loss with Downstream Task**

The pain point is that log-loss is agnostic to the downstream task; low loss does not guarantee a good downstream estimate. The authors' solution is based on a fact from proper scoring rule theory: any strictly proper scoring rule is uniquely characterized by a non-negative weight function $w_\ell(q) = H_\ell''(q)$, where the second-order expansion of its induced divergence at $q=p$ is $d_\ell(p,q) = \tfrac{1}{2} w_\ell(p)(p-q)^2 + o((p-q)^2)$. If the downstream error bound can also be expressed as a divergence-type upper bound $\mathcal{E}(\theta,\hat\theta) \le \mathbb{E}[d_{\text{task}}(p,\hat p)]$, its expansion at $q=p$ is likewise $d_{\text{task}}(p,q) = \tfrac{1}{2} w_{\text{task}}(p)(p-q)^2 + o((p-q)^2)$. Since the expansions are isomorphic, one merely sets $w_\ell(p) = w_{\text{task}}(p)$ and integrates $H_\ell''(q) = w_\ell(q)$ twice to obtain the entropy function $H_\ell$. The loss is then given in closed form by $\ell(y,q) = -H_\ell(q) - H_\ell'(q)(y - q)$.

The "provability" of this chain relies on the additive decomposition of proper scoring rules $\mathbb{E}[\ell(T,\hat p)] = \mathbb{E}[d_\ell(p,\hat p)] + \mathbb{E}[H_\ell(p)]$: minimizing the training loss is equivalent to minimizing $\mathbb{E}[d_\ell]$. Adding the curvature matching condition explicitly links "training loss $\to$ second-order upper bound of downstream error." Compared to covariate balancing methods that target a proxy like "covariate balance," this provides direct alignment with the downstream error itself.

**2. IPW-tailored Loss: Embedding IPW Boundary Fragility into Weights**

The explosion of $1/\hat e$ and $1/(1-\hat e)$ terms when propensity scores approach $0$ or $1$ is a textbook vulnerability of IPW, and log-loss fails to penalize boundaries sufficiently. The authors apply the universal framework to IPW by performing a bias-variance decomposition on the MSE of $\hat\tau_{\text{ATE}}$: the bias term uses Cauchy-Schwarz to extract $\mathbb{E}[(e/\hat e - 1)^2]$ corresponding to $d_{\text{bias}}$, and the variance term (under bounded second moment assumptions) extracts $\mathbb{E}[Y(1)^2 e (1/\hat e - 1/e)^2]$ corresponding to $d_{\text{var}}$. Taking the second derivative at $q=p$ yields the task weight:

$$w_{\text{task}}(p) = \Big(\tfrac{2}{p^2} + \tfrac{2}{(1-p)^2}\Big) + \Big(\tfrac{2}{p^3} + \tfrac{2}{(1-p)^3}\Big),$$

where the first pair comes from bias and the second pair from variance. Compared to the log-loss weight $w(q) = \tfrac{1}{q(1-q)}$, the new weight explodes at $1/p^2$ and $1/p^3$ as $q \to 0$ or $q \to 1$, respectively, applying strict pressure exactly where IPW is most fragile. This effectively embeds $1/p^3$ sensitivity into the loss ante-hoc during training, rather than resorting to trimming (losing consistency) or changing the estimator (changing the target). The paper further points out that MSE bounds for Hajek and AIPW share the same $d_{\text{task}}$ (up to a scaling constant), making the loss universal across these estimators.

**3. Canonical Probability Mapping $\sigma_\ell$: Pulling Exploding Gradients back to Linear Residuals**

Directly applying the new loss after a sigmoid would cause immediate instability: the logit gradient $\partial \ell / \partial z = w_\ell(p)(p-y)\,\sigma'(z)$, where $\sigma'(z) = p(1-p) \approx p$ and $w_\ell(p) \approx 1/p^3$, leading to a gradient explosion at $1/p^2$ near boundaries. This is precisely why works like CBSR fail to integrate with deep networks. The authors use a canonical link instead: solving the differential equation $(\sigma_\ell^{-1})'(p) = w_\ell(p)$, i.e., $z = \int \big(\tfrac{2}{p^2} + \tfrac{2}{(1-p)^2} + \tfrac{2}{p^3} + \tfrac{2}{(1-p)^3}\big)\,dp$. Letting $u = 1/[p(1-p)]$, the inversion reduces to finding the maximum real root of the quartic equation $u^4 - 12u^2 - 16u - z^2 = 0$, which is solvable in closed form. After pairing, the gradient miraculously simplifies to $\partial \ell / \partial z = p - y$ — a linear residual that never explodes or vanishes, making the new loss a drop-in replacement for standard log-loss that can be used seamlessly with any gradient-based learner like MLP or XGBoost. An engineering trick is also used: the forward pass calculates the quartic root, while the backward pass manually implements $p - y$ to bypass automatic differentiation, further reducing overhead.

### Loss & Training
- Training objective: tailored proper scoring rule $\ell$ (obtained by integrating $w_\ell = w_{\text{task}}$), with the final layer activation being $\sigma_\ell$ (closed-form quartic solution).
- Evaluation pipeline: 10-fold cross-fitting; downstream ATE calculated using IPW, Hajek, and AIPW estimators.
- Alternative: A "Bias-Only Loss" that removes the variance term ($1/p^3$), reducing the inversion to a quadratic equation; the full MSE version performed better in practice.

## Key Experimental Results

### Main Results

**Standard benchmarks (IHDP / Jobs / Kang-Schafer, linear backbone)**:

| Metric | Tailored ℓ (Ours) | Logistic (MLE) | Trim/Clip | CBPS / CBSR | EB / SBW |
|------|--------------------|----------------|-----------|-------------|----------|
| Std. MAE Distribution | Far left (Best) | Moderate | Competitive in some cases | Unstable | Unstable |
| Std. RMSE Distribution | Far left | Moderate | Competitive in some cases | Unstable | Unstable |
| Mean Bias | Closest to 0 | Obvious deviation | Biased | Inconsistent | Inconsistent |
| Mean Rank (Across IPW/Hajek/AIPW) | **1st on all three** | Middle | Occasionally high | Unstable | Unstable |

Highlight: The gain is largest on the most fragile vanilla IPW estimator — confirming that increased boundary penalty directly fixes the core pain point of IPW.

**ACIC 2017 ($N=4802$, $d=58$, 32 DGPs)**:

| Backbone | Wins for Ours vs. log-loss (Configurations / Total) |
|----------|------------------------------------------|
| MLP | **95 / 96** (≈ 99%) |
| XGBoost | 62 / 96 (≈ 65%) |

Note: 96 = 32 DGPs × 3 downstream estimators (IPW / Hajek / AIPW).

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Full Tailored Loss (bias + variance) | Best RMSE | $w = 2/p^2 + 2/(1-p)^2 + 2/p^3 + 2/(1-p)^3$ |
| Bias-Only Loss | Slightly worse but still better than log-loss | Removed $1/p^3$ term; quadratic inversion; simpler engineering |
| Tailored loss + standard sigmoid | Training unstable / Gradient explosion | Verified necessity of canonical pairing (Appendix B.1) |
| log-loss baseline | Large lag under high selection strength | Insufficient boundary penalty |

### Key Findings
- Gains are highly correlated with **selection strength**: when true propensity scores are concentrated near boundaries (strong selection), the RMSE improvement of Ours relative to log-loss is most significant; it rarely performs worse in weak selection scenarios. This aligns perfectly with theory — the loss adds pressure only where IPW is truly fragile.
- Canonical pairing is not optional: ablation shows that using sigmoid with the new loss causes gradients to explode at $1/p^2$, preventing convergence; this explains why CBSR is rarely used in deep networks.
- The same loss applies to all three downstream estimators (IPW / Hajek / AIPW), as verified by experiments (theoretical MSE bounds share $d_{\text{task}}$ up to a constant).
- Almost "total victory" on MLP (95/96), while the win rate on XGBoost is lower (62/96) — likely because boosting step-size designs inherently mitigate boundary gradient explosion, partially absorbing the benefits.

## Highlights & Insights
- **Marriage of "Probability Estimation" and "Downstream Task" at the second order**: Using Taylor second-order curvature as an information bridge to align loss geometry with task geometry. This recipe moves beyond post-processing heuristics and can be transferred to ATT, CATE, TMLE, and other non-causal two-stage ML workflows.
- **Closed-form + Canonical Double Closure**: Both the loss and the activation function (quartic root) are closed-form, and the gradient simplifies to $p - y$. This means it can practically replace log-loss in one line, being far more useful than covariate balancing methods.
- **Paradigm Value of "task-aware loss"**: Unlike DML which "debiases at the estimator level," Ours "aligns at the probability level." The two are orthogonal and can be combined. This opens a door: any downstream task that can define $d_{\text{task}}$ can follow this pipeline.
- **Honest reporting of experimental failures**: Cases like the 65% win rate on XGBoost and the divergence in gains under strong vs. weak selection are honestly reported, facilitating future selection of applicable scenarios.

## Limitations & Future Work
- **Local Curvature Assumption**: All conclusions are built on a local Taylor expansion of $\hat p$ around $p$; the authors admit "no global upper bound guarantees in regions far from $p$." If the model is severely underfit ($\hat p$ far from truth), the curvature matching alignment may fail.
- **Canonical Mappings aren't always closed-form**: IPW happens to result in a quartic equation with closed-form roots. For other downstream tasks (e.g., certain CATE estimators), the inversion of the differential equation might lack an analytical solution, requiring numerical root-finding.
- **Computational Overhead**: Solving a quartic equation in the forward pass is still slower than sigmoid. Although simplified $p - y$ in the backward pass helps, total training time remains slightly higher than the log-loss baseline (Appendix B.4).
- **Uncovered Downstream Tasks**: The authors only addressed ATE for IPW / Hajek / AIPW; other estimands like ATT / CATE / TMLE are left for future work, requiring new derivations for $d_{\text{task}}$.
- **Strong Second-Moment Assumption**: The derivation of $d_{\text{var}}$ depends on $\mathbb{E}[Y(1)^2 | X] \le M^2$, which may need re-evaluation for heavy-tailed outcome scenarios.

## Related Work & Insights
- **vs Covariate Balancing (CBPS / EB / SBW / Kernel Balancing)**: These use "covariate balance" as a proxy for bias with weak theoretical links; Ours directly aligns with the second-order curvature of downstream MSE, providing a much tighter link.
- **vs CBSR (Zhao 2019)**: CBSR also uses the proper scoring rule framework but forces moment constraints, breaking canonical pairing and causing gradient explosion in deep nets. Ours maintains canonical pairing, serving as an "engineered-to-work" version of CBSR.
- **vs DML (Chernozhukov 2018)**: DML uses Neyman orthogonality + cross-fitting at the estimator level to debias, but the nuisance model training objective is unchanged. Ours modifies the training objective itself; they are orthogonal and additive.
- **vs Trimming / Clipping / Overlap Weighting**: These are post-hoc heuristics or change the estimand; Ours is ante-hoc and keeps the estimation target unchanged.
- **vs Calibration (Deshpande & Kuleshov 2024, etc.)**: Calibration is also post-processing; Ours aligns geometry directly during the training phase.
- **Transferable Insight**: Any pipeline that "estimates probability first, then substitutes into a complex downstream formula" (medical risk scores into decision curves, CTR into revenue estimation, etc.) could try to define a $d_{\text{task}}$ and apply this curvature matching recipe.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Turning the "curvature matching" of proper scoring rule theory into a universal recipe for task-aware loss is clear and theoretically sound; it is the first to achieve closed-forms for both "tailored loss + canonical link" in IPW.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers IHDP/Jobs/Kang-Schafer and ACIC 2017 benchmarks with multiple estimators and backbones. It would have been perfect if other estimands like CATE/ATT were included.
- Writing Quality: ⭐⭐⭐⭐⭐ Derivations, motivations, and limitations are all thoroughly explained. Figure 1 ($d_\ell$ vs $d_{\text{task}}$ vs $d_{\text{KL}}$) instantly clarifies why KL under-penalizes boundaries. Remarks provide helpful engineering tricks.
- Value: ⭐⭐⭐⭐⭐ A true "drop-in replacement for log-loss" with direct value for causal inference practice; it establishes a reusable paradigm for "task-aware probabilistic learning."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[CVPR 2025\] Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](../../CVPR2025/causal_inference/joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)
- [\[ICML 2025\] Causal Abstraction Inference under Lossy Representations](../../ICML2025/causal_inference/causal_abstraction_inference_under_lossy_representations.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[CVPR 2026\] CGU-Bayes: Causal Graph Uncertainty-Guided Bayesian Inference for Domain Generalization](../../CVPR2026/causal_inference/cgu-bayes_causal_graph_uncertainty-guided_bayesian_inference_for_domain_generali.md)

</div>

<!-- RELATED:END -->
