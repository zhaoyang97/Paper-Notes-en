---
title: >-
  [Paper Note] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation
description: >-
  [ICML 2026][Test-time adaptation][PAC-Bayes bounds] The paper provides the first PAC-Bayes upper bound for test-time adaptation in the form of "target risk $\le$ source empirical risk + KL complexity + MMD distribution shift." It interprets MMD-balls as credal sets in the sense of Walley, naturally separating aleatoric and epistemic uncertainty via "upper/lower risk intervals," and providing computable criteria for "when to adapt and when to abstain."
tags:
  - "ICML 2026"
  - "Test-time adaptation"
  - "PAC-Bayes theory"
  - "Uncertainty quantification"
  - "PAC-Bayes bounds"
  - "MMD"
  - "credal set"
  - "epistemic uncertainty"
date: 2026-05-08
content_hash: ebf81e256777db2b
---

# MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation

**Conference**: ICML 2026  
**arXiv**: [2605.21783](https://arxiv.org/abs/2605.21783)  
**Code**: Not publicly available  
**Area**: Test-time adaptation / PAC-Bayes theory / Uncertainty quantification  
**Keywords**: PAC-Bayes bounds, MMD, credal set, test-time adaptation, epistemic uncertainty

## TL;DR
The paper provides the first PAC-Bayes upper bound for test-time adaptation in the form of "target risk $\le$ source empirical risk + KL complexity + MMD distribution shift." It interprets MMD-balls as credal sets in the sense of Walley, naturally separating aleatoric and epistemic uncertainty via "upper/lower risk intervals," and providing computable criteria for "when to adapt and when to abstain."

## Background & Motivation

**Background**: Test-time adaptation (TTA) methods, such as TENT, EATA, SAR, and MEMO, significantly improve accuracy under distribution shift by fine-tuning Batch Normalization (BN) or model parameters using test batch statistics.

**Limitations of Prior Work**: Existing TTA methods lack formal guarantees; they provide no indication of the degree of shift a model can tolerate, nor do they specify when adaptation should be avoided altogether. In safety-critical scenarios like autonomous driving or medical imaging, a model may silently degrade without the TTA process being aware.

**Key Challenge**: Current predictive uncertainty tools (Bayesian networks, ensembles, conformal prediction) conflate aleatoric uncertainty (intrinsic data noise) with epistemic uncertainty (distributional ignorance), whereas TTA specifically lacks "epistemic uncertainty at the distribution level." Furthermore, prior PAC-Bayes approaches for domain adaptation relied on the NP-hard $\mathcal{H}$-divergence (Germain 2013), making them computationally impractical.

**Goal**: To develop a unified theoretical framework that (i) is dominated by the computable MMD, (ii) provides finite-sample upper bounds, (iii) naturally separates epistemic and aleatoric uncertainty, and (iv) determines when adaptation is necessary.

**Key Insight**: The authors observe that the MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ mathematically satisfies all requirements of a Walley credal set—representing the set of all distributions indistinguishable from the source distribution at resolution $\varepsilon$.

**Core Idea**: By using the RKHS-Lipschitz assumption, the risk difference between the target and source distributions is upper-bounded by $L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$, which is then integrated into the classic PAC-Bayes bound. By taking the supremum over the MMD-ball substituted for $\varepsilon$, the worst-case risk bound on the credal set is obtained, yielding natural lower/upper risk intervals.

## Method

### Overall Architecture
The framework is not a single "algorithm" but a "theory + decision criterion" composed of four interconnected components:

1.  **PAC-Bayes Upper Bound (Theorem 1)**: Under the covariate shift and RKHS-Lipschitz loss assumptions, the target risk $R_{P_t}(\rho)$ is decomposed into "source empirical risk + KL complexity + MMD shift penalty."
2.  **Finite-Sample Version (Theorem 3)**: The population MMD is replaced with the unbiased MMD estimator $\widehat{\mathrm{MMD}}_u$, using the sub-Gaussian concentration of Sutherland/Tolstikhin to provide a closed-form width $\varepsilon_{m,n}(\delta)$.
3.  **Credal Set Geometry (Definition 5 + Proposition 7 + Corollary 9)**: Interpreting the MMD-ball as a credal set to yield the worst-case risk $\overline{R}_\varepsilon(\rho)$ and the best-case risk $\underline{R}_\varepsilon(\rho)$. The imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon$ directly quantifies epistemic uncertainty.
4.  **Geodesic Preservation (Proposition 10 + Corollary 11)**: Proving that within the RKHS geometry, the difference in geodesic distances between source and target neighborhoods is controlled by $\sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)$, providing a theoretical explanation for "kernel-guided adaptation protecting rare classes."

These four components together constitute "epistemic intelligence at the distribution level": monitoring MMD $\to$ calculating credal intervals $\to$ triggering adapt/abstain.

### Key Designs

**1. PAC-Bayes + MMD Shift Penalty: Explicitly incorporating "distribution shift" into the generalization bound (Theorem 1 / 3)**

TTA methods lack guarantees because past tools for bounding the "source-target risk difference" were either uncomputable ($\mathcal{H}$-divergence is NP-hard) or lacked finite-sample versions. This work employs MMD as a more practical metric. Under Assumption 1 (conditional expected loss resides in the RKHS with bounded norm, $L(w,\cdot)\in\mathcal{H}$ and $\|L(w,\cdot)\|_\mathcal{H}\le L_\mathcal{H}$), the reproducing property and Cauchy-Schwarz inequality bound the risk difference as $|R_{P_t}(\rho)-R_{P_s}(\rho)|\le L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$. Combined with the classic McAllester PAC-Bayes bound, this yields the first explicit bound with a distribution shift penalty for TTA:

$$R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}(\rho\|\pi)+\log(2\sqrt{n}/\delta)}{2n}}+L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t).$$

Theorem 3 replaces population MMD with an unbiased estimate $\widehat{\mathrm{MMD}}_u$ and concentration width $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$, making the entire bound computable with $O((m+n)^2)$ complexity and a minimax optimal rate of $O(1/\sqrt{n})$. The benefit is that the MMD term grows linearly; as the shift increases, the bound loosens smoothly rather than collapsing—precisely the behavior expected of epistemic uncertainty.

**2. MMD-balls as Credal Sets: Upgrading point estimates to lower-upper risk intervals to separate aleatoric and epistemic uncertainty**

An upper bound alone is insufficient for safety scenarios; one must know the magnitude of uncertainty and its source. The critical observation here is that the MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ is mathematically a credal set—the set of all distributions indistinguishable from the source at resolution $\varepsilon$. Due to the linearity of characteristic kernels, it is convex and weakly closed (Lemma 6), allowing for the worst-case risk over the set:

$$\sup_{Q\in\mathcal{C}_\varepsilon(P_s)}R_Q(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}+\log}{2n}}+L_\mathcal{H}\varepsilon.$$

The infimum direction uses Germain's PAC-Bayes lower bound to symmetrically derive the best-case risk $\underline{R}_\varepsilon$. The resulting imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{(\mathrm{KL}+\log)/2n}+2L_\mathcal{H}\varepsilon$ physically decouples the two types of uncertainty: the first term decays with source sample size $n$ (estimation uncertainty/aleatoric), while the second term increases linearly with $\varepsilon$ (distributional uncertainty/epistemic). This connects Walley’s behaviorist imprecise probability with PAC-Bayes.

**3. RKHS Geodesic Preservation: Geometrically explaining why MMD-bounded adaptation protects rare classes better than entropy minimization**

Empirically, kernel-guided adaptation is less likely to erase minority classes than entropy minimization (like TENT). This work elevates this intuition to a geometric theorem. Under Assumption 2 (encoder factorizes into a bounded linear layer and an RKHS feature map $f_\theta=W\cdot \phi_\theta$, $\|W\|_{op}\le C_W$), local linearization of the RBF kernel $d_k(x,y)=\sqrt{2\gamma}\|f_\theta(x)-f_\theta(y)\|+O(\bar\epsilon^2)$ combined with the reverse triangle inequality shows that geodesic distance drift is also controlled by MMD:

$$\big|\mathbb{E}_{y\sim P_s}[d_k(x_i,y)]-\mathbb{E}_{y\sim P_t}[d_k(x_i,y)]\big|\le \sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)+O(\bar\epsilon^2).$$

Crucially, this bound is independent of class frequency. Minority classes, which are small but structurally compact, have their local geometry naturally protected. In contrast, entropy minimization misinterprets low-density areas as high-entropy zones and flattens them, erasing rare classes.

## Loss & Training
The paper does not propose a new training algorithm; results rely on two assumptions: (1) Assumption 1 requires $L(w,\cdot)$ to be in the RKHS with bounded norm (approximate for softmax + RBF kernels via universality); (2) Assumption 2 requires the encoder to be decomposable into bounded linear + RKHS feature maps (approximate in the NTK regime, explicit MMD regularization, or spectral normalization). Section 8 discusses relaxing (1) to $\mathbb{E}_{w\sim\rho}[\|L(w,\cdot)\|_\mathcal{H}]\le L_\mathcal{H}$.

## Key Experimental Results

### Main Results
This is a theoretical work and **does not contain complete experimental tables**. The "data" are presented as theorems, summarized below:

| Inequality | Primary Term | Description |
|---|---|---|
| Theorem 1 (Population MMD) | $R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{(\mathrm{KL}+\log(2\sqrt n/\delta))/(2n)}+L_\mathcal{H}\,\mathrm{MMD}(P_s,P_t)$ | First explicit MMD shift penalty PAC-Bayes bound for TTA |
| Theorem 3 (Finite Sample) | Above + $L_\mathcal{H}\,(\widehat{\mathrm{MMD}}_u+\varepsilon_{m,n}(\delta/2))$, $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$ | Entirely computable, $O((m+n)^2)$ |
| Proposition 7 (Worst-case) | $\sup_{Q\in\mathcal{C}_\varepsilon(P_s)} R_Q(\rho)\le \hat R_{P_s}(\rho)+\sqrt{\cdots}+L_\mathcal{H}\varepsilon$ | Bounded worst-case risk over the entire credal set |
| Corollary 9 (Interval Width) | $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{\cdots/2n}+2L_\mathcal{H}\varepsilon$ | Epistemic uncertainty = estimation term + shift term |
| Proposition 10 (Geometry) | $|\mathbb{E}_{P_s}[d_k]-\mathbb{E}_{P_t}[d_k]|\le \sqrt{2\gamma}C_W\,\mathrm{MMD}+O(\bar\epsilon^2)$ | MMD controls RKHS geodesic distance drift |

### Ablation Study

| Assumption / Setting | Impact | Description |
|---|---|---|
| Disable RKHS-Lipschitz (Assum. 1) | Theorem 1 fails | Shift term no longer has a linear MMD upper bound |
| Use posterior mean ρ instead of distribution | KL term vanishes | PAC-Bayes complexity becomes 0, losing prior regularization |
| Replace MMD with $\mathcal{H}$-divergence | Reverts to Germain 2013 | Bound remains valid but becomes uncomputable |
| Disable Covariate Shift assumption | $L(w,x)$ differs between source/target | Shift term requires joint $(x,y)$ MMD, needing kernel expansion |
| Non-characteristic kernel | $\varepsilon=0$ no longer implies $Q=P_s$ | Credal set degenerates |

### Key Findings
- When $\mathrm{MMD}(P_s,P_t)\to 0$, the bound precisely recovers the classic PAC-Bayes bound. Under large distribution shifts, the bound loosens smoothly, matching the expected behavior of epistemic uncertainty.
- Decision Criterion: Given a tolerance $r_{\max}$, if $\underline{R}_\varepsilon(\rho)>r_{\max}$, even the best target distribution is unacceptable—the model should abstain. If $\overline{R}_\varepsilon(\rho)<r_{\max}$, the model is safe even under the worst distribution—adaptation is unnecessary. Adaptation is meaningful only when $\overline{R}_\varepsilon>r_{\max}>\underline{R}_\varepsilon$.
- $\varepsilon$ can be calibrated using the asymptotic null distribution of the MMD two-sample test at level $\alpha$: rejecting $H_0:P_t=P_s$ is equivalent to $\widehat{\mathrm{MMD}}_u>\varepsilon_\alpha$, linking credal set width to evidentiary strength.

## Highlights & Insights
- Connecting Walley's 1991 imprecise probability with PAC-Bayes generalization via MMD-balls is a rare fusion of 60s decision theory, 90s statistical learning, and 2010s kernel methods.
- The linear loosening of the bound with $\varepsilon$ indicates MMD is a "benign geometric quantity" for distribution shifts, offering guidance for designing distribution-shift-aware losses or OOD detectors.
- The geodesic preservation lemma provides a formal explanation for why entropy minimization harms minority classes (as it encourages flattening in low-density areas), while MMD-controlled adaptation is class-frequency independent.

## Limitations & Future Work
- Assumption 1 for deep networks + softmax loss is "informally supported by RBF universality" but lacks strict construction.
- Assumption 2 only approximately holds for standard ResNet/ViT in the NTK limit or with explicit MMD regularization; tight bounds for specific architectures are needed.
- While the $O(1/\sqrt n)$ MMD convergence rate is minimax optimal, it may be loose in practice; adaptive kernel selection could provide tighter rates.
- The paper contains **no empirical experiments** on TTA benchmarks (CIFAR-10-C, ImageNet-C), leaving the validation of bound tightness and decision criteria as an open next step.
- The integration with conformal prediction ($\alpha(\varepsilon)=\alpha_0+g(\varepsilon)$) is only a sketch.

## Related Work & Insights
- **vs. Germain et al. 2013 (PAC-Bayes domain adaptation)**: They use $\mathcal{H}$-divergence (NP-hard, no finite-sample version); this work uses MMD ($O((m+n)^2)$ and computable).
- **vs. TENT / EATA / SAR (TTA Algorithms)**: These rely on empirical tricks; this work provides formal criteria for when to adapt.
- **vs. Bayesian Neural Nets / Ensembles**: Those conflate aleatoric and epistemic uncertainty; this work explicitly separates them via credal set width.
- **vs. Conformal prediction**: Conformal prediction gives individual coverage guarantees but doesn't bound aggregate risk; this work provides distribution-level risk intervals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to bridge PAC-Bayes, kernel mean embedding, and Walley credal sets in TTA.
- Experimental Thoroughness: ⭐⭐ Purely theoretical short paper with no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, concise notation, and accessible proofs.
- Value: ⭐⭐⭐⭐ Provides computable criteria for adapt-vs-abstain decisions for safety-critical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICLR 2026\] Epistemic Uncertainty Quantification To Improve Decisions From Black-Box Models](../../ICLR2026/learning_theory/epistemic_uncertainty_quantification_to_improve_decisions_from_black-box_models.md)
- [\[ICML 2026\] Learning Credal Ensembles via Distributionally Robust Optimization](learning_credal_ensembles_via_distributionally_robust_optimization.md)
- [\[ICLR 2026\] Efficient Credal Prediction through Decalibration](../../ICLR2026/learning_theory/efficient_credal_prediction_through_decalibration.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
