---
title: >-
  [Paper Note] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation
description: >-
  [ICML 2026][learning_theory][MMD] The paper provides the first PAC-Bayes upper bound for test-time adaptation in the form of "Target Risk $\le$ Source Empirical Risk + KL Complexity + MMD Shift Term." It interprets MMD-balls as credal sets in the sense of Walley, naturally separating aleatoric and epistemic uncertainty via "upper and lower risk interva
tags:
  - ICML 2026
  - learning_theory
  - MMD
  - credal set
date: 2026-05-08
content_hash: 55f53a873b9eefe7
---
# MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation

**Conference**: ICML 2026  
**arXiv**: [2605.21783](https://arxiv.org/abs/2605.21783)  
**Code**: Not publicly available  
**Area**: Test-Time Adaptation / PAC-Bayes Theory / Uncertainty Quantization  
**Keywords**: PAC-Bayes Bound, MMD, Credal Set, Test-Time Adaptation, Epistemic Uncertainty

## TL;DR
The paper provides the first PAC-Bayes upper bound for test-time adaptation in the form of "Target Risk $\le$ Source Empirical Risk + KL Complexity + MMD Shift Term." It interprets MMD-balls as credal sets in the sense of Walley, naturally separating aleatoric and epistemic uncertainty via "upper and lower risk intervals," providing computable criteria for "when to adapt and when to abstain."

## Background & Motivation

**Background**: Test-time adaptation (TTA) methods, represented by TENT, EATA, SAR, and MEMO, significantly improve accuracy under distribution shift by fine-tuning BN layers or parameters using test batch statistics.

**Limitations of Prior Work**: Existing TTA methods lack formal guarantees; they do not indicate the degree of shift a model can tolerate or when adaptation should be avoided altogether. In safety-critical scenarios like autonomous driving or medical imaging, models may degrade silently without TTA detection.

**Key Challenge**: Current predictive uncertainty tools (Bayesian networks, ensembles, conformal prediction) often conflate aleatoric uncertainty (data noise) with epistemic uncertainty (unknown distribution). TTA specifically lacks "distribution-level epistemic uncertainty." Additionally, past PAC-Bayes applications in domain adaptation relied on the NP-hard $\mathcal{H}$-divergence (Germain 2013), which is computationally intractable.

**Goal**: To develop a unified theoretical framework that (i) is dominated by the computable MMD, (ii) provides finite-sample upper bounds, (iii) naturally separates epistemic and aleatoric uncertainty, and (iv) determines when adaptation is necessary.

**Key Insight**: The authors observe that an MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ mathematically satisfies all requirements of a Walley credal set—it represents the set of all distributions indistinguishable from the source distribution at resolution $\varepsilon$.

**Core Idea**: By assuming RKHS-Lipschitz continuity, the upper bound of the "loss difference between target and source distributions" is expressed as $L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$, which is integrated into the classical PAC-Bayes bound. By replacing MMD with $\varepsilon$ and taking the supremum, the worst-case risk bound over the credal set is obtained, naturally yielding lower/upper risk intervals.

## Method

### Overall Architecture
The framework is not a single "algorithm" but a "theory + decision criterion" composed of four interconnected components:

1.  **PAC-Bayes Upper Bound (Theorem 1)**: Under covariate shift and RKHS-Lipschitz loss assumptions, the target risk $R_{P_t}(\rho)$ is decomposed into "Source Empirical Risk + KL Complexity + MMD Shift Penalty."
2.  **Finite-Sample Version (Theorem 3)**: Replaces population MMD with the unbiased estimator $\widehat{\mathrm{MMD}}_u$, utilizing sub-Gaussian concentration from Sutherland/Tolstikhin to provide a closed-form width $\varepsilon_{m,n}(\delta)$.
3.  **Credal Set Geometry (Definition 5 + Proposition 7 + Corollary 9)**: Interprets the MMD-ball as a credal set, providing the worst-case risk $\overline{R}_\varepsilon(\rho)$ and best-case risk $\underline{R}_\varepsilon(\rho)$. The imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon$ directly quantifies epistemic uncertainty.
4.  **Geodesic Preservation (Proposition 10 + Corollary 11)**: Proves that the geodesic distance difference between source and target neighborhoods in RKHS geometry is controlled by $\sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)$, providing a theoretical explanation for "kernel-guided adaptation protecting rare classes."

These components together constitute "distribution-level epistemic intelligence": monitor MMD $\to$ calculate credal intervals $\to$ trigger adapt/abstain.

### Key Designs

**1. PAC-Bayes + MMD Shift Penalty: Explicitly incorporating "distribution shift" into the generalization bound (Theorem 1 / 3)**

The lack of guarantees in TTA stems from the fact that previous tools for bounding "source-target risk differences" were either intractable ($\mathcal{H}$-divergence) or lacked finite-sample versions. This work utilizes MMD. Under Assumption 1 (conditional expected loss resides in RKHS with bounded norm, $L(w,\cdot)\in\mathcal{H}$ and $\|L(w,\cdot)\|_\mathcal{H}\le L_\mathcal{H}$), the risk difference is bounded by an MMD term using the reproducing property and Cauchy-Schwarz: $|R_{P_t}(\rho)-R_{P_s}(\rho)|\le L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$. Combined with the McAllester PAC-Bayes bound, this yields the first TTA bound with an explicit distribution shift penalty:

$$R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}(\rho\|\pi)+\log(2\sqrt{n}/\delta)}{2n}}+L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t).$$

Theorem 3 substitutes the unobservable population MMD with the unbiased estimator $\widehat{\mathrm{MMD}}_u$ and a concentration width $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$, making the entire bound computable with $O((m+n)^2)$ complexity and an optimal $O(1/\sqrt{n})$ rate. The bound grows linearly with MMD, meaning it loosens smoothly as shift increases—characteristic of epistemic uncertainty.

**2. MMD-Ball as Credal Set: Upgrading point estimates to lower-upper risk intervals to separate aleatoric and epistemic uncertainty**

A single upper bound is insufficient for safety scenarios requiring uncertainty decomposition. The key observation is that the MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ is mathematically a credal set—all distributions indistinguishable from the source at resolution $\varepsilon$. Due to the linearity of characteristic kernels, this set is convex and weakly closed (Lemma 6), allowing for the worst-case risk:

$$\sup_{Q\in\mathcal{C}_\varepsilon(P_s)}R_Q(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}+\log}{2n}}+L_\mathcal{H}\varepsilon,$$

The best-case risk $\underline{R}_\varepsilon$ is derived symmetrically using Germain’s PAC-Bayes lower bound. The imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{(\mathrm{KL}+\log)/2n}+2L_\mathcal{H}\varepsilon$ separates the two uncertainties: the first term decays with source sample size $n$ (estimation/aleatoric uncertainty), while the second grows linearly with $\varepsilon$ (distribution/epistemic uncertainty).

**3. RKHS Geodesic Preservation: Geometrically explaining why MMD-bounded adaptation protects rare classes better than entropy minimization**

In practice, kernel-guided adaptation tends to preserve minority classes better than entropy minimization (like TENT). This is formalized as a geometric theorem. Under Assumption 2 (encoder factorizes into a bounded linear layer and an RKHS feature map $f_\theta=W\cdot \phi_\theta$, $\|W\|_{op}\le C_W$), applying local linearization to the RBF kernel $d_k(x,y)=\sqrt{2\gamma}\|f_\theta(x)-f_\theta(y)\|+O(\bar\epsilon^2)$ reveals that shift in neighborhood geodesic distance is controlled by MMD:

$$\big|\mathbb{E}_{y\sim P_s}[d_k(x_i,y)]-\mathbb{E}_{y\sim P_t}[d_k(x_i,y)]\big|\le \sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)+O(\bar\epsilon^2).$$

Crucially, this bound is independent of class frequency. Minority classes with "small but compact" local structures are naturally protected, whereas entropy minimization might misidentify low-density areas as high-entropy regions and collapse them, erasing rare classes.

### Loss & Training
The paper does not propose a new training algorithm. Results rely on two assumptions: (1) Assumption 1 requires $L(w,\cdot)$ to be in RKHS with a bounded norm (approximated via kernel universality for softmax + RBF kernels); (2) Assumption 2 requires the encoder to decompose into bounded linear + RKHS maps (approximated in the NTK regime or via explicit MMD regularization).

## Key Experimental Results

### Main Results
This is a theoretical work; **no full experimental tables are provided.** "Data" is presented through theorems summarized below:

| Inequality | Primary Terms | Description |
|---|---|---|
| Theorem 1 (Population MMD) | $R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{(\mathrm{KL}+\log)/(2n)}+L_\mathcal{H}\,\mathrm{MMD}$ | First TTA PAC-Bayes bound with explicit MMD penalty |
| Theorem 3 (Finite Sample) | Above + $L_\mathcal{H}\,(\widehat{\mathrm{MMD}}_u+\varepsilon_{m,n}(\delta/2))$ | Entirely computable, $O((m+n)^2)$ complexity |
| Proposition 7 (Worst-case) | $\sup_{Q\in\mathcal{C}_\varepsilon(P_s)} R_Q(\rho)\le \hat R_{P_s}(\rho)+\sqrt{\cdots}+L_\mathcal{H}\varepsilon$ | Bounded worst-case risk over the credal set |
| Corollary 9 (Interval Width) | $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{\cdots/2n}+2L_\mathcal{H}\varepsilon$ | Epistemic uncertainty = Estimation + Shift terms |
| Proposition 10 (Geometry) | $|\mathbb{E}_{P_s}[d_k]-\mathbb{E}_{P_t}[d_k]|\le \sqrt{2\gamma}C_W\,\mathrm{MMD}$ | MMD controls RKHS geodesic distance drift |

### Ablation Study
| Assumption / Setting | Impact | Description |
|---|---|---|
| Disable RKHS-Lipschitz (Assum. 1) | Theorem 1 fails | Shift term no longer has a linear MMD upper bound |
| Use posterior mean $\rho$ instead of dist. | KL term vanishes | PAC-Bayes complexity becomes 0, loses prior regularization |
| Replace MMD with $\mathcal{H}$-divergence | Returns to Germain 2013 | Bound holds but becomes computationally intractable |
| Disable covariate shift assumption | $L(w,x)$ inconsistent | Shift term requires joint $(x,y)$ MMD, requires kernel extension |
| Non-characteristic kernel | $\varepsilon=0 \not\implies Q=P_s$ | Credal set degenerates |

### Key Findings
- As $\mathrm{MMD}(P_s,P_t)\to 0$, the bound restores the classical PAC-Bayes bound. Under large shift, the bound loosens "smoothly" rather than "collapsing," which is the expected behavior for epistemic uncertainty.
- Decision Criterion: Given a tolerance $r_{\max}$, if $\underline{R}_\varepsilon(\rho)>r_{\max}$, even the best target distribution is unacceptable—model should **abstain**. If $\overline{R}_\varepsilon(\rho)<r_{\max}$, even the worst distribution is safe—model does not need to **adapt**. Adaptation is meaningful only when $\overline{R}_\varepsilon>r_{\max}>\underline{R}_\varepsilon$.
- $\varepsilon$ can be calibrated using the asymptotic null distribution of the MMD two-sample test at level $\alpha$: rejecting $H_0:P_t=P_s$ is equivalent to $\widehat{\mathrm{MMD}}_u>\varepsilon_\alpha$.

## Highlights & Insights
- Connecting Walley's 1991 imprecise probability with PAC-Bayes generalization via MMD-balls merges 60s decision theory, 90s statistical learning, and 2010s kernel methods.
- The linear growth of the bound relative to $\varepsilon$ indicates MMD is a "well-behaved geometric quantity" for distribution shifts, guiding future distribution-shift-aware loss designs.
- Geodesic preservation explained: Entropy minimization collapses low-density regions, whereas MMD-controlled adaptation is frequency-independent, preserving minority class geometry.

## Limitations & Future Work
- Assumption 1 is "informally supported" by RBF universality for deep nets but lacks a rigorous construction.
- Assumption 2 (factorization) is an approximation for general ResNet/ViT architectures; tighter bounds for specific architectures are needed.
- MMD convergence $O(1/\sqrt n)$ is minimax optimal but may be loose in practice; adaptive kernel selection could improve data-dependent rates.
- **No empirical experiments**: Theorems are not validated on standard TTA benchmarks (CIFAR-10-C, ImageNet-C), representing a clear next step.

## Related Work & Insights
- **vs. Germain et al. 2013**: They use $\mathcal{H}$-divergence (NP-hard, no finite-sample version); this work uses MMD ($O((m+n)^2)$ and computable).
- **vs. TENT / EATA / SAR**: These rely on empirical tricks (entropy, sharpness); this work provides formal criteria for when to adapt.
- **vs. BNN / Ensembles**: Traditional predictive uncertainty conflates error types; this work separates them using credal set widths.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Connects PAC-Bayes, kernel mean embedding, and Walley credal sets for TTA.
- Experimental Thoroughness: ⭐⭐ Purely theoretical short paper with no empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, concise notation, and elegant proofs.
- Value: ⭐⭐⭐⭐ Provides computable criteria for "abstain vs. adapt" for safety-critical ML.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICML 2026\] Realizable Bayes-Consistency for General Metric Losses](realizable_bayes-consistency_for_general_metric_losses.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)

</div>

<!-- RELATED:END -->
