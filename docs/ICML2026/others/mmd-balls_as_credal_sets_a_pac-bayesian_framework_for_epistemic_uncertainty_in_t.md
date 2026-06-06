---
title: >-
  [Paper Note] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation
description: >-
  [ICML 2026][PAC-Bayes Bound] The paper provides the first PAC-Bayes upper bound for test-time adaptation formulated as "Target Risk $\le$ Source Empirical Risk + KL Complexity + MMD Distribution Shift Term." It interpret…
tags:
  - "ICML 2026"
  - "PAC-Bayes Bound"
  - "MMD"
  - "credal set"
  - "Test-Time Adaptation"
  - "Epistemic Uncertainty"
date: 2026-05-08
content_hash: 50f9b89e5fe6e2ae
---

# MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation

**Conference**: ICML 2026  
**arXiv**: [2605.21783](https://arxiv.org/abs/2605.21783)  
**Code**: Paper not yet public  
**Area**: Test-Time Adaptation / PAC-Bayes Theory / Uncertainty Quantization  
**Keywords**: PAC-Bayes Bound, MMD, credal set, Test-Time Adaptation, Epistemic Uncertainty  

## TL;DR
The paper provides the first PAC-Bayes upper bound for test-time adaptation formulated as "Target Risk $\le$ Source Empirical Risk + KL Complexity + MMD Distribution Shift Term." It interprets MMD-balls as Walley-style credal sets, naturally separating aleatoric and epistemic uncertainty through "upper and lower risk intervals," and provides a computable criterion for "when to adapt and when to abstain."

## Background & Motivation

**Background**: Test-time adaptation (TTA) methods represented by TENT, EATA, SAR, and MEMO significantly improve accuracy under distribution shifts. Their approach typically involves fine-tuning BN layers/parameters using test batch statistics.

**Limitations of Prior Work**: All these TTA methods lack formal guarantees—they neither indicate the degree of shift the model can tolerate nor specify when adaptation should be avoided altogether. In safety-critical scenarios like autonomous driving or medical imaging, models may silently degrade without the TTA process being aware.

**Key Challenge**: Existing predictive uncertainty tools (Bayesian networks, ensembles, conformal prediction) conflate aleatoric (inherent data noise) and epistemic (uncertainty about the distribution) uncertainty. TTA specifically lacks "distribution-level epistemic uncertainty." Furthermore, previous PAC-Bayes applications to domain adaptation relied on the NP-hard $\mathcal{H}$ divergence (Germain 2013), which is computationally intractable in practice.

**Goal**: To provide a unified theoretical framework that (i) is dominated by computable MMD, (ii) provides finite-sample upper bounds, (iii) naturally separates epistemic and aleatoric uncertainty, and (iv) determines when to adapt.

**Key Insight**: The authors observe that the MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ mathematically satisfies all requirements for a Walley credal set—it represents the set of all distributions indistinguishable from the source distribution at resolution $\varepsilon$.

**Core Idea**: By assuming RKHS-Lipschitz continuity, the upper bound of the "loss difference between target and source distributions" is transformed into $L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$, which is then appended to the classic PAC-Bayes bound. By taking the supremum over the MMD radius $\varepsilon$, the worst-case risk bound on the credal set is obtained, naturally providing a lower/upper risk interval.

## Method

### Overall Architecture
The framework is not an "algorithm" but a "theory + decision criterion," comprising four interconnected components:

1.  **PAC-Bayes Upper Bound (Theorem 1)**: Under the covariate shift and RKHS-Lipschitz loss assumption, the target risk $R_{P_t}(\rho)$ is decomposed into "source empirical risk + KL complexity + MMD shift penalty."
2.  **Finite Sample Version (Theorem 3)**: The population MMD is replaced with the unbiased MMD estimator $\widehat{\mathrm{MMD}}_u$, and a closed-form width $\varepsilon_{m,n}(\delta)$ is provided using Sutherland/Tolstikhin sub-Gaussian concentration.
3.  **Credal Set Geometry (Definition 5 + Proposition 7 + Corollary 9)**: Interprets MMD-balls as credal sets to provide the worst-case risk $\overline{R}_\varepsilon(\rho)$ and best-case risk $\underline{R}_\varepsilon(\rho)$. The imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon$ directly quantifies epistemic uncertainty.
4.  **Geodesic Preservation (Proposition 10 + Corollary 11)**: Proves that the geodesic distance difference between source and target neighborhoods in RKHS geometry is controlled by $\sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)$, providing a theoretical explanation for "kernel-guided adaptation protecting rare classes."

The combination forms a "distribution-level epistemic intelligence": Monitor MMD $\to$ calculate credal intervals $\to$ trigger adapt / abstain.

### Key Designs

1.  **PAC-Bayes + MMD Shift Penalty (Theorem 1 / 3)**:
    - **Function**: The only PAC-Bayes inequality in TTA that explicitly incorporates "distribution shift magnitude" into the bound.
    - **Mechanism**: Under Assumption 1 (conditional expected loss $L(w,\cdot)\in\mathcal{H}$ with $\|L(w,\cdot)\|_\mathcal{H}\le L_\mathcal{H}$), the reproducing property and Cauchy-Schwarz inequality yield $|R_{P_t}(\rho)-R_{P_s}(\rho)|\le L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$. Combined with the classic McAllester PAC-Bayes bound, this results in $R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{(\mathrm{KL}(\rho\|\pi)+\log(2\sqrt{n}/\delta))/(2n)}+L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$. Theorem 3 achieves the $O(1/\sqrt{n})$ minimax optimal rate by replacing MMD with an unbiased estimator plus $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$.
    - **Design Motivation**: Unlike $\mathcal{H}$-divergence (NP-hard), MMD is $O((m+n)^2)$ computable. The MMD shift term exhibits "linear degradation" rather than "exponential collapse," meaning the bound loosens smoothly rather than failing when distribution shift is large.

2.  **MMD-ball = Credal Set, Deriving Lower-Upper Risk Decomposition**:
    - **Function**: Upgrades uncertainty from "point estimation" to "risk intervals" and clearly separates aleatoric and epistemic factors.
    - **Mechanism**: Define $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$. Lemma 6 proves it is a convex and weakly closed credal set via linearity of the feature kernel. The worst-case risk over this set is bounded as $\sup_{Q\in\mathcal{C}_\varepsilon(P_s)}R_Q(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\cdots/2n}+L_\mathcal{H}\varepsilon$ (Proposition 7), with the infimum symmetrically derived using the Germain PAC-Bayes lower bound. The final imprecision width $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{(\mathrm{KL}+\log)/2n}+2L_\mathcal{H}\varepsilon$ separates estimation uncertainty (first term, decaying with $n$) and distribution uncertainty (second term, linear with $\varepsilon$).
    - **Design Motivation**: This step connects Walley’s behavioral imprecise probability with PAC-Bayes for the first time, providing an operational "epistemic vs aleatoric" decomposition that addresses criticisms of ML uncertainty metrics (Hüllermeier & Waegeman 2021).

3.  **RKHS Geodesic Preservation + Rare Class Robustness**:
    - **Function**: Explains why kernel-guided/MMD-bounded adaptation protects minority classes better than entropy minimization.
    - **Mechanism**: Under Assumption 2 (encoder factorized as $f_\theta=W\cdot \phi_\theta$, $\|W\|_{op}\le C_W$), local linearization of the RBF kernel gives $d_k(x,y)=\sqrt{2\gamma}\|f_\theta(x)-f_\theta(y)\|+O(\bar\epsilon^2)$. Using the reverse triangle inequality yields $|\mathbb{E}_{y\sim P_s}[d_k(x_i,y)]-\mathbb{E}_{y\sim P_t}[d_k(x_i,y)]|\le \sqrt{2\gamma}C_W\,\mathrm{MMD}(P_s,P_t)+O(\bar\epsilon^2)$. Since this bound is independent of class frequency, the local geometry of rare classes (small but compact regions) is naturally preserved. TENT-style entropy minimization treats "low-density regions" as "high-entropy regions" and flattens them, erasing rare classes.
    - **Design Motivation**: Elevates the intuition of kernel safety to a geometric theorem, ensuring epistemic control constrains both risk and representation geometry.

### Loss & Training
The paper does not propose a new training algorithm; results are based on two key assumptions: (1) Assumption 1 requires $L(w,\cdot)$ to lie in an RKHS with bounded norm (approximated via kernel universality for softmax + RBF kernels); (2) Assumption 2 requires the encoder to be decomposable into a bounded linear operator + RKHS feature map (approximated in NTK regimes, explicit MMD regularization, or spectral normalization). Section 8 discusses relaxing (1) to $\mathbb{E}_{w\sim\rho}[\|L(w,\cdot)\|_\mathcal{H}]\le L_\mathcal{H}$.

## Key Experimental Results

### Main Results
The paper is a theoretical work and **does not include a complete experimental table**. Data is presented in the form of theorems, summarized below:

| Inequality | Main Term | Description |
|---|---|---|
| Theorem 1 (Population MMD) | $R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{(\mathrm{KL}+\log(2\sqrt n/\delta))/(2n)}+L_\mathcal{H}\,\mathrm{MMD}(P_s,P_t)$ | First TTA PAC-Bayes bound with explicit MMD drift penalty. |
| Theorem 3 (Finite Sample) | Above + $L_\mathcal{H}\,(\widehat{\mathrm{MMD}}_u+\varepsilon_{m,n}(\delta/2))$, $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$ | Fully computable in $O((m+n)^2)$. |
| Proposition 7 (Worst-case) | $\sup_{Q\in\mathcal{C}_\varepsilon(P_s)} R_Q(\rho)\le \hat R_{P_s}(\rho)+\sqrt{\cdots}+L_\mathcal{H}\varepsilon$ | Bounded worst-case risk over the entire credal set. |
| Corollary 9 (Interval Width) | $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{\cdots/2n}+2L_\mathcal{H}\varepsilon$ | Epistemic uncertainty = estimation term + shift term. |
| Proposition 10 (Geometry) | $|\mathbb{E}_{P_s}[d_k]-\mathbb{E}_{P_t}[d_k]|\le \sqrt{2\gamma}C_W\,\mathrm{MMD}+O(\bar\epsilon^2)$ | MMD controls RKHS geodesic distance drift. |

### Ablation Study
| Assumption / Setting | Impact | Description |
|---|---|---|
| Disable RKHS-Lipschitz (Assump. 1) | Theorem 1 fails | Shift term no longer has a linear MMD upper bound. |
| Use posterior mean $\rho$ instead of distribution $\rho$ | KL term degrades | PAC-Bayes complexity term becomes 0, but loses prior regularization. |
| Replace MMD with $\mathcal{H}$-divergence | Reverts to Germain 2013 | Bound still holds but becomes non-computable. |
| Disable covariate shift assumption | $L(w,x)$ inconsistent between source/target | Shift term requires joint $(x,y)$ MMD, requiring kernel expansion. |
| Non-characteristic kernel | $\varepsilon=0$ no longer implies $Q=P_s$ | Credal set degenerates. |

### Key Findings
- When $\mathrm{MMD}(P_s,P_t)\to 0$, the bound precisely recovers classic PAC-Bayes without redundancy. Under large distribution shifts, the bound "loosens smoothly" rather than "failing suddenly," characterizing desirable epistemic uncertainty properties.
- **Decision Criterion**: Given a tolerance $r_{\max}$, if $\underline{R}_\varepsilon(\rho)>r_{\max}$, even the best target distribution is unacceptable—the system should **abstain**. If $\overline{R}_\varepsilon(\rho)<r_{\max}$, even the worst-case distribution is safe—no need to **adapt**. Adaptation is meaningful only when $\overline{R}_\varepsilon>r_{\max}>\underline{R}_\varepsilon$.
- The radius $\varepsilon$ can be calibrated using the asymptotic null distribution of the MMD two-sample test at level $\alpha$: Rejecting $H_0:P_t=P_s$ is equivalent to $\widehat{\mathrm{MMD}}_u>\varepsilon_\alpha$, linking credal set width to the "strength of evidence against the null hypothesis."

## Highlights & Insights
- Successfully bridges Walley’s behaviorist imprecise probability (1991) with PAC-Bayes generalization on the geometric object of MMD-balls. This represents a fusion of 1960s decision theory, 1990s statistical learning, and 2010s kernel methods, with key proofs relying elegantly on the reproducing property and Cauchy-Schwarz.
- The linear (rather than exponential) loosening of the bound as $\varepsilon$ increases suggests MMD is a "benign geometric quantity" for distribution shift, providing guidance for future distribution-shift-aware loss or OOD detector designs.
- The geodesic preservation lemma provides a formal explanation for why entropy minimization harms rare classes: entropy naturally flattens low-density regions, whereas MMD-constrained adaptation is class-frequency independent, preserving local geometry.

## Limitations & Future Work
- Assumption 1 for deep networks + softmax loss is only "informally supported by RBF universality" without a rigorous construction. Relaxing this to expected norms remains an open problem.
- Assumption 2 for general ResNets/ViTs only holds approximately under NTK limits or explicit MMD regularization; tighter bounds for specific architectures are needed.
- While the $O(1/\sqrt n)$ MMD convergence rate is minimax optimal, it may be loose in practice. Adaptive kernel selection/bandwidth tuning could provide tighter data-dependent rates.
- **Lack of Empirical Evidence**: There are no experiments on real TTA benchmarks (CIFAR-10-C, ImageNet-C, etc.) to verify bound tightness or the effectiveness of the decision criterion.
- The proposed link with conformal prediction $\alpha(\varepsilon)=\alpha_0+g(\varepsilon)$ in Appendix F is only a sketch and requires rigorous calibration.

## Related Work & Insights
- **vs. Germain et al. 2013 (PAC-Bayes DA)**: They use $\mathcal{H}$-divergence (NP-hard, no finite-sample version); Ours uses MMD ($O((m+n)^2)$ and includes concentration inequalities).
- **vs. TENT / EATA / SAR (TTA Algorithms)**: These rely on empirical tricks (entropy minimization, sharpness regularization) without upper bounds; Ours provide formal criteria for when to adapt, which is orthogonal and additive.
- **vs. BNNs / Ensembles**: Predictive uncertainty conflates aleatoric and epistemic; Ours explicitly separates them using credal set width.
- **vs. Conformal Prediction**: Conformal prediction gives individual-level coverage guarantees but doesn't constrain aggregate risk; Ours provides distribution-level risk intervals. Both can be combined by adaptively tuning conformal coverage via $\varepsilon$.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Connects PAC-Bayes, kernel mean embedding, and Walley credal sets in a TTA context for the first time.
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical short paper with no empirical validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with theorems, proof sketches, and remarks; well-managed notation.
- **Value**: ⭐⭐⭐⭐ Provides a computable criterion for "adapt vs. abstain"; highly useful for safety-critical ML deployment if paired with proper calibration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](private_and_stable_test-time_adaptation_with_differential_privacy.md)
- [\[ICML 2026\] On the Epistemic Uncertainty of Overparametrized Neural Networks](on_the_epistemic_uncertainty_of_overparametrized_neural_networks.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
