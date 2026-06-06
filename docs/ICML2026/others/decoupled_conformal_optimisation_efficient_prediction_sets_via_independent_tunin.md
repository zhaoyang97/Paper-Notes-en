---
title: >-
  [Paper Note] Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration
description: >-
  [ICML2026][Split Conformal Prediction] This paper proposes DCO-Warmstart—a tripartite "Train–Tune–Calibrate" Bayesian conformal optimization paradigm. By placing efficiency search on an independent tuning split and reser…
tags:
  - "ICML2026"
  - "Split Conformal Prediction"
  - "Bayesian Optimization"
  - "Marginal Coverage"
  - "Data Partitioning"
  - "Efficiency–Validity Decoupling"
date: 2026-05-08
content_hash: 54c65f6668180f1c
---

# Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration

**Conference**: ICML2026  
**arXiv**: [2605.18354](https://arxiv.org/abs/2605.18354)  
**Code**: Public link not provided in the paper  
**Area**: Uncertainty Quantization / Conformal Prediction / Bayesian Optimization  
**Keywords**: Split Conformal Prediction, Bayesian Optimization, Marginal Coverage, Data Partitioning, Efficiency–Validity Decoupling

## TL;DR
This paper proposes DCO-Warmstart—a tripartite "Train–Tune–Calibrate" Bayesian conformal optimization paradigm. By placing efficiency search on an independent tuning split and reserving the conformal quantile for an untouched calibration split, it achieves standard finite-sample marginal coverage guarantees on arbitrary (even infinite) candidate structure classes without requiring a confidence parameter $\delta$. Empirically, the resulting prediction set sizes are typically smaller than those of coupled calibration methods like CRC or BQ.

## Background & Motivation

**Background**: Conformal Prediction (CP) has become a mainstream uncertainty quantization paradigm due to its "distribution-free + finite-sample marginal coverage" properties. Split CP divides data into $D_{\text{train}}$ and $D_{\text{cal}}$. After fixing a non-conformity score $S(x,y)$, the prediction set $C(x)=\{y:S(x,y)\le \hat q_{1-\alpha}\}$ is derived using the empirical $(1-\alpha)$ quantile $\hat q_{1-\alpha}$. Recently, researchers have pursued "smaller yet valid" prediction sets, leading to conformal optimization methods that minimize size by searching over scores, priors, model architectures, or threshold rules.

**Limitations of Prior Work**: Representative methods such as BCP-CRC, Bayesian Quadrature calibration, and Learn-then-Test perform both "optimal threshold search" and "coverage certification" on the **same** hold-out dataset $D_{\text{cal}}$. Consequently, the quantile is no longer calculated on data independent of the search process, invalidating the exchangeability proof of standard split CP. These methods must resort to PAC-style weak guarantees ("risk is satisfied with probability $1-\delta$") and apply multiple testing corrections—which significantly inflate thresholds and prediction sets when the candidate class is large.

**Key Challenge**: Statistically, optimization and calibration are distinct tasks—the former requires repeated evaluation of candidate rules on data, while the latter requires data uncontaminated by any prior search. Bundling them into a single $D_{\text{cal}}$ might seem data-efficient but replaces a strong, simple marginal coverage guarantee with a weaker, more complex guarantee requiring $\delta$.

**Goal**: In the context of Bayesian conformal optimization, **restore** the finite-sample marginal coverage guarantee $\mathbb P\{Y_{m+1}\in C(X_{m+1})\}\ge 1-\alpha$ while retaining the ability to explicitly optimize efficiency over structures (score types, prior hyperparameters, model architectures, etc.) and eliminating the impact of candidate class size on the final threshold.

**Key Insight**: The authors observe that "using a validation set to select a model and then performing split CP" is already valid in naive CP—provided the selection does not depend on $D_{\text{cal}}$. The issue is that Bayesian conformal optimization blurs the boundary between search and calibration. By explicitly partitioning data into three sets and enforcing that the calibration split is used only once in the final step, the theory reverts to the classical framework.

**Core Idea**: Utilize an independent tuning split $D_{\text{tune}}$ for all efficiency-oriented structural selections, while the calibration split $D_{\text{cal}}$ is used solely to compute the final conformal quantile. The threshold $\hat\lambda_{\text{tune}}$ obtained during the tuning phase serves only as a ranking tool and is **discarded upon deployment**.

## Method

### Overall Architecture
Given an exchangeable dataset $D_n$, DCO-Warmstart partitions it into three disjoint subsets: $D_{\text{train}}\cup D_{\text{tune}}\cup D_{\text{cal}}$:

1.  **Training Phase**: Fit the posterior $\pi(\theta\mid D_{\text{train}})$ on $D_{\text{train}}$ to fix a family of non-conformity scores $\{S_\phi(x,y)\}_{\phi\in\Phi}$. In the BCP setting, $S_\phi(x,y)=-\log p(y\mid x,D_{\text{train}})$ is the negative log-posterior predictive density, where $\phi$ encodes structural choices.
2.  **Tuning Phase**: Perform constrained optimization on $D_{\text{tune}}$: $(\hat\phi_{\text{tune}},\hat\lambda_{\text{tune}})=\arg\min_{(\phi,\lambda)\in\Phi\times\Lambda}\widehat{\mathcal S}_{\text{tune}}(\phi,\lambda)$ s.t. $\widehat R_{\text{tune}}(\phi,\lambda)\le\alpha$, where $\widehat{\mathcal S}_{\text{tune}}$ is the empirical average prediction set size and $\widehat R_{\text{tune}}$ is the empirical miscoverage rate. Line search can be applied to $\lambda$ for each fixed $\phi$ due to monotonicity.
3.  **Calibration Phase**: **Discard** $\hat\lambda_{\text{tune}}$, recalculate the quantile $\hat q_{1-\alpha}=S_{(k_\alpha)}$ on $D_{\text{cal}}$ (where $k_\alpha=\lceil(m+1)(1-\alpha)\rceil$), and deploy the prediction set $C_{\text{DCO}}(x)=\{y:S_{\hat\phi_{\text{tune}}}(x,y)\le \hat q_{1-\alpha}\}$.

The critical statistical observation is that since $\hat\phi_{\text{tune}}$ depends only on $D_{\text{train}}\cup D_{\text{tune}}$, the scores on $D_{\text{cal}}$ remain exchangeable with the test point score given this structure. Thus, the classical split CP proof applies directly, yielding $\mathbb P\{Y_{m+1}\in C_{\hat\phi_{\text{tune}},\hat q_{1-\alpha}}(X_{m+1})\}\ge 1-\alpha$ for **any candidate class** $\Phi$ (finite or infinite) without $\delta$ or multiple testing corrections.

### Key Designs

1.  **Tripartite Data Partitioning + "Discard Tuning Threshold"**:
    *   **Function**: Physically separates efficiency search from coverage certification, reviving standard split CP exchangeability in Bayesian conformal optimization.
    *   **Mechanism**: Separate splits handle model fitting, structural selection, and threshold determination. The optimization $\min \widehat{\mathcal S}_{\text{tune}}$ s.t. $\widehat R_{\text{tune}}\le\alpha$ yields both $\hat\phi_{\text{tune}}$ and $\hat\lambda_{\text{tune}}$, but the latter is used only to rank candidates within $\Phi$ and is **discarded**. The operational threshold is the subsequent $\hat q_{1-\alpha}$ from $D_{\text{cal}}$.
    *   **Design Motivation**: To ensure $\hat\phi_{\text{tune}}$ is measurable with respect to $D_{\text{cal}}$, obtaining finite-sample marginal coverage without $\delta$. Unlike BCP-CRC which bundles optimization into $D_{\text{cal}}$, DCO decouples these tasks.

2.  **Decoupling Candidate Class Size from Final Threshold**:
    *   **Function**: Prevents the conformal threshold from inflating with the number of candidate structures $K=|\Phi|$, allowing for richer search spaces.
    *   **Mechanism**: CRC/BQ-style methods must certify risks for multiple candidates on $D_{\text{cal}}$, requiring a penalty term that grows with $K$. DCO leaves candidate filtering entirely within $D_{\text{tune}}$, running the quantile calculation only once for the fixed $\hat\phi_{\text{tune}}$.
    *   **Design Motivation**: Theorem 3.1 explicitly states that the coverage guarantee holds for "any finite or infinite $\Phi$." The cost is borne only by the tuning sample complexity (Proposition 3.2): $m_{\text{tune}}\ge \max\{\log(4|\mathcal A|/\eta)/(2\varepsilon_R^2),\,B^2\log(4|\mathcal A|/\eta)/(2\varepsilon_S^2)\}$. Candidate size only affects "tuning quality," not "final validity."

3.  **Asymptotic Equivalence to PAC Methods + DirectTune Diagnostic Baseline**:
    *   **Function**: Clarifies the relationship between DCO and CRC/BQ styles—different guarantees in finite samples (marginal coverage vs. high-probability risk control) but convergence to the same population threshold $\lambda^\star=\inf\{\lambda:R(\lambda)\le\alpha\}$ in large samples.
    *   **Mechanism**: Under conditions where $R(\lambda)$ is continuous and strictly monotonic near $\lambda^\star$, the split conformal threshold is consistent $\hat\lambda_{\text{DCO}}\xrightarrow{p}\lambda^\star$, and the CRC bias term $b_m(\lambda,\delta_m)$ vanishes, Proposition 3.3 proves $\hat\lambda_{\text{CRC}}-\hat\lambda_{\text{DCO}}\xrightarrow{p}0$. DirectTune (tuning only without calibration) is used as a negative baseline to quantify the cost of skipping the calibration split.
    *   **Design Motivation**: To clarify that DCO does not "replace" CRC/BQ but serves different objectives; DirectTune empirically shows smaller sets but fails coverage, proving the necessity of the final calibration step.

### Loss & Training
The tuning objective follows equation (9): $\min_{(\phi,\lambda)} \widehat{\mathcal S}_{\text{tune}}(\phi,\lambda)$ s.t. $\widehat R_{\text{tune}}(\phi,\lambda)\le\alpha$. In practice, this is solved via grid search over $\Phi\times\Lambda$ combined with monotonic line search. If no candidate satisfies the constraint, the one with minimum empirical miscoverage is selected. Total complexity is $O(K|\Lambda|m_{\text{tune}})$ for tuning and $O(m_{\text{cal}}\log m_{\text{cal}})$ for calibration sorting.

## Key Experimental Results

### Main Results
The authors compared DCO-Warmstart against CRC/BQ-style calibration on ImageNet-A (classification), CIFAR-100 (classification), and Diabetes, California Housing, and Concrete (regression). The target coverage $1-\alpha$ was set to 90%. DCO consistently matched the nominal coverage while providing tighter prediction sets:

| Dataset | Metric | CRC/BQ style | DCO-Warmstart | Gain |
| :--- | :--- | :--- | :--- | :--- |
| ImageNet-A | Avg. Set Size | 26.52 | 25.26 | ↓ 1.26 |
| ImageNet-A | 95th Percentile Size | 58.95 | 53.73 | ↓ 5.22 |
| Diabetes (Reg.) | Avg. Interval Width | 2.098 | 1.914 | ↓ 0.184 |

### Ablation Study
Systematic ablations were performed on search ranges, split ratios, and coverage levels, using DirectTune as a "no-calibration" diagnostic baseline:

| Configuration | Coverage | Set Size | Description |
| :--- | :--- | :--- | :--- |
| DCO-Warmstart | Close to $1-\alpha$ | Smaller | Full method; theoretical + empirical validity |
| CRC/BQ-style (Coupled) | $\ge 1-\alpha$ (Conservative) | Larger | Multiple testing correction inflates threshold |
| DirectTune | Generally < $1-\alpha$ | Smallest | Low empirical threshold; no exchangeability guarantee |

### Key Findings
- **Calibration cannot be skipped**: While DirectTune yields the tightest sets, its miscoverage exceeds nominal levels, confirming that quantiles must be calculated on a $D_{\text{cal}}$ untouched by search to ensure $\ge 1-\alpha$ coverage.
- **Candidate class richness**: DCO's advantage over CRC/BQ increases as the candidate class becomes more diverse, because the latter suffers from heavier multiple testing corrections while DCO is immune.
- In scenarios with extremely few samples requiring a $\delta$-risk certificate, CRC/BQ-style remains appropriate. DCO is positioned for settings where a tripartite split is feasible and the goal is classic marginal coverage.

## Highlights & Insights
- **Reorienting the problem rather than inventing new algorithms**: The method is essentially "split data three ways and perform split CP," but the authors elevate it to a "design principle for Bayesian conformal optimization." By distinguishing marginal coverage from high-probability risk control, complex coupled calibration tricks can often be replaced by simply "cutting the data once more."
- **Unconditional coverage for infinite candidate classes**: The property in Theorem 3.1 allowing for infinite $\Phi$ is highly useful—neural architectures and continuous hyperparameters can be treated as candidates without the discretization and multiple testing costs required by LTT/CRC.
- **DirectTune as a counter-example** is more persuasive than pure theory. Showing that an "almost right" method fails miscoverage empirically is more effective at convincing practitioners to allocate 20–30% of data for a calibration split.

## Limitations & Future Work
- Tripartite splits partition data more finely. In small-sample regimes, both $D_{\text{tune}}$ and $D_{\text{cal}}$ may be insufficient, potentially inflating the variance of the tuning rank and the final quantile. While DKW inequalities provide a $m_{\text{cal}}^{-1/2}$ rate, the paper lacks guidance on optimal split ratios.
- The objective only guarantees marginal coverage, not conditional coverage or risk control. If a "high-probability risk $\le \alpha$" guarantee is required, one must revert to CRC/LTT.
- Experiments focused on small-to-medium classification/regression benchmarks. Efficiency gains in high-dimensional tasks like LLM generation or structured output require further validation.
- The constrained optimization still uses grid + line search. Computational overhead may become a bottleneck for very large candidate classes; future work could integrate smarter searchers like Bayesian Optimization.

## Related Work & Insights
- **vs BCP-CRC (Coupled Calibration)**: BCP-CRC optimizes thresholds and certifies risk on the same $D_{\text{cal}}$ to provide PAC-style guarantees; DCO decouples these for simpler marginal coverage at the cost of an extra tuning split.
- **vs Learn-then-Test (LTT)**: LTT certifies candidates via hypothesis testing at the population risk level; like CRC, it requires $\delta$ and corrections. DCO's value proposition is "no $\delta$, no corrections, candidate size is irrelevant."
- **vs ROCP (Risk-Optimal CP)**: ROCP also follows an "optimize then independently calibrate" approach but focuses on the entire prediction set construction + downstream decision. DCO refines this for Bayesian structures and threshold search.
- **vs Score-tuned CP (e.g., RAPS)**: These methods already "tune scores then perform split CP." DCO generalizes this principle to all Bayesian structural selections (priors, architectures, rules).

## Rating
- Novelty: ⭐⭐⭐ Clear and practical, though essentially a systematized packaging of "one more split" for classic CP.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks across classification/regression + three-dimensional ablations + DirectTune control support the arguments well.
- Writing Quality: ⭐⭐⭐⭐ Table 1 clearly situates six categories of methods across four dimensions (optimization/calibration/guarantee/confidence).
- Value: ⭐⭐⭐⭐ Provides a clear decision framework for when to use PAC vs. marginal coverage in conformal optimization; easily reproducible in engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/others/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](../../NeurIPS2025/others/prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ICML 2026\] Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation](industrializing_prediction-powered_inference_the_glide_library_for_reliable_gena.md)
- [\[ICML 2026\] Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning](vision_transformer_finetuning_benefits_from_non-smooth_components.md)

</div>

<!-- RELATED:END -->
