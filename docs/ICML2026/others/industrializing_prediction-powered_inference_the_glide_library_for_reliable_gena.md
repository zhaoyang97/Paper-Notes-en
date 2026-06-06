---
title: >-
  [Paper Note] Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation
description: >-
  [ICML2026][Prediction-Powered Inference] GLIDE unifies the latest estimators (PPI++, Stratified PPI, PTD, ASI) and samplers (uniform, stratified, active…
tags:
  - "ICML2026"
  - "Prediction-Powered Inference"
  - "LLM-as-Judge"
  - "Stratified Sampling"
  - "Active Sampling"
  - "Effective Sample Size"
date: 2026-05-08
content_hash: b94af5d87cc7cb8a
---

# Industrializing Prediction-Powered Inference: The GLIDE Library for Reliable GenAI and Agentic Systems Evaluation

**Conference**: ICML2026  
**arXiv**: [2605.31278](https://arxiv.org/abs/2605.31278)  
**Code**: https://github.com/EmertonData/glide  
**Area**: LLM Evaluation / Statistical Inference / Agentic Systems  
**Keywords**: Prediction-Powered Inference, LLM-as-Judge, Stratified Sampling, Active Sampling, Effective Sample Size  

## TL;DR
GLIDE unifies the latest estimators (PPI++, Stratified PPI, PTD, ASI) and samplers (uniform, stratified, active, cost-optimal) from the PPI (prediction-powered inference) family into a scipy-style mean estimation library. It specifically addresses the hybrid evaluation problem of "expensive human labels + cheap but biased LLM-as-judge," integrating Monte Carlo validation and a decision tree to industrialize the reliable assessment of GenAI and Agentic systems.

## Background & Motivation

**Background**: Evaluating the "quality" of GenAI and Agentic systems typically settles into a mean estimation task—accuracy, relevance, hallucination rate, toxicity rate, and tool-use success rate are all forms of $\theta^\star=\mathbb{E}[Y]$. Two prevailing approaches exist: (i) Full human annotation, which is reliable but slow and expensive (expert review of an agentic trajectory can cost several dollars); (ii) LLM-as-judge, which is cheap (cents per instance) but suffers from systematic bias, particularly in knowledge-intensive domains like medicine, law, and finance.

**Limitations of Prior Work**: The PPI framework proposed by Angelopoulos et al. (2023) was designed for this "small gold truth + large proxy predictions" scenario—providing unbiased estimates and nominal confidence intervals. However: (i) Extensions of PPI (PPI++, Stratified, PTD, ASI, cost-optimal) are scattered across various papers with inconsistent notation and fragmented implementations; (ii) The existing `ppi_py` library serves as a foundational implementation but lacks depth in mean estimation specialization and integration of newer methods; (iii) Agentic evaluation possesses unique attributes (extreme cost asymmetry, natural stratification, available proxy uncertainty) that map perfectly to PPI branches, but no library connects them end-to-end.

**Key Challenge**: In practical deployment, engineers need a pathway that provides unbiased estimation, valid confidence intervals, and maximizes label budget efficiency. This pathway must automatically select methods based on specific conditions such as "availability of cost estimates / proxy uncertainty / natural stratification / sufficient budget." Currently, academic fragmentation makes this engineering-wise infeasible.

**Goal**: To industrialize recent advances in the PPI family into a single scipy-style library covering: (1) Unified estimator encapsulation; (2) Unified sampler encapsulation; (3) Reproducible Monte Carlo validation suites; (4) Empirically calibrated decision trees for method selection; (5) Real-world agentic benchmark cases.

**Key Insight**: The authors intentionally focus **exclusively on mean estimation**—the form of 90% of deployment-side metrics. Removing the generality of GLM/M-estimators allows for significant codebase simplification. Multiple estimators that diverge in general M-estimation collapse into the same form for mean estimation, improving API consistency. Furthermore, explicitly separating "sampling / annotation / estimation" into three stages allows samplers and estimators to be independently combined.

**Core Idea**: Use the PPI++ style of "small human labels + large LLM-as-judge predictions → unbiased mean + valid confidence intervals" as the core. Integrate stratified, active, and cost-optimal components as orthogonal plugins, emphasizing the engineering slogan: "Better proxies do not replace human labels; they amplify the human label budget."

## Method

### Overall Architecture
GLIDE divides the evaluation pipeline into three steps: **Sampling → Annotation → Estimation**. Given a pool of $N$ proxy-labeled data points (from LLM-as-judge): (1) A sampler selects $n$ instances for human annotation; (2) The annotation step is domain-specific and handled externally; (3) An estimator combines the $n$ human labels with $N$ proxy predictions to produce an unbiased estimate $\hat\theta$ and a confidence interval. This sampler ↔ estimator decoupling allows for arbitrary mix-and-match and easy integration of new methods.

The core PPI formula (PPI++, Angelopoulos 2023b):
$$\hat\theta^{\text{PPI++}}_\lambda = \frac{1}{n}\sum_{i=1}^n Y_i + \lambda\left(\frac{1}{N}\sum_{j=1}^N f(X_j) - \frac{1}{n}\sum_{i=1}^n f(X_i)\right)$$

where $f$ is the proxy (LLM-as-judge) and $\lambda\in\mathbb{R}$ is the power-tuning parameter. $\lambda^\star$ has a closed-form solution to minimize asymptotic variance, ensuring PPI++ is asymptotically **never worse** than classical estimators using only human labels, even if the proxy is adversarial.

### Key Designs

1.  **Three-Step Decoupling + Scipy-style API**:
    - **Function**: Transforms "sampling" and "estimation" into orthogonal, pluggable objects.
    - **Mechanism**: Samplers expose a `sample` method returning $(\pi,\xi)$, where $\pi\in[0,1]^N$ is the sampling probability and $\xi\in\{0,1\}^N$ is the inclusion indicator. Estimators are stateful objects whose `estimate` interface returns a dataclass containing point estimates, confidence intervals, effective sample size $n_{\text{eff}}$, and metric labels. An end-to-end flow fits in 6 lines of Python.
    - **Design Motivation**: Adopting the scipy/scikit-learn paradigm minimizes learning costs for the scientific computing community. Decoupling allows independent improvements in literature (e.g., a new sampling method) to be layered seamlessly.

2.  **Comprehensive Toolbox (5 Samplers + 5 Estimators)**:
    - **Function**: Maps the four characteristics of Agentic evaluation to specific methodological choices.
    - **Mechanism**: 
        - Samplers: `UniformSampler` (baseline), `StratifiedSampler` (Neyman allocation $n_h\propto N_h\sigma_h$), `ActiveSampler` (Bernoulli sampling proportional to proxy uncertainty), `CostOptimalRandomSampler`, and `CostOptimalSampler` (based on cost ratios and uncertainty).
        - Estimators: `PPIMeanEstimator` (PPI++), `StratifiedPPIMeanEstimator`, `PTDMeanEstimator` (Predict-Then-Debias using bootstrap for small samples $n<50$), `StratifiedPTDMeanEstimator`, and `ASIMeanEstimator` (IPW debiasing for active sampling).
    - **Design Motivation**: Each component corresponds to a real-world "If X, use Y" scenario, organizing three years of discrete PPI research into a continuous menu for practitioners.

3.  **Heuristic Decision Tree**:
    - **Function**: Allows engineers with limited statistical background to select the optimal sampler/estimator combination in seconds.
    - **Mechanism**: The sampling route is determined by three binary signals: cost estimates available → CostOptimal; proxy uncertainty available → ActiveSampler; heterogeneous natural strata available → StratifiedSampler. The estimation route uses a threshold: if human labels $n \ge 50$ (per stratum), use CLT-based estimators (PPI++系); otherwise, use bootstrap-based PTD variants.
    - **Design Motivation**: Embedding statistical decision-making into the library itself fills a critical gap in PPI industrialization.

### Loss & Training
This work does not involve training but focuses on statistical inference. All estimators return a `PredictionPoweredMeanInferenceResult`. The effective sample size $n_{\text{eff}}=n\cdot\widehat{\text{Var}}(\bar Y_n)/\widehat{\text{Var}}(\hat\theta^{\text{PPI++}}_\lambda)$ is the core KPI; the ratio $n_{\text{eff}}/n\ge 1$ translates directly into saved annotation hours.

## Key Experimental Results

### Main Results
**Monte Carlo Validation**: A synthetic binary classification task with $\theta^\star=0.55$ and a biased proxy mean of $0.50$. Proxy quality is controlled by Pearson correlation $\rho$. $N_{\text{true}}=500$, $N_{\text{proxy}}=1000$, 90% confidence level, 1000 iterations.

| Correlation $\rho$ | Method | Empirical Coverage | Interval Width | $n_{\text{eff}}$ |
|-------------|------|-----------|----------|------------------------------|
| 0.1 | Labeled-only | 0.90 | 0.073 | 500 |
| 0.1 | PTD | 0.90 | 0.072 | ≈ 500 |
| 0.5 | Labeled-only | 0.90 | 0.073 | 500 |
| 0.5 | PTD | 0.90 | 0.060 | ≈ 750 |
| 0.9 | Labeled-only | 0.90 | 0.073 | 500 |
| 0.9 | **PTD** | **0.90** | **0.049** | **≈ 1100 (2.2×)** |

PTD maintains 90% nominal coverage across all $\rho$. Better proxies yield narrower intervals and higher $n_{\text{eff}}$. If the proxy is uninformative, PTD reverts to labeled-only width and never "performs worse."

**Agentic Case: R-Judge Safety Evaluation**: 568 user/agent dialogues across 5 domains (general, programming, finance, web, IoT), $\theta^\star\approx 0.525$. Proxy uses claude-sonnet-3.5 as LLM-as-judge with verbalized confidence. Proxy mean ≈ 0.655 (biased +13 pp), $\rho\approx 0.59$. Budget $n=100$ human labels, $N=468$ proxy-only.

| Protocol | 90% Coverage | Interval Width | $n_{\text{eff}}$ |
|------|-----------|----------|------------------|
| Labeled-only ($n=100$) | 0.90 | 0.164 | 100 |
| Proxy-only (No debiasing) | <0.05 | 0.066 | — |
| PPI++ (uniform) | 0.90 | 0.137 | ≈ 143 |
| ASI (active) | 0.90 | 0.135 | ≈ 148 |
| **Stratified PPI++ (Neyman)** | **0.90** | **0.131** | **≈ 157 (1.57×)** |

### Ablation Study
| Configuration | Empirical Coverage (90%) | Avg Interval Width | Note |
|------|------------------|--------------|------|
| Full: PPI++ + power tuning | 0.90 | 0.137 | Default combo |
| w/o power tuning ($\lambda=1$) | 0.90 | 0.142 | Slightly wider but maintains coverage |
| w/o stratification | 0.90 | 0.137 | Reverts to standard PPI++ |
| w/o active sampling | 0.90 | 0.137 | Same as PPI++ |
| Degraded Proxy ($\rho=0.1$) | 0.90 | ≈ baseline | Interval widens, coverage holds |
| Proxy-only | < 0.05 | 0.066 | Narrow but biased, coverage collapses |

### Key Findings
- **Robustness of Coverage**: All four protocols involving human labels maintained nominal coverage across all $\rho$. Only the "proxy-only" baseline failed, validating that PPI's theoretical guarantees hold in real LLM-as-judge scenarios.
- **Stratification > Active (on R-Judge)**: Stratification by domain outperformed active sampling by uncertainty on this benchmark and is more engineer-friendly as it requires no confidence signals.
- **Proxy Quality ↔ ESS Monotony**: Improving $\rho$ from 0.1 to 0.9 directly translates a 500-label budget into an effective 1100-label budget.
- **PTD Necessity**: CLT-based estimators require $n \gtrsim 50$ per stratum; otherwise, they underestimate width. PTD is the reliable fallback.

## Highlights & Insights
- **Scalable Library Design**: The three-stage decoupling is a software engineering victory, allowing the library to ingest fragmented academic work.
- **Amplification, Not Replacement**: Quantifying the ROI of better LLM judges as an $n_{\text{eff}}/n$ ratio provides "hard numbers" for product decisions.
- **Embedded Expertise**: Moving method selection from a research question to a table lookup is essential for PPI's industrialization.
- **Strategic Specialization**: Sacrificing the generality of GLMs for deeper mean estimation implementation is the correct trade-off for deployment realities.

## Limitations & Future Work
- **Mean Estimation Only**: Quantiles (P95 latency), regression coefficients, and GLM scenarios still require `ppi_py`.
- **Sample Size Constraints**: The validity of PTD for ultra-small samples ($n < 20$) lacks a strict upper bound.
- **i.i.d. Assumption**: Does not support covariate/label shifts or stream monitoring (anytime-valid).
- **Annotation Orthogonality**: GLIDE assumes an annotation pipeline already exists, which is often the most difficult/expensive part of vertical deployment.
- **Non-deterministic Evaluation**: How to allocate budget between "input coverage" and "output replication" in agentic systems remains an open question.
- **Multiple Annotators**: Extending the framework to handle inter-annotator disagreement.

## Related Work & Insights
- **vs `ppi_py` (2023a)**: Foundational bib but lacks newer methods (Stratified, PTD, ASI) and decision trees. GLIDE is a "combat-ready" enhancement for mean estimation.
- **vs HELM / DeepEval / RAGAS**: These are upstream orchestrators. GLIDE is the downstream statistical layer that turns their outputs into debiased estimates with coverage guarantees.
- **vs Csillag et al. 2025 (PPI e-values)**: Potential entry point for streaming monitoring in GLIDE’s roadmap.
- **vs Cowen-Breen et al. 2026 (Multi-proxy)**: Aggregating multiple judges is a high-priority future update.

## Rating
- Novelty: ⭐⭐⭐ (Synthesizes existing methods, but high value in **industrialization + decision tree**)
- Experimental Thoroughness: ⭐⭐⭐⭐ (MC validation + real agentic case study)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear framework and logical flow)
- Value: ⭐⭐⭐⭐⭐ (Directly addresses GenAI evaluation pain points; likely to gain significant community traction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](../../NeurIPS2025/others/prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ICML 2026\] Decoupled Conformal Optimisation: Efficient Prediction Sets via Independent Tuning and Calibration](decoupled_conformal_optimisation_efficient_prediction_sets_via_independent_tunin.md)
- [\[ICML 2026\] Inference of Online Newton Methods with Nesterov's Accelerated Sketching](inference_of_online_newton_methods_with_nesterovs_accelerated_sketching.md)
- [\[ICML 2026\] Beyond Model Readiness: Institutional Readiness for AI Deployment in Public Systems](beyond_model_readiness_institutional_readiness_for_ai_deployment_in_public_syste.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)

</div>

<!-- RELATED:END -->
