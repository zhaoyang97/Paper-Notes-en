---
title: >-
  [Paper Note] Controllable Generative Sandbox for Causal Inference
description: >-
  [ICML 2026][Causal Inference][CausalMix] This paper proposes CausalMix: a variational generative framework that jointly optimizes a data-type-specific multi-head decoder and a Bayesian Gaussian mixture latent prior with…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "CausalMix"
  - "conditional VAE"
  - "Bayesian GMM prior"
  - "overlap regularizer"
  - "CATE benchmarking"
date: 2026-05-08
content_hash: 626fa2a05adc24b4
---

# Controllable Generative Sandbox for Causal Inference

**Conference**: ICML 2026  
**arXiv**: [2603.03587](https://arxiv.org/abs/2603.03587)  
**Code**: https://github.com/zhangqiecho/causalmix  
**Area**: Causal Inference / Medical Statistics; Generative Models for Methodological Validation; Synthetic Data Benchmark  
**Keywords**: CausalMix, conditional VAE, Bayesian GMM prior, overlap regularizer, CATE benchmarking

## TL;DR
This paper proposes CausalMix: a variational generative framework that jointly optimizes a data-type-specific multi-head decoder and a Bayesian Gaussian mixture latent prior with three independently controllable causal "knobs" (overlap $\alpha(X)$, CATE function $\tau(X)$, unobserved confounding $\kappa(X,T)$). This enables users to freely design counterfactual benchmarks while maintaining fidelity to real data distributions. On real mCRPC (prostate cancer) cases, CausalMix demonstrates high-fidelity reproduction of mixed-type tabular data and stable, on-demand injection of overlap/confounding/heterogeneous effects, serving as a controllable stress test for CATE estimators.

## Background & Motivation

**Background**: Evaluation of causal inference methods (meta-learners, DR-learners, DML, causal forest, BCF) heavily relies on **synthetic data with ground-truth counterfactuals**—since in real data, both $Y(1)$ and $Y(0)$ cannot be observed simultaneously. Three common simulator types: fully parametric (controllable but unrealistic), semi-synthetic (simulate $T/Y$ on real $X$, limited control), and data-fitting generators (RealCause, WGAN, Credence, etc., use neural models to fit DGP, realistic but weak causal controllability).

**Limitations of Prior Work**: (i) Existing data-fit generators perform poorly on **mixed-type tabular data** (continuous + binary + categorical + integer), either forcibly one-hot encoding (introducing spurious correlations) or using a single likelihood loss for multi-type structures; (ii) **Causal knobs are missing or coupled**—RealCause only interpolates between fitted extremes, WGAN lacks effect control, Credence allows specification but lacks support for mixed-type multimodal data; (iii) Even if $\tau(X)$ can be specified, there is **no mechanism to verify if the generator truly implements it**, especially when the causal function is low-dimensional/weakly nonlinear and easily overwhelmed by reconstruction loss.

**Key Challenge**: There is a natural trade-off between distributional realism (fit observed data) and causal controllability (faithfully realize user-specified $\tau, \kappa, \alpha$)—the tighter the fit, the less freedom; the more freedom, the further from real data. Existing methods sacrifice either the latter (neural generators) or the former (parametric simulators).

**Goal**: (i) Jointly optimize distributional fidelity and causal constraints under a unified objective, avoiding a forced trade-off; (ii) Achieve high fidelity on mixed-type tabular data; (iii) Provide three **orthogonal, independently controllable** causal knobs (overlap/confounding/heterogeneity) with a quantification and verification pipeline; (iv) Demonstrate practical value in real clinical scenarios (mCRPC safety comparison).

**Key Insight**: Use a conditional VAE as the generative backbone (proven stable on tabular data with analytic ELBO), express causal constraints as differentiable penalties on decoder outputs, and use mean alignment + variance regularization to ensure faithful realization of low-dimensional causal functions; replace the isotropic Gaussian prior with a Bayesian GMM to recover the multimodal structure of mixed-type data.

**Core Idea**: Treat "distribution fitting" and "causal control" as two sets of terms in a unified loss, with rigidness hyperparameters $\lambda_\alpha, \lambda_\tau, \lambda_\kappa$ for explicit control; mixture prior handles multimodality, multi-head decoder handles mixed types, and three penalty layers address the three causal dimensions—one tool simultaneously solves fidelity, control, and mixed-type challenges.

## Method

### Overall Architecture
Given observations $\mathcal{O} = (X, T, Y)$, where $X$ is mixed-type covariates, $T\in\{0,1\}$, and $Y$ is the outcome. The generator $G_\theta$ is modularized into three parts:

- **Treatment model** $p(T)$: Bernoulli;
- **Pre-treatment model** $G_{X,\theta}$: conditional VAE modeling $X\mid T$;
- **Post-treatment model** $G_{Y,\theta}$: conditional VAE jointly modeling $(Y(0), Y(1))\mid X, T$, **outputting both potential outcomes**.

Generation proceeds as $T'\to X'\mid T'\to (Y'(0), Y'(1))\mid X', T'\to Y' = T'Y'(1)+(1-T')Y'(0)$. The decoder uses multi-heads: Gaussian for continuous, Bernoulli for binary, softmax for categorical. After training, the standard Gaussian latent prior is replaced with a Bayesian GMM (Dirichlet-process prior).

**Unified Objective**:
$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{VAE}} + \lambda_\alpha \mathcal{L}_\alpha + \lambda_\tau \mathcal{L}_\tau^{\text{mean}} + \lambda_\tau^{\text{var}}\mathcal{L}_\tau^{\text{var}} + \lambda_\kappa \mathcal{L}_\kappa^{\text{mean}} + \lambda_\kappa^{\text{var}}\mathcal{L}_\kappa^{\text{var}}
$$

### Key Designs

1. **Three Independent Causal "Knobs" + Huber-style Penalty**:

    - **Function**: Allow users to independently specify overlap $\alpha(X)$, CATE $\tau(X)$, and unobserved confounding $\kappa(X,T)$ during generation, ensuring $G_\theta$ truly satisfies them.
    - **Mechanism**: Define target and generator-induced causal quantities as $\alpha(x) = P(X=x\mid T=0)/P(X=x\mid T=1)$, $\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x]$, $\kappa(x,t) = \mathbb{E}[Y(t)\mid X=x,T=1] - \mathbb{E}[Y(t)\mid X=x,T=0]$. Corresponding training penalties:
        - **Overlap**: $\mathcal{L}_\alpha = \mathbb{E}_X[(\log\alpha_\theta(X) - \log\alpha(X))^2]$, directly aligns the decoder's log-density ratio via MSE;
        - **CATE**: $\mathcal{L}_\tau^{\text{mean}} = \mathbb{E}_X[\lambda_\tau^{\text{mse}}(\Delta\tau_\theta)^2 + \lambda_\tau^{\text{sl1}}\text{SmoothL1}(\Delta\tau_\theta)]$, a **Huber composite loss**—quadratic anchors the mean, SmoothL1 improves robustness to outliers and weakly identified regions, with an additional variance penalty $\mathcal{L}_\tau^{\text{var}} = \text{Var}[\Delta\tau_\theta]$ to suppress spurious unit-level discreteness;
        - **Confounding**: $\kappa$ uses the same Huber composite + variance structure.
    - **Design Motivation**: Pure MSE for $\tau, \kappa$ is easily overwhelmed by reconstruction loss when these are low-dimensional/weakly nonlinear—the model only ensures "$\tau_\theta$ is correct on average" but is unstable at the unit level; Huber + variance regularizer simultaneously anchor the mean and compress dispersion, enabling stable enforcement of causal constraints even in low-signal settings. Independent tuning of the three $\lambda$ allows factorial studies (simultaneously varying overlap and confounding to test CATE estimator robustness).

2. **Mixed-type Multi-head Decoder + Bayesian GMM Prior**:

    - **Function**: Faithfully reproduce the mixed-type and multimodal structure of real tabular data, avoiding spurious correlations from type confusion.
    - **Mechanism**: Each variable is assigned an independent likelihood head according to its data type—continuous variables use Gaussian NLL (unlike Credence's MSE), allowing the decoder to learn both location and dispersion, which is crucial for heteroscedastic or bounded-support variables; binary uses Bernoulli logits; categorical uses softmax; integers are treated as continuous and rounded post hoc. The encoder still outputs a diagonal Gaussian posterior, but after training, a BGMM (Dirichlet-process prior, truncated stick-breaking variational inference) is fitted to the latent means as the generation prior: $p_{\text{BGMM}}(z) = \sum_k \pi_k \mathcal{N}(z\mid\mu_k, \Sigma_k)$, with $K$ learned automatically.
    - **Design Motivation**: The standard isotropic Gaussian prior assumes a unimodal latent space, which is inconsistent with real clinical data (patients naturally cluster into subgroups); BGMM uses a Dirichlet-process to automatically select $K$, **post hoc fitting** without changing the VAE training objective—this increases expressiveness while keeping the decoder unchanged. The detail of Gaussian NLL vs MSE is often overlooked but critical—MSE does not learn variance, leading to gradient scale imbalance for heteroscedastic variables.

3. **Joint Optimization + Modular Staging (Decoupling X and Y)**:

    - **Function**: Enable distributional fit and causal control to co-train within the same mini-batch, but optimize the X-generator and Y-generator independently to reduce coupling.
    - **Mechanism**: Pre-treatment $G_{X,\theta}$ optimizes only $\mathcal{L}_{\text{VAE}}^X + \lambda_\alpha\mathcal{L}_\alpha$ (X reconstruction + overlap control); post-treatment $G_{Y,\theta}$ optimizes $\mathcal{L}_{\text{VAE}}^Y + \lambda_\tau\mathcal{L}_\tau^{\text{mean}} + \cdots + \lambda_\kappa\mathcal{L}_\kappa^{\text{mean}} + \cdots$. During training, the decoder evaluates both $Y(0), Y(1)$ potential outcomes (even if only one is observed), enabling computation of $\tau_\theta, \kappa_\theta$. Early stopping is based on validation loss.
    - **Design Motivation**: The causal mechanisms for X and Y differ—overlap on X is a marginal distribution problem, while $\tau,\kappa$ on Y are conditional expectation problems; separate training allows each module's rigidness hyperparameter to be tuned independently, avoiding interference from one module's penalty on another. **Evaluating both potential outcomes** is key for directly expressing $\tau$—in contrast to Credence and others that only model $Y\mid X,T$.

### Loss & Training

- Optimizer: Adam (lr = $10^{-3}$), 80/20 train/val split, PyTorch Lightning;
- Key hyperparameters: $\lambda_\tau, \lambda_\kappa$ fixed at $10^3$; $\lambda_\alpha$ in $10^1\text{–}10^2$ (overlap is more sensitive to misspecification);
- For low-dimensional/weakly nonlinear control functions: reduce MSE weight (0.2–0.4), increase SmoothL1 + variance reg;
- After training, fit BGMM (DP prior, max $K$ = latent dim) as the generation prior.

## Key Experimental Results

### Main Results (mCRPC Cases: abiraterone vs enzalutamide, 4,098 patients, 18 baseline covariates)

| Scenario | Setting | Key Phenomenon |
|----------|---------|---------------|
| Scenario 1 | $\tau\equiv 0.1, \kappa\equiv 0, \log\alpha\equiv 0$ (constant effect, no confounding, perfect overlap) | Sanity check: both BGMM and Gaussian prior successfully recover |
| Scenario 2 | Linear $\tau$ (CVD, age, Charlson), $\kappa\equiv 0.02$, $\log\alpha\equiv 1$ | Both priors perform well, BGMM slightly better |
| **Scenario 3** | Nonlinear tanh $\tau$ (CVD, age, Charlson, dementia), $\kappa$ jointly depends on $X,T$, $\log\alpha = 2(2\cdot\text{Abi\_prev}-1)$ | **BGMM wins by a large margin**: CATE correlation and decoder-level overlap reconstruction significantly outperform Gaussian |

### Ablation Study

| Configuration | Key Effect | Notes |
|---------------|------------|-------|
| Gaussian prior vs BGMM | BGMM comprehensively outperforms in Scenario 3 | Multimodal prior is necessary for complex scenarios |
| Gaussian NLL vs MSE (continuous) | NLL is significantly better (especially for heteroscedastic variables) | Learning variance is essential for correct modeling |
| Composite Huber (MSE+SmoothL1+var) vs pure MSE | Huber is stable for low-dimensional $\tau$, pure MSE is unstable | Variance regularizer is the key stabilizer |
| Privacy trade-off | Gaussian prior is more private, BGMM slightly weaker but protection > 0.5 | Controlled trade-off between realism and privacy |

### Key Findings

- **BGMM value increases with causal complexity**: In Scenarios 1/2, both priors perform well; in Scenario 3, BGMM overwhelmingly outperforms Gaussian in normalized Wasserstein, C2ST, CATE correlation, and overlap reconstruction—multimodal prior is essential for multi-peaked clinical data.
- **Privacy-realism trade-off is controlled**: BGMM is slightly weaker in privacy due to higher realism, but DCR protection fraction remains $>0.5$, median distance ratio $>1$, with no systematic memorization; the decline is concentrated in lower quantiles (local proximity rather than widespread leakage).
- **Causal knobs are faithfully implemented**: In the complex Scenario 3, CATE MAE/Pearson, $\kappa$ MAE, and overlap MSE all reach acceptable accuracy, demonstrating the effectiveness of the unified loss and Huber + variance regularization design.
- **CATE estimator benchmarking**: On Scenario 3 calibrated DGP, X-learner, DR-learner, DML, Causal Forest, and BCF are compared side-by-side (Fig. 4), revealing which estimators are more robust in different overlap/confounding regions—this is the practical value of CausalMix.
- **Causal Forest hyperparameter sensitivity visualization** (Fig. 5): PEHE vs min leaf size in Scenario 3 DGP shows a nontrivial pattern, providing direct guidance for tuning in clinical scenarios—insights unattainable from parametric simulators.

## Highlights & Insights

- **"Realism + controllability" is no longer a trade-off**: The unified loss incorporates both into a single objective, with rigidness hyperparameters giving users explicit control, allowing benchmark designers to "have it both ways" for the first time.
- **Multi-head decoder + Gaussian NLL is an underrated detail**: Many tabular generative models use MSE for reconstruction, causing gradient imbalance for heteroscedastic variables; switching to NLL, though seemingly minor, is key to mixed-type tabular fidelity.
- **BGMM post hoc fitting as an engineering philosophy**: The VAE training objective remains unchanged, only the latent space is post-processed—this increases expressiveness without sacrificing training stability, a "clever not complex" design.
- **Joint modeling of both potential outcomes**: Unlike Credence, which only models $Y\mid X,T$, this work's decoder outputs both $Y(0), Y(1)$, allowing direct computation and penalty supervision of $\tau_\theta$—crucial for causal control to "truly take effect" rather than just "appear correct".
- **Comprehensive evaluation pipeline**: Three-layer evaluation—distributional fidelity (marginal/pairwise/conditional/joint), causal fidelity (MAE/correlation/Wasserstein), and privacy (DCR)—sets an evaluation paradigm for future causal sandbox papers.
- **Real clinical deployment**: Not just a toy benchmark, but applied to real prostate cancer cases for CATE estimator benchmarking, hyperparameter tuning, and power analysis, providing direct value to clinical statisticians.

## Limitations & Future Work

- **Relies on correctly specified causal functions**: Users must provide analytic forms for $\tau(X), \kappa(X,T), \alpha(X)$; powerless for "unknown-shape" causal functions in real clinical settings—essentially a benchmarking tool, not a discovery tool.
- **Modeling of unobserved confounding remains a black box**: $\kappa(X,T)$ is implemented via the difference between two sets of potential outcomes from the decoder, but there is no explicit latent confounder variable; difficult to simulate structured scenarios where "strong unobserved confounder affects T and Y via specific mechanisms".
- **Multi-head decoder complexity in high-dimensional X**: One head per variable; scaling to hundreds of dimensions inflates network size, and experiments only use 18 dimensions.
- **Hyperparameter sensitivity not fully studied**: Rigidness $\lambda_\tau, \lambda_\kappa, \lambda_\alpha$ are set empirically ($10^3, 10^3, 10^{1\text{-}2}$), lacking automatic selection; the Pareto front of fidelity vs control under different $\lambda$ combinations is not quantified.
- **Variance regularizer may over-suppress true heterogeneity**: In genuinely high-heterogeneity scenarios, the variance penalty may flatten real unit-level dispersion; lacks a mechanism to distinguish "true heterogeneity vs noise".
- **Does not cover longitudinal/survival outcomes**: Currently supports only single time-point binary treatment + scalar outcome; longitudinal data, time-varying confounding, and survival analysis—common in clinical settings—are not supported.

## Related Work & Insights

- **vs RealCause (Neal 2020)**: RealCause uses normalizing flow, with causal control limited to interpolation between fitted extremes; CausalMix allows arbitrary design of $\tau, \kappa, \alpha$ with explicit penalties to ensure implementation.
- **vs WGAN-based generators (Athey 2024)**: WGAN excels at distributional fit but lacks any effect control interface; CausalMix incorporates causal constraints into the loss.
- **vs Credence (Parikh 2024)**: Credence also allows prespecification of $\tau, \kappa$, but uses MSE and does not support multi-modal/multi-type; this work combines Huber + BGMM + multi-head for greater stability on mixed-type multimodal real data.
- **vs Frengression (Zhang 2026)**: Frengression controls marginal causal quantities, but does not support fine-grained specification of conditional CATE/overlap/unobserved confounding.
- **vs Plasmode simulations**: Traditional plasmode uses real X + known Y model to inject truth; CausalMix extends this paradigm to fully synthetic $X', T', Y'$, with $X$ also generated.
- **Insights**: The unified objective + decoupled penalty design can be transferred to other scenarios requiring both data fit and structural constraints, such as fair generation (fit + group fairness), constrained simulation (fit + physical laws), etc.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating mixed-type VAE, multimodal prior, and three-layer causal penalty into a unified framework is a meaningful synthesis; Huber + variance reg for weak signals is a careful engineering contribution, though individual components are not original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three progressively complex scenarios + BGMM ablation + comprehensive fidelity/causal/privacy evaluation + real clinical case + CATE estimator benchmarking + hyperparameter sensitivity + power analysis; the evaluation pipeline sets an industry benchmark.
- Writing Quality: ⭐⭐⭐⭐ Method section is clearly organized, with thorough explanation of loss and design motivation; a few symbols (e.g., direction conventions for $\kappa$ and $\alpha$) require cross-referencing; friendly for statistical readers, but ML readers may need to adapt to causal notation.
- Value: ⭐⭐⭐⭐ Direct value for clinical statisticians, causal ML methodologists, and pharma RWE teams; open-source code + real clinical case enhance reproducibility and traction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)

</div>

<!-- RELATED:END -->
