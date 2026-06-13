---
title: >-
  [Paper Note] Controllable Generative Sandbox for Causal Inference
description: >-
  [ICML 2026][Causal Inference][CausalMix] This paper proposes CausalMix: a variational generative framework that jointly optimizes a type-specific multi-head decoder and a Bayesian Gaussian Mixture latent prior with three…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "CausalMix"
  - "conditional VAE"
  - "Bayesian GMM prior"
  - "overlap regularizer"
  - "CATE benchmarking"
date: 2026-05-08
content_hash: b00fc0e5b68f7cd9
---

# Controllable Generative Sandbox for Causal Inference

**Conference**: ICML 2026  
**arXiv**: [2603.03587](https://arxiv.org/abs/2603.03587)  
**Code**: https://github.com/zhangqiecho/causalmix  
**Area**: Causal Inference / Medical Statistics; Generative Models for Methodology Validation; Synthetic Data Benchmark  
**Keywords**: CausalMix, conditional VAE, Bayesian GMM prior, overlap regularizer, CATE benchmarking

## TL;DR
This paper proposes CausalMix: a variational generative framework that jointly optimizes a type-specific multi-head decoder and a Bayesian Gaussian Mixture latent prior with three categories of independently controllable causal "knobs" (overlap $\alpha(X)$, CATE function $\tau(X)$, and unobserved confounding $\kappa(X,T)$). While maintaining real-world data distribution fidelity, it allows users to freely design counterfactual benchmarks. Validation on real-world mCRPC (prostate cancer) cases shows that CausalMix high-faithfully reproduces mixed-type tables and stably injects overlap, confounding, and heterogeneous effects on demand, serving as a controllable stress test for CATE estimators.

## Background & Motivation

**Background**: Evaluation of causal inference methods (meta-learners, DR-learners, DML, causal forest, BCF) relies heavily on **synthetic data with ground-truth counterfactuals**, as $Y(1)$ and $Y(0)$ cannot be observed simultaneously in real data. Existing simulators include: purely parametric (controllable but unrealistic), semi-synthetic (using real X to simulate T/Y with limited control), and data-fit generators (RealCause, WGAN, Credence, etc., which use neural models to fit DGP with high realism but weak causal controllability).

**Limitations of Prior Work**: (i) Existing data-fit generators show poor fidelity on **mixed-type tabular data** (continuous + binary + categorical + integer), either introducing spurious correlations via forced one-hot encoding or losing multi-element structures with a single likelihood loss; (ii) **Causal knobs are missing or coupled**—RealCause only interpolates between fitted extremes, WGAN lacks an effect control interface, and Credence lacks support for multi-modal mixed-type data; (iii) Even when $\tau(X)$ can be specified, there is no mechanism to **verify whether the generator faithfully realizes it**, especially when the causal function is low-dimensional or weakly non-linear and easily overwhelmed by reconstruction loss.

**Key Challenge**: There is a natural trade-off between distributional realism (fitting observed data) and causal controllability (faithfully realizing user-specified $\tau, \kappa, \alpha$). Tighter fits reduce degrees of freedom, while higher freedom leads to greater deviations from real data. Existing methods sacrifice either controllability (neural generators) or realism (parametric simulators).

**Goal**: (i) Jointly optimize distribution fidelity and causal constraints under a unified objective to avoid the trade-off; (ii) Achieve high fidelity on mixed-type tabular data; (iii) Provide three **orthogonal and independently controllable** causal knobs for overlap, confounding, and heterogeneity, accompanied by a quantitative verification pipeline; (iv) Demonstrate utility in real clinical scenarios (safety comparisons in mCRPC).

**Key Insight**: Leveraging a conditional VAE as the generative backbone (proven stable for tabular data with analytical ELBO), causal constraints are formulated as differentiable penalties on the decoder output. Mean alignment and variance regularization ensure faithful realization of low-dimensional causal functions, while the standard isotropic Gaussian prior is replaced with a Bayesian GMM to recover the multi-modal structure of mixed-type data.

**Core Idea**: "Distribution fitting" and "causal control" are treated as two sets of terms in a unified loss function, explicitly controlled by rigidity hyperparameters $\lambda_\alpha, \lambda_\tau, \lambda_\kappa$. A mixture prior handles multi-modality, a multi-head decoder handles mixed types, and a three-layer penalty manages three causal dimensions—addressing fidelity, control, and mixed-type data simultaneously.

## Method

### Overall Architecture
Given observations $\mathcal{O} = (X, T, Y)$, where $X$ denotes mixed-type covariates, $T\in\{0,1\}$, and $Y$ is the outcome, a generator $G_\theta$ is learned and modularized into three parts:

- **Treatment Model** $p(T)$: Bernoulli;
- **Pre-treatment Model** $G_{X,\theta}$: A conditional VAE modeling $X\mid T$;
- **Post-treatment Model** $G_{Y,\theta}$: A conditional VAE jointly modeling $(Y(0), Y(1))\mid X, T$, **outputting both potential outcomes simultaneously**.

The generation sequence follows $T'\to X'\mid T'\to (Y'(0), Y'(1))\mid X', T'\to Y' = T'Y'(1)+(1-T')Y'(0)$. The decoder uses multi-heads: continuous (Gaussian), binary (Bernoulli), and categorical (softmax). After training, the standard Gaussian latent prior is replaced with a Bayesian GMM (Dirichlet-process prior).

**Unified Objective**:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{VAE}} + \lambda_\alpha \mathcal{L}_\alpha + \lambda_\tau \mathcal{L}_\tau^{\text{mean}} + \lambda_\tau^{\text{var}}\mathcal{L}_\tau^{\text{var}} + \lambda_\kappa \mathcal{L}_\kappa^{\text{mean}} + \lambda_\kappa^{\text{var}}\mathcal{L}_\kappa^{\text{var}}$$

### Key Designs

1. **Three Independent Causal "Knobs" + Huber-style Penalty**:
    - **Function**: Allows users to independently specify overlap $\alpha(X)$, CATE $\tau(X)$, and unobserved confounding $\kappa(X,T)$ during generation, ensuring the learned $G_\theta$ faithfully satisfies them.
    - **Mechanism**: The target causal quantities and the generator-induced quantities are defined as $\alpha(x) = P(X=x\mid T=0)/P(X=x\mid T=1)$, $\tau(x) = \mathbb{E}[Y(1)-Y(0)\mid X=x]$, and $\kappa(x,t) = \mathbb{E}[Y(t)\mid X=x,T=1] - \mathbb{E}[Y(t)\mid X=x,T=0]$. The training penalties are:
        - **Overlap**: $\mathcal{L}_\alpha = \mathbb{E}_X[(\log\alpha_\theta(X) - \log\alpha(X))^2]$, performing MSE alignment directly on the log-density ratio provided by the decoder;
        - **CATE**: $\mathcal{L}_\tau^{\text{mean}} = \mathbb{E}_X[\lambda_\tau^{\text{mse}}(\Delta\tau_\theta)^2 + \lambda_\tau^{\text{sl1}}\text{SmoothL1}(\Delta\tau_\theta)]$. This **composite Huber loss** uses a quadratic term to anchor the mean and SmoothL1 to improve robustness against outliers and weakly identified regions, plus a variance penalty $\mathcal{L}_\tau^{\text{var}} = \text{Var}[\Delta\tau_\theta]$ to suppress spurious unit-level dispersion;
        - **Confounding**: Uses the same composite Huber + variance structure for $\kappa$.
    - **Design Motivation**: Pure MSE is easily overwhelmed by reconstruction loss when $\tau, \kappa$ are low-dimensional or weakly non-linear; the Huber + variance regularizer anchors the mean while compressing dispersion, allowing causal constraints to be realized stably even in low-signal scenarios. Independent adjustment of the three $\lambda$ values enables factorial studies (e.g., varying overlap and confounding simultaneously to test CATE estimator robustness).

2. **Mixed-type Multi-head Decoder + Bayesian GMM Prior**:
    - **Function**: Faithfully reproduces the mixed-type and multi-modal structure of real tables, avoiding spurious correlations caused by type confusion.
    - **Mechanism**: Each variable is assigned an independent likelihood head based on its data type—Gaussian NLL for continuous variables (rather than MSE as in Credence), allowing the decoder to learn both location and dispersion, which is critical for heteroscedastic or bounded variables; Bernoulli logits for binary; softmax for categorical; and integers processed as continuous with rounding. The encoder still outputs a diagonal Gaussian posterior, but after training, a Bayesian GMM (Dirichlet-process prior with truncated stick-breaking variational inference) is fitted to the latent means as the generation prior: $p_{\text{BGMM}}(z) = \sum_k \pi_k \mathcal{N}(z\mid\mu_k, \Sigma_k)$, with $K$ learned automatically.
    - **Design Motivation**: The standard isotropic Gaussian prior assumes a unimodal latent space, which is highly inconsistent with real clinical data where patients naturally cluster into sub-populations. BGMM utilizes a Dirichlet process to automatically select $K$ and is **fitted post-hoc** without altering the VAE training objective—increasing expressiveness while maintaining decoder stability. The choice of Gaussian NLL over MSE is a detail often overlooked but crucial; MSE does not learn variance, leading to imbalanced gradient scales for heteroscedastic variables.

3. **Joint Optimization + Modular Phasing (Decoupling X and Y)**:
    - **Function**: Co-trains distributional fit and causal control within the same mini-batch while optimizing X-generator and Y-generator independently to reduce coupling.
    - **Mechanism**: Pre-treatment $G_{X,\theta}$ optimizes only $\mathcal{L}_{\text{VAE}}^X + \lambda_\alpha\mathcal{L}_\alpha$ (X reconstruction + overlap control); Post-treatment $G_{Y,\theta}$ optimizes $\mathcal{L}_{\text{VAE}}^Y + \lambda_\tau^{\text{mean}} + \cdots + \lambda_\kappa^{\text{mean}} + \cdots$. In the latter, the decoder evaluates both potential outcomes $Y(0)$ and $Y(1)$ simultaneously during training (even if only one is observed) to calculate $\tau_\theta, \kappa_\theta$. Early stopping is based on validation loss.
    - **Design Motivation**: Causal mechanisms for X and Y differ—overlap on X is a marginal distribution issue, while $\tau, \kappa$ on Y are conditional expectation issues. Separate training allows rigidity hyperparameters for each module to be tuned independently, preventing penalties in one module from interfering with another. **Evaluating both potential outcomes** is key to directly expressing $\tau$, contrasting with methods like Credence that only model $Y\mid X,T$.

### Loss & Training
- Optimizer: Adam (lr = $10^{-3}$), 80/20 train/val split, PyTorch Lightning;
- Key Hyperparameters: $\lambda_\tau, \lambda_\kappa$ fixed at $10^3$; $\lambda_\alpha$ between $10^1\text{–}10^2$ (overlap is more sensitive to misspecification);
- For low-dimensional/weakly non-linear control functions: Reduce MSE weight (0.2–0.4), increase SmoothL1 + variance reg;
- Post-training: Fit BGMM (DP prior, max K = latent dim) as the generative prior.

## Key Experimental Results

### Main Results (mCRPC Cases: Abiraterone vs. Enzalutamide, 4,098 patients, 18 baseline covariates)

| Scenario | Setting | Key PHenomena |
|------|------|---------|
| Scenario 1 | $\tau\equiv 0.1, \kappa\equiv 0, \log\alpha\equiv 0$ (Constant effect, no confounding, perfect overlap) | Sanity check: Both BGMM and Gaussian prior recover the distribution successfully |
| Scenario 2 | Linear $\tau$ (CVD, age, Charlson), $\kappa\equiv 0.02$, $\log\alpha\equiv 1$ | Both priors perform reasonably, BGMM slightly superior |
| **Scenario 3** | Non-linear tanh $\tau$ (CVD, age, Charlson, dementia), $\kappa$ jointly dependent on $X,T$, $\log\alpha = 2(2\cdot\text{Abi\_prev}-1)$ | **BGMM wins significantly**: CATE correlation and decoder-level overlap reconstruction are notably superior to Gaussian |

### Ablation Study

| Configuration | Key Effect | Remarks |
|------|---------|------|
| Gaussian prior vs BGMM | BGMM wins across the board in Scenario 3 | Multi-modal prior is necessary for complex scenarios |
| Gaussian NLL vs MSE (continuous) | NLL significantly better (especially for heteroscedastic variables) | Learning variance is essential for correct modeling |
| Composite Huber (MSE+SmoothL1+var) vs Pure MSE | Huber stable under low-dimensional $\tau$, pure MSE erratic | Variance regularizer is a key stabilizer |
| Privacy Trade-off | Gaussian prior provides stronger privacy, BGMM slightly weaker but protection > 0.5 | A controllable trade-off between realism and privacy |

### Key Findings
- **The value of BGMM scales with causal complexity**: While both priors performed adequately in Scenario 1/2, BGMM crushed Gaussian in Scenario 3 across normalized Wasserstein, C2ST, CATE correlation, and overlap reconstruction—multi-modal priors are essential for multi-peak clinical data.
- **The privacy-realism trade-off is manageable**: BGMM has slightly weaker privacy protection due to higher realism, but DCR protection fraction remains $>0.5$ and median distance ratio $>1$, showing no systemic memorization; decreases are concentrated in low percentiles (local proximity rather than broad leakage).
- **Causal knobs are faithfully realized**: Even in the complex Scenario 3, CATE MAE/Pearson, $\kappa$ MAE, and overlap MSE reached acceptable precision, proving the effectiveness of the unified loss and Huber + variance reg design.
- **CATE estimator benchmarking**: Comparing X-learner, DR-learner, DML, Causal Forest, and BCF on the Scenario 3 calibrated DGP (Fig. 4) reveals which estimators are more stable in specific overlap/confounding regions—highlighting the value of CausalMix.
- **Visualizing Causal Forest hyperparameter sensitivity** (Fig. 5): PEHE displays an atypical shape relative to min leaf size under the Scenario 3 DGP, providing direct answers on "how to tune for your clinical scenario"—an insight parametric simulators cannot provide.

## Highlights & Insights
- **"Realism vs. Controllability" is no longer a trade-off**: The unified loss places both objectives in a single target, with rigidity hyperparameters allowing users to explicitly tune weights, enabling benchmark designers to achieve both for the first time.
- **The multi-head decoder + Gaussian NLL is an underrated detail**: Many tabular generative models use MSE for reconstruction, leading to gradient imbalance for heteroscedastic variables. Replacing it with NLL seems minor but is core to mixed-type tabular fidelity.
- **The engineering philosophy of post-hoc BGMM fitting**: By not altering the VAE training objective and only post-processing the latent space, expressiveness is increased without breaking training stability—a "smart, not complex" design.
- **Joint modeling of both potential outcomes**: Unlike Credence which only models $Y\mid X,T$, this approach outputs $Y(0)$ and $Y(1)$ simultaneously, allowing $\tau_\theta$ to be calculated directly and supervised by the penalty—ensuring causal control "actually works" rather than just "looking correct."
- **Complete evaluation pipeline**: Distribution fidelity (marginal/pairwise/conditional/joint) + Causal fidelity (MAE/correlation/Wasserstein) + Privacy (DCR) evaluation layers set a new standard for causal sandbox papers.
- **Real-world clinical utility**: Beyond toy benchmarks, application to CATE estimator benchmarking, hyperparameter tuning, and power analysis in real mCRPC cases provides direct value to clinical statisticians.

## Limitations & Future Work
- **Reliance on correctly specified causal functions**: Users must provide analytical forms for $\tau(X), \kappa(X,T), \alpha(X)$; the tool cannot handle "unknown shape" causal functions—it is a benchmarking tool, not a discovery tool.
- **Modeling of unobserved confounding remains a black box**: $\kappa(X,T)$ is implemented via the difference between potential outcome outputs but lacks explicit latent confounder variables, making it difficult to simulate structured scenarios where strong unobserved confounders affect T and Y through specific mechanisms.
- **Multi-head decoder complexity in high dimensions**: Assigning a head to every variable causes the network size to expand as tables reach hundreds of dimensions; this study used only 18 dimensions.
- **Insufficient study of hyperparameter sensitivity**: Rigidness settings for $\lambda_\tau, \lambda_\kappa, \lambda_\alpha$ rely on heuristics ($10^3, 10^3, 10^{1\text{-}2}$), lacking an automated selection scheme or a quantified Pareto front for fidelity vs. control.
- **Variance regularizer may over-suppress legitimate heterogeneity**: In valid high-heterogeneity scenarios, the variance penalty might flatten true unit-level dispersion; the mechanism lacks the ability to distinguish "true heterogeneity" from "noise."
- **Lack of longitudinal/survival outcome support**: Currently restricted to single-timepoint binary treatments and scalar outcomes; longitudinal data, time-varying confounding, and survival analysis are common in clinical practice but not yet supported.

## Related Work & Insights
- **vs. RealCause (Neal 2020)**: RealCause uses normalizing flows, where causal control is restricted to interpolating between fitted extremes; CausalMix allows any $\tau, \kappa, \alpha$ design with explicit penalty guarantees.
- **vs. WGAN-based generators (Athey 2024)**: WGAN excels at distributional fit but lacks effect control interfaces; CausalMix integrates causal constraints into the loss.
- **vs. Credence (Parikh 2024)**: Credence also allows prespecifying $\tau, \kappa$, but uses MSE and lacks support for multi-modality/mixed-types. Ours uses Huber + BGMM + multi-head, demonstrating greater stability on real mixed-type data.
- **vs. Frengression (Zhang 2026)**: Frengression controls marginal causal quantities but does not support fine-grained specification of conditional CATE, overlap, or unobserved confounding.
- **vs. Plasmode simulations**: Traditional plasmode uses real X + known Y models. CausalMix extends this to fully synthetic $X', T', Y'$, where $X$ is also generated.
- **Insights**: The design of a unified objective with decoupled penalties is transferable to other scenarios requiring both data fitting and structural constraints, such as fair generation (fit + group fairness) or constrained simulation (fit + physical laws).

## Rating
- Novelty: ⭐⭐⭐⭐ Integrating mixed-type VAE, multi-modal priors, and three-layer causal penalties into a unified framework is a meaningful consolidation; Huber + variance reg for weak signals is a meticulous engineering contribution, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three increasing complexity scenarios, BGMM ablation, a complete three-layer fidelity/causal/privacy evaluation, real clinical cases, CATE estimator benchmarking, hyperparameter sensitivity, and power analysis. This evaluation pipeline sets an industry benchmark.
- Writing Quality: ⭐⭐⭐⭐ The Method section is well-organized, with clear design motivations for the loss functions. Some notation (e.g., directional conventions for $\kappa$ and $\alpha$) requires close attention; friendly to statistical readers, though ML readers may need to adapt to causal notation.
- Value: ⭐⭐⭐⭐ Direct utility for clinical statisticians, causal ML researchers, and pharmaceutical RWE teams. Public code and real clinical cases enhance reproducibility and traction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](../../ACL2026/causal_inference/learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[NeurIPS 2025\] Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference](../../NeurIPS2025/causal_inference/practical_do-shapley_explanations_with_estimand-agnostic_causal_inference.md)
- [\[AAAI 2026\] Causal Inference Under Threshold Manipulation: Bayesian Mixture Modeling and Heterogeneous Treatment Effects](../../AAAI2026/causal_inference/causal_inference_under_threshold_manipulation_bayesian_mixtu.md)
- [\[NeurIPS 2025\] GST-UNet: A Neural Framework for Spatiotemporal Causal Inference with Time-Varying Confounding](../../NeurIPS2025/causal_inference/gst-unet_a_neural_framework_for_spatiotemporal_causal_inference_with_time-varyin.md)

</div>

<!-- RELATED:END -->
