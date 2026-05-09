---
title: >-
  [Paper Note] An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes
description: >-
  [ICLR 2026][Medical Imaging][Q-function estimation] This paper systematically introduces semiparametric efficiency theory from causal inference into Q-function estimation for MDPs. It demonstrates that classical Q-regression and FQE are essentially naive plug-in learners subject to plug-in bias, and proposes the DRQQ-learner—a meta-learner that simultaneously achieves double robustness, Neyman orthogonality, and near-oracle efficiency. By deriving the efficient influence function (EIF) to construct a debiased two-stage loss, the method comprehensively outperforms baselines in Taxi and Frozen Lake environments.
tags:
  - ICLR 2026
  - Medical Imaging
  - Q-function estimation
  - double robustness
  - Neyman orthogonality
  - causal inference
  - offline policy evaluation
date: 2026-05-08
content_hash: a238ea59fad5900d
---

# An Orthogonal Learner for Individualized Outcomes in Markov Decision Processes

**Conference**: ICLR 2026
**arXiv**: [2509.26429](https://arxiv.org/abs/2509.26429)
**Code**: [EmilJavurek/Orthogonal-Q-in-MDPs](https://github.com/EmilJavurek/Orthogonal-Q-in-MDPs)
**Area**: Medical Imaging
**Keywords**: Q-function estimation, double robustness, Neyman orthogonality, causal inference, offline policy evaluation

## TL;DR

This paper systematically introduces semiparametric efficiency theory from causal inference into Q-function estimation for MDPs. It demonstrates that classical Q-regression and FQE are essentially naive plug-in learners subject to plug-in bias, and proposes the DRQQ-learner—a meta-learner that simultaneously achieves double robustness, Neyman orthogonality, and near-oracle efficiency. By deriving the efficient influence function (EIF) to construct a debiased two-stage loss, the method comprehensively outperforms baselines in Taxi and Frozen Lake environments.

## Background & Motivation

**State of the Field**: In settings such as personalized medicine, estimating individualized potential outcomes under a target policy from observational data—i.e., estimating the Q-function—is a core task. Existing methods such as Q-regression (Liu et al., 2018) and FQE (Le et al., 2019) primarily focus on breaking the "curse of horizon" (the exponential blow-up of cumulative density ratios over time steps), but lack theoretical guarantees regarding the statistical properties of the estimators themselves.

**Limitations of Prior Work**: Q-regression adjusts via inverse probability weighting (IPTW) using cumulative density ratios $\rho_{1:t}$, leading to variance explosion under long horizons. FQE avoids cumulative density ratios through recursive Bellman equation fitting, but suffers from the "deadly triad" (the unstable combination of function approximation, bootstrapping, and off-policy learning), which can cause divergence. More critically, neither method offers any theoretical guarantee of orthogonality or efficiency.

**Root Cause**: From a causal inference perspective, both Q-regression and FQE are plug-in learners—they directly substitute estimated nuisance parameters into the identification formula. The fundamental flaw of plug-in estimation is that first-order estimation errors in nuisance parameters propagate linearly into the final estimator, limiting convergence to the rate of the slowest nuisance component. In static causal inference, doubly robust (DR) and Neyman orthogonal methods have successfully addressed this issue, but their extension to the sequential decision-making framework of MDPs is nontrivial due to the recursive structure of the Bellman equation and cross-temporal error propagation.

**Paper Goals**: (1) How can one derive the efficient influence function (EIF) for Q-function estimation in MDPs from semiparametric efficiency theory? (2) How can one construct a Neyman orthogonal loss based on the EIF such that nuisance estimation errors affect the final estimate only through second-order terms? (3) Can the resulting estimator simultaneously achieve double robustness and near-oracle efficiency?

**Starting Point**: The authors observe that the Q-function is fundamentally a causal estimand (a potential outcome under evaluation policy $\pi_e$), which can be formalized via the potential outcomes framework. Once this causal interpretation is established, mature debiasing tools from semiparametric statistics (EIF, orthogonal learning, cross-fitting) can be systematically applied to improve estimation.

**Core Idea**: Derive the efficient influence function for the Q-function MSE loss, and construct a two-stage meta-learner that first estimates nuisance functions and then applies EIF-based debiasing, endowing the Q-function estimator with double robustness, Neyman orthogonality, and near-oracle efficiency simultaneously.

## Method

### Overall Architecture

The DRQQ-learner is a two-stage meta-learner. Its input is an observational dataset $\mathcal{D}_{\pi_b}$ generated by a behavior policy $\pi_b$ (decomposed into one-step transitions $(S, A, R, \tilde{S})$ from i.i.d. trajectories) and a target evaluation policy $\pi_e$. The first stage estimates three nuisance functions: the behavior policy $\hat{\pi}_b$, the density ratio $\hat{w}_{e/b}$, and an initial Q-function estimate $\hat{Q}^1_{\pi_e}$. The second stage refines the Q-function estimate by minimizing a Neyman orthogonal loss $\hat{L}^3_{\pi_e}$ derived from the EIF, yielding the final $\hat{Q}^{DR}_{\pi_e}$. The framework imposes no restrictions on the specific model choices in the first stage—neural networks, linear models, or tabular methods are all admissible—and the model class $\mathcal{G}$ in the second stage can likewise be specified flexibly (e.g., linear models when interpretability is required).

### Key Designs

1. **Causal Identification and Plug-in Learner Diagnosis**

    - **Function**: Formalizes the Q-function $Q_{\pi_e}(s,a)$ as a causal estimand and demonstrates that existing methods are biased plug-in learners.
    - **Mechanism**: Through the potential outcomes framework, the Q-function is defined as the causal parameter $\xi_{\pi_e}(s,a) = \mathbb{E}[R_0 + \sum_{t=1}^{\infty} \gamma^t R_t[\pi_e(\cdot|S_t)] \mid S_0=s, A_0=a]$. Under standard assumptions (Markovianity, no unobserved confounding, positivity), two identification paths exist: trajectory-based IPTW identification (requiring cumulative density ratios $\rho_{1:t}$, corresponding to Q-regression) and one-step-transition Bellman equation identification (corresponding to FQE). By computing the Gâteaux derivatives of these losses with respect to nuisance parameters and showing they are nonzero, bias is shown to propagate linearly with nuisance estimation error—the defining characteristic of plug-in bias.
    - **Design Motivation**: Rather than proposing new identification results, the goal is to diagnose the deficiencies of existing methods within a unified causal inference framework, providing a theoretical foundation for subsequent debiasing.

2. **EIF Derivation and Neyman Orthogonal Loss Construction**

    - **Function**: Derives the EIF of the standard MSE loss for the Q-function and uses it to construct the Neyman orthogonal debiased loss $L^3_{\pi_e}$.
    - **Mechanism**: The EIF is derived for the standard MSE population risk $L^1_{\pi_e}(\eta, g) = \mathbb{E}_{pb}[\sum_a \pi_e(a|S)(Q_{\pi_e}(S,a) - g(S,a))^2]$. The key correction term in the EIF contains the Bellman residual $R' + \gamma v_{\pi_e}(\tilde{S}') - Q_{\pi_e}(S', A')$ multiplied by density ratios $\pi_e/\pi_b$ and $w_{e/b}$, corresponding respectively to "local" debiasing (for action distribution shift at the current state) and "global" debiasing (for state visitation distribution shift). The final orthogonal loss incorporates two pseudo-outcomes $\phi_1, \phi_2$, replacing the original MSE objective with a debiased target containing correction terms. Neyman orthogonality implies $D_\eta D_g L^3(g^*, \eta)[\hat{g}-g, \hat{\eta}-\eta] = 0$, i.e., the loss gradient is first-order insensitive to perturbations in nuisance parameters around their true values.
    - **Design Motivation**: Neyman orthogonality ensures that nuisance estimation errors affect the final estimate only through second-order (product) terms, so that even if nuisance models converge slowly, the overall estimator can still achieve fast convergence rates.

3. **Double Robustness and Near-Oracle Efficiency Guarantees**

    - **Function**: Proves that the DRQQ-learner simultaneously achieves double robustness and near-oracle efficiency.
    - **Mechanism**: The central near-oracle efficiency result is the excess risk bound $\|g^* - \hat{g}\|^2 \lesssim \|\Delta^2 \hat{\pi}_b\|^2 \|\Delta^2 \hat{Q}_{\pi_e}\|^2 + \|\Delta^2 \hat{w}_{e/b}\|^2 \|\Delta^2 \hat{Q}_{\pi_e}\|^2$, i.e., the error depends only on products of nuisance estimation errors. Double robustness follows as a direct corollary: as long as either $\hat{Q}^1_{\pi_e}$ is consistent or $(\hat{\pi}_b, \hat{w}_{e/b})$ are jointly consistent, the final estimator is consistent—only one of the two nuisance groups need be correctly specified.
    - **Design Motivation**: Double robustness provides a safety net in high-stakes domains such as healthcare—model misspecification is nearly inevitable, and the DR property allows consistent estimation even when only one of the two model groups is correct.

### Loss & Training

The three nuisance models in the first stage are trained separately via standard supervised learning: $\hat{\pi}_b$ is a classification model fitting the behavior policy; $\hat{w}_{e/b}$ is a density ratio estimator obtained by estimating the ratio of transition probabilities and stationary distributions; $\hat{Q}^1_{\pi_e}$ can be initialized using any existing Q-estimation method (e.g., FQE). In the second stage, given the nuisance estimates, empirical risk minimization is performed over the orthogonal loss $\hat{L}^3_{\pi_e}$, with support for cross-fitting (sample splitting) to avoid overfitting bias. The entire pipeline is model-agnostic: both nuisance and target models can be instantiated with arbitrary ML architectures (neural networks, linear models, tabular methods, etc.).

## Key Experimental Results

### Main Results

Experiments are conducted on OpenAI Gym's Taxi and Frozen Lake environments. Both the behavior policy $\pi_b$ and evaluation policy $\pi_e$ are epsilon-greedy. The evaluation metric is relative MSE: $\text{rMSE} = \|\hat{Q} - Q_{\pi_e}\|_2^2 / \|Q_{\pi_e}\|_2^2$.

| Setting | Q-regression | FQE | MQL | DRQQ (Ours) |
|---------|-------------|-----|-----|------------|
| Taxi, n=4000, unrestricted $\mathcal{G}$ | High rMSE, affected by curse of horizon | Moderate rMSE | Moderate rMSE | **Lowest rMSE** |
| Taxi, long horizon ($h$=20) | rMSE increases significantly | Largely stable | Moderate | **Consistently optimal** |
| Taxi, low overlap (ε_e=0.1) | rMSE explodes | Moderate increase | Moderate | **Significantly outperforms all baselines** |
| Frozen Lake, unrestricted $\mathcal{G}$ | High rMSE | Moderate | Moderate | **Lowest rMSE** |
| Taxi, linear $\mathcal{G}$ | Moderate | Moderate | Moderate | **Best in most settings** |

### Ablation Study

The paper conducts systematic ablations across three dimensions to validate theoretical properties:

| Variable | Range | Theoretical Property Verified | DRQQ Performance |
|---------|---------|-------------|---------|
| Dataset size $n$ | 2000→6000 | Convergence rate | Steadily decreases with $n$; consistently outperforms baselines |
| Effective horizon $h=1/(1-\gamma)$ | 3→20 | Breaking curse of horizon | rMSE insensitive to $h$; Q-regression degrades sharply |
| Overlap (ε_e) | 0.1→0.9 | Low-overlap advantage of Neyman orthogonality | Largest advantage at low overlap; gap narrows at high overlap |
| Model class restriction | Unrestricted vs. linear | Applicability to restricted $\mathcal{G}$ | Still best under linear $\mathcal{G}$ |

### Key Findings

- **Low overlap is the setting where DRQQ-learner has the greatest advantage**: When the evaluation and behavior policies differ substantially, the bias of plug-in methods is amplified, whereas DRQQ's Neyman orthogonality renders it insensitive to such distribution shift. The improvement of DRQQ over FQE in low-overlap settings is substantially larger than in high-overlap settings.
- **Q-regression is severely affected by the curse of horizon**: As the effective horizon increases from 3 to 20, Q-regression's rMSE rises sharply due to the exponentially growing variance of cumulative density ratios $\rho_{1:t}$. DRQQ and FQE, which use one-step transitions, are unaffected.
- **Restricting the model class does not impair effectiveness**: DRQQ remains the best-performing method under linear $\mathcal{G}$, validating the generality of the theoretical framework and demonstrating applicability to healthcare scenarios requiring interpretable models.
- **DRQQ's advantage diminishes but does not reverse at high overlap**: When $\pi_e$ approaches a random policy, density ratios approach 1, nuisance estimation becomes easier for all methods, and the debiasing advantage of DRQQ naturally decreases—but it does not underperform baselines.

## Highlights & Insights

- **Theoretical bridge between causal inference and RL**: This is the first work to fully introduce semiparametric efficiency theory into Q-function estimation for MDPs, unifying Q-regression and FQE as plug-in learners and precisely diagnosing the source of their bias. Beyond proposing a new method, it provides a unified theoretical lens for analyzing and comparing different Q-estimation approaches.
- **"Coarse estimation then debiasing" meta-learning paradigm**: The two-stage structure of the DRQQ-learner is notably elegant—the first stage can use any off-the-shelf method (even biased FQE) for rough nuisance estimation, while the second stage automatically corrects bias through the EIF-derived orthogonal loss. This "model-agnostic debiasing wrapper" paradigm is directly transferable to other sequential decision-making estimation problems involving nuisance parameters.
- **Practical value of double robustness**: In settings such as precision medicine, both the behavior policy $\pi_b$ and transition dynamics are typically difficult to estimate accurately. The DR property provides fault tolerance—as long as either the initial Q-function estimate or the density model is correctly specified, the final result is consistent.

## Limitations & Future Work

- **Limited experimental scope**: Validation is restricted to two small-scale discrete environments—Taxi (25×5 states × 6 actions) and Frozen Lake (16 states × 4 actions). Although the theory supports continuous state spaces, no experimental validation in continuous or high-dimensional settings is provided, leaving a substantial gap from practical medical applications (high-dimensional patient features, continuous dosing).
- **Density ratio $w_{e/b}$ estimation is difficult in practice**: This nuisance involves the ratio of stationary distributions, which is extremely challenging to estimate in continuous or large-scale discrete state spaces and may become a practical bottleneck for the method.
- **Doubled computational cost**: Training three nuisance models followed by a second-stage optimization significantly increases computational cost compared to a single FQE training pass, with cross-fitting adding further overhead.
- **Extension to non-stationary MDPs**: The current theory strictly requires time-homogeneous MDPs and stationary policies, whereas in real medical settings patient state transitions and treatment policies often vary over time; extension to non-stationary settings requires re-deriving the EIF.

## Related Work & Insights

- **vs. Q-regression (Liu et al., 2018)**: Q-regression adjusts for distribution shift via IPTW using cumulative density ratios from full trajectories. This paper demonstrates it is a trajectory-based plug-in learner subject to the curse of horizon and first-order plug-in bias. DRQQ uses one-step transitions to avoid the horizon problem and employs an orthogonal loss to eliminate plug-in bias.
- **vs. FQE (Le et al., 2019)**: FQE uses one-step transitions via recursive Bellman equation fitting, successfully breaking the curse of horizon. This paper demonstrates it is a Bellman-based plug-in learner that still exhibits first-order plug-in bias. DRQQ augments FQE with an EIF correction term to achieve Neyman orthogonality.
- **vs. Shi et al. (2021) debiased OPE**: Shi et al. derive a pointwise iterative debiasing method for OPE confidence intervals, which coincides with a special case of DRQQ in the discrete setting. DRQQ's advantages include: (1) applicability to continuous state spaces (Shi's method contains Dirac delta functions that do not generalize directly); (2) support for restricted model classes $\mathcal{G}$; (3) complete theoretical proofs of orthogonality and near-oracle efficiency.
- **vs. DR methods in static causal inference**: DoubleML (Chernozhukov et al., 2018) and the DR-learner (Kennedy, 2020) serve as the direct theoretical foundation; DRQQ extends these orthogonal learning concepts from static/short-horizon settings to infinite-horizon MDPs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Systematically introducing semiparametric efficiency theory into Q-function estimation for MDPs is a significant theoretical contribution, though the core tools (EIF, orthogonal loss, cross-fitting) originate from the established causal inference literature.
- **Experimental Thoroughness**: ⭐⭐⭐ — The experimental design effectively validates theoretical predictions (varying data size, horizon, and overlap), but is limited to two small-scale discrete environments with no validation in continuous or high-dimensional settings.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The theoretical derivations are rigorous, and the narrative arc from plug-in diagnosis to EIF derivation to orthogonal loss construction is coherent and clear, making complex semiparametric statistical theory highly accessible.
- **Value**: ⭐⭐⭐⭐ — Provides a fundamentally new theoretical perspective and a rigorously grounded method for offline Q-function estimation, with significant implications for personalized medicine and reliable RL.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MTBBench: A Multimodal Sequential Clinical Decision-Making Benchmark in Oncology](../../NeurIPS2025/medical_imaging/mtbbench_a_multimodal_sequential_clinical_decision-making_benchmark_in_oncology.md)
- [\[AAAI 2026\] Bayesian Meta-Analyses Could Be More: A Case Study in Trial of Labor After a Cesarean-section Outcomes and Complications](../../AAAI2026/medical_imaging/bayesian_meta-analyses_could_be_more_a_case_study_in_trial_of_labor_after_a_cesa.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](../../AAAI2026/medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](dual_distillation_for_few-shot_anomaly_detection.md)
- [\[ICLR 2026\] DISCO: Densely-overlapping Cell Instance Segmentation via Adjacency-aware Collaborative Coloring](disco_densely-overlapping_cell_instance_segmentation_via_adjacency-aware_collabo.md)

<!-- RELATED:END -->
