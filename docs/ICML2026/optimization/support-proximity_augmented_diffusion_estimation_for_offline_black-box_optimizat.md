---
title: >-
  [Paper Note] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization
description: >-
  [ICML 2026][Optimization][Offline BBO] SPADE replaces the traditional regression surrogate with a conditional diffusion model to model $p(y\mid\boldsymbol{x})$…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Offline BBO"
  - "Conditional Diffusion Surrogate"
  - "kNN Support Regularization"
  - "LCB Acquisition Function"
date: 2026-05-08
content_hash: a9265b61c4d7031f
---

# Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.11246](https://arxiv.org/abs/2605.11246)  
**Code**: https://github.com/HarryYoung2018/spade (available)  
**Area**: Diffusion Models / Offline Black-Box Optimization  
**Keywords**: Offline BBO, Conditional Diffusion Surrogate, kNN Support Regularization, LCB Acquisition Function

## TL;DR
SPADE replaces the traditional regression surrogate with a conditional diffusion model to model $p(y\mid\boldsymbol{x})$, and implicitly injects data priors into the surrogate via "mean/rank calibration" and "kNN support regularization (mean shrinkage + variance inflation)", enabling offline black-box optimization to stably achieve SOTA on Design-Bench and LLM data mixture tasks.

## Background & Motivation

**Background**: Offline black-box optimization (offline BBO) can only use a static dataset $\mathcal{D}=\{(\boldsymbol{x}_i,y_i)\}$ to find the optimal design, without further access to the true oracle. Mainstream approaches fall into two categories: inverse methods directly learn $p(\boldsymbol{x}\mid y)$ to sample designs conditioned on high scores; forward methods learn a regression surrogate $f_\theta(\boldsymbol{x})$ and then perform gradient ascent or acquisition function search.

**Limitations of Prior Work**: Inverse methods are essentially ill-posed one-to-many mappings, making training difficult and prone to mode collapse. In forward methods, deterministic MLPs cannot provide epistemic uncertainty, leading to "hole punching"—the optimizer exploits regions where the surrogate overestimates, resulting in unreliable performance in the real environment.

**Key Challenge**: A good forward surrogate must simultaneously possess three properties—distributional expressiveness (providing both mean and variance), global accuracy (accurate means and correct ranking), and inherent conservativeness in OOD regions (automatically downweighting values far from the data manifold). Existing methods typically satisfy only one of these.

**Goal**: 1) Enable diffusion models to serve as forward surrogates, capturing the full distribution $p(y\mid\boldsymbol{x})$; 2) Calibrate the training objective for global mean and pairwise ranking; 3) Inject prior information into the surrogate without separately training a generative model $p(\boldsymbol{x})$.

**Key Insight**: Decompose Bayes' formula $p(\boldsymbol{x}\mid y)\propto p(y\mid\boldsymbol{x})\,p(\boldsymbol{x})$—model the forward part with conditional diffusion, estimate the prior part using kNN distance as a nonparametric density estimator, and theoretically prove that this geometric regularization is "first-order equivalent" to adding $\log p(\boldsymbol{x})$ in the acquisition function.

**Core Idea**: Use conditional diffusion as the forward surrogate + calibration loss to anchor global statistics + kNN distance-driven mean shrinkage/variance inflation to inject support priors, finally performing risk-aware search with LCB + evolutionary algorithms.

## Method

### Overall Architecture
SPADE consists of two stages. **Surrogate Training Stage**: On $\mathcal{D}$, simultaneously optimize three losses—basic diffusion denoising loss $\mathcal{L}_{\text{diff}}$, calibration loss $\mathcal{L}_{\text{calib}}$ (mean matching + pairwise ranking), and support-proximity loss $\mathcal{L}_{\text{prox}}$ (kNN distance-driven mean-shrink + variance-floor). **Optimization Stage**: Use an evolutionary algorithm to evolve a candidate population from high-scoring seeds; for each candidate $\boldsymbol{x}$, estimate $\hat\mu_\theta(\boldsymbol{x})$ and $\hat\sigma_\theta(\boldsymbol{x})$ via $M$-step MC sampling, and select the optimum according to LCB $=\hat\mu-\beta\hat\sigma$.

### Key Designs

1. **Conditional Diffusion Forward Surrogate**:

    - **Function**: Replace the traditional deterministic regression surrogate with DDPM, so $p_\theta(y\mid\boldsymbol{x})$ is a full predictive distribution rather than a point estimate.
    - **Mechanism**: Add noise to $y_0$ using variance schedule $\{\beta_t\}$ to obtain $q(y_t\mid y_0)=\mathcal{N}(\sqrt{\bar\alpha_t}y_0,(1-\bar\alpha_t)\mathbf{I})$, train a noise prediction network $\epsilon_\theta(y_t,t,\boldsymbol{x})$ conditioned on $\boldsymbol{x}$, with loss $\mathcal{L}_{\text{diff}}=\mathbb{E}\|\epsilon-\epsilon_\theta(y_t,t,\boldsymbol{x})\|_2^2$. At inference, perform $M$-step MC sampling to obtain $\{y^{(m)}\}$, from which mean and variance are estimated.
    - **Design Motivation**: MLP regression only provides point estimates; without $\sigma$, risk-aware acquisition functions like LCB/EI cannot be used. Diffusion models naturally capture multimodality and heteroscedasticity, and are more scalable than ensembles/BNNs.

2. **Calibrated Diffusion Estimation (Calibration Loss)**:

    - **Function**: Ensure the surrogate matches the real landscape in terms of "global mean" and "pairwise ranking", compensating for the local focus of standard denoising loss.
    - **Mechanism**: For each mini-batch, estimate $\hat\mu_\theta(\boldsymbol{x})\approx\frac{1}{M}\sum_m y^{(m)}$ via $M$-step MC sampling, then sum two terms: first-moment matching $(\hat\mu_\theta(\boldsymbol{x})-y)^2$ + pairwise rank consistency $\log(1+\exp\{-s[\hat\mu_\theta(\boldsymbol{x}_i)-\hat\mu_\theta(\boldsymbol{x}_j)]\})$ (computed only for ordered pairs with $y_i>y_j$, temperature $s=1$).
    - **Design Motivation**: BBO ultimately relies on the ranking of means, not the shape of the distribution; optimizing only $\mathcal{L}_{\text{diff}}$ does not guarantee correct ranking. The rank loss explicitly encodes "which is better" into the training objective, effectively propagating the BBO utility signal back to the diffusion network.

3. **Support-Proximity Regularization**:

    - **Function**: Without separately training a generative model $p(\boldsymbol{x})$, ensure the surrogate automatically lowers the mean and raises the variance in OOD regions far from the data manifold, making LCB naturally unfavorable in OOD.
    - **Mechanism**: Use the $k$-th nearest neighbor distance $R_k(\boldsymbol{x})$ as a density proxy, define $d(\boldsymbol{x})=\log R_k(\boldsymbol{x})$, so $-\log\hat p_{\text{knn}}(\boldsymbol{x})\propto d(\boldsymbol{x})$. The loss includes two hinge terms: mean-shrink $\max(0,\hat\mu_\theta-\mu_{\text{NN}}-\tau(d))$ pulls the mean towards the neighbors and more aggressively with distance, variance-floor $\max(0,\sigma_{\min}(d)-\hat\sigma_\theta)$ enforces a distance-dependent lower bound on variance, where $\tau(d)=ad$, $\sigma_{\min}(d)=a_0+a_1 d$, with default $a=0.02,a_0=0.02,a_1=0.005$ universal across tasks. The paper proves: for LCB-type acquisition functions (monotonic in $\mu$, decreasing in $\sigma$), $\widetilde{\mathcal{A}}(\boldsymbol{x})=\mathcal{A}(\mu,\sigma)+\kappa(\boldsymbol{x})\log\hat p_{\text{knn}}(\boldsymbol{x})+o(\cdot)$, i.e., first-order equivalent to adding a log-prior to the utility.
    - **Design Motivation**: Training an independent $p(\boldsymbol{x})$ generator is expensive and hard to tune; kNN is nonparametric and robust to high dimensions and non-uniform distributions; the hinge formulation ensures gradients are only applied when the "should be conservative" constraint is violated, not interfering with in-manifold fitting.

### Loss & Training
The total loss is $\mathcal{L}(\theta)=\mathcal{L}_{\text{diff}}+\lambda_1\mathcal{L}_{\text{calib}}+\lambda_2\mathcal{L}_{\text{prox}}$. At inference, LCB $\hat\mu_\theta(\boldsymbol{x})-\beta\hat\sigma_\theta(\boldsymbol{x})$ is used as the acquisition function. An evolutionary algorithm (EA) initializes the population from high-scoring seeds in $\mathcal{D}$; in each generation, each candidate's LCB is evaluated, followed by selection/mutation/crossover, and finally outputs $\arg\max_{\boldsymbol{x}\in\mathcal{P}}\text{LCB}(\boldsymbol{x})$.

## Key Experimental Results

### Main Results
On six tasks from Design-Bench (SuperConductor, Ant, D'Kitty, TF8, TF10) and LLM Data Mixture (LLM-DM), the 100th-percentile normalized score among $K=128$ candidates is reported (mean ± SE, 8 seeds).

| Task | $\mathcal{D}(\text{best})$ | Prev. SOTA Range | SPADE | Notes |
|------|---------------------------|------------------|-------|-------|
| SuperConductor | 0.399 | Baselines 0.40~0.55 | Among the best | Calibration improves surrogate ranking |
| Ant Morphology | 0.565 | 0.60~0.90 | Among the best | High-dimensional continuous control |
| D'Kitty | 0.884 | ~0.90 | Among the best | High OOD risk |
| LLM-DM | 1.000 | Near upper bound | On par/more stable than baselines | LLM data mixture optimization |
| TF8 / TF10 | 0.439 / 0.511 | Discrete design tasks | Among the best | Also works in discrete spaces |

SPADE ranks first in both mean rank and median rank, and is the only method consistently top across all six tasks.

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|---------------|---------------|-------------|
| Full SPADE | SOTA or tied on all 6 tasks | All three modules are indispensable |
| w/o $\mathcal{L}_{\text{calib}}$ | Ranking disorder, EA selects wrong candidates | Lacks global calibration |
| w/o $\mathcal{L}_{\text{prox}}$ | Classic OOD reward hacking, EA overestimates values | No prior constraint |
| w/o diffusion (plain MLP regression) | No $\sigma$, LCB degenerates to mean greedy | Loses risk-awareness |
| Replace kNN with KDE | Collapses in high dimensions, results worsen | kNN's adaptive bandwidth is more robust |
| Replace LCB with mean greedy | OOD risk amplified | Confirms LCB is the best regularizer |

### Key Findings
- $\mathcal{L}_{\text{prox}}$ contributes most to stability: removing it leads to reward hacking in most tasks, with scores 10–30% lower than the full model; it essentially uses geometry as a generative prior.
- The rank term in $\mathcal{L}_{\text{calib}}$ is more critical than the moment term, since BBO ultimately relies on relative ranking rather than absolute values.
- The number of diffusion steps $T$ is not sensitive (short runs suffice), but the MC sample size $M$ affects variance estimation accuracy; too small $M$ increases LCB noise.
- The hyperparameters $a, a_0, a_1$ are universal across tasks, requiring no per-task tuning, demonstrating the robustness of the kNN geometric prior.

## Highlights & Insights
- "Using diffusion as a forward surrogate" is a counterintuitive yet reasonable design: diffusion is usually used for inverse $p(\boldsymbol{x}\mid y)$, but here it is applied to $p(y\mid\boldsymbol{x})$; the key is that $y$ is a one-dimensional scalar, making diffusion lightweight yet still able to provide $\sigma$.
- Equating "geometric constraints" and "Bayesian priors" via a first-order equivalence theorem is a transferable proof strategy—it shows that if a hinge regularizer $\tau(d)$ grows linearly with $-\log p(\boldsymbol{x})$, it is equivalent to adding a log-prior in the acquisition function. This can be applied to other tasks (e.g., imitation learning, offline RL).
- Mean-shrink + variance-floor are natural partners: the former reduces $\mu$, the latter increases $\sigma$, and together they doubly discount LCB in OOD regions, providing more stability than either alone.

## Limitations & Future Work
- The authors acknowledge that Proposition 3.1 is only a "motivation" rather than a full algorithmic guarantee; actual behavior is still affected by EA, $\beta$, and MC noise.
- kNN may still degrade in design spaces with hundreds of dimensions (distance homogenization); extremely high-dimensional scenarios such as proteins require representation learning or manifold-aware distances.
- $\mathcal{L}_{\text{calib}}$ requires $M$ short-run MC samples per step, making training several times more expensive than pure regression surrogates—this is the main engineering overhead.
- The optimal range of LCB coefficient $\beta$ across tasks is not discussed; practical applications still require tuning $\beta$.

## Related Work & Insights
- **vs DDOM / inverse diffusion methods**: These model $p(\boldsymbol{x}\mid y)$ and suffer from ill-posed one-to-many mapping; SPADE takes the forward $p(y\mid\boldsymbol{x})$ route with explicit prior injection, avoiding the training difficulties of inverse methods.
- **vs COMs / ROMA and other conservative regression baselines**: These add adversarial or penalty terms to MLPs for conservativeness, but point estimates make LCB unusable; SPADE uses diffusion for distributional outputs and kNN geometry, with more explicit prior injection and a Bayesian interpretation.
- **vs GP / BNN and other probabilistic surrogates**: GPs do not scale to high dimensions; BNNs are expensive to train and may not be calibrated; diffusion + short-run MC strikes a good balance between expressiveness and scalability.

## Rating
- Novelty: ⭐⭐⭐⭐ Moves diffusion from inverse to forward perspective, complemented by a Bayes equivalence theorem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers the full Design-Bench suite + LLM-DM, complete ablations, and universal hyperparameters.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, pipeline diagrams clearly illustrate the two-stage training/optimization.
- Value: ⭐⭐⭐⭐ Provides a stable SOTA surrogate paradigm for offline BBO; the kNN-as-prior idea is transferable to other conservative offline settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] PPO-MI: Efficient Black-Box Model Inversion via Proximal Policy Optimization](../../ICML2025/image_generation/ppo-mi_efficient_black-box_model_inversion_via_proximal_policy_optimization.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](../../ICLR2026/image_generation/pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [\[ACL 2026\] MASH: Evading Black-Box AI-Generated Text Detectors via Style Humanization](../../ACL2026/image_generation/mash_evading_black-box_ai-generated_text_detectors_via_style_humanization.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[ICML 2026\] Threshold-Guided Optimization for Visual Generative Models](threshold-guided_optimization_for_visual_generative_models.md)

</div>

<!-- RELATED:END -->
