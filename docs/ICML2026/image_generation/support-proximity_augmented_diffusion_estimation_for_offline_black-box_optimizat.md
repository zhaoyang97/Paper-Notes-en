---
title: >-
  [Paper Note] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization
description: >-
  [ICML 2026][Image Generation][Offline BBO] SPADE replaces traditional regression surrogates with a conditional diffusion model to model $p(y\mid\boldsymbol{x})$. By incorporating "mean/rank calibration" and "kNN support…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Offline BBO"
  - "Conditional Diffusion Surrogate"
  - "kNN Support Regularization"
  - "LCB Acquisition Function"
date: 2026-05-08
content_hash: 9c9c50e4a531f069
---

# Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization

**Conference**: ICML 2026  
**arXiv**: [2605.11246](https://arxiv.org/abs/2605.11246)  
**Code**: https://github.com/HarryYoung2018/spade (Available)  
**Area**: Diffusion Models / Offline Black-Box Optimization  
**Keywords**: Offline BBO, Conditional Diffusion Surrogate, kNN Support Regularization, LCB Acquisition Function

## TL;DR
SPADE replaces traditional regression surrogates with a conditional diffusion model to model $p(y\mid\boldsymbol{x})$. By incorporating "mean/rank calibration" and "kNN support regularization (mean-shrink + variance-floor)", it implicitly injects data priors into the surrogate, enabling offline black-box optimization to consistently reach SOTA performance on Design-Bench and LLM data mixture tasks.

## Background & Motivation

**Background**: Offline black-box optimization (offline BBO) aims to find an optimal design using only a static dataset $\mathcal{D}=\{(\boldsymbol{x}_i,y_i)\}$ without querying the true oracle. Mainstream approaches are divided into two categories: inverse methods, which directly learn $p(\boldsymbol{x}\mid y)$ to sample designs conditioned on high scores; and forward methods, which learn a regression surrogate $f_\theta(\boldsymbol{x})$ followed by gradient ascent or acquisition function search.

**Limitations of Prior Work**: Inverse methods are inherently ill-posed one-to-many mappings, making them difficult to train and prone to mode collapse. In forward methods, deterministic MLPs fail to provide epistemic uncertainty, leading the optimizer to "drill holes" into regions where the surrogate overestimates, resulting in unreliable performance in real environments.

**Key Challenge**: An effective forward surrogate requires three properties: distributional expressiveness (providing both mean and variance), global accuracy (precise mean and ranking), and natural conservatism toward OOD regions (automatically lowering evaluations far from the data manifold). Existing methods typically satisfy only one of these.

**Goal**: 1) Enable diffusion models to serve as forward surrogates capturing the full distribution of $p(y\mid\boldsymbol{x})$; 2) Calibrate global means and pairwise rankings through training objectives; 3) Inject prior information into the surrogate without training an additional generative model for $p(\boldsymbol{x})$.

**Key Insight**: By decomposing Bayes' rule $p(\boldsymbol{x}\mid y)\propto p(y\mid\boldsymbol{x})\,p(\boldsymbol{x})$, the forward part is modeled via conditional diffusion while the prior part is estimated through kNN-based non-parametric density estimation. This geometric regularization is theoretically proven to be "first-order equivalent" to adding $\log p(\boldsymbol{x})$ to the acquisition function.

**Core Idea**: Utilize conditional diffusion as a forward surrogate + calibration loss to anchor global statistics + kNN-driven mean-shrink/variance-floor to inject support priors, finally performing risk-aware search using LCB and evolutionary algorithms.

## Method

### Overall Architecture
SPADE operates in two phases. **Surrogate Training Phase**: Three losses are optimized simultaneously on $\mathcal{D}$—the basic diffusion denoising loss $\mathcal{L}_{\text{diff}}$, the calibration loss $\mathcal{L}_{\text{calib}}$ (mean matching + pairwise ranking), and the support-proximity loss $\mathcal{L}_{\text{prox}}$ (kNN distance-driven mean-shrink + variance-floor). **Optimization Phase**: An evolutionary algorithm evolves a candidate population starting from high-score seeds. For each candidate $\boldsymbol{x}$, $\hat\mu_\theta(\boldsymbol{x})$ and $\hat\sigma_\theta(\boldsymbol{x})$ are estimated via $M$ MC samples, and the best is selected according to $\text{LCB} = \hat\mu - \beta\hat\sigma$.

### Key Designs

1.  **Conditional Diffusion Forward Surrogate**:
    - **Function**: Replaces traditional deterministic regression surrogates with DDPM to make $p_\theta(y\mid\boldsymbol{x})$ a full predictive distribution rather than a point estimate.
    - **Mechanism**: A noise schedule $\{\beta_t\}$ adds noise to $y_0$ to obtain $q(y_t\mid y_0)=\mathcal{N}(\sqrt{\bar\alpha_t}y_0,(1-\bar\alpha_t)\mathbf{I})$. A noise prediction network $\epsilon_\theta(y_t,t,\boldsymbol{x})$ conditioned on $\boldsymbol{x}$ is trained with $\mathcal{L}_{\text{diff}}=\mathbb{E}\|\epsilon-\epsilon_\theta(y_t,t,\boldsymbol{x})\|_2^2$. During inference, $M$ MC samples $\{y^{(m)}\}$ are generated via short-run sampling to estimate the predictive mean and variance.
    - **Design Motivation**: MLP regression only provides point estimates; without $\sigma$, risk-aware acquisition functions like LCB/EI cannot be utilized. Diffusion models naturally capture multimodality and heteroskedasticity, scaling better than ensembles or BNNs.

2.  **Calibrated Diffusion Estimation**:
    - **Function**: Aligns the surrogate with the true landscape in terms of "global mean" and "pairwise ranking," compensating for the denoising loss's focus on local distributions.
    - **Mechanism**: $\hat\mu_\theta(\boldsymbol{x})\approx\frac{1}{M}\sum_m y^{(m)}$ is estimated from mini-batches using $M$ MC samples, followed by two terms: first-order moment matching $(\hat\mu_\theta(\boldsymbol{x})-y)^2$ and pairwise rank consistency $\log(1+\exp\{-s[\hat\mu_\theta(\boldsymbol{x}_i)-\hat\mu_\theta(\boldsymbol{x}_j)]\})$ (calculated only for pairs where $y_i>y_j$, with temperature $s=1$).
    - **Design Motivation**: BBO relies on mean ranking rather than distribution shape. Since $\mathcal{L}_{\text{diff}}$ does not guarantee rank consistency, the rank loss explicitly propagates the "who is better" utility signal back into the diffusion network.

3.  **Support-Proximity Regularization**:
    - **Function**: Automatically lowers the mean and increases the variance in OOD regions far from the data manifold without training a separate $p(\boldsymbol{x})$, making LCB naturally unfavorable in OOD areas.
    - **Mechanism**: The $k$-th nearest neighbor distance $R_k(\boldsymbol{x})$ serves as a density proxy, defining $d(\boldsymbol{x})=\log R_k(\boldsymbol{x})$ such that $-\log\hat p_{\text{knn}}(\boldsymbol{x})\propto d(\boldsymbol{x})$. The loss includes two hinge terms: mean-shrink $\max(0,\hat\mu_\theta-\mu_{\text{NN}}-\tau(d))$ pulls the mean toward the neighbors' mean, and variance-floor $\max(0,\sigma_{\min}(d)-\hat\sigma_\theta)$ pushes the variance to a distance-dependent lower bound, where $\tau(d)=ad$ and $\sigma_{\min}(d)=a_0+a_1 d$. Theoretical proof shows that under "$\mu$ increasing, $\sigma$ decreasing" acquisition functions like LCB, $\widetilde{\mathcal{A}}(\boldsymbol{x})=\mathcal{A}(\mu,\sigma)+\kappa(\boldsymbol{x})\log\hat p_{\text{knn}}(\boldsymbol{x})+o(\cdot)$, which is first-order equivalent to adding a $\log p(\boldsymbol{x})$ prior to the utility.
    - **Design Motivation**: Training independent $p(\boldsymbol{x})$ generators is expensive; kNN is non-parametric and robust to high-dimensional distributions. The hinge formulation ensures gradients are applied only when "conservatism" constraints are violated.

### Loss & Training
The total loss is $\mathcal{L}(\theta)=\mathcal{L}_{\text{diff}}+\lambda_1\mathcal{L}_{\text{calib}}+\lambda_2\mathcal{L}_{\text{prox}}$. Inference utilizes LCB $\hat\mu_\theta(\boldsymbol{x})-\beta\hat\sigma_\theta(\boldsymbol{x})$ as the acquisition function. An evolutionary algorithm (EA) initializes the population from high-score seeds in $\mathcal{D}$, performing evaluation, selection, mutation, and crossover to output $\arg\max_{\boldsymbol{x}\in\mathcal{P}}\text{LCB}(\boldsymbol{x})$.

## Key Experimental Results

### Main Results
On 6 tasks comprising Design-Bench (SuperConductor, Ant, D'Kitty, TF8, TF10) and LLM Data Mixture (LLM-DM), the 100th-percentile normalized score among $K=128$ candidates is reported (mean ± SE, 8 seeds).

| Task | $\mathcal{D}(\text{best})$ | Prev. SOTA Range | SPADE | Remarks |
|------|---------------------------|---------------|-------|------|
| SuperConductor | 0.399 | 0.40~0.55 | Best | Calibration improves rank accuracy |
| Ant Morphology | 0.565 | 0.60~0.90 | Best | High-dim continuous control |
| D'Kitty | 0.884 | ~0.90 | Best | High OOD risk |
| LLM-DM | 1.000 | Near upper bound | Competitive/Stabler | LLM data optimization |
| TF8 / TF10 | 0.439 / 0.511 | Discrete tasks | Best | Effective in discrete spaces |

SPADE ranks first in both mean rank and median rank, being the only method to remain stable at the top across all 6 tasks.

### Ablation Study

| Configuration | Key Findings | Description |
|------|---------|------|
| Full SPADE | SOTA or tied on all 6 tasks | All three modules are essential |
| w/o $\mathcal{L}_{\text{calib}}$ | Ranking errors, EA selects wrong candidates | Missing global calibration |
| w/o $\mathcal{L}_{\text{prox}}$ | Classic OOD reward hacking, EA estimates explode | Lacks prior constraints |
| w/o Diffusion (MLP) | No $\sigma$, LCB degrades to greedy mean | Loss of risk-awareness |
| kNN replaced by KDE | Performance drops significantly in high dims | kNN adaptive bandwidth is stabler |
| LCB replaced by Greedy | OOD risk is amplified | Confirms LCB is the optimal partner |

### Key Findings
- **$\mathcal{L}_{\text{prox}}$ is the largest contributor to stability**: Removing it leads to reward hacking in most tasks, reducing scores by 10-30%. It effectively replaces generative priors with geometric ones.
- **$\mathcal{L}_{\text{calib}}$'s rank term is more critical than the moment term**, as BBO utilizes relative ranking rather than absolute values.
- **Diffusion steps $T$ are less sensitive** (short runs suffice), but the number of MC samples $M$ impacts variance estimation precision.
- **Hyperparameters ($a, a_0, a_1$) are universal across tasks**, demonstrating the robustness of the kNN geometric prior.

## Highlights & Insights
- Using diffusion as a forward surrogate is an counter-intuitive yet logical design: while diffusion is typically used for inverse $p(\boldsymbol{x}\mid y)$, applying it to $p(y\mid\boldsymbol{x})$ is lightweight since $y$ is a scalar, while still providing distributional $\sigma$.
- Equating "geometric constraints" with "Bayesian priors" via first-order equivalence is a powerful theoretical bridge. This suggests that any hinge regularization $\tau(d)$ growing linearly with $-\log p(\boldsymbol{x})$ acts as a log-prior on the acquisition function.
- The mean-shrink and variance-floor pair works synergistically: one lowers $\mu$ and the other raises $\sigma$, causing LCB to "double discount" in OOD regions, providing superior stability.

## Limitations & Future Work
- Proposition 3.1 serves as motivation rather than a complete algorithmic guarantee; actual behavior is influenced by EA, $\beta$, and MC noise.
- kNN may degrade in design spaces exceeding hundreds of dimensions (distance homogenization); extremely high-dimensional scenarios like proteins might require representation learning.
- $\mathcal{L}_{\text{calib}}$ requires $M$ short-run MC samples per step, increasing training time compared to simple regression.
- The optimal range of the LCB coefficient $\beta$ across different tasks was not discussed and may require tuning.

## Related Work & Insights
- **vs DDOM / Inverse Diffusion**: Inverse methods suffer from ill-posed one-to-many mapping. SPADE follows the forward $p(y\mid\boldsymbol{x})$ path with explicit prior injection, avoiding inverse training difficulties.
- **vs COMs / ROMA (Conservative Regression)**: These add adversarial or penalty terms to MLPs for conservatism, but deterministic outputs preclude LCB. SPADE provides native distributions via diffusion + kNN geometry for explicit prior injection.
- **vs GP / BNN**: GPs do not scale to high dimensions; BNN training is expensive and often poorly calibrated. Diffusion + short-run MC achieves a strong balance between expressiveness and scalability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Refreshing perspective on diffusion for forward modeling with Bayesian equivalence proofs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Full Design-Bench coverage + LLM-DM, complete ablations, and universal hyperparameters.
- **Writing Quality**: ⭐⭐⭐⭐ Clear derivations and a well-structured pipeline diagram.
- **Value**: ⭐⭐⭐⭐ Provides a stable SOTA surrogate paradigm for offline BBO; the kNN-as-prior concept is transferable to other conservative offline settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](../../ICLR2026/image_generation/pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[NeurIPS 2025\] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models](../../NeurIPS2025/image_generation/transferable_black-box_one-shot_forging_of_watermarks_via_image_preference_model.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](discoformer_plug-in_density_and_score_estimation_with_transformers.md)

</div>

<!-- RELATED:END -->
