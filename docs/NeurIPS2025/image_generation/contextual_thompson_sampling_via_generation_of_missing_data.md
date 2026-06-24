---
title: >-
  [Paper Note] Contextual Thompson Sampling via Generation of Missing Data
description: >-
  [NeurIPS 2025][Image Generation][Thompson sampling] This paper proposes Generative Thompson Sampling (TS-Gen), which reframes uncertainty in contextual bandits as missing data rather than unknown parameters. A generative model autoregressively imputes missing outcomes to implement Thompson sampling, and a regret bound directly tied to offline prediction loss is established.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Thompson sampling"
  - "contextual bandit"
  - "generative models"
  - "missing data imputation"
  - "meta-learning"
  - "regret bounds"
date: 2026-05-08
content_hash: 79aa702046c0e6bd
---

# Contextual Thompson Sampling via Generation of Missing Data

**Conference**: NeurIPS 2025
**arXiv**: [2502.07064](https://arxiv.org/abs/2502.07064)  
**Code**: To be confirmed  
**Area**: Generative Models / Decision Systems / Contextual Bandits
**Keywords**: Thompson sampling, contextual bandit, generative models, missing data imputation, meta-learning, regret bounds

## TL;DR
This paper proposes Generative Thompson Sampling (TS-Gen), which reframes uncertainty in contextual bandits as missing data rather than unknown parameters. A generative model autoregressively imputes missing outcomes to implement Thompson sampling, and a regret bound directly tied to offline prediction loss is established.

## Background & Motivation

**Background**: Classical Thompson sampling (TS) attributes environmental uncertainty to unknown latent parameters $\theta$, relying on three operations: specifying a prior, sampling from the posterior, and updating the posterior. All three are notoriously difficult in neural network settings (intractable posteriors, unintuitive priors, and costly online updates).

**Limitations of Prior Work**: (a) Bayesian uncertainty quantification methods for neural networks (variational inference, MC dropout, ensembles, etc.) are limited in effectiveness and computationally expensive; (b) existing TS theory is largely restricted to parametric models (linear/logistic regression) and does not naturally extend to complex industrial settings using foundation models; (c) there is no systematic framework for leveraging prior information from historical tasks in the meta-learning setting.

**Key Challenge**: How can Thompson sampling be made compatible with modern deep learning (sequence models, foundation models) while retaining provable theoretical guarantees?

**Key Insight**: Rather than treating uncertainty as arising from unknown parameters, this work treats it as missing observable data (counterfactual outcomes). If the outcomes of all actions were known, there would be no uncertainty—one could directly fit an optimal policy on the complete data. At each decision point, a generative model imputes the missing outcomes, a policy is fitted on the completed dataset, and an action is selected according to that policy.

## Method

### Core Framework: Missing Data Perspective

A task $\tau$ is defined as a potential outcomes table containing task information $Z_\tau$, a context sequence $X_{1:T}$, and potential outcomes $\{Y_t^{(a)}\}$ for all actions. Only the outcome of the selected action is observed; the rest constitute "missing data."

**Algorithm 1: Generative Thompson Sampling**

At each decision step $t$:
1. Observe context $X_t$
2. Sample a completed task from the conditional distribution using generative model $p$: $\hat{\tau}_t \sim p(\tau \in \cdot \mid \mathcal{H}_t)$
3. Fit the optimal policy $\pi^*(\cdot; \hat{\tau}_t)$ on the completed dataset $\hat{\tau}_t$
4. Select action $A_t \leftarrow \pi^*(X_t; \hat{\tau}_t)$
5. Observe outcome $Y_t = Y_t^{(A_t)}$ and update the history

### Autoregressive Posterior Sampling (Algorithm 3)

For each action $a$, observed outcomes are placed first and missing outcomes last; a sequence model $p_\theta$ then autoregressively generates the missing values:

$$\hat{Y}_i^{(a)} \sim p_\theta(\cdot \mid Z, \{\hat{X}_j, \hat{Y}_j^{(a)}\}_{j \prec_a i}, \hat{X}_i)$$

This ordering ensures the model always conditions on all observed outcomes before generating missing values, thereby implementing correct posterior sampling.

### Offline Training (Algorithm 2)

A sequence model is trained on historical task data $\mathcal{D}^{\text{offline}}$ using a standard autoregressive loss:

$$\ell(p_\theta) = -\mathbb{E}\left[\log p_\theta(X_{1:T}, \{Y_{1:t-1}^{(a)}\}_{a \in \mathcal{A}_\tau} \mid Z_\tau)\right]$$

In practice, simplifying assumptions (contexts independent of historical outcomes; outcomes independent across actions) allow optimization via mini-batch SGD.

### Theoretical Guarantees

**Theorem 1 (Perfect Model Regret Bound)**: Using the true distribution $p^*$,

$$\Delta(\mathbb{A}_{\text{TS-Gen}}(p^*)) \leq \sqrt{\frac{|\mathcal{A}_\tau|}{2T} \cdot H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau)}$$

**Theorem 2 (Approximate Model Regret Bound)**: Using an approximate model $p_\theta$,

$$\Delta(\mathbb{A}_{\text{TS-Gen}}(p_\theta)) \leq \underbrace{\sqrt{\frac{|\mathcal{A}_\tau|}{2T} \cdot H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau)}}_{\text{perfect TS regret}} + \underbrace{\sqrt{2\{\ell(p_\theta) - \ell(p^*)\}}}_{\text{model quality penalty}}$$

The quality of the generative model affects the regret bound solely through the offline prediction loss—independent of model architecture, enabling the use of any sequence model.

**Proposition 2 (Complexity Bound)**: The entropy is upper-bounded via VC dimension: $H(\boldsymbol{\pi}^*(X_{1:T}) \mid Z_\tau) = O(d \cdot \log(T \cdot |\mathcal{A}_\tau|))$, applicable directly to infinite policy classes.

## Key Experimental Results

### Setup
$T=500$, $|\mathcal{A}|=10$, binary outcomes $Y \in \{0,1\}$, prior information $Z^{(a)} \in \mathbb{R}^2$.

| Method | Cumulative Regret (Synthetic) | Cumulative Regret (Semi-synthetic / News Recommendation) |
|---|---|---|
| **TS-Gen** | **Lowest** | **Lowest** |
| Greedy | High | High |
| ε-Greedy | Medium-High | Medium-High |
| TS-Neural-Linear (uninformative) | Medium | Medium |
| TS-Neural-Linear (fitted prior) | Medium | Medium |
| LinUCB | Medium-High | Medium-High |
| TS-Linear | Medium-High | High |

- TS-Gen outperforms all baselines in both settings.
- Using the same $p_\theta$ model with alternative decision strategies (Greedy, ε-Greedy, TS-Neural-Linear) yields higher regret, validating the advantage of the autoregressive generation approach for uncertainty quantification.
- Lower offline prediction loss corresponds to lower TS-Gen regret, consistent with Theorem 2.
- Computational cost: in the semi-synthetic experiment, each decision requires 4.2s for generation and 2.2s for policy fitting (CPU).

## Highlights & Insights

- **The missing data perspective redefines the primitive operations of TS**: the paradigm shifts from "specify prior → sample posterior → update posterior" to "minimize prediction loss → autoregressively generate → fit policy," naturally compatible with modern deep learning.
- **Modular design**: the generative model $p_\theta$ and policy class $\Pi$ are fully decoupled—the generative model architecture and policy fitting method can be independently replaced (even with fairness constraints added).
- **Theoretical contribution**: this is the first work to derive regret bounds for contextual TS over infinite policy classes using information-theoretic methods, without discretizing the parameter space.
- **Meta-learning as in-context learning (ICL)**: after training, $p_\theta$ requires no online gradient updates and "learns in context" by conditioning on historical observations—naturally aligned with the ICL capabilities of large language models.

## Limitations & Future Work

- **Validation limited to two experimental settings**: synthetic and semi-synthetic; large-scale real-world evaluation is absent.
- **Training data requirements**: complete task datasets are needed to train $p_\theta$; in practice only partial data may be available (approximated via bootstrap, without theoretical guarantees).
- **High computational cost**: each decision requires autoregressive generation of complete task data plus policy fitting, limiting real-time applicability.
- **No guidance on policy class selection**: choosing an appropriate $\Pi$ remains an open problem.
- **MDP extension unexplored**: the current framework is restricted to the bandit setting.

## Related Work & Insights

- **vs. Classical TS**: classical TS requires parametric models and posterior inference; TS-Gen replaces this with generative imputation of missing data.
- **vs. Deep Ensemble TS**: ensemble methods require online gradient updates and cannot leverage task meta-information $Z$; TS-Gen exploits $Z$ through offline pretraining.
- **vs. Imitation Learning (Decision Transformer, etc.)**: imitation learning directly predicts optimal actions, whereas TS-Gen predicts future outcomes—the latter enables uncertainty quantification.
- **vs. Wen et al. 2021**: that work is non-contextual and requires modeling latent parameters; TS-Gen extends to the contextual setting without latent parameters.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The missing data perspective for reconstructing TS is highly novel, with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Acceptable as a proof of concept, but only two experimental scenarios are considered; large-scale real-world validation is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, theory is rigorous, and problem formulation is elegant.
- Value: ⭐⭐⭐⭐ — Establishes an important theoretical bridge between generative AI and online decision-making, with significant directional implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Models Meet Contextual Bandits](diffusion_models_meet_contextual_bandits.md)
- [\[CVPR 2026\] Beyond Objects: Contextual Synthetic Data Generation for Fine-Grained Classification](../../CVPR2026/image_generation/beyond_objects_contextual_synthetic_data_generation_for_fine-grained_classificat.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)
- [\[NeurIPS 2025\] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images](diffeye_diffusion-based_continuous_eye-tracking_data_generation_conditioned_on_n.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)

</div>

<!-- RELATED:END -->
