---
title: >-
  [Paper Note] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning
description: >-
  [ICML2026][Model Compression][scaling law] This paper models training budget allocation in scaling law experiments as a multi-round resource selection problem. By combining Successive Halving with learning curve surrogates to predict future potential, it approximates the full scaling law with up to 98.7% training cost savings on synthetic and nanoGPT learning curves.
tags:
  - "ICML2026"
  - "Model Compression"
  - "scaling law"
  - "Successive Halving"
  - "learning curves"
  - "Gaussian Process"
  - "budget allocation"
date: 2026-05-08
content_hash: e1769fd0e2ac91fe
---

# Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning

**Conference**: ICML2026  
**arXiv**: [2605.17234](https://arxiv.org/abs/2605.17234)  
**Code**: No specific repository (paper states code will be released)  
**Area**: Optimization / Scaling Law / Budget Allocation  
**Keywords**: scaling law, Successive Halving, learning curves, Gaussian Process, budget allocation  

## TL;DR
This paper models training budget allocation in scaling law experiments as a multi-round resource selection problem. By combining Successive Halving with learning curve surrogates to predict future potential, it approximates the full scaling law with up to 98.7% training cost savings on synthetic and nanoGPT learning curves.

## Background & Motivation
**Background**: Scaling laws use empirical learning curves to describe the relationship between loss and factors such as compute, model size, and data size. They are essential tools for planning training budgets, model sizes, and data requirements for large models. The classic approach typically requires training many models of different sizes to observe their loss-compute frontier.

**Limitations of Prior Work**: Complete scaling law studies are extremely expensive. To obtain a reliable frontier, researchers may need to train dozens or even hundreds of models over long compute intervals, many of which ultimately do not contribute to the optimal frontier. Traditional uniform allocation distributes the budget equally across all models, wasting resources on models that are too small, plateau early, or are large but perform poorly in the short term.

**Key Challenge**: Small models show rapid loss decline in early stages, making them appear better under short budgets. Large models may not be dominant in terms of short-term loss but possess higher long-term potential. If pruning is based solely on current loss, models capable of contributing to the future scaling frontier will be eliminated prematurely; if no pruning occurs, computational costs become unbearable.

**Goal**: The authors aim to actively decide which models to continue training and which to stop under a fixed total FLOPs budget. The goal is to ensure the final set of obtained learning curves is sufficient to fit an accurate scaling law while significantly reducing costs relative to "full training of all models."

**Key Insight**: Successive Halving from hyperparameter optimization can already allocate resources among multiple configurations, but it only considers observed loss. This paper further allows the surrogate model to predict the future continuation of each learning curve, deciding which models to retain based on the "potential future minimum loss" rather than the "current loss."

**Core Idea**: Use learning curve surrogates to correct the short-sighted pruning of Successive Halving, directing the budget toward models with greater potential to contribute to the loss-compute frontier.

## Method

### Overall Architecture
This paper addresses the problem that fitting an accurate scaling law requires training many models of different scales to sufficient compute, yet most do not fall on the loss-compute frontier, resulting in wasted computation. The authors decompose data collection into several rounds of resource selection: each round allocates the same additional compute to all current candidates, trains a segment of the learning curve, and then eliminates a batch based on those curves to concentrate the budget on the remaining models until the total budget is exhausted. The entire process follows the Successive Halving (SH) framework but modifies the pruning criterion—considering the predicted loss a model could achieve if training continued, rather than just the curve's current endpoint.

Specifically, the inputs are an initial model set $\mathcal M_0$, total budget $B$, and pruning factor $\eta$. In round $r$, each retained model receives a budget $C_r=\lfloor B/(|\mathcal M_r|\lceil\log_\eta |\mathcal M_0|\rceil)\rfloor$ to form an observed learning curve $L_m(C)$. While standard SH selects Top$_k$ models for the next round based on the lowest observed loss, SH LMC / SH DE uses a surrogate to predict the loss of each curve extended to the final round's budget, selecting based on the combination of observed segments and predicted continuations. The final output consists only of real training curves (surrogate continuations are never used as observed data), which are used to fit the compute-loss scaling law.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: Model set M₀ + Total budget B + Pruning factor η<br/>Proxy objective: Approximate minimum validation loss within budget"] --> B["Round r: Allocate equal additional budget Cr<br/>to each retained model"]
    B --> C["Train to cumulative budget to obtain<br/>observed segments of learning curves L(C)"]
    C --> D{"Criterion for Top-k selection?"}
    D -->|Standard SH (Current)| E["Select based on minimum observed loss"]
    D -->|SH LMC / SH DE (Future)| F["Surrogate predicts continuation to<br/>estimate future minimum loss<br/>(LMC GP or Deep Ensemble)"]
    E --> G["Retain Top-k models, prune others"]
    F --> G
    G -->|Budget not exhausted, next round| B
    G -->|Budget exhausted| H["Output real training curve set L<br/>(Surrogate continuations excluded)"]
    H --> I["Fit scaling law (C/α)^−γ<br/>GP mean / UCB / LCB provide optimistic/pessimistic intervals"]
```

### Key Designs

**1. Converting scaling law sampling to budget-constrained proxy optimization: Bypassing the lack of a ground-truth fitting target**

There is no direct optimization target for "which set of learning curves best fits the scaling law"—the true ground-truth frontier is only known after training models to completion, which is the most expensive part. Consequently, the authors use a computable proxy: find a set of models that can achieve the minimum validation loss within the total budget $B$. The byproduct of optimizing this proxy is a set of learning curves trained to varying degrees, which can then be used to fit the scaling law. This approach ensures the objective depends only on observed loss, allowing the application of anytime resource allocation algorithms like SH without prior knowledge of the frontier.

**2. Surrogate-guided Successive Halving: Pruning by "Future Potential" rather than "Current Loss"**

This is the core mechanism of the paper. Standard SH allocates a budget $C_r$ to all retained models in each round and selects Top$_k$ based on the **lowest current observed loss**. Its flaw is that small models decline faster early on and are judged "currently superior" under short budgets, while larger models that perform better later are eliminated prematurely. This paper modifies the Top$_k$ selection step: instead of looking at the observed endpoint, a surrogate predicts the **future minimum loss** each model would reach if it completed all remaining rounds. Re-ranking based on these predictions ensures the budget flows to models with real potential to contribute to the frontier. A crucial robustness detail is that surrogate-predicted continuations $\hat{\mathcal L}$ are **only used for pruning decisions**; the collection $\mathcal L$ used for fitting the scaling law contains only real training data, making the method more credible than "direct extrapolation as data."

**3. LMC Gaussian Process: "Vindicating" large models via cross-curve correlation**

The second design replaces the pruning criterion with "predicted future minimum loss," but how is this prediction generated? This is the author's first solution. It predicts the subsequent loss trajectory of a model based on multiple early curves, thereby retaining large models that are currently non-dominant but predicted to reach lower losses in the future. Specifically, curve extrapolation is modeled as a multi-input multi-output GP. The kernel combines exponential decay, white noise, and bias sub-kernels, while a co-regionalization matrix explicitly captures correlations between different model curves. Thus, patterns like "when small models plateau" and "when large model curves surpass others" are shared across curves, providing a prior that single-curve extrapolation cannot access.

**4. Deep Ensemble surrogate and scaling law extrapolation: Parameterized curve families + Uncertainty intervals for the frontier**

Beyond GP, the authors tested a parametric approach to compare non-parametric GP and parameterized curve families for budget allocation. Deep Ensemble uses a two-layer MLP to condition the coefficients of curve functions (e.g., power law, exponential, Morgan-Mercer-Flodin) to predict curve shapes. Since noise and shapes vary across datasets, multiple curve families provide flexibility. Furthermore, scaling law fitting often requires extrapolation beyond the trained compute range. After SH LMC, the authors use the GP's mean / UCB / LCB to extend the learning curves further, reducing the AbC gap between the fitted curve and the ground truth. UCB/LCB also provide optimistic/pessimistic curve boundaries, moving beyond a simple point estimate for budget decisions.

### Loss & Training
LMC GP is optimized using L-BFGS with 20 random restarts. Deep Ensemble is trained using five randomly initialized two-layer perceptrons for 1000 iterations. By default, 20 observation points per curve are sampled to train the surrogate. Scaling law fitting uses $L^{SL}(C)=(C/\alpha)^{-\gamma}$, and the Area between Curves (AbC) is used to measure the distance between the fitted curve and the ground-truth scaling law over a specified compute interval.

## Key Experimental Results

### Main Results
On synthetic learning curves, SH LMC shows consistent improvements over standard SH, while uniform allocation (UA) is significantly worse.

| Number of models $M_0$ | Budget $B$ (petaFLOPs) | SH mean loss | SH LMC mean rel. improv. | UA mean rel. degradation | Conclusion |
|------------------------|-------------------------|--------------|---------------------------|---------------------------|------------|
| 5 | $10^2$ | 6.40±9.07 | 5.15% (max 20.30%) | -10.17% | Surrogate provides significant improvement with few models |
| 5 | $10^4$ | 3.84±2.03 | 5.47% (max 16.70%) | -7.59% | Gains persist under high budget |
| 10 | $10^3$ | 3.86±0.38 | 2.38% (max 6.11%) | -14.06% | LMC continues to improve upon strong SH baseline |
| 20 | $10^4$ | 3.18±0.09 | 1.50% (max 6.53%) | -16.40% | Gains are stable and positive as model count increases |

In real-world nanoGPT learning curve experiments, SH LMC also outperforms SH and most DE surrogates, and all strategies surpass UA.

| $M_0$ | Budget $B$ | SH mean loss | SH LMC rel. improv. | Best DE rel. improv. | UA rel. degradation |
|-------|------------|--------------|----------------------|----------------------|---------------------|
| 5 | $10^4$ | 3.17±0.06 | 2.58% | 2.32% (DE EXP) | -5.09% |
| 5 | $10^5$ | 2.97±0.03 | 2.36% | 2.40% (DE PL) | -0.74% |
| 10 | $10^5$ | 3.00±0.02 | 2.82% | 2.14% (DE MMF) | -0.81% |
| 20 | $10^4$ | 3.30±0.02 | 2.84% | 2.02% (DE PL) | -11.46% |
| 20 | $10^5$ | 3.03±0.01 | 2.24% | 1.44% (DE EXP) | -2.96% |

Regarding scaling law fitting, both SH and SH LMC obtain laws close to the ground truth under budgets far lower than the cost of full training.

| Setting | Method | AbC vs Full Data SL | Loss regret | Cost savings vs full training |
|---------|--------|---------------------|-------------|-------------------------------|
| $M_0=5,B=10^4$ | SH | 0.09±0.05 | 0.43±0.09 | 94.00% |
| $M_0=5,B=10^4$ | SH LMC | 0.11±0.07 | 0.41±0.10 | 94.00% |
| $M_0=10,B=10^4$ | SH | 0.07±0.02 | 0.56±0.07 | 97.50% |
| $M_0=10,B=10^4$ | SH LMC | 0.09±0.04 | 0.51±0.06 | 97.50% |
| $M_0=20,B=10^4$ | SH | 0.12±0.04 | 0.67±0.03 | 98.70% |
| $M_0=20,B=10^4$ | SH LMC | 0.11±0.07 | 0.59±0.05 | 98.70% |

### Ablation Study
The key analysis determines whether surrogate extrapolation can compensate for the insufficient compute range of trained models.

| Budget $B$ | AbC SH LMC | AbC GP Mean | AbC UCB | AbC LCB | Notes |
|------------|------------|-------------|---------|---------|-------|
| $10^3$ | 5.84 | 0.51±0.27 | 0.62±0.27 | 0.49±0.16 | Direct curve deviation is high at low budgets; GP extrapolation corrects it significantly |
| $10^4$ | 3.88 | 0.36±0.42 | 0.48±0.13 | 0.45±0.19 | Uncertainty decreases as budget increases |
| $10^5$ | 2.17 | 0.00±0.00 | 0.53±0.31 | 0.38±0.16 | GP mean nearly recovers ground truth |

| Analysis Dimension | Observation | Insight |
|--------------------|-------------|---------|
| synthetic clean curves | SH LMC improvements up to 5.47% mean / 20.30% max | GP effectively utilizes cross-curve correlation when patterns are strong |
| noisy curves | SH LMC average minimum loss remains lower than SH under white/Brownian/OU noise | Surrogate is somewhat robust to short-term noise |
| nanoGPT real curves | Relative gains of ~2%-3%, smaller than synthetic | Real curves are closer and noisier, requiring refined prediction |
| UA baseline | Significant degradation in most settings | Simple uniform allocation is not a good strategy for scaling law sampling |

### Key Findings
- Standard SH is already much better than uniform allocation but tends to favor small models with fast early decline. Adding a surrogate makes it more likely to retain larger models with high later potential.
- SH LMC's benefits are more pronounced on synthetic data and more modest but stable on nanoGPT. Given the cost of training large models, even a 2%-3% loss improvement or reduction in incorrect pruning has practical value.
- Scaling law accuracy does not depend solely on minimum loss. Table 3 shows SH and SH LMC have similar AbC in some cases, indicating both can form usable frontiers; LMC's distinct advantage is reducing regret and providing extrapolation/uncertainty.
- Cost savings are the core value. Compared to training full learning curves for all selected models, the method saves between 75.61% and 98.70% of compute.

## Highlights & Insights
- This paper treats scaling law data collection as a formal resource allocation problem rather than "heuristically training a few models." This has practical significance for large-scale experiment planning.
- Using learning-curve surrogates to correct the short-sightedness of SH is natural. Early performance by small models does not represent the final frontier, and extrapolation addresses this gap.
- The authors do not mix surrogate-predicted curves into the final training data; they are used only for pruning decisions. Scaling laws are still fitted using real training curves, making the method more robust than pure extrapolation.
- Using GP UCB/LCB for scaling law interval estimation is insightful. Real-world budget decisions often require an optimistic/pessimistic range rather than just a point estimate.

## Limitations & Future Work
- Real-world experiments only covered the nanoGPT model family, up to 1.5B parameters. Effectiveness across larger scales, different architectural families, and diverse datasets still needs verification.
- Surrogate training relies on early learning curves being sufficiently predictive. "Late bloomer" models, shifts in training regimes, or data curricula might mislead early extrapolation.
- SH LMC does not always significantly outperform SH on AbC, indicating the minimum loss proxy is not perfectly aligned with the scaling law fitting objective.
- The method requires a pre-defined set of candidate models and compute range. If the candidate space is poorly defined, the correct frontier cannot be recovered even with optimal allocation.
- The complexity of GP/LMC and DE surrogates is higher than standard SH, requiring reliable engineering support for practical deployment.

## Related Work & Insights
- **vs Uniform Allocation**: UA is simple and fair but wastes budget; this paper uses multi-round pruning to concentrate training on models likely to contribute to the frontier.
- **vs Successive Halving / Hyperband**: Traditional SH prunes based on current performance; this paper uses surrogates to predict future learning curves, reducing the risk of premature elimination of large models.
- **vs Freeze-Thaw BO**: Sequential methods like Freeze-Thaw select one configuration at a time, which is unsuitable for scaling law scenarios requiring parallel training of multiple curves.
- **vs LC-PFN / Single-curve extrapolation**: Single-curve methods do not utilize cross-model correlations; LMC captures common trends across different model scales via co-regionalization.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Practically combines SH, learning curve surrogates, and scaling law data collection with a clear problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes synthetic, noise, nanoGPT, and scaling law AbC analysis; larger model families are still missing.
- Writing Quality: ⭐⭐⭐⭐☆ Logically complete with rich tables; the high volume of symbols and appendices increases reading cost.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for budget-constrained scaling law experiments, with direct impact on training planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](../../NeurIPS2025/model_compression/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[ICML 2026\] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search](a_language-guided_bayesian_optimization_for_efficient_lora_hyperparameter_search.md)
- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)

</div>

<!-- RELATED:END -->
