---
title: >-
  [Paper Note] Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning
description: >-
  [ICML2026][Model Compression][scaling law] This paper models the training budget allocation in scaling law experiments as a multi-round resource selection problem. By combining Successive Halving with learning curve surr…
tags:
  - "ICML2026"
  - "Model Compression"
  - "scaling law"
  - "Successive Halving"
  - "learning curves"
  - "Gaussian Process"
  - "budget allocation"
date: 2026-05-08
content_hash: b1e37ffcb3c16bcc
---

# Active Budget Allocation for Efficient Scaling Law Estimation via Surrogate-Guided Pruning

**Conference**: ICML2026  
**arXiv**: [2605.17234](https://arxiv.org/abs/2605.17234)  
**Code**: No explicit repository (paper states code will be released)  
**Area**: Optimization / Scaling Law / Budget Allocation  
**Keywords**: scaling law, Successive Halving, learning curves, Gaussian Process, budget allocation  

## TL;DR
This paper models the training budget allocation in scaling law experiments as a multi-round resource selection problem. By combining Successive Halving with learning curve surrogates to predict future potential, it approximates full scaling laws with up to 98.7% training cost savings on synthetic and nanoGPT learning curves.

## Background & Motivation
**Background**: Scaling laws use empirical learning curves to describe the relationship between loss and compute, model size, and data size. They are essential tools for planning training budgets, model sizes, and data requirements for large models. The classical approach typically involves training many models of various scales and observing their loss-compute frontier.

**Limitations of Prior Work**: Comprehensive scaling law research is extremely expensive. To obtain a reliable frontier, researchers may need to train dozens to hundreds of models over long compute intervals, many of which do not eventually contribute to the optimal frontier. Traditional uniform allocation distributes the budget equally across all models, wasting resources on models that are too small, plateau early, or are large but perform poorly in the short term.

**Key Challenge**: Small models exhibit rapid loss decay early on, making them appear superior under short budgets. Large models may not initially dominate in loss but possess higher long-term potential. If pruning is executed based solely on current loss, models that could eventually contribute to the scaling frontier may be eliminated prematurely; if pruning is not used, the computational cost becomes prohibitive.

**Goal**: The authors aim to actively decide which models to continue training and which to stop under a fixed total FLOPs budget, ensuring that the final set of obtained learning curves is sufficient to fit an accurate scaling law while significantly reducing costs compared to "full training of all models."

**Key Insight**: Successive Halving from hyperparameter optimization can already allocate resources among multiple configurations, but it only considers observed loss. This paper further allows a surrogate model to predict the future continuation of each learning curve, using "the lowest loss potentially achievable in the future" rather than "current loss" to decide which models to retain.

**Core Idea**: Use learning curve surrogates to correct the short-sighted pruning of Successive Halving, directing the budget toward models with greater potential for the loss-compute frontier.

## Method
The paper decomposes the scaling law data collection process into several rounds. In each round, the current candidate models are allocated an equal increment of compute to train part of their learning curves. Then, a subset of models is selected for the next round based on these curves. Unlike standard SH, SH LMC/SH DE does not look only at the endpoint of the observed curve; it first predicts the potential loss if the model is trained to subsequent budgets and then prunes based on predicted potential.

### Overall Architecture
The input consists of an initial model set $\mathcal M_0$, total budget $B$, and pruning factor $\eta$. In round $r$, each remaining model receives a budget $C_r = \lfloor B/(|\mathcal M_r|\lceil\log_\eta |\mathcal M_0|\rceil)\rfloor$. The model forms a learning curve $L_m(C)$ after training to the current cumulative budget.

Without a surrogate, Top_k selection is based directly on the minimum loss of observed curves. With a surrogate, LMC GP or Deep Ensemble is first used to predict the future loss when each curve extends to the final round's budget, and the selection combines observed curves with predicted continuations. The final output only preserves curves obtained through actual training (the surrogate continuation is not treated as observed data); these curves are then used to fit the compute-loss scaling law.

### Key Designs
1.  **Transforming scaling law sampling into proxy optimization**:
    - **Function**: Avoids directly solving the gold-standard-less and hard-to-optimize problem of "which set of curves fits the scaling law best."
    - **Mechanism**: The authors first optimize a proxy: find a set of models that achieves the lowest validation loss within the total budget. This process naturally produces a set of learning curves trained to varying degrees, which are then used to fit the scaling law.
    - **Design Motivation**: The true target of scaling laws requires knowing the ground-truth frontier after complete training, which is precisely the expensive part. The proxy target can be derived directly from current training loss, making it suitable for approximation using SH-like algorithms.

2.  **LMC Gaussian Process for learning cross-curve correlations**:
    - **Function**: Predicts the subsequent loss trend of a specific model based on the early learning curves of multiple models.
    - **Mechanism**: The LMC surrogate models curve extrapolation as a multi-input multi-output GP. The kernel combines exponential decay, white noise, and bias sub-kernels, capturing correlations between different model curves through a co-regionalisation matrix. When a small model plateaus or a large model's curve improves, it provides extrapolation signals for other curves.
    - **Design Motivation**: Ordinary SH is easily misled by early loss. LMC utilizes curve shape priors and cross-model correlations to ensure that larger models are retained if they are predicted to be superior in the future, even if their current loss is not the lowest.

3.  **Deep Ensemble surrogate and scaling law extrapolation**:
    - **Function**: Compares non-parametric GP with parameterized curve families for budget allocation and uses the surrogate to predict extended compute ranges.
    - **Mechanism**: Deep Ensemble uses a two-layer MLP to condition the coefficients of functions like power law, exponential, and Morgan-Mercer-Flodin to predict learning curve shapes. Subsequent synthetic experiments also use GP mean/UCB/LCB to extrapolate learning curves after SH LMC, reducing the AbC gap between the scaling law and ground truth.
    - **Design Motivation**: Curve noise and shapes vary across datasets, so a single surrogate might not be optimal; meanwhile, scaling laws often need to exceed the trained compute range, where the surrogate's uncertainty bounds can provide a decision interval.

### Loss & Training
LMC GP is optimized using L-BFGS with 20 random restarts. Deep Ensemble uses 5 randomly initialized two-layer perceptrons trained for 1000 iterations. By default, 20 observation points are sampled from each curve to train the surrogate. Scaling law fitting uses $L^{SL}(C)=(C/\alpha)^{-\gamma}$, and the Area between Curves (AbC) is used to measure the distance between the fitted curve and the ground truth scaling law over a specified compute interval.

## Key Experimental Results

### Main Results
On synthetic learning curves, SH LMC shows stable improvement over ordinary SH, while uniform allocation is significantly worse.

| Model Count $M_0$ | Budget $B$ (petaFLOPs) | SH mean loss | SH LMC mean rel. improv. | UA mean rel. degradation | Conclusion |
|-------------------|----------------------|--------------|---------------------------|---------------------------|------------|
| 5 | $10^2$ | 6.40±9.07 | 5.15% (max 20.30%) | -10.17% | Surrogate improvement is significant with few models |
| 5 | $10^4$ | 3.84±2.03 | 5.47% (max 16.70%) | -7.59% | Gains persist at high budgets |
| 10 | $10^3$ | 3.86±0.38 | 2.38% (max 6.11%) | -14.06% | SH is already strong, but LMC still improves |
| 20 | $10^4$ | 3.18±0.09 | 1.50% (max 6.53%) | -16.40% | Gains are smaller but stable with many models |

In real-world nanoGPT learning curve experiments, SH LMC also outperforms SH and most DE surrogates, and all strategies outperform UA.

| $M_0$ | Budget $B$ | SH mean loss | SH LMC rel. improv. | Strongest DE rel. improv. | UA rel. degradation |
|-------|----------|--------------|----------------------|----------------------|---------------------|
| 5 | $10^4$ | 3.17±0.06 | 2.58% | 2.32% (DE EXP) | -5.09% |
| 5 | $10^5$ | 2.97±0.03 | 2.36% | 2.40% (DE PL) | -0.74% |
| 10 | $10^5$ | 3.00±0.02 | 2.82% | 2.14% (DE MMF) | -0.81% |
| 20 | $10^4$ | 3.30±0.02 | 2.84% | 2.02% (DE PL) | -11.46% |
| 20 | $10^5$ | 3.03±0.01 | 2.24% | 1.44% (DE EXP) | -2.96% |

Regarding scaling law fitting, both SH and SH LMC obtain laws close to the ground truth at budgets far below the cost of full training.

| Setting | Method | AbC vs Full Data SL | Loss Regret | Rel. Cost Savings vs Full |
|---------|--------|---------------------|-------------|----------------------------|
| $M_0=5,B=10^4$ | SH | 0.09±0.05 | 0.43±0.09 | 94.00% |
| $M_0=5,B=10^4$ | SH LMC | 0.11±0.07 | 0.41±0.10 | 94.00% |
| $M_0=10,B=10^4$ | SH | 0.07±0.02 | 0.56±0.07 | 97.50% |
| $M_0=10,B=10^4$ | SH LMC | 0.09±0.04 | 0.51±0.06 | 97.50% |
| $M_0=20,B=10^4$ | SH | 0.12±0.04 | 0.67±0.03 | 98.70% |
| $M_0=20,B=10^4$ | SH LMC | 0.11±0.07 | 0.59±0.05 | 98.70% |

### Ablation Study
The key analysis in the paper is whether surrogate extrapolation can compensate for the lack of trained compute range.

| Budget $B$ | AbC SH LMC | AbC GP Mean | AbC UCB | AbC LCB | Note |
|------------|------------|-------------|---------|---------|------|
| $10^3$ | 5.84 | 0.51±0.27 | 0.62±0.27 | 0.49±0.16 | Large deviation without surrogate at low budget; GP significantly corrects |
| $10^4$ | 3.88 | 0.36±0.42 | 0.48±0.13 | 0.45±0.19 | Uncertainty decreases as budget increases |
| $10^5$ | 2.17 | 0.00±0.00 | 0.53±0.31 | 0.38±0.16 | GP mean almost restores ground truth |

| Analysis Dimension | Observation | Insight |
|--------------------|-------------|---------|
| Synthetic clean curves | SH LMC improvement up to 5.47% mean / 20.30% max | GP utilizes cross-curve correlations well when patterns are strong |
| Noisy curves | SH LMC minimum loss remains lower than SH under white/Brownian/OU noise | Surrogates possess robustness against short-term noise |
| nanoGPT real curves | Relative gains approx 2%-3%, smaller than synthetic | Real curves are closer and noisier, requiring finer prediction |
| UA baseline | Significant degradation in most settings | Simple uniform allocation is not a good strategy for scaling law sampling |

### Key Findings
- Ordinary SH is already much better than uniform allocation but tends to favor small models that drop quickly early on. With a surrogate, larger models with high long-term potential are more likely to be retained.
- The benefits of SH LMC are more pronounced on synthetic data and more moderate but stable on nanoGPT. Given the cost of LLM training, even a 2%-3% loss improvement or reduction in incorrect pruning has practical value.
- The accuracy of the scaling law does not depend solely on the minimum loss. Table 3 shows SH and SH LMC have similar AbC at times, indicating both can form usable frontiers; LMC's prominent advantage lies in reducing regret and providing extrapolation/uncertainty.
- Cost saving is the core value. The method can save 75.61% to 98.70% of compute compared to training all selected models to their full learning curves.

## Highlights & Insights
- The paper transforms scaling law data collection from "empirically training a few more models" into a well-defined resource allocation problem. This has significant practical implications for planning LLM experiments.
- Using a learning-curve surrogate to correct SH's short-sightedness is natural. Good performance of small models early on does not guarantee a good final frontier, and model extrapolation compensates for this flaw.
- The authors do not directly mix predicted surrogate curves into the final training data; they are used only for pruning decisions, and the final scaling law is still based on real curves. This makes the method more robust than pure extrapolation.
- Using GP UCB/LCB for scaling law interval estimation is insightful. Practical training budget decisions often require not just a point estimate but also an optimistic/pessimistic curve range.

## Limitations & Future Work
- Real experiments only cover the nanoGPT model family, up to 1.5B parameters. Effectiveness on larger scales, different architectural families, and diverse datasets still needs verification.
- Surrogate training relies on early learning curves being sufficiently predictive. If "late bloomer" models, training regime switches, or data curricula exist, early curves may mislead extrapolation.
- SH LMC does not always significantly outperform SH in terms of AbC, suggesting the minimum loss proxy is not perfectly aligned with the scaling law fitting objective.
- The method requires a pre-defined candidate set and compute range. If the candidate space itself is poorly covered, even optimal budget allocation cannot recover the correct frontier.
- The complexity of GP/LMC and DE surrogates is higher than standard SH, requiring reliable engineering tools for practical deployment.

## Related Work & Insights
- **vs Uniform Allocation**: UA is simple and fair but wasteful; this paper uses multi-round pruning to concentrate training on models likely to contribute to the frontier.
- **vs Successive Halving / Hyperband**: Traditional SH prunes based on current performance; this paper uses surrogates to predict future learning curves, reducing the risk of premature elimination of large models.
- **vs Freeze-Thaw BO**: Sequential methods like Freeze-Thaw select one configuration at a time, which is unsuitable for scaling law scenarios requiring parallel training of multiple curves; this paper emphasizes parallel resource allocation.
- **vs LC-PFN / Single-curve extrapolation**: Single-curve methods do not utilize correlations across model curves; LMC captures common trends among different scale models through co-regionalisation.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines SH, learning curve surrogates, and scaling law data collection practically; problem definition is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes synthetic, noise, nanoGPT, and scaling law AbC analysis; larger model families are still missing.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological logic is complete, tables are rich; many symbols and appendices result in slightly high reading cost.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for budget-constrained LLM scaling law experiments, directly impacting training planning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Tabular Augmentation via Policy-Guided Diffusion Inpainting](active_tabular_augmentation_via_policy-guided_diffusion_inpainting.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](../../NeurIPS2025/model_compression/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[ICML 2026\] A Language-Guided Bayesian Optimization for Efficient LoRA Hyperparameter Search](a_language-guided_bayesian_optimization_for_efficient_lora_hyperparameter_search.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)

</div>

<!-- RELATED:END -->
