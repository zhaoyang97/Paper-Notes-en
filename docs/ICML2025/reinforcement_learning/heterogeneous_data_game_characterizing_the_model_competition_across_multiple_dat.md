---
title: >-
  [Paper Note] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources
description: >-
  [ICML 2025][Reinforcement Learning][Data Heterogeneity] This paper proposes the Heterogeneous Data Game (HD-Game) framework, applying game theory to analyze the competitive behavior of multiple ML model providers over heterogeneous data sources. It uncovers three pure strategy Nash equilibrium (PNE) patterns—non-existence, homogenization, and heterogenization—and provides sufficient/necessary conditions for the existence of each type.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Data Heterogeneity"
  - "Nash Equilibrium"
  - "Game Theory"
  - "ML Market Competition"
  - "Choice Models"
date: 2026-05-08
content_hash: 5ead8fb51af2949f
---

# Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources

**Conference**: ICML 2025  
**arXiv**: [2505.07688](https://arxiv.org/abs/2505.07688)  
**Code**: None  
**Area**: Reinforcement Learning / Game Theory  
**Keywords**: Data Heterogeneity, Nash Equilibrium, Game Theory, ML Market Competition, Choice Models

## TL;DR
This paper proposes the Heterogeneous Data Game (HD-Game) framework, applying game theory to analyze the competitive behavior of multiple ML model providers over heterogeneous data sources. It uncovers three pure strategy Nash equilibrium (PNE) patterns—non-existence, homogenization, and heterogenization—and provides sufficient/necessary conditions for the existence of each type.

## Background & Motivation

**Background**: In real-world ML markets, data typically originates from multiple heterogeneous sources (e.g., patient data from different hospitals), while there exist multiple ML model providers competing against each other in the market. Most prior works focus on how a single model can perform robustly on heterogeneous data (such as invariant learning, distributionally robust optimization, etc.).

**Limitations of Prior Work**: Existing ML competition analysis works (such as Ben-Porat & Tennenholtz 2017; Jagadeesan et al. 2023) primarily assume a single data distribution, neglecting the heterogeneity between data sources. However, in reality, different data sources have distinct distributional characteristics, which directly impacts the strategic choices of model providers.

**Key Challenge**: The interaction between data heterogeneity and market competition remains unclear—when multiple providers compete in a heterogeneous data market, what do the equilibrium strategies look like? Is everyone training a "one-size-fits-all model" or do they individually specialize in certain data sources?

**Goal**: To establish a game-theoretic framework for multi-ML provider competition under heterogeneous data, and analyze the existence and concrete forms of pure strategy Nash equilibria.

**Key Insight**: To analogize ML model competition to competitive location models, using the Mahalanobis distance to measure the loss of models on each data source, and two choice models (proximity and probability) to characterize data source selection behaviors.

**Core Idea**: Characterize the Nash equilibrium structure under monopoly, duopoly, and multi-provider scenarios using two data source choice models (deterministic optimal choice vs. logit probabilistic choice), revealing how data heterogeneity, the temperature parameter, and the number of providers jointly determine the type of market equilibrium.

## Method

### Overall Architecture
Consider $K$ data sources (each with weight $w_k$, ground truth parameter $\theta_k$, and covariance matrix $\Sigma_k$), and $N$ model providers who each choose a model parameter $\hat{\theta}_n$. The loss of model $n$ on data source $k$ is defined as the squared Mahalanobis distance $\ell_{n,k} = (\hat{\theta}_n - \theta_k)^\top \Sigma_k (\hat{\theta}_n - \theta_k)$. Each data source is allocated to providers according to a choice model, and the utility of each provider is the sum of their obtained weighted market shares.

### Key Designs

1. **Two Data Source Choice Models**:

    - **Function**: Characterize how data sources choose from multiple ML models.
    - **Mechanism**: The Proximity choice model (PROX) directly selects the model with the minimum loss (deterministic); the Probability choice model (PROP) uses logit softmax with a temperature parameter $t$ to control noisy selection, where $t \to 0$ degenerates to PROX and $t \to \infty$ becomes uniform random.
    - **Design Motivation**: The two models respectively correspond to rational decision-making and boundedly rational decision-making, covering different choice behaviors in reality.

2. **Characterization of Equilibrium Strategy Set (Proposition 4.1)**:

    - **Function**: Restrict the strategy space that each player may choose in equilibrium.
    - **Mechanism**: Prove that any strategy in any PNE must belong to the set $\vartheta = \{\bar{\theta}(\boldsymbol{q}) : \boldsymbol{q} \in \Delta_K\}$, where $\bar{\theta}(\boldsymbol{q})$ is the weighted optimum of the truth parameters of the data sources. That is, each provider's optimal strategy is equivalent to minimizing the weighted loss under some allocation of data source weights.
    - **Design Motivation**: To reduce the search space from a high-dimensional continuous strategy space to a $K$-dimensional simplex, making theoretical analysis feasible.

3. **Full Classification of Three Types of Equilibria**:

    - **Function**: Provide the existence and form of equilibria under different market structures.
    - **Mechanism**: Under PROX, a duopoly PNE exists if and only if $w_1 \geq 0.5$ (there is a dominant data source), and the PNE must be heterogenized (both providers choose the truth parameter of the dominant source); under PROP, a duopoly PNE must be homogenized (both choosing the weighted optimum $\hat{\theta}^M$), and a necessary and sufficient condition for its existence is that the temperature $t \geq \underline{t}$. With multiple providers, PROX produces heterogenized PNE (providers allocate proportionally according to data source weights), while PROP produces homogenized PNE at high temperatures and heterogenized PNE at low temperatures.
    - **Design Motivation**: To comprehensively understand how different market parameters affect competitive outcomes, providing a theoretical foundation for regulatory policies.

### Loss & Training
This paper is a purely theoretical work and does not involve model training. The analysis is based on a loss function defined by the Mahalanobis distance, which is equivalent to MSE under linear models.

## Key Experimental Results

### Main Results
This paper validates theoretical results through synthetic experiments. Setting $K=2$ data sources, $D=2$ dimensional parameter space, 10 game configurations are randomly generated.

| Experimental Settings | Metrics | Results |
|---------|------|------|
| Homogenized PNE threshold temperature $\underline{t}$ | $\underline{t}/(2\ell_{max})$ | Around 0.1-0.2, far smaller than the theoretical upper bound of 1.0 |
| Maximum temperature for heterogenized PNE | $t$ value | Decreasing trend as $N$ increases |
| Coexistence of two types of PNE | Whether they coexist | Coexists in some configurations, unlikely for large $N$ |

### Ablation Study

| Configuration | Key Findings | Note |
|------|---------|------|
| $N$ increasing (2→30) | Homogenized PNE threshold increases and then stabilizes | Consistent with $\underline{t} \leq 2\ell_{max}$ |
| $t$ increasing | Heterogenized PNE appears first, followed by homogenized PNE | Validates that temperature determines the equilibrium type |
| Changes in data heterogeneity | Larger $\ell_{max}$ leads to higher threshold $\underline{t}$ | Stronger heterogeneity requires higher noise to have a homogenized equilibrium |

### Key Findings
- Under PROX, a duopoly market has a PNE if and only if the weight of the largest data source is $\geq 0.5$, and providers inevitably "cluster" around the dominant data source.
- Under PROP, the temperature parameter $t$ is key to determining the type of equilibrium: low temperatures promote heterogenization (model diversity), while high temperatures promote homogenization (one-size-fits-all model).
- The larger the number of providers, the more likely a PNE exists, and in heterogenized PNE, providers are allocated proportionally to the weights of the data sources.
- The two types of PNE can coexist in the same game (Example 5.2 validates the case of $N=8, K=2$).

## Highlights & Insights
- **Novelty of the Game-Theoretic Perspective**: For the first time, data heterogeneity is introduced into ML competition analysis, revealing the game-theoretic foundations of the "one-size-fits-all model" and "specialized model" market paradigms. This perspective can be extended to federated learning to analyze the strategic choices of different participants.
- **Policy Implications**: The theoretical results directly guide regulation—if model diversity is to be promoted, rational choices of data sources should be encouraged (i.e., lowering the "temperature"); if certain small data sources are neglected, balance can be achieved by increasing the number of providers or providing incentives.
- **Mahalanobis Distance Framework**: Simultaneously captures both concept shift and covariate shift, and is applicable to linear probing scenarios, giving the theoretical results broader applicability.

## Limitations & Future Work
- Theoretical analysis is limited to linear models and Mahalanobis distance; applicability to deep neural networks remains unverified.
- Assumes that data source weights and distribution parameters are known, which are typically uncertain in reality.
- Only pure strategy Nash equilibria are analyzed; mixed strategy equilibria might provide more insights when a PNE does not exist.
- Synthetic experiments only utilize a small-scale setting of $K=2, D=2$, lacking validation on real-world datasets.
- Dynamic games (such as strategy adjustments in multi-round competitions) and asymmetric information scenarios are not considered.

## Related Work & Insights
- **vs. Traditional ML Competition Analysis (Jagadeesan et al. 2023)**: They assume competition under a single data distribution, whereas this paper extends to heterogeneous data sources, discovering entirely new equilibrium structures.
- **vs. Competitive Location Models (Hotelling 1929)**: Classical models are limited to low-dimensional spaces and uniform distance metrics. This paper introduces source-specific distance metrics and high-dimensional strategy spaces, which is more challenging but closer to practical ML scenarios.
- **vs. Distributionally Robust Optimization (Duchi & Namkoong 2021)**: DRO focuses on the robustness of a single model, whereas this paper focuses on the equilibrium behavior of multiple competing models. The two can be combined—each provider in the competition can use DRO to choose their strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing data heterogeneity into the ML competition framework is a brand-new perspective, but is limited to linear models.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments validate the theory, but lack empirical validation in real-world scenarios.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous and clear, with a unified notation system.
- Value: ⭐⭐⭐⭐ Holds significant reference value for understanding ML market competition and formulating regulatory policies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Automatic Reward Shaping from Confounded Offline Data](automatic_reward_shaping_from_confounded_offline_data.md)
- [\[ICML 2025\] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks](the_impact_of_on-policy_parallelized_data_collection_on_deep_reinforcement_learn.md)
- [\[ICML 2025\] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration](leveraging_skills_from_unlabeled_prior_data_for_efficient_online_exploration.md)
- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [\[ICLR 2026\] Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels](../../ICLR2026/reinforcement_learning/webscale-rl_automated_data_pipeline_for_scaling_rl_data_to_pretraining_levels.md)

</div>

<!-- RELATED:END -->
