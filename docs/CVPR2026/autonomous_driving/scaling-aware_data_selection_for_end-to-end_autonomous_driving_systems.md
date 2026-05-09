---
title: >-
  [Paper Note] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems
description: >-
  [CVPR 2026][Autonomous Driving][Data Selection] This paper proposes MOSAIC, a framework that clusters training data into domains, fits per-domain scaling laws over evaluation metrics, and greedily selects samples with the highest marginal gain, enabling efficient data selection for end-to-end autonomous driving models that matches or surpasses baseline performance with 80% less data.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Data Selection
  - Neural Scaling Laws
  - Data Mixture Optimization
  - End-to-End Autonomous Driving
  - EPDMS
date: 2026-05-08
content_hash: a8bb82c3196135a0
---

# Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems

**Conference**: CVPR 2026
**arXiv**: [2604.08366](https://arxiv.org/abs/2604.08366)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Data Selection, Neural Scaling Laws, Data Mixture Optimization, End-to-End Autonomous Driving, EPDMS

## TL;DR

This paper proposes MOSAIC, a framework that clusters training data into domains, fits per-domain scaling laws over evaluation metrics, and greedily selects samples with the highest marginal gain, enabling efficient data selection for end-to-end autonomous driving models that matches or surpasses baseline performance with 80% less data.

## Background & Motivation

1. **Background**: Large-scale deep learning models rely on diverse training data, especially in physical AI applications such as autonomous driving, where data must cover varied locations, weather conditions, and traffic scenarios. However, training on the full dataset is computationally prohibitive, motivating intelligent data selection strategies.
2. **Limitations of Prior Work**: (A) Influence estimation and active learning methods operate in feature space but do not account for how different data affect different evaluation metrics. (B) Existing data mixture methods (e.g., DoReMi, ADO) assume domains are explicitly defined and homogeneous, ignoring the heterogeneous impact rates of data sources on different metrics. (C) Physical AI systems must simultaneously optimize multiple potentially competing metrics (e.g., route progress vs. driving comfort vs. collision avoidance).
3. **Key Challenge**: A single training sample contributes differently to different metrics, yet existing frameworks cannot model this many-to-many, heterogeneous relationship between data and metrics.
4. **Goal**: Given a fixed data budget, select a training subset from a heterogeneous data pool that maximizes the aggregated metric (EPDMS).
5. **Key Insight**: Cluster the data pool into domains with similar metric impact, fit per-domain scaling laws independently, and then iteratively select the optimal mixture via greedy search.
6. **Core Idea**: Cluster first, fit scaling laws second, then greedily select — decomposing the complex multi-metric data selection problem into independently estimable, domain-level scaling subproblems.

## Method

### Overall Architecture

MOSAIC proceeds in three stages: (1) **Clustering and Ranking** — the data pool is partitioned into $M$ domains, with samples within each domain ranked by importance scores; (2) **Scaling Law Fitting** — small-scale pilot runs are used to estimate the data-metric scaling relationship for each domain; (3) **Iterative Selection** — samples are greedily selected one by one from the domain with the highest marginal gain until the budget is exhausted.

### Key Designs

1. **Domain Clustering and Sample Ranking**:

    - **Function**: Partition the heterogeneous data pool into subsets with similar metric impact, and prioritize high-influence samples within each domain.
    - **Mechanism**: Feature representations (e.g., semantic descriptions, geographic location) are used to cluster the data pool into $M$ domains. Within each domain, samples are ranked by an importance score $\mathcal{I}(x) = U(\{\mathcal{G}_r(f(\cdot; \mathcal{D}_{train}), x)\}_{r=1}^R)$, i.e., the aggregated metric value produced by the current model on that sample. High-importance samples are selected first.
    - **Design Motivation**: Clustering decouples the heterogeneous influence of different data on different metrics, ensuring consistency within domains for subsequent scaling law estimation. Ranking ensures that the most valuable samples are retrieved for any given selection size.

2. **Domain-Level Scaling Law Fitting**:

    - **Function**: Estimate the marginal improvement in the aggregated metric when additional data from each domain is included.
    - **Mechanism**: The contributions of individual domains to the mixture utility are assumed to be linearly separable: $\Delta U_{mix}(n_1,...,n_M) \approx \sum_{i=1}^M \Delta U_i(n_i)$. A saturating exponential scaling law $\Delta \hat{U_i}(n) = a_i(1 - e^{-n/\tau_i})$ is fitted per domain, where $a_i$ denotes the asymptotic improvement and $\tau_i$ the saturation rate. Parameters are estimated via small-scale pilot runs that train lightweight models on varying quantities of domain-specific data.
    - **Design Motivation**: The saturating exponential form captures the intuition of diminishing returns as more data is added. The linear separability assumption reduces the combinatorial optimization problem to independent single-domain estimations, enabling tractable prediction of which domain yields the greatest gain from adding one more sample.

3. **Scaling-Aware Greedy Iterative Selection**:

    - **Function**: Determine the optimal sample count per domain under a given budget.
    - **Mechanism**: The method maintains per-domain selection counts $b_i$ and computes the marginal gain $\delta_i(b_i) = \Delta\hat{U_i}(b_i+1) - \Delta\hat{U_i}(b_i)$ at each step. The domain with the highest marginal gain $j = \arg\max_i \delta_i(b_i)$ is selected, and its top-ranked unselected sample is added to the training set. This repeats until the budget is exhausted. Because $\Delta\hat{U_i}(n)$ is concave, marginal gains within each domain decrease monotonically, naturally inducing balanced cross-domain allocation.
    - **Design Motivation**: This procedure is equivalent to first-order discrete gradient ascent on a concave objective, leveraging greedy guarantees from submodular optimization. It is substantially more efficient than grid search or joint mixture ratio optimization.

### Loss & Training

- Hydra-MDP (NAVSIM 2024 champion) with a VoVNetV2-99 backbone and trajectory vocabulary size of 16,384 is used as the base model.
- Evaluation metric: EPDMS — an aggregation of nine rule-compliance metrics, including penalty terms (NC, DAC, DDC, TLC) and weighted average terms (EP, TTC, LK, HC, EC).
- Pilot runs are used to estimate scaling law parameters; the main model is trained on the selected subset.

## Key Experimental Results

### Main Results

OpenScene experiments (selected from 31,539 clips):

| Budget | Method | EPDMS ↑ | BRMR ↓ |
|--------|--------|---------|--------|
| 250 | Random | 72.84 | 1.00 |
| 250 | Coreset | 76.26 | 0.20 |
| 250 | MOSAIC | **77.38** | **0.15** |
| 1000 | Random | 75.84 | 1.00 |
| 1000 | MOSAIC | **81.68** | **0.18** |
| 4000 | Random | 80.38 | 1.00 |
| 4000 | MOSAIC | **84.25** | **0.18** |

Navtrain experiments:

| Budget | Method | EPDMS ↑ | BRMR ↓ |
|--------|--------|---------|--------|
| 100 | Random | 84.66 | 1.00 |
| 100 | MOSAIC | **86.29** | **0.30** |
| 1600 | Random | 88.62 | 1.00 |
| 1600 | MOSAIC | **90.18** | **0.37** |

MOSAIC achieves equivalent EPDMS performance using approximately 18–30% of the data required by random selection (BRMR 0.15–0.37).

### Ablation Study

EPDMS sub-metric breakdown (OpenScene, 4,000 clips):

| Method | NC ↑ | DAC ↑ | EP ↑ | TTC ↑ | LK ↑ | EPDMS ↑ |
|--------|------|-------|------|-------|------|---------|
| Base | 94.05 | 83.9 | 85.96 | 92.95 | 93.26 | 72.0 |
| Random | 96.32 | 90.53 | 86.36 | 95.66 | 95.68 | 80.38 |
| Uncertainty | 94.67 | 85.11 | 84.26 | 93.72 | 93.26 | 73.46 |
| Coreset | 97.11 | 92.93 | 86.65 | 96.42 | 96.66 | 83.63 |
| MOSAIC | 96.97 | **93.59** | **87.14** | 96.18 | 96.62 | **84.25** |

### Key Findings

- Uncertainty sampling performs worst — high-entropy samples may be noisy or edge cases, and over-representing them degrades overall performance.
- MOSAIC outperforms Coreset at all budget levels, with the gap widening at smaller budgets, indicating that scaling law guidance is more critical under data scarcity.
- The combination of clustering and scaling laws substantially outperforms clustering alone (Chameleon), suggesting that domain-level improvement estimation from scaling laws compensates for imperfect clustering.
- MOSAIC reaches the EPDMS of full-data training using approximately 42% of the data.
- Different domains (e.g., Pittsburgh curves vs. Las Vegas urban areas) exhibit different contribution rates to different metrics, validating the heterogeneous impact hypothesis.

## Highlights & Insights

- **Scaling Laws as a Data Selection Signal**: Unlike sample-level signals such as influence functions or uncertainty estimates, scaling laws operate at the domain level, are more stable, and naturally model diminishing returns — making them well-suited for large-scale data selection.
- **Elegance of the Greedy Algorithm**: For a concave objective, selecting the domain with the highest marginal gain at each step is equivalent to first-order discrete optimization, providing both simplicity and theoretical guarantees. This strategy transfers directly to settings such as LLM data mixture optimization.
- **The BRMR Metric**: The proposed Budget Ratio to Match Random baseline offers a concise and intuitive measure of data efficiency and merits broader adoption.
- **Robustness to Clustering Choice**: The paper demonstrates that MOSAIC consistently outperforms baselines regardless of whether semantic descriptions or geographic locations are used for clustering, indicating that the primary gains stem from scaling law guidance rather than clustering quality.

## Limitations & Future Work

- The linear separability assumption ignores cross-domain interaction effects — certain domain combinations may yield super- or sub-additive outcomes.
- Fitting scaling laws requires multiple pilot runs, introducing non-trivial computational overhead.
- Validation is limited to NAVSIM/OpenScene; the framework has not been tested in closed-loop driving or other physical AI systems.
- The number of clusters $M$ relies on prior knowledge (the paper uses four geographic metadata domains).
- Promising directions for improvement include: nonlinear scaling law models incorporating cross-domain interaction terms; online adaptive updating of scaling law parameters; and extension to other multi-metric optimization settings (e.g., robotic manipulation, multi-task learning).

## Related Work & Insights

- **vs. Chameleon**: Chameleon uses kernel ridge regression scores in model feature space for domain weighting but does not explicitly model the data-quantity–performance scaling relationship. MOSAIC builds on Chameleon's clustering foundation and adds scaling laws, outperforming it across all settings.
- **vs. ADO**: ADO online-fits a scaling estimator during training for mixture reweighting, but does not model independent domain-level scaling and requires multiple hyperparameters such as time averaging. MOSAIC's offline scaling law fitting combined with greedy selection is simpler and more stable.
- **vs. CoreSet**: CoreSet maximizes feature-space diversity and ranks second under most MOSAIC settings, indicating that diversity is necessary but not sufficient — metric-sensitive selection yields additional gains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing scaling laws into a multi-metric data selection framework is a novel design; the greedy algorithm is simple but well-adapted to the problem.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple baselines and budgets, fine-grained metric decomposition, and robustness analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear and algorithmic descriptions are rigorous, though some mathematical notation could be simplified.
- **Value**: ⭐⭐⭐⭐ Offers practical guidance for data-efficient training; the framework is broadly applicable, though validation could be extended to more diverse settings.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CausalVAD: De-confounding End-to-End Autonomous Driving via Causal Intervention](causalvad_de-confounding_end-to-end_autonomous_driving_via_causal_intervention.md)
- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](../../AAAI2026/autonomous_driving/drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving](meanfuser_fast_one-step_multi-modal_trajectory_generation_and_adaptive_reconstru.md)

<!-- RELATED:END -->
