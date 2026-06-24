---
title: >-
  [Paper Note] Score Matching with Missing Data
description: >-
  [ICML 2025 (Spotlight)][score matching] This paper adapts score matching and its major extensions to missing data scenarios, proposing two variants—the Importance Weighting (IW) method and the variational method—and demonstrating their respective advantages in various scenarios such as graphical model estimation.
tags:
  - "ICML 2025 (Spotlight)"
  - "score matching"
  - "missing data"
  - "importance weighting"
  - "variational inference"
  - "graphical models"
date: 2026-05-08
content_hash: 5310af15d63a62ba
---

# Score Matching with Missing Data

**Conference**: ICML 2025 (Spotlight)  
**arXiv**: [2506.00557](https://arxiv.org/abs/2506.00557)  
**Code**: None  
**Area**: Statistical Machine Learning / Generative Models  
**Keywords**: score matching, missing data, importance weighting, variational inference, graphical models

## TL;DR
This paper adapts score matching and its major extensions to missing data scenarios, proposing two variants—the Importance Weighting (IW) method and the variational method—and demonstrating their respective advantages in various scenarios such as graphical model estimation.

## Background & Motivation
**Background**: Score matching is a core tool for learning data distributions, widely applied in diffusion models, energy-based models (EBMs), and graphical model estimation. It avoids the need to compute normalizing constants by matching the score function of the model and the data (the gradient of log density $\nabla_x \log p(x)$).

**Limitations of Prior Work**: Existing score matching methods assume fully observed data (all coordinates are observed), whereas missing values are extremely common in real-world data (survey data, sensor failures, privacy masking, etc.). When data is partially missing on arbitrary subsets of coordinates, standard score matching cannot be directly applied.

**Key Challenge**: The core advantage of score matching (not requiring a normalizing constant) no longer automatically holds under missing data, because the score of the marginal distribution differs from the score of the joint distribution.

**Goal**: To design score matching methods that are applicable to data with arbitrary patterns of missingness.

**Key Insight**: Deal with missing coordinates through importance weighting or variational approximation.

**Core Idea**: Connect the score on the observed coordinate subset with the score of the complete data, eliminating the influence of the missing variables through integration or approximation.

## Method

### Overall Architecture
Input: Partially observed data $\{(\mathbf{x}_i^{\text{obs}}, \mathbf{m}_i)\}_{i=1}^n$, where $\mathbf{m}_i$ is the missingness mask  
Output: Score function estimate $s_\theta(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$ of the data-generating distribution

### Key Designs

1. **Importance Weighting (IW) Method**:

    - **Function**: Translates the score matching objective with missing data into a computable form via importance weighting
    - **Mechanism**: For a missingness pattern $\mathbf{m}$, the marginal score matching objective is:  
    $\mathcal{J}_{\text{IW}} = \sum_{\mathbf{m}} \mathbb{E}_{p(\mathbf{x}^{\text{obs}} | \mathbf{m})} \left[ \sum_{j \in \text{obs}} \left( \partial_j s_\theta^j + \frac{1}{2} (s_\theta^j)^2 \right) \cdot w(\mathbf{m}) \right]$  
      where $w(\mathbf{m})$ is the importance weight of the missingness pattern, estimated through the missingness mechanism (e.g., the MAR assumption)
    - **Design Motivation**: The IW method preserves the closed-form/semi-closed-form advantages of score matching, being particularly efficient in low-dimensional and finite domain settings. It provides finite-sample bounds in finite domain setups

2. **Variational Method**:

    - **Function**: Introduces a variational distribution to approximate the conditional distribution of the missing variables
    - **Mechanism**: Approximates the true conditional distribution with a variational distribution $q_\phi(\mathbf{x}^{\text{mis}} | \mathbf{x}^{\text{obs}})$:  
    $\mathcal{J}_{\text{var}} = \mathbb{E}_{q_\phi(\mathbf{x}^{\text{mis}} | \mathbf{x}^{\text{obs}})} \left[ \sum_j \left( \partial_j s_\theta^j(\mathbf{x}) + \frac{1}{2} (s_\theta^j(\mathbf{x}))^2 \right) \right]$  
      alternately optimizing $\theta$ (the score model) and $\phi$ (the variational distribution)
    - **Design Motivation**: In high-dimensional complex scenarios, the variance of IW can be excessively large; the variational method provides more stable gradient estimates through parametric approximations

3. **Extension to Major Score Matching Variants**:

    - **Function**: Extends the aforementioned methods to denoising score matching, sliced score matching, etc.
    - **Mechanism**: Embeds the treatment of missing data (IW or variational) into the objective functions of each variant, maintaining their respective original strengths
    - **Design Motivation**: Ensures the generality of the method, allowing integration with various approaches in the score matching ecosystem

### Loss & Training
- IW Loss: Weighted Fisher divergence objective
- Variational Loss: A score matching objective in the form of a variational lower bound
- Finite-sample bound (IW method): Provides convergence rates under the setting of a finite domain $|\mathcal{X}| < \infty$

## Key Experimental Results

### Main Results

| Task/Data | Metric | IW Method | Variational Method | Complete Case Analysis | Mean Imputation + SM |
|---|---|---|---|---|---|
| Synthetic Gaussian (d=5, 20% missing) | Score MSE | **0.045** | 0.052 | 0.089 | 0.078 |
| Synthetic Gaussian (d=20, 40% missing) | Score MSE | 0.231 | **0.098** | 0.456 | 0.312 |
| Graphical Model Estimation (Synthetic) | Structure Learning F1 | 0.72 | **0.85** | 0.58 | 0.63 |
| Graphical Model Estimation (Real Gene Data) | Precision | 0.65 | **0.78** | 0.51 | 0.59 |

### Ablation Study

| Configuration | Score MSE (d=10) | Description |
|---|---|---|
| IW, 10% missing | **0.028** | IW is optimal at low missing rates |
| IW, 40% missing | 0.185 | Variance increases at high missing rates |
| Variational, 10% missing | 0.035 | Slightly worse than IW at low missing rates |
| Variational, 40% missing | **0.097** | Variational is more stable at high missing rates |
| MCAR vs MAR | Not significant | Both methods are robust to the missingness mechanism |

### Key Findings
- The IW method performs optimally in low-dimensional, small-sample, and low-missing-rate scenarios, with theoretical guarantees.
- The variational method is stronger in high-dimensional and high-missing-rate scenarios, showing a significant advantage particularly in graphical model estimation.
- The two methods are complementary, covering the requirements of different practical scenarios.
- Simple imputation (mean imputation followed by standard score matching) performs significantly worse than ours.

## Highlights & Insights
- First to systematically address the missing data problem in score matching, filling an important gap.
- The complementary design of the two methods is highly practical—users can choose based on the scenario.
- Prominent performance on the crucial application of graphical model estimation.
- Accepted as a Spotlight paper, with its quality being well-recognized.

## Limitations & Future Work
- The variance issue of the IW method in high dimensions or with high missing rates needs to be further mitigated.
- The choice and optimization of the variational distribution may require careful hyperparameter tuning.
- Combining with diffusion models (training diffusion models with missing data) is a valuable future direction.

## Related Work & Insights
- Directly connects with the classical score matching framework of Hyvärinen (2005).
- The EM algorithm concept for missing data is reflected in the variational method.
- Holds direct application value for fields where missing data frequently occurs, such as medical data and survey data.

## Rating
- Novelty: ⭐⭐⭐⭐ First to solve score matching + missing data
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both synthetic and real data, with clear comparison between complementary methods
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theory with clear practical guidance
- Value: ⭐⭐⭐⭐⭐ Addresses a high-frequency pain point, selected as a Spotlight paper

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)
- [\[ICML 2025\] Diversity By Design: Leveraging Distribution Matching for Offline Model-Based Optimization](diversity_by_design_leveraging_distribution_matching_for_offline_model-based_opt.md)
- [\[ICML 2025\] Fully Dynamic Euclidean Bi-Chromatic Matching in Sublinear Update Time](fully_dynamic_euclidean_bi-chromatic_matching_in_sublinear_update_time.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)

</div>

<!-- RELATED:END -->
