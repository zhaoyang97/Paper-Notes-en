---
title: >-
  [Paper Note] Perturbed Dynamic Time Warping: A Probabilistic Framework and Generalized Variants
description: >-
  [ICLR 2026][Time Series][Dynamic Time Warping] This paper reinterprets soft-DTW from the perspective of perturbed optimization—where random noise is added to alignment costs before taking the expected minimum—proving it to be a special case under Gumbel noise. The noise is generalized to the Generalized Extreme Value (GEV) distribution to derive nested-soft-DTW (ns-DTW) with adjustable skewness, which consistently outperforms soft-DTW in time series barycenter computation…
tags:
  - "ICLR 2026"
  - "Time Series"
  - "Dynamic Time Warping"
  - "soft-DTW"
  - "perturbed optimization"
  - "generalized extreme value distribution"
  - "random utility theory"
date: 2026-05-08
content_hash: ccf30ab28c8675b7
---

# Perturbed Dynamic Time Warping: A Probabilistic Framework and Generalized Variants

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=gxCOTacltM](https://openreview.net/forum?id=gxCOTacltM)  
**Code**: Provided in supplementary materials (UCR dataset is public)  
**Area**: Time Series / Differentiable Dynamic Programming  
**Keywords**: Dynamic Time Warping, soft-DTW, perturbed optimization, generalized extreme value distribution, random utility theory

## TL;DR
This paper reinterprets soft-DTW from the perspective of perturbed optimization—where random noise is added to alignment costs before taking the expected minimum—proving it to be a special case under Gumbel noise. The noise is generalized to the Generalized Extreme Value (GEV) distribution to derive nested-soft-DTW (ns-DTW) with adjustable skewness, which consistently outperforms soft-DTW in time series barycenter computation, clustering, and classification.

## Background & Motivation
**Background**: Dynamic Time Warping (DTW) is a classic method for measuring similarity between two time series. It tolerates stretching/compression along the time axis by seeking the minimum cost alignment path, making it more suitable than Euclidean distance for sequences with unequal lengths or inconsistent rhythms. It is widely used in speech recognition, action recognition, and trajectory clustering. However, the core of DTW is a $\min$ operation, which is non-differentiable and cannot be integrated into end-to-end gradient-based training.

**Limitations of Prior Work**: The soft-DTW proposed by Cuturi & Blondel (2017) replaces the hard $\min$ with a smooth soft-min ($\min_\gamma x = -\gamma \log \sum_i \exp(-x_i / \gamma)$), making it differentiable and allowing for recursive gradient computation. This has become the de facto standard. However, the soft-min in soft-DTW is essentially a **heuristic smoothing operator**. Soft-DTW itself cannot explain why log-sum-exp is used, what probabilistic structure it corresponds to, or whether other forms can be used to achieve more flexible alignments.

**Key Challenge**: Differentiability was "forced" into the algorithm using a specific analytical smoothing function rather than being derived from a unified principle. This results in soft-DTW being inflexible: it provides a fixed expected alignment distribution and cannot characterize structural preferences, such as "bias toward vertical or horizontal directions."

**Key Insight**: The authors adopt the perspective of perturbed optimizers (Berthet et al. 2020) and random utility theory. The choice of an alignment path is treated as a discrete choice problem where the utility of each path $-\langle A, C \rangle$ is perturbed by a random shock $\gamma \varepsilon$. The resulting differentiable quantity is the "expected minimum cost after perturbation." This perspective is naturally differentiable as the expectation smooths out the discontinuous $\arg\min$, and different noise distributions yield different DTW variants.

**Core Idea**: Replace the heuristic soft-min of soft-DTW with "alignment cost plus random perturbation + expected minimum." Prove that under Gumbel noise, it is exactly equal to soft-DTW (providing a probabilistic interpretation), and then replace the noise with a GEV distribution to obtain ns-DTW with adjustable skewness.

## Method

### Overall Architecture
The method follows a logical chain from general principles to a specific algorithm rather than a data flow pipeline. The starting point is replacing the non-differentiable $\min$ of DTW with a **perturbed minimum operator**: a random noise $\gamma \varepsilon$ is subtracted from the cost $\langle A, C \rangle$ of each valid alignment path. The minimum is taken across all paths, and then the expectation is taken over the noise distribution, resulting in perturbed-DTW (Definition 1). This quantity is inherently differentiable: the optimal alignment is no longer a single 0/1 matrix but a probability distribution over paths, whose gradient with respect to the cost matrix $C$ is exactly the "expected alignment matrix" $E$.

The authors then substitute two specific types of noise. The first is i.i.d. Gumbel noise. Using the analytical properties of the minimum of Gumbel variables (Lemma 1), they prove that perturbed-DTW degenerates exactly into soft-DTW (Proposition 1). The second is the more general Generalized Extreme Value (GEV) distribution, which introduces correlation between path groups to derive ns-DTW (Theorem 1), where the alignment distribution has **adjustable skewness**. Finally, this framework is implemented via dynamic programming: the hard $\min$ in the Bellman recursion is replaced with the perturbed minimum operator. This yields a recursion formula computable in $O(mn)$, which simultaneously computes the transition probability tensor $G$ in three directions to back-propagate the expected alignment matrix $E$ (Algorithm 1).

### Key Designs

**1. Perturbed-DTW: Replacing non-differentiable hard min with "noise + expected minimum"**

The root of DTW's non-differentiability is the hard minimum $\min_{A \in \mathcal{A}_{m, n}} \langle A, C \rangle$. Instead of forcing a smooth function, the authors inject random perturbation into the cost of each alignment path and take the expectation:

$$\text{perturbed-DTW}_\gamma(C) := \mathbb{E}_{\varepsilon \sim P}\Big[\min_{A \in \mathcal{A}_{m, n}}\{\langle A, C \rangle - \gamma \varepsilon\}\Big]$$

Where $\varepsilon$ is a perturbation vector of dimension $|\mathcal{A}_{m, n}|$ (total valid alignments) and $\gamma > 0$ is the temperature. Intuitively, each noise sample solves a hard DTW optimal path; taking the expectation aggregates these "randomly realized optimal paths" into a distribution. The expectation operator smooths the discontinuous $\arg\min$, making the quantity differentiable with respect to $C$, where the gradient is the expected alignment matrix $E = \sum_A A \cdot P(A; C)$ (Proposition 3). This corresponds to random utility theory, where choice is determined by utility perturbed by random shocks.

**2. Gumbel perturbation perfectly recovers soft-DTW: A probabilistic interpretation**

When noise is set as i.i.d. Gumbel$(-c, 1)$ ($c \approx 0.5772$, Euler's constant), and utilizing the property that the minimum of Gumbel variables is the negative of the log-sum-exp of their negatives (Lemma 1):

$$\mathbb{E}\big[\min\{x_1-\gamma\varepsilon_1, \dots, x_n-\gamma\varepsilon_n\}\big] = -\gamma \log \sum_{i=1}^n \exp(-x_i / \gamma)$$

The right side is precisely the soft-min. Substituting this into perturbed-DTW yields $-\gamma \log \sum_{A} \exp(-\langle A, C \rangle / \gamma)$, identical to soft-DTW. Furthermore (Proposition 4), if a perturbed cost matrix is defined as $[\tilde C_\gamma]_{i, j} = C_{i, j} - \gamma \varepsilon_{i, j}$, then $\text{soft-DTW}_\gamma(C) = \mathbb{E}[\text{DTW}(\tilde C_\gamma)]$. This equivalence demonstrates that the log-sum-exp in soft-DTW is a necessary result of the Gumbel noise prior.

**3. GEV perturbation leads to ns-DTW: Adjustable skewness via nested correlations**

Gumbel noise assumes independent paths, locking the shape of the alignment distribution. The authors replace the noise with a GEV distribution, a correlated multivariate generalization of Gumbel. Alignments $\mathcal{A}_{m, n}$ are partitioned into $J$ groups, with intra-group correlation characterized by similarity parameters $\tau_\ell \in (0, 1]$. After zero-centering, perturbed-DTW under GEV noise becomes nested-soft-DTW (Theorem 1):

$$\text{ns-DTW}_\gamma(C) = -\gamma \log \sum_{\ell=1}^{J} \Big( \sum_{A \in \ell} \exp\big( -\tfrac{\langle A, C \rangle}{\gamma \tau_\ell} \big) \Big)^{\tau_\ell}$$

When all $\tau_\ell = 1$, it reverts to soft-DTW. When $0 < \tau_\ell < 1$, correlations occur between paths within a group, allowing for **skewed** alignment distributions. This nested structure corresponds to the nested logit model in random utility theory. Smaller $\tau$ makes the model more "selective," emphasizing low-cost transitions and structurally pushing alignment paths toward lower-cost directions (e.g., vertical $\downarrow$).

**4. DP Recursion for the Perturbed Minimum: Computing ns-DTW in O(mn)**

Directly enumerating $\mathcal{A}_{m, n}$ is exponential. The authors replace the hard $\min$ in DTW's Bellman recursion with the perturbed minimum operator to define an accumulated cost matrix $V$. Under Gumbel noise, the recursion simplifies to the soft-DTW recursion. Under GEV noise, it is written in a nested form based on grouping schemes (e.g., $J_1 = \{\downarrow, \rightarrow\}, J_2 = \{\searrow\}$). The transition probability tensor $G \in (0, 1]^{m \times n \times 3}$ is calculated at each step, and the expected alignment matrix is obtained via backward recursion $E_{i, j} = G_{i, j+1, 1}E_{i, j+1} + G_{i+1, j, 2}E_{i+1, j} + G_{i+1, j+1, 3}E_{i+1, j+1}$. The authors note that local DP is technically a **computable approximation** of global ns-DTW rather than an exact value, as GEV operators do not maintain distribution family closure under nesting in the same way log-sum-exp does.

### Loss & Training
ns-DTW is used as a differentiable sequence similarity measure. In barycenter computation, the objective is to minimize $\min_x \sum_i \text{ns-DTW}(C(x, y_i))$ (Fréchet mean). Classification uses Nearest Centroid Classifier (NCC) and 1NN; clustering uses k-means. Key hyperparameters include temperature $\gamma \in \{0.1, 0.01, 0.001, 0.0001\}$, similarity parameter $\tau \in \{0.80, 0.85, 0.90, 0.95\}$, and grouping schemes $\{g_1, g_2, g_3\}$, all selected via cross-validation.

## Key Experimental Results

Experiments were conducted on 47 subsets of the UCR time series archive for barycenter, classification, and clustering tasks. Baselines include Subgradient methods, DBA, and soft-DTW.

### Main Results: Barycenter Computation (Lower DTW loss)

Percentage of datasets where ns-DTW (with optimal parameters) achieved lower DTW loss than competitors:

| $\gamma$ | vs Subgradient | vs DBA | vs soft-DTW |
|----------|----------------|--------|-------------|
| 0.1 | 68.09% | 46.81% | 36.17% |
| 0.01 | 80.85% | 72.34% | 59.57% |
| 0.001 | 95.74% | 87.23% | 80.85% |
| 0.0001 | 100.00% | 91.49% | 91.49% |

Under optimal $\gamma$ settings, ns-DTW outperforms Subgradient on 100%, DBA on 97.87%, and soft-DTW on 74.47% of datasets.

### Classification and Clustering (Percentage of datasets where ns-DTW $\ge$ baseline accuracy)

| Task | vs Subgradient | vs DBA | vs soft-DTW |
|------|----------------|--------|-------------|
| NCC Classification | 93.02% | 88.37% | 86.05% |
| 1NN Classification | — | 88.37% | 86.05% |
| k-means Clustering | — | 100.00% | 76.60% |

### Key Findings
- **Smaller $\gamma$ increases ns-DTW's advantage**: As $\gamma$ decreases from 0.1 to 0.0001, the win rate against soft-DTW rises from 36% to 91%.
- **Skewness (controlled by $\tau$ and grouping) is the core Gain source**: As $\tau \to 0$, the model prunes high-cost deviations and pushes the expected path toward lower-cost directions.
- **Grouping schemes $g_1, g_2, g_3$ provide directional flexibility**: This allows the method to adapt to different alignment structures.

## Highlights & Insights
- **Upgrading soft-DTW from "heuristic smoothing" to "special case of a probabilistic model"**: Proving soft-DTW $= \mathbb{E}[\text{DTW}(\tilde C_\gamma)]$ provides an elegant explanation for the log-sum-exp form.
- **Random Utility Theory bridge**: Mapping alignment selection to discrete choice problems allows tools from economics (Gumbel $\to$ logit, GEV $\to$ nested logit) to be applied to differentiable combinatorial optimization.
- **Distinction between theory and computability**: The authors honestly acknowledge that the DP implementation is an approximation of the global GEV definition, providing a clear path for future research.

## Limitations & Future Work
- **Ours' DP is an approximation**: The gap between the recursive $V_{m, n}$ and the theoretical ns-DTW is not quantitatively characterized.
- **Large hyperparameter search space**: Grouping schemes, $\tau$, and $\gamma$ require joint cross-validation, increasing deployment costs.
- **Classic task focus**: The experiments are limited to UCR barycenter/classification/clustering, without directly demonstrating gains in end-to-end deep learning training scenarios.

## Related Work & Insights
- **vs soft-DTW (Cuturi & Blondel, 2017)**: Ours replaces heuristic smoothing with a probabilistic framework, offering higher flexibility through GEV noise at the cost of additional hyperparameters.
- **vs Perturbed Optimizers (Berthet et al., 2020)**: This paper specifically applies their "noise + expectation" framework to the structured alignment of DTW.
- **vs Differentiable DP (Mensch & Blondel, 2018)**: While they use entropy-regularized DP, this paper generalizes the regularization from Shannon entropy to nested Shannon entropy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] End-to-End Probabilistic Framework for Learning with Hard Constraints](end-to-end_probabilistic_framework_for_learning_with_hard_constraints.md)
- [\[ICML 2026\] Dynamic-TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series](../../ICML2026/time_series/dynamic_tmoe_a_drift-aware_dynamic_mixture_of_experts_framework_for_non-stationa.md)
- [\[NeurIPS 2025\] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting](../../NeurIPS2025/time_series/timeperceiver_an_encoder-decoder_framework_for_generalized_time-series_forecasti.md)
- [\[ICLR 2026\] From Samples to Scenarios: A New Paradigm for Probabilistic Forecasting](from_samples_to_scenarios_a_new_paradigm_for_probabilistic_forecasting.md)
- [\[ICLR 2026\] Reliable Probabilistic Forecasting of Irregular Time Series via Marginal Consistent Flows](reliable_probabilistic_forecasting_of_irregular_time_series_through_marginalizat.md)

</div>

<!-- RELATED:END -->
