---
title: >-
  [Paper Note] On Optimal Steering to Achieve Exact Fairness
description: >-
  [NeurIPS 2025][Image Generation][fairness] This paper defines the concept of an *ideal distribution*—a data distribution under which the Bayes-optimal classifier for any cost-sensitive risk satisfies exact fairness—and p…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "fairness"
  - "optimal-transport"
  - "distribution-steering"
  - "LLM-representation"
  - "Bayes-optimal"
date: 2026-05-08
content_hash: 34b55a6a7bc0aa93
---

# On Optimal Steering to Achieve Exact Fairness

**Conference**: NeurIPS 2025
**arXiv**: [2509.15759](https://arxiv.org/abs/2509.15759)
**Code**: None
**Area**: Image Generation
**Keywords**: fairness, optimal-transport, distribution-steering, LLM-representation, Bayes-optimal

## TL;DR

This paper defines the concept of an *ideal distribution*—a data distribution under which the Bayes-optimal classifier for any cost-sensitive risk satisfies exact fairness—and proposes an optimization framework that identifies the nearest ideal distribution via KL divergence minimization, providing provable fairness guarantees for both fair preprocessing and LLM representation steering.

## Background & Motivation

1. **"Biased input, biased output" problem**: The root cause of unfair outcomes in ML systems lies in biases inherent to the training data. Imposing fairness constraints during training on biased data does not guarantee fairness on unbiased test sets, and post-processing based on biased validation data is equally unreliable.
2. **Limitations of fair preprocessing**: The classical reweighting method of Kamiran & Calders (2007) adjusts only class–group prior probabilities while ignoring feature distributions, and thus cannot provide fairness guarantees when combined with accuracy maximization.
3. **Lack of provable guarantees in prior work**: The preprocessing framework of Calmon et al. (2017) may be infeasible when group fairness and individual fairness are incompatible; Dutta et al. (2020) use the Chernoff Information gap as a surrogate measure, which cannot be directly translated into standard fairness metrics such as DP or EO.
4. **Absence of the ideal distribution concept**: Prior work focuses on *fair classifiers* (constrained optimal models given a fixed distribution) rather than formally defining distributions that are naturally fair for arbitrary downstream tasks, leaving this level of theory unaddressed.
5. **Rise of LLM representation steering**: Steerable internal representations in LLMs have attracted growing research interest, yet the theoretical foundations remain weak, underscoring the need for provable fairness guarantees to guide representation transformations.
6. **Tractability of parametric conditions**: When group–class conditional distributions belong to common parametric families (e.g., Gaussian, log-normal), the conditions for an ideal distribution reduce to parameter constraints, rendering the optimization feasible.

## Method

### Definition of the Ideal Distribution (Definition 3.1)

For group-aware classifiers $h: \mathcal{X} \times \mathcal{A} \to \mathcal{Y}$ in a hypothesis class $\mathcal{H}$, a distribution $D$ is **ideal** if and only if, for every cost matrix $C \in \mathbb{R}^{|\mathcal{Y}| \times |\mathcal{Y}|}$, the Bayes-optimal classifier $h^*_C$ with respect to the $C$-cost-sensitive risk on $D$ satisfies exact fairness (e.g., $\Delta\text{DP} = 0$ or $\Delta\text{EO} = 0$).

**Key property**: An ideal distribution admits no fairness–accuracy trade-off, since the optimal classifier is itself exactly fair.

### Parametric Conditions (Propositions 3.2 & 3.3)

When $X \mid Y=i, A=a \sim \mathcal{N}(\mu_{ia}, \Sigma_{ia})$ is multivariate Gaussian, sufficient conditions for an ideal distribution are:

1. **Consistent standardized mean differences**: $\Sigma_{ia}^{-1/2}(\mu_{ia} - \mu_{ja}) = \Sigma_{ia'}^{-1/2}(\mu_{ia'} - \mu_{ja'})$
2. **Consistent covariance ratios**: $\Sigma_{ia}^{1/2} \Sigma_{ja}^{-1} \Sigma_{ia}^{1/2} = \Sigma_{ia'}^{1/2} \Sigma_{ja'}^{-1} \Sigma_{ia'}^{1/2}$
3. **Consistent class prior ratios**: $q_{ia}/q_{ja} = q_{ia'}/q_{ja'}$

For the univariate Gaussian binary classification and binary group setting (Proposition 3.3), these conditions are necessary and sufficient, and simultaneously guarantee DP, EO, and Equalized Odds.

### Optimization for the Nearest Ideal Distribution

**Objective**: $\min_{\tilde{D}:\,\text{ideal}} D_{\mathrm{KL}}(\tilde{D} \| D)$

This problem is non-convex in general, but admits efficient solutions under specific intervention strategies.

### Affirmative Action Intervention (Theorem 4.1 & Corollary 4.2)

**Modifying only the disadvantaged group's** distributional parameters:
- In the multivariate Gaussian setting the objective is convex (Theorem 4.1), enabling efficient optimization.
- Closed-form solution for the univariate Gaussian case (Corollary 4.2): $\tilde{\sigma}_{i0} = \gamma^* \sigma_{i1}$, $\tilde{\mu}_{10}$ is a weighted average, and $\tilde{\mu}_{00} = \tilde{\mu}_{10} + \gamma^*(\mu_{01} - \mu_{11})$.

### All-Subgroup Intervention (Proposition 4.3)

**Modifying all four subgroup** distributional parameters:
- The optimization problem is non-convex, but can be efficiently approximated via line search over the scaling factor $\gamma$.
- All updated parameters are expressible as functions of $\gamma^*$ and the original parameters.

### Theoretical Guarantees on Accuracy–Fairness (Proposition 4.4)

$$|\mathrm{err}(\tilde{h}, D) - \mathrm{err}(\tilde{h}, \tilde{D})| \leq \sqrt{2\, D_{\mathrm{KL}}(\tilde{D}, D)}$$

$$\Delta_{\mathrm{EO}}(\tilde{h}, D) \leq \sqrt{8\, D_{\mathrm{KL}}(\tilde{D} \| D)}$$

When the ideal distribution is sufficiently close to the original, both accuracy loss and residual unfairness remain bounded.

### LLM Representation Steering

1. **Multi-class fair classification (Bios dataset)**: Using Llama-2 7B embeddings, the first two moments of each subgroup (occupation × gender) are estimated; target moments for the ideal distribution are computed; and representations are steered via an affine transformation.
2. **Sentiment steering (GCS framework)**: Gaussian Concept Steering is applied on Llama-3 8B; the EF intervention is applied to the steering vector of the disadvantaged group (horror film reviews) to narrow the inter-group gap in "joyfulness" scores.

## Key Experimental Results

### Table 1: Univariate Gaussian Intervention Comparison (Synthetic Data)

| Intervention | Bayes Error (↓) | ΔDP (↓) | ΔEO (↓) | KL Distance (↓) |
|---|---|---|---|---|
| Original distribution | 0.15 | 0.25 | 0.18 | 0 |
| EF Affirmative (disadvantaged group only) | 0.12 | 0 | 0 | Small |
| EF All Subgroups (all groups) | 0.11 | 0 | 0 | Smallest |
| Mean Matching | 0.14 | ~0.05 | ~0.03 | Medium |

**Key observation**: EF All Subgroups achieves perfect fairness and the lowest Bayes error while remaining closest to the original distribution; mean matching alone cannot guarantee exact fairness.

### Table 2: Multi-class Classification TPR-gap on Bios Dataset (Llama-2 7B Embeddings)

| Method | Mean Accuracy | Mean RMS TPR-gap (↓) |
|---|---|---|
| Original (no intervention) | 0.79 | High |
| LEACE (Belrose et al.) | 0.77 | Medium |
| MiMiC (Singh et al.) | 0.78 | Medium-low |
| EF Affirmative (Ours) | 0.78 | Lowest |

**Key observation**: The proposed method consistently reduces the TPR-gap across all occupational categories while maintaining accuracy in the range 0.77–0.79, outperforming or matching MiMiC and LEACE on most categories.

### Sentiment Steering Experiment

Steering movie review generation toward "joyful" sentiment on Llama-3 8B:
- The Comedy group responds well to baseline steering; the Horror group does not.
- Applying the EF intervention to the Horror group at moderate $\alpha$ values (e.g., 0.03–0.05) substantially improves its $\Delta\text{-Joyful}$ score, narrowing the gap with the Comedy group.
- Excessively large $\alpha$ distorts the steering vector, degrading performance.

## Highlights & Insights

- **Formal definition of the ideal distribution**: This work is the first to define a data distribution under which the Bayes-optimal classifier for arbitrary cost-sensitive risk is exactly fair, elegantly circumventing the constraints of impossibility theorems.
- **Addressing fairness at the data source**: Rather than constraining model training, fairness is guaranteed at the distributional level, so that any subsequent reasonable model inherits the fairness properties.
- **Efficient algorithms and closed-form solutions**: The affirmative action intervention reduces to convex optimization in the multivariate Gaussian setting and admits a closed-form solution in the univariate case, making the approach practically viable.
- **Transferable theoretical guarantees**: Proposition 4.4 provides upper bounds on accuracy and fairness deviation on the original distribution for a classifier trained on the ideal distribution, bridging theory and practice.
- **Theoretically grounded LLM representation steering**: Applying ideal distribution theory to affine steering of LLM internal representations provides provable guarantees for representation steering, a domain that has hitherto lacked rigorous theoretical foundations.

## Limitations & Future Work

- **Parametric assumptions**: The theoretical results rely on the assumption that group–class conditional distributions belong to parametric families such as the Gaussian, which real-world data may not satisfy.
- **Gap from Bayes optimality**: Practical models are not Bayes-optimal classifiers, creating a gap between the theoretical guarantees and empirical performance.
- **Global optimality of non-convex optimization**: The line search for the all-subgroup intervention (Proposition 4.3) yields only an approximate global optimum without formal global optimality guarantees.
- **Scalability to multiple classes and groups**: Although the framework supports multiple classes and groups, the number of parameters and constraints in the optimization problem grows rapidly with the number of class–group pairs.
- **Finite-sample setting**: The theory operates at the population level; how parameter estimation errors under finite samples propagate to fairness guarantees has not been thoroughly analyzed.

## Related Work & Insights

| Dimension | Ours | Dutta et al. (2020) |
|---|---|---|
| **Objective** | Find the nearest ideal distribution such that the Bayes-optimal classifier is exactly fair | Find the nearest distribution with zero Chernoff Information gap |
| **Fairness metric** | Directly guarantees DP/EO/Equalized Odds | Relates to fairness metrics indirectly via an information-theoretic surrogate |
| **Tractability** | Affirmative action intervention is convex with a closed-form solution | Efficiency of the optimization problem is unclear |
| **Applicability** | Parametric conditions correspond directly to standard fairness metrics | Chernoff Information cannot be directly translated into DP/EO |

| Dimension | Ours | Singh et al. (2024) MiMiC |
|---|---|---|
| **Method** | Affine steering based on ideal distribution theory with provable fairness guarantees | Affine steering via least-squares moment matching; empirically effective |
| **Theoretical guarantee** | Bayes-optimal classifier is exactly fair on the ideal distribution | No explicit theoretical fairness guarantee |
| **Intervention strategy** | Modifies only the disadvantaged group or all groups, as determined by the optimization | Matches the first two moments across groups |
| **Empirical performance** | Consistently lower or comparable TPR-gap on the Bios dataset | Reduces TPR-gap but less consistently than the proposed method |

| Dimension | Ours | Kamiran & Calders (2007) |
|---|---|---|
| **Intervention level** | Simultaneously adjusts prior probabilities and feature distribution parameters | Reweights prior probabilities $q_{ia}$ only |
| **Fairness guarantee** | Exact fairness for arbitrary cost-sensitive risk | No theoretical fairness guarantee |
| **Relationship** | The proposed conditions subsume K&C reweighting (Remark 3.4); can be viewed as the second stage of a two-stage approach | A special case of the proposed method (prior component only) |

## Rating

| Dimension | Score | Comment |
|---|---|---|
| Novelty | ⭐⭐⭐⭐⭐ | First to formalize the concept of the ideal distribution with parametric necessary and sufficient conditions; the theoretical framework is highly original |
| Theoretical Depth | ⭐⭐⭐⭐ | Parametric condition derivations are rigorous; convexity proofs are complete; the KL-to-fairness transfer bound is valuable |
| Experimental Thoroughness | ⭐⭐⭐ | Synthetic data analysis is thorough, but real-data experiments are limited (only Bios and sentiment steering); large-scale benchmark comparisons are absent |
| Value | ⭐⭐⭐⭐ | Closed-form solutions and convex optimization make the method directly applicable to representation steering, though parametric assumptions limit generality |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[NeurIPS 2025\] LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss](lineas_end-to-end_learning_of_activation_steering_with_a_distributional_loss.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
