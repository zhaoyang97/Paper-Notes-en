---
title: >-
  [Paper Note] BECAME: BayEsian Continual Learning with Adaptive Model MErging
description: >-
  [ICML 2025][Model Compression][Continual Learning] This paper proposes BECAME, which reformulates the model merging mechanism based on Bayesian continual learning principles. It derives a closed-form solution for the optimal merging coefficient using Laplace approximation. Combining gradient projection (stability) and unconstrained training (plasticity) into a two-stage framework, it significantly outperforms SOTA on multiple continual learning benchmarks.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Continual Learning"
  - "Model Merging"
  - "Bayesian Learning"
  - "Gradient Projection"
  - "Stability-Plasticity Trade-off"
date: 2026-05-08
content_hash: be437585430f2f6c
---

# BECAME: BayEsian Continual Learning with Adaptive Model MErging

**Conference**: ICML 2025  
**arXiv**: [2504.02666](https://arxiv.org/abs/2504.02666)  
**Code**: [https://github.com/limei0818/BECAME](https://github.com/limei0818/BECAME)  
**Area**: Continual Learning  
**Keywords**: Continual Learning, Model Merging, Bayesian Learning, Gradient Projection, Stability-Plasticity Trade-off

## TL;DR
This paper proposes BECAME, which reformulates the model merging mechanism based on Bayesian continual learning principles. It derives a closed-form solution for the optimal merging coefficient using Laplace approximation. Combining gradient projection (stability) and unconstrained training (plasticity) into a two-stage framework, it significantly outperforms SOTA on multiple continual learning benchmarks.

## Background & Motivation

**Background**: Continual learning aims to enable models to learn incrementally on new tasks without catastrophically forgetting old tasks. The core challenge is the trade-off between stability (retaining old knowledge) and plasticity (learning new tasks).

**Limitations of Prior Work**:
   - Gradient projection methods (such as OGD, GPM, etc.) guarantee stability by constraining gradients to the orthogonal complement of the old feature space—but strict constraints limit plasticity.
   - Model merging methods (such as IMM, CoMA) perform a weighted average of old and new model parameters—but merging coefficients are either manually tuned or determined via simple heuristics, lacking theoretical foundations.
   - Existing merging methods assume that tasks are independent (independent Gaussian posteriors), ignoring sequential dependencies between tasks.

**Key Challenge**: Gradient projection sacrifices plasticity for stability, while unconstrained training sacrifices stability for plasticity—both of which are suboptimal. Model merging has the potential to combine the advantages of both, but where is the optimal merging point?

**Goal**: Theoretically derive the optimal merging coefficient and validate it empirically.

**Key Insight**: In Bayesian continual learning, the old model provides a prior distribution, and the new data provides likelihood—the MAP estimation of the posterior naturally defines the optimal merging point.

**Core Idea**: On the linear path between the gradient projection solution $\theta^{GP}$ (high stability) and the unconstrained solution $\hat{\theta}$ (high plasticity), there exists an optimal merging point $\theta^* = (1-\alpha)\theta^{GP} + \alpha \hat{\theta}$, where $\alpha$ has a closed-form solution based on Laplace approximation.

## Method

### Overall Architecture
The two-stage workflow of BECAME (for each new task):
1. **Stage 1 - Gradient Projection Training**: Obtains $\theta^{GP}$, which is stable but lacks plasticity.
2. **Stage 1.5 - Unconstrained Continued Training**: Trains without constraints starting from $\theta^{GP}$ to obtain $\hat{\theta}$ (high plasticity but forgets old tasks).
3. **Stage 2 - Adaptive Merging**: Computes the optimal $\alpha^*$ and merges to obtain $\theta^* = (1-\alpha^*)\theta^{GP} + \alpha^* \hat{\theta}$.

### Key Designs

1. **Bayesian Reconstruction of Model Merging**:

    - **Function**: Reformulates linear model merging as a MAP estimation of the Bayesian posterior.
    - **Mechanism**:
        - Old posterior $p(\theta | D_{\text{old}}) \approx \mathcal{N}(\theta^{GP}, H_{\text{old}}^{-1})$ (Laplace approximation)
        - New data likelihood $p(D_{\text{new}} | \theta)$
        - Merged posterior $p(\theta | D_{\text{all}}) \propto p(D_{\text{new}} | \theta) \cdot p(\theta | D_{\text{old}})$
    - Difference from Prior Work: IMM/CoMA assume that posteriors of tasks are independent $\rightarrow$ merging coefficients lack theoretical justification; BECAME is based on sequential Bayesian updates $\rightarrow$ the merging coefficient is naturally derived from the posterior MAP estimation.
    - **Design Motivation**: The Bayesian framework provides an optimality guarantee—the merged point maximizes the joint posterior of all tasks.

2. **Closed-Form Solution for the Optimal Merging Coefficient**:

    - **Function**: Identifies the optimal $\alpha^*$ on the linear path between $\theta^{GP}$ and $\hat{\theta}$.
    - **Mechanism**:
        - Objective: $\alpha^* = \arg\min_\alpha \mathcal{L}_{\text{all}}((1-\alpha)\theta^{GP} + \alpha \hat{\theta})$
        - Key Theorem: The cumulative loss along the linear path is a convex function of $\alpha$ (under the Laplace approximation).
        - Closed-form Solution: $\alpha^* = \frac{\delta^T H_{\text{old}} \delta + \nabla \mathcal{L}_{\text{new}}(\theta^{GP})^T \delta}{\delta^T (H_{\text{old}} + H_{\text{new}}) \delta}$
       where $\delta = \hat{\theta} - \theta^{GP}$ and $H$ is the Hessian.
    - **Design Motivation**: The closed-form solution eliminates hyperparameter searching—different tasks and different loss landscapes automatically produce different $\alpha^*$.

3. **Plasticity Existence Theorem**:

    - **Function**: Proves that the merged point is guaranteed to outperform both endpoints (theoretical guarantee).
    - **Mechanism**: Along the linear path, the gradients of the cumulative loss at the two endpoints point in opposite directions $\rightarrow$ there must be an extreme point in between.
    - **Design Motivation**: Theoretically guarantees that BECAME will not perform worse than using gradient projection or unconstrained training alone.

### Loss & Training
- Stage 1: Standard cross-entropy + gradient projection constraint
- Stage 1.5: Standard cross-entropy (unconstrained)
- Stage 2: Calculates $\alpha^*$ (closed-form solution, no training required)
- The Hessian is approximated by the diagonal elements of the Fisher Information Matrix (computationally efficient)
- Plug-and-play—can be seamlessly integrated into any gradient projection method

## Key Experimental Results

### Main Results
10-split CIFAR-100 (Class-IL setting):

| Method | Average Accuracy ↑ | Forgetting Rate ↓ | Final Task Plasticity ↑ |
|------|-----------|---------|---------------|
| OGD | 68.2 | 12.3 | 76.5 |
| GPM | 71.4 | 9.8 | 78.2 |
| NSCL | 73.1 | 8.5 | 79.8 |
| IMM (Equal-weight merging) | 72.8 | 9.1 | 81.3 |
| CoMA (Manual tuning) | 74.5 | 8.2 | 82.1 |
| **BECAME (Closed-form optimal merging)** | **77.3** | **6.9** | **85.7** |

### Multi-Benchmark Comparison

| Benchmark | BECAME Gain (vs Best Baseline) |
|------|----------------------|
| 10-split CIFAR-100 | +2.8% |
| 20-split CIFAR-100 | +3.5% |
| 5-split miniImageNet | +2.1% |
| 10-split TinyImageNet | +4.2% |

### Ablation Study

| Configuration | Accuracy (CIFAR-100) | Description |
|------|-------------------|------|
| Gradient projection only ($\alpha=0$) | 73.1 | High stability, low plasticity |
| Unconstrained only ($\alpha=1$) | 65.8 | High plasticity, low stability |
| Equal-weight merging ($\alpha=0.5$) | 74.2 | Manual but reasonable |
| **Closed-form optimal $\alpha^*$** | **77.3** | Adaptive optimal |
| $\alpha^*$ validated by grid search | 77.1 | Confirms that the closed-form solution is close to optimal |

### Key Findings
- The optimal $\alpha^*$ of BECAME varies across tasks—simpler tasks have larger $\alpha^*$ (more plasticity), while challenging tasks have smaller $\alpha^*$ (more stability).
- The plasticity metric shows the most significant improvement (+5.9%)—indicating that merging primarily releases the plasticity suppressed by gradient projection.
- The difference between the closed-form solution and grid search is <0.2%—validating the accuracy of the theoretical derivation.
- Consistently outperforms SOTA across all 4 benchmarks—demonstrating the high generalizability of the method.
- Although the Laplace approximation is an approximation, the closed-form solution is sufficiently accurate—a diagonal approximation of the Hessian suffices in practice.

## Highlights & Insights
- **Bayesian Derivation $\rightarrow$ Closed-form Merging Coefficient**—elevates model merging from "empirical hyperparameter tuning" to "theoretically guaranteed optimal solutions."
- Clever two-stage framework: first employs gradient projection to "delineate a safe boundary", then uses unconstrained training to "explore the limits", and finally achieves the "optimal balance point" via optimal merging.
- Excellent intuition behind the Plasticity Existence Theorem: the gradient directions of the two endpoints are opposite $\rightarrow$ there must be a better point in between.
- Plug-and-play design: any existing gradient projection method (such as OGD/GPM/NSCL) can immediately benefit.
- The diagonal Fisher approximation makes the method computationally almost indistinguishable from standard training.

## Limitations & Future Work
- The Laplace approximation assumes a Gaussian posterior—which may be inaccurate for complex multimodal posteriors.
- The diagonal Hessian approximation ignores the correlations between parameters.
- The linear merging path assumption may not be optimal—nonlinear paths (such as curve merging) might discover better merging points.
- Only validated on classification tasks—generative tasks and NLP tasks remain to be explored.
- Task boundaries must be known—the task-incremental setting might not apply to task-free scenarios.

## Related Work & Insights
- **vs IMM**: Equal-weight or manual tuning merging, lacking theoretical guarantees; BECAME provides Bayesian optimality guarantees and a closed-form solution.
- **vs CoMA**: Searches for merging coefficients via manual hyperparameter tuning; BECAME computes them automatically.
- **vs EWC**: Regularization methods directly constrain parameter changes; BECAME does not constrain first and then merges, offering greater flexibility.
- **vs PackNet/HAT**: Architectural methods allocate dedicated parameters for each task; BECAME shares parameters but performs adaptive merging.
- **Insight**: The application of Bayesian derivation to model merging can be generalized to any multi-model combination scenario (such as federated learning, model ensemble).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The Bayesian derivation of the closed-form merging coefficient possesses both theoretical depth and practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 benchmarks $\times$ multiple methods $\times$ extensive ablations $\times$ theoretical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly intuitive loss landscape visualizations.
- Value: ⭐⭐⭐⭐⭐ Provides a solid theoretical foundation for model merging in continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Rethinking the Stability-Plasticity Trade-off in Continual Learning from an Architectural Perspective](rethinking_the_stability-plasticity_trade-off_in_continual_learning_from_an_arch.md)
- [\[NeurIPS 2025\] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging](../../NeurIPS2025/model_compression/mingle_mixture_of_null-space_gated_low-rank_experts_for_test-time_continual_mode.md)
- [\[ICML 2025\] Bring Reason to Vision: Understanding Perception and Reasoning through Model Merging](bring_reason_to_vision_understanding_perception_and_reasoning_through_model_merg.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICML 2025\] TreeLoRA: Efficient Continual Learning via Layer-Wise LoRAs Guided by a Hierarchical Gradient-Similarity Tree](treelora_efficient_continual_learning_via_layer-wise_loras_guided_by_a_hierarchi.md)

</div>

<!-- RELATED:END -->
