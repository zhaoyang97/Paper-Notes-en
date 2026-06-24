---
title: >-
  [Paper Note] Sufficient Invariant Learning for Distribution Shift
description: >-
  [CVPR 2025][Distribution Shift] This paper proposes the Sufficient Invariant Learning (SIL) framework to improve robustness under distribution shift by learning diverse subsets of invariant features instead of a single invariant feature. It designs the ASGDRO algorithm to implement SIL by seeking a common flat minimum across environments, achieving SOTA performance on multiple distribution shift benchmarks.
tags:
  - "CVPR 2025"
  - "Distribution Shift"
  - "Invariant Feature Learning"
  - "Robust Optimization"
  - "Flat Minima"
  - "Domain Generalization"
date: 2026-05-08
content_hash: c0081795cdf9419e
---

# Sufficient Invariant Learning for Distribution Shift

**Conference**: CVPR 2025  
**arXiv**: [2210.13533](https://arxiv.org/abs/2210.13533)  
**Code**: [https://github.com/MLAI-Yonsei/SIL-ASGDRO](https://github.com/MLAI-Yonsei/SIL-ASGDRO)  
**Area**: LLM Evaluation  
**Keywords**: Distribution Shift, Invariant Feature Learning, Robust Optimization, Flat Minima, Domain Generalization

## TL;DR

This paper proposes the Sufficient Invariant Learning (SIL) framework to improve robustness under distribution shift by learning diverse subsets of invariant features instead of a single invariant feature. It designs the ASGDRO algorithm to implement SIL by seeking a common flat minimum across environments, achieving SOTA performance on multiple distribution shift benchmarks.

## Background & Motivation

In real-world applications, mismatches between the distributions of training and testing data (distribution shift) often lead to significant degradation in model performance. Invariant learning is a mainstream approach to address distribution shift, with the core idea being to identify invariant features that remain consistent across environments. However, existing invariant learning methods suffer from a critical assumption flaw: they assume that the invariant features learned during training are fully visible in the test environment.

In practice, models might only learn a subset of invariant features (e.g., a bird's "feet"), which may be unobservable in test environments (e.g., when a bird is standing in water, obscuring its feet). In such cases, models relying solely on a single invariant feature will fail. The key insight of this work is: **models should learn a sufficiently diverse set of invariant features rather than relying on just one**. Even if some invariant features are missing, the model can still make reliable predictions using the remaining ones.

Based on this motivation, the authors propose the Sufficient Invariant Learning (SIL) framework and design the ASGDRO algorithm, which encourages models to learn diverse invariant mechanisms by finding common flat minima across environments.

## Method

### Overall Architecture

The method is built upon the problem formulation of invariant learning. The model $f = h \circ g$ consists of an encoder $g$ and a classifier $h$. Input features can be decomposed into invariant features $Z^I$ (consistent across environments) and spurious (non-invariant) features $Z^{NI}$ (varying across environments). The SIL framework requires the classifier to remain robust across any subset of invariant features, while ASGDRO achieves this goal by combining adaptive sharpness-aware optimization with group distributionally robust optimization (GDRO).

### Key Designs

1. **Sufficient Invariant Learning (SIL) Framework**:
    - Function: Defines a new learning principle requiring that the model not only performs well across all environments but also makes accurate predictions using any subset of invariant features.
    - Mechanism: Treating invariant features as a set $Z^I = \{Z_1^I, ..., Z_p^I\}$, SIL requires the optimal classifier to satisfy $\min_{\theta_h} \max_{e \in \mathcal{E}} \max_{\hat{Z}^I \subseteq Z^I} \mathbb{E}[\ell(h_{\theta_h}(\hat{Z}^I), Y^e)]$.
    - Design Motivation: Existing invariant learning methods may converge after learning just a single invariant feature because the solution that minimizes the worst-case environmental loss is not unique. SIL ensures that the model utilizes diverse invariant features through additional constraints.

2. **ASGDRO Algorithm (Adaptive Sharpness-Aware Group Distributionally Robust Optimization)**:
    - Function: Acts as the implementation algorithm for SIL, learning diverse invariant mechanisms by jointly optimizing sharpness and distributional robustness.
    - Mechanism: The objective function is $\max_{e \in \mathcal{E}_{tr}} \max_{\|\epsilon_e\| \leq \rho} \mathcal{R}^e(\theta + \epsilon_e)$, which searches for the maximum loss within a $\rho$-neighborhood in the parameter space for each environment, and then takes the worst outcome across all environments.
    - Design Motivation: Drawing insights from model ensemble and multi-task learning, the SIL solution $\theta^{SI}$ is hypothesized to lie within the linear interpolation of individual invariant mechanisms $\theta_i^I$. Optimizing in flat regions of the parameter space allows covering multiple invariant mechanisms simultaneously.

3. **Common Flat Minima Theory**:
    - Function: Explains theoretically why ASGDRO can realize SIL.
    - Mechanism: Theorem 1 proves that under a linear model, the optimal solution of ASGDRO will use all invariant mechanisms uniformly ($\lambda^* = (1/p, ..., 1/p)$). Proposition 1 shows that ASGDRO drives the model to converge to a common flat minimum by regularizing the gradient norm $\|\nabla \mathcal{R}^e(\theta)\|$.
    - Design Motivation: A flat minimum implies that loss changes minimally when parameters are perturbed within their neighborhood, which directly corresponds to the model's ability to maintain stable performance across multiple invariant mechanisms.

### Loss & Training

The training process of ASGDRO is as follows:
1. For each environment $e$, compute the adaptive sharpness perturbation $\epsilon_e^* = \rho \frac{T_\theta^2 \nabla \mathcal{R}^e(\theta)}{\|T_\theta \nabla \mathcal{R}^e(\theta)\|}$, where $T_\theta$ is a normalization matrix used to eliminate scale symmetry.
2. Compute the loss for each environment at the perturbed parameters $\theta + \epsilon_e^*$.
3. Update environmental weights using exponential weighting: $\lambda_e^{(t)} = \lambda_e^{(t-1)} \exp(\gamma \mathcal{R}^e(\theta_t^*))$ and normalize them.
4. Compute the gradient of the weighted loss to update model parameters.

In practice, for computational efficiency, a common perturbation $\epsilon^*$ is used instead of calculating separate perturbations for each environment.

## Key Experimental Results

### Main Results

| Dataset | Metric | ASGDRO | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| CMNIST | Worst Acc | 74.2% | 73.3% (LISA) | +0.9% |
| Waterbirds | Worst Acc | 91.4% | 90.6% (GDRO) | +0.8% |
| CelebA | Worst Acc | 91.0% | 89.3% (LISA) | +1.7% |
| CivilComments | Worst Acc | 71.8% | 72.6% (LISA) | -0.8% |
| DomainBed Avg | Avg Acc | 65.9% | 65.1% (GSAM) | +0.8% |
| H-CMNIST TestBed2 (Shape) | Acc | 69.17% | 61.44% (GDRO) | +7.73% |

Wilds Benchmark (no pre-training):

| Dataset | Metric | ASGDRO | GDRO | Gain |
|--------|------|--------|------|------|
| Camelyon17 | Avg Acc | 81.0% | 68.4% | +12.6% |
| CivilComments | Worst Acc | 71.8% | 70.0% | +1.8% |
| FMoW | Worst Acc | 35.0% | 30.8% | +4.2% |
| Amazon | 10th perc. | 54.5% | 53.3% | +1.2% |
| RxRx1 | Avg Acc | 32.2% | 23.0% | +9.2% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| GDRO (no flatness constraint) | 61.44% (H-CMNIST Shape) | Unable to learn multiple invariant features |
| ASAM (no robust optimization) | 57.07% (H-CMNIST Shape) | Unable to eliminate spurious features |
| ERM (baseline) | 57.41% (H-CMNIST Shape) | Learns only simple invariant features |
| ASGDRO | 69.17% (H-CMNIST Shape) | Learns both color and shape invariant features simultaneously |

### Key Findings

- Although GDRO can eliminate spurious correlations, it fails to learn multiple invariant features, and while ASAM considers flatness, it cannot eliminate spurious correlations—ASGDRO combines the strengths of both.
- Grad-CAM analysis demonstrates that ASGDRO focuses on diverse invariant feature regions (e.g., bird's head, wings, and tail) rather than focusing on a single region.
- Hessian analysis shows that ASGDRO finds lower eigenvalues (flatter minima) across all groups, with minimal differences between groups.
- On the Wilds benchmark, ASGDRO outperforms GDRO and IRM with rich representation pre-training without requiring any pre-training.

## Highlights & Insights

- **Invariant Learning from a Diversity Perspective**: Re-examines invariant learning from the perspective of "sufficiency," pointing out that merely eliminating spurious correlations is insufficient and that learning sufficiently diverse invariant features is also necessary, which represents a significant theoretical contribution.
- **Elegant Integration of SAM and DRO**: Organically combines sharpness-aware optimization and distributionally robust optimization, allowing flatness constraints to serve the learning of diverse invariant mechanisms.
- **H-CMNIST Benchmark**: Designs a new dataset specifically for evaluating whether diverse invariant features are learned, containing both color and shape invariant features, which directly validates the effectiveness of SIL.
- **Model Agnosticism**: ASGDRO can be combined with other methods such as DPLCLIP, demonstrating excellent plug-and-play characteristics.

## Limitations & Future Work

- The theoretical analysis is primarily based on linear model assumptions; behaviors in non-linear deep networks require more in-depth research.
- The number of invariant features $p$ is unknown in practice, and the selection of the hyperparameter $\rho$ relies on heuristics.
- Computational overhead is increased compared to ERM (requiring the computation of gradient perturbations for each environment).
- For textual datasets like CivilComments, ASGDRO fails to outperform LISA, potentially due to the different structure of invariant features in textual domains.
- Future directions include exploring methods to adaptively determine $\rho$ and extending SIL to a wider range of task setups.

## Related Work & Insights

- **vs GDRO**: GDRO only minimizes the worst-case group loss without considering flatness, potentially converging to sharp minima that utilize only a subset of invariant features. ASGDRO encourages diverse invariant mechanisms through flatness constraints.
- **vs SAM/ASAM**: SAM only considers the flatness of the overall loss across environments; ASGDRO independently imposes sharpness constraints on each environment.
- **vs LISA**: LISA enhances minority groups through mixup-based sampling, which belongs to data augmentation strategies, whereas ASGDRO addresses the problem at the optimization objective level.
- **vs SWAD**: SWAD searches for flat minima using weight averaging but may fail in the presence of strong spurious correlations. The DRO component of ASGDRO ensures the elimination of spurious correlations.

## Rating

- Novelty: ⭐⭐⭐⭐ Redefines the goal of invariant learning from a "sufficiency" perspective, offering a fresh theoretical angle.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes a toy example, the new H-CMNIST benchmark, multiple standard benchmarks, and comprehensive analyses.
- Writing Quality: ⭐⭐⭐⭐ Derivations are clear, problem motivations are well-articulated, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Provides a fresh perspective and practical algorithms for robust learning under distribution shift.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Open Set Label Shift with Test Time Out-of-Distribution Reference](open_set_label_shift_with_test_time_out-of-distribution_reference.md)
- [\[ICML 2026\] Target-Agnostic Calibration under Distribution Shift with Frequency-Aware Gradient Rectification](../../ICML2026/others/target-agnostic_calibration_under_distribution_shift_with_frequency-aware_gradie.md)
- [\[CVPR 2025\] HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition](hotspot_signed_distance_function_optimization_with_an_asymptotically_sufficient_.md)
- [\[ICML 2026\] Learning Permutation-Invariant Macroscopic Dynamics](../../ICML2026/others/learning_permutation-invariant_macroscopic_dynamics.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](../../ICML2026/others/continual_learning_of_domain-invariant_representations.md)

</div>

<!-- RELATED:END -->
