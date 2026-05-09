---
title: >-
  [Paper Note] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks
description: >-
  [AAAI 2026 (Oral)][AI Safety][hard-label attacks] To address the query efficiency bottleneck in hard-label black-box adversarial attacks, this paper proposes ARS-OPT, a momentum-based algorithm grounded in Nesterov Accelerated Gradient (NAG), and its enhanced variant PARS-OPT that incorporates surrogate model priors. Theoretical convergence guarantees are established, and both methods outperform 13 state-of-the-art approaches on ImageNet and CIFAR-10.
tags:
  - AAAI 2026 (Oral)
  - AI Safety
  - hard-label attacks
  - black-box adversarial examples
  - query efficiency
  - momentum acceleration
  - surrogate models
date: 2026-05-08
content_hash: c305a894051a8be1
---

# Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks

**Conference**: AAAI 2026 (Oral)
**arXiv**: [2512.21241](https://arxiv.org/abs/2512.21241)
**Code**: None
**Area**: AI Security / Adversarial Attacks
**Keywords**: hard-label attacks, black-box adversarial examples, query efficiency, momentum acceleration, surrogate models

## TL;DR

To address the query efficiency bottleneck in hard-label black-box adversarial attacks, this paper proposes ARS-OPT, a momentum-based algorithm grounded in Nesterov Accelerated Gradient (NAG), and its enhanced variant PARS-OPT that incorporates surrogate model priors. Theoretical convergence guarantees are established, and both methods outperform 13 state-of-the-art approaches on ImageNet and CIFAR-10.

## Background & Motivation

**Background**: Adversarial attack research is a central component of AI security. Among black-box attacks, the hard-label setting—where only the top-1 predicted label is accessible—represents the most practically relevant and challenging scenario. Ray search-based methods, which identify the optimal direction to minimize the $\ell_2$ norm of adversarial perturbations, are representative approaches in this setting.

**Limitations of Prior Work**: The fundamental obstacle in hard-label attacks is the extremely high query complexity. Because only discrete label information (rather than continuous gradients) is available, estimating gradients requires a large number of queries per step. Existing ray search optimization methods converge slowly, often requiring thousands to tens of thousands of queries to find sufficiently small adversarial perturbations.

**Key Challenge**: Minimal information (labels only) versus a vast search space (high-dimensional image space)—efficient navigation of a high-dimensional space is required under severely limited feedback.

**Goal**: (1) Improve the convergence rate of ray search optimization; (2) reduce the number of queries required to achieve equivalent attack performance; (3) provide theoretical convergence guarantees.

**Key Insight**: Drawing from optimization theory, the paper introduces the momentum concept of Nesterov Accelerated Gradient (NAG) into ray search—leveraging accumulated momentum to anticipate future gradient directions and accelerate convergence.

**Core Idea**: ARS-OPT performs lookahead gradient estimation along momentum-predicted future directions, while PARS-OPT further incorporates surrogate model priors, achieving faster convergence and reduced query counts.

## Method

### Overall Architecture

Given a benign image $x$ and a target model $f$ (returning only the top-1 label), the objective is to find a minimal $\ell_2$-norm perturbation $\delta$ such that $f(x + \delta) \neq f(x)$. The problem is reformulated as searching for the optimal ray direction on the unit sphere—each direction defines a ray from $x$, along which the nearest point on the decision boundary is located. The optimization goal is to find the direction that minimizes the distance from $x$ to the boundary.

### Key Designs

1. **Accelerated Ray Search with Momentum (ARS-OPT)**:

    - **Function**: Exploits historical gradient information to accelerate optimization of the current direction.
    - **Mechanism**: Inspired by NAG, at each step the method does not estimate the gradient at the current direction directly. Instead, it uses accumulated momentum to predict a "lookahead" direction and estimates the gradient there. Specifically, given current direction $d_t$ and momentum $m_t$, the gradient is estimated at the lookahead direction $d_t + \beta m_t$, enabling gradient estimates that better reflect the optimization trajectory.
    - **Design Motivation**: Standard ray search relies solely on current information at each step, which can cause oscillations in regions of high curvature. Momentum accumulates historical trend information, and lookahead gradient estimation yields smoother optimization paths and faster convergence. Theoretical analysis is provided showing that ARS-OPT achieves a superior convergence rate compared to standard ray search.

2. **Prior-Augmented Ray Search with Surrogate Models (PARS-OPT)**:

    - **Function**: Leverages gradient information from a surrogate model (e.g., ResNet) to further accelerate search.
    - **Mechanism**: The gradient estimation in ARS-OPT is augmented with surrogate model gradients as a prior. Although the surrogate model differs from the target model, its gradient directions typically transfer, providing prior knowledge about adversarial directions. In practice, surrogate gradients and finite-difference gradient estimates are combined via weighted averaging.
    - **Design Motivation**: Pure finite-difference gradient estimation is noisy and query-intensive. Surrogate model priors supply low-noise directional guidance, reducing the number of queries required for finite-difference estimation.

3. **Theoretical Convergence Analysis**:

    - **Function**: Provides mathematical guarantees for the superiority of the proposed methods.
    - **Mechanism**: Under standard assumptions, convergence rate upper bounds are derived for both ARS-OPT and PARS-OPT, proving that both outperform momentum-free ray search methods. The analysis covers the effects of momentum parameter selection and surrogate prior weighting on convergence.
    - **Design Motivation**: Many hard-label attack methods rely on empirical tuning without theoretical guidance. Methods with formal guarantees are more reliable and interpretable.

### Loss & Training

The optimization objective is to minimize the boundary distance along ray directions: $\min_{d \in \mathbb{S}^{n-1}} g(d)$, where $g(d)$ denotes the boundary distance corresponding to direction $d$. Gradients of $g$ are estimated via finite differences. The momentum parameter $\beta$ and learning rate are determined through theoretical analysis.

## Key Experimental Results

### Main Results

| Dataset | Metric | PARS-OPT | Best Baseline | # Methods Compared |
|---------|--------|----------|---------------|-------------------|
| ImageNet | Query efficiency ($\ell_2$ perturbation) | Best | 2nd best | Outperforms 13 SOTA |
| CIFAR-10 | Query efficiency ($\ell_2$ perturbation) | Best | 2nd best | Outperforms 13 SOTA |

### Ablation Study

| Configuration | Query Efficiency | Description |
|---------------|-----------------|-------------|
| PARS-OPT | Best | Momentum + surrogate prior |
| ARS-OPT | 2nd best | Momentum only, no surrogate prior |
| Momentum-free ray search | Baseline | Standard method |
| Momentum-free + surrogate prior | Moderate | Surrogate prior helps but less than momentum |

### Key Findings

- Momentum acceleration is the primary driver of improved query efficiency—ARS-OPT shows significant gains over its momentum-free counterpart.
- Surrogate model priors further reduce query requirements; PARS-OPT achieves the best overall performance.
- Theoretical convergence rates are consistent with experimental results, validating the analysis.
- The proposed methods achieve top performance among 13 SOTA methods and were accepted as an Oral presentation.

## Highlights & Insights

- **Introducing NAG-style acceleration into hard-label attacks** represents an elegant integration of theory and practice—applying well-established optimization principles to improve practical attack efficiency.
- **Theoretical guarantees** are relatively rare in adversarial attack research, substantially enhancing the credibility and interpretability of the proposed methods.
- The methodology is broadly applicable—momentum-accelerated finite-difference gradient estimation can be extended to other black-box optimization problems.

## Limitations & Future Work

- The choice of surrogate model influences performance; when the surrogate and target models differ substantially, the prior may be unreliable.
- The current work focuses on $\ell_2$-norm attacks; applicability to $\ell_\infty$ attacks remains to be verified.
- Attack efficiency against defense models (e.g., adversarially trained models) has yet to be evaluated.
- Adaptive momentum scheduling—dynamically adjusting the momentum magnitude based on search history—warrants further exploration.

## Related Work & Insights

- **vs. SignOPT/HSJA**: These are classical ray search methods; the proposed approach builds upon them by introducing momentum acceleration.
- **vs. Transfer-based attacks**: Transfer attacks directly exploit surrogate model gradients, whereas this work incorporates surrogate gradients as priors within ray search, resulting in greater robustness.
- **vs. Query-based gradient estimation**: Standard finite-difference estimation requires many queries per step; momentum's accumulated historical information reduces the per-step query budget.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying NAG acceleration to hard-label attacks is novel, with rigorous theoretical analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Outperforms 13 methods across two datasets with theoretical support
- Writing Quality: ⭐⭐⭐⭐ Strong balance of theory and experiments; Oral-level quality
- Value: ⭐⭐⭐⭐ Significant contribution to AI security and adversarial robustness research

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICLR 2026\] Efficient Resource-Constrained Training of Transformers via Subspace Optimization](../../ICLR2026/ai_safety/efficient_resource-constrained_training_of_transformers_via_subspace_optimizatio.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](../../NeurIPS2025/ai_safety/differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[AAAI 2026\] Alternative Fairness and Accuracy Optimization in Criminal Justice](alternative_fairness_and_accuracy_optimization_in_criminal_j.md)

<!-- RELATED:END -->
