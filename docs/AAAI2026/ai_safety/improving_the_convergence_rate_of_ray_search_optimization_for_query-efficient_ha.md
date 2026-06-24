---
title: >-
  [Paper Note] Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks
description: >-
  [AAAI 2026 (Oral)][AI Safety][Hard-label attack] Addressing the query efficiency bottleneck in hard-label black-box adversarial attacks, this paper proposes a momentum algorithm, ARS-OPT, based on Nesterov accelerated gradient, and introduces a surrogate model prior to obtain an enhanced version, PARS-OPT. It theoretically proves a faster convergence rate and outperforms 13 state-of-the-art (SOTA) methods on ImageNet and CIFAR-10.
tags:
  - "AAAI 2026 (Oral)"
  - "AI Safety"
  - "Hard-label attack"
  - "black-box adversarial"
  - "query efficiency"
  - "momentum acceleration"
  - "surrogate model"
date: 2026-05-08
content_hash: 6a715c83423637d9
---

# Improving the Convergence Rate of Ray Search Optimization for Query-Efficient Hard-Label Attacks

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2512.21241](https://arxiv.org/abs/2512.21241)  
**Code**: None  
**Area**: AI Security / Adversarial Attacks  
**Keywords**: Hard-label attack, black-box adversarial, query efficiency, momentum acceleration, surrogate model

## TL;DR

Addressing the query efficiency bottleneck in hard-label black-box adversarial attacks, this paper proposes a momentum algorithm, ARS-OPT, based on Nesterov accelerated gradient, and introduces a surrogate model prior to obtain an enhanced version, PARS-OPT. It theoretically proves a faster convergence rate and outperforms 13 state-of-the-art (SOTA) methods on ImageNet and CIFAR-10.

## Background & Motivation

**Background**: Adversarial attack research is an important component of AI security. In black-box attacks, the hard-label setting, which only allows access to the model's top-1 predicted label, represents the most realistic and challenging attack scenario. Ray-search-based methods, which search for the optimal direction to minimize the $\ell_2$ norm of adversarial perturbations, are representative approaches for hard-label attacks.

**Limitations of Prior Work**: The primary bottleneck of hard-label attacks is the extremely high query complexity. Because only discrete label information (rather than continuous gradients) is accessible, estimating the gradient at each step requires a large number of queries. Existing ray search optimization methods suffer from slow convergence, requiring thousands or even tens of thousands of queries to find sufficiently small adversarial perturbations.

**Key Challenge**: Extremely low information (labels only) vs. an immense search space (high-dimensional image space)—it requires efficient navigation of a high-dimensional space with extremely limited information.

**Goal**: (1) To improve the convergence rate of ray search optimization; (2) to reduce the number of queries required to achieve equivalent attack performance; and (3) to guarantee theoretical convergence.

**Key Insight**: Starting from optimization theory, the momentum concept of Nesterov Accelerated Gradient (NAG) is introduced into ray search—harnessing accumulated momentum to predict future gradient directions and accelerate convergence.

**Core Idea**: Use momentum-accelerated ray search optimization (ARS-OPT) to proactively estimate the gradient at future directions, and further accelerate this via a surrogate model prior (PARS-OPT) to achieve faster convergence and fewer queries.

## Method

### Overall Architecture

Attack workflow: Given a benign image $x$ and a target model $f$ (which only returns the top-1 label), the goal is to find a perturbation $\delta$ with the minimum $\ell_2$ norm such that $f(x + \delta) \neq f(x)$. The method formulates this as searching for the optimal ray direction on the unit sphere—each direction corresponds to a ray starting from $x$, along which the nearest point on the decision boundary is located. The optimization objective is to find the direction that minimizes the boundary distance to $x$.

### Key Designs

1. **Momentum-Accelerated Ray Search (ARS-OPT)**:

    - **Function**: Accelerating the optimization of the current direction using historical gradient information.
    - **Mechanism**: Inspired by NAG, instead of directly estimating the gradient of the current direction at each step, the accumulated momentum is first used to predict a "future" direction, and then the gradient is estimated at this future location. Specifically, if the current direction is $d_t$ and the momentum is $m_t$, the gradient is estimated at $d_t + \beta m_t$ (the look-ahead direction), allowing the gradient estimate to better reflect the optimization trajectory.
    - **Design Motivation**: Standard ray search makes decisions at each step based only on current information, which can easily lead to oscillations in areas with varying curvature. Momentum accumulates historical trend information, and the look-ahead gradient estimation renders the optimization path smoother and standardizes faster convergence. The authors provide a theoretical analysis proving that ARS-OPT enjoys a superior convergence rate compared to standard ray search.

2. **Surrogate Model Prior Enhancement (PARS-OPT)**:

    - **Function**: Accelerate the search process using gradient information from a surrogate model (e.g., ResNet).
    - **Mechanism**: Integrate the gradient of a surrogate model as a prior into the gradient estimation of ARS-OPT. Although the surrogate model differs from the target model, its gradient direction is typically transferable—providing prior knowledge about adversarial directions. This is realized by computing a weighted average of the surrogate gradient and the query-based gradient estimate.
    - **Design Motivation**: Query-based gradient estimation alone is highly noisy and query-expensive. The surrogate model prior offers low-noise directional guidance, thereby reducing the number of queries required for gradient estimation.

3. **Convergence Analysis**:

    - **Function**: Provide mathematical guarantees for the superiority of the proposed methods.
    - **Mechanism**: Derive the upper bounds of convergence rates for both ARS-OPT and PARS-OPT under standard assumptions, proving their superiority over non-momentum ray search methods. The analysis covers the impact of momentum parameter selection and surrogate prior weight on convergence.
    - **Design Motivation**: Many methods in the hard-label attack domain rely strictly on empirical hyperparameter tuning and lack theoretical guidance. Methods with theoretical guarantees are inherently more reliable.

### Loss & Training

The optimization objective is to minimize the boundary distance along the ray direction:

$$\min_{d \in \mathbb{S}^{n-1}} g(d)$$

where $g(d)$ is the boundary distance corresponding to direction $d$. The gradient of $g$ is estimated using query-based differences. The momentum parameter $\beta$ and the learning rate are determined through theoretical analysis.

## Key Experimental Results

### Main Results

| Dataset | Metric | PARS-OPT | Best Baseline | No. of Baselines |
|--------|------|----------|----------|-----------|
| ImageNet | Query efficiency ($\ell_2$ perturbation) | Best | Runner-up | Outperforms 13 SOTAs |
| CIFAR-10 | Query efficiency ($\ell_2$ perturbation) | Best | Runner-up | Outperforms 13 SOTAs |

### Ablation Study

| Configuration | Query Efficiency | Description |
|------|---------|------|
| PARS-OPT | Best | Momentum + Surrogate Prior |
| ARS-OPT | Runner-up | Momentum only, no surrogate prior |
| Ray Search w/o Momentum | Baseline | Standard method |
| No Momentum + Surrogate Prior | Moderate | Surrogate prior helps but not as significantly as momentum |

### Key Findings

- Momentum acceleration is the primary driver behind query efficiency improvements—ARS-OPT demonstrates significant gains over the non-momentum version.
- The surrogate model prior further reduces the query requirement, leading to the best overall performance for PARS-OPT.
- The theoretical convergence rates align well with the experimental results, validating the correctness of the analytical proofs.
- The method achieves the best performance among 13 SOTA baselines and has been accepted as an Oral presentation.

## Highlights & Insights

- **Introducing NAG acceleration to hard-label attacks** is an elegant combination of theory and practice—employing established optimization theory to resolve practical attack efficiency challenges.
- **Theoretical guarantees** are highly uncommon in the adversarial attack domain, which strengthens the reliability and interpretability of the method.
- The methodology is generic—momentum-accelerated query-based gradient estimation can be extended to other black-box optimization problems.

## Limitations & Future Work

- The choice of surrogate model impacts performance; if the gap between the surrogate and the target model is too large, the prior may become inaccurate.
- The study currently focuses on $\ell_2$-norm attacks; its applicability to $\ell_\infty$ attacks remains to be verified.
- The attack efficiency against defended models (such as adversarially trained models) warrants further evaluation.
- Adaptive momentum parameters can be explored to dynamically adjust the momentum step size based on the search history.

## Related Work & Insights

- **vs. SignOPT/HSJA**: These are classic ray search methods, upon which this work introduces momentum acceleration.
- **vs. Transfer-based Attacks**: Transfer attacks deploy surrogate gradients directly, whereas this work integrates surrogate gradients as a prior into ray search, yielding greater robustness.
- **vs. Query-based Gradient Estimation**: Standard query-based estimation requires numerous queries per step, whereas the accumulation of historical information in momentum reduces the per-step query demand.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing NAG acceleration into hard-label attacks is highly novel, backed by solid theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Surpasses 13 methods across two datasets with strong theoretical support.
- Writing Quality: ⭐⭐⭐⭐ Balances theory and experiments well, exhibiting Oral-level quality.
- Value: ⭐⭐⭐⭐ Offers significant contributions to AI security and adversarial robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Private Rate-Constrained Optimization with Applications to Fair Learning](../../ICLR2026/ai_safety/private_rate-constrained_optimization_with_applications_to_fair_learning.md)
- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[ICCV 2025\] Membership Inference Attacks with False Discovery Rate Control](../../ICCV2025/ai_safety/membership_inference_attacks_with_false_discovery_rate_control.md)
- [\[AAAI 2026\] Easy to Learn, Yet Hard to Forget: Towards Robust Unlearning Under Bias](easy_to_learn_yet_hard_to_forget_towards_robust_unlearning_under_bias.md)
- [\[CVPR 2026\] Improving Adversarial Transferability with Local Perturbation Augmentation](../../CVPR2026/ai_safety/improving_adversarial_transferability_with_local_perturbation_augmentation.md)

</div>

<!-- RELATED:END -->
